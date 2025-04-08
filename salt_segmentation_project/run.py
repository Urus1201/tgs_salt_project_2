import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from data.dataset import SaltDataset
from data.transforms import get_training_augmentation, get_validation_augmentation
from models.segmentation_model import SaltSegmentationModel
from models.encoder_swin import SwinEncoder
from train.trainer import Trainer
from train.pretrain_mae import pretrain_mae
from inference.predictor import Predictor
from inference.submission import SubmissionGenerator, mask_to_rle
from inference.refinement import RefinementNet, UncertaintyRefinement, train_refinement_model
from utils.path_utils import prepare_data_paths


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file and resolve paths."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve and prepare all data paths
    config = prepare_data_paths(config)
    
    return config


def create_dataloaders(config: Dict, local_rank: int = -1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        local_rank: Local rank for distributed training (-1 for non-distributed)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    train_transform = get_training_augmentation()
    val_transform = get_validation_augmentation()
    
    # Create datasets
    train_dataset = SaltDataset(
        csv_file=config['data']['train_csv'],
        image_dir=config['data']['train_images'],
        mask_dir=config['data']['train_masks'],
        depths_csv=config['data']['depths_csv'],
        transform=train_transform,
        use_2_5d=config['data']['use_2_5d'],
        mode='train'
    )
    
    val_dataset = SaltDataset(
        csv_file=config['data']['train_csv'],
        image_dir=config['data']['train_images'],
        mask_dir=config['data']['train_masks'],
        depths_csv=config['data']['depths_csv'],
        transform=val_transform,
        use_2_5d=config['data']['use_2_5d'],
        mode='train'
    )
    
    # Get indices for train/val split
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    val_size = int(dataset_size * config['data']['val_split'])
    
    # Use fixed seed for reproducible splits
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # Create samplers
    if local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        from torch.utils.data.sampler import SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
    
    # Create test dataset
    test_dataset = SaltDataset(
        csv_file=config['data']['test_csv'],
        image_dir=config['data']['test_images'],
        depths_csv=config['data']['depths_csv'],
        transform=val_transform,
        use_2_5d=config['data']['use_2_5d'],
        mode='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_worker(
    local_rank: int,
    world_size: int,
    config: dict
):
    """Worker function for distributed training.
    
    Args:
        local_rank: Local rank of this process
        world_size: Total number of processes
        config: Configuration dictionary
    """
    # Create dataloaders with DDP sampler
    train_loader, val_loader, _ = create_dataloaders(config, local_rank)
    
    # Create model
    model = SaltSegmentationModel(
        img_size=config['model']['img_size'],
        in_channels=3 if config['data']['use_2_5d'] else 1,
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size'],
        dropout_rate=config['model']['dropout_rate'],
        use_checkpoint=config['model']['use_checkpoint'],
        pretrained_encoder=config['model']['mae_checkpoint'] if config['model']['mae_pretrained'] else None
    )
    
    # Create trainer with DDP
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        local_rank=local_rank,
        device='cuda'
    )
    
    # Train model
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_dir=config['training']['save_dir'],  # Ensure checkpoints are saved directly in the specified directory
        early_stop_patience=config['training']['early_stop_patience']
    )


def pretrain(config_path: str):
    """Run MAE pretraining pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Validate config
    if 'mae' not in config:
        raise ValueError("Configuration file missing 'mae' section")
    
    # Create save directory
    os.makedirs(config['mae']['save_dir'], exist_ok=True)
    
    try:
        # Run pretraining
        metrics = pretrain_mae(config)
        
        print("MAE pretraining completed!")
        # Safely get minimum validation loss
        val_losses = [m['val_loss'] for m in metrics if 'val_loss' in m]
        if val_losses:
            best_val_loss = min(val_losses)
            print(f"Best validation loss: {best_val_loss:.4f}")
        else:
            print("Warning: No validation metrics were recorded during training")
        print(f"Encoder weights saved to: {config['mae']['save_dir']}/best_encoder.pth")
    except Exception as e:
        print(f"Error during MAE pretraining: {str(e)}")
        raise


def train(config_path: str):
    """Run training pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Validate config sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Configuration file missing '{section}' section")
    
    if config['training'].get('distributed', False):
        # Launch distributed training
        world_size = torch.cuda.device_count()
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        device = torch.device(
            config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Create dataloaders
        train_loader, val_loader, _ = create_dataloaders(config)
        
        # Create model
        model = SaltSegmentationModel(
            img_size=config['model']['img_size'],
            in_channels=3 if config['data']['use_2_5d'] else 1,
            embed_dim=config['model']['embed_dim'],
            depths=config['model']['depths'],
            num_heads=config['model']['num_heads'],
            window_size=config['model']['window_size'],
            dropout_rate=config['model']['dropout_rate'],
            use_checkpoint=config['model']['use_checkpoint'],
            pretrained_encoder=config['model']['mae_checkpoint'] if config['model']['mae_pretrained'] else None
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Train model
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            save_dir=config['training']['save_dir'],  # Ensure checkpoints are saved directly in the specified directory
            early_stop_patience=config['training']['early_stop_patience']
        )


def train_refinement(config_path: str):
    """Train the refinement model after main training."""
    # 1. Load config
    config = load_config(config_path)
    
    # Check if refinement section exists in config
    if 'refinement' not in config:
        raise ValueError("Configuration file missing 'refinement' section. Please add refinement parameters to your config.")
    
    # 2. Set device
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # 3. Load validation data
    _, val_loader, _ = create_dataloaders(config)
    
    # 4. Load main model for generating predictions on validation set
    main_model = SaltSegmentationModel(
        img_size=config['model']['img_size'],
        in_channels=3 if config['data']['use_2_5d'] else 1,
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Load checkpoint
    main_checkpoint_path = config['refinement'].get('main_model_checkpoint')
    if not main_checkpoint_path:
        raise ValueError("Please specify 'main_model_checkpoint' in the refinement config section")
    
    checkpoint = torch.load(main_checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    main_model.load_state_dict(state_dict)
    main_model = main_model.to(device)
    
    # 5. Create predictor
    predictor = Predictor(
        model=main_model,
        device=device,
        threshold=config['inference'].get('threshold', 0.5),
        use_tta=config['inference'].get('use_tta', True),
        use_mc_dropout=config['inference'].get('use_mc_dropout', True),
        mc_samples=config['inference'].get('mc_samples', 10)
    )
    
    print("Generating data for refinement model training...")
    # 6. Generate training data for refinement model
    train_inputs = []
    train_targets = []
    
    # Enable progress bar
    from tqdm import tqdm
    
    for images, masks in tqdm(val_loader, desc="Generating refinement data"):
        # Move batch to device
        images = images.to(device)
        masks = masks.to(device)
        
        # Get predictions with uncertainty for entire batch
        with torch.no_grad():
            predictions = predictor.predict_batch(images, progress_bar=False)
            
            # Process entire batch at once
            x_batch = torch.cat([
                images,
                F.interpolate(predictions['seg_pred'], size=images.shape[2:], mode='bilinear', align_corners=True),
                F.interpolate(predictions['seg_var'], size=images.shape[2:], mode='bilinear', align_corners=True)
            ], dim=1)
            
            train_inputs.append(x_batch)
            train_targets.append(masks)
    
    # Concatenate all batches
    train_inputs = torch.cat(train_inputs, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    
    # Split data into train/val
    dataset_size = len(train_inputs)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.8)  # 80/20 split
    
    # Use fixed seed for reproducible splits
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    
    # Create final train/val sets
    train_data = (
        train_inputs[indices[:split]],
        train_targets[indices[:split]]
    )
    
    val_data = (
        train_inputs[indices[split:]],
        train_targets[indices[split:]]
    )
    
    print(f"Created refinement training data: {train_data[0].shape} inputs, {train_data[1].shape} targets")
    print(f"Created refinement validation data: {val_data[0].shape} inputs, {val_data[1].shape} targets")
    
    # 7. Create and train refinement model
    refinement_model = RefinementNet(
        in_channels=5,  # image (3) + mask (1) + uncertainty (1)
        base_channels=config['refinement'].get('base_channels', 16),
        num_levels=config['refinement'].get('num_levels', 3),
        dropout_rate=config['refinement'].get('dropout_rate', 0.2)
    )
    
    print("Training refinement model...")
    # Parse learning rate as float (in case it's a string in the config)
    learning_rate = float(config['refinement'].get('learning_rate', 1e-4))
    weight_decay = float(config['refinement'].get('weight_decay', 1e-5))
    
    refinement_model = train_refinement_model(
        model=refinement_model,
        train_data=train_data,
        val_data=val_data,
        num_epochs=config['refinement'].get('epochs', 50),
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=config['refinement'].get('patience', 10)
    )
    
    # 8. Save refinement model checkpoint
    checkpoint_path = config['refinement'].get('checkpoint_path', os.path.join(config['training']['save_dir'], 'refinement_model.pth'))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(refinement_model.state_dict(), checkpoint_path)
    print(f"Refinement model trained and saved to: {checkpoint_path}")


def predict(config_path: str, checkpoint_path: str):
    """Run inference pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create dataloaders
    _, _, test_loader = create_dataloaders(config)
    
    # Create model and load checkpoint
    model = SaltSegmentationModel(
        img_size=config['model']['img_size'],
        in_channels=3 if config['data']['use_2_5d'] else 1,
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Handle distributed checkpoint if needed
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Create predictor
    predictor = Predictor(
        model=model,
        device=device,
        threshold=config['inference']['threshold'],
        use_tta=config['inference']['use_tta'],
        use_mc_dropout=config['inference']['use_mc_dropout'],
        mc_samples=config['inference']['mc_samples']
    )
    
    # Optional: Load and use refinement model
    refinement = None
    if config['inference']['refinement']['enabled']:
        print("Loading refinement model...")
        refinement_path = config['refinement'].get('checkpoint_path')
        if not refinement_path or not os.path.exists(refinement_path):
            print(f"Warning: Refinement model path not found: {refinement_path}")
            print("Prediction will continue without refinement.")
        else:
            try:
                refinement_model = RefinementNet(
                    in_channels=5,  # image (3) + mask (1) + uncertainty (1)
                    base_channels=config['refinement'].get('base_channels', 16),
                    num_levels=config['refinement'].get('num_levels', 3),
                    dropout_rate=config['refinement'].get('dropout_rate', 0.2)
                )
                
                refinement_model.load_state_dict(
                    torch.load(refinement_path, map_location=device)
                )
                
                refinement = UncertaintyRefinement(
                    model=refinement_model,
                    device=device,
                    threshold=config['inference']['threshold']
                )
                print("Refinement model loaded successfully.")
            except Exception as e:
                print(f"Error loading refinement model: {str(e)}")
                print("Prediction will continue without refinement.")
    
    # Create submission generator with optional refinement
    submission_path = config['inference'].get('submission_path',
        os.path.join(config['training']['save_dir'], 'submission.csv')
    )
    
    # Create SubmissionGenerator
    if refinement:
        # Create a custom submission generator to use refinement
        # This is necessary because the refinement process needs to process each image
        print("Generating predictions with refinement...")
        predictions_list = []
        image_ids = []
        
        for images, batch_ids in test_loader:
            # Get predictions with uncertainty
            predictions = predictor.predict_batch(images)
            
            # Apply refinement to each prediction
            refined_masks = []
            for i in range(len(images)):
                image_slice = images[i:i+1].to(device)
                pred_slice = {k: v[i:i+1] for k, v in predictions.items()}
                
                # Apply refinement
                refined_mask = refinement.refine_prediction(
                    image_slice, pred_slice
                )
                
                # Handle empty mask cases using classifier
                if (1 - pred_slice['cls_pred']).item() > config['inference'].get('empty_threshold', 0.9):
                    refined_mask.zero_()
                
                refined_masks.append(refined_mask.cpu().numpy()[0, 0])
            
            predictions_list.extend(refined_masks)
            image_ids.extend(batch_ids)
            
        # Convert to RLE format
        print("Converting to RLE format...")
        rle_strings = []
        for mask in predictions_list:
            rle = mask_to_rle(mask)
            rle_strings.append(rle)
            
        # Create submission DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'id': image_ids,
            'rle_mask': rle_strings
        })
        
        # Save submission
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        
        # Print statistics
        empty_masks = df['rle_mask'].str.len() == 0
        print(f"\nSubmission statistics:")
        print(f"Total images: {len(df)}")
        print(f"Empty masks: {empty_masks.sum()} ({empty_masks.mean()*100:.1f}%)")
    else:
        # Use standard SubmissionGenerator
        submission = SubmissionGenerator(
            predictor=predictor,
            test_loader=test_loader,
            submission_path=submission_path
        )
        
        # Generate predictions and submission file
        submission.generate(
            uncertainty_threshold=config['inference'].get('threshold'),
            empty_threshold=config['inference'].get('empty_threshold', 0.9)
        )


def main():
    parser = argparse.ArgumentParser(description='TGS Salt Segmentation')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['pretrain', 'train', 'predict', 'train_refinement'],
                      help='Run mode')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint (for prediction)')
    
    args = parser.parse_args()
    
    if args.mode == 'pretrain':
        pretrain(args.config)
    elif args.mode == 'train':
        train(args.config)
    elif args.mode == 'train_refinement':
        train_refinement(args.config)
    elif args.mode == 'predict':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for prediction mode")
        predict(args.config, args.checkpoint)


if __name__ == '__main__':
    main()