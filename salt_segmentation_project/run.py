import os
import yaml
import argparse
import torch
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
from inference.submission import SubmissionGenerator
from inference.refinement import RefinementNet, UncertaintyRefinement
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
        save_dir=config['training']['save_dir'],
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
            save_dir=config['training']['save_dir'],
            early_stop_patience=config['training']['early_stop_patience']
        )


def predict(config_path: str, checkpoint_path: str):
    """Run inference pipeline."""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['training']['device'])
    
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
    if config['inference']['refinement']['enabled']:
        refinement_model = RefinementNet(
            in_channels=3,  # image + mask + uncertainty
            base_channels=16
        )
        refinement_model.load_state_dict(
            torch.load(config['inference']['refinement']['checkpoint'])
        )
        refinement = UncertaintyRefinement(
            model=refinement_model,
            device=device,
            threshold=config['inference']['threshold']
        )
    else:
        refinement = None
    
    # Create submission generator
    submission = SubmissionGenerator(
        predictor=predictor,
        test_loader=test_loader,
        refinement=refinement,
        submission_path=os.path.join(
            config['training']['save_dir'],
            'submission.csv'
        )
    )
    
    # Generate predictions and submission file
    submission.generate()


def main():
    parser = argparse.ArgumentParser(description='TGS Salt Segmentation')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['pretrain', 'train', 'predict'],
                      help='Run mode')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint (for prediction)')
    
    args = parser.parse_args()
    
    if args.mode == 'pretrain':
        pretrain(args.config)
    elif args.mode == 'train':
        train(args.config)
    elif args.mode == 'predict':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path required for prediction mode")
        predict(args.config, args.checkpoint)


if __name__ == '__main__':
    main()