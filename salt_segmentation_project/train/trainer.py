import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from pathlib import Path

from models.segmentation_model import SaltSegmentationModel
from losses.combined_loss import CombinedLoss
from utils.logging import ExperimentLogger


def setup_ddp(local_rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)


class Trainer:
    """Trainer class for TGS Salt Segmentation model."""
    def __init__(
        self,
        model: SaltSegmentationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        local_rank: int = -1,
        device: str = 'cuda',
        experiment_name: Optional[str] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            local_rank: Local rank for distributed training (-1 for non-distributed)
            device: Device to train on
            experiment_name: Optional name for experiment logging
        """
        self.config = config
        self.local_rank = local_rank
        self.device = device if local_rank == -1 else local_rank
        
        # Setup distributed training if needed
        self.distributed = local_rank != -1
        if self.distributed:
            setup_ddp(local_rank, dist.get_world_size())
            model = DDP(model, device_ids=[local_rank])
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup loss function
        self.criterion = CombinedLoss(
            dice_weight=config['loss']['dice_weight'],
            focal_weight=config['loss']['focal_weight'],
            boundary_weight=config['loss']['boundary_weight'],
            cls_weight=config['loss']['cls_weight'],
            focal_gamma=config['loss']['focal_gamma'],
            focal_alpha=config['loss']['focal_alpha']
        )
        
        # Setup optimizer with gradient clipping
        if config['training']['optimizer'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['training']['lr'],
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['training']['lr']
            )
        
        # Setup learning rate scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs'],
                eta_min=config['training']['lr'] * 0.01
            )
        elif config['training']['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
            
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Setup logging (only on main process for distributed training)
        if not self.distributed or self.local_rank == 0:
            log_dir = Path(config['training']['save_dir']) / 'logs'
            self.logger = ExperimentLogger(
                log_dir=str(log_dir),
                experiment_name=experiment_name,
                config=config
            )
        else:
            self.logger = None

        # Validate data paths
        self._validate_data_paths()

    def _validate_data_paths(self):
        """Validate that data directories exist before starting training."""
        if not self.distributed or self.local_rank == 0:
            try:
                # Get correct data path by checking both absolute and relative paths
                data_path = Path(self.config.get('data', {}).get('data_dir', '/root/tgs_salt_mode_2/data'))
                if not data_path.is_absolute():
                    # Try relative to the project root
                    project_root = Path(__file__).resolve().parents[1]
                    data_path = project_root / data_path

                train_images_path = data_path / 'train' / 'images'
                train_masks_path = data_path / 'train' / 'masks'
                
                paths_to_check = [
                    (data_path, "Data directory"),
                    (train_images_path, "Training images directory"),
                    (train_masks_path, "Training masks directory")
                ]
                
                for path, desc in paths_to_check:
                    if not path.exists():
                        self.logger.logger.error(f"{desc} does not exist: {path}")
                        raise FileNotFoundError(f"{desc} not found: {path}")
                
                # Check for image files
                image_files = list(train_images_path.glob('*.png'))
                mask_files = list(train_masks_path.glob('*.png'))
                
                if len(image_files) == 0:
                    self.logger.logger.error(f"No image files found in {train_images_path}")
                    raise FileNotFoundError(f"No image files found in {train_images_path}")
                
                if len(mask_files) == 0:
                    self.logger.logger.error(f"No mask files found in {train_masks_path}")
                    raise FileNotFoundError(f"No mask files found in {train_masks_path}")
                
                if len(image_files) != len(mask_files):
                    self.logger.logger.warning(f"Mismatch between number of images ({len(image_files)}) and masks ({len(mask_files)})")
                
                self.logger.logger.info(f"Data validation passed. Found {len(image_files)} images and {len(mask_files)} masks.")
            except Exception as e:
                self.logger.logger.error(f"Error validating data paths: {str(e)}")
                raise

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0
        metric_totals = {'iou': 0, 'dice': 0, 'cls_acc': 0}
        num_batches = len(self.train_loader)
        
        # Progress bar only on main process
        if not self.distributed or self.local_rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        else:
            pbar = self.train_loader
        
        # Track skipped batches due to errors
        skipped_batches = 0
        
        for batch_idx, batch_data in enumerate(pbar):
            try:
                images, masks = batch_data
                # Efficient batch handling
                B = images.size(0)
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Create classification targets from masks
                cls_target = (masks.sum(dim=(2,3)) > 0).float()
                
                # Forward pass with automatic mixed precision
                with torch.amp.autocast('cuda'):
                    seg_logits, cls_logits = self.model(images)
                seg_logits = F.interpolate(seg_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
                loss, loss_dict = self.criterion(
                    seg_logits, masks,
                    cls_logits, cls_target
                )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                if self.config['training'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                metrics = self.criterion.compute_metrics(
                    seg_logits, masks,
                    cls_logits, cls_target
                )
                
                # Update averages
                epoch_loss += loss_dict['total']
                for k, v in metrics.items():
                    metric_totals[k] += v
                
                # Update progress bar on main process
                if not self.distributed or self.local_rank == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total']:.4f}",
                        'iou': f"{metrics['iou']:.4f}"
                    })
            
            except FileNotFoundError as e:
                skipped_batches += 1
                if not self.distributed or self.local_rank == 0:
                    self.logger.logger.error(f"File not found error in batch {batch_idx}: {str(e)}")
                    if skipped_batches >= 5:  # Stop training if too many batches are skipped
                        raise RuntimeError(f"Too many file errors ({skipped_batches}). Check your dataset paths.")
                continue  # Skip this batch and continue
            except Exception as e:
                if not self.distributed or self.local_rank == 0:
                    self.logger.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                raise  # Re-raise other exceptions
        
        # Log how many batches were skipped
        if skipped_batches > 0 and (not self.distributed or self.local_rank == 0):
            self.logger.logger.warning(f"Skipped {skipped_batches} batches due to file errors")
        
        # Average metrics across processes for distributed training
        if self.distributed:
            avg_loss = epoch_loss / num_batches
            avg_metrics = {k: v/num_batches for k, v in metric_totals.items()}
            
            avg_loss = self._reduce_tensor(torch.tensor(avg_loss, device=self.device))
            for k in avg_metrics:
                avg_metrics[k] = self._reduce_tensor(
                    torch.tensor(avg_metrics[k], device=self.device)
                ).item()
        else:
            avg_loss = epoch_loss / num_batches
            avg_metrics = {k: v/num_batches for k, v in metric_totals.items()}
        
        avg_metrics['loss'] = avg_loss
        
        # Log metrics (main process only)
        if not self.distributed or self.local_rank == 0:
            self.logger.log_metrics(avg_metrics, phase='train', epoch=self.current_epoch)
        
        return avg_metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0
        metric_totals = {'iou': 0, 'dice': 0, 'cls_acc': 0}
        num_batches = len(self.val_loader)
        
        val_images = []
        val_masks = []
        val_predictions = {'seg_pred': [], 'cls_pred': []}
        val_ids = []
        
        for batch_idx, (images, masks) in enumerate(self.val_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            cls_target = (masks.sum(dim=(2,3)) > 0).float()
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast('cuda'):
                seg_logits, cls_logits = self.model(images)
            seg_logits = F.interpolate(seg_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss, loss_dict = self.criterion(
                seg_logits, masks,
                cls_logits, cls_target
            )
            
            # Calculate metrics
            metrics = self.criterion.compute_metrics(
                seg_logits, masks,
                cls_logits, cls_target
            )
            
            # Update totals
            val_loss += loss_dict['total']
            for k, v in metrics.items():
                metric_totals[k] += v
                
            # Store predictions for visualization (main process only)
            if not self.distributed or self.local_rank == 0:
                if len(val_images) < self.config['logging']['num_validation_images']:
                    val_images.append(images[:4].cpu())  # Store first 4 images
                    val_masks.append(masks[:4].cpu())
                    val_predictions['seg_pred'].append(torch.sigmoid(seg_logits[:4]).cpu())
                    val_predictions['cls_pred'].append(torch.sigmoid(cls_logits[:4]).cpu())
        
        # Average metrics across processes for distributed training
        if self.distributed:
            avg_loss = val_loss / num_batches
            avg_metrics = {k: v/num_batches for k, v in metric_totals.items()}
            
            avg_loss = self._reduce_tensor(torch.tensor(avg_loss, device=self.device))
            for k in avg_metrics:
                avg_metrics[k] = self._reduce_tensor(
                    torch.tensor(avg_metrics[k], device=self.device)
                ).item()
        else:
            avg_loss = val_loss / num_batches
            avg_metrics = {k: v/num_batches for k, v in metric_totals.items()}
            
        avg_metrics['loss'] = avg_loss
        
        # Log metrics and examples (main process only)
        if not self.distributed or self.local_rank == 0:
            self.logger.log_metrics(avg_metrics, phase='val', epoch=self.current_epoch)
            
            if len(val_images) > 0:
                self.logger.log_validation_examples(
                    torch.cat(val_images),
                    torch.cat(val_masks),
                    {
                        'seg_pred': torch.cat(val_predictions['seg_pred']),
                        'seg_var': torch.zeros_like(torch.cat(val_predictions['seg_pred'])),
                        'cls_pred': torch.cat(val_predictions['cls_pred'])
                    },
                    val_ids if val_ids else None
                )
        
        return avg_metrics
        
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stop_patience: int = 15
    ) -> Dict:
        """Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stop_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary with training history
        """
        if not self.distributed or self.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            
        patience_counter = 0
        
        for epoch in range(num_epochs):
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                
            self.current_epoch = epoch
            if self.logger is not None:
                self.logger.update_current_epoch(epoch)
            
            # Training epoch
            train_metrics = self.train_epoch()
            
            # Validation epoch
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Save best model (main process only)
            if not self.distributed or self.local_rank == 0:
                if val_metrics['iou'] > self.best_metric:
                    self.best_metric = val_metrics['iou']
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    # Save checkpoint
                    self.logger.save_model(
                        self.model.module if self.distributed else self.model,
                        self.optimizer,
                        epoch,
                        self.best_metric,
                        'val_iou'
                    )
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stop_patience:
                    if self.logger is not None:
                        self.logger.logger.info(
                            f"Early stopping triggered! No improvement for {early_stop_patience} epochs"
                        )
                    break
            
            # Synchronize processes
            if self.distributed:
                dist.barrier()
        
        # Plot final training curves (main process only)
        if not self.distributed or self.local_rank == 0:
            self.logger.plot_metrics()
            
        # Clean up distributed training
        if self.distributed:
            dist.destroy_process_group()
        
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across distributed processes."""
        if not self.distributed:
            return tensor
            
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Handle distributed vs non-distributed state dict
        if self.distributed:
            if not next(iter(state_dict.keys())).startswith('module.'):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
        else:
            if next(iter(state_dict.keys())).startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('val_iou', 0.0)
        
        if not self.distributed or self.local_rank == 0:
            self.logger.logger.info(f"Loaded checkpoint from epoch {self.current_epoch+1}")
            self.logger.logger.info(f"Best metric: {self.best_metric:.4f}")