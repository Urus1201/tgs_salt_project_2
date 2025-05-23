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
from typing import Dict, Optional, Tuple, Union, List, Any
from pathlib import Path
import time

from models.segmentation_model import SaltSegmentationModel
from losses.combined_loss import CombinedLoss
from losses.deep_supervision_loss import DeepSupervisionLoss
from utils.logging import ExperimentLogger
from utils.lr_scheduler import create_scheduler_with_warmup


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
        
        # Check if using deep supervision
        self.use_deep_supervision = config['model'].get('use_deep_supervision', False)
        
        # Setup loss function - use DeepSupervisionLoss if deep supervision is enabled
        if self.use_deep_supervision:
            self.criterion = DeepSupervisionLoss(
                dice_weight=config['loss']['dice_weight'],
                focal_weight=config['loss']['focal_weight'],
                boundary_weight=config['loss']['boundary_weight'],
                cls_weight=config['loss']['cls_weight'],
                aux_weight=config['loss'].get('aux_weight', 0.4),
                focal_gamma=config['loss']['focal_gamma'],
                focal_alpha=config['loss']['focal_alpha'],
                aux_decay=config['loss'].get('aux_decay', True)
            )
            # Enable deep supervision in model during training
            if not self.distributed:
                self.model.enable_deep_supervision()
            else:
                # For DDP model, access the module directly
                self.model.module.enable_deep_supervision()
        else:
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
        
        # Setup learning rate scheduler with warmup support
        self.scheduler = create_scheduler_with_warmup(
            self.optimizer,
            config,
            config['training']['num_epochs']
        )
            
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

        # Initialize gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=config['training'].get('amp', True))

    def _validate_data_paths(self):
        """Validate that data directories exist before starting training."""
        if not self.distributed or self.local_rank == 0:
            try:
                # Get data path from config, using the correct key 'base_dir'
                data_path = Path(self.config.get('data', {}).get('base_dir', '/root/tgs_salt_mode_2/data'))
                
                # Handle paths properly - don't try to make absolute paths relative
                if not data_path.is_absolute():
                    # If it's a relative path, make it absolute from current directory
                    data_path = Path.cwd() / data_path
                
                # Form paths to training data directories
                train_dir = data_path / self.config.get('data', {}).get('train_dir', 'train')
                train_images_path = data_path / self.config.get('data', {}).get('train_images', 'train/images')
                train_masks_path = data_path / self.config.get('data', {}).get('train_masks', 'train/masks')
                
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
        # Initialize metrics dictionary with all possible metrics
        metric_totals = {
            'iou': 0, 'dice': 0, 'cls_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0  # Add the new metrics
        }
        num_batches = len(self.train_loader)
        
        # Progress bar only on main process
        if not self.distributed or self.local_rank == 0:
            pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        else:
            pbar = self.train_loader
        
        # Track skipped batches due to errors
        skipped_batches = 0
        
        use_amp = self.config['training'].get('amp', True) and torch.cuda.is_available()
        data_times = []
        compute_times = []
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(pbar):
            try:
                images, masks = batch_data
                # Efficient batch handling
                B = images.size(0)
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                # Create classification targets from masks
                cls_target = (masks.sum(dim=(2,3)) > 0).float()
                
                data_end_time = time.time()
                data_times.append(data_end_time - start_time)
                
                # Forward pass with automatic mixed precision
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(images)
                
                # For deep supervision model, outputs is a dictionary with 'seg_logits', 'cls_logits', and 'aux_outputs'
                # For standard model, outputs is a tuple of (seg_logits, cls_logits)
                if not isinstance(outputs, dict):
                    seg_logits, cls_logits = outputs
                    if seg_logits.shape[2:] != masks.shape[2:]:
                        seg_logits = F.interpolate(seg_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
                else:
                    # Model with deep supervision - outputs already processed by the model
                    if outputs['seg_logits'].shape[2:] != masks.shape[2:]:
                        outputs['seg_logits'] = F.interpolate(
                            outputs['seg_logits'], 
                            size=masks.shape[2:], 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # Also resize auxiliary outputs if present
                        if 'aux_outputs' in outputs:
                            for i in range(len(outputs['aux_outputs'])):
                                outputs['aux_outputs'][i] = F.interpolate(
                                    outputs['aux_outputs'][i], 
                                    size=masks.shape[2:], 
                                    mode='bilinear', 
                                    align_corners=False
                                )
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    outputs,
                    masks,
                    cls_target
                )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                self.scaler.scale(loss).backward()
                if self.config['training'].get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                compute_end_time = time.time()
                compute_times.append(compute_end_time - data_end_time)
                start_time = time.time()
                
                # Calculate metrics
                metrics = self.criterion.compute_metrics(
                    outputs,
                    masks,
                    cls_target
                )
                
                # Update averages - handle potentially missing metrics gracefully
                epoch_loss += loss_dict['total']
                for k, v in metrics.items():
                    if k in metric_totals:
                        metric_totals[k] += v
                
                # Update progress bar on main process
                if not self.distributed or self.local_rank == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total']:.4f}",
                        'iou': f"{metrics.get('iou', 0):.4f}"
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
            self.logger.logger.info(
                f"Avg data loading time: {np.mean(data_times):.4f}s, "
                f"Avg compute time: {np.mean(compute_times):.4f}s"
            )
        
        return avg_metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        # During validation, disable deep supervision to get only main outputs
        if self.use_deep_supervision:
            if not self.distributed:
                self.model.disable_deep_supervision()
            else:
                self.model.module.disable_deep_supervision()
        
        self.model.eval()
        val_loss = 0
        # Initialize metrics with all possible metrics
        metric_totals = {
            'iou': 0, 'dice': 0, 'cls_acc': 0,
            'precision': 0, 'recall': 0, 'f1': 0  # Add the new metrics
        }
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
                outputs = self.model(images)
            
            # Handle different output formats
            if not isinstance(outputs, dict):
                seg_logits, cls_logits = outputs
                if seg_logits.shape[2:] != masks.shape[2:]:
                    seg_logits = F.interpolate(seg_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
            else:
                # Model returning dictionary for deep supervision
                seg_logits = outputs['seg_logits']
                cls_logits = outputs['cls_logits'] 
                if seg_logits.shape[2:] != masks.shape[2:]:
                    seg_logits = F.interpolate(seg_logits, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            # Calculate loss - in validation we'll always get (seg_logits, cls_logits) tuple thanks to disabling deep supervision
            loss, loss_dict = self.criterion(
                (seg_logits, cls_logits),
                masks,
                cls_target
            )
            
            # Calculate metrics
            metrics = self.criterion.compute_metrics(
                (seg_logits, cls_logits),
                masks,
                cls_target
            )
            
            # Update totals - handle potentially missing metrics gracefully
            val_loss += loss_dict['total']
            for k, v in metrics.items():
                if k in metric_totals:
                    metric_totals[k] += v
                
            # Store predictions for visualization (main process only)
            if not self.distributed or self.local_rank == 0:
                if len(val_images) < self.config['logging']['num_validation_images']:
                    val_images.append(images[:4].cpu())  # Store first 4 images
                    val_masks.append(masks[:4].cpu())
                    val_predictions['seg_pred'].append(torch.sigmoid(seg_logits[:4]).cpu())
                    val_predictions['cls_pred'].append(torch.sigmoid(cls_logits[:4]).cpu())
        
        # Re-enable deep supervision for training if needed
        if self.use_deep_supervision:
            if not self.distributed:
                self.model.enable_deep_supervision()
            else:
                self.model.module.enable_deep_supervision()
                
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
        learning_rates = []
        
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
                
                # Record current learning rate for logging
                if not self.distributed or self.local_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    learning_rates.append(current_lr)
                    if self.logger is not None:
                        self.logger.logger.info(f"Current learning rate: {current_lr:.7f}")
            
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
        
        # Plot learning rate curve at the end of training
        if not self.distributed or self.local_rank == 0:
            if self.logger is not None and learning_rates:
                from utils.visualization import plot_learning_rate_schedule
                plot_learning_rate_schedule(
                    learning_rates, 
                    os.path.join(save_dir, 'learning_rate_schedule.png')
                )
            
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
        
        # Verify model compatibility
        if 'model_config' in checkpoint:
            current_config = {
                'embed_dim': self.model.embed_dim,
                'depths': self.model.depths,
                'num_heads': self.model.num_heads
            }
            saved_config = checkpoint['model_config']
            if current_config != saved_config:
                self.logger.logger.warning(
                    f"Model configuration mismatch. Current: {current_config}, Saved: {saved_config}"
                )
        
        if not self.distributed or self.local_rank == 0:
            self.logger.logger.info(f"Loaded checkpoint from epoch {self.current_epoch+1}")
            self.logger.logger.info(f"Best metric: {self.best_metric:.4f}")