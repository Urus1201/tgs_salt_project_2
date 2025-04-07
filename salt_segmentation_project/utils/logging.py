import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import yaml
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from .visualization import (
    plot_training_history,
    plot_prediction_grid,
    plot_boundary_refinement,
    plot_depth_distribution
)


class ExperimentLogger:
    """Logger for training experiments."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        # Create log directory
        self.base_log_dir = Path(log_dir)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(str(self.experiment_dir / 'tensorboard'))
        
        # Set up file logging
        self.log_file = self.experiment_dir / 'experiment.log'
        self.setup_file_logging()
        
        # Save config if provided
        if config is not None:
            self.save_config(config)
            
        # Initialize metric tracking
        self.metrics = {
            'train': {},
            'val': {}
        }
        
        # Initialize current epoch tracking
        self.current_epoch = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized experiment in {self.experiment_dir}")

    def update_current_epoch(self, epoch: int) -> None:
        """Update the current epoch number.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        
    def setup_file_logging(self) -> None:
        """Set up file logging."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = self.experiment_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: int,
        phase: str = 'train'
    ) -> None:
        """Log metrics to TensorBoard and update tracking.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current epoch
            phase: 'train' or 'val'
        """
        # Update tracking
        for name, value in metrics.items():
            if name not in self.metrics[phase]:
                self.metrics[phase][name] = []
            self.metrics[phase][name].append(value)
            
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{name}', value, epoch)
            
        # Log to file
        metrics_str = ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'{phase.capitalize()} epoch {epoch}: {metrics_str}')
        
    def log_images(
        self,
        images: List[torch.Tensor],
        masks: List[torch.Tensor],
        predictions: List[torch.Tensor],
        uncertainties: Optional[List[torch.Tensor]] = None,
        step: int = 0,
        max_images: int = 8
    ) -> None:
        """Log images to TensorBoard.
        
        Args:
            images: List of input images
            masks: List of ground truth masks
            predictions: List of predicted masks
            uncertainties: Optional list of uncertainty maps
            step: Current step/iteration
            max_images: Maximum number of images to log
        """
        num_images = min(len(images), max_images)
        
        # Convert tensors to numpy arrays
        images = [img.cpu().numpy() for img in images[:num_images]]
        masks = [mask.cpu().numpy() for mask in masks[:num_images]]
        predictions = [pred.cpu().numpy() for pred in predictions[:num_images]]
        
        # Fix the boolean check - check if uncertainties is not None and not empty
        if uncertainties is not None and len(uncertainties) > 0:
            uncertainties = [unc.cpu().numpy() for unc in uncertainties[:num_images]]
            
        # Create visualization grid
        save_path = self.experiment_dir / f'predictions_step_{step}.png'
        plot_prediction_grid(
            images=images,
            masks=masks,
            predictions=predictions,
            uncertainties=uncertainties,
            num_examples=num_images,
            save_path=str(save_path)
        )
        
        # Log individual images to TensorBoard
        for i in range(num_images):
            self.writer.add_image(
                f'image_{i}/input',
                images[i],
                step,
                dataformats='HW'
            )
            self.writer.add_image(
                f'image_{i}/mask',
                masks[i],
                step,
                dataformats='HW'
            )
            self.writer.add_image(
                f'image_{i}/prediction',
                predictions[i],
                step,
                dataformats='HW'
            )
            if uncertainties is not None and len(uncertainties) > 0:
                self.writer.add_image(
                    f'image_{i}/uncertainty',
                    uncertainties[i],
                    step,
                    dataformats='HW'
                )
                
    def log_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float = None,
        metric_name: str = None,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metric_value: Optional best metric value
            metric_name: Optional name of the metric
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        # Add metric value if provided
        if metric_value is not None and metric_name is not None:
            checkpoint[metric_name] = metric_value
        
        # Save latest checkpoint
        checkpoint_path = self.experiment_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save numbered checkpoint
        checkpoint_path = self.experiment_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = self.experiment_dir / 'model_best.pth'
            torch.save(checkpoint, best_path)
            
    def save_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float = None,
        metric_name: str = None
    ) -> None:
        """Alias for log_model that matches trainer's expected interface.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch
            metric_value: Best metric value
            metric_name: Name of the metric
        """
        self.log_model(model, optimizer, epoch, metric_value, metric_name, is_best=True)
        
    def log_validation_examples(
        self,
        images,
        masks,
        predictions,
        image_ids=None,
        step=None
    ):
        """Log validation examples to tensorboard.
        
        Args:
            images: Tensor of validation images (B,C,H,W)
            masks: Tensor of ground truth masks (B,1,H,W)
            predictions: Dict of prediction tensors
            image_ids: Optional list of image IDs
            step: Optional global step
        """
        if step is None:
            step = self.current_epoch
            
        # Ensure proper tensor format and normalization
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
            
        masks = masks.float()
        if masks.max() > 1.0:
            masks = masks / 255.0
            
        # Log images using the visualization utility
        from .visualization import log_images
        log_images(
            logger=self.writer,
            images=images,
            masks=masks,
            predictions=predictions,
            tag='validation',
            global_step=step
        )
        
    def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Save additional artifacts (e.g., submission file).
        
        Args:
            artifacts: Dictionary of artifact names and contents
        """
        artifacts_dir = self.experiment_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        
        for name, content in artifacts.items():
            path = artifacts_dir / name
            
            if isinstance(content, (dict, list)):
                # Save JSON
                with open(path.with_suffix('.json'), 'w') as f:
                    json.dump(content, f, indent=2)
            elif isinstance(content, np.ndarray):
                # Save numpy array
                np.save(path.with_suffix('.npy'), content)
            elif isinstance(content, (pd.DataFrame, pd.Series)):
                # Save pandas object
                content.to_csv(path.with_suffix('.csv'))
            else:
                # Save as text
                with open(path.with_suffix('.txt'), 'w') as f:
                    f.write(str(content))
                    
    def plot_training_curves(self) -> None:
        """Plot and save training curves."""
        save_path = self.experiment_dir / 'training_curves.png'
        plot_training_history(
            train_metrics=self.metrics['train'],
            val_metrics=self.metrics['val'],
            save_path=str(save_path)
        )
        
    def plot_metrics(self) -> None:
        """Alias for plot_training_curves to match trainer's expected interface."""
        self.plot_training_curves()
        
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """Log hyperparameters and their corresponding metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of metric values
        """
        self.writer.add_hparams(hparams, metrics)
        
    def close(self) -> None:
        """Close the logger and save final artifacts."""
        # Plot final training curves
        self.plot_training_curves()
        
        # Save final metrics
        metrics_path = self.experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Close TensorBoard writer
        self.writer.close()