import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns
from pathlib import Path
import pandas as pd
import os
from PIL import Image
from inference.submission import rle_to_mask


def plot_training_history(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: str
) -> None:
    """Plot training and validation metrics history.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        save_path: Path to save the plot
    """
    num_metrics = len(train_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
    if num_metrics == 1:
        axes = [axes]
        
    for idx, (metric_name, train_values) in enumerate(train_metrics.items()):
        ax = axes[idx]
        epochs = range(1, len(train_values) + 1)
        
        # Plot training metrics
        ax.plot(epochs, train_values, 'b-', label=f'Train {metric_name}')
        
        # Plot validation metrics if available
        if metric_name in val_metrics:
            val_values = val_metrics[metric_name]
            ax.plot(epochs, val_values, 'r-', label=f'Val {metric_name}')
            
        ax.set_title(f'{metric_name} vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_prediction_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    predictions: List[np.ndarray],
    uncertainties: Optional[List[np.ndarray]] = None,
    num_examples: int = 8,
    save_path: str = None
) -> None:
    """Plot a grid of images with their corresponding masks and predictions.
    
    Args:
        images: List of input images
        masks: List of ground truth masks
        predictions: List of predicted masks
        uncertainties: Optional list of uncertainty maps
        num_examples: Number of examples to plot
        save_path: Path to save the plot
    """
    num_rows = num_examples
    num_cols = 3 if uncertainties is None else 4
    
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3*num_cols, 3*num_rows)
    )
    
    for idx in range(num_examples):
        # Plot original image
        axes[idx, 0].imshow(np.moveaxis(images[idx], 0, -1))
        axes[idx, 0].set_title('Input')
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(masks[idx].squeeze(0), cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Plot prediction
        axes[idx, 2].imshow(predictions[idx].squeeze(0), cmap='gray')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # Plot uncertainty if available
        if uncertainties is not None:
            axes[idx, 3].imshow(uncertainties[idx].squeeze(0), cmap='magma')
            axes[idx, 3].set_title('Uncertainty')
            axes[idx, 3].axis('off')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_boundary_refinement(
    original_pred: np.ndarray,
    refined_pred: np.ndarray,
    boundary_mask: np.ndarray,
    save_path: str = None
) -> None:
    """Plot boundary refinement results.
    
    Args:
        original_pred: Original prediction mask
        refined_pred: Refined prediction mask
        boundary_mask: Boundary uncertainty mask
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original_pred, cmap='gray')
    axes[0].set_title('Original Prediction')
    axes[0].axis('off')
    
    axes[1].imshow(boundary_mask, cmap='magma')
    axes[1].set_title('Boundary Uncertainty')
    axes[1].axis('off')
    
    axes[2].imshow(refined_pred, cmap='gray')
    axes[2].set_title('Refined Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_depth_distribution(
    depths: np.ndarray,
    masks: np.ndarray,
    save_path: str = None
) -> None:
    """Plot distribution of salt presence across depths.
    
    Args:
        depths: Array of depth values
        masks: Array of mask coverage percentages
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(depths, masks, alpha=0.5)
    
    # Add trend line
    z = np.polyfit(depths, masks, 1)
    p = np.poly1d(z)
    plt.plot(depths, p(depths), "r--", alpha=0.8)
    
    plt.xlabel('Depth')
    plt.ylabel('Salt Coverage (%)')
    plt.title('Salt Presence vs. Depth')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: str = None
) -> None:
    """Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 4))
    
    steps = range(len(learning_rates))
    plt.plot(steps, learning_rates)
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: str = None
) -> None:
    """Plot confusion matrix for binary segmentation.
    
    Args:
        confusion_matrix: 2x2 confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Salt', 'Salt'],
        yticklabels=['Non-Salt', 'Salt']
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def log_images(
    logger,
    images,
    masks=None,
    predictions=None,
    tag='images',
    global_step=0
):
    """Log images to tensorboard.
    
    Args:
        logger: Tensorboard logger instance
        images: Tensor of images (B, C, H, W)
        masks: Optional tensor of masks (B, 1, H, W)
        predictions: Optional dict containing prediction tensors
        tag: Base tag for tensorboard
        global_step: Global step for tensorboard
    """
    # Convert CHW to HWC format for visualization
    images = images.permute(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
    
    # Handle grayscale images
    if images.shape[-1] == 1:
        images = images.repeat(1, 1, 1, 3)
    
    # Log original images
    logger.add_images(f'{tag}/input', images, global_step, dataformats='NHWC')
    
    if masks is not None:
        # Convert mask to HW format and add as image
        mask_vis = masks.squeeze(1)  # Remove channel dim for masks
        logger.add_images(f'{tag}/mask', mask_vis.unsqueeze(-1), global_step, dataformats='NHWC')
    
    if predictions is not None:
        for pred_name, pred_tensor in predictions.items():
            # Handle different prediction types
            if pred_name == 'seg_pred':
                # Convert segmentation predictions to HW format
                pred_vis = pred_tensor.squeeze(1)  # Remove channel dim
                logger.add_images(f'{tag}/{pred_name}', pred_vis.unsqueeze(-1), 
                                global_step, dataformats='NHWC')
            elif pred_name == 'cls_pred':
                # Classification predictions can be logged as scalars
                logger.add_scalars(f'{tag}/{pred_name}', 
                                {f'sample_{i}': p for i, p in enumerate(pred_tensor)},
                                global_step)
            else:
                # For other prediction types, try to visualize as images
                if pred_tensor.ndim == 4:  # B,C,H,W format
                    pred_vis = pred_tensor.permute(0, 2, 3, 1)
                    if pred_vis.shape[-1] == 1:
                        pred_vis = pred_vis.repeat(1, 1, 1, 3)
                    logger.add_images(f'{tag}/{pred_name}', pred_vis, 
                                    global_step, dataformats='NHWC')


def visualize_predictions(submission_path, test_images_dir, num_samples=5):
    """Visualize sample predictions from the submission file.

    Args:
        submission_path (str): Path to the submission CSV file.
        test_images_dir (str): Directory containing test images.
        num_samples (int): Number of samples to visualize.
    """
    # Load submission file
    submission = pd.read_csv(submission_path)

    # Randomly select samples
    samples = submission.sample(n=num_samples, random_state=42)

    # Plot each sample
    for _, row in samples.iterrows():
        image_id = row['id']
        rle_mask = row['rle_mask']

        # Load test image
        image_path = os.path.join(test_images_dir, f"{image_id}.png")
        image = np.array(Image.open(image_path))

        # Decode RLE mask
        mask = rle_to_mask(rle_mask, shape=image.shape)

        # Plot image and mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Test Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        plt.show()