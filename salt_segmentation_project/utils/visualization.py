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
        # Process the image based on its current format
        img = images[idx]
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3:
                if img.shape[0] in [1, 3]:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
                # Otherwise assume it's already in HWC format
            if len(img.shape) > 2 and img.shape[-1] == 1:  # Remove singleton dimensions
                img = img.squeeze(-1)
        
        # Process the mask
        mask = masks[idx]
        if isinstance(mask, np.ndarray) and len(mask.shape) > 2:
            if mask.shape[0] == 1:  # CHW format with single channel
                mask = mask.squeeze(0)
            elif mask.shape[-1] == 1:  # HWC format with single channel
                mask = mask.squeeze(-1)
        
        # Process the prediction
        pred = predictions[idx]
        if isinstance(pred, np.ndarray) and len(pred.shape) > 2:
            if pred.shape[0] == 1:  # CHW format with single channel
                pred = pred.squeeze(0)
            elif pred.shape[-1] == 1:  # HWC format with single channel
                pred = pred.squeeze(-1)
        
        # Plot original image
        axes[idx, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[idx, 0].set_title('Input')
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Plot prediction
        axes[idx, 2].imshow(pred, cmap='gray')
        axes[idx, 2].set_title('Prediction')
        axes[idx, 2].axis('off')
        
        # Plot uncertainty if available
        if uncertainties is not None:
            uncert = uncertainties[idx]
            if isinstance(uncert, np.ndarray) and len(uncert.shape) > 2:
                if uncert.shape[0] == 1:  # CHW format with single channel
                    uncert = uncert.squeeze(0)
                elif uncert.shape[-1] == 1:  # HWC format with single channel
                    uncert = uncert.squeeze(-1)
            
            axes[idx, 3].imshow(uncert, cmap='magma')
            axes[idx, 3].set_title('Uncertainty')
            axes[idx, 3].axis('off')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()  # Changed from plt.close() to actually display the plot


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

def create_overlay(image, mask, alpha=0.5, mask_color=[1, 0, 0]):
    """
    Create an overlay of the mask on the image.
    
    Args:
        image: Grayscale or RGB image array
        mask: Binary mask array
        alpha: Transparency of the overlay (0-1)
        mask_color: Color of the mask overlay in RGB
    
    Returns:
        Overlay image
    """
    # Ensure image is in the range [0, 1]
    if image.max() > 1:
        image = image / 255.0
    
    # Handle different image and mask shapes
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image_rgb = np.stack([image] * 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
        image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = image.squeeze(-1)
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image
    else:
        image_rgb = image
        
    # Normalize mask to be binary
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    
    # Create colored mask
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[..., 0] = mask * mask_color[0]
    colored_mask[..., 1] = mask * mask_color[1]
    colored_mask[..., 2] = mask * mask_color[2]
    
    # Create overlay
    overlay = image_rgb * (1 - alpha) + colored_mask * alpha
    
    # Clip values to be between 0 and 1
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def visualize_refinement_comparison(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    original_preds: List[np.ndarray],
    refined_preds: List[np.ndarray],
    uncertainties: Optional[List[np.ndarray]] = None,
    num_examples: int = 4,
    save_path: str = None
) -> None:
    """
    Visualize a grid comparing original and refined predictions.
    
    Args:
        images: List of input images
        masks: List of ground truth masks
        original_preds: List of original prediction masks
        refined_preds: List of refined prediction masks
        uncertainties: Optional list of uncertainty maps
        num_examples: Number of examples to plot
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(
        num_examples, 
        5 if uncertainties else 4, 
        figsize=(20, 5*num_examples)
    )
    
    for i in range(num_examples):
        # Process images for visualization
        img = images[i]
        if isinstance(img, np.ndarray) and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if len(img.shape) > 2 and img.shape[-1] == 1:
            img = img.squeeze(-1)
            
        # Original image
        axes[i, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt = masks[i]
        if len(gt.shape) > 2:
            gt = gt.squeeze()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Original prediction
        orig_pred = original_preds[i]
        if len(orig_pred.shape) > 2:
            orig_pred = orig_pred.squeeze()
        axes[i, 2].imshow(orig_pred, cmap='gray')
        axes[i, 2].set_title('Original Prediction')
        axes[i, 2].axis('off')
        
        # Refined prediction
        ref_pred = refined_preds[i]
        if len(ref_pred.shape) > 2:
            ref_pred = ref_pred.squeeze()
        axes[i, 3].imshow(ref_pred, cmap='gray')
        axes[i, 3].set_title('Refined Prediction')
        axes[i, 3].axis('off')
        
        # Uncertainty if provided
        if uncertainties:
            uncert = uncertainties[i]
            if len(uncert.shape) > 2:
                uncert = uncert.squeeze()
            axes[i, 4].imshow(uncert, cmap='magma')
            axes[i, 4].set_title('Uncertainty')
            axes[i, 4].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_gradcam(
    model,
    image,
    target_layer,
    target_class=None,
    mask=None,
    save_path=None
):
    """
    Visualize GradCAM for model explanation.
    
    Args:
        model: PyTorch model
        image: Input image tensor (C,H,W)
        target_layer: Target layer for GradCAM
        target_class: Target class for GradCAM (None for segmentation)
        mask: Optional ground truth mask
        save_path: Path to save the visualization
    """
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Please install pytorch-grad-cam: pip install pytorch-grad-cam")
        return
        
    # Create a simple wrapper for the model if necessary
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            preds = self.model(x)
            if isinstance(preds, tuple):
                return preds[0]  # Assume segmentation prediction is first output
            elif isinstance(preds, dict):
                return preds['seg_pred']  # Assume it's in a dict with this key
            return preds
    
    wrapped_model = ModelWrapper(model)
    
    # Setup GradCAM
    cam = GradCAM(
        model=wrapped_model, 
        target_layers=[target_layer],
        use_cuda=next(model.parameters()).is_cuda
    )
    
    # Preprocess image if it's not already a tensor
    if isinstance(image, np.ndarray):
        if image.shape[0] not in [1, 3]:  # Not in CHW format
            if len(image.shape) == 3:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = np.expand_dims(image, 0)  # Add channel dim
        image_tensor = torch.from_numpy(image.copy()).float().unsqueeze(0)
    else:
        image_tensor = image.clone().unsqueeze(0) if image.dim() == 3 else image.clone()
    
    # Generate GradCAM
    grayscale_cam = cam(input_tensor=image_tensor, target_category=target_class)
    grayscale_cam = grayscale_cam[0, :]
    
    # Convert image for visualization
    if isinstance(image, torch.Tensor):
        image_for_vis = image.cpu().numpy()
    else:
        image_for_vis = image.copy()
    
    # Handle CHW format
    if image_for_vis.shape[0] in [1, 3]:
        image_for_vis = np.transpose(image_for_vis, (1, 2, 0))
    
    # Ensure image is normalized and convert to RGB if grayscale
    if image_for_vis.max() > 1:
        image_for_vis = image_for_vis / 255.0
    
    if len(image_for_vis.shape) == 2 or image_for_vis.shape[-1] == 1:
        if len(image_for_vis.shape) > 2:
            image_for_vis = image_for_vis.squeeze(-1)
        image_for_vis = np.stack([image_for_vis] * 3, axis=-1)
    
    # Create visualization
    cam_image = show_cam_on_image(image_for_vis, grayscale_cam, use_rgb=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(12, 4))
    
    axes[0].imshow(image_for_vis)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam_image)
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return grayscale_cam

def analyze_sample(
    image,
    mask=None,
    predictions=None,
    refined_prediction=None,
    uncertainty=None,
    gradcam=None,
    save_path=None
):
    """
    Comprehensive visualization and analysis of a single sample.
    
    Args:
        image: Input image
        mask: Ground truth mask (optional)
        predictions: Model prediction (optional)
        refined_prediction: Refined prediction (optional)
        uncertainty: Uncertainty map (optional)
        gradcam: GradCAM output (optional)
        save_path: Path to save the visualization
    """
    # Determine number of rows and columns based on available data
    cols = 2  # At minimum, show original image and one other visualization
    rows = 1
    
    # Count components
    components = []
    if image is not None:
        components.append(('Original', image, 'gray'))
    
    if mask is not None:
        components.append(('Ground Truth', mask, 'gray'))
    
    if predictions is not None:
        components.append(('Prediction', predictions, 'gray'))
    
    if refined_prediction is not None:
        components.append(('Refined', refined_prediction, 'gray'))
    
    if uncertainty is not None:
        components.append(('Uncertainty', uncertainty, 'magma'))
    
    if gradcam is not None:
        components.append(('GradCAM', gradcam, None))  # GradCAM is already colored
    
    # Calculate layout
    cols = min(3, len(components))  # Max 3 columns
    rows = (len(components) + cols - 1) // cols  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    # Preprocess image for visualization if needed
    if isinstance(image, np.ndarray) and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) > 2 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    # Create overlays if both image and masks are available
    if mask is not None and predictions is not None:
        gt_overlay = create_overlay(image, mask, mask_color=[0, 1, 0])  # Green for ground truth
        pred_overlay = create_overlay(image, predictions, mask_color=[1, 0, 0])  # Red for prediction
        components.append(('GT Overlay', gt_overlay, None))
        components.append(('Pred Overlay', pred_overlay, None))
        
        # Add refinement overlay if available
        if refined_prediction is not None:
            refined_overlay = create_overlay(image, refined_prediction, mask_color=[0, 0, 1])  # Blue for refined
            components.append(('Refined Overlay', refined_overlay, None))
    
    # Recalculate layout
    cols = min(3, len(components))
    rows = (len(components) + cols - 1) // cols
    
    # Plot all components
    for i, (title, data, cmap) in enumerate(components):
        plt.subplot(rows, cols, i + 1)
        
        # Ensure data is properly formatted
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if len(data.shape) > 2 and data.shape[0] in [1, 3]:
            data = np.transpose(data, (1, 2, 0))
            
        if len(data.shape) > 2 and data.shape[-1] == 1:
            data = data.squeeze(-1)
            
        plt.imshow(data, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_uncertainty(
    image,
    prediction,
    uncertainty,
    mask=None,
    threshold=0.5,
    save_path=None
):
    """
    Visualize uncertainty in prediction with error analysis.
    
    Args:
        image: Input image
        prediction: Model prediction
        uncertainty: Uncertainty map
        mask: Ground truth mask (optional for error analysis)
        threshold: Threshold to convert probability to binary mask
        save_path: Path to save visualization
    """
    # Preprocess arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Ensure proper shapes
    if len(image.shape) > 2 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) > 2 and image.shape[-1] == 1:
        image = image.squeeze(-1)
        
    if len(prediction.shape) > 2:
        prediction = prediction.squeeze()
    if len(uncertainty.shape) > 2:
        uncertainty = uncertainty.squeeze()
    if mask is not None and len(mask.shape) > 2:
        mask = mask.squeeze()
    
    # Create binary prediction
    binary_prediction = (prediction > threshold).astype(np.float32)
    
    # Create figure
    if mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction with uncertainty
    # Create a cool visualization where high uncertainty areas are colored differently
    prediction_viz = np.zeros((*prediction.shape, 3))
    
    # Areas of high confidence (salt) - blue
    prediction_viz[..., 2] = binary_prediction * (1 - uncertainty)
    # Areas of high confidence (not salt) - black
    # (this is the default)
    
    # Areas of uncertainty - use a gradient
    uncertainty_mask = uncertainty > 0.2  # Adjust threshold as needed
    
    # Red channel - show uncertainty in predicted salt areas
    prediction_viz[..., 0] = binary_prediction * uncertainty * uncertainty_mask
    
    # Green channel - show uncertainty in predicted non-salt areas
    prediction_viz[..., 1] = (1 - binary_prediction) * uncertainty * uncertainty_mask
    
    axes[1].imshow(prediction_viz)
    axes[1].set_title('Prediction with Uncertainty\nBlue: High confidence salt\nRed: Uncertain salt\nGreen: Uncertain background')
    axes[1].axis('off')
    
    # Raw uncertainty heatmap
    uncertainty_img = axes[2].imshow(uncertainty, cmap='magma')
    axes[2].set_title('Uncertainty Heatmap')
    axes[2].axis('off')
    plt.colorbar(uncertainty_img, ax=axes[2], shrink=0.8)
    
    # If we have ground truth, show error analysis
    if mask is not None:
        # True Positive: prediction=1, mask=1 (blue)
        # True Negative: prediction=0, mask=0 (black)
        # False Positive: prediction=1, mask=0 (red)
        # False Negative: prediction=0, mask=1 (green)
        error_viz = np.zeros((*prediction.shape, 3))
        
        # True Positives (blue)
        error_viz[..., 2] = (binary_prediction == 1) & (mask == 1)
        
        # False Positives (red)
        error_viz[..., 0] = (binary_prediction == 1) & (mask == 0)
        
        # False Negatives (green)
        error_viz[..., 1] = (binary_prediction == 0) & (mask == 1)
        
        axes[3].imshow(error_viz)
        axes[3].set_title('Error Analysis\nBlue: True Positive\nRed: False Positive\nGreen: False Negative')
        axes[3].axis('off')
        
        # Ground truth
        axes[4].imshow(mask, cmap='gray')
        axes[4].set_title('Ground Truth')
        axes[4].axis('off')
        
        # Overlay high uncertainty on error map
        high_uncertainty = uncertainty > np.percentile(uncertainty, 90)  # Top 10% uncertainty
        error_uncertainty = error_viz.copy()
        
        # Highlight high uncertainty areas with yellow
        error_uncertainty[high_uncertainty, 0] = 1
        error_uncertainty[high_uncertainty, 1] = 1
        error_uncertainty[high_uncertainty, 2] = 0
        
        axes[5].imshow(error_uncertainty)
        axes[5].set_title('Errors with High Uncertainty\nYellow: High uncertainty regions')
        axes[5].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_uncertainty(
    image,
    prediction,
    uncertainty,
    mask=None,
    threshold=0.5,
    save_path=None
):
    """
    Visualize uncertainty in prediction with error analysis.
    
    Args:
        image: Input image
        prediction: Model prediction
        uncertainty: Uncertainty map
        mask: Ground truth mask (optional for error analysis)
        threshold: Threshold to convert probability to binary mask
        save_path: Path to save visualization
    """
    # Preprocess arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Ensure proper shapes
    if len(image.shape) > 2 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    if len(image.shape) > 2 and image.shape[-1] == 1:
        image = image.squeeze(-1)
        
    if len(prediction.shape) > 2:
        prediction = prediction.squeeze()
    if len(uncertainty.shape) > 2:
        uncertainty = uncertainty.squeeze()
    if mask is not None and len(mask.shape) > 2:
        mask = mask.squeeze()
    
    # Create binary prediction
    binary_prediction = (prediction > threshold).astype(np.float32)
    
    # Create figure
    if mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction with uncertainty
    # Create a cool visualization where high uncertainty areas are colored differently
    prediction_viz = np.zeros((*prediction.shape, 3))
    
    # Areas of high confidence (salt) - blue
    prediction_viz[..., 2] = binary_prediction * (1 - uncertainty)
    # Areas of high confidence (not salt) - black
    # (this is the default)
    
    # Areas of uncertainty - use a gradient
    uncertainty_mask = uncertainty > 0.2  # Adjust threshold as needed
    
    # Red channel - show uncertainty in predicted salt areas
    prediction_viz[..., 0] = binary_prediction * uncertainty * uncertainty_mask
    
    # Green channel - show uncertainty in predicted non-salt areas
    prediction_viz[..., 1] = (1 - binary_prediction) * uncertainty * uncertainty_mask
    
    axes[1].imshow(prediction_viz)
    axes[1].set_title('Prediction with Uncertainty\nBlue: High confidence salt\nRed: Uncertain salt\nGreen: Uncertain background')
    axes[1].axis('off')
    
    # Raw uncertainty heatmap
    uncertainty_img = axes[2].imshow(uncertainty, cmap='magma')
    axes[2].set_title('Uncertainty Heatmap')
    axes[2].axis('off')
    plt.colorbar(uncertainty_img, ax=axes[2], shrink=0.8)
    
    # If we have ground truth, show error analysis
    if mask is not None:
        # True Positive: prediction=1, mask=1 (blue)
        # True Negative: prediction=0, mask=0 (black)
        # False Positive: prediction=1, mask=0 (red)
        # False Negative: prediction=0, mask=1 (green)
        error_viz = np.zeros((*prediction.shape, 3))
        
        # True Positives (blue)
        error_viz[..., 2] = (binary_prediction == 1) & (mask == 1)
        
        # False Positives (red)
        error_viz[..., 0] = (binary_prediction == 1) & (mask == 0)
        
        # False Negatives (green)
        error_viz[..., 1] = (binary_prediction == 0) & (mask == 1)
        
        axes[3].imshow(error_viz)
        axes[3].set_title('Error Analysis\nBlue: True Positive\nRed: False Positive\nGreen: False Negative')
        axes[3].axis('off')
        
        # Ground truth
        axes[4].imshow(mask, cmap='gray')
        axes[4].set_title('Ground Truth')
        axes[4].axis('off')
        
        # Overlay high uncertainty on error map
        high_uncertainty = uncertainty > np.percentile(uncertainty, 90)  # Top 10% uncertainty
        error_uncertainty = error_viz.copy()
        
        # Highlight high uncertainty areas with yellow
        error_uncertainty[high_uncertainty, 0] = 1
        error_uncertainty[high_uncertainty, 1] = 1
        error_uncertainty[high_uncertainty, 2] = 0
        
        axes[5].imshow(error_uncertainty)
        axes[5].set_title('Errors with High Uncertainty\nYellow: High uncertainty regions')
        axes[5].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()