import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation with optional class weights."""
    def __init__(self, smooth: float = 1.0, square_in_union: bool = True):
        """Initialize DiceLoss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero
            square_in_union: Whether to square predictions in denominator (orig. Dice)
        """
        super().__init__()
        self.smooth = smooth
        self.square_in_union = square_in_union

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss.
        
        Args:
            pred: Predicted probabilities (after sigmoid) (B, 1, H, W)
            target: Target binary masks (B, 1, H, W)
            
        Returns:
            Dice loss value
        """
        # Flatten predictions and targets
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection
        intersection = (pred_flat * target_flat).sum(-1)
        
        if self.square_in_union:
            denominator = (pred_flat * pred_flat).sum(-1) + (target_flat * target_flat).sum(-1)
        else:
            denominator = pred_flat.sum(-1) + target_flat.sum(-1)
            
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        
        # Return Dice loss
        return 1. - dice.mean()