import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .boundary_loss import BoundaryLoss


class CombinedLoss(nn.Module):
    """Combined loss for TGS Salt Segmentation.
    Combines Dice, Focal, Boundary losses for segmentation and BCE for classification.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 1.0,
        cls_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.8, #0.25
        boundary_theta: float = 0.7
    ):
        """Initialize combined loss with component weights.
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            boundary_weight: Weight for Boundary loss
            cls_weight: Weight for classification BCE loss
            focal_gamma: Focusing parameter for Focal loss
            focal_alpha: Alpha weighting for Focal loss
            boundary_theta: Distance decay for Boundary loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.cls_weight = cls_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss(square_in_union=True)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss(theta=boundary_theta)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate combined loss.
        
        Args:
            seg_logits: Segmentation logits (B, 1, H, W)
            seg_target: Ground truth masks (B, 1, H, W)
            cls_logits: Classification logits (B, 1)
            cls_target: Ground truth labels (B, 1)
            
        Returns:
            tuple:
                - total_loss: Combined weighted loss
                - loss_dict: Dictionary with individual loss components
        """
        # Calculate segmentation losses
        seg_pred = torch.sigmoid(seg_logits)
        
        dice_loss = self.dice_loss(seg_pred, seg_target)
        focal_loss = self.focal_loss(seg_logits, seg_target)
        boundary_loss = self.boundary_loss(seg_logits, seg_target)
        
        # Calculate classification loss
        cls_loss = self.bce_loss(cls_logits, cls_target)
        
        # Combine losses with weights
        total_loss = (
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss +
            self.boundary_weight * boundary_loss +
            self.cls_weight * cls_loss
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'boundary': boundary_loss.item(),
            'cls': cls_loss.item()
        }
        
        return total_loss, loss_dict

    @torch.no_grad()
    def compute_metrics(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_target: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            seg_logits: Segmentation logits (B, 1, H, W)
            seg_target: Ground truth masks (B, 1, H, W)
            cls_logits: Classification logits (B, 1)
            cls_target: Ground truth labels (B, 1)
            threshold: Threshold for binary segmentation
            
        Returns:
            Dictionary of metrics
        """
        # Get binary predictions
        seg_pred = (torch.sigmoid(seg_logits) > threshold).float()
        cls_pred = (torch.sigmoid(cls_logits) > threshold).float()
        
        # Calculate IoU for segmentation
        intersection = (seg_pred * seg_target).sum((1, 2, 3))
        union = seg_pred.sum((1, 2, 3)) + seg_target.sum((1, 2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        mean_iou = iou.mean().item()
        
        # Calculate Dice score
        dice = (2 * intersection + 1e-6) / (union + intersection + 1e-6)
        mean_dice = dice.mean().item()
        
        # Handle edge case for empty ground truth masks
        if (seg_target.sum(dim=(1, 2, 3)) == 0).all():
            if (torch.sigmoid(seg_logits) < 0.5).all():
                mean_iou = 1.0
                mean_dice = 1.0
            else:
                mean_iou = 0.0
                mean_dice = 0.0
        
        # Calculate classification accuracy
        cls_acc = (cls_pred == cls_target).float().mean().item()
        
        return {
            'iou': mean_iou,
            'dice': mean_dice,
            'cls_acc': cls_acc
        }