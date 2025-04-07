import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dealing with class imbalance.
    Based on: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (salt pixels)
            gamma: Focusing parameter that reduces loss contribution from easy examples
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Focal Loss.
        
        Args:
            pred_logits: Raw model output before sigmoid (B, 1, H, W)
            target: Binary ground truth masks (B, 1, H, W)
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        pred = torch.sigmoid(pred_logits)
        
        # Calculate standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none'
        )
        
        # Get probabilities for ground truth class
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Calculate focal weights
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = torch.where(
                target == 1,
                torch.ones_like(target) * self.alpha,
                torch.ones_like(target) * (1 - self.alpha)
            )
            focal_weight = alpha_weight * focal_weight
            
        # Calculate focal loss
        loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss