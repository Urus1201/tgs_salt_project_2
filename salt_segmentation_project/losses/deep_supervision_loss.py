import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Union, Any

from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .boundary_loss import BoundaryLoss
from .combined_loss import CombinedLoss


class DeepSupervisionLoss(nn.Module):
    """
    Enhanced combined loss with deep supervision support for salt segmentation.
    
    Features:
    - Supports deep supervision with auxiliary outputs
    - Combines Dice, Focal, Boundary losses for segmentation
    - Handles BCE for classification
    - Applies weighting to auxiliary outputs
    - Provides detailed loss breakdown for monitoring
    
    Deep supervision applies the same loss functions to outputs from intermediate
    layers of the decoder, which helps with gradient flow and improves training.
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 1.0,
        cls_weight: float = 0.5,
        aux_weight: float = 0.4,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.8,
        boundary_theta: float = 0.7,
        aux_decay: bool = True
    ):
        """Initialize deep supervision loss with component weights.
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            boundary_weight: Weight for Boundary loss
            cls_weight: Weight for classification BCE loss
            aux_weight: Base weight for auxiliary outputs
            focal_gamma: Focusing parameter for Focal loss
            focal_alpha: Alpha weighting for Focal loss
            boundary_theta: Distance decay for Boundary loss
            aux_decay: Whether to decay auxiliary weights by depth
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.cls_weight = cls_weight
        self.aux_weight = aux_weight
        self.aux_decay = aux_decay
        
        # Initialize component losses
        self.dice_loss = DiceLoss(square_in_union=True)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss(theta=boundary_theta)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # For backward compatibility
        self.base_loss = CombinedLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            boundary_weight=boundary_weight,
            cls_weight=cls_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            boundary_theta=boundary_theta
        )

    def _calculate_seg_loss(self, seg_logits, seg_target):
        """Calculate segmentation loss components and return individual losses."""
        # Apply sigmoid for dice and boundary losses (not for focal which expects logits)
        seg_pred = torch.sigmoid(seg_logits)
        
        dice_loss = self.dice_loss(seg_pred, seg_target)
        focal_loss = self.focal_loss(seg_logits, seg_target)
        boundary_loss = self.boundary_loss(seg_logits, seg_target)
        
        # Combined segmentation loss
        seg_loss = (
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss +
            self.boundary_weight * boundary_loss
        )
        
        return seg_loss, {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'boundary': boundary_loss.item()
        }

    def forward(
        self,
        outputs: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]],
        seg_target: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate deep supervision loss.
        
        Args:
            outputs: Either a tuple of (seg_logits, cls_logits) or a dictionary with keys:
                    - 'seg_logits': Main segmentation output (B, 1, H, W)
                    - 'cls_logits': Classification output (B, 1)
                    - 'aux_outputs': List of auxiliary segmentation outputs
            seg_target: Ground truth masks (B, 1, H, W)
            cls_target: Ground truth labels (B, 1)
            
        Returns:
            tuple:
                - total_loss: Combined weighted loss
                - loss_dict: Dictionary with individual loss components
        """
        # Handle different output formats
        if isinstance(outputs, dict):
            seg_logits = outputs['seg_logits']
            cls_logits = outputs['cls_logits']
            aux_outputs = outputs.get('aux_outputs', None)
        else:
            seg_logits, cls_logits = outputs
            aux_outputs = None
            
        # For backward compatibility with non-deep supervision models
        if aux_outputs is None:
            return self.base_loss(seg_logits, seg_target, cls_logits, cls_target)
        
        # Calculate main segmentation loss
        main_seg_loss, seg_losses = self._calculate_seg_loss(seg_logits, seg_target)
        
        # Calculate classification loss
        cls_loss = self.bce_loss(cls_logits, cls_target)
        
        # Start with main losses
        total_loss = main_seg_loss + self.cls_weight * cls_loss
        
        # Initialize loss dictionary
        loss_dict = {
            'main_seg': main_seg_loss.item(),
            'cls': cls_loss.item(),
            **{f'main_{k}': v for k, v in seg_losses.items()}
        }
        
        # Add auxiliary losses with decaying weights if enabled
        aux_loss_sum = 0.0
        for i, aux_output in enumerate(aux_outputs):
            # Ensure auxiliary output has same size as target
            if aux_output.shape[-2:] != seg_target.shape[-2:]:
                aux_output = F.interpolate(
                    aux_output, 
                    size=seg_target.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
            # Calculate loss for this auxiliary output
            aux_seg_loss, aux_seg_losses = self._calculate_seg_loss(aux_output, seg_target)
            
            # Apply decaying weight based on depth if enabled
            if self.aux_decay:
                depth_factor = 0.5 ** (len(aux_outputs) - i)
            else:
                depth_factor = 1.0
                
            weight = self.aux_weight * depth_factor
            aux_loss = weight * aux_seg_loss
            aux_loss_sum += aux_loss
            
            # Add to loss dictionary
            loss_dict[f'aux{i}_seg'] = aux_seg_loss.item()
            for k, v in aux_seg_losses.items():
                loss_dict[f'aux{i}_{k}'] = v
        
        # Add auxiliary losses to total
        total_loss = total_loss + aux_loss_sum
        loss_dict['aux_total'] = aux_loss_sum.item()
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict

    @torch.no_grad()
    def compute_metrics(
        self,
        outputs: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]],
        seg_target: torch.Tensor,
        cls_target: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            outputs: Either a tuple of (seg_logits, cls_logits) or a dictionary with keys:
                    - 'seg_logits': Main segmentation output
                    - 'cls_logits': Classification output
                    - 'aux_outputs': List of auxiliary segmentation outputs (ignored for metrics)
            seg_target: Ground truth masks (B, 1, H, W)
            cls_target: Ground truth labels (B, 1)
            threshold: Threshold for binary segmentation
            
        Returns:
            Dictionary of metrics
        """
        # Extract main outputs
        if isinstance(outputs, dict):
            seg_logits = outputs['seg_logits']
            cls_logits = outputs['cls_logits']
        else:
            seg_logits, cls_logits = outputs
            
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
        empty_masks = (seg_target.sum(dim=(1, 2, 3)) == 0).float()
        if empty_masks.sum() > 0:
            # Calculate TN IoU for empty masks (predict empty -> IoU=1, predict something -> IoU=0)
            empty_iou = ((seg_pred.sum((1, 2, 3)) == 0).float() * empty_masks).sum() / max(empty_masks.sum(), 1)
            # Adjust mean IoU for empty masks
            if empty_masks.sum() == empty_masks.numel():  # All masks are empty
                mean_iou = empty_iou
        
        # Calculate classification accuracy
        cls_acc = (cls_pred == cls_target).float().mean().item()
        
        # Calculate more metrics
        tp = (seg_pred * seg_target).sum((1, 2, 3))
        fp = seg_pred.sum((1, 2, 3)) - tp
        fn = seg_target.sum((1, 2, 3)) - tp
        
        precision = (tp / (tp + fp + 1e-6)).mean().item()
        recall = (tp / (tp + fn + 1e-6)).mean().item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        
        return {
            'iou': mean_iou,
            'dice': mean_dice,
            'cls_acc': cls_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }