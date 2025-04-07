import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional
import cv2


class BoundaryLoss(nn.Module):
    """Boundary loss for segmentation focusing on region interfaces.
    Based on: "Boundary loss for highly unbalanced segmentation" (https://arxiv.org/abs/1812.07032)
    """
    def __init__(
        self,
        theta: float = 0.7,
        reduction: str = 'mean',
        max_dist: Optional[float] = None
    ):
        """Initialize Boundary Loss.
        
        Args:
            theta: Distance weight decay factor
            reduction: 'mean' or 'sum'
            max_dist: Maximum distance to consider for boundary weighting
        """
        super().__init__()
        self.theta = theta
        self.reduction = reduction
        self.max_dist = max_dist
        
        # Cache for distance maps to avoid recomputing for same masks
        self.cache = {}
        self.cache_size = 1000  # Maximum number of cached maps
        
    def compute_distance_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute distance transform for boundary weighting.
        
        Args:
            mask: Binary mask tensor (B, 1, H, W)
            
        Returns:
            Distance weight map emphasizing boundaries
        """
        B, _, H, W = mask.shape
        distance_maps = torch.zeros_like(mask)
        
        for b in range(B):
            # Convert to numpy for distance transform
            mask_np = mask[b, 0].cpu().numpy().astype(bool)
            
            # Try to get from cache first
            cache_key = hash(mask_np.tobytes())
            if cache_key in self.cache:
                distance_maps[b, 0] = self.cache[cache_key].to(mask.device)
                continue
                
            # Get boundaries using morphological operations
            kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.morphologyEx(
                mask_np.astype(np.uint8),
                cv2.MORPH_GRADIENT,
                kernel
            ).astype(bool)
            
            # Compute distance from boundary
            dist = distance_transform_edt(~boundary)
            
            # Optional distance clipping
            if self.max_dist is not None:
                dist = np.clip(dist, 0, self.max_dist)
            
            # Convert to weight map with exponential decay
            weight_map = np.exp(-dist * self.theta)
            
            # Convert to tensor and cache
            weight_tensor = torch.from_numpy(weight_map).float()
            distance_maps[b, 0] = weight_tensor.to(mask.device)
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = weight_tensor
            
        return distance_maps
        
    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate boundary-weighted loss.
        
        Args:
            pred_logits: Raw model output before sigmoid (B, 1, H, W)
            target: Binary ground truth masks (B, 1, H, W)
            
        Returns:
            Boundary loss value
        """
        # Compute distance-based weights
        weights = self.compute_distance_map(target)
        
        # Calculate weighted BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            target,
            reduction='none'
        )
        
        # Apply boundary weights
        weighted_loss = weights * bce_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss