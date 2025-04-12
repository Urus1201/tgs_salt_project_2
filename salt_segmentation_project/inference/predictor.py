import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import torch.nn.functional as F

from models.segmentation_model import SaltSegmentationModel
from data.transforms import get_validation_augmentation


class Predictor:
    """Predictor class for TGS Salt Segmentation inference with uncertainty."""
    def __init__(
        self,
        model: SaltSegmentationModel,
        device: str = 'cuda',
        threshold: float = 0.5,
        use_tta: bool = True,
        use_mc_dropout: bool = True,
        mc_samples: int = 10
    ):
        """Initialize predictor.
        
        Args:
            model: Trained model instance
            device: Device to run inference on
            threshold: Probability threshold for binary segmentation
            use_tta: Whether to use test-time augmentation
            use_mc_dropout: Whether to use Monte Carlo dropout
            mc_samples: Number of MC samples if using dropout
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.use_tta = use_tta
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        
    def tta_predict(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use test-time augmentation for prediction.
        
        Uses horizontal and vertical flips as augmentations.
        Predictions are averaged, and variance is computed.
        
        Args:
            image: Input image tensor (1, C, H, W)
            
        Returns:
            Tuple containing:
            - Mean prediction
            - Variance of predictions
        """
        self.model.eval()
        predictions = []
        
        # Original image
        seg_logits, _ = self.model(image)
        predictions.append(torch.sigmoid(seg_logits))
        
        # Horizontal flip
        flipped_h = torch.flip(image, dims=[3])
        seg_logits, _ = self.model(flipped_h)
        pred_h = torch.sigmoid(seg_logits)
        predictions.append(torch.flip(pred_h, dims=[3]))
        
        # Vertical flip
        flipped_v = torch.flip(image, dims=[2])
        seg_logits, _ = self.model(flipped_v)
        pred_v = torch.sigmoid(seg_logits)
        predictions.append(torch.flip(pred_v, dims=[2]))
        
        # Both flips
        flipped_hv = torch.flip(image, dims=[2, 3])
        seg_logits, _ = self.model(flipped_hv)
        pred_hv = torch.sigmoid(seg_logits)
        predictions.append(torch.flip(pred_hv, dims=[2, 3]))
        
        # Stack and compute statistics
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        var_pred = predictions.var(dim=0)
        
        return mean_pred, var_pred
        
    def mc_dropout_predict(
        self,
        image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use Monte Carlo dropout for uncertainty estimation.
        
        Args:
            image: Input image tensor (1, C, H, W)
            
        Returns:
            Tuple containing:
            - Mean segmentation prediction
            - Segmentation variance (uncertainty)
            - Mean classification prediction
            - Classification variance
        """
        self.model.enable_mc_dropout()
        predictions_seg = []
        predictions_cls = []
        
        # Multiple forward passes with dropout
        for _ in range(self.mc_samples):
            seg_logits, cls_logits = self.model(image)
            predictions_seg.append(torch.sigmoid(seg_logits))
            predictions_cls.append(torch.sigmoid(cls_logits))
            
        # Stack predictions
        predictions_seg = torch.stack(predictions_seg, dim=0)
        predictions_cls = torch.stack(predictions_cls, dim=0)
        
        # Compute statistics
        mean_seg = predictions_seg.mean(dim=0)
        var_seg = predictions_seg.var(dim=0)
        mean_cls = predictions_cls.mean(dim=0)
        var_cls = predictions_cls.var(dim=0)
        
        self.model.disable_mc_dropout()
        return mean_seg, var_seg, mean_cls, var_cls
        
    @torch.no_grad()
    def predict_single(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Predict single image with uncertainty estimation.
        
        Args:
            image: Input image tensor (1, C, H, W)
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        image = image.to(self.device)
        
        # Initialize predictions
        seg_pred = None
        seg_var = None
        cls_pred = None
        cls_var = None
        
        # TTA predictions
        if self.use_tta:
            seg_pred, seg_var = self.tta_predict(image)
        
        # MC dropout predictions
        if self.use_mc_dropout:
            mc_seg_pred, mc_seg_var, mc_cls_pred, mc_cls_var = self.mc_dropout_predict(image)
            
            if seg_pred is None:
                seg_pred = mc_seg_pred
                seg_var = mc_seg_var
            else:
                # Combine TTA and MC dropout predictions using weighted average
                # Weight by inverse variance (more weight to more certain predictions)
                epsilon = 1e-6  # To avoid division by zero
                w1 = 1.0 / (seg_var + epsilon)
                w2 = 1.0 / (mc_seg_var + epsilon)
                seg_pred = (w1 * seg_pred + w2 * mc_seg_pred) / (w1 + w2)
                # Take maximum variance as combined uncertainty
                seg_var = torch.maximum(seg_var, mc_seg_var)
                
            cls_pred = mc_cls_pred
            cls_var = mc_cls_var
        
        # If neither TTA nor MC dropout, do regular forward pass
        if seg_pred is None:
            seg_logits, cls_logits = self.model(image)
            seg_pred = torch.sigmoid(seg_logits)
            cls_pred = torch.sigmoid(cls_logits)
            seg_var = torch.zeros_like(seg_pred)
            cls_var = torch.zeros_like(cls_pred)
        
        return {
            'seg_pred': seg_pred,
            'seg_var': seg_var,
            'cls_pred': cls_pred,
            'cls_var': cls_var if cls_var is not None else torch.zeros_like(cls_pred)
        }
        
    def predict_batch(
        self,
        images: torch.Tensor,
        progress_bar: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Predict batch of images efficiently.
        
        Args:
            images: Batch of images (B, C, H, W)
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            if self.use_tta:
                # Initialize predictions
                all_seg_preds = []
                all_cls_preds = []
                
                # Original images
                seg_logits, cls_logits = self.model(images)
                all_seg_preds.append(torch.sigmoid(seg_logits))
                all_cls_preds.append(torch.sigmoid(cls_logits))
                
                # Horizontal flip
                flipped_h = torch.flip(images, dims=[3])
                seg_logits, cls_logits = self.model(flipped_h)
                all_seg_preds.append(torch.flip(torch.sigmoid(seg_logits), dims=[3]))
                all_cls_preds.append(torch.sigmoid(cls_logits))
                
                # Vertical flip
                flipped_v = torch.flip(images, dims=[2])
                seg_logits, cls_logits = self.model(flipped_v)
                all_seg_preds.append(torch.flip(torch.sigmoid(seg_logits), dims=[2]))
                all_cls_preds.append(torch.sigmoid(cls_logits))
                
                # Both flips
                flipped_hv = torch.flip(images, dims=[2, 3])
                seg_logits, cls_logits = self.model(flipped_hv)
                all_seg_preds.append(torch.flip(torch.sigmoid(seg_logits), dims=[2, 3]))
                all_cls_preds.append(torch.sigmoid(cls_logits))
                
                # Stack and compute statistics
                all_seg_preds = torch.stack(all_seg_preds, dim=0)  # (num_aug, B, 1, H, W)
                all_cls_preds = torch.stack(all_cls_preds, dim=0)  # (num_aug, B, 1)
                
                seg_pred = all_seg_preds.mean(dim=0)  # (B, 1, H, W)
                seg_var = all_seg_preds.var(dim=0)    # (B, 1, H, W)
                cls_pred = all_cls_preds.mean(dim=0)  # (B, 1)
                cls_var = all_cls_preds.var(dim=0)    # (B, 1)
                
            if self.use_mc_dropout:
                # Multiple forward passes with dropout
                mc_seg_preds = []
                mc_cls_preds = []
                
                self.model.enable_mc_dropout()
                for _ in range(self.mc_samples):
                    seg_logits, cls_logits = self.model(images)
                    mc_seg_preds.append(torch.sigmoid(seg_logits))
                    mc_cls_preds.append(torch.sigmoid(cls_logits))
                self.model.disable_mc_dropout()
                
                # Stack predictions
                mc_seg_preds = torch.stack(mc_seg_preds, dim=0)  # (num_samples, B, 1, H, W)
                mc_cls_preds = torch.stack(mc_cls_preds, dim=0)  # (num_samples, B, 1)
                
                mc_seg_pred = mc_seg_preds.mean(dim=0)  # (B, 1, H, W)
                mc_seg_var = mc_seg_preds.var(dim=0)    # (B, 1, H, W)
                mc_cls_pred = mc_cls_preds.mean(dim=0)  # (B, 1)
                mc_cls_var = mc_cls_preds.var(dim=0)    # (B, 1)
                
                if 'seg_pred' in locals():
                    # Combine TTA and MC dropout predictions
                    epsilon = 1e-6
                    w1 = 1.0 / (seg_var + epsilon)
                    w2 = 1.0 / (mc_seg_var + epsilon)
                    seg_pred = (w1 * seg_pred + w2 * mc_seg_pred) / (w1 + w2)
                    seg_var = torch.maximum(seg_var, mc_seg_var)
                    cls_pred = mc_cls_pred
                    cls_var = mc_cls_var
                else:
                    seg_pred = mc_seg_pred
                    seg_var = mc_seg_var
                    cls_pred = mc_cls_pred
                    cls_var = mc_cls_var
            
            if not (self.use_tta or self.use_mc_dropout):
                # Regular forward pass
                seg_logits, cls_logits = self.model(images)
                seg_pred = torch.sigmoid(seg_logits)
                cls_pred = torch.sigmoid(cls_logits)
                seg_var = torch.zeros_like(seg_pred)
                cls_var = torch.zeros_like(cls_pred)
        
        return {
            'seg_pred': seg_pred,
            'seg_var': seg_var,
            'cls_pred': cls_pred,
            'cls_var': cls_var
        }
        
    def get_binary_prediction(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: Optional[float] = None,
        cls_low_threshold: float = 0.1,
        cls_medium_threshold: float = 0.4,
        seg_threshold_adjustment: float = 1.5  # Multiplier for threshold
    ) -> torch.Tensor:
        """Get binary mask from predictions using classification guidance.
        
        Args:
            predictions: Dictionary with predictions from predict_single/predict_batch
            threshold: Optional override for default threshold
            cls_low_threshold: Classification threshold below which to zero out masks
            cls_medium_threshold: Classification threshold below which to adjust segmentation threshold
            seg_threshold_adjustment: How much to adjust segmentation threshold for low confidence
            
        Returns:
            Binary mask tensor
        """
        if threshold is None:
            threshold = self.threshold
        
        # Get classification confidence
        cls_confidence = predictions['cls_pred']
        batch_size = cls_confidence.size(0)
        seg_masks = []
        
        # Process each image in the batch
        for i in range(batch_size):
            single_confidence = cls_confidence[i].item()
            single_seg_pred = predictions['seg_pred'][i:i+1]  # Keep batch dimension
            
            if single_confidence < cls_low_threshold:
                # Very low confidence of salt presence - return empty mask
                seg_masks.append(torch.zeros_like(single_seg_pred))
                
            elif single_confidence < cls_medium_threshold:
                # Low to medium confidence - use higher threshold to be more conservative
                adjusted_threshold = threshold * seg_threshold_adjustment
                seg_masks.append((single_seg_pred > adjusted_threshold).float())
                
            else:
                # High confidence - use normal threshold
                seg_masks.append((single_seg_pred > threshold).float())
        
        # Combine results back into a batch
        if batch_size > 1:
            return torch.cat(seg_masks, dim=0)
        else:
            return seg_masks[0]