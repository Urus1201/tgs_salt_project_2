import os
import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

from .predictor import Predictor


def mask_to_rle(mask: np.ndarray) -> str:
    """Convert binary mask to RLE format for Kaggle submission.
    
    Args:
        mask: Binary mask array (H, W)
        
    Returns:
        RLE encoded string
    """
    # Flatten mask in column-major order (Fortran style)
    pixels = mask.flatten(order='F')
    
    # Add sentinels to beginning and end
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Find runs of same value
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    
    # Convert to start positions and lengths
    runs[1::2] -= runs[::2]
    
    # Convert to string
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle_string: str, shape: tuple) -> np.ndarray:
    """Convert RLE string back to binary mask (for validation).
    
    Args:
        rle_string: RLE encoded string
        shape: Shape of the mask (H, W)
        
    Returns:
        Binary mask array
    """
    if rle_string == '':
        return np.zeros(shape)
        
    # Parse runs from string
    runs = np.array([int(x) for x in rle_string.split()])
    runs[1::2] += runs[::2]
    
    # Create empty array and fill runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(runs[::2], runs[1::2]):
        img[start:end] = 1
        
    # Reshape to original size (column-major order)
    return img.reshape(shape, order='F')


class SubmissionGenerator:
    """Handle test set prediction and Kaggle submission generation."""
    def __init__(
        self,
        predictor: Predictor,
        test_loader: DataLoader,
        submission_path: str
    ):
        """Initialize submission generator.
        
        Args:
            predictor: Trained predictor instance
            test_loader: DataLoader for test set
            submission_path: Path to save submission CSV
        """
        self.predictor = predictor
        self.test_loader = test_loader
        self.submission_path = submission_path
        
    def generate(
        self,
        uncertainty_threshold: float = None,
        empty_threshold: float = 0.9
    ):
        """Generate predictions and create submission file.
        
        Args:
            uncertainty_threshold: Threshold for uncertainty-based refinement
            empty_threshold: Classification confidence threshold for empty masks
        """
        print("Generating predictions...")
        predictions_list = []
        image_ids = []
        
        for images, batch_ids in self.test_loader:
            # Get predictions with uncertainty
            predictions = self.predictor.predict_batch(images)
            
            # Get binary masks
            binary_masks = self.predictor.get_binary_prediction(
                predictions,
                uncertainty_threshold=uncertainty_threshold
            )
            
            # Handle empty mask cases using classifier
            cls_probs = predictions['cls_pred']
            no_salt_conf = (1 - cls_probs) > empty_threshold
            binary_masks[no_salt_conf] = 0
            
            # Convert to numpy and add to list
            masks_np = binary_masks.cpu().numpy()
            predictions_list.extend([m[0] for m in masks_np])
            image_ids.extend(batch_ids)
            
        print("Converting to RLE format...")
        rle_strings = []
        for mask in predictions_list:
            rle = mask_to_rle(mask)
            rle_strings.append(rle)
            
        # Create submission DataFrame
        df = pd.DataFrame({
            'id': image_ids,
            'rle_mask': rle_strings
        })
        
        # Save submission
        df.to_csv(self.submission_path, index=False)
        print(f"Submission saved to {self.submission_path}")
        
        # Print statistics
        empty_masks = df['rle_mask'].str.len() == 0
        print(f"\nSubmission statistics:")
        print(f"Total images: {len(df)}")
        print(f"Empty masks: {empty_masks.sum()} ({empty_masks.mean()*100:.1f}%)")
        
    @staticmethod
    def validate_rle(
        rle_string: str,
        original_mask: np.ndarray,
        mask_shape: tuple = (101, 101)
    ) -> float:
        """Validate RLE encoding by decoding and comparing to original mask.
        
        Args:
            rle_string: RLE encoded string
            original_mask: Original binary mask
            mask_shape: Shape of the mask
            
        Returns:
            IoU between original and decoded masks
        """
        decoded_mask = rle_to_mask(rle_string, mask_shape)
        
        intersection = (decoded_mask & original_mask).sum()
        union = (decoded_mask | original_mask).sum()
        
        if union == 0:  # Both masks are empty
            return 1.0  # Perfect match
        iou = intersection / union
        return iou