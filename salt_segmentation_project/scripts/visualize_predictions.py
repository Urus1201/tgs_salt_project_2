import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.submission import rle_to_mask

def visualize_predictions(submission_path, test_images_dir, num_samples=5, save_dir=None):
    """Visualize sample predictions from the submission file.
    
    Args:
        submission_path: Path to submission CSV file
        test_images_dir: Directory containing test images
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save visualizations
    """
    # Load submission file
    submission_df = pd.read_csv(submission_path)
    
    # Get samples with different characteristics
    empty_masks = submission_df[submission_df['rle_mask'].str.len() == 0]
    non_empty_masks = submission_df[submission_df['rle_mask'].str.len() > 0]
    
    # Select diverse samples
    samples = []
    if len(empty_masks) > 0:
        samples.extend(empty_masks.sample(min(2, len(empty_masks))).to_dict('records'))
    if len(non_empty_masks) > 0:
        samples.extend(non_empty_masks.sample(min(num_samples - len(samples), len(non_empty_masks))).to_dict('records'))
    
    for i, sample in enumerate(samples):
        image_id = sample['id']
        rle_mask = sample['rle_mask']
        
        # Load test image
        image_path = os.path.join(test_images_dir, f"{image_id}.png")
        image = np.array(Image.open(image_path))
        
        # Convert RLE to mask
        mask = rle_to_mask(rle_mask, shape=image.shape)
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot original image
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title(f'Test Image (ID: {image_id})')
        plt.axis('off')
        
        # Plot predicted mask
        plt.subplot(122)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Predicted Mask ({"Empty" if len(rle_mask)==0 else "Has Salt"})')
        plt.axis('off')
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'prediction_{i+1}.png'))
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    # Paths
    submission_path = os.path.join(project_root, 'checkpoints/submission.csv')
    test_images_dir = os.path.join(project_root, '../data/test/images')
    save_dir = os.path.join(project_root, 'checkpoints/visualization')
    
    # Visualize predictions
    visualize_predictions(
        submission_path=submission_path,
        test_images_dir=test_images_dir,
        num_samples=5,
        save_dir=save_dir
    )