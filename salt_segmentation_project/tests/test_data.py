import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A

from ..data.dataset import SaltDataset
from ..data.transforms import get_training_augmentation, get_validation_augmentation


class TestDataPipeline(unittest.TestCase):
    """Test suite for data pipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.img_size = 101
        
        # Create temporary test data
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        
        # Create dummy CSV files
        self.train_csv = pd.DataFrame({
            'id': ['test1', 'test2', 'test3'],
            'coverage': [0.0, 0.3, 0.7]
        })
        self.train_csv.to_csv(self.test_dir / 'train.csv', index=False)
        
        self.depths_csv = pd.DataFrame({
            'id': ['test1', 'test2', 'test3'],
            'z': [100, 200, 300]
        })
        self.depths_csv.to_csv(self.test_dir / 'depths.csv', index=False)
        
        # Create dummy image and mask directories
        self.img_dir = self.test_dir / 'images'
        self.mask_dir = self.test_dir / 'masks'
        self.img_dir.mkdir(exist_ok=True)
        self.mask_dir.mkdir(exist_ok=True)
        
        # Create dummy images and masks
        for i in range(3):
            img = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
            mask = np.random.randint(0, 2, (101, 101), dtype=np.uint8)
            np.save(self.img_dir / f'test{i+1}.npy', img)
            np.save(self.mask_dir / f'test{i+1}.npy', mask)
    
    def test_dataset_creation(self):
        """Test dataset initialization."""
        dataset = SaltDataset(
            csv_file=str(self.test_dir / 'train.csv'),
            image_dir=str(self.img_dir),
            mask_dir=str(self.mask_dir),
            depths_csv=str(self.test_dir / 'depths.csv'),
            transform=None,
            use_2_5d=True,
            mode='train'
        )
        
        self.assertEqual(len(dataset), 3)
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = SaltDataset(
            csv_file=str(self.test_dir / 'train.csv'),
            image_dir=str(self.img_dir),
            mask_dir=str(self.mask_dir),
            depths_csv=str(self.test_dir / 'depths.csv'),
            transform=None,
            use_2_5d=True,
            mode='train'
        )
        
        image, mask = dataset[0]
        
        # Check shapes
        self.assertEqual(image.shape, (3, self.img_size, self.img_size))
        self.assertEqual(mask.shape, (1, self.img_size, self.img_size))
        
        # Check value ranges
        self.assertTrue(torch.all(image >= 0))
        self.assertTrue(torch.all(image <= 1))
        self.assertTrue(torch.all(mask >= 0))
        self.assertTrue(torch.all(mask <= 1))
        
    def test_training_augmentations(self):
        """Test training augmentations."""
        transform = get_training_augmentation()
        
        # Create dummy data
        image = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
        mask = np.random.randint(0, 2, (101, 101), dtype=np.uint8)
        
        # Apply augmentation
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Check shapes
        self.assertEqual(aug_image.shape, (101, 101))
        self.assertEqual(aug_mask.shape, (101, 101))
        
        # Check value ranges
        self.assertTrue(np.all(aug_image >= 0))
        self.assertTrue(np.all(aug_image <= 255))
        self.assertTrue(np.all(aug_mask >= 0))
        self.assertTrue(np.all(aug_mask <= 1))
        
    def test_validation_augmentations(self):
        """Test validation augmentations."""
        transform = get_validation_augmentation()
        
        # Create dummy data
        image = np.random.randint(0, 255, (101, 101), dtype=np.uint8)
        mask = np.random.randint(0, 2, (101, 101), dtype=np.uint8)
        
        # Apply augmentation
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Check shapes are unchanged (no resizing in validation)
        self.assertEqual(aug_image.shape, (101, 101))
        self.assertEqual(aug_mask.shape, (101, 101))
        
        # Verify normalization
        self.assertTrue(np.all(aug_image >= 0))
        self.assertTrue(np.all(aug_image <= 1))
        
    def test_2_5d_input(self):
        """Test 2.5D input creation."""
        dataset = SaltDataset(
            csv_file=str(self.test_dir / 'train.csv'),
            image_dir=str(self.img_dir),
            mask_dir=str(self.mask_dir),
            depths_csv=str(self.test_dir / 'depths.csv'),
            transform=None,
            use_2_5d=True,
            mode='train'
        )
        
        image, _ = dataset[1]  # Middle sample
        
        # Check that we get 3 channels
        self.assertEqual(image.shape[0], 3)
        
        # Check that channels are different (representing different depths)
        self.assertFalse(torch.allclose(image[0], image[1]))
        self.assertFalse(torch.allclose(image[1], image[2]))
        
    def test_augmentation_consistency(self):
        """Test that augmentations preserve mask-image alignment."""
        transform = get_training_augmentation()
        
        # Create test pattern
        image = np.zeros((101, 101), dtype=np.uint8)
        mask = np.zeros((101, 101), dtype=np.uint8)
        
        # Add test pattern
        image[40:60, 40:60] = 255
        mask[40:60, 40:60] = 1
        
        # Apply augmentation
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image']
        aug_mask = augmented['mask']
        
        # Check alignment (bright regions in image should correspond to mask regions)
        img_bright = aug_image > 127
        mask_positive = aug_mask > 0.5
        
        overlap = np.sum(img_bright & mask_positive) / np.sum(mask_positive)
        self.assertGreater(overlap, 0.8)  # Allow some margin due to interpolation
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.test_dir)


if __name__ == '__main__':
    unittest.main()