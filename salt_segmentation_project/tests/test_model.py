import unittest
import torch
import numpy as np
from pathlib import Path

from ..models.segmentation_model import SaltSegmentationModel
from ..losses.combined_loss import CombinedLoss
from ..inference.predictor import Predictor


class TestSaltSegmentation(unittest.TestCase):
    """Test suite for core model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.img_size = 101
        
        # Create model
        self.model = SaltSegmentationModel(
            img_size=self.img_size,
            in_channels=3,  # 2.5D input
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create loss function
        self.criterion = CombinedLoss(
            dice_weight=1.0,
            focal_weight=1.0,
            boundary_weight=1.0,
            cls_weight=0.5
        )
        
        # Create predictor
        self.predictor = Predictor(
            model=self.model,
            device=self.device,
            threshold=0.5,
            use_tta=True,
            use_mc_dropout=True,
            mc_samples=5
        )
        
    def test_model_forward(self):
        """Test model forward pass."""
        # Create dummy input
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        
        # Forward pass
        seg_logits, cls_logits = self.model(x)
        
        # Check output shapes
        self.assertEqual(
            seg_logits.shape,
            (self.batch_size, 1, self.img_size, self.img_size)
        )
        self.assertEqual(cls_logits.shape, (self.batch_size, 1))
        
    def test_loss_computation(self):
        """Test loss computation."""
        # Create dummy data
        seg_logits = torch.randn(
            self.batch_size, 1, self.img_size, self.img_size
        ).to(self.device)
        seg_target = torch.randint(
            0, 2, (self.batch_size, 1, self.img_size, self.img_size)
        ).float().to(self.device)
        cls_logits = torch.randn(self.batch_size, 1).to(self.device)
        cls_target = torch.randint(0, 2, (self.batch_size, 1)).float().to(self.device)
        
        # Compute loss
        loss, loss_dict = self.criterion(
            seg_logits, seg_target,
            cls_logits, cls_target
        )
        
        # Check loss values
        self.assertGreater(loss.item(), 0)
        for k, v in loss_dict.items():
            self.assertGreater(v, 0)
            
    def test_predictor(self):
        """Test prediction with uncertainty estimation."""
        # Create dummy input
        x = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        
        # Get prediction
        predictions = self.predictor.predict_single(x)
        
        # Check prediction components
        self.assertIn('seg_pred', predictions)
        self.assertIn('seg_var', predictions)
        self.assertIn('cls_pred', predictions)
        self.assertIn('cls_var', predictions)
        
        # Check shapes
        self.assertEqual(
            predictions['seg_pred'].shape,
            (1, 1, self.img_size, self.img_size)
        )
        self.assertEqual(
            predictions['seg_var'].shape,
            (1, 1, self.img_size, self.img_size)
        )
        self.assertEqual(predictions['cls_pred'].shape, (1, 1))
        self.assertEqual(predictions['cls_var'].shape, (1, 1))
        
        # Check value ranges
        self.assertTrue(torch.all(predictions['seg_pred'] >= 0))
        self.assertTrue(torch.all(predictions['seg_pred'] <= 1))
        self.assertTrue(torch.all(predictions['seg_var'] >= 0))
        self.assertTrue(torch.all(predictions['cls_pred'] >= 0))
        self.assertTrue(torch.all(predictions['cls_pred'] <= 1))
        
    def test_batch_prediction(self):
        """Test batch prediction."""
        # Create dummy batch
        x = torch.randn(
            self.batch_size, 3, self.img_size, self.img_size
        ).to(self.device)
        
        # Get predictions
        predictions = self.predictor.predict_batch(x)
        
        # Check shapes
        self.assertEqual(
            predictions['seg_pred'].shape,
            (self.batch_size, 1, self.img_size, self.img_size)
        )
        self.assertEqual(predictions['cls_pred'].shape, (self.batch_size, 1))
        
    def test_tta_consistency(self):
        """Test TTA prediction consistency."""
        # Create dummy input
        x = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        
        # Get predictions with and without TTA
        self.predictor.use_tta = True
        pred_tta = self.predictor.predict_single(x)
        
        self.predictor.use_tta = False
        pred_normal = self.predictor.predict_single(x)
        
        # Check that TTA predictions are similar but not identical
        diff = torch.abs(pred_tta['seg_pred'] - pred_normal['seg_pred']).mean()
        self.assertLess(diff.item(), 0.5)  # Not too different
        self.assertGreater(diff.item(), 0)  # Not identical
        
    def test_mc_dropout_uncertainty(self):
        """Test MC dropout uncertainty estimation."""
        # Create dummy input
        x = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        
        # Get predictions with MC dropout
        self.predictor.use_mc_dropout = True
        pred_mc = self.predictor.predict_single(x)
        
        # Check that uncertainty varies across the image
        var_std = torch.std(pred_mc['seg_var'])
        self.assertGreater(var_std.item(), 0)
        

if __name__ == '__main__':
    unittest.main()