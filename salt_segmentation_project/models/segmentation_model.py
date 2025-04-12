import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union

from .encoder_swin import SwinEncoder
from .decoder_unet import UNetDecoder
from .classifier_head import ClassifierHead

class SaltSegmentationModel(nn.Module):
    """
    Enhanced Swin Transformer + UNet Decoder + Classification Head (multi-task learning).
    
    Features:
    - Swin Transformer encoder
    - Enhanced UNet decoder with attention gates and residual connections
    - ASPP module for multi-scale feature aggregation
    - Deep supervision for better gradient flow
    - Multi-task learning with segmentation and classification heads
    
    Input:  (B, 3, H, W)  →  3 stacked grayscale slices (2.5D)
    Output: segmentation map + salt presence logit (+ auxiliary outputs in training)
    """
    def __init__(
        self,
        model_name='microsoft/swin-tiny-patch4-window7-224',
        in_channels=3,
        seg_out_channels=1,
        cls_out_channels=1,
        pretrained=True,
        use_deep_supervision=True,
        use_checkpoint=False,
        dropout_rate=0.1
    ):
        super().__init__()

        # Configuration
        self.use_deep_supervision = use_deep_supervision
        
        # 1. Swin Transformer Encoder from timm
        self.encoder = SwinEncoder(
            model_name=model_name,
            in_channels=in_channels,
            pretrained=pretrained
        )

        # 2. Enhanced UNet-style Decoder
        self.decoder = UNetDecoder(
            enc_feature_channels=self.encoder.out_channels,
            dropout_rate=dropout_rate,
            use_checkpoint=use_checkpoint
        )

        # 3. Segmentation Head (1x1 conv)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, seg_out_channels, kernel_size=1)
        )

        # 4. Classification Head (salt presence)
        self.classifier_head = ClassifierHead(
            in_features=self.encoder.out_channels[-1],  # deepest feature map (e.g., 768)
            num_classes=cls_out_channels,
            dropout_rate=dropout_rate
        )
        
        # Track if we should apply sigmoid during inference
        self.apply_sigmoid = False

    def enable_sigmoid(self):
        """Enable sigmoid activation for inference."""
        self.apply_sigmoid = True
        
    def disable_sigmoid(self):
        """Disable sigmoid activation (raw logits for training with BCE loss)."""
        self.apply_sigmoid = False

    def enable_deep_supervision(self):
        """Enable deep supervision for training."""
        self.use_deep_supervision = True
        self.decoder.enable_deep_supervision()
        
    def disable_deep_supervision(self):
        """Disable deep supervision for inference."""
        self.use_deep_supervision = False
        self.decoder.disable_deep_supervision()

    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout by forcing dropout layers into train mode."""
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train()

    def disable_mc_dropout(self):
        """Disable Monte Carlo dropout (normal eval mode for dropout layers)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.eval()

    def forward(self, x):
        """
        Args:
            x: Tensor (B, 3, H, W)

        Returns:
            Training mode with deep supervision:
                dict with keys:
                - 'seg_logits': Main segmentation output (B, 1, H_out, W_out)
                - 'cls_logits': Classification output (B, 1)
                - 'aux_outputs': List of auxiliary segmentation outputs at different scales
            
            Inference mode or without deep supervision:
                seg_logits: (B, 1, H_out, W_out) - segmentation output
                cls_logits: (B, 1) - classification output
        """
        # Save original input size for upsampling
        original_size = (x.shape[2], x.shape[3])
        
        # Encoder features
        encoder_feats = self.encoder(x)  # [c1, c2, c3, c4]
        
        # Decoder with potential deep supervision outputs
        if self.use_deep_supervision and self.training:
            # Returns main output and deep supervision outputs
            decoder_output, deep_outputs = self.decoder(encoder_feats)
            
            # Process main segmentation output
            seg_logits = self.segmentation_head(decoder_output)
            
            # Ensure all outputs match input resolution
            seg_logits = F.interpolate(
                seg_logits, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
            
            # Process and resize auxiliary outputs
            aux_outputs = []
            for aux_out in deep_outputs:
                aux_out = F.interpolate(
                    aux_out,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
                aux_outputs.append(aux_out)
                
        else:
            # Regular forward pass without deep supervision
            decoder_output = self.decoder(encoder_feats)
            seg_logits = self.segmentation_head(decoder_output)
            
            # Ensure output matches input resolution
            if seg_logits.shape[2:] != original_size:
                seg_logits = F.interpolate(
                    seg_logits, 
                    size=original_size, 
                    mode='bilinear', 
                    align_corners=False
                )
        
        # Apply sigmoid during inference if enabled
        if self.apply_sigmoid and not self.training:
            seg_logits = torch.sigmoid(seg_logits)

        # Classification from last encoder stage
        cls_logits = self.classifier_head(encoder_feats[-1])

        # Return different outputs depending on mode
        if self.use_deep_supervision and self.training:
            return {
                'seg_logits': seg_logits,
                'cls_logits': cls_logits,
                'aux_outputs': aux_outputs
            }
        else:
            return seg_logits, cls_logits
