import torch
import torch.nn as nn
from typing import Tuple, Optional

from .encoder_swin import SwinEncoder
from .decoder_unet import UNetDecoder
from .classifier_head import ClassifierHead


# class SaltSegmentationModel(nn.Module):
#     """Combined model with Swin encoder, U-Net decoder, and classification head."""
#     def __init__(
#         self,
#         img_size: int = 101,
#         in_channels: int = 3,  # 3 for 2.5D input
#         embed_dim: int = 96,
#         depths: list = [2, 2, 6, 2],
#         num_heads: list = [3, 6, 12, 24],
#         window_size: int = 7,
#         dropout_rate: float = 0.1,
#         decoder_dropout: bool = True,
#         use_checkpoint: bool = False,
#         pretrained_encoder: Optional[str] = None
#     ):
#         super().__init__()
        
#         # Create Swin Transformer encoder
#         self.encoder = SwinEncoder(
#             img_size=img_size,
#             patch_size=4,  # Fixed patch size for 101x101 input
#             in_channels=in_channels,
#             embed_dim=embed_dim,
#             depths=depths,
#             num_heads=num_heads,
#             window_size=window_size,
#             dropout=dropout_rate,
#             use_checkpoint=use_checkpoint
#         )
        
#         # Create U-Net decoder
#         self.decoder = UNetDecoder(
#             enc_channels=self.encoder.out_channels,
#             use_dropout=decoder_dropout
#         )
        
#         # Create classification head
#         self.classifier = ClassifierHead(
#             in_features=self.encoder.out_channels[-1],
#             hidden_dim=256,
#             dropout_rate=dropout_rate
#         )
        
#         # Load pretrained encoder if specified
#         if pretrained_encoder is not None:
#             encoder_state = torch.load(pretrained_encoder, map_location='cpu')
#             self.encoder.load_state_dict(encoder_state)
            
#         self._mc_dropout_enabled = False
        
#     def enable_mc_dropout(self):
#         """Enable MC Dropout for uncertainty estimation."""
#         self._mc_dropout_enabled = True
        
#     def disable_mc_dropout(self):
#         """Disable MC Dropout (normal inference mode)."""
#         self._mc_dropout_enabled = False
        
#     def train(self, mode: bool = True):
#         """Override train mode to handle MC Dropout."""
#         super().train(mode)
#         if self._mc_dropout_enabled:
#             # Keep dropout enabled even in eval mode
#             for m in self.modules():
#                 if isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout):
#                     m.train()
#         return self
    
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x: input tensor of shape (B, 3, 101, 101) representing 3 stacked slices.
#         Returns segmentation mask logits and classification logits.
#         """
#         # Encoder forward: get multi-scale feature maps
#         features, padding_info = self.encoder(x)  
        
#         # Decoder uses features from encoder
#         seg_out = self.decoder(features, padding_info)
        
#         # Classification head uses the last encoder feature
#         class_out = self.classifier(features[-1])
        
#         return seg_out, class_out

import torch
import torch.nn as nn

from models.encoder_swin import SwinEncoder
from models.decoder_unet import UNetDecoder
from models.classifier_head import ClassifierHead

class SaltSegmentationModel(nn.Module):
    """
    Swin Transformer + UNet Decoder + Classification Head (multi-task learning).
    
    Input:  (B, 3, H, W)  →  3 stacked grayscale slices (2.5D)
    Output: segmentation map + salt presence logit
    """
    def __init__(
        self,
        model_name='microsoft/swin-tiny-patch4-window7-224',
        in_channels=3,
        seg_out_channels=1,
        cls_out_channels=1,
        pretrained=True
    ):
        super().__init__()

        # 1. Swin Transformer Encoder from timm
        self.encoder = SwinEncoder(
            model_name=model_name,
            in_channels=in_channels,
            pretrained=pretrained
        )

        # 2. UNet-style Decoder
        self.decoder = UNetDecoder(enc_feature_channels=self.encoder.out_channels)

        # 3. Segmentation Head (1x1 conv)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # 4. Classification Head (salt presence)
        self.classifier_head = ClassifierHead(
            in_features=self.encoder.out_channels[-1],  # deepest feature map (e.g., 768)
            num_classes=cls_out_channels
        )
        
        # Track if we should apply sigmoid during inference
        self.apply_sigmoid = False

    def enable_sigmoid(self):
        """Enable sigmoid activation for inference."""
        self.apply_sigmoid = True
        
    def disable_sigmoid(self):
        """Disable sigmoid activation (raw logits for training with BCE loss)."""
        self.apply_sigmoid = False

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
            seg_logits: (B, 1, H_out, W_out) - raw segmentation logits
            cls_logits: (B, 1)               - raw classification logit (salt/no salt)
        """
        # Save original input size for upsampling
        original_size = (x.shape[2], x.shape[3])
        
        encoder_feats = self.encoder(x)              # [c1, c2, c3, c4]
        decoder_output = self.decoder(encoder_feats) # final decoder feature map
        seg_logits = self.segmentation_head(decoder_output)
        
        # Ensure output matches input resolution with bilinear upsampling
        if seg_logits.shape[2:] != original_size:
            seg_logits = nn.functional.interpolate(
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

        return seg_logits, cls_logits
