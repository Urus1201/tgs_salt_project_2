import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block with skip connection."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Split operations for better memory efficiency with checkpointing
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(0.1) if use_dropout else None

    def _forward_conv1(self, x):
        return self.conv1_block(x)
        
    def _forward_conv2(self, x):
        x = self.conv2_block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_conv1, x)
            x = checkpoint.checkpoint(self._forward_conv2, x)
        else:
            x = self._forward_conv1(x)
            x = self._forward_conv2(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net decoder that takes multi-scale features from Swin encoder."""
    def __init__(
        self,
        enc_channels: List[int],  # List of encoder output channels at each scale
        dec_channels: Optional[List[int]] = None,  # Optional custom decoder channel sizes
        use_dropout: bool = True,
        use_checkpoint: bool = False,
        final_channel: int = 1  # Number of output channels (1 for binary mask)
    ):
        super().__init__()
        if dec_channels is None:
            # By default, decoder channels are half of encoder channels
            dec_channels = [c // 2 for c in enc_channels]
        
        # Verify matching number of scales
        assert len(enc_channels) == len(dec_channels)
        
        self.use_checkpoint = use_checkpoint
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Create decoder stages
        for i in range(len(enc_channels)-1, -1, -1):
            # Input channels = current encoder channels + previous decoder channels (if not first)
            in_channels = enc_channels[i]
            if i < len(enc_channels)-1:
                in_channels += dec_channels[i+1]
                
            # Add upsampling and decoder block
            if i < len(enc_channels)-1:  # No need to upsample at first (deepest) level
                self.upsamples.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            dec_channels[i+1],
                            dec_channels[i+1],
                            kernel_size=2,
                            stride=2
                        ),
                        nn.BatchNorm2d(dec_channels[i+1])
                    )
                )
            
            self.stages.append(
                UNetDecoderBlock(
                    in_channels=in_channels,
                    out_channels=dec_channels[i],
                    use_dropout=use_dropout,
                    use_checkpoint=use_checkpoint
                )
            )
        
        # Final convolution to get desired number of output channels
        self.final_conv = nn.Conv2d(dec_channels[0], final_channel, kernel_size=1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def _upsample_add(self, x1, x2):
        """Memory efficient upsampling and addition of skip connection."""
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(
                x1,
                size=x2.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        return torch.cat([x1, x2], dim=1)
        
    def forward(self, encoder_features: List[torch.Tensor], padding_info: Optional[Dict] = None):
        """Forward pass through U-Net decoder.
        
        Args:
            encoder_features: List of feature maps from encoder at different scales
            padding_info: Dictionary containing padding information:
                - original_size: (H, W) tuple of original input size
                - pad_h: Height padding
                - pad_w: Width padding
        
        Returns:
            Final segmentation logits at original image size
        """
        # Reverse encoder features to process from deepest layer first
        encoder_features = encoder_features[::-1]
        
        x = encoder_features[0]
        
        # Process through decoder stages
        for i in range(len(self.stages)):
            if i > 0:  # Skip connection + upsampling (except first block)
                # Memory efficient upsampling of previous decoder output
                x = self.upsamples[i-1](x)
                
                # Apply skip connection with memory efficient concatenation
                if self.use_checkpoint and x.requires_grad:
                    x = checkpoint.checkpoint(
                        self._upsample_add,
                        x,
                        encoder_features[i]
                    )
                else:
                    x = self._upsample_add(x, encoder_features[i])
            
            # Apply decoder block
            x = self.stages[i](x)
        
        # Final 1x1 conv
        x = self.final_conv(x)
        
        # Remove padding if needed
        if padding_info is not None:
            # padding_info format: (left, right, top, bottom)
            # If we need to remove padding, we extract the original image size portion
            # Calculate start and end positions for height and width
            b, c, h, w = x.shape
            top, bottom = padding_info[2], padding_info[3]
            left, right = padding_info[0], padding_info[1]
            
            # Only crop if there's padding to remove
            if top > 0 or bottom > 0 or left > 0 or right > 0:
                # Extract the non-padded region
                h_start, h_end = 0, h - bottom if bottom > 0 else h
                w_start, w_end = 0, w - right if right > 0 else w
                x = x[:, :, h_start:h_end, w_start:w_end]
        
        return x