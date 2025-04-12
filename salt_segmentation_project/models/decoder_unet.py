import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, List

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) module for focusing on relevant features from skip connections.
    Implements the attention mechanism from Attention U-Net.
    """
    def __init__(self, g_channels, x_channels, int_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels),
            nn.ReLU(inplace=True)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels),
            nn.ReLU(inplace=True)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Resize gating signal to match skip connection spatial dimensions
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
        

class EnhancedDecoderBlock(nn.Module):
    """
    Enhanced U-Net decoder block with:
    - Residual connections
    - Attention gates for skip connections
    - Layer normalization (better with transformers)
    - Dropout for regularization
    - Gradient checkpointing for memory efficiency
    """
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Attention gate for skip connection
        self.attention_gate = AttentionGate(
            g_channels=in_channels,
            x_channels=skip_channels,
            int_channels=skip_channels // 2
        )
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # First conv block with residual connection
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        # Second conv block with residual connection
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # 1x1 conv for residual connection if dimensions don't match
        self.skip_conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)

    def _forward_conv1(self, x):
        identity = self.skip_conv(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        return out + identity
        
    def _forward_conv2(self, x):
        identity = x
        out = self.conv2(x)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        return out + identity

    def forward(self, x, skip):
        # Apply attention to skip connection
        attended_skip = self.attention_gate(x, skip)
        
        # Upsampling
        x = self.upsample(x)
        
        # Handle size mismatches
        if x.shape[2:] != attended_skip.shape[2:]:
            x = F.interpolate(x, size=attended_skip.shape[2:], mode='bilinear', align_corners=True)
            
        # Concatenate
        x = torch.cat([attended_skip, x], dim=1)
        
        # Apply convolutional blocks with residual connections using checkpoint if needed
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self._forward_conv1, x)
            x = checkpoint.checkpoint(self._forward_conv2, x)
        else:
            x = self._forward_conv1(x)
            x = self._forward_conv2(x)
            
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for capturing multi-scale contexts
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        dilations = [1, 6, 12, 18]
        
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        
        aspp_outputs = []
        for block in self.aspp_blocks:
            aspp_outputs.append(block(x))
            
        # Global pooling branch
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=(h, w), mode='bilinear', align_corners=True)
        
        aspp_outputs.append(global_features)
        output = torch.cat(aspp_outputs, dim=1)
        
        return self.output_conv(output)


class EnhancedUNetDecoder(nn.Module):
    """
    Enhanced U-Net decoder with:
    - Attention gates for skip connections
    - Residual connections
    - ASPP module for multi-scale context
    - Deep supervision
    - Gradient checkpointing for memory efficiency
    
    Assumes encoder outputs 4 feature maps with the following channel sizes (for Swin-Tiny):
        - c1: 96     (1/4 scale)
        - c2: 192    (1/8 scale)
        - c3: 384    (1/16 scale)
        - c4: 768    (1/32 scale)
    """

    def __init__(self, enc_feature_channels, dropout_rate=0.1, use_checkpoint=False):
        super().__init__()
        assert len(enc_feature_channels) == 4, "Expected 4 encoder feature maps"

        c1, c2, c3, c4 = enc_feature_channels
        
        # ASPP at the deepest level for better contextual information
        self.aspp = ASPP(in_channels=c4, out_channels=c4)
        
        # Decoder blocks with attention gates and residual connections
        self.dec4 = EnhancedDecoderBlock(
            in_channels=c4, 
            skip_channels=c3, 
            out_channels=c3,
            dropout_rate=dropout_rate, 
            use_checkpoint=use_checkpoint
        )
        self.dec3 = EnhancedDecoderBlock(
            in_channels=c3, 
            skip_channels=c2, 
            out_channels=c2,
            dropout_rate=dropout_rate, 
            use_checkpoint=use_checkpoint
        )
        self.dec2 = EnhancedDecoderBlock(
            in_channels=c2, 
            skip_channels=c1, 
            out_channels=c1,
            dropout_rate=dropout_rate, 
            use_checkpoint=use_checkpoint
        )
        
        # Final processing
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        
        # Deep supervision outputs
        self.deep_sup3 = nn.Conv2d(c3, 1, kernel_size=1)
        self.deep_sup2 = nn.Conv2d(c2, 1, kernel_size=1)
        
        self.out_channels = c1
        self.use_checkpoint = use_checkpoint
        self.training_mode = True

    def enable_deep_supervision(self):
        self.training_mode = True
        
    def disable_deep_supervision(self):
        self.training_mode = False

    def forward(self, feats):
        """
        feats: list of encoder feature maps [c1, c2, c3, c4]
        
        Returns:
            training mode: (main_output, [deep_sup_output2, deep_sup_output3]) 
            inference mode: main_output
        """
        c1, c2, c3, c4 = feats  # shallowest to deepest
        
        # Apply ASPP at the deepest level
        c4 = self.aspp(c4)
        
        # Decoder pathway with attention gates and skip connections
        x = self.dec4(c4, c3)
        deep_out3 = self.deep_sup3(x) if self.training_mode else None
        
        x = self.dec3(x, c2)
        deep_out2 = self.deep_sup2(x) if self.training_mode else None
        
        x = self.dec2(x, c1)
        
        # Final processing
        x = self.dec1(x)
        
        if self.training_mode:
            return x, [deep_out2, deep_out3]
        else:
            return x


# For backward compatibility, keep UNetDecoder as an alias to EnhancedUNetDecoder
UNetDecoder = EnhancedUNetDecoder
