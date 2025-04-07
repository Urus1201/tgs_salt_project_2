import torch
import torch.nn as nn
from typing import Tuple, Optional

from .encoder_swin import SwinEncoder
from .decoder_unet import UNetDecoder
from .classifier_head import ClassifierHead


class SaltSegmentationModel(nn.Module):
    """Combined model with Swin encoder, U-Net decoder, and classification head."""
    def __init__(
        self,
        img_size: int = 101,
        in_channels: int = 3,  # 3 for 2.5D input
        embed_dim: int = 96,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        dropout_rate: float = 0.1,
        decoder_dropout: bool = True,
        use_checkpoint: bool = False,
        pretrained_encoder: Optional[str] = None
    ):
        super().__init__()
        
        # Create Swin Transformer encoder
        self.encoder = SwinEncoder(
            img_size=img_size,
            patch_size=4,  # Fixed patch size for 101x101 input
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout_rate,
            use_checkpoint=use_checkpoint
        )
        
        # Create U-Net decoder
        self.decoder = UNetDecoder(
            enc_channels=self.encoder.out_channels,
            use_dropout=decoder_dropout
        )
        
        # Create classification head
        self.classifier = ClassifierHead(
            in_features=self.encoder.out_channels[-1],
            hidden_dim=256,
            dropout_rate=dropout_rate
        )
        
        # Load pretrained encoder if specified
        if pretrained_encoder is not None:
            encoder_state = torch.load(pretrained_encoder, map_location='cpu')
            self.encoder.load_state_dict(encoder_state)
            
        self._mc_dropout_enabled = False
        
    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation."""
        self._mc_dropout_enabled = True
        
    def disable_mc_dropout(self):
        """Disable MC Dropout (normal inference mode)."""
        self._mc_dropout_enabled = False
        
    def train(self, mode: bool = True):
        """Override train mode to handle MC Dropout."""
        super().train(mode)
        if self._mc_dropout_enabled:
            # Keep dropout enabled even in eval mode
            for m in self.modules():
                if isinstance(m, nn.Dropout2d) or isinstance(m, nn.Dropout):
                    m.train()
        return self
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: input tensor of shape (B, 3, 101, 101) representing 3 stacked slices.
        Returns segmentation mask logits and classification logits.
        """
        # Encoder forward: get multi-scale feature maps
        features, padding_info = self.encoder(x)  
        
        # Decoder uses features from encoder
        seg_out = self.decoder(features, padding_info)
        
        # Classification head uses the last encoder feature
        class_out = self.classifier(features[-1])
        
        return seg_out, class_out