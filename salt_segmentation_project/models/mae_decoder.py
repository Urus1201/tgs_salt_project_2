import torch
import torch.nn as nn
from typing import Tuple

class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE pretraining that reconstructs from masked patches."""
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 1,
        encoder_embed_dim: int = 96,
        decoder_embed_dim: int = 64,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Embed tokens for masked patches (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Project encoder features to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Predict original patches
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * in_channels, bias=True
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features from visible patches [B, N, C]
            mask: Boolean mask indicating masked patches [B, N], True = masked
        Returns:
            Reconstructed image patches [B, N, patch_size*patch_size*in_channels]
        """
        B, N, C = x.shape
        
        # Project encoder features to decoder dim
        x = self.decoder_embed(x)
        
        # Append mask tokens for masked patches
        mask_tokens = self.mask_token.expand(B, N, -1)
        w = mask.unsqueeze(-1).type_as(mask_tokens)
        x = x * (1 - w) + mask_tokens * w
        
        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values for each patch
        x = self.decoder_pred(x)
        
        return x

class TransformerBlock(nn.Module):
    """Basic Transformer block with self-attention and MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, bias=qkv_bias, batch_first=True
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x