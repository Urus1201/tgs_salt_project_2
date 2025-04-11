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
        self.decoder_embed_dim = decoder_embed_dim  # Store as instance variable
        self.encoder_embed_dim = encoder_embed_dim  # Store encoder dim for prediction
        
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
        
        # Predict patches in encoder embedding space for reconstruction
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, encoder_embed_dim, bias=True
        )

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features from visible patches [B, N_visible, C]
            ids_restore: Indices to restore the original order of patches [B, N_total]
        Returns:
            Reconstructed image patches [B, N_total, encoder_embed_dim]
        """
        B, N_vis, C = x.shape
        N_total = ids_restore.shape[1]  # Total number of patches
        
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens
        mask_tokens = self.mask_token.repeat(B, N_total - N_vis, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed_dim))  # unshuffle
        
        # decoder blocks
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        
        # decoder to patch
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        
        return x_

class TransformerBlock(nn.Module):
    """Basic Transformer block with self-attention and MLP."""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer = nn.LayerNorm,
        use_checkpoint: bool = True,  # Enable by default for memory efficiency
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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

    def _forward(self, x: torch.Tensor) -> torch.Tensor: #or) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)