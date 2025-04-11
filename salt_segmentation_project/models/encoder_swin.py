import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Dict
import timm
import huggingface_hub
import requests
import warnings

# Disable SSL verification completely
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
warnings.filterwarnings('ignore', message='.*SSL.*')

# This is the most direct way to disable SSL verification in the requests library
# which is used by huggingface_hub
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey patch the request session to disable verification
old_init = requests.Session.__init__
def new_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = new_init

# Also monkey patch huggingface_hub's get_session
original_get_session = huggingface_hub.utils._http.get_session
def patched_get_session(*args, **kwargs):
    session = original_get_session(*args, **kwargs)
    session.verify = False
    return session
huggingface_hub.utils._http.get_session = patched_get_session

# --------------------------------------------------------------------------
#  Window partition helpers 
# --------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, 
        H // window_size, window_size, 
        W // window_size, window_size, 
        C
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reverse window partition:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, 
        H // window_size, 
        W // window_size,
        window_size, 
        window_size, 
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# --------------------------------------------------------------------------
#  Window-based multi-head self attention
# --------------------------------------------------------------------------
class WindowAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        window_size: int, 
        num_heads: int, 
        qkv_bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)  # for 2D
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads
            )
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B*nW, win_size*win_size, C)
        mask: (nW, win_size*win_size, win_size*win_size) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # add relative position bias
        win_size_sq = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(win_size_sq, win_size_sq, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask shape => [nW, N, N]
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --------------------------------------------------------------------------
#  Basic Swin block
# --------------------------------------------------------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        window_size: int = 7, 
        shift_size: int = 0, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.0, 
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward_part1(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]):
        B, N, C = x.shape
        # We have a “square” resolution in the sense H*W = N
        H = W = int(np.sqrt(N + 0.5))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad to multiples of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

        _, Hp, Wp, _ = x.shape

        # shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(x, self.window_size)  # (nW*B, wsize, wsize, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention
        attn_windows = self.attn(x_windows, mask=mask_matrix)  # (nW*B, wsize^2, C)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # remove any padding
        x = x[:, :H, :W, :].contiguous()

        # flatten back
        x = x.view(B, H * W, C)
        x = shortcut + x
        return x

    def forward_part2(self, x: torch.Tensor):
        return x + self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None):
        """
        x: (B, N, C)
        """
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
            x = checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = self.forward_part1(x, mask_matrix)
            x = self.forward_part2(x)
        return x

# --------------------------------------------------------------------------
#  SwinEncoder (Single forward_features method returns [B, L, embed_dim])
# --------------------------------------------------------------------------
class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder using timm with pretrained weights.

    This encoder returns 4 feature maps at increasing depth and decreasing spatial resolution,
    which makes it compatible with U-Net style decoders.

    Example output channel sizes (for swin_tiny_patch4_window7_224):
        [96, 192, 384, 768]
    """

    def __init__(
        self,
        model_name: str = 'microsoft/swin-tiny-patch4-window7-224',
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__()
        
        # The model requires images to be 224x224
        self.target_size = (224, 224)
        
        # Create backbone with specified params
        self.backbone = timm.create_model(
            model_name.replace("-", "_"),  # Convert hyphens to underscores for timm compatibility
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True
        )

        # Store output channels - these need to match the decoder's expected inputs
        self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]
        self.embed_dim = self.out_channels[-1]  # e.g. 768 for swin-tiny
        self.patch_size = 4  # match "patch4" from model_name
        
        # Disable strict size checking in the model
        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'strict_img_size'):
            self.backbone.patch_embed.strict_img_size = False

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            List of feature maps [c1, c2, c3, c4] with correct channel dimensions
        """
        # Save original dimensions
        B, C, H, W = x.shape
        
        # Resize input to match the target size expected by the model
        if (H, W) != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
            
        # Pass through the backbone to get features
        features = self.backbone(x)
        
        # Target sizes for the feature maps to match the expected downsampling ratios
        # For a 101x101 input, the feature maps should be roughly:
        # c1: 25x25, c2: 13x13, c3: 7x7, c4: 4x4
        target_sizes = [
            (max(1, H // 4), max(1, W // 4)),    # c1 - 1/4 scale
            (max(1, H // 8), max(1, W // 8)),    # c2 - 1/8 scale
            (max(1, H // 16), max(1, W // 16)),  # c3 - 1/16 scale
            (max(1, H // 32), max(1, W // 32))   # c4 - 1/32 scale
        ]
        
        # Process features: handle channel dimension and resize
        processed_features = []
        for i, (feat, target_size) in enumerate(zip(features, target_sizes)):
            # First fix channel dimension if needed - timm returns [B, H, W, C] but we need [B, C, H, W]
            if len(feat.shape) == 4 and feat.shape[1] not in [96, 192, 384, 768]:
                # This assumes channels are in the last dimension
                feat = feat.permute(0, 3, 1, 2).contiguous()
            
            # Now resize spatial dimensions while keeping channel dims intact
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            processed_features.append(feat)
        
        return processed_features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns patch embeddings of shape [B, L, embed_dim].
        """
        feats = self.forward(x)  # returns [c1, c2, c3, c4]
        last_feat = feats[-1]    # [B, embed_dim, H', W']
        B, C, H, W = last_feat.shape
        return last_feat.flatten(2).transpose(1, 2)