import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, Tuple, Dict


def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition the input tensor into local windows for self-attention.
    Args:
        x: (B, H, W, C)
        window_size: Window size for local self-attention
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reverse window partitioning.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Height and width of the image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention module."""
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
        self.window_size = (window_size if isinstance(window_size, int) else window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads
            )
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with shifted window attention."""
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

        # Layer norm before attention
        self.norm1 = nn.LayerNorm(dim)
        
        # Window attention with shifted windows
        self.attn = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer norm before MLP
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward_part1(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]):
        """Forward pass part 1: Window attention.
        
        Args:
            x: Input tensor of shape (B, H*W, C) or (B, H, W, C)
            mask_matrix: Optional attention mask
            
        Returns:
            Output tensor of shape (B, H*W, C)
        """
        B, L, C = x.shape
        H = W = int(np.sqrt(L + 0.5))  # Adding 0.5 for numerical stability
        
        shortcut = x
        x = self.norm1(x)  # LayerNorm over channel dimension
        
        # Reshape to spatial dimensions
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=mask_matrix)  # [nW*B, Mh*Mw, C]

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # Flatten back to sequence
        x = x.view(B, H * W, C)
        
        # Add skip connection
        x = shortcut + x
        
        return x
        
    def forward_part2(self, x: torch.Tensor):
        return x + self.mlp(self.norm2(x))
        
    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None):
        """Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input tensor
            mask_matrix: Optional attention mask for shifted windows
            
        Returns:
            Output tensor
        """
        if self.use_checkpoint and x.requires_grad:
            # Split computation for better memory efficiency
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
            x = checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = self.forward_part1(x, mask_matrix)
            x = self.forward_part2(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
        padding_mode: str = 'reflect'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.padding_mode = padding_mode
        
        # Calculate padding needed for 101x101 input
        self.input_size = 101
        self.pad_h = (patch_size - (self.input_size % patch_size)) % patch_size
        self.pad_w = (patch_size - (self.input_size % patch_size)) % patch_size
        
        # Calculate output dimensions after padding and patching
        self.padded_size = self.input_size + self.pad_h  # Same for width since input is square
        self.grid_size = self.padded_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Patch embedding projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Optional normalization
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        B, C, H, W = x.shape
        
        # Pad input to be divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = F.pad(
                x,
                (0, self.pad_w, 0, self.pad_h),
                mode=self.padding_mode
            )
        
        # Project to embedding dimension
        x = self.proj(x)  # Shape: [B, embed_dim, grid_size, grid_size]
        
        # Rearrange to [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
        
        # Return shape info for reconstruction
        padding_info = {
            'original_size': (H, W),
            'pad_h': self.pad_h,
            'pad_w': self.pad_w,
            'grid_size': self.grid_size
        }
        
        return x, padding_info


class SwinEncoder(nn.Module):
    def __init__(self, img_size=101, patch_size=4, in_channels=3, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super().__init__()
        
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        
        # Calculate padding needed for 101x101 input
        self.pad_h = (patch_size - (self.img_size[0] % patch_size)) % patch_size
        self.pad_w = (patch_size - (self.img_size[1] % patch_size)) % patch_size
        
        # Calculate padded dimensions
        self.padded_h = self.img_size[0] + self.pad_h
        self.padded_w = self.img_size[1] + self.pad_w
        
        # Calculate patch grid size
        self.grid_h = self.padded_h // patch_size
        self.grid_w = self.padded_w // patch_size
        self.num_patches = self.grid_h * self.grid_w
        
        print(f"DEBUG - Input: {self.img_size}, Padded: ({self.padded_h},{self.padded_w}), "
              f"Grid: ({self.grid_h},{self.grid_w}), Patches: {self.num_patches}")
        
        # Use simple Conv2d for patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Output channels for each stage
        self.out_channels = [embed_dim]
        for i in range(self.num_layers-1):
            self.out_channels.append(self.out_channels[-1] * 2)
        
        # Calculate stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        
        # Build Swin Transformer layers
        self.layers = nn.ModuleList()
        curr_dim = embed_dim
        
        for i_layer in range(self.num_layers):
            # Create blocks for this layer
            blocks = nn.ModuleList()
            for i_block in range(depths[i_layer]):
                blocks.append(
                    SwinTransformerBlock(
                        dim=curr_dim,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=0 if (i_block % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        dropout=dpr[sum(depths[:i_layer]) + i_block],
                        use_checkpoint=False
                    )
                )
            
            # Create layer with blocks
            layer = nn.ModuleDict({
                'blocks': blocks,
                'norm': nn.LayerNorm(curr_dim)
            })
            
            # Comment out downsample to keep resolution
            # if i_layer < self.num_layers - 1:
            #     layer['downsample'] = nn.Sequential(
            #         nn.LayerNorm(4 * curr_dim),
            #         nn.Linear(4 * curr_dim, 2 * curr_dim)
            #     )
            #     curr_dim *= 2
            
            self.layers.append(layer)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare tokens by adding positional embeddings.
        Args:
            x: Input tensor of shape (B, L, D) where L is sequence length and D is embedding dimension
        Returns:
            Tensor of same shape with positional embeddings added
        """
        B, L, D = x.shape
        
        # Create positional embeddings with the exact expected sequence length (676 for 26x26 grid)
        # Registering this as a buffer rather than a Parameter to avoid accumulating gradients
        if not hasattr(self, 'pos_embed') or self.pos_embed.shape[1] != L:
            pos_embed = torch.zeros(1, L, D, device=x.device)
            nn.init.trunc_normal_(pos_embed, std=.02)
            self.register_buffer('pos_embed', pos_embed, persistent=False)
        
        # Add positional embeddings to patch embeddings
        x = x + self.pos_embed
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through transformer blocks without final reshaping.
        Args:
            x: Input tensor of shape (B, L, D) where L is sequence length
        Returns:
            Processed features of shape (B, L, embed_dim)
        """
        B, L, D = x.shape
        
        # Adapt to the input sequence length by recalculating grid dimensions
        # This allows flexibility when input has a different patch count
        grid_size = int(np.sqrt(L + 0.5))  # Adding 0.5 for numerical stability
        
        # Debugging grid dimensions and sequence length
        print(f"DEBUG - encoder forward_features - Input: B={B}, L={L}, D={D}, calculated grid_size={grid_size}")
        
        # Update grid dimensions for this forward pass and store as object attributes
        # so they can be accessed by the decoder if needed
        self.grid_h = self.grid_w = grid_size
        self.current_num_patches = L
        
        # Project to embedding dimension if needed
        if D != self.embed_dim:
            proj = nn.Linear(D, self.embed_dim).to(x.device)
            x = proj(x)
        
        # Add positional embeddings
        x = self.prepare_tokens(x)
        
        # Process through ALL Swin blocks for all layers (not just first layer)
        for i_layer, layer_dict in enumerate(self.layers):
            # Apply all blocks in this layer
            for block in layer_dict['blocks']:
                x = block(x)
            
            # Apply normalization
            x = layer_dict['norm'](x)
        
        # Only return the tensor to maintain compatibility with existing code
        return x

    def forward(self, x):
        """Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            List of features at different scales for U-Net style skip connections
        """
        B, C, H, W = x.shape
        
        # Debug input shape
        # print(f"DEBUG - Input shape: {x.shape}")
        
        # Pad input to be divisible by patch_size
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='reflect')
            padding_info = (0, self.pad_w, 0, self.pad_h)  # left, right, top, bottom
        else:
            padding_info = None
        
        # Debug padded shape
        # print(f"DEBUG - After padding: {x.shape}")
        
        # Apply patch embedding using Conv2d
        x = self.patch_embed(x)  # B, embed_dim, grid_h, grid_w
        
        # Debug after patch embedding
        # print(f"DEBUG - After patch embed: {x.shape}")
        
        # Initialize feature storage
        features = []
        
        # Track spatial dimensions
        curr_h, curr_w = self.grid_h, self.grid_w
        
        # Convert from spatial to sequential format for transformer blocks
        x = x.flatten(2).transpose(1, 2)  # B, grid_h*grid_w, embed_dim
        x = self.pos_drop(x)
        
        # Debug sequential format
        # print(f"DEBUG - Sequential format: {x.shape}, expected spatial: ({curr_h},{curr_w})")
        
        # Process through Swin blocks
        for i_layer, layer_dict in enumerate(self.layers):
            # Apply Swin blocks
            for i_block, block in enumerate(layer_dict['blocks']):
                x = block(x)
            
            # Apply normalization
            x = layer_dict['norm'](x)
            
            # Get current feature dimensions
            curr_dim = self.out_channels[i_layer]
            
            # Reshape to create feature maps
            # Verify dimensions match before reshaping
            expected_size = B * curr_h * curr_w * curr_dim
            actual_size = x.numel()
            
            # print(f"DEBUG - Layer {i_layer}: Expected shape [{B}, {curr_h}, {curr_w}, {curr_dim}] "
            #       f"(size={expected_size}), Actual tensor size={actual_size}")
            
            if expected_size != actual_size:
                # This is where the issue occurs - fix dimensions
                # Recalculate spatial dimensions from sequence length
                seq_len = x.shape[1]
                spatial_size = int(np.sqrt(seq_len + 0.5))  # Adding 0.5 for numerical stability
                curr_h = curr_w = spatial_size
                # print(f"DEBUG - Fixing dimensions: seq_len={seq_len}, new spatial=({curr_h},{curr_w})")
            
            # Reshape to spatial format for feature storage
            curr_feat = x.view(B, curr_h, curr_w, curr_dim).permute(0, 3, 1, 2).contiguous()
            features.append(curr_feat)
            
            # Apply downsampling if not the last layer
            if i_layer < self.num_layers - 1 and 'downsample' in layer_dict:
                # For PatchMerging downsampling
                # We need to convert the sequence format to a format suitable for merging patches
                # 1. Reform to spatial
                x_spatial = x.view(B, curr_h, curr_w, curr_dim)
                
                # Pad if curr_h or curr_w is odd
                if (curr_h % 2) != 0 or (curr_w % 2) != 0:
                    pad_bottom = curr_h % 2
                    pad_right = curr_w % 2
                    x_spatial = x_spatial.permute(0, 3, 1, 2)
                    x_spatial = F.pad(x_spatial, (0, pad_right, 0, pad_bottom), mode='reflect')
                    x_spatial = x_spatial.permute(0, 2, 3, 1)
                    curr_h += pad_bottom
                    curr_w += pad_right
                
                # Prepare for 2×2 merging
                x_reshaped = x_spatial.view(B, curr_h//2, 2, curr_w//2, 2, curr_dim)
                
                # 2. Reshape to prepare for merging (merging 2x2 neighborhoods)
                x_merged = x_reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
                x_merged = x_merged.view(B, (curr_h//2)*(curr_w//2), 4*curr_dim)
                
                # 3. Apply linear projection to get new dimensions
                x = layer_dict['downsample'](x_merged)
                
                # 4. Update spatial dimensions (halved)
                curr_h, curr_w = curr_h // 2, curr_w // 2
                
                # print(f"DEBUG - After downsample: seq shape={x.shape}, spatial=({curr_h},{curr_w})")
        
        return features, padding_info