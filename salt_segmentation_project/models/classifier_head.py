import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    Enhanced auxiliary classification head for predicting salt presence.
    
    Features:
    - LayerNorm instead of BatchNorm (better with transformers)
    - Optional GeLU activation for improved performance
    - Residual connections
    - Proper weight initialization
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        num_classes: int = 1,
        use_gelu: bool = True
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Combined pooling features
        self.use_combined_pooling = True
        pooled_features = in_features * 2 if self.use_combined_pooling else in_features
        
        # Activation function
        act_fn = nn.GELU() if use_gelu else nn.ReLU(inplace=True)
        
        # First fully connected block with residual connection
        self.fc1 = nn.Linear(pooled_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = act_fn
        self.drop1 = nn.Dropout(dropout_rate)
        
        # Second fully connected block
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.act2 = act_fn
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Final classification layer
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Skip connection - transform input dimension if needed
        self.skip_proj = nn.Linear(pooled_features, hidden_dim) if pooled_features != hidden_dim else None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Transformer-style initialization
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Feature map from encoder (B, C, H, W)
        Returns:
            Logits for binary classification (B, 1)
        """
        if self.use_combined_pooling:
            # Combine avg and max pooling for better feature representation
            avg_feat = self.avg_pool(x).flatten(1)  # (B, C)
            max_feat = self.max_pool(x).flatten(1)  # (B, C)
            x = torch.cat([avg_feat, max_feat], dim=1)  # (B, 2C)
        else:
            x = self.avg_pool(x).flatten(1)  # (B, C)
        
        # Store input for residual connection
        identity = x
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        
        # First block with residual connection
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = x + identity  # Residual connection
        
        # Second block
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        # Final classification
        x = self.fc3(x)
        
        return x