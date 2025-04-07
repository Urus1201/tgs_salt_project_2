import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Auxiliary classification head for predicting salt presence."""
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Feature map from encoder (B, C, H, W)
        Returns:
            Logits for binary classification (B, 1)
        """
        x = self.avg_pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)      # (B, C)
        x = self.fc(x)        # (B, 1)
        return x