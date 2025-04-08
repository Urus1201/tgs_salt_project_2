import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class RefinementNet(nn.Module):
    """Small U-Net for uncertainty-guided mask refinement."""
    def __init__(
        self,
        in_channels: int = 3,  # image + initial mask + uncertainty map
        base_channels: int = 16,
        num_levels: int = 3,
        dropout_rate: float = 0.2
    ):
        """Initialize the refinement network.
        
        Args:
            in_channels: Number of input channels (image, mask, uncertainty)
            base_channels: Base number of channels for the network
            num_levels: Number of U-Net levels
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        # Encoder blocks
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        channels = []
        
        for level in range(num_levels):
            out_channels = base_channels * (2 ** level)
            encoder = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_rate)
            )
            self.encoders.append(encoder)
            channels.append(out_channels)
            current_channels = out_channels
            
        # Decoder blocks with skip connections
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        for level in range(num_levels - 1, 0, -1):
            # Upsampling
            self.upsamples.append(
                nn.ConvTranspose2d(
                    channels[level],
                    channels[level - 1],
                    kernel_size=2,
                    stride=2
                )
            )
            
            # Decoder block
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels[level - 1] * 2,  # *2 for skip connection
                        channels[level - 1],
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(channels[level - 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels[level - 1],
                        channels[level - 1],
                        kernel_size=3,
                        padding=1
                    ),
                    nn.BatchNorm2d(channels[level - 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=dropout_rate)
                )
            )
            
        # Final output layer
        self.final = nn.Conv2d(channels[0], 1, kernel_size=1)
        
        # Optional attention blocks for focusing on uncertain regions
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], 1, kernel_size=1),
                nn.Sigmoid()
            ) for i in range(num_levels)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor containing [image, initial_mask, uncertainty_map]
            
        Returns:
            Refined mask logits
        """
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # Apply attention, weighing features by uncertainty
            attn = self.attention[i](x)
            x = x * attn
            
            encoder_features.append(x)
            if i < len(self.encoders) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Decoder path with skip connections
        for i in range(len(self.decoders)):
            # Upsample
            x = self.upsamples[i](x)
            
            # Handle size mismatch if any
            if x.shape != encoder_features[-i-2].shape:
                x = F.interpolate(
                    x,
                    size=encoder_features[-i-2].shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            
            # Concatenate with encoder features
            x = torch.cat([x, encoder_features[-i-2]], dim=1)
            
            # Apply decoder block
            x = self.decoders[i](x)
            
        # Final 1x1 conv to get logits
        return self.final(x)


class UncertaintyRefinement:
    """Handle uncertainty-guided mask refinement."""
    def __init__(
        self,
        model: RefinementNet,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """Initialize refinement module.
        
        Args:
            model: Trained refinement model
            device: Device to run inference on
            threshold: Probability threshold for binary prediction
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        
    def refine_prediction(
        self,
        image: torch.Tensor,
        initial_pred: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Refine segmentation using uncertainty information.
        
        Args:
            image: Original input image
            initial_pred: Dictionary with initial predictions and uncertainties
            
        Returns:
            Refined binary mask
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure predictions match image size
            seg_pred = F.interpolate(initial_pred['seg_pred'], 
                                   size=image.shape[2:], 
                                   mode='bilinear', 
                                   align_corners=True)
            seg_var = F.interpolate(initial_pred['seg_var'], 
                                  size=image.shape[2:], 
                                  mode='bilinear', 
                                  align_corners=True)
            
            # Prepare input: concatenate image, initial mask, and uncertainty map
            x = torch.cat([image.to(self.device), 
                         seg_pred.to(self.device), 
                         seg_var.to(self.device)], dim=1)
            
            # Get refined prediction
            refined_logits = self.model(x)
            refined_pred = torch.sigmoid(refined_logits)
            
            # Get binary mask
            binary_mask = (refined_pred > self.threshold).float()
            
            # If classifier is very confident about no salt, return empty mask
            if 'cls_pred' in initial_pred and initial_pred['cls_pred'].item() < 0.1:
                binary_mask.zero_()
                
        return binary_mask
        
    @staticmethod
    def create_training_pair(
        image: torch.Tensor,
        pred_mask: torch.Tensor,
        uncertainty: torch.Tensor,
        true_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input-target pair for training refinement model.
        
        Args:
            image: Original image
            pred_mask: Initial predicted mask
            uncertainty: Uncertainty map
            true_mask: Ground truth mask
            
        Returns:
            tuple:
                - Input tensor [image, pred_mask, uncertainty]
                - Target mask
        """
        # Resize pred_mask and uncertainty to match the image size
        pred_mask = F.interpolate(pred_mask, size=image.shape[2:], mode='bilinear', align_corners=True)
        uncertainty = F.interpolate(uncertainty, size=image.shape[2:], mode='bilinear', align_corners=True)

        # Create input by concatenating channels
        x = torch.cat([image, pred_mask, uncertainty], dim=1)
        return x, true_mask


def train_refinement_model(
    model: RefinementNet,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    num_epochs: int = 50,
    device: str = 'cuda',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 10
) -> RefinementNet:
    """Train refinement model on validation predictions.
    
    Args:
        model: Untrained refinement model
        train_data: Tuple of (inputs, targets) for training
        val_data: Tuple of (inputs, targets) for validation
        num_epochs: Number of epochs to train
        device: Device to train on
        learning_rate: Initial learning rate
        weight_decay: L2 regularization factor
        patience: Early stopping patience
        
    Returns:
        Trained refinement model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience // 2,
        verbose=True
    )
    
    # Load data
    x_train, y_train = train_data[0].to(device), train_data[1].to(device)
    x_val, y_val = val_data[0].to(device), val_data[1].to(device)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        train_loss = F.binary_cross_entropy_with_logits(pred, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = F.binary_cross_entropy_with_logits(val_pred, y_val)
            
        # Update learning rate
        scheduler.step(val_loss)
            
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model