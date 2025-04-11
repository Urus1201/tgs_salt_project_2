import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm  # Added tqdm import

from models.encoder_swin import SwinEncoder
from models.mae_decoder import MAEDecoder
from data.dataset import SaltDataset


class MAEPretrainer:
    def __init__(
        self,
        encoder: SwinEncoder,
        config: Dict,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        """
        - encoder: A SwinEncoder that already handles patch embedding internally.
        - decoder: A MAEDecoder that expects [B, L, hidden_dim] from the encoder.
        """
        self.encoder = encoder.to(device)
        self.decoder = MAEDecoder(
            # In your original code, these come from config['mae'].
            # The patch_size=... is not used directly here (we rely on the encoder).
            patch_size=config['mae']['patch_size'],  # Use config instead of encoder.patch_size
            in_channels=1,  # single channel for grayscale
            encoder_embed_dim=encoder.embed_dim,
            decoder_embed_dim=config['mae']['decoder_embed_dim'],
            decoder_depth=config['mae']['decoder_depth'],
            decoder_num_heads=config['mae']['decoder_num_heads'],
        ).to(device)
        
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config['mae']['learning_rate'],  # Changed from 'lr' to 'learning_rate'
            weight_decay=config['mae']['weight_decay']
        )
        
        # For logging/early stopping
        self.best_val_loss = float('inf')
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking on the sequence of patch embeddings x, shape [B, L, D].
        Returns:
          x_masked: masked embeddings, shape [B, L*(1 - mask_ratio), D]
          mask: shape [B, L], with 1 where patches were removed
          ids_restore: used for un-shuffling if needed
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Shuffle each sample's patches by random noise
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only the first "len_keep" patches in each sample
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_loss(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        1. Pass the full images through the SwinEncoder to get patch embeddings [B, L, D].
        2. Randomly mask them, get x_masked, plus 'mask' for which patches were removed.
        3. Decoder reconstructs all L patches in that embedding space -> pred [B, L, D].
        4. MSE loss only for the masked patches.
        """
        # Step 1: The encoder does patch embed + forward
        # Make sure your images are already padded to multiples of patch_size if needed
        x = self.encoder.forward_features(imgs)  # shape [B, L, embed_dim]
        
        # Step 2: Random masking in embedding space
        x_masked, mask, ids_restore = self.random_masking(
            x, self.config['mae']['mask_ratio']
        )
        
        # Step 3: Decode (reconstruct) all L patches
        pred = self.decoder(x_masked, ids_restore)  # shape [B, L, D]
        
        # Step 4: MSE loss over the masked patches only
        # Expand mask to [B, L, 1] to match [B, L, D]
        mask_3d = mask.unsqueeze(-1)  # shape [B, L, 1]
        loss = (pred - x) ** 2
        loss = (loss * mask_3d).mean()  # average only the masked patches
        
        return loss, {
            'loss': loss.item(),
            'masked_ratio': mask.sum() / mask.numel()
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Training for one epoch. We only need the images; masks are not used for MAE.
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        # Add progress bar for batches
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for imgs, _ in pbar:
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.to(self.device)
                
                loss, metrics = self.forward_loss(imgs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += metrics['loss']
                
                # Update progress bar description with current loss
                pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
        
        return {'train_loss': total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validation loop. Same logic as training but no backward/optim.
        """
        if self.val_loader is None:
            return {}
            
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        # Add progress bar for validation
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        for imgs, _ in pbar:
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.to(self.device)
                
                loss, metrics = self.forward_loss(imgs)
                total_loss += metrics['loss']
                
                # Fix: use metrics['loss'] instead of metrics['val_loss']
                pbar.set_postfix({"val_loss": f"{metrics['loss']:.4f}"})
        
        return {'val_loss': total_loss / len(self.val_loader)}

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stop_patience: int = 10
    ) -> Dict:
        """
        Full training loop with optional early stopping.
        """
        patience_counter = 0
        train_metrics = []
        
        # Add an outer progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="MAE Pretraining")
        for epoch in epoch_pbar:
            metrics = self.train_epoch()
            
            if self.val_loader is not None:
                val_metrics = self.validate()
                metrics.update(val_metrics)
                
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_encoder(os.path.join(save_dir, "best_encoder.pth"))
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    epoch_pbar.write(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
            
            # Still log to console but use tqdm's write to prevent progress bar interference
            epoch_pbar.write(f"Epoch {epoch+1}/{num_epochs}: " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            train_metrics.append(metrics)
        
        return train_metrics

    def save_encoder(self, path: str):
        """Saves only the encoder weights."""
        torch.save(self.encoder.state_dict(), path)


def pretrain_mae(config: Dict):
    """
    Main function to run MAE pretraining with the above classes.
    - We assume you have a SwinEncoder that does patch embedding internally.
    - We pass the full images in, and rely on that to produce [B, L, D].
    """
    # 1) Initialize encoder
    encoder = SwinEncoder(
        model_name=config['mae'].get('model_name', 'microsoft/swin-tiny-patch4-window7-224'),
        pretrained=config['mae'].get('pretrained', False),
        in_channels=config['mae']['in_channels']
    )
    
    # 2) Setup data with train/val split
    full_dataset = SaltDataset(
        csv_file=os.path.join(config['data']['base_dir'], config['data']['train_csv']),
        image_dir=os.path.join(config['data']['base_dir'], config['data']['train_images']),
        mask_dir=None,   # no masks needed for MAE
        depths_csv=None, # not needed for MAE
        transform=None,  # you can add padding transform here if you like
        use_2_5d=False,  # single channel
        mode='test'      # so it won't load actual segmentation masks
    )
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mae']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['mae']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 3) Initialize the pretrainer
    trainer = MAEPretrainer(
        encoder=encoder,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['mae']['device']  # Changed from 'training.device' to 'mae.device'
    )
    
    # 4) Train
    metrics = trainer.train(
        num_epochs=config['mae']['num_epochs'],
        save_dir=config['mae']['save_dir'],
        early_stop_patience=config['mae']['early_stop_patience']
    )
    
    return metrics
