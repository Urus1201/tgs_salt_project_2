import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple

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
        self.encoder = encoder.to(device)
        self.decoder = MAEDecoder(
            patch_size=encoder.patch_size,
            in_channels=1,  # Single channel for MAE pretraining
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
            lr=config['mae']['lr'],
            weight_decay=config['mae']['weight_decay']
        )
        
        # For logging
        self.best_val_loss = float('inf')
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images to a sequence of patches."""
        p = self.encoder.patch_size
        c = 1  # number of channels (1 for MAE pretraining)
        
        x = imgs.reshape(shape=(imgs.shape[0], c, imgs.shape[2] // p, p, imgs.shape[3] // p, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], -1, p**2 * c))
        return x
    
    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Convert a sequence of patches back to an image."""
        p = self.encoder.patch_size
        c = 1
        
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w))
        return imgs

    def forward_loss(self, samples: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with masking and loss computation."""
        # Get patches
        x = self.patchify(samples)
        
        # Add positional encoding in encoder
        x = self.encoder.prepare_tokens(x)
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(
            x, self.config['mae']['mask_ratio']
        )
        
        # Encode visible patches
        latent = self.encoder.forward_features(x_masked)
        
        # Decode all patches
        pred = self.decoder(latent, mask)
        
        # Unpatchify
        pred = self.unpatchify(pred, samples.shape[2], samples.shape[3])
        
        # Compute loss only on masked patches
        loss = F.mse_loss(pred, samples, reduction='none')
        loss = (loss * mask.unsqueeze(1)).mean()
        
        return loss, {
            'loss': loss.item(),
            'masked_ratio': mask.sum() / mask.numel()
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        for batch in self.train_loader:
            imgs = batch['image'][:, 0:1].to(self.device)  # Take only first channel for MAE
            
            # Forward pass and loss
            loss, metrics = self.forward_loss(imgs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += metrics['loss']
        
        return {'train_loss': total_loss / len(self.train_loader)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the current model."""
        if self.val_loader is None:
            return {}
            
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        for batch in self.val_loader:
            imgs = batch['image'][:, 0:1].to(self.device)
            loss, metrics = self.forward_loss(imgs)
            total_loss += metrics['loss']
        
        return {'val_loss': total_loss / len(self.val_loader)}

    def train(
        self,
        num_epochs: int,
        save_dir: str,
        early_stop_patience: int = 10
    ) -> Dict:
        """Full training loop with validation and early stopping."""
        patience_counter = 0
        train_metrics = []
        
        for epoch in range(num_epochs):
            # Train
            metrics = self.train_epoch()
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                metrics.update(val_metrics)
                
                # Early stopping check
                val_loss = val_metrics['val_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_encoder(f"{save_dir}/best_encoder.pth")
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            print(f"Epoch {epoch + 1}/{num_epochs}:", end=" ")
            print(", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            train_metrics.append(metrics)
        
        return train_metrics

    def save_encoder(self, path: str):
        """Save only the encoder weights for later use in segmentation."""
        torch.save(self.encoder.state_dict(), path)


def pretrain_mae(config: Dict):
    """Main function to run MAE pretraining."""
    # Initialize encoder
    encoder = SwinEncoder(
        img_size=config['model']['img_size'],
        patch_size=4,  # Fixed for MAE
        in_channels=1,  # Single channel for pretraining
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size'],
        drop_path_rate=0.1
    )
    
    # Setup data
    train_dataset = SaltDataset(
        data_dir=config['data']['train_dir'],
        transform=None,  # Add transforms if needed
        use_depth=False,  # Not needed for MAE
        use_neighbors=False  # Single channel for MAE
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['mae']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize trainer
    trainer = MAEPretrainer(
        encoder=encoder,
        config=config,
        train_loader=train_loader,
        device=config['training']['device']
    )
    
    # Train
    metrics = trainer.train(
        num_epochs=config['mae']['num_epochs'],
        save_dir=config['mae']['save_dir'],
        early_stop_patience=config['mae']['early_stop_patience']
    )
    
    return metrics