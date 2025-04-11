import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MAEPretrainer(nn.Module):
    """
    Masked Autoencoder Pretrainer for the TGS Salt Identification Challenge.
    
    This implementation adapts the MAE approach using a Swin Transformer as the encoder.
    The model masks random patches of input images, encodes only visible patches,
    and reconstructs the full image including masked regions.
    
    Attributes:
        encoder: SwinEncoder backbone for encoding visible patches
        decoder: MAEDecoder for reconstructing the full image
        patch_size: Size of each image patch (height=width)
        in_chans: Number of input channels (1 for grayscale)
        img_size: Size of input images (101x101 for TGS Salt)
        mask_ratio: Ratio of patches to mask during training
    """
    
    def __init__(self, encoder, decoder, patch_size=4, in_chans=1, img_size=101, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        # Validate input dimensions
        if img_size % patch_size != 0:
            raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")
            
        self.num_patches = (img_size // patch_size) ** 2
    
    def patchify(self, imgs):
        """
        Convert a batch of images into a batch of patches.
        
        Args:
            imgs: Tensor of shape [B, C, H, W]
            
        Returns:
            patches: Tensor of shape [B, N, P*P*C] where N is number of patches
                    and P is patch_size
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0, \
            f"Image dimensions must be square and divisible by patch size {p}"
        
        h = w = imgs.shape[2] // p
        patches = imgs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(imgs.shape[0], h * w, -1)
        
        return patches
    
    def unpatchify(self, x):
        """
        Convert a batch of patches back into a batch of images.
        
        Args:
            x: Tensor of shape [B, N, P*P*C]
            
        Returns:
            imgs: Tensor of shape [B, C, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        # Reshape with memory optimization - avoid unnecessary intermediate tensors
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        
        # Use reshape directly instead of contiguous + reshape to reduce memory usage
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by per-sample shuffling.
        
        Args:
            x: Tensor of shape [B, N, D] where N is number of patches and D is patch dimension
            mask_ratio: Ratio of patches to mask (between 0 and 1)
            
        Returns:
            x_masked: Masked patches containing only the visible patches
            mask: Binary mask indicating which patches are masked (1=masked, 0=visible)
            ids_restore: Indices for restoring the original patch order
        """
        # Validate mask_ratio
        if not 0 <= mask_ratio < 1:
            raise ValueError(f"Mask ratio must be between 0 and 1, got {mask_ratio}")
        
        N = x.shape[1]
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(x.shape[0], N, device=x.device)  # Noise for random shuffling
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascending order
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        mask = torch.ones(x.shape[0], N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio=None):
        x = self.patchify(x)
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio or self.mask_ratio)
        x_encoded = self.encoder(x_masked)
        return x_encoded, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x_decoded = self.decoder(x, ids_restore)
        x_reconstructed = self.unpatchify(x_decoded)
        return x_reconstructed
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute the reconstruction loss (Mean Squared Error) on masked patches.
        
        Args:
            imgs: Original input images [B, C, H, W]
            pred: Reconstructed images [B, C, H, W]
            mask: Binary mask indicating masked patches [B, N], where N is number of patches
            
        Returns:
            loss: MSE loss on masked patches
        """
        # Validate mask shape
        expected_mask_shape = (imgs.shape[0], self.num_patches)
        if mask.shape != expected_mask_shape:
            raise ValueError(f"Mask shape {mask.shape} does not match expected shape {expected_mask_shape}")
        
        target = self.patchify(imgs)
        
        # Validate target and prediction dimensions match
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}")
        
        # Compute loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Handle edge case: if mask is all False (no masked patches)
        if not mask.any():
            return torch.tensor(0.0, device=loss.device)
            
        loss = (loss * mask).sum() / mask.sum()  # Mean on masked patches
        
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        x_encoded, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x_encoded, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        # Return additional info for visualization
        return loss, pred, mask
        
    def visualize_reconstruction(self, imgs, pred, mask, save_path=None):
        """
        Visualize the original images, reconstructed images, and masks.
        
        Args:
            imgs: Original input images [B, C, H, W]
            pred: Reconstructed images [B, C, H, W]
            mask: Binary mask indicating masked patches [B, N]
            save_path: Optional path to save the visualization
            
        Returns:
            fig: Matplotlib figure object
        """
        # Convert tensors to numpy arrays for visualization
        imgs = imgs.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        
        # Create a figure with 3 rows (original, recon, masked) and B columns
        batch_size = imgs.shape[0]
        fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))
        
        # If batch_size is 1, axes needs to be reshaped
        if batch_size == 1:
            axes = axes.reshape(3, 1)
        
        # Create mask visualization
        mask_vis = mask.reshape(batch_size, int(np.sqrt(mask.shape[1])), int(np.sqrt(mask.shape[1])))
        mask_vis = np.repeat(np.repeat(mask_vis, self.patch_size, axis=1), self.patch_size, axis=2)
        
        for i in range(batch_size):
            # Original image
            axes[0, i].imshow(imgs[i, 0], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstructed image
            axes[1, i].imshow(pred[i, 0], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
            
            # Mask visualization
            axes[2, i].imshow(mask_vis[i], cmap='binary')
            axes[2, i].set_title('Mask (white=masked)')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig