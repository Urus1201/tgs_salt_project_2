# TGS Salt Segmentation: Advanced 2.5D Swin U-Net Framework

A PyTorch-based solution for the TGS Salt Identification Challenge using a hybrid Swin Transformer U-Net architecture with additional advanced techniques.

## Overview

This repository implements an advanced salt body segmentation pipeline with:

- **Hybrid Swin Transformer + U-Net** architecture for multi-scale feature extraction
- **2.5D input handling** to leverage volumetric context
- **Self-supervised pretraining** via Masked Autoencoder (MAE)
- **Multi-task learning** combining segmentation and classification
- **Boundary-aware compound loss** for precise edge detection
- **Uncertainty estimation** using Monte Carlo dropout
- **Synthetic data integration** support (with placeholders for diffusion models)

## Project Structure

```
salt_segmentation_project/
├── configs/              # Configuration files
│   └── default.yaml
├── data/                 # Data handling
│   ├── dataset.py        # Dataset with 2.5D support
│   └── transforms.py     # Custom augmentations
├── models/               # Model components
│   ├── encoder_swin.py   # Swin Transformer encoder
│   ├── decoder_unet.py   # U-Net decoder
│   ├── segmentation_model.py
│   └── classifier_head.py
├── losses/               # Loss functions
│   ├── dice_loss.py
│   ├── focal_loss.py
│   ├── boundary_loss.py
│   └── combined_loss.py
├── train/                # Training pipeline
│   ├── trainer.py
│   └── pretrain_mae.py
├── inference/            # Inference utilities
│   ├── predictor.py      # With MC dropout support
│   ├── uncertainty.py
│   └── submission.py     # Kaggle submission generator
├── utils/                # Utilities
│   ├── metrics.py
│   ├── rle.py
│   └── config_parser.py
└── run.py                # Main script
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tgs-salt-segmentation.git
cd tgs-salt-segmentation

# Create and activate conda environment
conda create -n salt-seg python=3.8
conda activate salt-seg

# Install requirements
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training
python run.py --config configs/default.yaml --mode train

# Training with pretrained encoder
python run.py --config configs/default.yaml --mode train --pretrained path/to/encoder.pth
```

### Inference & Submission

```bash
# Generate predictions with uncertainty
python run.py --mode predict --checkpoint path/to/model.pth --data path/to/test --output predictions.csv
```

## Key Components

### Model Architecture

The model combines a Swin Transformer encoder with a CNN-based U-Net decoder, plus a classification head for salt presence detection:

```python
class SaltSegmentationModel(nn.Module):
    def __init__(self, img_size=101, in_channels=3, **kwargs):
        super().__init__()
        # Initialize Swin Transformer encoder (pretrained weights can be loaded separately)
        self.encoder = SwinEncoder(img_size=img_size, in_channels=in_channels, **kwargs)
        # Initialize U-Net style decoder; it will receive encoder feature maps
        self.decoder = UNetDecoder(enc_feature_channels=self.encoder.out_channels)
        # Auxiliary classification head to predict salt presence
        self.classifier = ClassifierHead(in_features=self.encoder.out_channels[-1], num_classes=1)
    
    def forward(self, x):
        """
        x: input tensor of shape (B, 3, 101, 101) representing 3 stacked slices.
        Returns segmentation mask logits and classification logits.
        """
        # Encoder forward: get multi-scale feature maps
        enc_feats = self.encoder(x)  
        # enc_feats might be a list of feature maps from various stages
        seg_out = self.decoder(enc_feats)        # segmentation output (before sigmoid)
        class_out = self.classifier(enc_feats[-1])  # classification from last encoder feature
        return seg_out, class_out
```

### 2.5D Input Handling

The dataset provides 3-channel inputs by stacking consecutive seismic slices, capturing context from adjacent slices without the complexity of full 3D convolutions.

### Self-Supervised Pretraining (MAE)

Masked Autoencoder pretraining helps the encoder learn meaningful representations from unlabeled seismic data:

```python
# Pseudocode for MAE pretraining loop
encoder = SwinEncoder(img_size=img_size, in_channels=1)
mae_decoder = MAEDecoder(latent_dim=encoder.embed_dim)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mae_decoder.parameters()), lr=1e-4)

for epoch in range(num_epochs):
    for images in unlabeled_loader:
        masked_imgs, mask = mask_random_patches(images, mask_ratio=0.75)
        latent = encoder(masked_imgs)
        recon = mae_decoder(latent, mask)
        loss = reconstruction_loss(recon, images) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Multi-Task Learning

The model jointly learns segmentation and classification, improving performance for empty masks (approximately 38% of the dataset):

```python
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, feat_map):
        x = self.avgpool(feat_map)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
```

### Compound Loss Function

Combines multiple loss components for optimal performance:

```python
def combined_loss(pred_logits, target_mask, pred_class, target_class):
    # Segmentation losses
    l_dice = dice_loss(torch.sigmoid(pred_logits), target_mask)
    l_focal = focal_loss(pred_logits, target_mask)
    l_boundary = boundary_loss(pred_logits, target_mask)
    seg_loss = l_dice + l_focal + l_boundary
    # Classification loss
    cls_loss = F.binary_cross_entropy_with_logits(pred_class, target_class)
    return seg_loss + 0.5 * cls_loss
```

### Uncertainty Estimation

Monte Carlo Dropout provides uncertainty maps for more reliable predictions:

```python
def predict_with_uncertainty(model, image, num_samples=10):
    model.train()  # ensure dropout is on
    preds = []
    for _ in range(num_samples):
        with torch.no_grad():
            seg_logit, class_logit = model(image)
            preds.append(torch.sigmoid(seg_logit))
    preds = torch.stack(preds, dim=0)
    mean_pred = preds.mean(dim=0)
    var_pred = preds.var(dim=0)  # uncertainty estimate
    return mean_pred, var_pred
```

### Synthetic Data Integration

The codebase includes scaffolding for adding synthetic seismic data:

```python
class SaltDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, use_synthetic=False, synth_generator=None, transform=None):
        self.df = df
        self.use_synthetic = use_synthetic
        self.synth_generator = synth_generator
        # ...existing code...
        
    def __getitem__(self, idx):
        if self.use_synthetic and idx % 5 == 0:  # e.g., every 5th sample is synthetic
            img, mask = self.synth_generator.generate_sample()
        else:
            img = load_image(self.df.iloc[idx]["id"])
            mask = load_mask(self.df.iloc[idx]["id"])
        # ...existing code...
```

## Deployment Features

- **ONNX/TorchScript export** for production deployment
- **Batched inference** for speed optimization
- **Knowledge distillation** hooks for model compression
- **Quantization support** for faster inference

## References

- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," 2021
- He et al., "Masked Autoencoders Are Scalable Vision Learners," 2021
- Kervadec et al., "Boundary loss for highly unbalanced segmentation," Medical Image Analysis 2021
- Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," ICML 2016

## License

MIT