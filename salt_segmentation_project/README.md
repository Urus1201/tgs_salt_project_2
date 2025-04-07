# TGS Salt Identification Challenge Solution

This repository contains a deep learning solution for the TGS Salt Identification Challenge, using a Swin Transformer-based architecture with uncertainty estimation.

## Features

- Swin Transformer encoder with U-Net style decoder architecture
- 2.5D input processing for improved depth context
- Multi-task learning with auxiliary classification head
- Uncertainty estimation using Monte Carlo dropout
- Test-time augmentation for robust predictions
- Uncertainty-guided mask refinement
- Comprehensive experiment tracking and visualization

## Project Structure

```
salt_segmentation_project/
├── configs/                # Configuration files
│   └── default.yaml       # Default training configuration
├── data/                  # Data handling
│   ├── dataset.py        # Dataset implementation
│   └── transforms.py     # Data augmentation
├── inference/             # Inference pipeline
│   ├── predictor.py      # Prediction with uncertainty
│   ├── refinement.py     # Mask refinement
│   └── submission.py     # Kaggle submission
├── losses/               # Loss functions
│   ├── boundary_loss.py  # Boundary loss
│   ├── combined_loss.py  # Combined loss module
│   ├── dice_loss.py     # Dice loss
│   └── focal_loss.py    # Focal loss
├── models/               # Model architecture
│   ├── classifier_head.py # Classification head
│   ├── decoder_unet.py   # U-Net decoder
│   ├── encoder_swin.py   # Swin Transformer encoder
│   └── segmentation_model.py # Full model
├── train/                # Training pipeline
│   └── trainer.py       # Training loop
├── utils/               # Utilities
│   ├── logging.py      # Experiment logging
│   └── visualization.py # Visualization tools
├── requirements.txt     # Dependencies
└── run.py              # Main script
```

## Installation

1. Create a new Python environment:
```bash
conda create -n salt_seg python=3.8
conda activate salt_seg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the competition data from [Kaggle](https://www.kaggle.com/c/tgs-salt-identification-challenge/data)
2. Extract the data and organize it as follows:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── test/
│   └── images/
├── depths.csv
├── train.csv
└── sample_submission.csv
```

## Training

1. Review and modify the configuration in `configs/default.yaml`
2. Start training:
```bash
python run.py --config configs/default.yaml --mode train
```

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir checkpoints/logs
```

## Inference

1. Generate predictions and create submission:
```bash
python run.py --config configs/default.yaml --mode predict --checkpoint checkpoints/best_model.pth
```

## Configuration

Key configuration options in `default.yaml`:

```yaml
data:
  use_2_5d: true    # Use 2.5D input
  val_split: 0.15   # Validation split ratio

model:
  img_size: 101
  embed_dim: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]

training:
  num_epochs: 100
  optimizer: 'adamw'
  lr: 0.0001
  scheduler: 'cosine'

inference:
  use_tta: true
  use_mc_dropout: true
  mc_samples: 10
```

## Model Architecture

The model consists of:
- Swin Transformer encoder for hierarchical feature extraction
- U-Net style decoder with skip connections
- Auxiliary classification head
- Multiple loss components:
  - Dice loss for segmentation
  - Focal loss for handling class imbalance
  - Boundary loss for edge precision
  - Binary cross-entropy for classification

## Uncertainty Estimation

The model provides uncertainty estimates through:
1. Monte Carlo dropout sampling
2. Test-time augmentation
3. Optional uncertainty-guided mask refinement

## Visualization

The project includes comprehensive visualization tools:
- Training metrics and learning curves
- Prediction visualization with uncertainty maps
- TensorBoard integration for experiment tracking

## Contributing

Feel free to submit issues and enhancement requests!