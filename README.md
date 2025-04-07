Perfect, thank you! I’ll now create a modular PyTorch codebase for your integrated 2.5D salt segmentation pipeline, incorporating Swin Transformer + CNN hybrid architecture, masked autoencoder pretraining, multi-task segmentation/classification, boundary-aware loss, synthetic data integration via diffusion models (placeholders), Bayesian uncertainty modeling, and full training/evaluation/submission support. I’ll get back to you shortly with the structured workspace and initial implementation for each module.

# Modular PyTorch Codebase for TGS Salt Segmentation (Swin U-Net & Advanced Features)

To address the **TGS Salt Identification Challenge** with state-of-the-art methods, we design a modular PyTorch codebase. This project integrates a **hybrid Swin Transformer U-Net** model, **self-supervised pretraining**, **multi-task learning**, advanced **loss functions**, and **uncertainty estimation**. The code is organized into a clear directory structure for flexibility and maintainability. Below we outline the architecture and provide initial implementations for each module.

## Project Structure and Organization

Our codebase is structured as a Python package with separate modules for models, losses, data handling, training, inference, etc. This modular design makes it easy to extend or swap components. A typical layout might look like:

```plaintext
salt_segmentation_project/
├── configs/              # Configuration files (YAML/JSON) for training, model hyperparams, etc.
│   └── default.yaml
├── data/                 # Data loading and augmentation logic
│   ├── dataset.py        # Dataset class for TGS data (with 2.5D stacking and depth)
│   └── transforms.py     # (Optional) Custom transforms or augmentations
├── models/               # Model architecture components
│   ├── encoder_swin.py   # Swin Transformer encoder (patch embedding for 101x101 inputs)
│   ├── decoder_unet.py   # U-Net style CNN decoder with skip connections
│   ├── segmentation_model.py  # Combined model (encoder + decoder + classifier head)
│   └── classifier_head.py    # Auxiliary classification head for salt presence
├── losses/               # Loss function implementations
│   ├── dice_loss.py      # Dice loss for segmentation
│   ├── focal_loss.py     # Focal loss for segmentation
│   ├── boundary_loss.py  # Boundary loss (distance transform based)
│   └── combined_loss.py  # Composite loss combining dice, focal, boundary, and BCE
├── train/                # Training pipeline code
│   ├── trainer.py        # Training loop, validation, early stopping, logging
│   └── pretrain_mae.py   # (Optional) Self-supervised pretraining (MAE) routine
├── inference/            # Inference and deployment code
│   ├── predictor.py      # Model inference with MC dropout for uncertainty
│   ├── uncertainty.py    # Functions for uncertainty estimation and refinement
│   └── submission.py     # Kaggle RLE submission generation
├── utils/                # Utility functions (metrics, etc.)
│   ├── metrics.py        # e.g., IoU, accuracy calculations
│   ├── rle.py            # Utility to convert masks to run-length encoding
│   └── config_parser.py  # Helper to load configurations
└── run.py                # Main script to run training or inference using modules above
```

Each directory contains an `__init__.py` to allow easy imports (e.g., `from models import SegmentationModel`). This separation ensures each concern (model architecture, loss definitions, data loading, etc.) can be developed and tested in isolation.

## Model Architecture: Swin Transformer Encoder + U-Net Decoder (2.5D Input)

**Architecture Overview:** We implement a **hybrid Swin Transformer U-Net** architecture. The encoder is a **Swin Transformer** that extracts multi-scale features with self-attention (to capture global context), and the decoder is a **U-Net style CNN** that uses those features (with skip connections) to produce a segmentation mask. Additionally, we include an **auxiliary classifier head** to predict if any salt is present in the image (multi-task learning). The model accepts **2.5D inputs**: each sample is a stack of 3 consecutive seismic slices treated as a 3-channel input image (simulating volumetric context).

- **Swin Transformer Encoder:** The Swin Transformer is a hierarchical Vision Transformer that partitions the image into patches and applies **shifted window self-attention** to capture both local and global features ([ST-Unet: Swin Transformer boosted U-Net with Cross-Layer Feature ...](https://www.sciencedirect.com/science/article/abs/pii/S0010482522012240#:~:text=,features%20of%20different%20perception)) ([Microsoft Word - conference-template-a4-230425-v2](https://arxiv.org/pdf/2304.12615#:~:text=%EF%82%B7%20The%20Swin%20Transformer%20block,National%20Natural%20Science%20Founda%02tion%20of)). We adapt it to 101×101 grayscale seismic images by customizing the patch partitioning:
  - The input (after stacking slices) has shape `3×101×101` (3 channels for 3 slices). We use a small patch size (e.g. 4×4) and possibly pad the image to the nearest patch-grid size (e.g., pad to 104×104) so that patch embedding divides the image evenly. The patch embedding layer is a learned conv layer that maps 3 input channels to an initial latent dimension ([ST-Unet: Swin Transformer boosted U-Net with Cross-Layer Feature ...](https://www.sciencedirect.com/science/article/abs/pii/S0010482522012240#:~:text=,features%20of%20different%20perception)).
  - The encoder produces a hierarchy of feature maps at different resolutions (e.g., 1/4, 1/8, 1/16 of input size) through patch merging and transformer blocks ([ST-Unet: Swin Transformer boosted U-Net with Cross-Layer Feature ...](https://www.sciencedirect.com/science/article/abs/pii/S0010482522012240#:~:text=,features%20of%20different%20perception)). We capture features from multiple stages to use in the decoder (similar to U-Net skip connections).

- **U-Net CNN Decoder:** The decoder is a convolutional upsampling path that gradually reconstructs the segmentation mask:
  - We take the encoded feature maps from the Swin encoder’s multiple stages. For each encoder stage (at a given resolution), there is a corresponding decoder block. The encoder features are **skip-connected** to the decoder (either by direct concatenation or add after upsampling) ([Image Segmentation: Kaggle experience | by Vlad Shmyhlo | TDS Archive | Medium](https://medium.com/data-science/image-segmentation-kaggle-experience-9a41cb8924f0#:~:text=U,wise%20segmentation)), combining fine-grained low-level features with high-level context ([Image Segmentation: Kaggle experience | by Vlad Shmyhlo | TDS Archive | Medium](https://medium.com/data-science/image-segmentation-kaggle-experience-9a41cb8924f0#:~:text=U,wise%20segmentation)).
  - Each decoder block may consist of upsampling (e.g., transposed convolution or interpolation + conv) followed by convolutional layers that refine the features.
  - The final decoder output is a feature map of the original image size (101×101) with a single channel (salt mask probability per pixel). A sigmoid is applied to produce the segmentation probability map.

- **2.5D Input Handling:** To incorporate volumetric context, our `Dataset` yields inputs as 3-channel images: for a given seismic slice, it also provides the previous and next slice (when available) as additional channels. (If at volume boundaries or if such ordering is not inherent, we can duplicate the slice or use a placeholder for missing neighbors.) This **"2.5D" approach** allows the model to use adjacent-slice information without the complexity of full 3D convolutions ([Comparing 3D, 2.5D, and 2D Approaches to Brain Image Auto ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC9952534/#:~:text=Comparing%203D%2C%202,dimensional%20segmentation.%203D%20segmentation)), and is known to improve segmentation consistency across slices ([Comparing 3D, 2.5D, and 2D Approaches to Brain Image Auto ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC9952534/#:~:text=Comparing%203D%2C%202,dimensional%20segmentation.%203D%20segmentation)). The depth value provided for each slice (a scalar indicating how deep that slice is) can be optionally fed into the model (e.g., appended as an extra input channel by repeating it as a constant image or used in a positional encoding). For simplicity, our initial implementation uses only the image stacks as input channels and does not explicitly incorporate the depth scalar.

Below is a **code snippet** for the combined model architecture (`models/segmentation_model.py`), illustrating the integration of the Swin encoder, U-Net decoder, and classification head:

```python
import torch
import torch.nn as nn
from models.encoder_swin import SwinEncoder
from models.decoder_unet import UNetDecoder
from models.classifier_head import ClassifierHead

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

In this design, `SwinEncoder` is a module (perhaps adapted from the official Swin Transformer code) that returns a list of feature maps (`enc_feats`), e.g., features after each stage. `UNetDecoder` upsamples and combines these to produce `seg_out`. `ClassifierHead` takes the deepest encoder feature (after the final Swin block) and outputs a single logit for binary classification (salt present vs not).

**Skip Connections:** If the Swin encoder is implemented to gradually downsample (like a CNN), we can naturally take outputs at 1/2, 1/4, 1/8, etc. of the input resolution. These are fed into corresponding decoder blocks (which upsample to the same spatial size and concatenate). This retains U-Net's strength of combining context and fine details ([Image Segmentation: Kaggle experience | by Vlad Shmyhlo | TDS Archive | Medium](https://medium.com/data-science/image-segmentation-kaggle-experience-9a41cb8924f0#:~:text=U,wise%20segmentation)). If needed, projection layers adapt the channel dimensions when merging encoder features into the decoder.

**Patch Embedding Adaptation:** Because our images are 101×101, we handle the slight misalignment with patch size. One approach is to pad the input to 104×104 (if patch size 4, since 104 is divisible by 4) at the encoder input and then crop the decoder output back to 101×101. The `SwinEncoder` can include logic to pad and unpad as needed. Another approach is allowing the last patch window to overlap partially at image edges (Swin supports non-full windows). We ensure the code is documented for how this is handled so that the segmentation aligns correctly with the original image size.

## Self-Supervised Pretraining (Masked Autoencoder - MAE)

To improve the encoder’s feature learning, we incorporate **self-supervised pretraining** using a **Masked Autoencoder (MAE)** strategy ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=The%20pre,training%20a%20seismic)) ([[2111.06377] Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377#:~:text=,models%20efficiently%20and%20effectively%3A%20we)). In MAE pretraining, the model learns to reconstruct missing parts of the input when a large portion of the image is masked out. This helps the encoder learn rich, domain-specific representations of seismic textures and structures without labels ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=applications,seismic%20salt%20and%20facies%20classification)).

Our codebase provides a **pretraining script** (`train/pretrain_mae.py`) that uses the same Swin Transformer encoder in an MAE setup:
- During pretraining, we **mask** a high percentage (e.g. 75%) of input pixels or patches and feed the partially visible image into the encoder ([[2111.06377] Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377#:~:text=,supervisory%20task.%20Coupling%20these%20two)).
- A lightweight decoder (different from the U-Net decoder) is attached for pretraining purposes to reconstruct the original image from the encoder’s latent representation ([[2111.06377] Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377#:~:text=random%20patches%20of%20the%20input,models%20efficiently%20and%20effectively%3A%20we)).
- After pretraining, we discard the MAE decoder and transfer the weights of the encoder into our segmentation model.

This approach follows He et al. (2021) ([[2111.06377] Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377#:~:text=,models%20efficiently%20and%20effectively%3A%20we)), who showed that an asymmetric encoder-decoder with high masking yields excellent learned features. In geophysics, recent studies confirm that **ViT-MAE pretraining on seismic data can boost salt segmentation performance** ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=datasets%20with%20synthetic%20data%20and,seismic%20salt%20and%20facies%20classification)) ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=of%2060%2C000%202D%20crops%20for,seismic%20salt%20and%20facies%20classification)) by leveraging large unlabeled datasets. We design our code to easily swap the pretrained encoder; for instance, one could plug in a contrastive-pretrained encoder (e.g., SimCLR or MoCo) by implementing the same `SwinEncoder` interface.

**Implementation Details:** In `train/pretrain_mae.py`, we define a training loop that optimizes a reconstruction loss (e.g., MSE between reconstructed and original pixels) on the masked images. We save the encoder weights after pretraining:
```python
# Pseudocode for MAE pretraining loop
encoder = SwinEncoder(img_size=img_size, in_channels=1)            # For MAE, might use 1-channel input
mae_decoder = MAEDecoder(latent_dim=encoder.embed_dim, ... )       # a lightweight decoder for reconstruction
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(mae_decoder.parameters()), lr=1e-4)

for epoch in range(num_epochs):
    for images in unlabeled_loader:  # iterate over seismic images
        masked_imgs, mask = mask_random_patches(images, mask_ratio=0.75)
        latent = encoder(masked_imgs)            # encoder processes visible patches
        recon = mae_decoder(latent, mask)        # decoder tries to reconstruct original
        loss = reconstruction_loss(recon, images) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
# Save encoder weights for later fine-tuning
torch.save(encoder.state_dict(), "pretrained_encoder.pth")
```
*(Note: The actual implementation might integrate an existing MAE library or the official MAE code ([facebookresearch/mae: PyTorch implementation of MAE ... - GitHub](https://github.com/facebookresearch/mae#:~:text=facebookresearch%2Fmae%3A%20PyTorch%20implementation%20of%20MAE,Autoencoders%20Are%20Scalable%20Vision%20Learners)) for efficiency.)*

After pretraining, the main training script can load these weights:
```python
model = SaltSegmentationModel(...)
model.encoder.load_state_dict(torch.load("pretrained_encoder.pth"))
```
This modular design means you can toggle pretraining on/off via config. The encoder class is designed to be reusable outside the segmentation model, facilitating tasks like pretraining or separate evaluation of the encoder.

By providing the option for self-supervised learning, our codebase ensures the encoder has **flexible initialization**: one can start from scratch, use MAE pretrained weights, or even use **contrastive learning** weights (with a compatible architecture). The key is that any encoder must output the expected multi-scale features to be consumed by the decoder, adhering to the interface defined in `SwinEncoder`.

## Multi-Task Learning: Segmentation + Salt Presence Classification

To exploit multi-task learning benefits, our model jointly learns **pixel-wise segmentation** and **image-level classification**. A **shared encoder** feeds into two heads:
1. **Segmentation head:** the U-Net decoder producing a mask.
2. **Classification head:** a small head producing a binary prediction of whether the image contains any salt deposit.

Multi-task learning can improve representation quality by providing **additional supervision signals** and encouraging the encoder to learn features useful for both localizing salt and recognizing its presence ([Silver(top 3%) solution for tgs-salt-identification-challenge](https://www.linkedin.com/pulse/silvertop-3-solution-yunfei-duan#:~:text=Try%20a%20dedicated%20classifier)). In the TGS Salt dataset, a significant fraction of images have no salt at all (approximately 38% had empty masks) ([TGS Salt Identification Challenge | Kaggle](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933#:~:text=TGS%20Salt%20Identification%20Challenge%20,test%20images%20with%20zero%20mask)). A classification branch helps detect these cases, which can prevent false-positive segmentations by allowing the model to output an “all-clear” when no salt is present.

**Auxiliary Classification Head:** Implemented in `models/classifier_head.py`, this might simply be a fully-connected layer or small ConvNet:
- We apply global average pooling to the final encoder feature map (of size `B × C × H_enc × W_enc`) to get a `B × C` feature vector (C is last encoder channels).
- Then a linear layer (with sigmoid) produces a probability of salt presence.

For example:
```python
class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # pool to 1x1
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, feat_map):
        x = self.avgpool(feat_map)         # (B, C, 1, 1)
        x = x.view(x.size(0), -1)          # (B, C)
        logits = self.fc(x)               # (B, num_classes)
        return logits
```

During training, we use the classification head’s output to compute a binary cross-entropy loss (salt present vs absent) as part of the overall loss (explained below). The segmentation decoder and classification head share the encoder’s features, effectively **regularizing** the encoder. Kaggle practitioners found that training a dedicated classifier in parallel with a U-Net can slightly boost performance ([Silver(top 3%) solution for tgs-salt-identification-challenge](https://www.linkedin.com/pulse/silvertop-3-solution-yunfei-duan#:~:text=Try%20a%20dedicated%20classifier)) – our integrated approach accomplishes this in one model.

At inference time, the classification prediction can be used in post-processing: if the classifier outputs a very low probability for salt presence, we might threshold the segmentation output to an all-zero mask (since the model is confident no salt). However, to avoid missing faint salt, we can still rely on the segmentation but use the classification as an additional guide (for example, adjust confidence or apply a higher mask threshold when the classifier says "no salt"). This logic can be implemented in the `inference/predictor.py` module.

## Loss Functions: Dice + Boundary + Focal + Auxiliary Loss

Training uses a **compound loss** that combines several components, each addressing a different aspect of the task:
- **Dice Loss:** Measures overlap between prediction and ground truth, focusing on the **region** of salt. Dice loss is effective for segmentation with class imbalance because it directly optimizes for the Dice coefficient (similar to IoU) ([Several Loss Functions - Kaggle](https://www.kaggle.com/code/frantotti/several-loss-functions#:~:text=This%20loss%20combines%20Dice%20loss,the%20default%20for%20segmentation%20models)).
- **Focal Loss:** A variant of cross-entropy that down-weights easy negatives and focuses on hard examples ([Loss Function Compilations - Kaggle](https://www.kaggle.com/code/nafin59/loss-function-compilations#:~:text=Purpose%3A%20Focal%20Loss%20is%20designed,s)). In our context, many pixels are non-salt; focal loss prevents the vast number of easy background pixels from overwhelming the gradient ([Image Segmentation: Kaggle experience | by Vlad Shmyhlo | TDS Archive | Medium](https://medium.com/data-science/image-segmentation-kaggle-experience-9a41cb8924f0#:~:text=Why%20this%20is%20bad%3F%20This,such%20as%20focal%20loss)).
- **Boundary Loss:** Emphasizes agreement on the **contour** of the salt region. We implement this using a distance transform on the ground truth mask: pixels near the true boundary are given higher weight. This is inspired by Kervadec et al. (2019) who proposed a loss based on the distance in contour space to better handle highly unbalanced segmentation ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=for%20highly%20unbalanced%20segmentations%2C%20such,of%20contours%20as%20a%20regional)) ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=performance%20and%20stability,expressed%20with%20the%20regional%20softmax)). By focusing on region interfaces instead of just region overlap, boundary loss helps recover small or thin salt structures that Dice or CE might miss ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=for%20highly%20unbalanced%20segmentations%2C%20such,contour%20flows%2C%20we)).
- **Binary Cross-Entropy (BCE) Loss for classification:** For the auxiliary salt presence prediction (a single sigmoid output), we use standard BCE.

These losses are combined linearly with tuning coefficients. For example, the **combined loss** (in `losses/combined_loss.py`) might be:
$$ L_{\text{total}} = \lambda_{dice} L_{dice} + \lambda_{focal} L_{focal} + \lambda_{boundary} L_{boundary} + \lambda_{cls} L_{BCE(cls)}. $$

We may start with equal weighting or adjust based on validation. A typical choice might be $\lambda_{dice}=1, \lambda_{focal}=1, \lambda_{boundary}=1, \lambda_{cls}=0.5$ (giving slightly less weight to classification if segmentation is primary).

**Implementation:** Each loss is implemented in its own module for clarity:
- `losses/dice_loss.py` contains a function `dice_loss(pred_mask, true_mask)` that computes the Dice coefficient and returns $1 - \text{Dice}$ (so that minimizing corresponds to maximizing Dice).
- `losses/focal_loss.py` contains `focal_loss(logits, true_mask, alpha=0.8, gamma=2)` (with typical focal loss parameters).
- `losses/boundary_loss.py` may precompute a distance transform of the ground truth mask edges (using `scipy.ndimage.distance_transform_edt` or similar) to weight pixel-wise BCE or Dice errors. Alternatively, one can use the formula from the Boundary Loss paper ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=for%20highly%20unbalanced%20segmentations%2C%20such,of%20contours%20as%20a%20regional)) which effectively computes an $L_2$ distance on the predicted probabilistic contour.
- `losses/combined_loss.py` imports the above and sums them up.

Example snippet combining losses:
```python
import torch.nn.functional as F

def dice_loss(pred, target):
    # pred, target: shape (B, 1, H, W)
    smooth = 1.0  # for numerical stability
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice_coef = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1.0 - dice_coef.mean()

def focal_loss(pred_logits, target, alpha=0.8, gamma=2.0):
    # pred_logits: raw model output (before sigmoid), target: {0,1} mask
    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy(p, target, reduction='none')
    p_t = p*target + (1-p)*(1-target)  
    # Modulating factor (1-p_t)^gamma and alpha balancing
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha is not None:
        alpha_t = alpha*target + (1-alpha)*(1-target)
        loss = alpha_t * loss
    return loss.mean()

def boundary_loss(pred, target):
    # Compute distance map from target edges
    # (For brevity, using a placeholder; in practice, precompute weight map or use a differentiable approach)
    # Here we weight each pixel by its distance to the nearest true boundary
    target_np = target.cpu().numpy().astype(bool)
    dist_map = compute_distance_map(target_np)    # e.g., Euclidean DT for background and inverse for foreground
    dist_map = torch.from_numpy(dist_map).to(pred.device)
    # Use weighted BCE as boundary loss (higher weight near boundaries)
    p = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(p, target, reduction='none')
    bce_weighted = bce * dist_map  # weight matrix emphasizing boundary
    return bce_weighted.mean()

def combined_loss(pred_logits, target_mask, pred_class, target_class):
    # Segmentation losses
    l_dice = dice_loss(torch.sigmoid(pred_logits), target_mask)
    l_focal = focal_loss(pred_logits, target_mask)
    l_boundary = boundary_loss(pred_logits, target_mask)
    seg_loss = l_dice + l_focal + l_boundary
    # Classification loss (BCE with logits for numerical stability)
    cls_loss = F.binary_cross_entropy_with_logits(pred_class, target_class)
    # We can apply weighting factors here if needed
    return seg_loss + 0.5 * cls_loss
```

In practice, these loss functions would be class instances or functions configured via `configs/default.yaml` (e.g., enabling/disabling boundary loss or adjusting $\gamma$ for focal loss). The training loop will call `loss = combined_loss(seg_logits, mask, class_logits, has_salt)` and backpropagate.

By combining regional (Dice, Focal) and boundary-based losses, we aim to **capture both area overlap and edge accuracy**. The **boundary loss** is particularly useful to delineate salt edges more precisely, complementing the Dice loss ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=performance%20and%20stability,expressed%20with%20the%20regional%20softmax)). Focal loss helps when most images are largely background with a small salt region (common in this challenge). The auxiliary classification loss ensures the encoder learns to detect the presence of salt globally, which indirectly helps segmentation focus on true positive regions ([Silver(top 3%) solution for tgs-salt-identification-challenge](https://www.linkedin.com/pulse/silvertop-3-solution-yunfei-duan#:~:text=Try%20a%20dedicated%20classifier)). This comprehensive loss formulation should drive the model to high IoU and accuracy.

## Synthetic Data Interface (Diffusion Model Placeholder)

To future-proof the codebase for data augmentation via generative methods, we include scaffolding for integrating **synthetic seismic data** (e.g., from diffusion models). While not implemented fully in the initial version, the idea is to allow plugging in a **data generator** that can create new training examples.

In `data/dataset.py`, we can include a placeholder for synthetic data:
- The Dataset class could have a flag `use_synthetic` and a reference to a `SyntheticDataGenerator` object.
- If `use_synthetic` is true, the dataset could on-the-fly mix real samples with generated samples (for example, for each batch, a fraction of images come from the generator).
- The `SyntheticDataGenerator` (perhaps in `data/synthetic.py`) might use a pre-trained diffusion model or GAN to generate realistic seismic slices and corresponding masks.

Example pseudo-code:
```python
class SyntheticDataGenerator:
    def __init__(self, model_checkpoint):
        # Load a generative model (diffusion or GAN) trained on seismic data
        self.model = load_diffusion_model(model_checkpoint)
    def generate_sample(self):
        # Use the model to sample a synthetic seismic image and mask
        img, mask = self.model.sample()
        return img, mask

class SaltDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, use_synthetic=False, synth_generator=None, transform=None):
        # df: dataframe with file IDs and depths
        self.df = df
        self.use_synthetic = use_synthetic
        self.synth_generator = synth_generator
        ...
    def __getitem__(self, idx):
        if self.use_synthetic and idx % 5 == 0:  # e.g., every 5th sample is synthetic
            img, mask = self.synth_generator.generate_sample()
        else:
            img = load_image(self.df.iloc[idx]["id"])
            mask = load_mask(self.df.iloc[idx]["id"])
        # If using 2.5D, also load neighbor slices for img
        img = self._stack_neighbors(idx, img)
        # Apply transforms if any (augmentations, normalization)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask
```

Initially, `SyntheticDataGenerator.load_diffusion_model` could just be a stub or return random noise (since we don't have a real model yet). The key is that the architecture anticipates an eventual integration of synthetic data:
- **Use case:** One might train a diffusion model (like Stable Diffusion or a custom conditional diffusion) on seismic data to generate plausible seismic slices and their salt masks. This could address limited data by augmenting the training set with more variety.
- Our design allows adding such data without changing the training loop logic – we just treat them as additional samples.

In the future, when a diffusion model is ready, replacing the placeholder is straightforward. For example, *Sheng et al. 2023* and others have explored using large seismic datasets to pre-train foundation models and even generate synthetic seismic examples ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=The%20pre,training%20a%20seismic)). While implementing a diffusion model is beyond this task, the code hooks ensure we can incorporate such advances by providing a unified interface to real and synthetic data.

## Uncertainty Estimation and Model Refinement

To gauge the model’s confidence and improve reliability, we incorporate **uncertainty estimation** via **Monte Carlo Dropout** and an optional **uncertainty-guided refinement** module.

- **Monte Carlo Dropout (MC Dropout):** At inference time, we perform multiple forward passes with dropout layers active to sample from the model’s predictive distribution ([[PDF] A Quantitative Comparison of Epistemic Uncertainty Maps Applied to ...](https://www.melba-journal.org/pdf/2021:013.pdf#:~:text=,Carlo)). This gives us a mean prediction and an uncertainty map (e.g., pixel-wise variance) as an approximation of epistemic uncertainty (model uncertainty) ([[PDF] A Quantitative Comparison of Epistemic Uncertainty Maps Applied to ...](https://www.melba-journal.org/pdf/2021:013.pdf#:~:text=,Carlo)). We modify our model such that certain layers (e.g., Transformer blocks or decoder convs) use `nn.Dropout(p=0.1)` and **do not turn off dropout during evaluation** if MC sampling is requested. In `inference/predictor.py`, we can implement:
  ```python
  def predict_with_uncertainty(model, image, num_samples=10):
      model.train()  # ensure dropout is on
      preds = []
      for _ in range(num_samples):
          with torch.no_grad():
              seg_logit, class_logit = model(image)
              preds.append(torch.sigmoid(seg_logit))  # accumulate probability maps
      preds = torch.stack(preds, dim=0)  # shape (num_samples, 1, H, W)
      mean_pred = preds.mean(dim=0)
      var_pred = preds.var(dim=0)  # variance of predictions
      return mean_pred, var_pred
  ```
  Here, `mean_pred` is the average segmentation probability, and `var_pred` is an uncertainty estimate per pixel (high variance means the model is unsure). We still get a classification prediction for each sample; its variance indicates uncertainty in presence/absence as well.

  *Note:* We only enable dropout in layers we intentionally included for MC dropout (to avoid messing batchnorm etc.). Alternatively, we can use a Bayesian approach or deep ensemble, but MC Dropout is straightforward and was shown by Gal & Ghahramani (2016) to approximate a Bayesian neural network ([[PDF] A Quantitative Comparison of Epistemic Uncertainty Maps Applied to ...](https://www.melba-journal.org/pdf/2021:013.pdf#:~:text=,Carlo)).

- **Uncertainty-Guided Refinement:** We include an optional second stage model – essentially a smaller U-Net (or any post-processor) that takes the initial segmentation and the uncertainty map to refine the prediction. The rationale is that regions of high uncertainty often coincide with object boundaries or areas where the model might have missed small segments. By training a refiner network to focus on these ambiguous regions, we can improve final accuracy:
  - The **refinement U-Net** could take as input the original image, the initial segmentation (perhaps thresholded or as probabilities), and the uncertainty map. It then outputs a corrected mask.
  - Training this refiner would require generating dropout predictions on the training set (or a validation set) to get uncertainty maps as training data. For simplicity, one could generate these once and treat the refiner training as a separate phase after the main model is trained.

In `inference/uncertainty.py`, we provide functions to get uncertainty maps and a template for a refiner:
```python
class UncertaintyRefinementNet(nn.Module):
    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        # A small U-Net that takes [initial_mask, uncertainty_map] as input (2 channels)
        # and outputs a correction mask (or directly refined mask).
        # Define a tiny U-Net architecture:
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, 2*base_channels, kernel_size=3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*base_channels, base_channels, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )
    def forward(self, mask_pred, uncertainty_map):
        x = torch.cat([mask_pred, uncertainty_map], dim=1)  # concatenate along channel
        features = self.encoder(x)
        refined = self.decoder(features)
        return refined
```

This `UncertaintyRefinementNet` is a placeholder example. In practice, one might include skip connections or use the original image as an additional input channel for refinement. The goal is that the refiner learns to **add or remove mask regions** where the model was uncertain. For instance, if a thin salt filament was only partially detected, the uncertainty there will be high, and the refiner can learn from ground truth that those uncertain regions should actually be filled in.

**Usage:** During inference, if enabled, we would:
1. Obtain `mean_pred` and `var_pred` from MC Dropout.
2. Threshold `mean_pred` at some probability (e.g., 0.5) to get initial binary mask.
3. Feed `mask_pred_binary` and `var_pred` into the refinement net to get a refined mask.
4. (Optionally, combine or take a logical OR of the refined mask and initial mask).

The refined output is then used for the final prediction. This two-stage approach (predict -> refine) guided by uncertainty can yield more accurate results, especially in tricky regions ([Uncertainty‐guided U‐Net for soil boundary segmentation using ...](https://onlinelibrary.wiley.com/doi/10.1111/mice.13396#:~:text=,layers%20using%20binary%20variables)) ([Unified medical image segmentation by learning from uncertainty in ...](https://www.sciencedirect.com/science/article/abs/pii/S0950705122000594#:~:text=Unified%20medical%20image%20segmentation%20by,for%20automatic%20medical%20image%20segmentation)). Our codebase makes this an **optional** component: the main `SaltSegmentationModel` does not include it, but the `inference` module can utilize it if a trained refiner is available. We could train the refiner by a script that uses saved predictions on the training set as input.

By capturing model uncertainty, we also gain more insight into predictions – the `var_pred` can be visualized to highlight where the model is unsure. This is valuable in a field like seismic interpretation, where a human may want to review regions of high uncertainty.

## Full Training Pipeline

The training pipeline brings together the above components in a configurable way. The main script (`run.py`) and the `train/trainer.py` handle the process:

**Data Loading:** We use the `SaltDataset` (in `data/dataset.py`) to load images, and PyTorch `DataLoader` for batching. Key points:
- Before training, we might preprocess all images to 101×101 resolution (the Kaggle data is already 101×101) and load the depth values.
- The dataset can be configured to use 2.5D stacking. If `use_neighbors=True` in config, the dataset will load the slice with index-1 and index+1 (if exists) to form the 3-channel input. We assume the dataset is sorted by some spatial order or the CSV of the competition provides indices that allow neighbor identification. (If not, the simplest is to treat the data as sorted by depth and take neighbors in that sorted order, acknowledging it’s an approximation.)
- We include common **data augmentations** (implemented in `data/transforms.py`) such as horizontal flips, vertical flips, small rotations or elastic distortions (as seismic images can be flipped without losing meaning). Augmentations help generalize the model and were commonly used in the Kaggle solutions.

**Training Loop:** Implemented in `train/trainer.py`, it will:
1. Parse configuration for hyperparameters (learning rate, batch size, loss weights, etc.).
2. Initialize the model (`SaltSegmentationModel`), and if specified, load pretrained encoder weights.
3. Move model to GPU if available, set up optimizer (e.g., Adam or AdamW), and possibly a learning rate scheduler (like cosine annealing or ReduceLROnPlateau).
4. Loop over epochs:
   - For each batch from DataLoader:
     - Forward pass to get `seg_logits, class_logits`.
     - Compute the combined loss: `loss = combined_loss(seg_logits, true_mask, class_logits, true_label)`.
     - Backpropagate `loss` and update model parameters.
   - Compute validation metrics on a hold-out set periodically (e.g., end of each epoch).
   - Log training loss and validation score (IoU/Dice) using `print` or a logging library (could integrate with TensorBoard or Weights & Biases for richer logging).
   - Apply **early stopping**: if validation score hasn't improved for X epochs, or a maximum number of epochs reached, stop training to avoid overfitting.

We will also save checkpoints:
- Save the **best model** (according to validation Dice or IoU) to `models/best_model.pth`.
- Possibly save last epoch model and intermediate ones for safety.

**Pseudo-code (simplified):**
```python
for epoch in range(1, config.epochs+1):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        labels = (masks.sum(dim=(1,2,3)) > 0).float().unsqueeze(1)  # whether each mask has any salt
        seg_logits, class_logit = model(images)
        loss = combined_loss(seg_logits, masks, class_logit, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # Validation
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            seg_logits, class_logit = model(images)
            preds = torch.sigmoid(seg_logits) > 0.5
            # compute IoU or Dice on preds vs masks
            val_iou += iou_metric(preds, masks)
    val_iou /= len(val_loader)
    print(f"Epoch {epoch}: Train loss = {epoch_loss/len(train_loader):.4f}, Val IoU = {val_iou:.4f}")
    # Checkpointing
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.early_stop_patience:
            print("Early stopping triggered.")
            break
    scheduler.step(val_iou)  # if using a scheduler that monitors val score
```

This loop ensures the model trains until convergence on validation performance. We monitor IoU (or Dice) because that is the primary metric in the competition (and it aligns with our Dice loss objective).

We also log the metrics each epoch. If integrated with a logger, these would be recorded for later analysis or graphing. Our code can easily be extended to k-fold cross-validation if needed (the Kaggle competition allowed using local validation splits since test labels were not provided).

**Kaggle Submission (RLE generation):** After training, we use `inference/submission.py` to generate results on the test set. The Kaggle competition expects run-length encoding (RLE) of the binary mask for each test image. We implement a utility function (in `utils/rle.py`) to convert a binary mask (101×101) into RLE format:
```python
def mask_to_rle(mask):
    # mask: 2D numpy array of {0,1}
    pixels = mask.flatten(order='F')  # flatten in column-major (Fortran) order as Kaggle expects
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    # runs now contains pairs of (start_index, length)
    rle = ' '.join(str(x) for x in runs)
    return rle
```
We iterate over each test image, predict the mask (with thresholding), and write out the RLE string along with the image ID in a submission CSV. The `run.py` could have a mode like `--inference` to do this after training. 

By encapsulating this in `submission.py`, we make it easy to adapt to any output format changes or to use the same logic for validation set evaluation if needed.

## Efficiency and Deployment

We incorporate features to make the model efficient and ready for deployment:

- **Model Export (ONNX/TorchScript):** After training, users may want to deploy the model. We ensure our model is compatible with TorchScript (by avoiding dynamic control flows that TorchScript cannot handle) and ONNX export. For example, in `inference/predictor.py` or a separate `export.py`, we include:
  ```python
  # TorchScript export
  scripted = torch.jit.trace(model, example_input=torch.rand(1, 3, 101, 101))
  scripted.save("model_scripted.pt")
  # ONNX export
  torch.onnx.export(model, torch.rand(1, 3, 101, 101), "model.onnx", opset_version=12,
                    input_names=["input"], output_names=["mask", "class"])
  ```
  This allows the trained model to be used in C++ environments or accelerated inference frameworks. The codebase might include minor adjustments (e.g., replacing some layers not supported by ONNX with equivalent operations).

- **Inference Speed:** We can utilize the fact that the images are small (101×101) to batch many for inference. The `predictor.py` can handle both single image and batch inference. We also have the option to disable the classification head during final test time if, for instance, we found it unnecessary (just not use its output).

- **Optional Knowledge Distillation:** Hooks for knowledge distillation can be added to train a smaller model to mimic the large model. For instance, if we wanted a lighter model for deployment, we could train a student (say a smaller CNN) to replicate the outputs of the Swin U-Net model. While we won't implement a full KD pipeline here, our code is structured to allow multiple models during training. For example, one could add a `teacher_model` in trainer and define a distillation loss:
  ```python
  # Pseudo-distillation step
  teacher_model.eval()
  with torch.no_grad():
      teacher_logits, _ = teacher_model(images)
  student_logits, student_class = student_model(images)
  kd_loss = F.kl_div(torch.log_softmax(student_logits, dim=1),
                     torch.softmax(teacher_logits, dim=1), reduction='batchmean')
  loss = combined_loss(student_logits, mask, student_class, label) + 0.1 * kd_loss
  ```
  This way, we leverage the powerful Swin U-Net as a teacher to train a faster student. The code supports this by modular design – one can add a new script or extend `trainer.py` for this purpose without entangling it with base training logic.

- **Model Compression:** Similarly, we anticipate using PyTorch’s quantization or pruning for compression. The `utils/` folder might include a `quantize.py` that uses `torch.quantization` to post-train quantize the model for int8 inference, or a `prune.py` to remove insignificant weights. Because our model is built from standard PyTorch layers, it should be compatible with these APIs. For example, after training:
  ```python
  from torch.quantization import quantize_dynamic
  q_model = quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
  torch.jit.save(torch.jit.script(q_model), "model_quantized.pt")
  ```
  This would produce a size- and latency-optimized version. We keep these as optional steps, invoked if specified in the `configs` (e.g., `config.export.onnx=True`, `config.export.quantize=True`).

Finally, we ensure that the codebase is **extensible**. New encoders (e.g., different transformer variants or CNN backbones) can be added to `models/encoder_*.py` and plugged into the `SaltSegmentationModel`. Loss functions can be toggled or new ones added without altering the model code. The data pipeline can accommodate new input types (synthetic data, or additional modalities). This modular design makes the codebase suitable not only for the TGS Salt competition, but also as a template for similar segmentation challenges.

## Conclusion and Usage

In summary, we have created a comprehensive PyTorch codebase for salt body segmentation with a focus on modern techniques. The **hybrid Swin U-Net** architecture leverages transformer-based contextual learning with CNN decoding for precise masks. **Self-supervised MAE pretraining** provides a strong initialization by learning seismic structures without labels ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=datasets%20with%20synthetic%20data%20and,seismic%20salt%20and%20facies%20classification)). **Multi-task learning** (segmentation + classification) improves the model’s ability to identify difficult cases ([Silver(top 3%) solution for tgs-salt-identification-challenge](https://www.linkedin.com/pulse/silvertop-3-solution-yunfei-duan#:~:text=Try%20a%20dedicated%20classifier)). A combination of **Dice, Focal, and Boundary losses** guides the model to maximize overlap while capturing fine boundaries ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=for%20highly%20unbalanced%20segmentations%2C%20such,of%20contours%20as%20a%20regional)). We prepared for **synthetic data augmentation** by including hooks for diffusion model integration. **Uncertainty estimation** with MC Dropout gives insight into model confidence and allows an optional refinement stage for even better results. The training pipeline covers data loading (with 2.5D context and augmentations), training loops with early stopping, and Kaggle-specific output generation (RLE masks). We also included considerations for **deployability and compression** (export to ONNX/TorchScript, potential distillation, quantization).

With this modular setup, you can run `python run.py --config configs/default.yaml` to train the model. The `run.py` can parse arguments to switch between training and inference modes (for example, `--mode train` vs `--mode predict`). After training, use `--mode predict` along with a path to the test data and a trained model checkpoint to output the submission CSV.

This codebase provides an **initial implementation** for each part of the project. While some components (like synthetic data generation or refinement training) are placeholders, they establish a clear blueprint. Developers can fill in these parts as the relevant models or data become available. The result is a flexible and powerful starting point for not just the TGS Salt Identification challenge, but for tackling segmentation tasks in geoscience with the latest deep learning tools. 

**References:**

- Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," 2021 – introduced the Swin Transformer block for efficient vision transformers ([ST-Unet: Swin Transformer boosted U-Net with Cross-Layer Feature ...](https://www.sciencedirect.com/science/article/abs/pii/S0010482522012240#:~:text=,features%20of%20different%20perception)).
- Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation," arXiv 2021 – demonstrated a U-Net architecture using Swin Transformer in encoder and decoder ([Microsoft Word - conference-template-a4-230425-v2](https://arxiv.org/pdf/2304.12615#:~:text=,wise%20perspective%20with)).
- He et al., "Masked Autoencoders Are Scalable Vision Learners," 2021 – proposed the MAE approach for self-supervised pretraining of vision transformers ([[2111.06377] Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377#:~:text=,models%20efficiently%20and%20effectively%3A%20we)).
- Kaggle TGS Salt Identification Challenge (2018) discussions – highlighted the high proportion of empty masks (~38%) ([TGS Salt Identification Challenge | Kaggle](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933#:~:text=TGS%20Salt%20Identification%20Challenge%20,test%20images%20with%20zero%20mask)) and successful use of an auxiliary classifier ([Silver(top 3%) solution for tgs-salt-identification-challenge](https://www.linkedin.com/pulse/silvertop-3-solution-yunfei-duan#:~:text=Try%20a%20dedicated%20classifier)).
- Kervadec et al., "Boundary loss for highly unbalanced segmentation," Medical Image Analysis 2021 – introduced the boundary loss concept for segmentation focusing on contour agreement ([[1812.07032] Boundary loss for highly unbalanced segmentation](https://arxiv.org/abs/1812.07032#:~:text=for%20highly%20unbalanced%20segmentations%2C%20such,of%20contours%20as%20a%20regional)).
- Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," ICML 2016 – showed that enabling dropout at inference (MC Dropout) provides a way to estimate model uncertainty ([[PDF] A Quantitative Comparison of Epistemic Uncertainty Maps Applied to ...](https://www.melba-journal.org/pdf/2021:013.pdf#:~:text=,Carlo)).
- TGS Article (Sansal et al., First Break 2025) – discussed scaling seismic foundation models and the efficacy of MAE pretraining for salt interpretation ([Scaling Seismic Foundation Models](https://www.tgs.com/articles/scaling-seismic-foundation-models#:~:text=datasets%20with%20synthetic%20data%20and,seismic%20salt%20and%20facies%20classification)).