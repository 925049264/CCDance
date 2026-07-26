# LeViT-Hybrid Baseline

**Reference:** Wang, "Hybrid Model Integrating LeViT Transformer and Distillation Techniques for Dance Classification," *Scientific Reports*, 2025.

## Overview

This directory contains the implementation of the LeViT-Hybrid baseline, adapted from the original image-based LeViT architecture for SMPL pose sequence classification and teacher comment generation on the CCDance dataset.

## Key Adaptation: Image -> Pose Sequence

The original LeViT-Hybrid (Wang, 2025) operates on 224x224 RGB images:

- CNN backbone (3x3 conv, 128 filters) extracts low-level features
- Patch embedding divides feature maps into patches
- LeViT-128 transformer processes patch embeddings
- Knowledge distillation from ViT-B/16 teacher (larger ViT model)
- Classification head for dance grade prediction

Since CCDance provides SMPL pose sequences (300 frames of 69-D axis-angle vectors) rather than RGB images, we adapt the architecture as follows:

### Option A (Implemented) - Pose Patch Embedding

The core idea is to treat temporal segments of pose sequences as "patches," analogous to image patches in the original LeViT.

| Original (Images) | Adapted (Pose Sequences) |
|---|---|
| 224x224x3 RGB image | 300x69 SMPL pose sequence |
| Conv patch embedding (16x16 patches) | Temporal patch embedding (15-frame patches) |
| 14x14 = 196 image patches | 300/15 = 20 temporal patches |
| Each patch: 16*16*3 = 768-D | Each patch: 69*15 = 1035-D |
| Linear projection to 128-D | Linear projection to 256-D |
| LeViT-128 (4 layers, 4 heads) | TransformerEncoder (4 layers, 8 heads, 256 dim) |
| ViT-B/16 teacher (12 layers, 768 dim) | Larger Transformer (8 layers, 8 heads, 512 dim) |
| Classification head (MLP) | MLP classifier (256 -> 256 -> 3) |

### Why This Adaptation Makes Sense

1. **Temporal patches as image patches:** Just as an image is divided into spatial patches, a pose sequence is divided into temporal patches. Each patch captures a short motion segment (15 frames ~ 0.5 seconds at 30fps).

2. **Pose Patch Embedding replaces Conv Embedding:** The original LeViT uses a convolutional stem to create patch embeddings. Since poses lack spatial locality (each of the 69 dimensions is an independent axis-angle parameter), we use a simple linear projection instead.

3. **Transformer preserves the core mechanism:** Multi-head self-attention over patches works identically whether patches represent image regions or temporal segments.

4. **Knowledge distillation follows the same principle:** A larger teacher model provides soft targets to regularize the smaller student model.

## Architecture Details

### Classification Model (`classification/model.py`)

```
LeViTHybridModel:
  - PosePatchEmbedding
      Input:  (B, 300, 69)
      Reshape: (B, 20, 1035) where each patch = 15 frames * 69 dims
      Linear:  1035 -> 256
      + CLS token + positional encoding
      Output:  (B, 21, 256)

  - Student Encoder (LeViTEncoder)
      TransformerEncoderLayer x4 (nhead=8, d_model=256, GELU, pre-norm)
      Output:  (B, 256) from CLS token

  - Teacher Encoder (LeViTEncoder)
      TransformerEncoderLayer x8 (nhead=8, d_model=512, GELU, pre-norm)
      Teacher projection: Linear(256 -> 512)
      Output:  (B, 512) from CLS token

  - Student Classifier (MLPClassifier)
      Linear(256 -> 256) -> ReLU -> Dropout -> Linear(256 -> 3)

  - Teacher Classifier (MLPClassifier)
      Linear(512 -> 512) -> ReLU -> Dropout -> Linear(512 -> 3)
```

### Distillation Loss

```
Loss = alpha * KL(student_softmax(T) || teacher_softmax(T)) * T^2
     + (1 - alpha) * CE(student_logits, labels)

Where:
  - T = temperature (default 4.0)
  - alpha = distillation weight (default 0.5)
  - T^2 scaling preserves gradient magnitude (Hinton et al., 2015)
```

## Training Protocol

### Classification (3 phases)

1. **Teacher Pretraining** (default: 20 epochs):
   - Train teacher encoder + classifier on classification task
   - Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
   - Cosine LR schedule
   - Teacher learns to produce high-quality soft targets

2. **Distillation Training** (default: 50 epochs):
   - Freeze teacher, train student
   - Combined KD + CE loss
   - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
   - Student learns from both ground truth labels and teacher soft targets

3. **Student Fine-tuning** (optional, default: 10 epochs):
   - Continue training student without distillation
   - Lower LR (lr=1e-4)
   - Allows student to adapt beyond teacher's knowledge

### Generation

- **LeViTHybridGenerator:** PosePatchEmbedding -> LeViTEncoder -> LSTMDecoder
- Teacher forcing ratio: 0.5
- Same training protocol as other CCDance generation baselines

## Usage

### Classification Training

```bash
# Full 3-phase training (recommended)
python classification/train.py --gpu 0

# Skip teacher pretraining
python classification/train.py --gpu 0 --skip-teacher-pretrain

# Direct student training (no distillation)
python classification/train.py --gpu 0 --skip-distillation --skip-teacher-pretrain

# Custom configuration
python classification/train.py --gpu 0 \
    --lr 1e-3 \
    --epochs 50 \
    --teacher-epochs 20 \
    --temperature 4.0 \
    --alpha 0.5 \
    --patch-size 15 \
    --embed-dim 256 \
    --teacher-embed-dim 512
```

### Classification Evaluation

```bash
# Evaluate distilled model (default)
python classification/test.py --gpu 0 --seed 42

# Evaluate with specific checkpoint
python classification/test.py --gpu 0 \
    --checkpoint results/seed_42/model_distilled.pt \
    --seed 42 \
    --distilled
```

### Generation Training

```bash
python generation/train.py --gpu 0 --epochs 100
```

## Differences from Original Paper

1. **Input modality:** Original uses RGB images (224x224); we use SMPL pose sequences (300x69).

2. **Patch embedding:** Original uses CNN feature extraction + patch flattening; we use direct temporal patching + linear projection.

3. **Model dimensions:** Original LeViT-128 uses 128-dim embeddings, 4 layers, 4 heads; we use 256-dim, 4 layers, 8 heads for the student to accommodate the larger patch dimension (1035 vs 768).

4. **Teacher model:** Original uses ViT-B/16 (12 layers, 768-dim, 12 heads); we use a TransformerEncoder with 8 layers, 512-dim, 8 heads as a lighter proxy.

5. **Learning rate:** Original uses 1e-5 (standard for ViT fine-tuning); we use 1e-3 (standard for training from scratch on pose data).

6. **Batch size:** Original uses 32; we use 16 (consistent with other CCDance baselines for fair comparison).

7. **Training epochs:** Original uses 100 epochs; we use 50 for classification (consistent across baselines).

## Option B (Not Implemented)

An alternative approach would be to render SMPL 3D keypoints projected to 2D as images and feed them to an actual LeViT model. This was not implemented because:

- Computationally expensive (rendering frames for each sample)
- Information loss in 3D->2D projection
- Introduces rendering pipeline dependencies
- Less direct than operating on pose vectors directly

## Files

```
levit_hybrid/
  README.md                           # This file
  classification/
    model.py                          # LeViTHybridModel, PosePatchEmbedding, LeViTEncoder
    train.py                          # 3-phase training (pretrain + distillation + finetune)
    test.py                           # Evaluation script
  generation/
    model.py                          # LeViTHybridGenerator
    train.py                          # Generation training
  checkpoints/                        # Saved model weights
  logs/                               # Training logs
  results/                            # Results per seed + aggregated
```
