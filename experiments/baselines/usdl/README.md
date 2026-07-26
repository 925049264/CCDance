# USDL Baseline -- Uncertainty-Aware Score Distribution Learning

**Paper:** Tang et al., "Uncertainty-Aware Score Distribution Learning for Action Quality
Assessment", CVPR 2020.

## Adaptation from Original Paper

### Original Architecture (Tang et al.)

```
Input video (N=10 clips)
    -> I3D backbone (shared across clips, output D=2048 per clip)
    -> 3 FC layers: 2048 -> 256 -> 128 -> m (score bins)
    -> Temporal average pooling over N clips
    -> Softmax -> score distribution over m bins
    -> Argmax -> score (in continuous score range, e.g. 0-100)
```

### Our Adaptation (SMPL Pose Input)

```
SMPL pose sequence (T=300 frames, D=69)
    -> STGCNEncoder (output 256-D)
    -> ScoreDistributionHead: 256 -> 256 -> 128 -> 10 bins
    -> Softmax -> score distribution over 10 bins
    -> Argmax -> bin index -> grade map {0-3:A, 4-6:B, 7-9:C}
```

**Key changes:**
1. **I3D -> STGCN:** The original paper uses I3D (RGB video) as backbone. We replace
   it with a Spatial-Temporal Graph Convolutional Network (STGCN) that operates on
   SMPL pose sequences. This is necessary because our input modality is 3D pose
   (69-D axis-angle vectors over 300 frames) rather than RGB video.
2. **Single-pass vs clip-based:** The original processes N=10 clips independently and
   averages them. Our STGCN processes the full 300-frame sequence in one pass with
   temporal convolutions, achieving the same temporal aggregation effect.
3. **Score bins -> discrete grades:** The original predicts continuous scores (0-100).
   Our task is 3-class grade classification (A/B/C). We map the 10-bin distribution
   to grades via a fixed mapping: bins 0-3 -> A, 4-6 -> B, 7-9 -> C.
4. **Lower learning rate:** Following the paper's observation that USDL benefits from
   lower LR, we use 1e-4 vs the default 1e-3.

## Score Distribution Learning

The core idea of USDL is to model each sample's score as a distribution rather than
a single value, capturing the inherent uncertainty in AQA.

### Training

1. **Ground-truth distribution:** For each grade label g in {0, 1, 2}, we create a
   discretized Gaussian distribution centered at g with sigma=0.5, evaluated at 10
   evenly spaced bin centers covering [0, 2].

   ```
   P_target(bin_i) = exp(-(c_i - g)^2 / (2*sigma^2))
                    / sum_j exp(-(c_j - g)^2 / (2*sigma^2))
   ```

2. **Loss:** KL divergence between predicted and ground-truth distributions.

   ```
   L_KL = sum_i P_target(i) * log(P_target(i) / P_pred(i))
   ```

### Inference

1. Model outputs softmax distribution over 10 bins.
2. Argmax gives the most likely bin index.
3. Bin index mapped to grade: bins 0-3 -> A(0), 4-6 -> B(1), 7-9 -> C(2).

## Alternative: Direct Classification

For comparison, we also provide a standard 3-class classification head:
- STGCNEncoder -> MLPClassifier (256 -> 256 -> 3) -> CrossEntropy loss

This matches conventional supervised classification and serves as a baseline to
measure the benefit of distribution learning.

## Files

```
usdl/
  README.md                         -- This file
  classification/
    model.py                        -- USDLModel (encoder + distribution + classifier)
    train.py                        -- Train both distribution and classification
    test.py                         -- Evaluate a trained checkpoint
  generation/
    model.py                        -- USDLGenerator (pose -> distribution -> text)
    train.py                        -- Train grade-conditioned text generation
```

## Hyperparameters

| Parameter            | Value       | Notes                              |
|----------------------|-------------|------------------------------------|
| Sequence length      | 300 frames  | Uniform sampling/padding           |
| Encoder output dim   | 256         | STGCN output dimension             |
| Distribution bins    | 10          | Uniform over [0, 2]                |
| Gaussian sigma       | 0.5         | For ground-truth soft labels       |
| Learning rate        | 1e-4        | Lower than default (USDL-specific) |
| Batch size           | 16          | Shared across baselines            |
| Epochs (classif.)    | 50          |                                    |
| Epochs (generation)  | 100         |                                    |
| Optimizer            | AdamW       |                                    |
| LR schedule          | Cosine      | Cosine annealing to 0              |
| Weight decay         | 1e-4        |                                    |
| Seeds                | 5           | 42, 123, 456, 789, 1024           |

## Usage

### Classification

```bash
# Train both distribution and direct classification
python classification/train.py --gpu 0

# Train only distribution learning
python classification/train.py --gpu 0 --approach distribution

# Train only direct classification
python classification/train.py --gpu 0 --approach classification

# Test a trained distribution model
python classification/test.py --gpu 0 \
    --checkpoint results/distribution/seed_42/model.pt \
    --mode distribution

# Test a trained classification model
python classification/test.py --gpu 0 \
    --checkpoint results/classification/seed_42/model.pt \
    --mode classification
```

### Generation

```bash
python generation/train.py --gpu 0 --epochs 100
```

## References

- Tang, C.-Y., et al. "Uncertainty-Aware Score Distribution Learning for Action Quality
  Assessment." CVPR 2020.
- Yan, S., et al. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition." AAAI 2018. (STGCN backbone)
