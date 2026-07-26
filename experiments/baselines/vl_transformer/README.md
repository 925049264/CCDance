# VL-Transformer Baseline

Implementation of the Visual Language Transformer Framework for Multimodal
Dance Performance Evaluation (Chen, SciRep 2025).

## Architecture

### Classification (`classification/`)

```
SMPL Pose (B, T, 69)              Audio Features (B, T_a, 64)
         |                                  |
  STGCNEncoder (256-D)          MusicLSTMEncoder (256-D)
         |                                  |
         +---------- Concat(512-D) ---------+
                            |
                     Linear(512, 256)
                            |
                 MLP(256 -> 128 -> 3)
                            |
                       Grade (A/B/C)
```

### Generation (`generation/`)

Same encoder architecture with:
- **Teacher embedding regression**: Predicts Sentence-BERT 768-D embeddings
- **LSTM text decoder** (optional): Token-level comment generation with
  teacher forcing during training and greedy decoding at inference

## Training Protocol

### Phase 1: InfoNCE Contrastive Pretraining
- 100 epochs
- Aligns pose and music embeddings in a shared latent space
- Symmetric InfoNCE loss (pose2music + music2pose)
- Temperature tau = 0.07
- Only encoder parameters are updated
- Cosine annealing LR schedule from 1e-4

### Phase 2: Classification Fine-tuning
- 50 epochs
- All parameters fine-tuned end-to-end
- AdamW optimizer (lr=1e-4, wd=1e-4)
- Cosine annealing LR schedule
- Early stopping with patience 10 (monitoring validation accuracy)
- Stratified 70/15/15 train/val/test split

## Data

- **Pose**: SMPL axis-angle parameters (69-D), 300 frames per sequence,
  uniform sampling/padding
- **Audio**: 64-D features per frame (MFCC 20 + delta 20 + chroma 12 +
  tempogram 11 + onset strength 1)
- **Grades**: A=0 (excellent), B=1 (good), C=2 (needs improvement)
- **Teacher comments**: English text evaluations, encoded as Sentence-BERT
  768-D embeddings for the generation task

## Files

| File | Description |
|------|-------------|
| `classification/model.py` | VLTransformerClassifier with InfoNCE loss |
| `classification/train.py` | Two-stage training (pretrain + finetune) |
| `classification/test.py` | Evaluation on held-out test set |
| `generation/model.py` | VLTransformerGenerator with embedding regression |
| `generation/train.py` | Generation training (teacher embedding prediction) |

## Usage

```bash
# Classification: InfoNCE pretrain + fine-tuning
python classification/train.py --gpu 0

# Classification: fine-tuning only (skip pretraining)
python classification/train.py --gpu 0 --no-pretrain

# Classification: evaluate existing checkpoint
python classification/test.py --gpu 0 --seed 42 \
    --checkpoint results/seed_42/model.pt

# Generation: train teacher embedding predictor
python generation/train.py --gpu 0
```

## Key Hyperparameters

| Parameter | Classification | Generation |
|-----------|:-------------:|:----------:|
| Epochs | 50 (+100 pretrain) | 100 |
| Batch size | 16 | 16 |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Dropout | 0.3 | 0.3 |
| Hidden dim | 256 | 256 |
| Optimizer | AdamW | AdamW |
| LR schedule | Cosine annealing | Cosine annealing |
| Seeds | 42, 123, 456, 789, 1024 | same |

## Results Format

Per-seed output in `results/seed_{N}/`:
- `model.pt` - Trained model state dict
- `results.json` - Test metrics and summary
- `train_log.json` - Per-epoch training log
- `pretrain_log.json` - InfoNCE pretraining log (classification only)

Aggregated results (mean +/- std across 5 seeds):
- `results/aggregated_results.json`

## Dependencies

- PyTorch >= 1.10
- NumPy
- scikit-learn
- librosa (audio feature extraction)
- sentence-transformers (generation teacher embeddings, optional)
- Shared modules in `experiments/baselines/shared/`

## References

- Chen, J. "Visual language transformer framework for multimodal dance
  performance evaluation." Scientific Reports, 2025.
- SMPL: Loper et al. "SMPL: A Skinned Multi-Person Linear Model." SIGGRAPH 2015.
- InfoNCE: Oord et al. "Representation Learning with Contrastive
  Predictive Coding." 2018.
