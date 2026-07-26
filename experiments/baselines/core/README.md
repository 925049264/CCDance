# CoRe Baseline -- Group-Aware Contrastive Regression

**Reference:** Yu et al., "Group-Aware Contrastive Regression for Action Quality Assessment", ICCV 2021.

## Overview

This directory reproduces the CoRe (Group-Aware Contrastive Regression) baseline adapted for the CCDance dataset with SMPL pose input.

## Architecture

### Original CoRe (Yu et al. ICCV 2021)
- I3D backbone for video feature extraction
- Group-Aware Regression Tree (GART): binary decision tree, depth=5, 32 leaves
- Each internal node: linear layer + sigmoid for routing decision
- Each leaf: predicts a continuous score (regression)
- Multi-exemplar voting (M=10) for inference
- Optimizer: Adam (lr=1e-3 tree, lr=1e-4 backbone), 0 weight decay

### CCDance Adaptation
- **Backbone:** PoseLSTMEncoder (bidirectional LSTM with attention, 256-D output) replaces I3D
- **GART preserved** with binary routing, depth=5, 32 leaves, node_dim=256
- **Classification head:** Each leaf predicts a 3-class distribution (A/B/C) instead of continuous score
- **Contrastive loss:** Pulls same-grade samples together, pushes different-grade samples apart (weighted by grade difference)
- **Multi-exemplar voting:** M=10 training samples used as exemplars; test predictions averaged over exemplar-conditioned outputs

## Files

| File | Description |
|------|-------------|
| `classification/model.py` | `CoReModel` (PoseLSTMEncoder + GART), `CoReSimple` (LSTM + MLP ablation), `CoReLoss`, `GroupAwareRegressionTree` |
| `classification/train.py` | Training pipeline with contrastive+CE loss, per-parameter-group LR (backbone vs tree), multi-seed eval |
| `classification/test.py` | Evaluation with direct forward pass and multi-exemplar voting (M=10, stratified by grade) |
| `generation/model.py` | `CoReGenerator` with grade-conditioned pose encoding + LSTM decoder |
| `generation/train.py` | Generation training with teacher forcing, vocabulary building, BLEU/ROUGE metrics |

## Key Design Decisions

### Group-Aware Regression Tree (GART)
- **Routing:** Each internal node uses Linear(256,1) + Sigmoid to produce P(left). Features are propagated left (feat + delta) or right (feat - delta) with residual-style updates.
- **Soft assignment:** Samples are not hard-assigned to a single leaf. Instead, the cumulative product of routing probabilities gives a soft distribution over all 32 leaves.
- **Aggregation:** The final prediction is the probability-weighted average over all leaf classifiers.

### Loss Function
The CoReLoss combines three terms:
1. **CE on aggregated logits** (standard classification loss)
2. **Leaf-weighted CE** (each leaf classifier should predict correctly, weighted by routing probability)
3. **Contrastive grouping** (same-grade pairs pulled together via cosine similarity maximization; different-grade pairs pushed apart proportional to grade difference)

### Training Details
- **Optimizer:** AdamW with two parameter groups:
  - Backbone: lr=1e-4, weight_decay=0.0
  - GART: lr=1e-3, weight_decay=0.0
- **Scheduler:** Cosine annealing over N epochs
- **Batch size:** 16 (shared baseline default)

### Multi-Exemplar Voting
- At test time, M=10 training exemplars are selected (stratified by grade, ~3 per grade).
- For each test sample, the backbone embedding is compared to exemplar embeddings via cosine similarity.
- Similarity-weighted exemplar predictions are averaged to produce the final output.
- The evaluation script runs multiple voting rounds and reports mean +/- std.

## Usage

### Classification Training
```bash
python core/classification/train.py --gpu 0
```

### Classification Testing
```bash
python core/classification/test.py --gpu 0 --seed 42
python core/classification/test.py --gpu 0 --seed 42 --checkpoint path/to/model.pt
```

### Generation Training
```bash
python core/generation/train.py --gpu 0
```

## Results

Results are saved under:
- `core/classification/results/` -- per-seed subdirectories with `model.pt`, `results.json`, `train_log.json`
- `core/classification/results/aggregated_results.json` -- aggregated across 5 seeds
- `core/generation/results/` -- same structure for generation

## Ablation: CoReSimple

`CoReSimple` replaces the GART with a standard MLP classifier to isolate the contribution of the tree structure. Run with the same training pipeline by swapping the model class.
