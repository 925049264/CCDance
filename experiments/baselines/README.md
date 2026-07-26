# CCDance Baseline Reproduction Suite

## Directory Structure
```
experiments/baselines/
├── usdl/                  # USDL (CVPR 2020)
├── gdlt/                  # GDLT (ACM MM 2020)
├── pose_lstm_aqa/         # Pose-LSTM AQA Baseline
├── stgcn_aqa/             # ST-GCN AQA Baseline
└── transformer_aqa/       # Transformer AQA Baseline
    ├── classification/    # Grade classification code + logs
    └── generation/        # Comment generation code + logs
```

## Data Path
- Dataset root: ~/software/emc_results/ (22 dance genres, 175 videos)
- SMPL-H poses: per-frame JSON files in each student directory
- Audio: 1.mp3 per genre
- Teacher evaluations: *_EN.txt files per student

## Environment
- Ubuntu, 4× NVIDIA L20 (48GB each), CUDA 12.x
- Python 3.10, PyTorch 2.x
- Key packages: torch, xgboost, scikit-learn, librosa, sentence-transformers

## Quick Start
```bash
# Activate environment
cd ~/software/emc_results/experiments/baselines/

# Run individual baseline (example)
cd usdl/classification/
python train.py --data_root ~/software/emc_results/ --gpu 0
```

## Baselines to Reproduce (5 papers × 2 tasks = 10 experiments)

### Task 1: Grade Classification (A/B/C)
| Baseline | Input | Architecture |
|----------|-------|-------------|
| USDL | SMPL poses | Score distribution learning |
| GDLT | SMPL poses | Group-aware feature learning |
| Pose-LSTM | SMPL poses | Bidirectional LSTM + attention |
| ST-GCN | SMPL poses (graph) | Spatial-temporal GCN |
| Transformer | SMPL poses (sequence) | Transformer encoder |

### Task 2: Comment Generation
| Baseline | Input | Architecture |
|----------|-------|-------------|
| USDL-ext | SMPL + grade | Distribution → text decoder |
| GDLT-ext | SMPL + grade | Group features → LSTM decoder |
| Pose-LSTM-gen | SMPL | LSTM encoder-decoder |
| ST-GCN-gen | SMPL (graph) | GCN encoder + LSTM decoder |
| Transformer-gen | SMPL (seq) | Transformer encoder-decoder |

## Evaluation Protocol
- Random split: 70/15/15 stratified by grade
- Cross-performer: leave-one-performer-out (18 genres)
- Cross-genre: leave-one-genre-out (22 genres)
- 5 random seeds (42, 123, 456, 789, 1024)
- Metrics: Accuracy, Macro-F1, QWK (classification); BLEU-1/4, ROUGE-L, BERTScore (generation)
