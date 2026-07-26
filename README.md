# CCDance: A Large-Scale Multi-Modal Benchmark for Chinese Dance Quality Assessment

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**CCDance (Chinese College Dance)** is the first large-scale, multi-modal benchmark dataset for Chinese dance quality assessment. It contains **175 professionally annotated videos** spanning **22 distinct Chinese dance genres** organized into **Chinese Folk Dance** (16 genres) and **Chinese Modern Dance** (6 genres), with per-frame SMPL-H parameters, synchronized music, and bilingual teacher evaluations.

## Dataset Overview

| Property | Value |
|----------|-------|
| Total Videos | 175 |
| Dance Genres | 22 (16 Folk + 6 Modern) |
| Total Duration | >15,000 seconds (4.1 hours) |
| SMPL-H Frames | 371,452 |
| Proficiency Levels | A (Excellent), B (Satisfactory), C (Developing) |
| Modalities | SMPL-H Pose, Music (MP3), Teacher Evaluations (CN/EN) |
| Annotation Team | 6 professional dance instructors |
| Inter-Rater Agreement | κ_w = 0.84, ICC = 0.91 |

## Repository Structure

```
CCDance/
├── README.md
├── requirements.txt
├── data/
│   ├── metadata.csv                    # Complete sample metadata
│   ├── music/                          # 22 MP3 files (one per genre)
│   ├── teacher_evaluations/            # 175 bilingual evaluations
│   └── smpl_archives/                  # SMPL-H JSON per dance (tar.gz)
├── experiments/
│   ├── baselines/
│   │   ├── shared/                     # Shared data loading & models
│   │   ├── usdl/                       # USDL (Tang et al. CVPR 2020)
│   │   ├── core/                       # CoRe (Yu et al. ICCV 2021)
│   │   ├── vl_transformer/             # VL-Transformer (Chen, SciRep 2025)
│   │   ├── levit_hybrid/               # LeViT-Hybrid (Wang, SciRep 2025)
│   │   ├── graph_transformer/          # Graph-Transformer (Han et al., SciRep 2026)
│   │   ├── dancemvp/                   # DanceMVP (SciRep 2025)
│   │   └── run_generation.py           # Unified generation baseline
│   ├── run_baselines.py                # Master experiment orchestration
│   └── results_summary/                # Aggregated benchmark results
└── paper/
    └── ccdance.pdf                     # Dataset paper (KDD 2027)
```

## Quick Start

### 1. Extract SMPL Data

```bash
cd data/smpl_archives/
for f in *.tar.gz; do tar xzf "$f" -C ../smpl/; done
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Data Loading

```python
import sys; sys.path.insert(0, 'experiments')
from baselines.shared.data_loader import load_all_pose_sequences

pose_seqs, metadata, valid_idx = load_all_pose_sequences(
    data_root='.', use_cache=False
)
print(f"Loaded {len(pose_seqs)} pose sequences")
```

### 4. Run Baseline Experiments

```bash
cd experiments

# Run classification for a single baseline
python baselines/usdl/classification/train.py --gpu 0 --epochs 50

# Run all classification experiments
python run_baselines.py --all --gpus 0,1,2,3

# Run generation experiments
python baselines/run_generation.py --baseline usdl --gpu 0 --epochs 30

# Aggregate results
python run_baselines.py --aggregate
```

## Benchmark Tasks

### Task 1: Dance Quality Grade Classification
Predict A/B/C proficiency grade from SMPL-H pose sequences.

**Metrics**: Accuracy, Macro-F1, Quadratic Weighted Kappa (QWK)

| Method | Accuracy | Macro-F1 | QWK |
|--------|----------|----------|-----|
| PoseLSTM | 0.343±0.075 | 0.324±0.065 | 0.097±0.133 |
| Pose Transformer | 0.332±0.009 | 0.166±0.003 | 0.000±0.000 |
| DanceMVP | 0.304±0.076 | 0.238±0.069 | 0.076±0.141 |
| USDL | 0.326±0.028 | 0.204±0.031 | 0.020±0.047 |
| CoRe | 0.326±0.015 | 0.191±0.036 | -0.000±0.000 |
| VL-Transformer | 0.341±0.054 | 0.254±0.072 | 0.047±0.085 |
| LeViT-Hybrid | 0.319±0.018 | 0.170±0.015 | -0.036±0.044 |
| Graph-Transformer | 0.333±0.041 | 0.247±0.034 | 0.032±0.056 |

### Task 2: Dance Quality Comment Generation
Generate professional pedagogical feedback from pose input.

**Metrics**: BLEU-1/4, ROUGE-L, Cosine Similarity (CosSim)

| Method | BLEU-1 | ROUGE-L | CosSim |
|--------|--------|---------|--------|
| DanceMVP | 0.052±0.008 | 0.045±0.006 | — |
| CoRe | 0.284±0.023 | 0.187±0.013 | 0.456±0.009 |
| USDL | 0.271±0.025 | 0.178±0.014 | 0.454±0.011 |
| VL-Transformer | 0.270±0.025 | 0.178±0.014 | 0.454±0.010 |
| LeViT-Hybrid | 0.287±0.031 | 0.190±0.023 | 0.452±0.012 |
| Graph-Transformer | 0.287±0.031 | 0.190±0.023 | 0.452±0.012 |
| *Human Agreement* | *0.47* | *0.52* | *—* |

## Data Format

### SMPL-H Parameters
Each frame is stored as a JSON file with SMPL-H body model parameters:
```json
[{
  "poses": [69-D joint rotations (axis-angle)],
  "shapes": [10-D PCA shape coefficients],
  "Rh": [3-D global root orientation],
  "Th": [3-D global translation]
}]
```

### Metadata (metadata.csv)
| Column | Description |
|--------|-------------|
| dance_id | 1-22, unique genre identifier |
| category | Folk or Modern |
| grade | A, B, or C proficiency level |
| student_name | Anonymous student identifier |
| n_smpl_frames | Number of valid SMPL-H frames |
| duration_seconds | Performance duration |
| music_file | Corresponding MP3 filename |
| teacher_comment_en | English teacher evaluation |
| teacher_comment_cn | Chinese teacher evaluation |

## Baseline Methods

| Baseline | Original Paper | Adaptation for CCDance |
|----------|---------------|----------------------|
| **USDL** | Tang et al. CVPR 2020 | I3D → ST-GCN for SMPL input |
| **CoRe** | Yu et al. ICCV 2021 | I3D → PoseLSTM + GART |
| **VL-Transformer** | Chen, SciRep 2025 | ST-GCN+LSTM + InfoNCE pre-training |
| **LeViT-Hybrid** | Wang, SciRep 2025 | Image → Pose patch embedding + KD |
| **Graph-Transformer** | Han et al., SciRep 2026 | IMU → SMPL joint graph + dual attention |
| **DanceMVP** | SciRep 2025 | ST-GCN+LSTM + InfoNCE + classification FT |

Each baseline directory contains:
```
baseline_name/
├── classification/
│   ├── model.py          # Model architecture
│   ├── train.py           # Training script
│   └── test.py            # Evaluation script
├── generation/
│   ├── model.py
│   ├── train.py
│   └── test.py
└── README.md              # Implementation notes
```

## Training Protocol (Unified)

| Parameter | Value |
|-----------|-------|
| Data Split | 70/15/15 train/val/test, stratified by grade |
| Random Seeds | 42, 123, 456, 789, 1024 |
| Sequence Length | 300 frames (uniform sampling/padding) |
| Batch Size | 16 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | Cosine Annealing |
| Epochs | 50 (classification), 30 (generation) |
| Early Stopping | Patience=10 on validation metric |

## Hardware Requirements

- **GPU**: NVIDIA GPU with ≥8GB VRAM (tested on L20 48GB)
- **RAM**: ≥32GB
- **Storage**: ≥2GB for extracted dataset
- **Software**: Python 3.10+, PyTorch 2.0+, CUDA 12+

## License

This dataset is released under the **CC BY-NC 4.0** license for academic research purposes only.
- ✅ Academic use, research, and education
- ❌ Commercial use without explicit permission
- ❌ Personnel scoring or educational punishment applications

```

## Contact

For dataset access requests, questions, or collaboration inquiries, please open a GitHub issue or contact the authors.

---
*CCDance was collected at Wuhan College of Communication during the 2025–2026 academic year with institutional ethics approval.*
