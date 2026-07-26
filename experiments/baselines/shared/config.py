"""
Shared configuration for all CCDance baseline experiments.
All baselines MUST use these settings for fair comparison.
"""
import os
from pathlib import Path

# Paths
DATA_ROOT = "/home/doudou/software/emc_results"
BASELINE_ROOT = Path("/home/doudou/software/emc_results/experiments/baselines")
RESULTS_SUMMARY = Path("/home/doudou/software/emc_results/experiments/results_summary")

# Dataset
N_DANCE_GENRES = 22
N_VIDEOS = 175
GRADE_MAP = {'A': 0, 'B': 1, 'C': 2}
REV_GRADE_MAP = {0: 'A', 1: 'B', 2: 'C'}
N_CLASSES = 3

# SMPL
SMPL_POSE_DIM = 69
SMPL_NUM_JOINTS = 23
SMPL_JOINTS_3D = 25  # 25 keypoints (COCO-WholeBody)

# Audio features
AUDIO_SR = 22050
AUDIO_HOP_LENGTH = 512
AUDIO_N_MFCC = 20
AUDIO_N_CHROMA = 12
AUDIO_FEATURE_DIM = 54  # 20 MFCC + 20 MFCC-delta + 12 chroma + 1 onset + 1 smooth

# Training (UNIFIED across all baselines)
SEQUENCE_LENGTH = 300  # uniform sampling/padding to 300 frames
BATCH_SIZE = 16
N_EPOCHS_CLASSIFICATION = 50
N_EPOCHS_GENERATION = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_SCHEDULE = "cosine"
EARLY_STOP_PATIENCE = 10
SEEDS = [42, 123, 456, 789, 1024]

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# GPU
N_GPUS = 4
GPU_MEMORY_GB = 48

# Generation
MAX_COMMENT_LENGTH = 512
TEACHER_EMBED_DIM = 768  # Sentence-BERT dimension

# Model-specific overrides (where papers specify different values)
MODEL_CONFIGS = {
    "usdl": {
        "lr": 1e-4,
        "hidden_dims": [256, 128],
        "n_bins": 10,  # score distribution bins
    },
    "core": {
        "lr_backbone": 1e-4,
        "lr_tree": 1e-3,
        "tree_depth": 5,
        "n_groups": 32,
        "node_dim": 256,
        "n_exemplars": 10,
    },
    "vl_transformer": {
        "lr": 1e-4,
        "pretrain_epochs": 100,
        "temperature": 0.07,
        "dropout": 0.3,
    },
    "levit_hybrid": {
        "lr": 1e-5,
        "image_size": 224,
        "teacher_model": "vit_base_patch16_224",
    },
    "graph_transformer": {
        "lr": 1e-3,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
    },
}
