"""
Graph-Transformer / X-DANCENET Classification Training

Architecture (Han et al., SciRep 2026):
- GraphTransformerEncoder: dual spatial-temporal attention over SMPL joints
- MLPClassifier: 256 -> 128 -> 3 classification head
- Standard cross-entropy loss with AdamW optimizer
- Cosine annealing LR schedule + early stopping

Simplifications vs. original X-DANCENET:
- No sensor normalization pipeline (SMPL is pre-cleaned)
- No tempo-conditioned multi-scale feature extraction
- No prototype-based explainable decision layer
- Standard CE loss instead of prototype + confidence + saliency losses

Usage:
    python train.py [--gpu 0]
"""
import sys
import os
import json
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           N_EPOCHS_CLASSIFICATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, GRADE_MAP,
                           MODEL_CONFIGS)
from shared.data_loader import (load_all_pose_sequences,
                                create_data_splits,
                                create_dataloaders)
from shared.train_utils import (run_classification_experiment,
                                run_classification_with_seeds)
from shared.metrics import compute_classification_metrics_mean_std
from model import GraphTransformerModel

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Graph-Transformer / X-DANCENET Classification Training'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_CLASSIFICATION,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device configuration
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory for results
    results_dir = Path(args.output_dir) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    print(f"Loaded {len(valid_indices)} pose sequences")

    # Build dataloaders for each seed
    seeds = args.seed
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for seed in seeds:
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create data splits
        train_idx, val_idx, test_idx = create_data_splits(
            valid_indices, metadata, seed=seed
        )

        # Create dataloaders (pose-only, no audio)
        train_loader, val_loader, test_loader = create_dataloaders(
            pose_sequences, metadata, train_idx, val_idx, test_idx,
            batch_size=args.batch_size, audio_features=None
        )

        train_loaders[seed] = train_loader
        val_loaders[seed] = val_loader
        test_loaders[seed] = test_loader

        print(f"  Seed {seed}: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

    # Create model factory function
    def model_fn(**kwargs):
        return GraphTransformerModel(
            pose_dim=69,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            num_joints=23,
            num_classes=3,
            dropout=args.dropout,
        )

    # Run classification across all seeds
    print(f"\n{'='*60}")
    print(f"Graph-Transformer Classification Training")
    print(f"d_model={args.d_model}, nhead={args.nhead}, "
          f"layers={args.num_layers}")
    print(f"lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}")
    print(f"{'='*60}")

    aggregated = run_classification_with_seeds(
        model_fn=model_fn,
        model_args={},
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        device=device,
        model_name='GraphTransformer',
        output_dir=results_dir,
        seeds=seeds,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_audio=False,
    )

    # Print aggregated results
    print(f"\n{'='*60}")
    print("Aggregated Results (across all seeds)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {aggregated['accuracy']:.3f} +/- "
          f"{aggregated['accuracy_std']:.3f}")
    print(f"  Macro F1:  {aggregated['macro_f1']:.3f} +/- "
          f"{aggregated['macro_f1_std']:.3f}")
    print(f"  QWK:       {aggregated['qwk']:.3f} +/- "
          f"{aggregated['qwk_std']:.3f}")

    return aggregated


if __name__ == '__main__':
    main()
