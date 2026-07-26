"""
LeViT-Hybrid Classification Evaluation

Loads a trained LeViTHybridModel checkpoint and evaluates on the test set.
Supports evaluation of both distilled and non-distilled models.

Usage:
    python test.py [--gpu 0] [--checkpoint path/to/model.pt] [--seed 42]
    python test.py [--gpu 0] --checkpoint path/to/model_distilled.pt --distilled
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
import torch.nn as nn
import torch.nn.functional as F

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           SEEDS, GRADE_MAP, N_CLASSES)
from shared.data_loader import (load_all_pose_sequences,
                                create_data_splits,
                                create_dataloaders)
from shared.train_utils import eval_epoch_classification
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import LeViTHybridModel

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='LeViT-Hybrid Classification Evaluation'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split (must match training)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--distilled', action='store_true',
                        help='Flag if checkpoint is from distillation training')
    parser.add_argument('--student-only', action='store_true',
                        help='Use student-only forward (no teacher)')
    parser.add_argument('--patch-size', type=int, default=15,
                        help='Frames per patch (must match training)')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Student embedding dimension (must match training)')
    parser.add_argument('--teacher-embed-dim', type=int, default=512,
                        help='Teacher embedding dimension (must match training)')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device, student_only=True):
    """Run full evaluation on test set.

    Args:
        model: LeViTHybridModel
        test_loader: DataLoader
        device: torch device
        student_only: if True, use student-only forward pass

    Returns:
        dict with predictions, labels, probs, and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in test_loader:
        if len(batch) >= 3:
            x, y = batch[0], batch[1]
        else:
            x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        if student_only:
            logits = model(x, return_teacher=False)
        else:
            # Use teacher for evaluation
            _, logits = model(x, return_teacher=True)

        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds, probs)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probs': probs.tolist(),
        'metrics': metrics,
    }


def main():
    args = parse_args()

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve checkpoint path
    if args.checkpoint is None:
        # Default: look for distilled model, then fallback to direct model
        default_ckpt = (
            Path(__file__).parent.parent / 'results' / f'seed_{args.seed}' / 'model_distilled.pt'
        )
        if default_ckpt.exists():
            args.checkpoint = str(default_ckpt)
            print(f"Found distilled checkpoint: {args.checkpoint}")
        else:
            fallback = (
                Path(__file__).parent.parent / 'results' / f'seed_{args.seed}' / 'model.pt'
            )
            if fallback.exists():
                args.checkpoint = str(fallback)
                print(f"Found direct checkpoint: {args.checkpoint}")
            else:
                # Try classification directory
                local_ckpt = Path(__file__).parent / 'model.pt'
                if local_ckpt.exists():
                    args.checkpoint = str(local_ckpt)
                else:
                    print("ERROR: No checkpoint found. Specify --checkpoint.")
                    sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")

    # Load data
    print("Loading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)

    # Create data split for the specified seed
    _, _, test_idx = create_data_splits(valid_indices, metadata, seed=args.seed)
    print(f"Test samples: {len(test_idx)}")

    # Create test loader
    _, _, test_loader = create_dataloaders(
        pose_sequences, metadata, [], [], test_idx,
        batch_size=args.batch_size
    )

    # Initialize model
    model = LeViTHybridModel(
        pose_dim=69,
        seq_length=SEQUENCE_LENGTH,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        teacher_embed_dim=args.teacher_embed_dim,
        nhead=8,
        student_layers=4,
        teacher_layers=8,
        dropout=0.1,
        num_classes=N_CLASSES,
        temperature=4.0,
        alpha=0.5,
    ).to(device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    # Determine student_only mode
    # If checkpoint is from direct training (no teacher keys), use student_only
    student_only = not args.distilled
    if student_only:
        print("Using student-only evaluation mode.")

    # Evaluate
    results = evaluate(model, test_loader, device, student_only=student_only)

    # Print results
    metrics = results['metrics']
    print(f"\n{'='*60}")
    print(f"Test Results (seed={args.seed})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  QWK:       {metrics['qwk']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f}")
    print(f"  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    for i, row in enumerate(cm):
        grade = {0: 'A', 1: 'B', 2: 'C'}[i]
        print(f"    {grade}: {row}")

    # Save results
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent / 'eval_results'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'test_results_seed{args.seed}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
