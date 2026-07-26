"""
VL-Transformer Classification Evaluation

Loads a trained model checkpoint and evaluates on test set.

Usage:
    python test.py [--gpu 0] [--checkpoint path/to/model.pt] [--seed 42]
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

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           SEEDS, GRADE_MAP, N_CLASSES)
from shared.data_loader import (load_all_pose_sequences,
                                extract_audio_features,
                                build_audio_features_per_sample,
                                create_data_splits,
                                create_dataloaders)
from shared.train_utils import eval_epoch_classification
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import VLTransformerClassifier

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='VL-Transformer Classification Evaluation'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (model.pt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split (must match training)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Run full evaluation on test set.

    Args:
        model: VLTransformerClassifier
        test_loader: DataLoader
        device: torch device

    Returns:
        dict with predictions, labels, probs, and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in test_loader:
        pose, audio, y, _ = batch
        pose = pose.to(device)
        y = y.to(device)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(device)

        logits = model(pose, audio)

        probs = torch.softmax(logits, dim=-1)
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
        # Default: look for latest trained model
        default_ckpt = Path(__file__).parent.parent / 'results' / f'seed_{args.seed}' / 'model.pt'
        if default_ckpt.exists():
            args.checkpoint = str(default_ckpt)
        else:
            # Fallback to classification/model.pt
            fallback = Path(__file__).parent / 'model.pt'
            if fallback.exists():
                args.checkpoint = str(fallback)
            else:
                print("ERROR: No checkpoint found. Specify --checkpoint.")
                sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")

    # Load data
    print("Loading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    audio_features = extract_audio_features(DATA_ROOT)
    per_sample_audio = build_audio_features_per_sample(
        valid_indices, metadata, audio_features
    )

    # Create data split for the specified seed
    _, _, test_idx = create_data_splits(valid_indices, metadata, seed=args.seed)
    print(f"Test samples: {len(test_idx)}")

    # Create test loader
    _, _, test_loader = create_dataloaders(
        pose_sequences, metadata, [], [], test_idx,
        batch_size=args.batch_size, audio_features=per_sample_audio
    )

    # Initialize model
    model = VLTransformerClassifier(
        pose_dim=69,
        audio_dim=64,
        hidden_dim=256,
        num_classes=N_CLASSES,
        dropout=0.3,
        temperature=0.07,
        num_joints=23,
    ).to(device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    # Evaluate
    results = evaluate(model, test_loader, device)

    # Print results
    metrics = results['metrics']
    print(f"\n{'='*60}")
    print(f"Test Results (seed={args.seed})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"  QWK:       {metrics['qwk']:.4f}")
    print(f"  ECE:       {metrics['ece']:.4f}")

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
