"""
CoRe Classification Evaluation with Multi-Exemplar Voting

Multi-exemplar voting (original CoRe paper): randomly sample M exemplars
from the training set and average their predictions with the test sample
for more robust inference.

Usage:
    python test.py [--gpu 0] [--checkpoint path/to/model.pt] [--seed 42]
"""
import sys
import os
import json
import argparse
import random
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           SEEDS, GRADE_MAP, N_CLASSES, MODEL_CONFIGS)
from shared.data_loader import (load_all_pose_sequences,
                                create_data_splits,
                                CCDancePoseDataset)
from shared.train_utils import eval_epoch_classification
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import CoReModel

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoRe Classification Evaluation'
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (model.pt)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for data split (must match training)')
    parser.add_argument('--n-exemplars', type=int,
                        default=MODEL_CONFIGS['core']['n_exemplars'],
                        help='Number of exemplars for multi-exemplar voting')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    return parser.parse_args()


def build_exemplar_set(train_loader, model, device, n_exemplars=10):
    """Build a multi-exemplar set from training samples.

    Randomly selects n_exemplars training samples and stores their
    backbone embeddings and labels.

    Args:
        train_loader: DataLoader for training set.
        model: CoReModel instance.
        device: torch device.
        n_exemplars: Number of exemplars to select.

    Returns:
        dict with 'embeddings' (n_exemplars, D) and 'labels' (n_exemplars,).
    """
    model.eval()

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            emb = model.encode_pose(x).cpu().numpy()
            all_embs.append(emb)
            all_labels.extend(y.numpy().tolist())

    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.array(all_labels)

    # Select exemplars: stratified by grade
    exemplar_indices = []
    for grade in range(N_CLASSES):
        grade_idx = np.where(all_labels == grade)[0]
        if len(grade_idx) > 0:
            n_per_grade = max(1, n_exemplars // N_CLASSES)
            selected = np.random.choice(
                grade_idx, size=min(n_per_grade, len(grade_idx)), replace=False
            )
            exemplar_indices.extend(selected.tolist())

    # If we still need more, sample randomly
    if len(exemplar_indices) < n_exemplars:
        remaining = list(set(range(len(all_embs))) - set(exemplar_indices))
        extra = np.random.choice(
            remaining, size=min(n_exemplars - len(exemplar_indices), len(remaining)),
            replace=False
        )
        exemplar_indices.extend(extra.tolist())

    exemplar_indices = exemplar_indices[:n_exemplars]

    return {
        'embeddings': torch.FloatTensor(all_embs[exemplar_indices]).to(device),
        'labels': torch.LongTensor(all_labels[exemplar_indices]).to(device),
    }


@torch.no_grad()
def evaluate_with_exemplars(model, test_loader, exemplar_set, device):
    """Evaluate with multi-exemplar voting.

    For each test sample, compute backbone embedding, then compare
    with exemplar embeddings. The final prediction averages over
    exemplar-conditioned leaf probabilities (weighted by embedding
    similarity to each exemplar).

    Args:
        model: CoReModel instance.
        test_loader: DataLoader for test set.
        exemplar_set: dict with 'embeddings' and 'labels'.
        device: torch device.

    Returns:
        dict with predictions, labels, probs, and metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    exemplar_embs = exemplar_set['embeddings']     # (M, D)
    exemplar_labels = exemplar_set['labels']       # (M,)

    for batch in test_loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)
        B = x.size(0)

        # Get test sample backbone embeddings
        test_embs = model.encode_pose(x)           # (B, D)

        # Compute similarity between test samples and exemplars
        sim = F.cosine_similarity(
            test_embs.unsqueeze(1),                 # (B, 1, D)
            exemplar_embs.unsqueeze(0),             # (1, M, D)
            dim=-1
        )                                           # (B, M)

        # Weight exemplar predictions by similarity
        # First, get exemplar model outputs
        exemplar_outputs = model.gart(exemplar_embs)   # M samples
        exemplar_logits = exemplar_outputs['logits']   # (M, C)

        # Each test sample's prediction = weighted average of exemplar predictions
        # Weight = softmax over similarities
        sim_weights = F.softmax(sim / 0.1, dim=-1)     # (B, M)
        weighted_logits = sim_weights @ exemplar_logits  # (B, C)

        probs = F.softmax(weighted_logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(weighted_logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds, probs)

    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probs': probs.tolist(),
        'metrics': metrics,
    }


@torch.no_grad()
def evaluate_direct(model, test_loader, device):
    """Direct evaluation without exemplar voting (standard forward pass).

    Args:
        model: CoReModel instance.
        test_loader: DataLoader for test set.
        device: torch device.

    Returns:
        dict with predictions, labels, probs, and metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in test_loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        output = model(x)
        logits = output['logits']

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

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Resolve checkpoint path
    if args.checkpoint is None:
        default_ckpt = (Path(__file__).parent.parent / 'results'
                        / f'seed_{args.seed}' / 'model.pt')
        if default_ckpt.exists():
            args.checkpoint = str(default_ckpt)
        else:
            fallback = Path(__file__).parent / 'model.pt'
            if fallback.exists():
                args.checkpoint = str(fallback)
            else:
                print("ERROR: No checkpoint found. Specify --checkpoint.")
                sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")

    # Load data
    print("\nLoading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)

    # Create data splits
    train_idx, _, test_idx = create_data_splits(
        valid_indices, metadata, seed=args.seed
    )
    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Create dataloaders
    from torch.utils.data import DataLoader
    train_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in train_idx])
    test_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in test_idx])

    train_dataset = CCDancePoseDataset(
        train_idx, pose_sequences, train_labels,
        [metadata[i]['grade'] for i in train_idx]
    )
    test_dataset = CCDancePoseDataset(
        test_idx, pose_sequences, test_labels,
        [metadata[i]['grade'] for i in test_idx]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Initialize model
    model = CoReModel(
        pose_dim=69,
        hidden_dim=256,
        tree_depth=MODEL_CONFIGS['core']['tree_depth'],
        num_classes=N_CLASSES,
        dropout=0.3,
    ).to(device)

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    # ---- Evaluation ----
    print(f"\n{'='*60}")
    print("1. Direct Evaluation (standard forward pass)")
    print(f"{'='*60}")
    direct_results = evaluate_direct(model, test_loader, device)
    m = direct_results['metrics']
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Macro F1:  {m['macro_f1']:.4f}")
    print(f"  QWK:       {m['qwk']:.4f}")
    print(f"  ECE:       {m['ece']:.4f}")

    # Build exemplar set from training data
    print(f"\n{'='*60}")
    print(f"2. Multi-Exemplar Voting (M={args.n_exemplars})")
    print(f"{'='*60}")

    # Run exemplar voting multiple times for robustness
    n_runs = 5
    exemplar_metrics = []
    for run in range(n_runs):
        exemplar_set = build_exemplar_set(
            train_loader, model, device,
            n_exemplars=args.n_exemplars
        )
        ex_results = evaluate_with_exemplars(
            model, test_loader, exemplar_set, device
        )
        exemplar_metrics.append(ex_results['metrics'])
        print(f"  Run {run+1}/{n_runs}: "
              f"acc={ex_results['metrics']['accuracy']:.4f}, "
              f"f1={ex_results['metrics']['macro_f1']:.4f}")

    # Average exemplar voting results
    avg_ex_metrics = {
        'accuracy': float(np.mean([m2['accuracy'] for m2 in exemplar_metrics])),
        'accuracy_std': float(np.std([m2['accuracy'] for m2 in exemplar_metrics])),
        'macro_f1': float(np.mean([m2['macro_f1'] for m2 in exemplar_metrics])),
        'macro_f1_std': float(np.std([m2['macro_f1'] for m2 in exemplar_metrics])),
        'qwk': float(np.mean([m2['qwk'] for m2 in exemplar_metrics])),
        'qwk_std': float(np.std([m2['qwk'] for m2 in exemplar_metrics])),
    }

    print(f"\n  Averaged exemplar voting: "
          f"acc={avg_ex_metrics['accuracy']:.4f}+/-{avg_ex_metrics['accuracy_std']:.4f}, "
          f"f1={avg_ex_metrics['macro_f1']:.4f}")

    # Save results
    all_results = {
        'seed': args.seed,
        'direct': direct_results,
        'exemplar_voting': {
            'n_exemplars': args.n_exemplars,
            'n_runs': n_runs,
            'per_run': exemplar_metrics,
            'averaged': avg_ex_metrics,
        },
    }

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).parent / 'eval_results'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'test_results_seed{args.seed}.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
