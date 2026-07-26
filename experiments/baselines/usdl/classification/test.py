#!/usr/bin/env python3
"""
USDL Classification Test Script.
Evaluates a trained model on the test set and reports metrics.

Usage:
    python test.py --gpu 0 --checkpoint ../results/distribution/seed_42/model.pt
    python test.py --gpu 0 --checkpoint ../results/classification/seed_42/model.pt --mode classification
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import (
    DATA_ROOT,
    SEQUENCE_LENGTH,
    N_CLASSES,
    GRADE_MAP,
    REV_GRADE_MAP,
    BATCH_SIZE,
    SEEDS,
    MODEL_CONFIGS,
)
from shared.data_loader import (
    load_all_pose_sequences,
    create_data_splits,
    create_dataloaders,
)
from shared.train_utils import eval_epoch_classification
from shared.metrics import compute_classification_metrics
from model import USDLModel


def gaussian_soft_labels(labels, n_bins=10, sigma=0.5, device="cpu"):
    """Generate Gaussian-distributed soft labels."""
    bin_centers = torch.linspace(0.0, 2.0, n_bins, device=device)
    labels_float = labels.float().unsqueeze(1)
    diff = bin_centers.unsqueeze(0) - labels_float
    unnormalized = torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    soft_labels = unnormalized / (unnormalized.sum(dim=1, keepdim=True) + 1e-8)
    return soft_labels


def distribution_to_grade(dist, n_bins=10):
    """Convert predicted distribution to grade label."""
    bin_idx = dist.argmax(dim=1)
    grade_map = torch.zeros(n_bins, dtype=torch.long, device=dist.device)
    grade_map[:4] = 0
    grade_map[4:7] = 1
    grade_map[7:] = 2
    return grade_map[bin_idx]


@torch.no_grad()
def test_distribution(model, dataloader, device, n_bins=10, sigma=0.5):
    """Evaluate model in distribution mode."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_kl = 0.0

    for batch in dataloader:
        x, y, lengths = batch
        x, y = x.to(device), y.to(device)

        dist, _, _ = model(x)

        soft_labels = gaussian_soft_labels(
            y, n_bins=n_bins, sigma=sigma, device=device
        )
        kl_loss = F.kl_div(
            torch.log(dist + 1e-8), soft_labels, reduction="batchmean"
        )
        total_kl += kl_loss.item() * x.size(0)

        pred_grades = distribution_to_grade(dist, n_bins=n_bins)
        all_preds.extend(pred_grades.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.append(dist.cpu().numpy())

    n = len(dataloader.dataset)
    avg_kl = total_kl / n
    probs = np.concatenate(all_probs, axis=0)

    return {
        "kl_loss": avg_kl,
        **compute_classification_metrics(all_labels, all_preds, probs),
        "predictions": all_preds,
        "labels": all_labels,
    }


@torch.no_grad()
def test_classification(model, dataloader, device):
    """Evaluate model in direct classification mode."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        x, y, lengths = batch
        x, y = x.to(device), y.to(device)

        _, logits_cls, _ = model(x)
        loss = criterion(logits_cls, y)
        total_loss += loss.item() * x.size(0)

        probs = F.softmax(logits_cls, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits_cls.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    probs = np.concatenate(all_probs, axis=0)

    return {
        "ce_loss": avg_loss,
        **compute_classification_metrics(all_labels, all_preds, probs),
        "predictions": all_preds,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="USDL Classification Test"
    )
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--mode", type=str, default="distribution",
                        choices=["distribution", "classification"],
                        help="Inference mode of the checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Gaussian sigma (distribution mode)")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for data split")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load data
    print("Loading pose sequences...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(
        data_root=DATA_ROOT
    )
    _, _, test_idx = create_data_splits(
        valid_indices, metadata, seed=args.seed
    )
    _, _, test_loader = create_dataloaders(
        pose_sequences, metadata,
        [], [], test_idx,
        batch_size=args.batch_size,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    model = USDLModel(
        hidden_dim=256,
        n_bins=args.n_bins,
        dropout=0.3,
        use_classifier=True,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Run evaluation
    if args.mode == "distribution":
        results = test_distribution(
            model, test_loader, device,
            n_bins=args.n_bins, sigma=args.sigma,
        )
        print(f"\nDistribution mode results:")
    else:
        results = test_classification(model, test_loader, device)
        print(f"\nClassification mode results:")

    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Macro F1:  {results['macro_f1']:.4f}")
    print(f"  QWK:       {results['qwk']:.4f}")
    print(f"  ECE:       {results.get('ece', 0.0):.4f}")
    if "kl_loss" in results:
        print(f"  KL Loss:   {results['kl_loss']:.4f}")
    if "ce_loss" in results:
        print(f"  CE Loss:   {results['ce_loss']:.4f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Confusion matrix
    cm = results.get("confusion_matrix", [])
    if cm:
        print("\nConfusion Matrix:")
        print(f"          Pred A  Pred B  Pred C")
        for i, row in enumerate(cm):
            print(f"  True {REV_GRADE_MAP[i]}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")


if __name__ == "__main__":
    main()
