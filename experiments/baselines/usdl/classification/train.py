#!/usr/bin/env python3
"""
USDL Classification Training Script.
Tests two approaches:
  1. Distribution learning: KL divergence between predicted distribution
     and ground-truth Gaussian distribution (core USDL method).
  2. Direct classification: Cross-entropy loss on 3 classes (3-layer MLP).

Usage:
    python train.py --gpu 0
    python train.py --gpu 0 --approach distribution  # only distribution learning
    python train.py --gpu 0 --approach classification # only direct classification
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
    N_EPOCHS_CLASSIFICATION,
    SEEDS,
    MODEL_CONFIGS,
)
from shared.data_loader import (
    load_all_pose_sequences,
    create_data_splits,
    create_dataloaders,
)
from shared.train_utils import (
    run_classification_with_seeds,
    train_epoch_classification,
    eval_epoch_classification,
    EarlyStopping,
)
from shared.metrics import (
    compute_classification_metrics,
    compute_classification_metrics_mean_std,
)
from model import USDLModel

# ---------------------------------------------------------------------------
# Gaussian soft label generation (ground-truth distribution for KL div)
# ---------------------------------------------------------------------------


def gaussian_soft_labels(labels, n_bins=10, sigma=0.5, device="cpu"):
    """Generate Gaussian-distributed soft labels.

    For each grade label g in {0, 1, 2}, create a discretized Gaussian
    distribution centered at g with standard deviation sigma over 10 bins
    uniformly spaced in [0, 2].

    Args:
        labels: (B,) integer labels in {0, 1, 2}
        n_bins: number of distribution bins
        sigma: standard deviation of the Gaussian
        device: torch device

    Returns:
        soft_labels: (B, n_bins) normalized Gaussian distributions
    """
    bin_centers = torch.linspace(0.0, 2.0, n_bins, device=device)  # (n_bins,)
    labels_float = labels.float().unsqueeze(1)  # (B, 1)
    diff = bin_centers.unsqueeze(0) - labels_float  # (B, n_bins)
    unnormalized = torch.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    soft_labels = unnormalized / (unnormalized.sum(dim=1, keepdim=True) + 1e-8)
    return soft_labels


def distribution_to_grade(dist, n_bins=10):
    """Convert predicted score distribution to grade label.

    Mapping: bins 0-3 -> grade 0 (A), bins 4-6 -> grade 1 (B),
             bins 7-9 -> grade 2 (C).

    Args:
        dist: (B, n_bins) predicted distribution (softmax)
        n_bins: number of bins

    Returns:
        grades: (B,) integer grades {0, 1, 2}
    """
    bin_idx = dist.argmax(dim=1)  # (B,)
    grade_map = torch.zeros(n_bins, dtype=torch.long, device=dist.device)
    grade_map[:4] = 0   # bins 0, 1, 2, 3  -> A
    grade_map[4:7] = 1  # bins 4, 5, 6     -> B
    grade_map[7:] = 2   # bins 7, 8, 9     -> C
    return grade_map[bin_idx]


# ---------------------------------------------------------------------------
# Distribution learning: training and evaluation loops
# ---------------------------------------------------------------------------


def train_epoch_distribution(model, dataloader, optimizer, device,
                             sigma=0.5, n_bins=10, scheduler=None):
    """Train one epoch for distribution learning (KL divergence)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        x, y, lengths = batch
        x, y = x.to(device), y.to(device)

        dist, _, _ = model(x)  # (B, n_bins)

        # Gaussian soft labels as ground-truth distribution
        soft_labels = gaussian_soft_labels(
            y, n_bins=n_bins, sigma=sigma, device=device
        )

        # KL divergence: KL(soft_labels || dist)
        kl_loss = F.kl_div(
            torch.log(dist + 1e-8),
            soft_labels,
            reduction="batchmean",
        )

        optimizer.zero_grad()
        kl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += kl_loss.item() * x.size(0)

        pred_grades = distribution_to_grade(dist, n_bins=n_bins)
        all_preds.extend(pred_grades.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "predictions": all_preds,
        "labels": all_labels,
    }


@torch.no_grad()
def eval_epoch_distribution(model, dataloader, device,
                            sigma=0.5, n_bins=10):
    """Evaluate one epoch for distribution learning."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        x, y, lengths = batch
        x, y = x.to(device), y.to(device)

        dist, _, _ = model(x)

        soft_labels = gaussian_soft_labels(
            y, n_bins=n_bins, sigma=sigma, device=device
        )

        kl_loss = F.kl_div(
            torch.log(dist + 1e-8),
            soft_labels,
            reduction="batchmean",
        )
        total_loss += kl_loss.item() * x.size(0)

        pred_grades = distribution_to_grade(dist, n_bins=n_bins)
        all_preds.extend(pred_grades.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.append(dist.cpu().numpy())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds, probs)

    return {
        "loss": avg_loss,
        **metrics,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Full experiment runner for distribution learning
# ---------------------------------------------------------------------------


def run_distribution_experiment(model, train_loader, val_loader, test_loader,
                                device, model_name, seed, output_dir,
                                n_epochs=50, lr=1e-4, weight_decay=1e-4,
                                sigma=0.5, n_bins=10):
    """Run a complete distribution-learning experiment (multi-seed wrapper)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    early_stop = EarlyStopping(patience=10, mode="max")

    best_val_acc = 0.0
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_distribution(
            model, train_loader, optimizer, device,
            sigma=sigma, n_bins=n_bins, scheduler=scheduler,
        )
        val_res = eval_epoch_distribution(
            model, val_loader, device, sigma=sigma, n_bins=n_bins,
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_res["loss"],
            "train_acc": train_res["accuracy"],
            "val_loss": val_res["loss"],
            "val_acc": val_res["accuracy"],
            "val_macro_f1": val_res["macro_f1"],
            "val_qwk": val_res["qwk"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={train_res['loss']:.4f}, "
                f"train_acc={train_res['accuracy']:.3f}, "
                f"val_acc={val_res['accuracy']:.3f}, "
                f"val_f1={val_res['macro_f1']:.3f}"
            )

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if early_stop(val_res["accuracy"]):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_distribution(
        model, test_loader, device, sigma=sigma, n_bins=n_bins,
    )

    results = {
        "model": model_name,
        "seed": seed,
        "approach": "distribution_learning",
        "n_epochs": len(train_log),
        "best_val_acc": best_val_acc,
        "test_metrics": {
            "accuracy": test_res["accuracy"],
            "macro_f1": test_res["macro_f1"],
            "qwk": test_res["qwk"],
            "ece": test_res.get("ece", 0.0),
        },
        "train_log": train_log,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    return results


def run_distribution_with_seeds(model_fn, model_args, train_loaders, val_loaders,
                                test_loaders, device, model_name, output_dir,
                                seeds, n_epochs=50, lr=1e-4, weight_decay=1e-4,
                                sigma=0.5, n_bins=10):
    """Run distribution learning across multiple seeds and aggregate results."""
    all_seed_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Running {model_name} [distribution] with seed {seed}")
        print(f"{'=' * 60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = model_fn(**model_args)

        seed_output_dir = Path(output_dir) / "distribution" / f"seed_{seed}"
        res = run_distribution_experiment(
            model,
            train_loaders[seed], val_loaders[seed], test_loaders[seed],
            device, model_name, seed, seed_output_dir,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            sigma=sigma, n_bins=n_bins,
        )
        all_seed_results.append(res["test_metrics"])

    aggregated = compute_classification_metrics_mean_std(all_seed_results)
    agg_dir = Path(output_dir) / "distribution"
    agg_dir.mkdir(parents=True, exist_ok=True)
    with open(agg_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{model_name} [distribution] aggregated results:")
    print(f"  Accuracy: {aggregated['accuracy']:.4f} +/- {aggregated['accuracy_std']:.4f}")
    print(f"  Macro F1: {aggregated['macro_f1']:.4f} +/- {aggregated['macro_f1_std']:.4f}")
    print(f"  QWK:      {aggregated['qwk']:.4f} +/- {aggregated['qwk_std']:.4f}")
    print(f"{'=' * 60}")

    return aggregated


# ---------------------------------------------------------------------------
# Direct classification wrapper (reuses shared train_utils infrastructure)
# ---------------------------------------------------------------------------


def run_direct_classification(model_fn, model_args, train_loaders, val_loaders,
                              test_loaders, device, model_name, output_dir,
                              seeds, n_epochs=50, lr=1e-4, weight_decay=1e-4):
    """Run direct classification (CE loss) using the MLPClassifier head."""
    use_audio = model_args.get("use_audio", False)

    # The shared run_classification_with_seeds expects models that return
    # logits for CrossEntropy.  We wrap USDLModel to output only the
    # classifier logits.
    def _classifier_only(**kwargs):
        base_model = model_fn(**kwargs)
        base_model.use_classifier = True

        class Wrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, x):
                _, logits_cls, _ = self.base(x)
                return logits_cls

        return Wrapper(base_model)

    # Build fresh loaders keyed by seed
    cls_loaders = {"train": {}, "val": {}, "test": {}}

    aggregated = run_classification_with_seeds(
        _classifier_only, model_args,
        train_loaders, val_loaders, test_loaders,
        device, model_name, Path(output_dir) / "classification",
        seeds=seeds, n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        use_audio=use_audio,
    )

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="USDL Classification Training"
    )
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--approach", type=str, default="both",
                        choices=["distribution", "classification", "both"],
                        help="Which approach to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS_CLASSIFICATION)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (USDL uses lower LR)")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Gaussian sigma for soft labels")
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Number of score distribution bins")
    parser.add_argument("--output-dir", type=str,
                        default=str(
                            Path(__file__).parent.parent / "results"
                        ),
                        help="Output directory")
    args = parser.parse_args()

    # Device
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data (once, reused across seeds and approaches)
    # ------------------------------------------------------------------
    print("Loading pose sequences...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(
        data_root=DATA_ROOT
    )
    print(f"  Loaded {len(valid_indices)} valid samples")
    print(f"  Pose dimension: {pose_sequences[valid_indices[0]].shape}")

    # Create per-seed data splits and loaders
    train_loaders, val_loaders, test_loaders = {}, {}, {}

    for seed in SEEDS:
        train_idx, val_idx, test_idx = create_data_splits(
            valid_indices, metadata, seed=seed
        )
        tr_loader, val_loader, te_loader = create_dataloaders(
            pose_sequences, metadata,
            train_idx, val_idx, test_idx,
            batch_size=args.batch_size,
        )
        train_loaders[seed] = tr_loader
        val_loaders[seed] = val_loader
        test_loaders[seed] = te_loader

    print(f"  Split sizes (seed={SEEDS[0]}): "
          f"train={len(train_loaders[SEEDS[0]].dataset)}, "
          f"val={len(val_loaders[SEEDS[0]].dataset)}, "
          f"test={len(test_loaders[SEEDS[0]].dataset)}")

    # ------------------------------------------------------------------
    # Model arguments
    # ------------------------------------------------------------------
    model_name = "USDL"
    model_fn = USDLModel
    model_args = {
        "hidden_dim": 256,
        "n_bins": args.n_bins,
        "dropout": 0.3,
        "use_classifier": True,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Approach 1: Distribution learning
    # ------------------------------------------------------------------
    if args.approach in ("distribution", "both"):
        print("\n" + "=" * 60)
        print("Approach 1: Distribution Learning (KL divergence)")
        print("=" * 60)

        run_distribution_with_seeds(
            model_fn, model_args,
            train_loaders, val_loaders, test_loaders,
            device, model_name, output_dir,
            seeds=SEEDS, n_epochs=args.epochs, lr=args.lr,
            sigma=args.sigma, n_bins=args.n_bins,
        )

    # ------------------------------------------------------------------
    # Approach 2: Direct classification
    # ------------------------------------------------------------------
    if args.approach in ("classification", "both"):
        print("\n" + "=" * 60)
        print("Approach 2: Direct Classification (CE loss)")
        print("=" * 60)

        run_direct_classification(
            model_fn, model_args,
            train_loaders, val_loaders, test_loaders,
            device, model_name, output_dir,
            seeds=SEEDS, n_epochs=args.epochs, lr=args.lr,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
