"""
CoRe Classification Training

Trains the Group-Aware Contrastive Regression (CoRe) model with:
  - Cross-entropy loss (aggregated + leaf-weighted)
  - Contrastive grouping loss (same-grade pull, diff-grade push)
  - Multi-seed evaluation (5 seeds)

Usage:
    python train.py [--gpu 0] [--epochs 50]
"""
import sys
import os
import json
import argparse
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           N_EPOCHS_CLASSIFICATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, GRADE_MAP, N_CLASSES,
                           MODEL_CONFIGS)
from shared.data_loader import (load_all_pose_sequences,
                                create_data_splits,
                                create_dataloaders)
from shared.train_utils import (train_epoch_classification,
                                eval_epoch_classification,
                                EarlyStopping)
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import CoReModel, CoReLoss

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoRe Classification Training'
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_CLASSIFICATION,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr-backbone', type=float,
                        default=MODEL_CONFIGS['core']['lr_backbone'],
                        help='Backbone learning rate')
    parser.add_argument('--lr-tree', type=float,
                        default=MODEL_CONFIGS['core']['lr_tree'],
                        help='GART learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay (CoRe uses 0 weight decay)')
    parser.add_argument('--tree-depth', type=int,
                        default=MODEL_CONFIGS['core']['tree_depth'],
                        help='GART tree depth')
    parser.add_argument('--n-exemplars', type=int,
                        default=MODEL_CONFIGS['core']['n_exemplars'],
                        help='Number of exemplars for inference voting')
    parser.add_argument('--ce-weight', type=float, default=1.0,
                        help='Weight for aggregated CE loss')
    parser.add_argument('--leaf-ce-weight', type=float, default=1.0,
                        help='Weight for leaf-weighted CE loss')
    parser.add_argument('--contrastive-weight', type=float, default=0.1,
                        help='Weight for contrastive grouping loss')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def train_epoch_core(model, dataloader, optimizer, criterion, device,
                     scheduler=None):
    """Train one epoch using the CoRe combined loss.

    Args:
        model: CoReModel instance.
        dataloader: DataLoader yielding (pose, label, T).
        optimizer: torch optimizer.
        criterion: CoReLoss instance.
        device: torch device.
        scheduler: optional LR scheduler.

    Returns:
        dict with loss, accuracy, predictions, labels.
    """
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_leaf_ce = 0.0
    total_contrast = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x, y = x.to(device), y.to(device)

        output = model(x)
        loss_dict = criterion(output, y)
        loss = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_dict['loss_ce'].item() * bs
        total_leaf_ce += loss_dict['loss_leaf_ce'].item() * bs
        total_contrast += loss_dict['loss_contrast'].item() * bs

        preds = output['logits'].argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        'loss': avg_loss,
        'loss_ce': total_ce / n,
        'loss_leaf_ce': total_leaf_ce / n,
        'loss_contrast': total_contrast / n,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def eval_epoch_core(model, dataloader, criterion, device):
    """Evaluate one epoch using the CoRe combined loss.

    Args:
        model: CoReModel instance.
        dataloader: DataLoader.
        criterion: CoReLoss instance.
        device: torch device.

    Returns:
        dict with loss, accuracy, macro_f1, qwk, ece, etc.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss_dict = criterion(output, y)

        total_loss += loss_dict['loss'].item() * x.size(0)

        probs = F.softmax(output['logits'], dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(output['logits'].argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds, probs)

    return {
        'loss': avg_loss,
        **metrics,
        'predictions': all_preds,
        'labels': all_labels,
    }


def run_core_experiment(model, train_loader, val_loader, test_loader,
                        device, model_name, seed, output_dir,
                        n_epochs=50, lr_backbone=1e-4, lr_tree=1e-3,
                        weight_decay=0.0, ce_weight=1.0, leaf_ce_weight=1.0,
                        contrastive_weight=0.1):
    """Run a complete CoRe training experiment.

    Uses separate learning rates for backbone and GART (as in the CoRe paper).
    """
    model = model.to(device)

    # Separate parameter groups (CoRe paper: different lr for backbone vs tree)
    backbone_params = list(model.backbone.parameters())
    tree_params = list(model.gart.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': tree_params, 'lr': lr_tree, 'weight_decay': weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = CoReLoss(
        ce_weight=ce_weight,
        leaf_ce_weight=leaf_ce_weight,
        contrastive_weight=contrastive_weight,
    )
    early_stop = EarlyStopping(patience=10, mode='max')

    best_val_acc = 0.0
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_core(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_res = eval_epoch_core(model, val_loader, criterion, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_loss_ce': train_res['loss_ce'],
            'train_loss_leaf_ce': train_res['loss_leaf_ce'],
            'train_loss_contrast': train_res['loss_contrast'],
            'train_acc': train_res['accuracy'],
            'val_loss': val_res['loss'],
            'val_acc': val_res['accuracy'],
            'val_macro_f1': val_res['macro_f1'],
            'val_qwk': val_res['qwk'],
            'val_ece': val_res.get('ece', 0.0),
            'lr_backbone': optimizer.param_groups[0]['lr'],
            'lr_tree': optimizer.param_groups[1]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f} "
                  f"(ce={train_res['loss_ce']:.3f} "
                  f"leaf={train_res['loss_leaf_ce']:.3f} "
                  f"con={train_res['loss_contrast']:.3f}) "
                  f"train_acc={train_res['accuracy']:.3f} "
                  f"val_acc={val_res['accuracy']:.3f} "
                  f"val_f1={val_res['macro_f1']:.3f}")

        if val_res['accuracy'] > best_val_acc:
            best_val_acc = val_res['accuracy']
            best_model_state = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}

        if early_stop(val_res['accuracy']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_core(model, test_loader, criterion, device)

    results = {
        'model': model_name,
        'seed': seed,
        'n_epochs': len(train_log),
        'best_val_acc': best_val_acc,
        'test_metrics': {
            'accuracy': test_res['accuracy'],
            'macro_f1': test_res['macro_f1'],
            'qwk': test_res['qwk'],
            'ece': test_res.get('ece', 0.0),
        },
        'train_log': train_log,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model and results
    torch.save(model.state_dict(), output_dir / 'model.pt')

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'train_log.json', 'w') as f:
        json.dump(train_log, f, indent=2)

    return results


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_dir = Path(args.output_dir)
    log_dir = base_dir / 'logs'
    ckpt_dir = base_dir / 'checkpoints'
    results_dir = base_dir / 'results'
    for d in [log_dir, ckpt_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    print(f"Loaded {len(valid_indices)} pose sequences")

    seeds = args.seed
    all_results = []

    for seed in seeds:
        seed_dir = results_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create data splits
        train_idx, val_idx, test_idx = create_data_splits(
            valid_indices, metadata, seed=seed
        )
        print(f"\nSeed {seed}: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            pose_sequences, metadata, train_idx, val_idx, test_idx,
            batch_size=args.batch_size,
        )

        # Initialize model
        model = CoReModel(
            pose_dim=69,
            hidden_dim=256,
            tree_depth=args.tree_depth,
            num_classes=N_CLASSES,
            dropout=args.dropout,
        ).to(device)

        print(f"\n{'='*60}")
        print(f"CoRe Training (seed={seed})")
        print(f"  Tree depth: {args.tree_depth}, "
              f"Leaves: {2**args.tree_depth}")
        print(f"  lr_backbone={args.lr_backbone}, lr_tree={args.lr_tree}")
        print(f"  contrastive_weight={args.contrastive_weight}")
        print(f"{'='*60}")

        results = run_core_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            model_name='CoRe',
            seed=seed,
            output_dir=seed_dir,
            n_epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_tree=args.lr_tree,
            weight_decay=args.weight_decay,
            ce_weight=args.ce_weight,
            leaf_ce_weight=args.leaf_ce_weight,
            contrastive_weight=args.contrastive_weight,
        )

        all_results.append(results['test_metrics'])

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregated Results (across all seeds)")
    print(f"{'='*60}")

    aggregated = compute_classification_metrics_mean_std(all_results)
    print(f"  Accuracy:  {aggregated['accuracy']:.3f} +/- {aggregated['accuracy_std']:.3f}")
    print(f"  Macro F1:  {aggregated['macro_f1']:.3f} +/- {aggregated['macro_f1_std']:.3f}")
    print(f"  QWK:       {aggregated['qwk']:.3f} +/- {aggregated['qwk_std']:.3f}")

    aggregated_path = results_dir / 'aggregated_results.json'
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {aggregated_path}")

    return aggregated


if __name__ == '__main__':
    main()
