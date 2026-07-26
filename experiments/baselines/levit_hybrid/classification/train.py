"""
LeViT-Hybrid Classification Training

Training protocol (adapted from Wang, SciRep 2025):
  1. Teacher pretraining: train the larger teacher Transformer (8 layers, 512 dim)
     on the classification task for a warmup phase.
  2. Joint distillation training: train student (4 layers, 256 dim) with
     CE loss + KL distillation loss from the (frozen) teacher.
  3. Student fine-tuning: optionally fine-tune student without distillation.

Key differences from original LeViT paper:
  - Original: 224x224 RGB images, CNN backbone, LeViT-128 + ViT-B/16 teacher
  - Adapted: SMPL pose sequences (300 frames of 69-D), pose patch embedding,
    Transformer-based student/teacher, MLP classifier

Usage:
    python train.py [--gpu 0] [--lr 1e-3] [--epochs 50]
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
from torch.utils.data import DataLoader

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           N_EPOCHS_CLASSIFICATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, GRADE_MAP,
                           MODEL_CONFIGS)
from shared.data_loader import (load_all_pose_sequences,
                                create_data_splits,
                                CCDancePoseDataset,
                                create_dataloaders)
from shared.train_utils import (train_epoch_classification,
                                eval_epoch_classification,
                                run_classification_experiment,
                                EarlyStopping)
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import LeViTHybridModel

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='LeViT-Hybrid Classification Training'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_CLASSIFICATION,
                        help='Number of epochs for distillation training')
    parser.add_argument('--teacher-epochs', type=int, default=20,
                        help='Number of epochs for teacher pretraining')
    parser.add_argument('--finetune-epochs', type=int, default=10,
                        help='Number of epochs for student fine-tuning without distillation')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for student (adapted pose version)')
    parser.add_argument('--teacher-lr', type=float, default=5e-4,
                        help='Learning rate for teacher pretraining')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--patch-size', type=int, default=15,
                        help='Frames per patch (default: 15 for 300/15=20 patches)')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Student embedding dimension')
    parser.add_argument('--teacher-embed-dim', type=int, default=512,
                        help='Teacher embedding dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--student-layers', type=int, default=4,
                        help='Number of student transformer layers')
    parser.add_argument('--teacher-layers', type=int, default=8,
                        help='Number of teacher transformer layers')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Distillation loss weight (0.5 = equal weight)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--skip-teacher-pretrain', action='store_true',
                        help='Skip teacher pretraining phase')
    parser.add_argument('--skip-distillation', action='store_true',
                        help='Skip distillation training (train student directly)')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def train_epoch_distillation(model, dataloader, optimizer, device,
                             temperature=4.0, alpha=0.5, scheduler=None):
    """Train one epoch with knowledge distillation.

    Uses both CE loss with ground-truth labels and KL divergence loss
    between teacher and student softened outputs.

    Args:
        model: LeViTHybridModel (teacher must be in eval mode)
        dataloader: DataLoader yielding (pose, label, T)
        optimizer: torch optimizer
        device: torch device
        temperature: distillation temperature
        alpha: distillation loss weight
        scheduler: optional LR scheduler

    Returns:
        dict with 'loss', 'distill_loss', 'ce_loss', 'accuracy', etc.
    """
    model.train()
    # Teacher should be in eval mode for stable soft targets
    model.teacher_encoder.eval()
    model.teacher_classifier.eval()

    total_loss = 0.0
    total_distill_loss = 0.0
    total_ce_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        # batch: (pose, label, T)
        if len(batch) == 3:
            x, y, lengths = batch
        else:
            x, y = batch[0], batch[1]
            lengths = batch[-1] if len(batch) > 2 else None

        x, y = x.to(device), y.to(device)

        # Forward with teacher
        student_logits, teacher_logits = model(x, return_teacher=True)

        # Compute distillation loss
        loss, distill_loss, ce_loss = model.compute_distillation_loss(
            student_logits, teacher_logits.detach(), labels=y
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_distill_loss += distill_loss.item() * x.size(0)
        total_ce_loss += ce_loss.item() * x.size(0) if isinstance(ce_loss, torch.Tensor) else 0.0

        preds = student_logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    avg_distill = total_distill_loss / n
    avg_ce = total_ce_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        'loss': avg_loss,
        'distill_loss': avg_distill,
        'ce_loss': avg_ce,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def eval_epoch_distillation(model, dataloader, device):
    """Evaluate student model on validation/test set.

    Args:
        model: LeViTHybridModel
        dataloader: DataLoader
        device: torch device

    Returns:
        dict with metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if len(batch) == 3:
            x, y, lengths = batch
        else:
            x, y = batch[0], batch[1]
            lengths = batch[-1] if len(batch) > 2 else None

        x, y = x.to(device), y.to(device)

        # Student-only forward during evaluation
        logits = model(x, return_teacher=False)

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds, probs)

    return {
        **metrics,
        'predictions': all_preds,
        'labels': all_labels,
    }


def train_teacher(model, train_loader, val_loader, device, n_epochs, lr,
                  weight_decay, output_dir, seed):
    """Pretrain teacher Transformer on classification task.

    The teacher is a larger Transformer (8 layers, 512 dim) trained with
    standard cross-entropy loss to serve as a soft label generator for
    student distillation.

    Args:
        model: LeViTHybridModel
        train_loader, val_loader: DataLoaders
        device: torch device
        n_epochs: number of pretraining epochs
        lr: learning rate
        weight_decay: weight decay
        output_dir: directory to save checkpoints
        seed: random seed

    Returns:
        best_val_acc: best validation accuracy achieved
    """
    print(f"\n{'='*60}")
    print(f"Teacher Pretraining (seed={seed})")
    print(f"{'='*60}")

    # Only optimize teacher parameters + patch embedding
    teacher_params = (
        list(model.patch_embed.parameters()) +
        list(model.teacher_proj.parameters()) +
        list(model.teacher_encoder.parameters()) +
        list(model.teacher_classifier.parameters())
    )
    optimizer = torch.optim.AdamW(teacher_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=10, mode='max')

    best_val_acc = 0.0
    best_teacher_state = None
    train_log = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch[0], batch[1]

            x, y = x.to(device), y.to(device)

            # Teacher forward: project patches, encode, classify
            patches = model.encode_patches(x)
            teacher_patches = model.teacher_proj(patches)
            teacher_emb = model.teacher_encoder(teacher_patches)
            logits = model.teacher_classifier(teacher_emb)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())

        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        val_res = eval_epoch_distillation(model, val_loader, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_res['accuracy'],
            'val_f1': val_res['macro_f1'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                  f"val_acc={val_res['accuracy']:.3f}")

        if val_res['accuracy'] > best_val_acc:
            best_val_acc = val_res['accuracy']
            best_teacher_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if early_stop(val_res['accuracy']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best teacher weights
    if best_teacher_state is not None:
        model.load_state_dict(best_teacher_state)

    # Save teacher checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_teacher_state, output_dir / 'teacher_pretrained.pt')

    with open(output_dir / 'teacher_pretrain_log.json', 'w') as f:
        json.dump(train_log, f, indent=2)

    print(f"  Teacher pretraining complete. Best val acc: {best_val_acc:.3f}")
    return best_val_acc


def train_distillation(model, train_loader, val_loader, test_loader, device,
                       n_epochs, lr, weight_decay, temperature, alpha,
                       output_dir, model_name, seed):
    """Train student with knowledge distillation from teacher.

    The teacher is frozen during this phase. The student learns from both
    the ground-truth labels (CE loss) and the teacher's soft outputs
    (KL divergence).

    Args:
        model: LeViTHybridModel (with pretrained teacher)
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        n_epochs: number of distillation epochs
        lr: learning rate
        weight_decay: weight decay
        temperature: distillation temperature
        alpha: distillation loss weight
        output_dir: directory to save results
        model_name: name for logging
        seed: random seed

    Returns:
        dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Distillation Training (seed={seed}, alpha={alpha}, T={temperature})")
    print(f"{'='*60}")

    # Freeze teacher
    model.freeze_teacher()

    # Optimize student + patch embedding parameters
    student_params = (
        list(model.patch_embed.parameters()) +
        list(model.student_encoder.parameters()) +
        list(model.student_classifier.parameters())
    )
    optimizer = torch.optim.AdamW(
        student_params, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    early_stop = EarlyStopping(patience=10, mode='max')

    best_val_acc = 0.0
    best_student_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_distillation(
            model, train_loader, optimizer, device,
            temperature=temperature, alpha=alpha, scheduler=scheduler
        )
        val_res = eval_epoch_distillation(model, val_loader, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_distill_loss': train_res['distill_loss'],
            'train_ce_loss': train_res['ce_loss'],
            'train_acc': train_res['accuracy'],
            'val_acc': val_res['accuracy'],
            'val_macro_f1': val_res['macro_f1'],
            'val_qwk': val_res['qwk'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f} "
                  f"(CE={train_res['ce_loss']:.4f}, "
                  f"KD={train_res['distill_loss']:.4f}), "
                  f"train_acc={train_res['accuracy']:.3f}, "
                  f"val_acc={val_res['accuracy']:.3f}")

        if val_res['accuracy'] > best_val_acc:
            best_val_acc = val_res['accuracy']
            best_student_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if early_stop(val_res['accuracy']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best student checkpoint
    if best_student_state is not None:
        model.load_state_dict(best_student_state)

    # Evaluate on test set
    test_res = eval_epoch_distillation(model, test_loader, device)

    # Save results
    results = {
        'model': model_name,
        'seed': seed,
        'phase': 'distillation',
        'temperature': temperature,
        'alpha': alpha,
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

    # Save student model
    torch.save(model.state_dict(), output_dir / 'model_distilled.pt')
    with open(output_dir / 'results_distillation.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(output_dir / 'train_log_distillation.json', 'w') as f:
        json.dump(train_log, f, indent=2)

    return results


def finetune_student(model, train_loader, val_loader, test_loader, device,
                     n_epochs, lr, weight_decay, output_dir, seed):
    """Fine-tune student without distillation (after distillation phase).

    This optional phase allows the student to adapt beyond the teacher's
    soft targets, potentially improving performance.

    Args:
        model: LeViTHybridModel
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        n_epochs: number of fine-tuning epochs
        lr: learning rate
        weight_decay: weight decay
        output_dir: directory to save results
        seed: random seed

    Returns:
        dict with test results
    """
    print(f"\n{'='*60}")
    print(f"Student Fine-tuning (seed={seed}, no distillation)")
    print(f"{'='*60}")

    # Unfreeze student (teacher remains frozen)
    # Student params are already trainable; ensure teacher is frozen
    model.freeze_teacher()

    # Use shared train_utils for standard classification training
    results = run_classification_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model_name='LeViTHybrid',
        seed=seed,
        output_dir=output_dir,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    return results


def main():
    args = parse_args()

    # Device configuration
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directories
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

    for seed_idx, seed in enumerate(seeds):
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

        # Create dataloaders (pose-only, no audio)
        train_loader, val_loader, test_loader = create_dataloaders(
            pose_sequences, metadata, train_idx, val_idx, test_idx,
            batch_size=args.batch_size
        )

        # Initialize model
        model = LeViTHybridModel(
            pose_dim=69,
            seq_length=SEQUENCE_LENGTH,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            teacher_embed_dim=args.teacher_embed_dim,
            nhead=args.nhead,
            student_layers=args.student_layers,
            teacher_layers=args.teacher_layers,
            dropout=args.dropout,
            num_classes=3,
            temperature=args.temperature,
            alpha=args.alpha,
        ).to(device)

        # Phase 1: Teacher pretraining
        if not args.skip_teacher_pretrain:
            train_teacher(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                n_epochs=args.teacher_epochs,
                lr=args.teacher_lr,
                weight_decay=args.weight_decay,
                output_dir=seed_dir,
                seed=seed,
            )

        # Phase 2: Distillation training
        if not args.skip_distillation:
            distill_results = train_distillation(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                n_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                alpha=args.alpha,
                output_dir=seed_dir,
                model_name='LeViTHybrid',
                seed=seed,
            )
            final_results = distill_results
        else:
            # Direct student training (no distillation)
            print(f"\n{'='*60}")
            print(f"Direct Student Training (seed={seed}, no distillation)")
            print(f"{'='*60}")
            model.freeze_teacher()
            direct_results = run_classification_experiment(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                model_name='LeViTHybrid',
                seed=seed,
                output_dir=seed_dir / 'direct',
                n_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            final_results = direct_results

        # Phase 3 (optional): Student fine-tuning without distillation
        if args.finetune_epochs > 0 and not args.skip_distillation:
            finetune_results = finetune_student(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                n_epochs=args.finetune_epochs,
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=args.weight_decay,
                output_dir=seed_dir / 'finetune',
                seed=seed,
            )
            final_results = finetune_results  # Report finetuned results

        all_results.append(final_results['test_metrics'])

    # Aggregate results across seeds
    print(f"\n{'='*60}")
    print("Aggregated Results (across all seeds)")
    print(f"{'='*60}")

    aggregated = compute_classification_metrics_mean_std(all_results)
    print(f"  Accuracy:  {aggregated['accuracy']:.3f} +/- {aggregated['accuracy_std']:.3f}")
    print(f"  Macro F1:  {aggregated['macro_f1']:.3f} +/- {aggregated['macro_f1_std']:.3f}")
    print(f"  QWK:       {aggregated['qwk']:.3f} +/- {aggregated['qwk_std']:.3f}")

    # Save aggregated results
    aggregated_path = results_dir / 'aggregated_results.json'
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {aggregated_path}")

    return aggregated


if __name__ == '__main__':
    main()
