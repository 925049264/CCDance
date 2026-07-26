"""
VL-Transformer Classification Training

Two-stage protocol:
1. InfoNCE contrastive pretraining (100 epochs, pose-music alignment)
2. Classification fine-tuning (50 epochs, fused embedding -> grade)

Usage:
    python train.py [--gpu 0]
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
                           WEIGHT_DECAY, SEEDS, GRADE_MAP,
                           MODEL_CONFIGS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
from shared.data_loader import (load_all_pose_sequences,
                                extract_audio_features,
                                build_audio_features_per_sample,
                                create_data_splits,
                                create_dataloaders)
from shared.train_utils import (train_epoch_classification,
                                eval_epoch_classification,
                                run_classification_experiment,
                                EarlyStopping)
from shared.metrics import (compute_classification_metrics,
                            compute_classification_metrics_mean_std)
from model import VLTransformerClassifier, InfoNCELoss

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='VL-Transformer Classification Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_CLASSIFICATION,
                        help='Number of classification epochs')
    parser.add_argument('--pretrain-epochs', type=int, default=100,
                        help='Number of contrastive pretraining epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='InfoNCE temperature')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Skip InfoNCE pretraining phase')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def train_epoch_contrastive(model, dataloader, optimizer, device, scheduler=None):
    """Train one epoch of InfoNCE contrastive learning.

    Args:
        model: VLTransformerClassifier
        dataloader: DataLoader yielding (pose, audio, label, T)
        optimizer: torch optimizer
        device: torch device
        scheduler: optional LR scheduler

    Returns:
        dict with 'loss' and 'acc'
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        # batch: (pose, audio, label, T)
        pose, audio, _, _ = batch
        pose = pose.to(device)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(device)

        # Encode both modalities
        pose_emb = model.encode_pose(pose)
        music_emb = model.encode_music(audio)

        # Compute contrastive loss
        loss, acc = model.compute_contrastive_loss(pose_emb, music_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'acc': total_acc / n_batches,
    }


def run_pretraining(model, train_loader, val_loader, device, n_epochs, lr,
                    weight_decay, output_dir, seed):
    """Run InfoNCE contrastive pretraining phase.

    Args:
        model: VLTransformerClassifier
        train_loader, val_loader: DataLoaders with pose+audio
        device: torch device
        n_epochs: number of pretraining epochs
        lr: learning rate
        weight_decay: weight decay
        output_dir: directory to save checkpoints
        seed: random seed for logging

    Returns:
        dict with pretraining log
    """
    print(f"\n{'='*60}")
    print(f"InfoNCE Contrastive Pretraining (seed={seed})")
    print(f"{'='*60}")

    # Only optimize encoder parameters during pretraining
    encoder_params = list(model.pose_encoder.parameters()) + \
                     list(model.music_encoder.parameters()) + \
                     list(model.fusion_proj.parameters())

    optimizer = torch.optim.AdamW(encoder_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )

    pretrain_log = []
    best_val_loss = float('inf')
    best_encoder_state = None

    for epoch in range(n_epochs):
        train_res = train_epoch_contrastive(
            model, train_loader, optimizer, device, scheduler
        )

        # Validation: compute contrastive loss
        val_res = evaluate_contrastive(model, val_loader, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_acc': train_res['acc'],
            'val_loss': val_res['loss'],
            'val_acc': val_res['acc'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        pretrain_log.append(log_entry)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f}, "
                  f"train_acc={train_res['acc']:.3f}, "
                  f"val_loss={val_res['loss']:.4f}, "
                  f"val_acc={val_res['acc']:.3f}")

        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_encoder_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    # Restore best encoder weights
    if best_encoder_state is not None:
        model.load_state_dict(best_encoder_state)

    # Save pretraining log
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'pretrain_log.json', 'w') as f:
        json.dump(pretrain_log, f, indent=2)

    return pretrain_log


@torch.no_grad()
def evaluate_contrastive(model, dataloader, device):
    """Evaluate InfoNCE contrastive loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        pose, audio, _, _ = batch
        pose = pose.to(device)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(device)

        pose_emb = model.encode_pose(pose)
        music_emb = model.encode_music(audio)
        loss, acc = model.compute_contrastive_loss(pose_emb, music_emb)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'acc': total_acc / n_batches,
    }


def run_classification_finetuning(model, train_loader, val_loader, test_loader,
                                   device, n_epochs, lr, weight_decay,
                                   output_dir, model_name, seed):
    """Fine-tune the classifier on top of frozen/unfrozen encoders.

    Uses run_classification_experiment from shared.train_utils.
    """
    print(f"\n{'='*60}")
    print(f"Classification Fine-tuning (seed={seed})")
    print(f"{'='*60}")

    # Fine-tune all parameters
    results = run_classification_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        model_name=model_name,
        seed=seed,
        output_dir=output_dir,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        use_audio=True,
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

    # Extract audio features
    print("Extracting audio features...")
    audio_features = extract_audio_features(DATA_ROOT)
    per_sample_audio = build_audio_features_per_sample(
        valid_indices, metadata, audio_features
    )
    print(f"Audio features available for {len(per_sample_audio)} samples")

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

        # Create dataloaders with audio features
        train_loader, val_loader, test_loader = create_dataloaders(
            pose_sequences, metadata, train_idx, val_idx, test_idx,
            batch_size=args.batch_size, audio_features=per_sample_audio
        )

        # Initialize model
        model = VLTransformerClassifier(
            pose_dim=69,
            audio_dim=54,  # AUDIO_FEATURE_DIM from shared.config
            hidden_dim=256,
            num_classes=3,
            dropout=args.dropout,
            temperature=args.temperature,
            num_joints=23,
        ).to(device)

        # Phase 1: InfoNCE Contrastive Pretraining
        if not args.no_pretrain:
            run_pretraining(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                n_epochs=args.pretrain_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                output_dir=seed_dir,
                seed=seed,
            )

        # Phase 2: Classification Fine-tuning
        results = run_classification_finetuning(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=seed_dir,
            model_name='VLTransformer',
            seed=seed,
        )

        all_results.append(results['test_metrics'])

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
