"""
Training utilities: training loop, logging, checkpointing for CCDance baselines.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from .config import N_CLASSES
from .metrics import (compute_classification_metrics,
                      compute_classification_metrics_mean_std,
                      compute_generation_metrics)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch_classification(model, dataloader, optimizer, criterion, device,
                               scheduler=None):
    """Train one epoch for classification."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        # Handle different dataset formats (pose-only or pose+audio)
        if len(batch) == 3:
            x, y, lengths = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        elif len(batch) == 4:
            if isinstance(batch[2], torch.Tensor) and batch[2].dim() == 1:
                x, audio, y, lengths = batch
                x, y = x.to(device), y.to(device)
                if isinstance(audio, torch.Tensor):
                    audio = audio.to(device)
                logits = model(x, audio)
            else:
                x, y, lengths = batch[:3]
                x, y = x.to(device), y.to(device)
                logits = model(x)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.detach().cpu().numpy().tolist())

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'predictions': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def eval_epoch_classification(model, dataloader, criterion, device):
    """Evaluate one epoch for classification."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        if len(batch) == 3:
            x, y, lengths = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        elif len(batch) == 4:
            if isinstance(batch[2], torch.Tensor) and batch[2].dim() == 1:
                x, audio, y, lengths = batch
                x, y = x.to(device), y.to(device)
                if isinstance(audio, torch.Tensor):
                    audio = audio.to(device)
                logits = model(x, audio)
            else:
                x, y, lengths = batch[:3]
                x, y = x.to(device), y.to(device)
                logits = model(x)

        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
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


def run_classification_experiment(model, train_loader, val_loader, test_loader,
                                  device, model_name, seed, output_dir,
                                  n_epochs=50, lr=1e-3, weight_decay=1e-4,
                                  use_audio=False):
    """Run a complete classification experiment with training, validation, and testing.
    Returns final test metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    early_stop = EarlyStopping(patience=10, mode='max')

    best_val_acc = 0.0
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_classification(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_res = eval_epoch_classification(model, val_loader, criterion, device)

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_acc': train_res['accuracy'],
            'val_loss': val_res['loss'],
            'val_acc': val_res['accuracy'],
            'val_macro_f1': val_res['macro_f1'],
            'val_qwk': val_res['qwk'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f}, "
                  f"train_acc={train_res['accuracy']:.3f}, "
                  f"val_acc={val_res['accuracy']:.3f}, "
                  f"val_f1={val_res['macro_f1']:.3f}")

        if val_res['accuracy'] > best_val_acc:
            best_val_acc = val_res['accuracy']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stop(val_res['accuracy']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_classification(model, test_loader, criterion, device)

    # Save results
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

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save training log
    with open(output_dir / 'train_log.json', 'w') as f:
        json.dump(train_log, f, indent=2)

    return results


def run_classification_with_seeds(model_fn, model_args, train_loaders, val_loaders,
                                  test_loaders, device, model_name, output_dir,
                                  seeds, n_epochs=50, lr=1e-3, weight_decay=1e-4,
                                  use_audio=False):
    """Run classification experiment across multiple seeds.
    Returns aggregated results.
    """
    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running {model_name} with seed {seed}")
        print(f"{'='*60}")

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = model_fn(**model_args)

        seed_output_dir = Path(output_dir) / f'seed_{seed}'
        res = run_classification_experiment(
            model, train_loaders[seed], val_loaders[seed],
            test_loaders[seed], device, model_name, seed, seed_output_dir,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
            use_audio=use_audio
        )
        all_seed_results.append(res['test_metrics'])

    # Aggregate
    aggregated = compute_classification_metrics_mean_std(all_seed_results)

    # Save aggregated results
    with open(Path(output_dir) / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def setup_logging(output_dir, model_name):
    """Setup logging for an experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f'{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    return log_file
