"""
Graph-Transformer / X-DANCENET Generation Training

Architecture (Han et al., SciRep 2026 - adapted for comment generation):
- GraphTransformerEncoder: dual spatial-temporal attention over SMPL joints
  -> (B, d_model=256)
- LSTMDecoder: 2-layer LSTM with teacher forcing
  - Encoder output initializes LSTM hidden state
  - Token embedding -> LSTM -> projection -> vocab logits
- Cross-entropy loss on token prediction

Simplifications:
- No sensor normalization pipeline
- No prototype-based explanation layer
- Standard LSTM decoder (simpler than full Transformer decoder)

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shared.config import (DATA_ROOT, SEQUENCE_LENGTH, BATCH_SIZE,
                           N_EPOCHS_GENERATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, MODEL_CONFIGS,
                           MAX_COMMENT_LENGTH)
from shared.data_loader import (load_all_pose_sequences,
                                load_teacher_comments,
                                create_data_splits,
                                CCDanceGenerationDataset)
from shared.metrics import compute_generation_metrics
from shared.train_utils import EarlyStopping
from model import GraphTransformerGenerator, WordTokenizer, PAD_TOKEN

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Graph-Transformer / X-DANCENET Generation Training'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_GENERATION,
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
    parser.add_argument('--decoder-hidden', type=int, default=512,
                        help='Decoder LSTM hidden dimension')
    parser.add_argument('--decoder-layers', type=int, default=2,
                        help='Number of decoder LSTM layers')
    parser.add_argument('--teacher-forcing', type=float, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max-vocab', type=int, default=5000,
                        help='Maximum vocabulary size')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def build_tokenizer_from_splits(pose_sequences, metadata, valid_indices,
                                train_idx, seed, max_vocab=5000):
    """Build a word-level tokenizer from training set comments."""
    comments = load_teacher_comments(DATA_ROOT)
    train_comments = [comments[i] for i in train_idx if comments[i]]
    tokenizer = WordTokenizer(max_vocab_size=max_vocab)
    tokenizer.fit(train_comments)
    return tokenizer, comments


def create_gen_dataloaders(pose_sequences, metadata, valid_indices,
                           train_idx, val_idx, test_idx, tokenizer,
                           comments, batch_size=16):
    """Create DataLoaders for generation task.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Training data
    train_labels = [metadata[i]['grade'] for i in train_idx]
    train_dataset = CCDanceGenerationDataset(
        train_idx, pose_sequences, comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH,
    )

    # Validation data
    val_labels = [metadata[i]['grade'] for i in val_idx]
    val_dataset = CCDanceGenerationDataset(
        val_idx, pose_sequences, comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH,
    )

    # Test data
    test_labels = [metadata[i]['grade'] for i in test_idx]
    test_dataset = CCDanceGenerationDataset(
        test_idx, pose_sequences, comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch_gen(model, dataloader, optimizer, criterion, device,
                    teacher_forcing_ratio, scheduler=None):
    """Train one generation epoch with teacher forcing.

    Args:
        model: GraphTransformerGenerator
        dataloader: DataLoader yielding (pose, tokens, T, comment_text)
        optimizer: torch optimizer
        criterion: CrossEntropyLoss (ignore_index=PAD_TOKEN)
        device: torch device
        teacher_forcing_ratio: probability of using teacher forcing
        scheduler: optional LR scheduler

    Returns:
        dict with average loss and perplexity
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        # batch: (pose, tokens, T, comment_text)
        pose, tokens, _, _ = batch
        pose = pose.to(device)
        tokens = tokens.to(device)  # (B, max_len)

        # Forward pass with teacher forcing
        logits = model(
            pose,
            target_tokens=tokens,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )  # (B, seq_len, vocab_size)

        # Compute loss (shift: predict next token)
        # logits: (B, seq_len, V), targets: (B, seq_len)
        # Predict token at position t from logits at position t
        # Loss computed over all positions except the last
        loss = criterion(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        n_tokens = (tokens[:, 1:] != PAD_TOKEN).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
    }


@torch.no_grad()
def eval_epoch_gen(model, dataloader, criterion, device,
                   tokenizer, max_len):
    """Evaluate generation model on validation/test set.

    Computes both token prediction loss and generation metrics
    (BLEU, ROUGE-L, BERTScore) from decoded text.

    Args:
        model: GraphTransformerGenerator
        dataloader: DataLoader
        criterion: CrossEntropyLoss
        device: torch device
        tokenizer: WordTokenizer for decoding
        max_len: maximum generation length

    Returns:
        dict with loss, perplexity, and generation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_references = []
    all_candidates = []

    for batch in dataloader:
        pose, tokens, _, comment_text = batch
        pose = pose.to(device)
        tokens = tokens.to(device)

        # Teacher-forced loss (same as training for metric logging)
        logits = model(
            pose,
            target_tokens=tokens,
            teacher_forcing_ratio=1.0,  # full teacher forcing for loss
        )
        loss = criterion(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1)
        )
        n_tokens = (tokens[:, 1:] != PAD_TOKEN).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Greedy decoding for generation metrics
        gen_ids = model.generate(pose, device)  # (B, max_len)
        for i in range(len(comment_text)):
            ref = comment_text[i] if comment_text[i] else ""
            cand = tokenizer.decode(gen_ids[i].cpu())
            all_references.append(ref)
            all_candidates.append(cand)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    # Compute generation metrics
    gen_metrics = compute_generation_metrics(all_references, all_candidates)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        **gen_metrics,
        'references': all_references,
        'candidates': all_candidates,
    }


def run_generation_experiment(model, train_loader, val_loader, test_loader,
                              device, model_name, seed, tokenizer, output_dir,
                              n_epochs=100, lr=1e-3, weight_decay=1e-4,
                              teacher_forcing_ratio=0.5, max_len=512):
    """Run a complete generation experiment.

    Returns final test metrics.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    early_stop = EarlyStopping(patience=10, mode='min')

    best_val_loss = float('inf')
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_gen(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio, scheduler
        )
        val_res = eval_epoch_gen(
            model, val_loader, criterion, device, tokenizer, max_len
        )

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_perplexity': train_res['perplexity'],
            'val_loss': val_res['loss'],
            'val_perplexity': val_res['perplexity'],
            'val_bleu1': val_res['bleu1'],
            'val_bleu2': val_res['bleu2'],
            'val_rouge_l': val_res['rouge_l'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f}, "
                  f"train_ppl={train_res['perplexity']:.2f}, "
                  f"val_loss={val_res['loss']:.4f}, "
                  f"val_bleu1={val_res['bleu1']:.4f}")

        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if early_stop(-val_res['loss']):  # mode='min', negate for EarlyStopping
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_gen(
        model, test_loader, criterion, device, tokenizer, max_len
    )

    # Print generation examples
    print(f"\n  Generation examples (seed={seed}):")
    for i in range(min(3, len(test_res['references']))):
        print(f"    Ref:  {test_res['references'][i][:80]}")
        print(f"    Gen:  {test_res['candidates'][i][:80]}")
        print()

    # Save results
    results = {
        'model': model_name,
        'seed': seed,
        'n_epochs': len(train_log),
        'best_val_loss': best_val_loss,
        'test_metrics': {
            'loss': test_res['loss'],
            'perplexity': test_res['perplexity'],
            'bleu1': test_res['bleu1'],
            'bleu4': test_res['bleu4'],
            'rouge_l': test_res['rouge_l'],
            'bertscore': test_res['bertscore'],
        },
        'generation_examples': [
            {'reference': test_res['references'][i],
             'candidate': test_res['candidates'][i]}
            for i in range(min(5, len(test_res['references'])))
        ],
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


def run_generation_with_seeds(model_fn, model_args, pose_sequences, metadata,
                               valid_indices, device, model_name, output_dir,
                               seeds, n_epochs=100, lr=1e-3,
                               weight_decay=1e-4, teacher_forcing_ratio=0.5,
                               max_vocab=5000, batch_size=16, max_len=512):
    """Run generation experiment across multiple seeds.

    Returns aggregated results.
    """
    all_seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running {model_name} Generation with seed {seed}")
        print(f"{'='*60}")

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create data splits
        train_idx, val_idx, test_idx = create_data_splits(
            valid_indices, metadata, seed=seed
        )
        print(f"  Split: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")

        # Build tokenizer from training set
        comments = load_teacher_comments(DATA_ROOT)
        train_comments = [comments[i] for i in train_idx if comments[i]]
        tokenizer = WordTokenizer(max_vocab_size=max_vocab)
        tokenizer.fit(train_comments)
        print(f"  Vocabulary size: {tokenizer.vocab_size}")

        # Create dataloaders
        train_loader, val_loader, test_loader = create_gen_dataloaders(
            pose_sequences, metadata, valid_indices,
            train_idx, val_idx, test_idx, tokenizer, comments,
            batch_size=batch_size,
        )

        # Initialize model
        model = model_fn(vocab_size=tokenizer.vocab_size, **model_args)

        # Run experiment
        seed_output_dir = Path(output_dir) / f'seed_{seed}'
        res = run_generation_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            seed=seed,
            tokenizer=tokenizer,
            output_dir=seed_output_dir,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_len=max_len,
        )
        all_seed_results.append(res['test_metrics'])

    # Aggregate
    bleu1s = [r['bleu1'] for r in all_seed_results]
    bleu4s = [r['bleu4'] for r in all_seed_results]
    rouge_ls = [r['rouge_l'] for r in all_seed_results]
    bertscores = [r['bertscore'] for r in all_seed_results]

    aggregated = {
        'bleu1': float(np.mean(bleu1s)),
        'bleu1_std': float(np.std(bleu1s)),
        'bleu4': float(np.mean(bleu4s)),
        'bleu4_std': float(np.std(bleu4s)),
        'rouge_l': float(np.mean(rouge_ls)),
        'rouge_l_std': float(np.std(rouge_ls)),
        'bertscore': float(np.mean(bertscores)),
        'bertscore_std': float(np.std(bertscores)),
        'per_seed': all_seed_results,
    }

    # Save aggregated results
    with open(Path(output_dir) / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def main():
    args = parse_args()

    # Device configuration
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    results_dir = Path(args.output_dir) / 'generation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    print(f"Loaded {len(valid_indices)} pose sequences")

    # Create model factory
    def model_fn(vocab_size, **kwargs):
        return GraphTransformerGenerator(
            vocab_size=vocab_size,
            pose_dim=69,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            num_joints=23,
            decoder_hidden_dim=args.decoder_hidden,
            decoder_layers=args.decoder_layers,
            dropout=args.dropout,
            max_len=MAX_COMMENT_LENGTH,
        )

    model_args = {}  # model_fn handles all args except vocab_size

    # Run generation across seeds
    print(f"\n{'='*60}")
    print(f"Graph-Transformer Generation Training")
    print(f"d_model={args.d_model}, nhead={args.nhead}, "
          f"layers={args.num_layers}")
    print(f"lr={args.lr}, batch={args.batch_size}, "
          f"epochs={args.epochs}")
    print(f"{'='*60}")

    aggregated = run_generation_with_seeds(
        model_fn=model_fn,
        model_args=model_args,
        pose_sequences=pose_sequences,
        metadata=metadata,
        valid_indices=valid_indices,
        device=device,
        model_name='GraphTransformer-Gen',
        output_dir=results_dir,
        seeds=args.seed,
        n_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        teacher_forcing_ratio=args.teacher_forcing,
        max_vocab=args.max_vocab,
        batch_size=args.batch_size,
        max_len=MAX_COMMENT_LENGTH,
    )

    # Print aggregated results
    print(f"\n{'='*60}")
    print("Aggregated Generation Results (across all seeds)")
    print(f"{'='*60}")
    print(f"  BLEU-1:    {aggregated['bleu1']:.4f} +/- "
          f"{aggregated['bleu1_std']:.4f}")
    print(f"  BLEU-4:    {aggregated['bleu4']:.4f} +/- "
          f"{aggregated['bleu4_std']:.4f}")
    print(f"  ROUGE-L:   {aggregated['rouge_l']:.4f} +/- "
          f"{aggregated['rouge_l_std']:.4f}")
    print(f"  BERTScore: {aggregated['bertscore']:.4f} +/- "
          f"{aggregated['bertscore_std']:.4f}")

    return aggregated


if __name__ == '__main__':
    main()
