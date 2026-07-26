"""
LeViT-Hybrid Generation Training

Trains the LeViT-Hybrid generator to produce teacher evaluation comments
from SMPL pose sequences. The model uses a LeViT-style patch-based
Transformer encoder to encode poses, and an LSTM decoder with
teacher forcing to generate text.

Usage:
    python train.py [--gpu 0] [--epochs 100]
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
                           N_EPOCHS_GENERATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, MAX_COMMENT_LENGTH)
from shared.data_loader import (load_all_pose_sequences,
                                load_teacher_comments,
                                create_data_splits,
                                CCDanceGenerationDataset)
from shared.train_utils import EarlyStopping
from shared.metrics import compute_generation_metrics
from model import LeViTHybridGenerator

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='LeViT-Hybrid Generation Training'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_GENERATION,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--patch-size', type=int, default=15,
                        help='Frames per patch')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer encoder layers')
    parser.add_argument('--teacher-forcing', type=float, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def build_tokenizer(comments, max_vocab_size=5000):
    """Build a simple word-level tokenizer from teacher comments.

    Args:
        comments: list of comment strings
        max_vocab_size: maximum vocabulary size

    Returns:
        tokenizer dict with 'word2idx', 'idx2word', 'vocab_size'
    """
    word_counts = {}
    for comment in comments:
        if comment:
            for word in comment.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])

    # Build vocabulary with special tokens
    word2idx = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }

    for i, (word, _) in enumerate(sorted_words[:max_vocab_size - 4]):
        word2idx[word] = i + 4

    idx2word = {v: k for k, v in word2idx.items()}

    tokenizer = {
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab_size': len(word2idx),
        'max_vocab_size': max_vocab_size,
    }

    return tokenizer


def tokenize_comment(comment, tokenizer, max_len=MAX_COMMENT_LENGTH):
    """Convert a comment string to token indices.

    Args:
        comment: text string
        tokenizer: tokenizer dict
        max_len: maximum sequence length

    Returns:
        tokens: (max_len,) LongTensor with token indices
    """
    word2idx = tokenizer['word2idx']
    unk_idx = word2idx.get('<UNK>', 3)
    sos_idx = word2idx.get('<SOS>', 1)
    eos_idx = word2idx.get('<EOS>', 2)
    pad_idx = word2idx.get('<PAD>', 0)

    words = comment.lower().split() if comment else []
    token_ids = [sos_idx]
    for word in words:
        token_ids.append(word2idx.get(word, unk_idx))
        if len(token_ids) >= max_len - 1:
            break
    token_ids.append(eos_idx)

    # Pad to max_len
    tokens = torch.full((max_len,), pad_idx, dtype=torch.long)
    act_len = min(len(token_ids), max_len)
    tokens[:act_len] = torch.tensor(token_ids[:act_len], dtype=torch.long)

    return tokens


def create_generation_datasets(pose_sequences, comments, train_idx,
                                val_idx, test_idx, batch_size, tokenizer):
    """Create DataLoaders for generation task.

    Args:
        pose_sequences: dict of pose sequences
        comments: dict mapping index -> comment text
        train_idx, val_idx, test_idx: index lists
        batch_size: batch size
        tokenizer: tokenizer dict

    Returns:
        train_loader, val_loader, test_loader
    """
    train_comments = [comments.get(i, "") for i in train_idx]
    val_comments = [comments.get(i, "") for i in val_idx]
    test_comments = [comments.get(i, "") for i in test_idx]

    train_dataset = CCDanceGenerationDataset(
        train_idx, pose_sequences, train_comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH
    )
    val_dataset = CCDanceGenerationDataset(
        val_idx, pose_sequences, val_comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH
    )
    test_dataset = CCDanceGenerationDataset(
        test_idx, pose_sequences, test_comments,
        tokenizer=tokenizer, max_pose_len=SEQUENCE_LENGTH,
        max_comment_len=MAX_COMMENT_LENGTH
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


def train_epoch_generation(model, dataloader, optimizer, criterion, device,
                           teacher_forcing_ratio=0.5, scheduler=None):
    """Train one epoch for generation.

    Args:
        model: LeViTHybridGenerator
        dataloader: DataLoader yielding (pose, tokens, T, comment_text)
        optimizer: torch optimizer
        criterion: CrossEntropyLoss (ignore_index=0 for padding)
        device: torch device
        teacher_forcing_ratio: probability of teacher forcing
        scheduler: optional LR scheduler

    Returns:
        dict with 'loss' and 'perplexity'
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        # batch: (pose, tokens, T, comment_text)
        pose = batch[0].to(device)
        tokens = batch[1].to(device)
        T = batch[2]

        # Forward pass
        logits = model(pose, target_tokens=tokens,
                       teacher_forcing_ratio=teacher_forcing_ratio)

        # Compute loss: predict next token at each position
        # logits: (B, seq_len, V), tokens: (B, seq_len)
        # Shift: predict token i+1 from position i
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets_flat = tokens[:, 1:].reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tokens = (tokens[:, 1:] != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
    }


@torch.no_grad()
def eval_epoch_generation(model, dataloader, criterion, device, tokenizer):
    """Evaluate generation model on validation/test set.

    Args:
        model: LeViTHybridGenerator
        dataloader: DataLoader
        criterion: CrossEntropyLoss
        device: torch device
        tokenizer: tokenizer dict

    Returns:
        dict with loss, perplexity, and generated text examples
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_references = []
    all_candidates = []

    idx2word = tokenizer['idx2word']

    for batch in dataloader:
        pose = batch[0].to(device)
        tokens = batch[1].to(device)
        comment_texts = batch[3]  # original text strings

        # Compute loss with teacher forcing
        logits = model(pose, target_tokens=tokens, teacher_forcing_ratio=1.0)
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets_flat = tokens[:, 1:].reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        n_tokens = (tokens[:, 1:] != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Greedy decoding for evaluation
        generated_ids = model(pose, target_tokens=None).argmax(dim=-1)

        # Decode generated tokens
        for i in range(len(generated_ids)):
            gen_ids = generated_ids[i].cpu().numpy()
            # Filter out PAD, SOS, EOS
            gen_words = []
            for tid in gen_ids:
                if tid == 0:  # PAD
                    continue
                if tid == 2:  # EOS
                    break
                if tid == 1:  # SOS
                    continue
                gen_words.append(idx2word.get(tid, '<UNK>'))

            candidate = ' '.join(gen_words)
            reference = comment_texts[i] if i < len(comment_texts) else ""

            all_references.append(reference)
            all_candidates.append(candidate)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    # Compute generation metrics
    gen_metrics = compute_generation_metrics(all_references, all_candidates)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        **gen_metrics,
        'references': all_references[:5],  # First 5 examples
        'candidates': all_candidates[:5],
    }


def run_generation_experiment(model, train_loader, val_loader, test_loader,
                               device, n_epochs, lr, weight_decay,
                               teacher_forcing_ratio, tokenizer,
                               output_dir, model_name, seed):
    """Run a complete generation experiment.

    Args:
        model: LeViTHybridGenerator
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        n_epochs: number of epochs
        lr: learning rate
        weight_decay: weight decay
        teacher_forcing_ratio: teacher forcing probability
        tokenizer: tokenizer dict
        output_dir: output directory
        model_name: name for logging
        seed: random seed

    Returns:
        dict with test results
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        train_res = train_epoch_generation(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio=teacher_forcing_ratio,
            scheduler=scheduler,
        )
        val_res = eval_epoch_generation(
            model, val_loader, criterion, device, tokenizer
        )

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'train_perplexity': train_res['perplexity'],
            'val_loss': val_res['loss'],
            'val_perplexity': val_res['perplexity'],
            'val_bleu1': val_res.get('bleu1', 0.0),
            'val_bleu4': val_res.get('bleu4', 0.0),
            'val_rouge_l': val_res.get('rouge_l', 0.0),
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f}, "
                  f"train_ppl={train_res['perplexity']:.2f}, "
                  f"val_loss={val_res['loss']:.4f}, "
                  f"val_ppl={val_res['perplexity']:.2f}, "
                  f"BLEU-1={val_res.get('bleu1', 0.):.3f}")

        if val_res['loss'] < best_val_loss:
            best_val_loss = val_res['loss']
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if early_stop(val_res['loss']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    test_res = eval_epoch_generation(
        model, test_loader, criterion, device, tokenizer
    )

    # Save results
    results = {
        'model': model_name,
        'seed': seed,
        'n_epochs': len(train_log),
        'best_val_loss': best_val_loss,
        'test_metrics': {
            'loss': test_res['loss'],
            'perplexity': test_res['perplexity'],
            'bleu1': test_res.get('bleu1', 0.0),
            'bleu2': test_res.get('bleu2', 0.0),
            'bleu4': test_res.get('bleu4', 0.0),
            'rouge_l': test_res.get('rouge_l', 0.0),
            'bertscore': test_res.get('bertscore', 0.0),
        },
        'train_log': train_log,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / 'model_gen.pt')

    # Save results
    with open(output_dir / 'results_generation.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(output_dir / 'train_log_generation.json', 'w') as f:
        json.dump(train_log, f, indent=2)

    return results


def main():
    args = parse_args()

    # Device
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
    print("\nLoading pose data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    print(f"Loaded {len(valid_indices)} pose sequences")

    print("Loading teacher comments...")
    comments = load_teacher_comments(DATA_ROOT)
    print(f"Loaded comments for {len(comments)} samples")

    # Build tokenizer from all comments
    all_comments = [comments.get(i, "") for i in valid_indices]
    tokenizer = build_tokenizer(all_comments)
    print(f"Vocabulary size: {tokenizer['vocab_size']}")

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

        # Create generation dataloaders
        train_loader, val_loader, test_loader = create_generation_datasets(
            pose_sequences, comments, train_idx, val_idx, test_idx,
            args.batch_size, tokenizer
        )

        # Initialize model
        model = LeViTHybridGenerator(
            vocab_size=tokenizer['vocab_size'],
            pose_dim=69,
            seq_length=SEQUENCE_LENGTH,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            decoder_embed_dim=256,
            decoder_hidden_dim=512,
            decoder_num_layers=2,
            max_comment_length=MAX_COMMENT_LENGTH,
        ).to(device)

        # Train
        results = run_generation_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            teacher_forcing_ratio=args.teacher_forcing,
            tokenizer=tokenizer,
            output_dir=seed_dir,
            model_name='LeViTHybrid-Gen',
            seed=seed,
        )
        all_results.append(results['test_metrics'])

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregated Generation Results (across all seeds)")
    print(f"{'='*60}")

    # Compute mean/std for each metric
    metric_keys = ['loss', 'perplexity', 'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'bertscore']
    aggregated = {}
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in all_results]
        aggregated[key] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))

    for key in ['perplexity', 'bleu1', 'bleu4', 'rouge_l']:
        print(f"  {key}: {aggregated[key]:.3f} +/- {aggregated[f'{key}_std']:.3f}")

    # Save aggregated results
    aggregated_path = results_dir / 'aggregated_generation_results.json'
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {aggregated_path}")

    return aggregated


if __name__ == '__main__':
    main()
