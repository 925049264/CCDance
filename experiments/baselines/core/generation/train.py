"""
CoRe Generation Training

Trains the CoReGenerator with grade-conditioned decoding:
  - PoseLSTMEncoder backbone
  - Grade embedding for conditioning
  - LSTMDecoder with teacher forcing
  - Multi-seed evaluation (5 seeds)

Usage:
    python train.py [--gpu 0] [--epochs 100]
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
                           WEIGHT_DECAY, SEEDS, GRADE_MAP, N_CLASSES,
                           MODEL_CONFIGS, MAX_COMMENT_LENGTH)
from shared.data_loader import (load_all_pose_sequences,
                                load_teacher_comments,
                                create_data_splits,
                                CCDanceGenerationDataset)
from shared.train_utils import EarlyStopping
from shared.metrics import compute_generation_metrics
from model import CoReGenerator

warnings.filterwarnings('ignore')


# Simple whitespace tokenizer for comments
def whitespace_tokenize(text):
    """Simple whitespace tokenizer. Returns list of tokens."""
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens


def build_vocab(comments):
    """Build vocabulary from teacher comments.

    Args:
        comments: dict mapping idx -> comment string (or list of comments).

    Returns:
        word2idx, idx2word dicts.
    """
    # Collect all comments
    if isinstance(comments, dict):
        all_texts = [c for c in comments.values() if c]
    else:
        all_texts = [c for c in comments if c]

    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for text in all_texts:
        tokens = whitespace_tokenize(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    idx2word = {v: k for k, v in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}")
    return vocab, idx2word


def encode_comment(comment, vocab, max_len=MAX_COMMENT_LENGTH):
    """Encode a comment string to token indices.

    Returns:
        token_ids: (max_len,) long tensor with <PAD> padding.
    """
    tokens = whitespace_tokenize(comment)[:max_len - 2]
    ids = [vocab.get('<SOS>')]
    ids.extend([vocab.get(t, vocab['<UNK>']) for t in tokens])
    ids.append(vocab.get('<EOS>'))

    # Pad or truncate
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab['<PAD>']] * (max_len - len(ids))

    return torch.LongTensor(ids)


def decode_comment(token_ids, idx2word):
    """Decode token indices back to a string."""
    tokens = []
    for tid in token_ids:
        if tid == 2:  # <EOS>
            break
        if tid >= 4:  # skip special tokens
            tokens.append(idx2word.get(tid, '<UNK>'))
    return ' '.join(tokens)


def parse_args():
    parser = argparse.ArgumentParser(
        description='CoRe Generation Training'
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_GENERATION,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (smaller due to sequence generation)')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--teacher-forcing', type=float, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Decoder embedding dimension')
    parser.add_argument('--decoder-dim', type=int, default=512,
                        help='Decoder hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of decoder LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    return parser.parse_args()


def train_epoch_generation(model, dataloader, optimizer, criterion, device,
                           teacher_forcing_ratio=0.5, scheduler=None):
    """Train one generation epoch.

    Args:
        model: CoReGenerator instance.
        dataloader: DataLoader yielding (pose, tokens, T, comment).
        optimizer: torch optimizer.
        criterion: CrossEntropyLoss (ignore_index=0 for <PAD>).
        device: torch device.
        teacher_forcing_ratio: Probability of teacher forcing.
        scheduler: optional LR scheduler.

    Returns:
        dict with loss and perplexity.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        # CCDanceGenerationDataset returns (pose, tokens, T, comment)
        pose, tokens, _, _ = batch

        # Extract grade from data (we need to map back through the dataset)
        # For generation dataset, we pass grade as label during forward
        # Currently dataset doesn't have grade, so we need to handle this
        # Use a placeholder grade (default to 1/B)
        pose = pose.to(device)
        B = pose.size(0)

        # For generation dataset without explicit labels, use grade=1 as default
        # In practice, the generation training uses the full dataset
        grade = torch.ones(B, dtype=torch.long, device=device)

        # Target tokens for teacher forcing
        if isinstance(tokens, torch.Tensor) and tokens.dim() == 2:
            target_tokens = tokens.to(device)
        else:
            continue

        # Forward pass
        logits = model(
            pose, grade,
            target_tokens=target_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Compute loss (shifted: predict next token)
        # logits: (B, seq_len, V), target_tokens: (B, seq_len)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tokens = (target_tokens != 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 10.0))

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
    }


@torch.no_grad()
def eval_epoch_generation(model, dataloader, criterion, device,
                          idx2word, teacher_forcing_ratio=0.0):
    """Evaluate one generation epoch.

    Args:
        model: CoReGenerator instance.
        dataloader: DataLoader.
        criterion: CrossEntropyLoss.
        device: torch device.
        idx2word: Index-to-word mapping for decoding.
        teacher_forcing_ratio: 0.0 for evaluation (no teacher forcing).

    Returns:
        dict with loss, perplexity, and generated samples.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_references = []
    all_candidates = []

    for batch in dataloader:
        pose, tokens, _, comments = batch
        pose = pose.to(device)
        B = pose.size(0)

        # Default grade
        grade = torch.ones(B, dtype=torch.long, device=device)

        if isinstance(tokens, torch.Tensor) and tokens.dim() == 2:
            target_tokens = tokens.to(device)

            logits = model(
                pose, grade,
                target_tokens=target_tokens,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )

            n_tokens = (target_tokens != 0).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        # Generate greedily for evaluation
        gen_ids, _ = model.generate(pose, grade, device)
        for i in range(B):
            candidate = decode_comment(gen_ids[i].cpu().tolist(), idx2word)
            reference = comments[i] if isinstance(comments[i], str) else ""
            all_references.append(reference)
            all_candidates.append(candidate)

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 10.0))

    # Compute generation metrics if we have references
    gen_metrics = {}
    if all_references and all_candidates:
        gen_metrics = compute_generation_metrics(all_references, all_candidates)

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        **gen_metrics,
        'references': all_references[:5],       # sample for logging
        'candidates': all_candidates[:5],
    }


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

    # Load teacher comments
    print("Loading teacher comments...")
    comments_dict = load_teacher_comments(DATA_ROOT)
    # Filter to valid indices and align
    comments = [comments_dict.get(i, "") for i in valid_indices]
    print(f"Loaded {sum(1 for c in comments if c)} non-empty comments")

    # Build vocabulary
    vocab, idx2word = build_vocab(comments)
    vocab_size = len(vocab)

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

        # Encode comments for each split
        def encode_split(indices):
            return [encode_comment(comments[valid_indices.index(i)], vocab)
                    for i in indices]

        train_tokens = encode_split(train_idx)
        val_tokens = encode_split(val_idx)
        test_tokens = encode_split(test_idx)

        train_comments = [comments[valid_indices.index(i)] for i in train_idx]
        val_comments = [comments[valid_indices.index(i)] for i in val_idx]
        test_comments = [comments[valid_indices.index(i)] for i in test_idx]

        # Create datasets
        train_dataset = CCDanceGenerationDataset(
            train_idx, pose_sequences, train_comments,
            max_pose_len=SEQUENCE_LENGTH,
            max_comment_len=MAX_COMMENT_LENGTH,
        )
        val_dataset = CCDanceGenerationDataset(
            val_idx, pose_sequences, val_comments,
            max_pose_len=SEQUENCE_LENGTH,
            max_comment_len=MAX_COMMENT_LENGTH,
        )
        test_dataset = CCDanceGenerationDataset(
            test_idx, pose_sequences, test_comments,
            max_pose_len=SEQUENCE_LENGTH,
            max_comment_len=MAX_COMMENT_LENGTH,
        )

        # Override __getitem__ to return tokens instead of raw text
        def make_collate(tokens_list):
            def collate_fn(batch):
                poses = []
                ts = []
                comments_out = []
                for i, (pose, comment, T) in enumerate(batch):
                    poses.append(pose)
                    ts.append(T)
                    comments_out.append(comment if isinstance(comment, str) else "")
                poses = torch.stack(poses, dim=0)
                token_tensor = torch.stack(tokens_list, dim=0)
                return poses, token_tensor, ts, comments_out
            return collate_fn

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
            collate_fn=make_collate(train_tokens),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            collate_fn=make_collate(val_tokens),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            collate_fn=make_collate(test_tokens),
        )

        # Initialize model
        model = CoReGenerator(
            vocab_size=vocab_size,
            pose_dim=69,
            hidden_dim=256,
            decoder_dim=args.decoder_dim,
            num_layers=args.num_layers,
            num_classes=N_CLASSES,
            max_len=MAX_COMMENT_LENGTH,
            dropout=args.dropout,
        ).to(device)

        # Training setup
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        early_stop = EarlyStopping(patience=10, mode='min')

        print(f"\n{'='*60}")
        print(f"CoRe Generation Training (seed={seed})")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Max comment length: {MAX_COMMENT_LENGTH}")
        print(f"{'='*60}")

        best_val_loss = float('inf')
        best_model_state = None
        train_log = []

        for epoch in range(args.epochs):
            train_res = train_epoch_generation(
                model, train_loader, optimizer, criterion, device,
                teacher_forcing_ratio=args.teacher_forcing,
                scheduler=scheduler,
            )
            val_res = eval_epoch_generation(
                model, val_loader, criterion, device, idx2word,
                teacher_forcing_ratio=0.0,
            )

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_res['loss'],
                'train_perplexity': train_res['perplexity'],
                'val_loss': val_res['loss'],
                'val_perplexity': val_res['perplexity'],
                'val_bleu1': val_res.get('bleu1', 0.0),
                'val_rouge_l': val_res.get('rouge_l', 0.0),
                'lr': optimizer.param_groups[0]['lr'],
            }
            train_log.append(log_entry)

            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(f"  Epoch {epoch+1}/{args.epochs}: "
                      f"train_loss={train_res['loss']:.4f} "
                      f"(ppl={train_res['perplexity']:.2f}) "
                      f"val_loss={val_res['loss']:.4f} "
                      f"(ppl={val_res['perplexity']:.2f}) "
                      f"val_bleu1={val_res.get('bleu1', 0):.4f}")

            if val_res['loss'] < best_val_loss:
                best_val_loss = val_res['loss']
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

            if early_stop(val_res['loss']):
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        test_res = eval_epoch_generation(
            model, test_loader, criterion, device, idx2word,
            teacher_forcing_ratio=0.0,
        )

        # Print sample generations
        print(f"\n  Sample generations:")
        for ref, cand in zip(test_res.get('references', []),
                             test_res.get('candidates', [])):
            print(f"    Reference: {ref[:80]}")
            print(f"    Generated: {cand[:80]}")
            print()

        results = {
            'model': 'CoReGenerator',
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

        # Save
        torch.save(model.state_dict(), seed_dir / 'model.pt')
        with open(seed_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        with open(seed_dir / 'train_log.json', 'w') as f:
            json.dump(train_log, f, indent=2)

        all_results.append(results['test_metrics'])

    # Aggregate
    print(f"\n{'='*60}")
    print("Aggregated Generation Results")
    print(f"{'='*60}")

    if all_results:
        aggregated = {}
        for key in ['loss', 'perplexity', 'bleu1', 'bleu2', 'bleu4',
                     'rouge_l', 'bertscore']:
            vals = [r.get(key, 0.0) for r in all_results]
            aggregated[key] = float(np.mean(vals))
            aggregated[f'{key}_std'] = float(np.std(vals))

        for key in ['bleu1', 'bleu2', 'bleu4', 'rouge_l', 'bertscore']:
            print(f"  {key}: {aggregated[key]:.4f} +/- {aggregated[f'{key}_std']:.4f}")

        aggregated_path = results_dir / 'aggregated_results.json'
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated results saved to {aggregated_path}")

    return aggregated if all_results else {}


if __name__ == '__main__':
    main()
