#!/usr/bin/env python3
"""
USDL Generation Training Script.

Trains a USDLGenerator (STGCNEncoder -> score distribution -> LSTMDecoder)
to produce teacher comments conditioned on the predicted grade.

Usage:
    python train.py --gpu 0
    python train.py --gpu 0 --epochs 100 --lr 1e-4
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import (
    DATA_ROOT,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    N_EPOCHS_GENERATION,
    SEEDS,
    MAX_COMMENT_LENGTH,
    MODEL_CONFIGS,
)
from shared.data_loader import (
    load_all_pose_sequences,
    create_data_splits,
    CCDanceGenerationDataset,
    load_teacher_comments,
)
from shared.train_utils import EarlyStopping
from shared.metrics import compute_generation_metrics
from model import USDLGenerator, build_vocab, CommentTokenizer


# ============================================================================
# Collation
# ============================================================================


def collate_gen_batch(batch):
    """Collate function for variable-length token sequences.

    Each item: (pose_tensor, tokens_tensor, T_original, comment_text)
    """
    poses = []
    token_lists = []
    lengths = []
    texts = []

    for item in batch:
        if len(item) == 4:
            pose, tokens, T, comment = item
        else:
            pose, tokens, comment = item[0], item[1], item[-1]
            T = 0

        poses.append(pose)
        if isinstance(tokens, torch.Tensor):
            token_lists.append(tokens)
        else:
            token_lists.append(torch.LongTensor([1]))  # SOS only
        lengths.append(T)
        texts.append(comment)

    # Pad poses
    poses = torch.stack(poses, dim=0)

    # Pad token sequences
    token_lengths = [t.size(0) for t in token_lists]
    max_tok_len = max(token_lengths)
    padded_tokens = torch.zeros(len(token_lists), max_tok_len, dtype=torch.long)
    for i, t in enumerate(token_lists):
        padded_tokens[i, :t.size(0)] = t

    lengths_tensor = torch.LongTensor(lengths)
    token_lengths_tensor = torch.LongTensor(token_lengths)

    return poses, padded_tokens, token_lengths_tensor, lengths_tensor, texts


# ============================================================================
# Training / Evaluation
# ============================================================================


def train_epoch_generation(model, dataloader, optimizer, device,
                           pad_idx=0, scheduler=None, clip_norm=1.0):
    """Train one generation epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        poses, tokens, tok_lengths, _, _ = batch
        poses = poses.to(device)
        tokens = tokens.to(device)
        B, seq_len = tokens.shape

        # Forward pass (teacher forcing)
        logits, _ = model(poses, target_tokens=tokens, teacher_forcing_ratio=0.5)

        # Loss: CrossEntropy (ignore padding)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))        # (B*(T-1), V)
        targets = tokens[:, 1:].reshape(-1)                             # (B*(T-1),)

        loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * B
        n_samples += B

    return {"loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def eval_epoch_generation(model, dataloader, tokenizer, device,
                          pad_idx=0, max_gen_len=128):
    """Evaluate generation: loss + text metrics."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_references = []
    all_candidates = []

    for batch in dataloader:
        poses, tokens, tok_lengths, _, texts = batch
        poses = poses.to(device)
        tokens = tokens.to(device)
        B, seq_len = tokens.shape

        # Loss
        logits, dist = model(poses, target_tokens=tokens, teacher_forcing_ratio=0.0)
        logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        targets = tokens[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)
        total_loss += loss.item() * B
        n_samples += B

        # Generate comments for metric computation
        gen_tokens, _ = model.generate(poses, max_len=max_gen_len)
        for i in range(B):
            candidate = tokenizer.decode(gen_tokens[i].cpu().tolist())
            reference = texts[i] if texts[i] else ""
            all_references.append(reference)
            all_candidates.append(candidate)

    avg_loss = total_loss / max(n_samples, 1)

    # Compute text generation metrics
    gen_metrics = compute_generation_metrics(all_references, all_candidates)

    return {
        "loss": avg_loss,
        **gen_metrics,
        "references": all_references,
        "candidates": all_candidates,
    }


def run_generation_experiment(model, train_loader, val_loader, test_loader,
                              tokenizer, device, model_name, seed, output_dir,
                              n_epochs=100, lr=1e-4, weight_decay=1e-4):
    """Run complete generation experiment."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    early_stop = EarlyStopping(patience=10, mode="max")

    best_val_bleu4 = 0.0
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_generation(
            model, train_loader, optimizer, device,
            pad_idx=tokenizer.word2idx.get("<PAD>", 0),
            scheduler=scheduler,
        )
        val_res = eval_epoch_generation(
            model, val_loader, tokenizer, device,
            pad_idx=tokenizer.word2idx.get("<PAD>", 0),
            max_gen_len=128,
        )

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_res["loss"],
            "val_loss": val_res["loss"],
            "val_bleu1": val_res["bleu1"],
            "val_bleu2": val_res["bleu2"],
            "val_bleu4": val_res["bleu4"],
            "val_rouge_l": val_res["rouge_l"],
            "val_bertscore": val_res["bertscore"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch+1}/{n_epochs}: "
                f"train_loss={train_res['loss']:.4f}, "
                f"val_loss={val_res['loss']:.4f}, "
                f"val_BLEU-4={val_res['bleu4']:.4f}, "
                f"val_ROUGE-L={val_res['rouge_l']:.4f}"
            )

        if val_res["bleu4"] > best_val_bleu4:
            best_val_bleu4 = val_res["bleu4"]
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if early_stop(val_res["bleu4"]):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Best model test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_generation(
        model, test_loader, tokenizer, device,
        pad_idx=tokenizer.word2idx.get("<PAD>", 0),
        max_gen_len=128,
    )

    results = {
        "model": model_name,
        "seed": seed,
        "approach": "generation",
        "n_epochs": len(train_log),
        "best_val_bleu4": best_val_bleu4,
        "test_metrics": {
            "loss": test_res["loss"],
            "bleu1": test_res["bleu1"],
            "bleu2": test_res["bleu2"],
            "bleu4": test_res["bleu4"],
            "rouge_l": test_res["rouge_l"],
            "bertscore": test_res["bertscore"],
        },
        "train_log": train_log,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": tokenizer.word2idx,
        },
        output_dir / "model.pt",
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    # Save sample generations
    samples = []
    for ref, cand in zip(test_res["references"][:10],
                         test_res["candidates"][:10]):
        samples.append({"reference": ref, "generated": cand})
    with open(output_dir / "samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    return results


def run_generation_with_seeds(model_fn, model_args, train_loaders, val_loaders,
                              test_loaders, tokenizers, device, model_name,
                              output_dir, seeds, n_epochs=100, lr=1e-4,
                              weight_decay=1e-4):
    """Run generation across multiple seeds."""
    all_seed_results = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Running {model_name} [generation] with seed {seed}")
        print(f"{'=' * 60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = model_fn(**model_args)

        seed_output_dir = Path(output_dir) / f"seed_{seed}"
        res = run_generation_experiment(
            model,
            train_loaders[seed], val_loaders[seed], test_loaders[seed],
            tokenizers[seed], device, model_name, seed,
            seed_output_dir,
            n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        )
        all_seed_results.append(res["test_metrics"])

    # Aggregate
    bleu4s = [r["bleu4"] for r in all_seed_results]
    rouges = [r["rouge_l"] for r in all_seed_results]
    bert_scores = [r["bertscore"] for r in all_seed_results]

    aggregated = {
        "bleu4": float(np.mean(bleu4s)),
        "bleu4_std": float(np.std(bleu4s)),
        "rouge_l": float(np.mean(rouges)),
        "rouge_l_std": float(np.std(rouges)),
        "bertscore": float(np.mean(bert_scores)),
        "bertscore_std": float(np.std(bert_scores)),
        "per_seed": all_seed_results,
    }

    with open(Path(output_dir) / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{model_name} [generation] aggregated results:")
    print(f"  BLEU-4:    {aggregated['bleu4']:.4f} +/- {aggregated['bleu4_std']:.4f}")
    print(f"  ROUGE-L:   {aggregated['rouge_l']:.4f} +/- {aggregated['rouge_l_std']:.4f}")
    print(f"  BERTScore: {aggregated['bertscore']:.4f} +/- {aggregated['bertscore_std']:.4f}")
    print(f"{'=' * 60}")

    return aggregated


# ============================================================================
# Data loaders for generation
# ============================================================================


def create_gen_dataloaders(pose_sequences, metadata, comments_dict,
                           tokenizer, train_idx, val_idx, test_idx,
                           batch_size=16, max_pose_len=300,
                           max_comment_len=512):
    """Create DataLoaders for the generation task."""
    train_comments = [comments_dict.get(i, "") for i in train_idx]
    val_comments = [comments_dict.get(i, "") for i in val_idx]
    test_comments = [comments_dict.get(i, "") for i in test_idx]

    train_dataset = CCDanceGenerationDataset(
        train_idx, pose_sequences, train_comments,
        tokenizer=tokenizer, max_pose_len=max_pose_len,
        max_comment_len=max_comment_len,
    )
    val_dataset = CCDanceGenerationDataset(
        val_idx, pose_sequences, val_comments,
        tokenizer=tokenizer, max_pose_len=max_pose_len,
        max_comment_len=max_comment_len,
    )
    test_dataset = CCDanceGenerationDataset(
        test_idx, pose_sequences, test_comments,
        tokenizer=tokenizer, max_pose_len=max_pose_len,
        max_comment_len=max_comment_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate_gen_batch,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_gen_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_gen_batch,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="USDL Generation Training"
    )
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS_GENERATION)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-embed-dim", type=int, default=256)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--min-word-freq", type=int, default=2,
                        help="Minimum word frequency for vocabulary")
    parser.add_argument("--max-comment-len", type=int, default=MAX_COMMENT_LENGTH)
    parser.add_argument("--output-dir", type=str,
                        default=str(
                            Path(__file__).parent.parent / "results" / "generation"
                        ),
                        help="Output directory")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading pose sequences...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(
        data_root=DATA_ROOT
    )
    print(f"  Loaded {len(valid_indices)} valid samples")

    print("Loading teacher comments...")
    comments_dict = load_teacher_comments(data_root=DATA_ROOT)
    print(f"  Loaded {len(comments_dict)} comments")

    # Build vocabulary from ALL comments (word2idx used as tokenizer arg)
    all_comments = [comments_dict.get(i, "") for i in valid_indices]
    word2idx, idx2word = build_vocab(all_comments, min_freq=args.min_word_freq)
    print(f"  Vocabulary size: {len(word2idx)} (min_freq={args.min_word_freq})")

    # ------------------------------------------------------------------
    # Build per-seed loaders and tokenizers
    # ------------------------------------------------------------------
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    tokenizers = {}

    for seed in SEEDS:
        train_idx, val_idx, test_idx = create_data_splits(
            valid_indices, metadata, seed=seed
        )

        # Build a tokenizer from training comments only (no data leakage)
        train_comments = [comments_dict.get(i, "") for i in train_idx]
        train_word2idx, _ = build_vocab(train_comments, min_freq=args.min_word_freq)
        # Merge unknown token for words not seen in train
        tokenizer = CommentTokenizer(train_word2idx)
        tokenizers[seed] = tokenizer

        tr_loader, val_loader, te_loader = create_gen_dataloaders(
            pose_sequences, metadata, comments_dict,
            tokenizer, train_idx, val_idx, test_idx,
            batch_size=args.batch_size,
            max_pose_len=SEQUENCE_LENGTH,
            max_comment_len=args.max_comment_len,
        )
        train_loaders[seed] = tr_loader
        val_loaders[seed] = val_loader
        test_loaders[seed] = te_loader

        print(f"  Seed {seed}: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}, "
              f"vocab={tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    model_name = "USDL-Generator"
    model_fn = USDLGenerator
    model_args = {
        "vocab_size": tokenizers[SEEDS[0]].vocab_size,
        "hidden_dim": args.hidden_dim,
        "n_bins": args.n_bins,
        "decoder_hidden_dim": args.decoder_hidden_dim,
        "decoder_embed_dim": args.decoder_embed_dim,
        "grade_embed_dim": 64,
        "dropout": 0.3,
        "max_comment_len": args.max_comment_len,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    run_generation_with_seeds(
        model_fn, model_args,
        train_loaders, val_loaders, test_loaders,
        tokenizers, device, model_name, output_dir,
        seeds=SEEDS, n_epochs=args.epochs, lr=args.lr,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
