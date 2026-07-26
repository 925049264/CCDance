"""
VL-Transformer Generation Training

Trains a generator that produces teacher evaluation comments from
pose+audio input. Uses teacher embedding regression (Sentence-BERT
768-D targets) as the primary objective.

Architecture:
- STGCN motion encoder -> 256-D
- Music LSTM encoder -> 256-D
- Fusion (concat + Linear 512->256)
- Teacher embedding regression head (256 -> 768)

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
                           N_EPOCHS_GENERATION, LEARNING_RATE,
                           WEIGHT_DECAY, SEEDS, GRADE_MAP,
                           MODEL_CONFIGS, TEACHER_EMBED_DIM,
                           MAX_COMMENT_LENGTH)
from shared.data_loader import (load_all_pose_sequences,
                                extract_audio_features,
                                build_audio_features_per_sample,
                                load_teacher_comments,
                                create_data_splits)
from shared.train_utils import EarlyStopping
from shared.metrics import compute_generation_metrics
from model import VLTransformerGenerator

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description='VL-Transformer Generation Training'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS_GENERATION,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--seed', type=int, nargs='+', default=SEEDS,
                        help='Random seeds')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent),
                        help='Output directory')
    parser.add_argument('--compute-embeddings', action='store_true',
                        help='Compute Sentence-BERT embeddings on the fly')
    return parser.parse_args()


def compute_sbert_embeddings(comments, device='cpu'):
    """Compute Sentence-BERT embeddings for teacher comments.

    Falls back to simple averaged GloVe-like embeddings if Sentence-BERT
    is not available.

    Args:
        comments: dict mapping idx -> comment string
        device: torch device

    Returns:
        embeddings: dict mapping idx -> 768-D numpy array
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        texts = [comments[idx] if comments[idx] else "" for idx in comments]
        if len(texts) == 0:
            return {}
        emb_list = model.encode(texts, show_progress_bar=False)
        embeddings = {
            idx: emb_list[i].astype(np.float32)
            for i, idx in enumerate(comments)
        }
        return embeddings
    except ImportError:
        print("  Warning: sentence-transformers not available. "
              "Using fallback random embeddings for test.")
        # Fallback: return random 768-D embeddings (will not produce useful results)
        rng = np.random.RandomState(42)
        return {
            idx: rng.randn(TEACHER_EMBED_DIM).astype(np.float32)
            for idx in comments
        }


def load_precomputed_embeddings(data_root=DATA_ROOT):
    """Load precomputed Sentence-BERT teacher embeddings from cache."""
    cache_file = Path(data_root) / "experiments" / "results" / "teacher_embeddings.pkl"
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None


class GenerationDataset(torch.utils.data.Dataset):
    """Dataset for teacher embedding regression."""

    def __init__(self, sample_indices, pose_sequences, audio_features,
                 teacher_embeddings, max_pose_len=SEQUENCE_LENGTH):
        self.sample_indices = sample_indices
        self.pose_sequences = pose_sequences
        self.audio_features = audio_features
        self.teacher_embeddings = teacher_embeddings
        self.max_pose_len = max_pose_len

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]
        pose = self.pose_sequences[real_idx].astype(np.float32)
        audio = self.audio_features.get(real_idx)
        if audio is None:
            audio = np.zeros((1, 64), dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        teacher_emb = self.teacher_embeddings.get(real_idx, np.zeros(
            TEACHER_EMBED_DIM, dtype=np.float32
        ))

        # Pad/truncate pose
        T = pose.shape[0]
        if T > self.max_pose_len:
            indices = np.linspace(0, T - 1, self.max_pose_len, dtype=int)
            pose = pose[indices]
        elif T < self.max_pose_len:
            pad = np.zeros((self.max_pose_len - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        # Handle audio
        if audio.ndim == 2 and audio.shape[0] > 1:
            if audio.shape[0] > self.max_pose_len // 10:
                a_idx = np.linspace(0, audio.shape[0] - 1,
                                    self.max_pose_len // 10, dtype=int)
                audio = audio[a_idx]

        return (torch.FloatTensor(pose),
                torch.FloatTensor(audio) if audio.ndim == 2
                else torch.FloatTensor(audio),
                torch.FloatTensor(teacher_emb))


def create_generation_dataloaders(valid_indices, metadata, pose_sequences,
                                   audio_features, teacher_embeddings,
                                   train_idx, val_idx, test_idx,
                                   batch_size=BATCH_SIZE):
    """Create DataLoaders for generation task."""
    train_dataset = GenerationDataset(
        train_idx, pose_sequences, audio_features, teacher_embeddings
    )
    val_dataset = GenerationDataset(
        val_idx, pose_sequences, audio_features, teacher_embeddings
    )
    test_dataset = GenerationDataset(
        test_idx, pose_sequences, audio_features, teacher_embeddings
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch_generation(model, dataloader, optimizer, criterion, device,
                           scheduler=None):
    """Train one epoch for teacher embedding regression."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        pose, audio, target_emb = batch
        pose = pose.to(device)
        target_emb = target_emb.to(device)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(device)

        # Forward
        outputs = model(pose, audio, mode='embedding')
        pred_emb = outputs['teacher_embed']

        # Loss: cosine embedding loss + MSE
        loss_mse = criterion(pred_emb, target_emb)
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)
        loss_cos = (1.0 - cos_sim).mean()
        loss = loss_mse + 0.5 * loss_cos

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / n_batches}


@torch.no_grad()
def eval_epoch_generation(model, dataloader, criterion, device):
    """Evaluate one epoch for teacher embedding regression."""
    model.eval()
    total_loss = 0.0
    all_pred_embs = []
    all_target_embs = []
    n_batches = 0

    for batch in dataloader:
        pose, audio, target_emb = batch
        pose = pose.to(device)
        target_emb = target_emb.to(device)
        if isinstance(audio, torch.Tensor):
            audio = audio.to(device)

        outputs = model(pose, audio, mode='embedding')
        pred_emb = outputs['teacher_embed']

        loss_mse = criterion(pred_emb, target_emb)
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)
        loss_cos = (1.0 - cos_sim).mean()
        loss = loss_mse + 0.5 * loss_cos

        total_loss += loss.item()
        all_pred_embs.append(pred_emb.cpu().numpy())
        all_target_embs.append(target_emb.cpu().numpy())
        n_batches += 1

    pred_embs = np.concatenate(all_pred_embs, axis=0)
    target_embs = np.concatenate(all_target_embs, axis=0)

    # Compute average cosine similarity
    cos_sim_values = []
    for i in range(len(pred_embs)):
        cos_sim_values.append(
            np.dot(pred_embs[i], target_embs[i]) /
            (np.linalg.norm(pred_embs[i]) * np.linalg.norm(target_embs[i]) + 1e-8)
        )
    avg_cos_sim = float(np.mean(cos_sim_values))

    return {
        'loss': total_loss / n_batches,
        'cosine_similarity': avg_cos_sim,
        'pred_embeddings': pred_embs,
        'target_embeddings': target_embs,
    }


def run_generation_experiment(model, train_loader, val_loader, test_loader,
                               device, model_name, seed, output_dir,
                               n_epochs=100, lr=1e-4, weight_decay=1e-4):
    """Run generation experiment with training, validation, and testing."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs
    )
    early_stop = EarlyStopping(patience=10, mode='max')

    best_val_sim = -1.0
    best_model_state = None
    train_log = []

    for epoch in range(n_epochs):
        train_res = train_epoch_generation(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_res = eval_epoch_generation(
            model, val_loader, criterion, device
        )

        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_res['loss'],
            'val_loss': val_res['loss'],
            'val_cosine_similarity': val_res['cosine_similarity'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train_loss={train_res['loss']:.4f}, "
                  f"val_loss={val_res['loss']:.4f}, "
                  f"val_cos_sim={val_res['cosine_similarity']:.4f}")

        if val_res['cosine_similarity'] > best_val_sim:
            best_val_sim = val_res['cosine_similarity']
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if early_stop(val_res['cosine_similarity']):
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_res = eval_epoch_generation(model, test_loader, criterion, device)

    # Save results
    results = {
        'model': model_name,
        'seed': seed,
        'n_epochs': len(train_log),
        'best_val_cosine_similarity': best_val_sim,
        'test_metrics': {
            'loss': test_res['loss'],
            'cosine_similarity': test_res['cosine_similarity'],
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

    return results, test_res


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
    print("\nLoading data...")
    pose_sequences, metadata, valid_indices = load_all_pose_sequences(DATA_ROOT)
    print(f"Loaded {len(valid_indices)} pose sequences")

    # Load audio features
    print("Loading audio features...")
    audio_features = extract_audio_features(DATA_ROOT)
    per_sample_audio = build_audio_features_per_sample(
        valid_indices, metadata, audio_features
    )

    # Load teacher comments and compute embeddings
    print("Loading teacher comments...")
    comments = load_teacher_comments(DATA_ROOT)
    print(f"Loaded {sum(1 for c in comments.values() if c)} non-empty comments")

    # Compute or load precomputed teacher embeddings
    precomputed = load_precomputed_embeddings(DATA_ROOT)
    if precomputed is not None:
        print("Using precomputed teacher embeddings")
        teacher_embeddings = precomputed
    else:
        print("Computing Sentence-BERT embeddings...")
        teacher_embeddings = compute_sbert_embeddings(
            comments, device='cpu'
        )
        print(f"Computed {len(teacher_embeddings)} embeddings")

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
        train_loader, val_loader, test_loader = create_generation_dataloaders(
            valid_indices, metadata, pose_sequences, per_sample_audio,
            teacher_embeddings, train_idx, val_idx, test_idx,
            batch_size=args.batch_size
        )

        # Initialize model
        model = VLTransformerGenerator(
            pose_dim=69,
            audio_dim=64,
            hidden_dim=256,
            num_joints=23,
            dropout=args.dropout,
            vocab_size=None,  # Embedding regression mode
            max_comment_length=MAX_COMMENT_LENGTH,
            teacher_embed_dim=TEACHER_EMBED_DIM,
        ).to(device)

        # Train
        results, test_res = run_generation_experiment(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            model_name='VLTransformer_Gen',
            seed=seed,
            output_dir=seed_dir,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        all_results.append({
            'seed': seed,
            'cosine_similarity': test_res['cosine_similarity'],
            'loss': test_res['loss'],
        })

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregated Generation Results")
    print(f"{'='*60}")

    cos_sims = [r['cosine_similarity'] for r in all_results]
    losses = [r['loss'] for r in all_results]

    aggregated = {
        'cosine_similarity': float(np.mean(cos_sims)),
        'cosine_similarity_std': float(np.std(cos_sims)),
        'loss': float(np.mean(losses)),
        'loss_std': float(np.std(losses)),
        'per_seed': all_results,
    }

    print(f"  Cosine Similarity: {aggregated['cosine_similarity']:.4f} "
          f"+/- {aggregated['cosine_similarity_std']:.4f}")
    print(f"  Loss:             {aggregated['loss']:.4f} "
          f"+/- {aggregated['loss_std']:.4f}")

    aggregated_path = results_dir / 'aggregated_results.json'
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to {aggregated_path}")


if __name__ == '__main__':
    main()
