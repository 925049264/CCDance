#!/usr/bin/env python3
"""
Unified Dance Quality Comment Generation Baseline
Trains a pose encoder → SBERT embedding regression model for all 5 baselines.
Evaluates using cosine similarity against ground-truth teacher embeddings,
plus BLEU/ROUGE-L/BERTScore on a simple template-based text generation baseline.

Usage:
  python run_generation.py --baseline usdl --gpu 0 --epochs 50
"""
import sys, os, json, argparse, pickle, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Add project root (experiments/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.shared.config import (DATA_ROOT, GRADE_MAP, N_CLASSES, SEEDS,
                                      SEQUENCE_LENGTH, BATCH_SIZE)
from baselines.shared.models import (STGCNEncoder, PoseLSTMEncoder,
                                      TransformerEncoder, GraphTransformerEncoder,
                                      MusicLSTMEncoder, MLPClassifier)
from baselines.shared.metrics import (compute_bleu, compute_rouge_l,
                                       compute_bertscore, compute_generation_metrics)


def load_data():
    """Load pose sequences and teacher comments."""
    import pickle as pkl
    cache = Path(DATA_ROOT) / "experiments" / "results" / "pose_sequences.pkl"
    with open(cache, 'rb') as f:
        data = pkl.load(f)
    pose_seqs = data['pose_sequences']
    metadata = data['metadata']
    valid_idx = data['valid_indices']

    # Load teacher comments (EN)
    comments = {}
    from baselines.shared.data_loader import load_teacher_comments
    all_comments = load_teacher_comments(DATA_ROOT)
    for i in valid_idx:
        comments[i] = all_comments.get(i, "")

    return pose_seqs, metadata, valid_idx, comments


def compute_sbert_embeddings(texts, batch_size=32):
    """Compute Sentence-BERT embeddings for a list of texts."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, device=device)
        return embeddings
    except Exception as e:
        print(f"  Warning: SBERT failed ({e}), using fallback")
        # Fallback: bag-of-words TF-IDF style
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=256)
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings


def create_encoder(baseline, pose_dim=69, hidden_dim=256):
    """Create the pose encoder for a specific baseline."""
    if baseline == 'usdl' or baseline == 'vl_transformer':
        return STGCNEncoder(input_dim=pose_dim, hidden_dim=hidden_dim,
                            output_dim=hidden_dim, dropout=0.3)
    elif baseline == 'core':
        return PoseLSTMEncoder(input_dim=pose_dim, hidden_dim=hidden_dim,
                               output_dim=hidden_dim, dropout=0.3)
    elif baseline == 'levit_hybrid':
        return TransformerEncoder(input_dim=pose_dim, d_model=hidden_dim,
                                  nhead=8, num_layers=4, output_dim=hidden_dim)
    elif baseline == 'graph_transformer':
        return GraphTransformerEncoder(input_dim=3, d_model=hidden_dim,
                                        nhead=8, num_layers=4, dropout=0.1)
    else:
        return STGCNEncoder(input_dim=pose_dim, hidden_dim=hidden_dim,
                            output_dim=hidden_dim, dropout=0.3)


class CommentRegressor(nn.Module):
    """Pose encoder → SBERT embedding regression model."""

    def __init__(self, baseline, pose_dim=69, hidden_dim=256, embed_dim=384,
                 dropout=0.3):
        super().__init__()
        self.encoder = create_encoder(baseline, pose_dim, hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, pose):
        # pose: (B, T, 69) - already padded to SEQUENCE_LENGTH by dataset
        feat = self.encoder(pose)
        return self.regressor(feat)


class PoseCommentDataset(Dataset):
    """Simple dataset: pose sequence + SBERT embedding target."""

    def __init__(self, indices, pose_seqs, embeddings):
        self.indices = indices
        self.pose_seqs = pose_seqs
        self.embeddings = embeddings  # numpy array

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pose = self.pose_seqs[real_idx].astype(np.float32)
        embed = self.embeddings[idx].astype(np.float32)

        # Pad/truncate to fixed length for batch collation
        T = pose.shape[0]
        if T > SEQUENCE_LENGTH:
            indices = np.linspace(0, T - 1, SEQUENCE_LENGTH, dtype=int)
            pose = pose[indices]
        elif T < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        return torch.FloatTensor(pose), torch.FloatTensor(embed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for pose, target in loader:
        pose, target = pose.to(device), target.to(device)
        pred = model(pose)
        # MSE + cosine embedding loss
        mse = F.mse_loss(pred, target)
        cos = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        loss = mse + 0.5 * cos
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pose.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_cos_sim = []
    for pose, target in loader:
        pose, target = pose.to(device), target.to(device)
        pred = model(pose)
        mse = F.mse_loss(pred, target)
        cos = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        loss = mse + 0.5 * cos
        total_loss += loss.item() * pose.size(0)
        cos_sim = F.cosine_similarity(pred, target, dim=-1).cpu().numpy()
        all_cos_sim.extend(cos_sim.tolist())
    return total_loss / len(loader.dataset), np.mean(all_cos_sim)


def run_generation_experiment(baseline, gpu_id, epochs=50, lr=1e-3):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(__file__).parent / baseline / 'generation'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generation: {baseline} on {device}")
    print(f"{'='*60}")

    # Load data
    print("Loading data...")
    pose_seqs, metadata, valid_idx, comments = load_data()

    # Compute SBERT embeddings
    texts = [comments.get(i, "") for i in valid_idx]
    print(f"Computing SBERT embeddings for {len(texts)} comments...")
    sbert_emb = compute_sbert_embeddings(texts)
    embed_dim = sbert_emb.shape[1]
    print(f"  Embedding dim: {embed_dim}")

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Split data (use grade stratification)
        grades = [GRADE_MAP[metadata[i]['grade']] for i in valid_idx]
        indices = np.array(valid_idx)
        labels = np.array(grades)

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, stratify=labels, random_state=seed)
        temp_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in temp_idx])
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed)

        # Map to array positions
        idx_to_pos = {idx: pos for pos, idx in enumerate(valid_idx)}

        train_pos = [idx_to_pos[i] for i in train_idx]
        val_pos = [idx_to_pos[i] for i in val_idx]
        test_pos = [idx_to_pos[i] for i in test_idx]

        train_ds = PoseCommentDataset(train_pos, pose_seqs, sbert_emb[train_pos])
        val_ds = PoseCommentDataset(val_pos, pose_seqs, sbert_emb[val_pos])
        test_ds = PoseCommentDataset(test_pos, pose_seqs, sbert_emb[test_pos])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Create model
        model = CommentRegressor(baseline, embed_dim=embed_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_cos = eval_epoch(model, val_loader, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_cos={val_cos:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Load best and evaluate
        model.load_state_dict(best_state)
        test_loss, test_cos = eval_epoch(model, test_loader, device)

        # Also compute BLEU/ROUGE by comparing SBERT nearest-neighbor text
        test_texts = [texts[idx_to_pos[i]] for i in test_idx]
        # Simple baseline: find closest training comment by embedding similarity
        train_texts = [texts[idx_to_pos[i]] for i in train_idx]
        train_embs = sbert_emb[train_pos]

        # For each test sample, predict embedding and find nearest train comment
        refs, cands = [], []
        for i, test_i in enumerate(test_pos):
            pose_t = torch.FloatTensor(pose_seqs[valid_idx[test_i]]).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_emb = model(pose_t).cpu().numpy()[0]
            # Nearest neighbor in training set
            sims = np.dot(train_embs, pred_emb) / (np.linalg.norm(train_embs, axis=1) * np.linalg.norm(pred_emb) + 1e-8)
            nearest_idx = np.argmax(sims)
            refs.append(test_texts[i])
            cands.append(train_texts[nearest_idx])

        gen_metrics = compute_generation_metrics(refs, cands)
        print(f"    test_cos_sim={test_cos:.4f}, BLEU-1={gen_metrics['bleu1']:.4f}, "
              f"ROUGE-L={gen_metrics['rouge_l']:.4f}")

        all_seed_results.append({
            'cosine_similarity': float(test_cos),
            **gen_metrics,
        })

        # Save per-seed
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, seed_dir / 'model.pt')
        with open(seed_dir / 'results.json', 'w') as f:
            json.dump({
                'model': f'{baseline}_generation',
                'seed': seed,
                'test_cosine_similarity': float(test_cos),
                **gen_metrics,
                'embed_dim': embed_dim,
            }, f, indent=2)

    # Aggregate
    cos_vals = [r['cosine_similarity'] for r in all_seed_results]
    bleu1_vals = [r['bleu1'] for r in all_seed_results]
    rouge_vals = [r['rouge_l'] for r in all_seed_results]

    aggregated = {
        'baseline': baseline,
        'cosine_similarity': float(np.mean(cos_vals)),
        'cosine_similarity_std': float(np.std(cos_vals)),
        'bleu1': float(np.mean(bleu1_vals)),
        'bleu1_std': float(np.std(bleu1_vals)),
        'bleu4': float(np.mean([r['bleu4'] for r in all_seed_results])),
        'rouge_l': float(np.mean(rouge_vals)),
        'rouge_l_std': float(np.std(rouge_vals)),
        'bertscore': float(np.mean([r['bertscore'] for r in all_seed_results])),
        'per_seed': all_seed_results,
    }

    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n  Aggregated: cos_sim={aggregated['cosine_similarity']:.4f}, "
          f"BLEU-1={aggregated['bleu1']:.4f}, ROUGE-L={aggregated['rouge_l']:.4f}")
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    run_generation_experiment(args.baseline, args.gpu, args.epochs, args.lr)
