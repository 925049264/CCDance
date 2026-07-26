#!/usr/bin/env python3
"""
DanceMVP Classification Implementation
Two-stage self-supervised multi-modal framework:
  Stage 1: InfoNCE contrastive pre-training (pose + music)
  Stage 2: Classification fine-tuning
Runs 5 seeds and outputs mean±std results.
"""
import sys, os, json, argparse, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import (DATA_ROOT, GRADE_MAP, N_CLASSES, SEQUENCE_LENGTH,
                            BATCH_SIZE, SEEDS)
from shared.models import STGCNEncoder, MusicLSTMEncoder, MLPClassifier
from shared.metrics import compute_classification_metrics, compute_classification_metrics_mean_std

OUTPUT_DIR = Path(__file__).parent
DEVICE = None


class DanceMVPClassifier(nn.Module):
    """DanceMVP: ST-GCN pose encoder + LSTM music encoder with InfoNCE pre-training."""

    def __init__(self, pose_dim=69, audio_dim=54, hidden_dim=256, num_classes=3, dropout=0.3, temperature=0.07):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pose_encoder = STGCNEncoder(input_dim=pose_dim, hidden_dim=hidden_dim,
                                          output_dim=hidden_dim, dropout=dropout)
        self.music_encoder = MusicLSTMEncoder(input_dim=audio_dim, hidden_dim=hidden_dim // 2,
                                               output_dim=hidden_dim, dropout=dropout)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = MLPClassifier(input_dim=hidden_dim, hidden_dim=hidden_dim // 2,
                                         num_classes=num_classes, dropout=dropout)
        self.temperature = temperature

    def forward(self, pose, audio=None):
        p_emb = self.pose_encoder(pose)
        if audio is not None and audio.dim() >= 2:
            m_emb = self.music_encoder(audio)
            fused = self.fusion(torch.cat([p_emb, m_emb], dim=-1))
        else:
            fused = p_emb
        return self.classifier(fused)

    def get_embeddings(self, pose, audio=None):
        p_emb = self.pose_encoder(pose)
        if audio is not None and audio.dim() >= 2:
            m_emb = self.music_encoder(audio)
            return self.fusion(torch.cat([p_emb, m_emb], dim=-1))
        return p_emb

    def infonce_loss(self, pose_emb, music_emb):
        B = pose_emb.size(0)
        pose_emb = F.normalize(pose_emb, dim=-1)
        music_emb = F.normalize(music_emb, dim=-1)
        sim = pose_emb @ music_emb.T / self.temperature
        labels = torch.arange(B, device=pose_emb.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
        acc = (sim.argmax(-1) == labels).float().mean()
        return loss, acc.item()


def load_data():
    cache = Path(DATA_ROOT) / "experiments" / "results" / "pose_sequences.pkl"
    with open(cache, 'rb') as f:
        data = pickle.load(f)
    pose_seqs = data['pose_sequences']
    metadata = data['metadata']
    valid_idx = data['valid_indices']

    # Load audio
    audio_cache = Path(DATA_ROOT) / "experiments" / "results" / "audio_features_shared.pkl"
    if audio_cache.exists():
        with open(audio_cache, 'rb') as f:
            audio_feats = pickle.load(f)
    else:
        audio_feats = {}

    per_sample_audio = {}
    for idx in valid_idx:
        dance_id = metadata[idx]['dance_id']
        af = audio_feats.get(dance_id)
        if af is not None:
            per_sample_audio[idx] = af.astype(np.float32)
        else:
            per_sample_audio[idx] = np.zeros((100, 54), dtype=np.float32)

    return pose_seqs, metadata, valid_idx, per_sample_audio


class CCDanceDataset(Dataset):
    def __init__(self, indices, pose_seqs, audio_feats, labels, max_len=SEQUENCE_LENGTH):
        self.indices = indices
        self.pose_seqs = pose_seqs
        self.audio_feats = audio_feats
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        pose = self.pose_seqs[real_idx].astype(np.float32)
        audio = self.audio_feats.get(real_idx, np.zeros((100, 54), dtype=np.float32))
        label = self.labels[idx]

        T = pose.shape[0]
        if T > self.max_len:
            indices = np.linspace(0, T - 1, self.max_len, dtype=int)
            pose = pose[indices]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        # Handle audio
        if isinstance(audio, np.ndarray):
            if audio.ndim == 2:
                if audio.shape[0] > 30:
                    idx_a = np.linspace(0, audio.shape[0] - 1, 30, dtype=int)
                    audio = audio[idx_a]
                audio_t = torch.FloatTensor(audio)
            else:
                audio_t = torch.FloatTensor(audio)
        else:
            audio_t = torch.zeros(30, 54)

        return torch.FloatTensor(pose), audio_t, torch.LongTensor([label])[0]


def pretrain_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for pose, audio, _ in loader:
        pose, audio = pose.to(device), audio.to(device)
        p_emb = model.pose_encoder(pose)
        m_emb = model.music_encoder(audio)
        loss, acc = model.infonce_loss(p_emb, m_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pose.size(0)
        total_acc += acc * pose.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


def train_cls_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for pose, audio, labels in loader:
        pose, audio, labels = pose.to(device), audio.to(device), labels.to(device)
        logits = model(pose, audio)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pose.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += pose.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for pose, audio, labels in loader:
        pose, audio, labels = pose.to(device), audio.to(device), labels.to(device)
        logits = model(pose, audio)
        loss = criterion(logits, labels)
        total_loss += loss.item() * pose.size(0)
        probs = F.softmax(logits, -1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    metrics = compute_classification_metrics(all_labels, all_preds, np.concatenate(all_probs))
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


def run_seed(seed, device, pose_seqs, metadata, valid_idx, audio_feats, epochs=50, pretrain_epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split
    grades = [GRADE_MAP[metadata[i]['grade']] for i in valid_idx]
    indices = np.array(valid_idx)
    labels = np.array(grades)
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=seed)
    temp_l = np.array([GRADE_MAP[metadata[i]['grade']] for i in temp_idx])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_l, random_state=seed)

    train_l = np.array([GRADE_MAP[metadata[i]['grade']] for i in train_idx])
    val_l = np.array([GRADE_MAP[metadata[i]['grade']] for i in val_idx])
    test_l = np.array([GRADE_MAP[metadata[i]['grade']] for i in test_idx])

    train_ds = CCDanceDataset(train_idx, pose_seqs, audio_feats, train_l)
    val_ds = CCDanceDataset(val_idx, pose_seqs, audio_feats, val_l)
    test_ds = CCDanceDataset(test_idx, pose_seqs, audio_feats, test_l)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DanceMVPClassifier().to(device)
    criterion = nn.CrossEntropyLoss()

    # Stage 1: InfoNCE pre-training
    pretrain_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(pretrain_epochs):
        loss, acc = pretrain_epoch(model, train_loader, pretrain_opt, device)

    # Stage 2: Classification fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        train_loss, train_acc = train_cls_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_metrics = eval_epoch(model, test_loader, criterion, device)

    # Save
    seed_dir = OUTPUT_DIR / f'seed_{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, seed_dir / 'model.pt')
    with open(seed_dir / 'results.json', 'w') as f:
        json.dump({'model': 'DanceMVP', 'seed': seed, 'test_metrics': test_metrics}, f, indent=2)

    return test_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pretrain_epochs', type=int, default=30)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pose_seqs, metadata, valid_idx, audio_feats = load_data()
    print(f"Loaded {len(valid_idx)} samples")

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}")
        res = run_seed(seed, device, pose_seqs, metadata, valid_idx, audio_feats,
                       epochs=args.epochs, pretrain_epochs=args.pretrain_epochs)
        all_results.append(res)
        print(f"  Test: acc={res['accuracy']:.4f}, f1={res['macro_f1']:.4f}, qwk={res['qwk']:.4f}")

    # Aggregate
    agg = compute_classification_metrics_mean_std(all_results)
    with open(OUTPUT_DIR / 'aggregated_results.json', 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*50}")
    print(f"DanceMVP Aggregated: acc={agg['accuracy']:.4f}±{agg['accuracy_std']:.4f}, "
          f"f1={agg['macro_f1']:.4f}±{agg['macro_f1_std']:.4f}, "
          f"qwk={agg['qwk']:.4f}±{agg['qwk_std']:.4f}")
