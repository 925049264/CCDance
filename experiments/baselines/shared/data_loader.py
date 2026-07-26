"""
Unified CCDance data loader for all baselines.
Loads SMPL pose sequences, audio features, and teacher evaluations.
"""
import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_loader import CCDanceDataset, extract_motion_features

from .config import (DATA_ROOT, GRADE_MAP, SEQUENCE_LENGTH, SEEDS,
                     TRAIN_RATIO, VAL_RATIO, TEST_RATIO, N_CLASSES)


class CCDancePoseDataset(Dataset):
    """Dataset for SMPL pose sequence input (used by all baselines)."""

    def __init__(self, sample_indices, pose_sequences, labels, grades,
                 max_len=SEQUENCE_LENGTH):
        self.sample_indices = sample_indices
        self.pose_sequences = pose_sequences  # dict: idx -> (T, 69)
        self.labels = labels  # numpy array of int labels
        self.grades = grades
        self.max_len = max_len

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]
        pose = self.pose_sequences[real_idx].astype(np.float32)
        label = self.labels[idx]

        T = pose.shape[0]
        if T > self.max_len:
            indices = np.linspace(0, T - 1, self.max_len, dtype=int)
            pose = pose[indices]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        return torch.FloatTensor(pose), torch.LongTensor([label])[0], T


class CCDancePoseAudioDataset(Dataset):
    """Dataset for SMPL pose + audio features (VL-Transformer, etc.)."""

    def __init__(self, sample_indices, pose_sequences, audio_features,
                 labels, grades, max_len=SEQUENCE_LENGTH,
                 audio_dim=64):
        self.sample_indices = sample_indices
        self.pose_sequences = pose_sequences
        self.audio_features = audio_features  # dict: idx -> (T_audio, D_audio)
        self.labels = labels
        self.grades = grades
        self.max_len = max_len
        self.audio_dim = audio_dim

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]
        pose = self.pose_sequences[real_idx].astype(np.float32)
        audio = self.audio_features.get(real_idx)
        if audio is None:
            audio = np.zeros((1, self.audio_dim), dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        label = self.labels[idx]

        T = pose.shape[0]
        if T > self.max_len:
            indices = np.linspace(0, T - 1, self.max_len, dtype=int)
            pose = pose[indices]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        # Handle audio: align temporal dimension or use global mean
        if audio.ndim == 2 and audio.shape[0] > 1:
            if audio.shape[0] > self.max_len // 10:
                audio_idx = np.linspace(0, audio.shape[0] - 1, self.max_len // 10, dtype=int)
                audio = audio[audio_idx]
        else:
            audio = audio.flatten()  # global audio feature vector

        return (torch.FloatTensor(pose),
                torch.FloatTensor(audio) if audio.ndim == 2 else torch.FloatTensor(audio),
                torch.LongTensor([label])[0],
                T)


class CCDanceGenerationDataset(Dataset):
    """Dataset for teacher comment generation task."""

    def __init__(self, sample_indices, pose_sequences, comments,
                 tokenizer=None, max_pose_len=SEQUENCE_LENGTH,
                 max_comment_len=512):
        self.sample_indices = sample_indices
        self.pose_sequences = pose_sequences
        self.comments = comments  # list of text strings
        self.max_pose_len = max_pose_len
        self.max_comment_len = max_comment_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        real_idx = self.sample_indices[idx]
        pose = self.pose_sequences[real_idx].astype(np.float32)
        comment = self.comments[idx] if self.comments[idx] else ""

        T = pose.shape[0]
        if T > self.max_pose_len:
            indices = np.linspace(0, T - 1, self.max_pose_len, dtype=int)
            pose = pose[indices]
        elif T < self.max_pose_len:
            pad = np.zeros((self.max_pose_len - T, pose.shape[1]), dtype=np.float32)
            pose = np.concatenate([pose, pad], axis=0)

        # Return raw comment text; tokenization is handled by each baseline's training loop
        return torch.FloatTensor(pose), comment, T


def load_all_pose_sequences(data_root=DATA_ROOT, use_cache=True):
    """Load all SMPL pose sequences from the dataset.
    Returns: pose_sequences dict, metadata dict, valid_indices list.
    """
    cache_file = Path(data_root) / "experiments" / "results" / "pose_sequences.pkl"

    if use_cache and cache_file.exists():
        print(f"Loading cached pose sequences from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['pose_sequences'], data['metadata'], data['valid_indices']

    print("Loading dataset and extracting pose sequences...")
    dataset = CCDanceDataset(data_root)
    pose_sequences = {}
    metadata = {}
    valid_indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        smpl = dataset.load_smpl(sample)
        if smpl is not None:
            pose_sequences[idx] = smpl['poses'].astype(np.float32)
            metadata[idx] = {
                'dance_id': sample['dance_id'],
                'grade': sample['grade'],
                'category': sample['category'],
                'student_name': sample['student_name'],
            }
            valid_indices.append(idx)

    print(f"Loaded {len(valid_indices)} valid samples")

    # Cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'pose_sequences': pose_sequences,
            'metadata': metadata,
            'valid_indices': valid_indices,
        }, f)

    return pose_sequences, metadata, valid_indices


def load_teacher_comments(data_root=DATA_ROOT):
    """Load teacher evaluation comments (English) for all samples."""
    dataset = CCDanceDataset(data_root)
    comments = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        eval_data = dataset.load_teacher_eval(sample)
        if eval_data and 'en_text' in eval_data:
            comments[idx] = eval_data['en_text']
        else:
            comments[idx] = ""
    return comments


def extract_audio_features(data_root=DATA_ROOT, use_cache=True):
    """Extract audio features for all 22 dance genres.
    Returns dict mapping dance_id -> audio feature array.
    """
    import librosa
    cache_file = Path(data_root) / "experiments" / "results" / "audio_features_shared.pkl"

    if use_cache and cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("Extracting audio features...")
    audio_features = {}
    for dance_id in range(1, 23):
        mp3_path = Path(data_root) / str(dance_id) / f"{dance_id}.mp3"
        if not mp3_path.exists():
            continue

        try:
            y, sr = librosa.load(str(mp3_path), sr=22050)
            hop = 512
            n_frames = len(y) // hop + 1

            # MFCC: (20, T)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
            n_frames = mfcc.shape[1]

            # MFCC delta: (20, T)
            mfcc_delta = librosa.feature.delta(mfcc)

            # Chroma: (12, T)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_chroma=12)

            # Onset strength: (T,)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
            # Pad/trim onset_env to match MFCC frames
            if len(onset_env) < n_frames:
                onset_env = np.pad(onset_env, (0, n_frames - len(onset_env)))
            else:
                onset_env = onset_env[:n_frames]
            onset_strength = onset_env.reshape(1, -1)  # (1, T)

            # Tempogram: (n_freq, T_tempo) — use tempogram ratio instead for fixed dim
            # Use a simplified onset-based feature to avoid dimension mismatch
            onset_smooth = np.convolve(onset_env, np.ones(5)/5, mode='same').reshape(1, -1)
            onset_smooth = onset_smooth[:, :n_frames]

            # Concatenate all features along feature axis (axis=0)
            # Result should be (D, T) where D = 20+20+12+1+1 = 54
            features = np.concatenate([
                mfcc[:, :n_frames],
                mfcc_delta[:, :n_frames],
                chroma[:, :n_frames],
                onset_strength[:, :n_frames],
                onset_smooth[:, :n_frames],
            ], axis=0)  # (54, T)

            # Transpose to (T, D) format
            audio_features[dance_id] = features.T.astype(np.float32)  # (T, 54)
        except Exception as e:
            print(f"  Warning: Failed to extract audio for dance {dance_id}: {e}")
            # Create a valid default feature with correct shape
            audio_features[dance_id] = np.zeros((100, 54), dtype=np.float32)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(audio_features, f)

    return audio_features


def build_audio_features_per_sample(valid_indices, metadata, audio_features):
    """Create per-sample audio feature lookup from dance-level audio features."""
    per_sample_audio = {}
    for idx in valid_indices:
        dance_id = metadata[idx]['dance_id']
        per_sample_audio[idx] = audio_features.get(dance_id, np.zeros((1, 64), dtype=np.float32))
    return per_sample_audio


def create_data_splits(valid_indices, metadata, seed=42):
    """Create train/val/test splits stratified by grade.
    Returns: train_idx, val_idx, test_idx lists.
    """
    grades = [GRADE_MAP[metadata[i]['grade']] for i in valid_indices]
    indices = np.array(valid_indices)
    labels = np.array(grades)

    # Stratified split: 70/15/15
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels, random_state=seed
    )
    temp_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in temp_idx])
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        stratify=temp_labels, random_state=seed
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_dataloaders(pose_sequences, metadata, train_idx, val_idx, test_idx,
                       batch_size=16, audio_features=None):
    """Create DataLoader objects for classification task."""
    train_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in train_idx])
    val_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in val_idx])
    test_labels = np.array([GRADE_MAP[metadata[i]['grade']] for i in test_idx])

    if audio_features is not None:
        train_dataset = CCDancePoseAudioDataset(
            train_idx, pose_sequences, audio_features,
            train_labels, [metadata[i]['grade'] for i in train_idx]
        )
        val_dataset = CCDancePoseAudioDataset(
            val_idx, pose_sequences, audio_features,
            val_labels, [metadata[i]['grade'] for i in val_idx]
        )
        test_dataset = CCDancePoseAudioDataset(
            test_idx, pose_sequences, audio_features,
            test_labels, [metadata[i]['grade'] for i in test_idx]
        )
    else:
        train_dataset = CCDancePoseDataset(
            train_idx, pose_sequences, train_labels,
            [metadata[i]['grade'] for i in train_idx]
        )
        val_dataset = CCDancePoseDataset(
            val_idx, pose_sequences, val_labels,
            [metadata[i]['grade'] for i in val_idx]
        )
        test_dataset = CCDancePoseDataset(
            test_idx, pose_sequences, test_labels,
            [metadata[i]['grade'] for i in test_idx]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    pose_sequences, metadata, valid_indices = load_all_pose_sequences()
    print(f"Loaded {len(pose_sequences)} pose sequences")
    print(f"Sample pose shape: {pose_sequences[valid_indices[0]].shape}")

    train_idx, val_idx, test_idx = create_data_splits(valid_indices, metadata, seed=42)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
