"""
VL-Transformer Classifier Model

Architecture (Chen, SciRep 2025):
- STGCN motion encoder (input: SMPL 69-D poses) -> 256-D embedding
- LSTM music encoder (input: MFCC+chroma+onset 64-D features) -> 256-D embedding
- Fusion: concat(pose_emb, music_emb) -> Linear(512, 256) -> MLP(256, 128, 3)

Supports InfoNCE contrastive pretraining and classification fine-tuning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import STGCNEncoder, MusicLSTMEncoder, MLPClassifier
from shared.config import (SMPL_POSE_DIM, N_CLASSES,
                           AUDIO_FEATURE_DIM, MODEL_CONFIGS)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for aligning pose and music embeddings.

    Given a batch of N pose-music pairs, the loss encourages
    matching pairs to have high cosine similarity and non-matching
    pairs to have low similarity.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pose_emb, music_emb):
        """Compute InfoNCE loss.

        Args:
            pose_emb: (B, D) pose embeddings
            music_emb: (B, D) music embeddings

        Returns:
            loss: scalar InfoNCE loss
            acc: matching accuracy
        """
        B = pose_emb.size(0)
        # Normalize embeddings
        pose_emb = F.normalize(pose_emb, dim=-1)
        music_emb = F.normalize(music_emb, dim=-1)

        # Similarity matrix: (B, B)
        sim = pose_emb @ music_emb.T / self.temperature

        # Symmetric InfoNCE: pose2music and music2pose
        labels = torch.arange(B, device=pose_emb.device)

        loss_p2m = self.criterion(sim, labels)
        loss_m2p = self.criterion(sim.T, labels)

        loss = (loss_p2m + loss_m2p) / 2.0

        # Matching accuracy
        with torch.no_grad():
            p2m_pred = sim.argmax(dim=-1)
            m2p_pred = sim.T.argmax(dim=-1)
            acc = ((p2m_pred == labels).float().mean() +
                   (m2p_pred == labels).float().mean()).item() / 2.0

        return loss, acc


class VLTransformerClassifier(nn.Module):
    """VL-Transformer classification model.

    Pose sequence -> STGCNEncoder -> 256-D
    Audio features -> MusicLSTMEncoder -> 256-D
    Concat(512-D) -> Linear(512, 256) -> MLP(256, 128, 3)
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, audio_dim=AUDIO_FEATURE_DIM,
                 hidden_dim=256, num_classes=N_CLASSES, dropout=0.3,
                 temperature=0.07, num_joints=23):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Encoders
        self.pose_encoder = STGCNEncoder(
            input_dim=pose_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_joints=num_joints,
            dropout=dropout,
        )
        self.music_encoder = MusicLSTMEncoder(
            input_dim=audio_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Fusion: concat(pose_emb, music_emb) -> 512 -> 256
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # InfoNCE contrastive loss
        self.infonce_loss = InfoNCELoss(temperature=temperature)

        # Classification head: 256 -> 128 -> 3
        self.classifier = MLPClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=2,
        )

    def encode_pose(self, pose):
        """Encode pose sequence to embedding.

        Args:
            pose: (B, T, 69) SMPL pose sequence

        Returns:
            pose_emb: (B, 256) pose embedding
        """
        return self.pose_encoder(pose)

    def encode_music(self, audio):
        """Encode audio features to embedding.

        Args:
            audio: (B, T_a, D_a) or (B, D_a) audio features

        Returns:
            music_emb: (B, 256) music embedding
        """
        return self.music_encoder(audio)

    def fuse(self, pose_emb, music_emb):
        """Fuse pose and music embeddings.

        Args:
            pose_emb: (B, 256)
            music_emb: (B, 256)

        Returns:
            fused: (B, 256) fused embedding
        """
        combined = torch.cat([pose_emb, music_emb], dim=-1)  # (B, 512)
        fused = self.fusion_proj(combined)
        return fused

    def forward(self, pose, audio=None):
        """Forward pass for classification.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            audio: (B, T_a, D_a) or (B, D_a) audio features, optional

        Returns:
            logits: (B, 3) classification logits
        """
        pose_emb = self.encode_pose(pose)

        if audio is not None:
            music_emb = self.encode_music(audio)
            fused = self.fuse(pose_emb, music_emb)
        else:
            # Pose-only fallback (should not happen in VL-Transformer)
            fused = pose_emb

        return self.classifier(fused)

    def compute_contrastive_loss(self, pose_emb, music_emb):
        """Compute InfoNCE contrastive loss between pose and music embeddings.

        Args:
            pose_emb: (B, 256) pose embeddings
            music_emb: (B, 256) music embeddings

        Returns:
            loss: scalar InfoNCE loss
            acc: matching accuracy
        """
        return self.infonce_loss(pose_emb, music_emb)

    def get_embeddings(self, pose, audio):
        """Get fused embeddings for a batch.

        Used for feature extraction and downstream tasks.
        """
        pose_emb = self.encode_pose(pose)
        music_emb = self.encode_music(audio)
        fused = self.fuse(pose_emb, music_emb)
        return {
            'pose': pose_emb,
            'music': music_emb,
            'fused': fused,
        }
