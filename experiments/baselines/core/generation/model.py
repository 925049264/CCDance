"""
CoRe Generation Model

Architecture:
- PoseLSTMEncoder backbone (256-D output)
- Grade embedding (3-D one-hot -> 256-D) injected into the decoder
- LSTM decoder for teacher comment generation with teacher forcing
- The grade-conditioning allows generating different-quality comments
  for different performance levels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import PoseLSTMEncoder
from shared.config import (SMPL_POSE_DIM, N_CLASSES, MODEL_CONFIGS,
                           MAX_COMMENT_LENGTH)


class CoReGenerator(nn.Module):
    """CoRe Generation model with grade-conditioned decoding.

    Encodes a pose sequence and generates a teacher comment, conditioned
    on the grade (A/B/C). The grade embedding is concatenated with the
    pose embedding to form the decoder initial state.

    Uses its own LSTM decoder (rather than shared LSTMDecoder) to avoid
    hidden-state dimension bugs in the shared module (bidirectional vs
    unidirectional mismatch).

    Args:
        vocab_size: Size of the token vocabulary.
        pose_dim: SMPL pose dimension (69).
        hidden_dim: Feature dimension (256).
        decoder_dim: LSTM decoder hidden dimension (512).
        num_layers: Number of LSTM decoder layers.
        num_classes: Number of grade classes (3).
        max_len: Maximum generation length.
        dropout: Dropout rate.
    """

    def __init__(self, vocab_size, pose_dim=SMPL_POSE_DIM, hidden_dim=256,
                 decoder_dim=512, num_layers=2, num_classes=N_CLASSES,
                 max_len=MAX_COMMENT_LENGTH, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Pose backbone
        self.backbone = PoseLSTMEncoder(
            input_dim=pose_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Grade embedding
        self.grade_embed = nn.Embedding(num_classes, hidden_dim)

        # Project pose + grade to decoder initial state
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, decoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Embedding layer for decoder tokens
        self.token_embed = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        # Unidirectional LSTM decoder
        self.lstm = nn.LSTM(
            256, decoder_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Linear(decoder_dim, vocab_size)

    def forward(self, pose, grade, target_tokens=None,
                teacher_forcing_ratio=0.5):
        """Forward pass for generation.

        Args:
            pose: (B, T, 69) SMPL pose sequence.
            grade: (B,) integer grade labels (0=A, 1=B, 2=C).
            target_tokens: (B, seq_len) target token indices for training.
            teacher_forcing_ratio: Probability of using teacher forcing.

        Returns:
            logits: (B, seq_len, vocab_size) output logits.
        """
        device = pose.device
        B = pose.size(0)

        # Encode pose
        pose_emb = self.backbone(pose)  # (B, hidden_dim)

        # Grade embedding
        grade_emb = self.grade_embed(grade)  # (B, hidden_dim)

        # Concatenate pose and grade embeddings
        combined = torch.cat([pose_emb, grade_emb], dim=-1)  # (B, 2*hidden_dim)

        # Project to decoder dimension -> initial hidden state
        decoder_init = self.state_proj(combined)  # (B, decoder_dim)

        h0 = decoder_init.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, H)
        c0 = torch.zeros_like(h0)

        if target_tokens is not None and self.training:
            seq_len = target_tokens.size(1)
            embedded = self.embed_dropout(self.token_embed(target_tokens))  # (B, S, 256)
            lstm_out, _ = self.lstm(embedded, (h0, c0))                     # (B, S, H)
            return self.output_proj(lstm_out)                                # (B, S, V)
        else:
            # Greedy decoding (inference or teacher_forcing_ratio=0)
            return self._greedy_decode(h0, c0, B, device)

    @torch.no_grad()
    def _greedy_decode(self, h0, c0, B, device):
        """Greedy decoding loop.

        Args:
            h0: (num_layers, B, decoder_dim) initial hidden state.
            c0: (num_layers, B, decoder_dim) initial cell state.
            B: Batch size.
            device: torch device.

        Returns:
            logits: (B, max_len, vocab_size) output logits.
        """
        hidden = (h0, c0)
        input_token = torch.ones(B, 1, dtype=torch.long, device=device)  # <SOS>=1
        outputs = []

        for _ in range(self.max_len):
            embedded = self.embed_dropout(self.token_embed(input_token))  # (B, 1, 256)
            lstm_out, hidden = self.lstm(embedded, hidden)                # (B, 1, H)
            logits = self.output_proj(lstm_out)                           # (B, 1, V)
            outputs.append(logits)
            input_token = logits.argmax(dim=-1)                           # (B, 1)

        return torch.cat(outputs, dim=1)  # (B, max_len, V)

    @torch.no_grad()
    def generate(self, pose, grade, device):
        """Generate a comment greedily.

        Args:
            pose: (B, T, 69) SMPL pose sequence.
            grade: (B,) integer grade labels.
            device: torch device.

        Returns:
            token_ids: (B, max_len) generated token indices.
            logits: (B, max_len, vocab_size) output logits.
        """
        self.eval()
        logits = self.forward(pose, grade, target_tokens=None)
        token_ids = logits.argmax(dim=-1)
        return token_ids, logits

    def get_embedding(self, pose, grade):
        """Get the combined pose+grade embedding.

        Args:
            pose: (B, T, 69) SMPL pose sequence.
            grade: (B,) integer grade labels.

        Returns:
            (B, decoder_dim) combined embedding.
        """
        pose_emb = self.backbone(pose)
        grade_emb = self.grade_embed(grade)
        combined = torch.cat([pose_emb, grade_emb], dim=-1)
        return self.state_proj(combined)
