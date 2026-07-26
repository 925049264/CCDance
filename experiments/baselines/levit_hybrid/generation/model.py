"""
LeViT-Hybrid Generation Model

Adapts the LeViT-Hybrid architecture for teacher comment generation.
Uses the LeViT-style pose patch embedding and Transformer encoder as a
motion feature extractor, feeding into an LSTM-based decoder for
teacher evaluation text generation.

Architecture:
  1. PosePatchEmbedding: (B, T=300, D=69) -> (B, N+1=21, 256)
  2. LeViT-style TransformerEncoder: (B, 21, 256) -> CLS token (B, 256)
  3. LSTM Decoder: (B, 256) -> generated comment tokens

This mirrors the approach used in other CCDance generation baselines
(e.g., STGCN + LSTMDecoder), but replaces the motion encoder with
the LeViT-Hybrid's patch-based Transformer encoder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.config import (SMPL_POSE_DIM, N_CLASSES,
                           SEQUENCE_LENGTH, MAX_COMMENT_LENGTH,
                           GRADE_MAP)


class PosePatchEmbedding(nn.Module):
    """Pose patch embedding for generation (same as classification model).

    Divides the 300-frame SMPL pose sequence into 20 patches of 15 frames.
    Each patch (69*15=1035-D) is linearly projected to embed_dim.
    A CLS token and positional embeddings are added.
    """

    def __init__(self, seq_length=SEQUENCE_LENGTH, pose_dim=SMPL_POSE_DIM,
                 patch_size=15, embed_dim=256, dropout=0.1):
        super().__init__()
        self.seq_length = seq_length
        self.pose_dim = pose_dim
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.embed_dim = embed_dim

        assert seq_length % patch_size == 0, \
            f"Sequence length {seq_length} must be divisible by patch_size {patch_size}"

        self.proj = nn.Linear(pose_dim * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Create patch embeddings.

        Args:
            x: (B, T, 69) SMPL pose sequence

        Returns:
            embeddings: (B, N+1, embed_dim) patch + CLS token embeddings
        """
        B = x.shape[0]

        if x.size(1) != self.seq_length:
            if x.size(1) > self.seq_length:
                start = (x.size(1) - self.seq_length) // 2
                x = x[:, start:start + self.seq_length, :]
            else:
                pad_len = self.seq_length - x.size(1)
                pad = torch.zeros(B, pad_len, self.pose_dim,
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)

        x = x.reshape(B, self.num_patches, self.patch_size * self.pose_dim)
        x = self.proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        return self.dropout(x)


class LeViTEncoder(nn.Module):
    """LeViT-style Transformer encoder for pose patches.

    Processes patch embeddings with a Transformer encoder and returns
    the CLS token representation.
    """

    def __init__(self, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Process patch embeddings.

        Args:
            x: (B, N+1, d_model)

        Returns:
            cls_out: (B, d_model) CLS token representation
        """
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]
        return cls_out


class PoseLSTMDecoder(nn.Module):
    """LSTM decoder for comment generation from pose embeddings.

    This is a self-contained replacement for shared.models.LSTMDecoder
    that correctly handles bidirectional hidden state initialization.

    Architecture:
      - Token embedding -> LSTM (bidirectional) -> Linear projection
      - Encoder output used as initial hidden state
      - Supports teacher forcing during training and greedy decoding at inference
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim * 2, vocab_size)
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def _init_hidden(self, encoder_output, device):
        """Initialize LSTM hidden state from encoder output.

        Args:
            encoder_output: (B, hidden_dim) pose embedding
            device: torch device

        Returns:
            h0: (num_layers * 2, B, hidden_dim) initial hidden state
            c0: (num_layers * 2, B, hidden_dim) initial cell state
        """
        B = encoder_output.size(0)
        # Repeat encoder output for each layer and direction
        h0 = encoder_output.unsqueeze(0).repeat(self.num_layers * 2, 1, 1)
        c0 = torch.zeros_like(h0)
        return h0.contiguous(), c0.contiguous()

    def forward(self, encoder_output, target_tokens=None,
                teacher_forcing_ratio=0.5):
        """Forward pass with optional teacher forcing.

        Args:
            encoder_output: (B, hidden_dim) pose embedding
            target_tokens: (B, seq_len) target token indices (training)
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            logits: (B, seq_len, vocab_size) or (B, max_len, vocab_size)
        """
        B = encoder_output.size(0)
        device = encoder_output.device

        if target_tokens is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Training with teacher forcing
            embedded = self.dropout(self.embed(target_tokens))
            h0, c0 = self._init_hidden(encoder_output, device)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            return self.output_proj(lstm_out)
        else:
            # Inference or non-teacher-forcing training: greedy decoding
            h0, c0 = self._init_hidden(encoder_output, device)
            hidden = (h0, c0)

            # Start token (SOS=1, assuming 0=PAD)
            input_token = torch.ones(B, 1, dtype=torch.long, device=device)
            outputs = []

            for _ in range(self.max_len):
                embedded = self.dropout(self.embed(input_token))
                lstm_out, hidden = self.lstm(embedded, hidden)
                logits = self.output_proj(lstm_out)  # (B, 1, V)
                outputs.append(logits)
                input_token = logits.argmax(dim=-1)  # (B, 1)

            return torch.cat(outputs, dim=1)  # (B, max_len, V)


class LeViTHybridGenerator(nn.Module):
    """LeViT-Hybrid generation model for teacher comments.

    Encodes SMPL pose sequences with a LeViT-style patch-based Transformer
    and generates teacher evaluation text with an LSTM decoder.

    Architecture:
      - PosePatchEmbedding: (B, T, 69) -> patch embeddings
      - LeViTEncoder: patch embeddings -> CLS representation
      - LSTMDecoder: CLS representation -> token sequence
    """

    def __init__(self, vocab_size, pose_dim=SMPL_POSE_DIM,
                 seq_length=SEQUENCE_LENGTH, patch_size=15,
                 embed_dim=256, nhead=8, num_layers=4,
                 mlp_ratio=2, dropout=0.1,
                 decoder_embed_dim=256, decoder_hidden_dim=512,
                 decoder_num_layers=2,
                 max_comment_length=MAX_COMMENT_LENGTH):
        """Initialize the LeViT-Hybrid generator.

        Args:
            vocab_size: Size of the tokenizer vocabulary
            pose_dim: SMPL pose dimension (69)
            seq_length: Number of frames (300)
            patch_size: Frames per patch (15)
            embed_dim: Embedding dimension for patch projection
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: Feedforward dimension ratio
            dropout: Dropout rate
            decoder_embed_dim: Embedding dimension for decoder
            decoder_hidden_dim: Hidden dimension for decoder LSTM
            decoder_num_layers: Number of decoder LSTM layers
            max_comment_length: Maximum generation length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_comment_length = max_comment_length

        # Pose patch embedding
        self.patch_embed = PosePatchEmbedding(
            seq_length=seq_length,
            pose_dim=pose_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # LeViT-style encoder
        self.encoder = LeViTEncoder(
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
        )

        # Project encoder output to decoder hidden dimension
        # LSTMDecoder uses encoder_output as initial hidden state,
        # which must match decoder_hidden_dim (from shared.models.LSTMDecoder)
        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_hidden_dim)

        # LSTM Decoder for text generation
        # Uses a self-contained implementation with correct bidirectional
        # hidden state initialization, avoiding a bug in shared.models.LSTMDecoder
        self.decoder = PoseLSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=decoder_embed_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            max_len=max_comment_length,
        )

    def encode_pose(self, pose):
        """Encode pose sequence to a fixed-size embedding.

        Args:
            pose: (B, T, 69) SMPL pose sequence

        Returns:
            pose_emb: (B, decoder_hidden_dim) pose embedding for decoder
        """
        patches = self.patch_embed(pose)        # (B, N+1, embed_dim)
        cls_out = self.encoder(patches)          # (B, embed_dim)
        pose_emb = self.encoder_to_decoder(cls_out)  # (B, decoder_hidden_dim)
        return pose_emb

    def forward(self, pose, target_tokens=None, teacher_forcing_ratio=0.5):
        """Forward pass for comment generation.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            target_tokens: (B, seq_len) target token indices (for training)
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            If target_tokens is provided:
                logits: (B, seq_len, vocab_size)
            Otherwise:
                logits: (B, max_len, vocab_size) from greedy decoding
        """
        pose_emb = self.encode_pose(pose)  # (B, decoder_hidden_dim)
        return self.decoder(pose_emb, target_tokens, teacher_forcing_ratio)

    def get_embeddings(self, pose):
        """Extract pose embeddings for downstream tasks.

        Args:
            pose: (B, T, 69)

        Returns:
            dict with 'pose_emb' key
        """
        return {'pose_emb': self.encode_pose(pose)}
