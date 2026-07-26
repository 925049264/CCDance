"""
VL-Transformer Generator Model

Architecture:
- STGCN motion encoder (input: SMPL 69-D poses) -> 256-D embedding
- LSTM music encoder (input: MFCC+chroma+onset 64-D features) -> 256-D embedding
- Fusion: concat(pose_emb, music_emb) -> Linear(512, 256)
- LSTM Decoder: 256-D fused embedding -> comment tokens

Two decoding modes:
1. Text generation: LSTM decoder with vocabulary
2. Teacher embedding regression: predict Sentence-BERT 768-D embeddings
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import STGCNEncoder, MusicLSTMEncoder, LSTMDecoder
from shared.config import (SMPL_POSE_DIM, TEACHER_EMBED_DIM,
                           MODEL_CONFIGS)


class VLTransformerGenerator(nn.Module):
    """VL-Transformer generation model.

    Encodes pose + audio into a fused embedding, then decodes
    into teacher comments either via LSTM text generation or
    teacher embedding regression.
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, audio_dim=64,
                 hidden_dim=256, num_joints=23, dropout=0.3,
                 vocab_size=None, max_comment_length=512,
                 teacher_embed_dim=TEACHER_EMBED_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.teacher_embed_dim = teacher_embed_dim

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

        # Fusion
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Teacher Embedding Regression Head
        # Predict Sentence-BERT 768-D embeddings from fused 256-D
        self.teacher_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, teacher_embed_dim),
        )

        # Text Decoder (LSTM with vocabulary)
        if vocab_size is not None:
            self.text_decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                num_layers=2,
                dropout=dropout,
                max_len=max_comment_length,
            )

    def encode(self, pose, audio):
        """Encode pose and audio into fused embedding.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            audio: (B, T_a, D) or (B, D) audio features

        Returns:
            dict with 'pose', 'music', 'fused' embeddings
        """
        pose_emb = self.pose_encoder(pose)  # (B, 256)

        if audio is not None:
            music_emb = self.music_encoder(audio)  # (B, 256)
            combined = torch.cat([pose_emb, music_emb], dim=-1)  # (B, 512)
        else:
            # Pose-only fallback: zero-pad music embedding
            music_emb = torch.zeros_like(pose_emb)
            combined = torch.cat([pose_emb, music_emb], dim=-1)

        fused = self.fusion_proj(combined)
        fused = self.fusion_norm(fused)
        fused = self.dropout(fused)

        return {
            'pose': pose_emb,
            'music': music_emb,
            'fused': fused,
        }

    def predict_teacher_embedding(self, fused):
        """Predict Sentence-BERT teacher embedding from fused encoding.

        Args:
            fused: (B, 256) fused embedding

        Returns:
            teacher_emb: (B, 768) predicted teacher embedding
        """
        return self.teacher_embed_head(fused)

    def forward(self, pose, audio=None, target_tokens=None,
                teacher_forcing_ratio=0.5, mode='embedding'):
        """Forward pass.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            audio: (B, T_a, D) or (B, D) audio features, optional
            target_tokens: (B, seq_len) token indices for teacher forcing
            teacher_forcing_ratio: ratio for teacher forcing (0.0-1.0)
            mode: 'embedding' for teacher embedding regression,
                  'text' for token generation,
                  'both' for both objectives

        Returns:
            dict with outputs depending on mode
        """
        enc = self.encode(pose, audio)
        fused = enc['fused']

        outputs = {'embeddings': enc}

        if mode == 'embedding' or mode == 'both':
            outputs['teacher_embed'] = self.predict_teacher_embedding(fused)

        if mode == 'text' or mode == 'both':
            if self.vocab_size is not None:
                if target_tokens is not None:
                    # Training mode with teacher forcing
                    outputs['logits'] = self.text_decoder(
                        fused, target_tokens, teacher_forcing_ratio
                    )
                else:
                    # Inference mode (greedy decoding)
                    outputs['logits'] = self.text_decoder(fused)
            else:
                raise ValueError("vocab_size must be provided for text generation mode")

        return outputs

    @torch.no_grad()
    def generate(self, pose, audio=None, max_len=512, device=None):
        """Generate comment from pose+audio input.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            audio: (B, T_a, D) or (B, D) audio features, optional
            max_len: maximum generation length
            device: torch device

        Returns:
            token_ids: (B, max_len) generated token indices
            teacher_embed: (B, 768) predicted teacher embedding
        """
        if device is not None:
            pose = pose.to(device)
            if isinstance(audio, torch.Tensor):
                audio = audio.to(device)

        enc = self.encode(pose, audio)
        fused = enc['fused']

        teacher_embed = self.predict_teacher_embedding(fused)

        if self.vocab_size is not None:
            # Override decoder max_len for generation
            orig_max_len = self.text_decoder.max_len
            self.text_decoder.max_len = max_len
            logits = self.text_decoder(fused)  # greedy decode
            self.text_decoder.max_len = orig_max_len
            token_ids = logits.argmax(dim=-1)  # (B, seq_len)
        else:
            token_ids = None

        return {
            'token_ids': token_ids,
            'teacher_embed': teacher_embed,
        }
