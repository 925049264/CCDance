"""
Graph-Transformer / X-DANCENET Generation Model

Architecture (Han et al., SciRep 2026 - adapted for comment generation):
- GraphTransformerEncoder: spatial-temporal dual attention over SMPL joints
  - Reshape 69-D SMPL to (23 joints x 3 DoF)
  - Joint embedding (3 -> d_model=256)
  - Spatial attention (cross-joint) + Temporal attention (cross-time)
  - Global mean pooling -> (B, 256)
- LSTMDecoder: 2-layer LSTM with teacher forcing
  - Encoder output initializes LSTM hidden state
  - Token embedding -> LSTM -> projection -> vocab logits

Simplifications vs. original X-DANCENET:
- No sensor normalization (ASN/Kalman/Butterworth) - SMPL data is clean
- No prototype-based explanation layer
- Standard LSTM decoder instead of Transformer-based decoder
- Built-in word-level tokenizer from training corpus
"""
import sys
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import GraphTransformerEncoder, LSTMDecoder
from shared.config import (SMPL_POSE_DIM, N_CLASSES,
                           MODEL_CONFIGS, MAX_COMMENT_LENGTH)


# Special tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']


class WordTokenizer:
    """Simple word-level tokenizer built from a text corpus.

    Handles vocabulary construction, text-to-ids, and ids-to-text.
    """

    def __init__(self, max_vocab_size=5000):
        self.word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
        self.vocab_size = len(SPECIAL_TOKENS)
        self.max_vocab_size = max_vocab_size

    def fit(self, texts):
        """Build vocabulary from a list of text strings."""
        counter = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)

        # Add most common words up to max_vocab_size
        for word, _ in counter.most_common(self.max_vocab_size - len(SPECIAL_TOKENS)):
            if word not in self.word2idx:
                idx = self.vocab_size
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size += 1

        print(f"Vocabulary size: {self.vocab_size}")

    def _tokenize(self, text):
        """Tokenize text into words (lowercase, split on non-alphanumeric)."""
        text = text.lower().strip()
        return re.findall(r"[a-z0-9']+|[.,!?;:\"()]", text)

    def encode(self, text, max_len=512):
        """Convert text to token IDs with SOS/EOS/padding.

        Args:
            text: input string
            max_len: maximum sequence length

        Returns:
            token_ids: (max_len,) long tensor of token IDs
        """
        tokens = self._tokenize(text)
        ids = [SOS_TOKEN]
        for tok in tokens[:max_len - 2]:
            ids.append(self.word2idx.get(tok, UNK_TOKEN))
        ids.append(EOS_TOKEN)

        # Pad to max_len
        if len(ids) < max_len:
            ids.extend([PAD_TOKEN] * (max_len - len(ids)))
        else:
            ids = ids[:max_len]

        return torch.LongTensor(ids)

    def decode(self, ids, skip_special=True):
        """Convert token IDs back to text string.

        Args:
            ids: list or tensor of token IDs
            skip_special: whether to skip <PAD>, <SOS>, <EOS>, <UNK>

        Returns:
            text: decoded string
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = []
        for idx in ids:
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special and word in SPECIAL_TOKENS:
                if word == '<EOS>':
                    break
                continue
            tokens.append(word)
        return ' '.join(tokens)


class GraphTransformerGenerator(nn.Module):
    """Graph-Transformer generation model.

    Pose sequence -> GraphTransformerEncoder -> (B, 256)
    -> LSTMDecoder -> (B, max_len, vocab_size) logits.
    """

    def __init__(self, vocab_size, pose_dim=SMPL_POSE_DIM,
                 d_model=256, nhead=8, num_layers=4, num_joints=23,
                 decoder_hidden_dim=512, decoder_layers=2,
                 dropout=0.1, max_len=MAX_COMMENT_LENGTH):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size

        # Graph-Transformer encoder
        self.encoder = GraphTransformerEncoder(
            input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_joints=num_joints,
            dropout=dropout,
        )

        # LSTM decoder for comment generation
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=d_model,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, pose, target_tokens=None, teacher_forcing_ratio=0.5):
        """Forward pass for generation.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            target_tokens: (B, seq_len) target token IDs (training only)
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            logits: (B, seq_len, vocab_size) if target_tokens provided
                    (B, max_len, vocab_size) if greedy decoding
        """
        # Encode pose sequence
        pose_emb = self.encoder(pose)  # (B, d_model)

        # Decode with or without teacher forcing
        logits = self.decoder(
            encoder_output=pose_emb,
            target_tokens=target_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return logits

    @torch.no_grad()
    def generate(self, pose, device):
        """Generate a comment from a pose sequence (greedy decoding).

        Args:
            pose: (B, T, 69) SMPL pose sequence
            device: torch device

        Returns:
            token_ids: (B, max_len) generated token IDs
        """
        self.eval()
        pose = pose.to(device)
        pose_emb = self.encoder(pose)  # (B, d_model)
        return self.decoder._greedy_decode(pose_emb, device)

    def get_embedding(self, pose):
        """Get the pose embedding from the encoder.

        Args:
            pose: (B, T, 69) SMPL pose sequence

        Returns:
            emb: (B, d_model) pose embedding
        """
        return self.encoder(pose)
