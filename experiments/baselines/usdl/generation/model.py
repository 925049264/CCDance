"""
USDL Generator for teacher comment generation, conditioned on the predicted
score distribution.

Architecture:
  STGCNEncoder -> pose encoding (256-D)
    +-> ScoreDistributionHead -> score distribution over 10 bins
    +-> Grade embedding (from argmax of distribution)
  Fusion(pose_encoding || grade_embedding) -> LSTMDecoder -> text tokens

Inference:
  Pose sequence -> distribution -> grade -> conditioning -> greedy decode -> comment
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import STGCNEncoder, ScoreDistributionHead, LSTMDecoder
from shared.config import SMPL_POSE_DIM, MODEL_CONFIGS


# ============================================================================
# Tokenizer utilities
# ============================================================================


def build_vocab(comments_list, min_freq=1):
    """Build word-level vocabulary from a list of comment strings.

    Args:
        comments_list: list of text strings
        min_freq: minimum word frequency to include

    Returns:
        word2idx: dict mapping word -> index
        idx2word: dict mapping index -> word
    """
    word_counts = Counter()
    for comment in comments_list:
        if comment:
            word_counts.update(comment.lower().split())

    vocab = sorted([w for w, c in word_counts.items() if c >= min_freq])

    word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for w in vocab:
        word2idx[w] = len(word2idx)

    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


class CommentTokenizer:
    """Simple word-level tokenizer for dance teacher comments."""

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.vocab_size = len(word2idx)

    def encode(self, text, max_len=512):
        """Encode a text string into token IDs.

        Returns (seq_len,) LongTensor with SOS at start and EOS at end.
        """
        tokens = [self.SOS]
        for w in text.lower().split():
            tokens.append(self.word2idx.get(w, self.UNK))
        tokens.append(self.EOS)

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        return torch.LongTensor(tokens)

    def decode(self, token_ids):
        """Decode token IDs back into a text string."""
        words = []
        for tid in token_ids:
            tid = int(tid)
            if tid == self.EOS:
                break
            if tid in (self.PAD, self.SOS):
                continue
            words.append(self.idx2word.get(tid, "<UNK>"))
        return " ".join(words)


# ============================================================================
# USDL Generator
# ============================================================================


class USDLGenerator(nn.Module):
    """USDL-based comment generator with grade-conditioned decoding.

    The model first encodes the pose sequence, predicts a score distribution,
    and uses both the pose encoding and the predicted grade to condition
    an LSTM text decoder.
    """

    def __init__(self, vocab_size, input_dim=SMPL_POSE_DIM, hidden_dim=256,
                 n_bins=10, grade_embed_dim=64, decoder_embed_dim=256,
                 decoder_hidden_dim=512, decoder_num_layers=2,
                 dropout=0.3, max_comment_len=512):
        super().__init__()
        self.n_bins = n_bins
        self.max_comment_len = max_comment_len
        self.grade_embed_dim = grade_embed_dim

        # ---- Pose Encoder ----
        self.encoder = STGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # ---- Score Distribution Head ----
        usdl_cfg = MODEL_CONFIGS.get("usdl", {})
        dist_hidden = usdl_cfg.get("hidden_dims", [256, 128])
        self.dist_head = ScoreDistributionHead(
            input_dim=hidden_dim,
            hidden_dims=dist_hidden,
            n_bins=n_bins,
            dropout=dropout,
        )

        # ---- Grade embedding (3 grades -> embed_dim) ----
        self.grade_embed = nn.Embedding(3, grade_embed_dim)

        # ---- Fusion: pose encoding + grade embedding -> decoder condition ----
        fusion_dim = hidden_dim + grade_embed_dim
        self.condition_proj = nn.Linear(fusion_dim, decoder_hidden_dim)

        # ---- LSTM Decoder ----
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=decoder_embed_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=dropout,
            max_len=max_comment_len,
        )

        # Override the default decoder embedding to allow different dims
        # NOTE: LSTMDecoder creates its own embed layer.  We need to replace
        # it so the decoder can have a different embed_dim than the grade
        # embedding.
        if decoder_embed_dim != grade_embed_dim:
            self.decoder.embed = nn.Embedding(
                vocab_size, decoder_embed_dim, padding_idx=0
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def distribution_to_grade(self, dist):
        """Convert predicted distribution to grade index.

        Mapping: bins 0-3 -> 0 (A), bins 4-6 -> 1 (B), bins 7-9 -> 2 (C).
        """
        bin_idx = dist.argmax(dim=1)
        grade_map = torch.zeros(self.n_bins, dtype=torch.long, device=dist.device)
        grade_map[:4] = 0
        grade_map[4:7] = 1
        grade_map[7:] = 2
        return grade_map[bin_idx]

    def _build_condition(self, encoding, dist):
        """Build the decoder conditioning vector.

        Args:
            encoding: (B, hidden_dim) pose encoding
            dist: (B, n_bins) score distribution

        Returns:
            condition: (B, decoder_hidden_dim) conditioning vector
        """
        grade = self.distribution_to_grade(dist)          # (B,)
        grade_emb = self.grade_embed(grade)               # (B, grade_embed_dim)
        fusion_in = torch.cat([encoding, grade_emb], dim=1)  # (B, hidden+grade_dim)
        condition = self.condition_proj(fusion_in)        # (B, decoder_hidden_dim)
        return condition

    # ------------------------------------------------------------------
    # Forward (training with teacher forcing)
    # ------------------------------------------------------------------

    def forward(self, pose, target_tokens=None, teacher_forcing_ratio=0.5,
                return_dist=True):
        """Forward pass for generation with optional teacher forcing.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            target_tokens: (B, seq_len) ground-truth token IDs for teacher forcing
            teacher_forcing_ratio: probability of teacher forcing during training
            return_dist: whether to also return the score distribution

        Returns:
            logits: (B, seq_len, vocab_size) if target_tokens provided,
                    else (B, max_len, vocab_size) from greedy decode
            dist: (B, n_bins) score distribution (if return_dist=True)
        """
        encoding = self.encoder(pose)            # (B, hidden_dim)
        dist = self.dist_head(encoding)          # (B, n_bins)  softmax
        condition = self._build_condition(encoding, dist)

        if target_tokens is not None:
            # Teacher-forced training
            logits = self._teacher_forced_decode(condition, target_tokens)
        else:
            # Free-running inference (greedy)
            logits = self._greedy_decode(condition)

        if return_dist:
            return logits, dist
        return logits

    def _teacher_forced_decode(self, condition, target_tokens):
        """Run decoder with teacher forcing.

        Args:
            condition: (B, decoder_hidden_dim)
            target_tokens: (B, seq_len) ground-truth tokens

        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = target_tokens.shape
        device = condition.device

        # Embed target tokens
        embedded = self.decoder.dropout(self.decoder.embed(target_tokens))

        # Use condition to initialize LSTM state
        h0 = condition.unsqueeze(0).repeat(
            self.decoder.lstm.num_layers * 2, 1, 1
        )  # (num_layers*2, B, decoder_hidden_dim)
        c0 = torch.zeros_like(h0)

        lstm_out, _ = self.decoder.lstm(embedded, (h0, c0))
        logits = self.decoder.output_proj(lstm_out)
        return logits  # (B, seq_len, vocab_size)

    def _greedy_decode(self, condition):
        """Greedy decoding from condition vector.

        Args:
            condition: (B, decoder_hidden_dim)

        Returns:
            logits: (B, max_len, vocab_size)
        """
        B = condition.size(0)
        device = condition.device

        h0 = condition.unsqueeze(0).repeat(
            self.decoder.lstm.num_layers * 2, 1, 1
        )
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        input_token = torch.ones(B, 1, dtype=torch.long, device=device)  # SOS
        outputs = []

        for _ in range(self.max_comment_len):
            embedded = self.decoder.dropout(self.decoder.embed(input_token))
            lstm_out, hidden = self.decoder.lstm(embedded, hidden)
            logits = self.decoder.output_proj(lstm_out)  # (B, 1, V)
            outputs.append(logits)
            input_token = logits.argmax(dim=-1)  # (B, 1)

        return torch.cat(outputs, dim=1)  # (B, max_len, V)

    # ------------------------------------------------------------------
    # Generation (no target)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, pose, max_len=None):
        """Generate a comment from a pose sequence (inference).

        Args:
            pose: (1, T, 69) or (B, T, 69) pose sequence
            max_len: maximum generation length (default: self.max_comment_len)

        Returns:
            token_ids: (B, max_len) generated token indices
            dist: (B, n_bins) score distribution
        """
        max_len = max_len or self.max_comment_len
        device = next(self.parameters()).device
        if pose.device != device:
            pose = pose.to(device)

        encoding = self.encoder(pose)
        dist = self.dist_head(encoding)
        condition = self._build_condition(encoding, dist)

        B = condition.size(0)
        h0 = condition.unsqueeze(0).repeat(
            self.decoder.lstm.num_layers * 2, 1, 1
        )
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        input_token = torch.ones(B, 1, dtype=torch.long, device=device)
        token_ids = []

        for _ in range(max_len):
            embedded = self.decoder.dropout(self.decoder.embed(input_token))
            lstm_out, hidden = self.decoder.lstm(embedded, hidden)
            logits = self.decoder.output_proj(lstm_out)
            next_token = logits.argmax(dim=-1)  # (B, 1)
            token_ids.append(next_token)
            input_token = next_token

            # Stop if all sequences emit EOS
            if (next_token == 2).all():
                break

        tokens_tensor = torch.cat(token_ids, dim=1)  # (B, generated_len)
        # Pad to max_len
        if tokens_tensor.size(1) < max_len:
            pad = torch.zeros(B, max_len - tokens_tensor.size(1),
                              dtype=torch.long, device=device)
            tokens_tensor = torch.cat([tokens_tensor, pad], dim=1)

        return tokens_tensor[:, :max_len], dist
