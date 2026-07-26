"""
Shared model components for CCDance baseline reproduction.
Includes encoders (STGCN, PoseLSTM, Transformer), classification heads,
and generation decoders reused across baselines.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SMPL_POSE_DIM, SMPL_NUM_JOINTS, N_CLASSES


# ============================================================================
# Positional Encoding
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# ST-GCN Encoder (used by USDL, VL-Transformer, Graph-Transformer)
# ============================================================================
class STGCNEncoder(nn.Module):
    """Spatial-Temporal Graph Convolutional Network encoder for SMPL poses."""

    def __init__(self, input_dim=SMPL_POSE_DIM, hidden_dim=256, output_dim=256,
                 num_joints=SMPL_NUM_JOINTS, dropout=0.3):
        super().__init__()
        self.num_joints = num_joints
        self.output_dim = output_dim

        # Reshape 69-D axis-angle to 23 joints x 3 DoF
        self.input_proj = nn.Linear(input_dim, num_joints * 3)

        self.conv1 = nn.Conv2d(3, hidden_dim // 2, (9, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(hidden_dim // 2)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, (9, 1), padding=(4, 0))
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, (9, 1), padding=(4, 0))
        self.bn3 = nn.BatchNorm2d(hidden_dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, 69)
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, 69)
        x = x.reshape(B, T, self.num_joints, 3)  # (B, T, 23, 3)
        x = x.permute(0, 3, 1, 2)  # (B, 3, T, 23)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, hidden_dim)
        x = self.output_proj(x)
        return x  # (B, output_dim)


# ============================================================================
# PoseLSTM Encoder (used by CoRe)
# ============================================================================
class PoseLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder with attention pooling for SMPL poses."""

    def __init__(self, input_dim=SMPL_POSE_DIM, hidden_dim=256, output_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        lstm_out, _ = self.lstm(x)  # (B, T, 2H)
        attn_weights = self.attention(lstm_out).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (B, 2H)
        return self.output_proj(self.dropout(pooled))  # (B, output_dim)


# ============================================================================
# Transformer Encoder (used by Pose Transformer baseline)
# ============================================================================
class TransformerEncoder(nn.Module):
    """Transformer encoder for pose sequences with CLS token."""

    def __init__(self, input_dim=SMPL_POSE_DIM, d_model=256, nhead=8,
                 num_layers=4, output_dim=256, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.output_proj(cls_out)  # (B, output_dim)


# ============================================================================
# Music LSTM Encoder (used by VL-Transformer)
# ============================================================================
class MusicLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for audio features."""

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=256,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T_audio, D_audio) or (B, D_global)
        if x.dim() == 2:
            # Global audio feature - project directly
            x = x.unsqueeze(1)  # (B, 1, D)
        lstm_out, _ = self.lstm(x)
        # Average pooling over time
        pooled = lstm_out.mean(dim=1)  # (B, 2H)
        return self.output_proj(self.dropout(pooled))  # (B, output_dim)


# ============================================================================
# Classification Heads
# ============================================================================
class MLPClassifier(nn.Module):
    """Standard MLP classification head."""

    def __init__(self, input_dim=256, hidden_dim=256, num_classes=N_CLASSES,
                 dropout=0.3, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class ScoreDistributionHead(nn.Module):
    """Score distribution prediction head (USDL-style)."""

    def __init__(self, input_dim=256, hidden_dims=[256, 128], n_bins=10,
                 dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# ============================================================================
# Generation Decoder (LSTM-based)
# ============================================================================
class LSTMDecoder(nn.Module):
    """LSTM decoder for comment generation."""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, target_tokens=None, teacher_forcing_ratio=0.5):
        """
        encoder_output: (B, enc_dim) - pose/multimodal embedding
        target_tokens: (B, seq_len) - target token indices (optional for training)
        """
        B = encoder_output.size(0)
        device = encoder_output.device

        if target_tokens is not None:
            # Training with teacher forcing
            embedded = self.dropout(self.embed(target_tokens))
            # Concatenate encoder output as first "token"
            enc_expanded = encoder_output.unsqueeze(1)  # (B, 1, enc_dim)
            # We use encoder output to initialize LSTM hidden state instead
            h0 = encoder_output.unsqueeze(0).repeat(self.lstm.num_layers * 2, 1, 1)
            c0 = torch.zeros_like(h0)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            return self.output_proj(lstm_out)
        else:
            # Inference: greedy decoding
            return self._greedy_decode(encoder_output, device)

    def _greedy_decode(self, encoder_output, device):
        B = encoder_output.size(0)
        h0 = encoder_output.unsqueeze(0).repeat(self.lstm.num_layers * 2, 1, 1)
        c0 = torch.zeros_like(h0)
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


# ============================================================================
# Graph Transformer Components (X-DANCENET adaptation)
# ============================================================================
class GraphTransformerEncoder(nn.Module):
    """Graph Transformer encoder with spatial + temporal attention."""

    def __init__(self, input_dim=3, d_model=256, nhead=8, num_layers=4,
                 num_joints=SMPL_NUM_JOINTS, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        # Project 69-D SMPL to joint features
        self.input_proj = nn.Linear(SMPL_POSE_DIM, num_joints * input_dim)

        # Joint embedding
        self.joint_embed = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)

        # Spatial attention (cross-joint)
        self.spatial_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Temporal attention (cross-time)
        self.temporal_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 3)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_dim = d_model

    def forward(self, x):
        # x: (B, T, 69)
        B, T, _ = x.shape

        # Reshape SMPL params to per-joint features
        x = self.input_proj(x)  # (B, T, 69)
        x = x.reshape(B, T, self.num_joints, 3)  # (B, T, 23, 3)
        x = self.joint_embed(x)  # (B, T, 23, d_model)

        # Add positional encoding
        x = x.permute(0, 2, 1, 3).reshape(B * self.num_joints, T, self.d_model)
        x = self.pos_encoding(x)
        x = x.reshape(B, self.num_joints, T, self.d_model)

        for i in range(len(self.spatial_attn_layers)):
            # Spatial attention (across joints per timestep)
            x_s = x.permute(0, 2, 1, 3).reshape(B * T, self.num_joints, self.d_model)
            norm = self.norm_layers[i * 3](x_s)
            attn_s, _ = self.spatial_attn_layers[i](norm, norm, norm)
            x_s = x_s + self.dropout(attn_s)
            x_s = x_s.reshape(B, T, self.num_joints, self.d_model)

            # Temporal attention (across time per joint)
            x_t = x_s.permute(0, 2, 1, 3).reshape(B * self.num_joints, T, self.d_model)
            norm = self.norm_layers[i * 3 + 1](x_t)
            attn_t, _ = self.temporal_attn_layers[i](norm, norm, norm)
            x_t = x_t + self.dropout(attn_t)

            # FFN
            norm = self.norm_layers[i * 3 + 2](x_t)
            ffn_out = self.ffn_layers[i](norm)
            x_t = x_t + self.dropout(ffn_out)

            x = x_t.reshape(B, self.num_joints, T, self.d_model)

        # Global pooling: average over joints and time
        x = x.mean(dim=1).mean(dim=1)  # (B, d_model)
        return x
