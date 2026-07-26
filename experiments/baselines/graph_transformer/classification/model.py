"""
Graph-Transformer / X-DANCENET Classifier Model

Architecture (Han et al., SciRep 2026):
- GraphTransformerEncoder: spatial-temporal dual attention over SMPL joints
  - Input: SMPL 69-D pose -> reshape to (23 joints x 3 DoF)
  - Joint embedding: Linear(3 -> d_model=256)
  - Spatial attention across joints per timestep
  - Temporal attention across time per joint
  - Feed-forward networks with residual connections + LayerNorm
  - Global mean pooling -> (B, 256)
- MLPClassifier: 256 -> 128 -> 3 (grade logits)

Simplifications vs. original X-DANCENET:
- No sensor normalization (ASN/Kalman/Butterworth) - SMPL data is clean
- No tempo-conditioned multi-scale feature extraction (MMFE)
- No prototype-based classification (use standard MLP + CE loss)
- No confidence/saliency explanation outputs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from shared.models import GraphTransformerEncoder, MLPClassifier
from shared.config import (SMPL_POSE_DIM, N_CLASSES,
                           MODEL_CONFIGS)


class GraphTransformerModel(nn.Module):
    """Graph-Transformer classification model.

    Pose sequence -> GraphTransformerEncoder (spatial+temporal dual attention)
    -> (B, d_model=256) -> MLPClassifier -> (B, 3) grade logits.
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, d_model=256, nhead=8,
                 num_layers=4, num_joints=23, num_classes=N_CLASSES,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Graph-Transformer encoder with dual spatial-temporal attention
        self.encoder = GraphTransformerEncoder(
            input_dim=3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_joints=num_joints,
            dropout=dropout,
        )

        # Classification head: 256 -> 128 -> 3
        self.classifier = MLPClassifier(
            input_dim=d_model,
            hidden_dim=d_model // 2,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=2,
        )

    def forward(self, x):
        """Forward pass for classification.

        Args:
            x: (B, T, 69) SMPL pose sequence

        Returns:
            logits: (B, 3) classification logits
        """
        # Encode pose sequence with spatial-temporal attention
        emb = self.encoder(x)  # (B, d_model)

        # Classify
        logits = self.classifier(emb)  # (B, 3)

        return logits

    def get_embedding(self, x):
        """Get the pose embedding (without classification head).

        Args:
            x: (B, T, 69) SMPL pose sequence

        Returns:
            emb: (B, d_model) pose embedding
        """
        return self.encoder(x)
