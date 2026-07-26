"""
USDL Model for Action Quality Assessment.
Adapted from Tang et al. CVPR 2020 - "Uncertainty-Aware Score Distribution Learning
for Action Quality Assessment".

Architecture (adapted for SMPL input):
  Original: I3D backbone -> 3 FC layers (256, 128, m) shared across N=10 clips
            -> temporal average pool -> softmax -> score distribution
  Adaptation: Replace I3D with STGCNEncoder (output 256-D), then predict
              score distribution over 10 bins.

Components:
  - STGCNEncoder: (B, T, 69) -> (B, 256)    [replaces I3D backbone]
  - ScoreDistributionHead: 256 -> 256 -> 128 -> 10, softmax output
  - MLPClassifier (optional): 256 -> 256 -> 3, standard classification head
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from shared.models import STGCNEncoder, ScoreDistributionHead, MLPClassifier
from shared.config import SMPL_POSE_DIM, N_CLASSES, MODEL_CONFIGS


class USDLModel(nn.Module):
    """USDL model with STGCN encoder + score distribution head + optional classifier.

    Supports two inference modes:
      1. Distribution learning (KL divergence):
         pose -> STGCNEncoder -> ScoreDistributionHead -> softmax over 10 bins
      2. Direct classification (Cross-entropy):
         pose -> STGCNEncoder -> MLPClassifier -> logits over 3 classes
    """

    def __init__(
        self,
        input_dim=SMPL_POSE_DIM,
        hidden_dim=256,
        n_bins=10,
        dropout=0.3,
        use_classifier=True,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.use_classifier = use_classifier

        usdl_cfg = MODEL_CONFIGS.get("usdl", {})
        dist_hidden = usdl_cfg.get("hidden_dims", [256, 128])

        # ---- Encoder (replaces I3D backbone) ----
        self.encoder = STGCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # ---- Score Distribution Head ----
        self.dist_head = ScoreDistributionHead(
            input_dim=hidden_dim,
            hidden_dims=dist_hidden,
            n_bins=n_bins,
            dropout=dropout,
        )

        # ---- Optional direct classifier (for comparison) ----
        if use_classifier:
            self.classifier = MLPClassifier(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_classes=N_CLASSES,
                dropout=dropout,
            )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, 69) SMPL pose sequence

        Returns:
            If use_classifier=True:
                dist: (B, n_bins) predicted score distribution (softmax)
                logits_cls: (B, N_CLASSES) classification logits
                encoding: (B, hidden_dim) pose encoding
            If use_classifier=False:
                dist: (B, n_bins) predicted score distribution (softmax)
                encoding: (B, hidden_dim) pose encoding
        """
        encoding = self.encoder(x)  # (B, hidden_dim)
        dist = self.dist_head(encoding)  # (B, n_bins), softmax output

        if self.use_classifier:
            logits_cls = self.classifier(encoding)  # (B, N_CLASSES)
            return dist, logits_cls, encoding

        return dist, encoding
