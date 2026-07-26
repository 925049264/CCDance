"""
CoRe Classification Model

Reference: Yu et al., "Group-Aware Contrastive Regression for
Action Quality Assessment", ICCV 2021.

Architecture (adapted for SMPL pose input):
- PoseLSTMEncoder backbone (256-D output) replaces original I3D
- Group-Aware Regression Tree (GART): binary decision tree, depth=5, 32 leaves
  - Each internal node: Linear(256->1) + Sigmoid for binary routing,
    Linear(256->256) + ReLU for feature update (residual)
  - Each leaf: Linear(256 -> 3) class predictor + Linear(256 -> 256) embedding
- CoReSimple ablation: PoseLSTMEncoder + MLPClassifier (no GART)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import PoseLSTMEncoder, MLPClassifier
from shared.config import SMPL_POSE_DIM, N_CLASSES, MODEL_CONFIGS


# ==============================================================================
# Group-Aware Regression Tree (GART)
# ==============================================================================

class GroupAwareRegressionTree(nn.Module):
    """Binary decision tree for hierarchical group-aware regression.

    Each internal node learns a soft binary routing decision and a feature
    transformation (residual style). Each leaf stores a class predictor and
    an embedding projector used for contrastive learning.

    Args:
        depth: Tree depth. Total leaves = 2^depth.
        node_dim: Feature dimension throughout the tree.
        num_classes: Number of output classes.
    """

    def __init__(self, depth=5, node_dim=256, num_classes=N_CLASSES):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.node_dim = node_dim
        self.num_internal = self.num_leaves - 1  # 31 for depth=5

        # Routing: binary decision at each internal node (sigmoid output ~ P(left))
        self.routings = nn.ModuleList([
            nn.Linear(node_dim, 1) for _ in range(self.num_internal)
        ])

        # Feature transformation with residual connection
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ) for _ in range(self.num_internal)
        ])

        # Leaf classifiers (3-class predictions)
        self.leaf_classifiers = nn.ModuleList([
            nn.Linear(node_dim, num_classes) for _ in range(self.num_leaves)
        ])

        # Leaf embeddings (for contrastive learning)
        self.leaf_projectors = nn.ModuleList([
            nn.Linear(node_dim, node_dim) for _ in range(self.num_leaves)
        ])

    def forward(self, x):
        """Forward pass through the tree.

        Args:
            x: (B, node_dim) feature vectors from backbone.

        Returns:
            dict with keys:
                logits: (B, num_classes) aggregated prediction.
                leaf_logits: (B, num_leaves, num_classes) per-leaf predictions.
                leaf_probs: (B, num_leaves) soft routing probabilities.
                leaf_embs: (B, num_leaves, node_dim) per-leaf embeddings.
                features: (B, node_dim) aggregated feature embedding.
        """
        B = x.size(0)
        device = x.device

        # BFS through the tree: track features and cumulative probabilities
        cur_features = [x]               # each (B, node_dim)
        cur_probs = [torch.ones(B, 1, device=device)]

        for level in range(self.depth):
            n_nodes = 2 ** level
            next_features = []
            next_probs = []

            for i in range(n_nodes):
                node_idx = (2 ** level - 1) + i
                feat = cur_features[i]
                prob = cur_probs[i]

                # Routing decision
                route_logit = self.routings[node_idx](feat)          # (B, 1)
                p_left = torch.sigmoid(route_logit)                  # P(left)

                # Feature transformation (residual)
                delta = self.transforms[node_idx](feat)

                # Left child
                left_feat = feat + delta
                left_prob = prob * p_left

                # Right child
                right_feat = feat - delta
                right_prob = prob * (1.0 - p_left)

                next_features.extend([left_feat, right_feat])
                next_probs.extend([left_prob, right_prob])

            cur_features = next_features
            cur_probs = next_probs

        # ---- Leaf layer ----
        leaf_logits = []   # (B, num_classes) each
        leaf_embs = []     # (B, node_dim) each

        for leaf_idx in range(self.num_leaves):
            leaf_feat = cur_features[leaf_idx]
            leaf_logits.append(self.leaf_classifiers[leaf_idx](leaf_feat))
            leaf_embs.append(self.leaf_projectors[leaf_idx](leaf_feat))

        leaf_logits = torch.stack(leaf_logits, dim=1)   # (B, L, C)
        leaf_embs = torch.stack(leaf_embs, dim=1)       # (B, L, D)

        leaf_probs = torch.cat(cur_probs, dim=1)        # (B, L)
        leaf_probs = leaf_probs / (leaf_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Aggregate: probability-weighted sum over leaves
        aggregated_logits = torch.bmm(
            leaf_probs.unsqueeze(1),                    # (B, 1, L)
            leaf_logits                                 # (B, L, C)
        ).squeeze(1)                                     # (B, C)

        aggregated_feats = torch.bmm(
            leaf_probs.unsqueeze(1),                    # (B, 1, L)
            leaf_embs                                   # (B, L, D)
        ).squeeze(1)                                     # (B, D)

        return {
            'logits': aggregated_logits,
            'leaf_logits': leaf_logits,
            'leaf_probs': leaf_probs,
            'leaf_embs': leaf_embs,
            'features': aggregated_feats,
        }


# ==============================================================================
# Full CoRe Model
# ==============================================================================

class CoReModel(nn.Module):
    """CoRe: Group-Aware Contrastive Regression for Action Quality Assessment.

    Pose backbone (LSTM) + Group-Aware Regression Tree (GART).

    Args:
        pose_dim: SMPL pose feature dimension (69).
        hidden_dim: Latent feature dimension (256).
        tree_depth: Depth of GART (default 5 -> 32 leaves).
        num_classes: Number of grade classes (3).
        dropout: Dropout rate for backbone.
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, hidden_dim=256,
                 tree_depth=5, num_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.backbone = PoseLSTMEncoder(
            input_dim=pose_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.gart = GroupAwareRegressionTree(
            depth=tree_depth,
            node_dim=hidden_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, 69) SMPL pose sequence.

        Returns:
            dict with 'logits', 'leaf_logits', 'leaf_probs', 'leaf_embs', 'features'.
        """
        emb = self.backbone(x)
        return self.gart(emb)

    def encode_pose(self, x):
        """Extract pose embedding before tree routing.

        Args:
            x: (B, T, 69) SMPL pose sequence.

        Returns:
            (B, hidden_dim) pose embedding.
        """
        return self.backbone(x)


# ==============================================================================
# Simplified CoRe (Ablation): LSTM + MLP, no GART
# ==============================================================================

class CoReSimple(nn.Module):
    """Simplified CoRe baseline: PoseLSTMEncoder + MLPClassifier.

    Ablation model to measure the contribution of the GART component.
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, hidden_dim=256,
                 num_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.backbone = PoseLSTMEncoder(
            input_dim=pose_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.classifier = MLPClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=2,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, 69) SMPL pose sequence.

        Returns:
            logits: (B, num_classes) classification logits.
        """
        emb = self.backbone(x)
        return self.classifier(emb)


# ==============================================================================
# Loss Functions
# ==============================================================================

class CoReLoss(nn.Module):
    """Combined loss for CoRe training.

    Components:
    1. Cross-entropy on the aggregated logits.
    2. Leaf-weighted CE: each leaf classifier should predict correctly,
       weighted by how much each sample belongs to that leaf.
    3. Contrastive grouping: same-grade pairs pulled together,
       different-grade pairs pushed apart (weighted by grade difference).

    Args:
        ce_weight: Weight for aggregated CE loss.
        leaf_ce_weight: Weight for leaf-weighted CE loss.
        contrastive_weight: Weight for contrastive grouping loss.
    """

    def __init__(self, ce_weight=1.0, leaf_ce_weight=1.0, contrastive_weight=0.1):
        super().__init__()
        self.ce_weight = ce_weight
        self.leaf_ce_weight = leaf_ce_weight
        self.contrastive_weight = contrastive_weight

    def forward(self, output, labels):
        """Compute combined loss.

        Args:
            output: dict from CoReModel.forward() with keys:
                logits, leaf_logits, leaf_probs, features.
            labels: (B,) integer grade labels.

        Returns:
            dict with 'loss', 'loss_ce', 'loss_leaf_ce', 'loss_contrast'.
        """
        logits = output['logits']              # (B, C)
        leaf_logits = output['leaf_logits']    # (B, L, C)
        leaf_probs = output['leaf_probs']      # (B, L)
        features = output['features']          # (B, D)

        B, L, C = leaf_logits.shape
        device = labels.device

        # ---- 1. Aggregated CE ----
        loss_ce = F.cross_entropy(logits, labels)

        # ---- 2. Leaf-weighted CE ----
        leaf_labels = labels.unsqueeze(1).expand(-1, L)                   # (B, L)
        ce_all = F.cross_entropy(
            leaf_logits.reshape(-1, C),
            leaf_labels.reshape(-1),
            reduction='none',
        ).reshape(B, L)
        loss_leaf_ce = (ce_all * leaf_probs).sum(dim=1).mean()

        # ---- 3. Contrastive grouping ----
        norm_feats = F.normalize(features, dim=-1)                       # (B, D)
        sim = norm_feats @ norm_feats.T                                   # (B, B)

        grade = labels.unsqueeze(1)                                       # (B, 1)
        same_grade = (grade == grade.T).float()                           # (B, B)
        diff_grade = 1.0 - same_grade
        eye = torch.eye(B, device=device)
        same_grade = same_grade - eye                                     # remove self
        same_grade = (same_grade > 0).float()

        abs_grade_diff = torch.abs(grade - grade.T).float()               # (B, B)

        loss_pull = torch.tensor(0.0, device=device)
        loss_push = torch.tensor(0.0, device=device)

        n_pos = same_grade.sum()
        n_neg = diff_grade.sum()

        if n_pos > 0:
            # Pull same-grade pairs together (maximize cosine similarity)
            pos_sim = (sim * same_grade).sum() / n_pos
            loss_pull = -torch.log(pos_sim.clamp(min=1e-8))

        if n_neg > 0:
            # Push different-grade pairs apart (weighted by grade difference)
            neg_sim = sim * diff_grade * abs_grade_diff
            loss_push = (neg_sim.sum() / n_neg).clamp(max=0.0)  # want negative

        loss_contrast = loss_pull - loss_push  # minimize pull distance, maximize push

        total_loss = (
            self.ce_weight * loss_ce
            + self.leaf_ce_weight * loss_leaf_ce
            + self.contrastive_weight * loss_contrast
        )

        return {
            'loss': total_loss,
            'loss_ce': loss_ce,
            'loss_leaf_ce': loss_leaf_ce,
            'loss_contrast': loss_contrast,
        }
