"""
LeViT-Hybrid Classifier Model

Architecture (Wang, SciRep 2025 - "Hybrid Model Integrating LeViT Transformer and
Distillation Techniques for Dance Classification"):

Original formulation (RGB images):
  - CNN backbone (3x3 conv, 128 filters) -> patch embedding -> Transformer encoder
  - Knowledge distillation from ViT-B/16 teacher
  - LeViT-128 + ViT-B/16, 224x224 images, AdamW(lr=1e-5), batch 32, 100 epochs

Adaptation for CCDance SMPL pose sequences (Option A - preferred):
  Since we have 69-D SMPL pose vectors rather than RGB images, we adapt LeViT's
  patch-based design to pose sequences:

  1. PosePatchEmbedding: divide the 300-frame sequence into 20 non-overlapping
     patches of 15 frames each. Each patch (69*15 = 1035-D) is linearly projected
     to a 256-D patch embedding, with learned positional encoding.

  2. "LeViT-like" Transformer encoder: a TransformerEncoder (from shared.models)
     operating on patch embeddings. 4 layers, 8 heads, d_model=256.

  3. Knowledge distillation: a larger Transformer (8 layers, d_model=512, 8 heads)
     serves as the teacher. The teacher is pretrained/frozen and provides soft
     targets via KL divergence during student training.

  4. Classification head: MLP(256 -> 256 -> 3) acting on the CLS token.

The key adaptation is replacing the CNN patch embedding with a pose-patch embedding:
  - Images: conv patch embedding (16x16 patches of 3-channel pixels)
  -> SMPL sequences: temporal patch embedding (15-frame patches of 69-D poses)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models import TransformerEncoder, MLPClassifier
from shared.config import (SMPL_POSE_DIM, N_CLASSES,
                           SEQUENCE_LENGTH, MODEL_CONFIGS)


class PosePatchEmbedding(nn.Module):
    """Pose patch embedding: divides pose sequence into patches and projects.

    For an input of (B, T=300, D=69):
      - Divide T into N patches of patch_size frames each: N = T / patch_size
      - Each patch has dimension patch_size * SMPL_POSE_DIM
      - Linear project to embed_dim
      - Add learned positional embedding

    Default configuration:
      - T = 300, patch_size = 15 -> N = 20 patches
      - Each patch flattened to 69 * 15 = 1035-D
      - Projected to 256-D
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

        # Linear projection: (pose_dim * patch_size) -> embed_dim
        self.proj = nn.Linear(pose_dim * patch_size, embed_dim)

        # Learned positional embedding for patches + CLS token
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        # CLS token (prepended to patch sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Create patch embeddings from pose sequence.

        Args:
            x: (B, T, 69) SMPL pose sequence, T = SEQUENCE_LENGTH

        Returns:
            embeddings: (B, N+1, embed_dim) patch + CLS token embeddings
        """
        B = x.shape[0]

        # Ensure exact sequence length
        if x.size(1) != self.seq_length:
            if x.size(1) > self.seq_length:
                # Center crop
                start = (x.size(1) - self.seq_length) // 2
                x = x[:, start:start + self.seq_length, :]
            else:
                # Pad
                pad_len = self.seq_length - x.size(1)
                pad = torch.zeros(B, pad_len, self.pose_dim, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)

        # Reshape into patches: (B, N, patch_size * pose_dim)
        x = x.reshape(B, self.num_patches, self.patch_size * self.pose_dim)

        # Project patches to embedding dimension
        x = self.proj(x)  # (B, N, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        return self.dropout(x)


class LeViTEncoder(nn.Module):
    """Transformer encoder with LeViT-inspired architecture for pose patches.

    This wraps TransformerEncoder from shared.models but is configured with
    LeViT-specific parameters (pre-normalization, GELU activations) to better
    match the original LeViT design.

    Architecture:
      - Input: (B, N+1, d_model) patch embeddings with CLS token
      - LeViT-style transformer blocks with pre-norm, GELU, etc.
      - Output: (B, d_model) CLS token embedding
    """

    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = d_model

        # Use PyTorch's native transformer encoder for simplicity
        # We use BatchFirst=True and standard ReLU in feedforward for consistency
        # with shared.models.TransformerEncoder, but the key architectural
        # choice (CLS token, multi-head self-attention) is the same as LeViT.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm (LeViT-style)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Process patch embeddings through transformer layers.

        Args:
            x: (B, num_patches+1, d_model) patch + CLS embeddings

        Returns:
            cls_out: (B, d_model) CLS token representation
        """
        x = self.transformer(x)
        x = self.norm(x)
        # Extract CLS token (first position)
        cls_out = x[:, 0, :]
        return cls_out


class LeViTHybridModel(nn.Module):
    """LeViT-Hybrid classification model with knowledge distillation.

    Components:
      1. PosePatchEmbedding: (B, T, 69) -> (B, N+1, 256)
      2. Student encoder: LeViTEncoder(256, 8, 4) -> (B, 256)
      3. Teacher encoder: LeViTEncoder(512, 8, 8) -> (B, 512) [larger]
         - The teacher is a larger Transformer used for knowledge distillation
         - Teacher can be pretrained or frozen during student training
      4. Student classifier: MLP(256 -> 256 -> 3)
      5. Teacher classifier: MLP(512 -> 256 -> 3) [optional, for teacher training]
      6. Distillation: KL divergence between teacher and student distributions

    Forward modes:
      - Training (distill=True): returns student logits, teacher logits, and
        patch embeddings for distillation loss computation
      - Evaluation (distill=False): returns student logits only
    """

    def __init__(self, pose_dim=SMPL_POSE_DIM, seq_length=SEQUENCE_LENGTH,
                 patch_size=15, embed_dim=256, teacher_embed_dim=512,
                 nhead=8, student_layers=4, teacher_layers=8,
                 mlp_ratio=2, dropout=0.1, num_classes=N_CLASSES,
                 temperature=4.0, alpha=0.5):
        """Initialize LeViT-Hybrid model.

        Args:
            pose_dim: SMPL pose dimension (69)
            seq_length: Number of frames in sequence (300)
            patch_size: Frames per patch (15)
            embed_dim: Student embedding dimension (256)
            teacher_embed_dim: Teacher embedding dimension (512)
            nhead: Number of attention heads
            student_layers: Number of student transformer layers (4)
            teacher_layers: Number of teacher transformer layers (8)
            mlp_ratio: Feedforward dimension ratio relative to d_model
            dropout: Dropout rate
            num_classes: Number of output classes (3)
            temperature: Distillation temperature
            alpha: Weight for distillation loss (0.5) vs CE loss
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.teacher_embed_dim = teacher_embed_dim
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha

        # Shared patch embedding (student and teacher use the same patches)
        self.patch_embed = PosePatchEmbedding(
            seq_length=seq_length,
            pose_dim=pose_dim,
            patch_size=patch_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Student projection: for teacher with larger d_model,
        # project patch embeddings to teacher dimension
        self.student_proj = nn.Identity()  # patches already at student embed_dim

        # Teacher projection: project patch embeddings to teacher dimension
        self.teacher_proj = nn.Linear(embed_dim, teacher_embed_dim)

        # Student encoder
        self.student_encoder = LeViTEncoder(
            d_model=embed_dim,
            nhead=nhead,
            num_layers=student_layers,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
        )

        # Teacher encoder (larger)
        self.teacher_encoder = LeViTEncoder(
            d_model=teacher_embed_dim,
            nhead=nhead,
            num_layers=teacher_layers,
            dim_feedforward=teacher_embed_dim * mlp_ratio,
            dropout=dropout,
        )

        # Student classification head
        self.student_classifier = MLPClassifier(
            input_dim=embed_dim,
            hidden_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=2,
        )

        # Teacher classification head
        self.teacher_classifier = MLPClassifier(
            input_dim=teacher_embed_dim,
            hidden_dim=teacher_embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=2,
        )

    def encode_patches(self, pose):
        """Convert pose sequence to patch embeddings.

        Args:
            pose: (B, T, 69) SMPL pose sequence

        Returns:
            patches: (B, N+1, embed_dim) patch embeddings with CLS token
        """
        return self.patch_embed(pose)

    def encode_student(self, patches):
        """Encode patch embeddings with student transformer.

        Args:
            patches: (B, N+1, embed_dim)

        Returns:
            student_emb: (B, embed_dim) student CLS embedding
        """
        return self.student_encoder(patches)

    def encode_teacher(self, patches):
        """Encode patch embeddings with teacher transformer.

        Args:
            patches: (B, N+1, embed_dim)

        Returns:
            teacher_emb: (B, teacher_embed_dim) teacher CLS embedding
        """
        # Project patches to teacher dimension
        teacher_patches = self.teacher_proj(patches)
        return self.teacher_encoder(teacher_patches)

    def forward(self, pose, return_teacher=False):
        """Forward pass.

        Args:
            pose: (B, T, 69) SMPL pose sequence
            return_teacher: If True, also return teacher logits (for distillation)

        Returns:
            If return_teacher:
                student_logits: (B, num_classes)
                teacher_logits: (B, num_classes)
            Else:
                student_logits: (B, num_classes)
        """
        # Shared patch embedding
        patches = self.encode_patches(pose)  # (B, N+1, embed_dim)

        # Student forward
        student_emb = self.encode_student(patches)  # (B, embed_dim)
        student_logits = self.student_classifier(student_emb)  # (B, num_classes)

        if return_teacher:
            # Teacher forward
            teacher_emb = self.encode_teacher(patches)  # (B, teacher_embed_dim)
            teacher_logits = self.teacher_classifier(teacher_emb)  # (B, num_classes)
            return student_logits, teacher_logits

        return student_logits

    def compute_distillation_loss(self, student_logits, teacher_logits,
                                  labels=None):
        """Compute combined distillation + classification loss.

        Loss = alpha * KL(teacher_soft || student_soft) * T^2
             + (1 - alpha) * CE(student_logits, labels)

        The T^2 scaling factor ensures gradients are properly scaled
        (from Hinton et al., 2015 "Distilling the Knowledge in a Neural Network").

        Args:
            student_logits: (B, num_classes) raw student logits
            teacher_logits: (B, num_classes) raw teacher logits
            labels: (B,) ground-truth labels (optional, for CE loss)

        Returns:
            loss: scalar distillation loss
            distill_loss: scalar KL divergence component (for logging)
            ce_loss: scalar CE component (for logging)
        """
        # Soften logits with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence: D_KL(teacher || student)
        # We use the standard KL div formulation for distillation
        distill_loss = F.kl_div(
            student_soft, teacher_soft,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)

        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            loss = self.alpha * distill_loss + (1.0 - self.alpha) * ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)
            loss = distill_loss

        return loss, distill_loss, ce_loss

    def get_embeddings(self, pose):
        """Extract learned representations for downstream tasks.

        Returns both student and teacher embeddings.

        Args:
            pose: (B, T, 69) SMPL pose sequence

        Returns:
            dict with 'student' and 'teacher' embeddings
        """
        patches = self.encode_patches(pose)
        student_emb = self.encode_student(patches)
        teacher_emb = self.encode_teacher(patches)

        return {
            'student': student_emb,
            'teacher': teacher_emb,
            'patches': patches,
        }

    def freeze_teacher(self):
        """Freeze teacher network parameters (for student-only training)."""
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False
        for param in self.teacher_proj.parameters():
            param.requires_grad = False

    def unfreeze_teacher(self):
        """Unfreeze teacher network parameters."""
        for param in self.teacher_encoder.parameters():
            param.requires_grad = True
        for param in self.teacher_classifier.parameters():
            param.requires_grad = True
        for param in self.teacher_proj.parameters():
            param.requires_grad = True
