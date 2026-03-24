"""
model.py — Improved EfficientNet + GRU Deepfake Detector

Architecture highlights:
  • EfficientNet-B4 backbone (timm) with optional gradient checkpointing
  • Multi-scale feature fusion from multiple backbone stages
  • Bidirectional GRU for temporal modelling
  • Multi-head self-attention over GRU outputs (temporal attention)
  • Deep classification head with residual dropout
  • Optional frame-level auxiliary head for richer supervision
  • Frequency-domain branch (DCT) optionally fused with spatial features
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import cfg, ModelConfig


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w


class FrequencyBranch(nn.Module):
    """
    Lightweight DCT-based frequency-domain branch.
    Converts each face frame to DCT and extracts frequency artifacts
    that are invisible in pixel space.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    @staticmethod
    def _dct_approx(x: torch.Tensor) -> torch.Tensor:
        """Approximate DCT via FFT magnitude spectrum."""
        x_f = torch.fft.fft2(x)
        return torch.abs(x_f) + 1e-8   # log-scale is informative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self._dct_approx(x)
        freq = torch.log(freq)
        freq = self.conv(freq)
        return self.fc(freq.flatten(1))


class MultiScaleFusion(nn.Module):
    """
    Fuse feature maps extracted from multiple EfficientNet stages
    into a single fixed-size vector.
    """

    def __init__(self, stage_dims: List[int], out_dim: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(d, out_dim // len(stage_dims), bias=False),
                nn.LayerNorm(out_dim // len(stage_dims)),
            ) for d in stage_dims]
        )
        self.out_dim = out_dim

    def forward(self, stage_feats: List[torch.Tensor]) -> torch.Tensor:
        parts = [proj(f) for proj, f in zip(self.projections, stage_feats)]
        return torch.cat(parts, dim=-1)   # (B, out_dim)


class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the sequence of GRU hidden states.
    Allows the model to weight important frames differently.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H)
        residual = x
        x, _ = self.attn(x, x, x)
        x = self.norm(self.dropout(x) + residual)
        return x


class ClassificationHead(nn.Module):
    """
    Deep residual classification head:
      Linear → BN → ReLU → Dropout → … → Linear(1)
    """

    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h, bias=False),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Improved EfficientNet + GRU deepfake detector.

    Input:  (B, T, C, H, W)  — batch of video clips
    Output: (B,)             — logit per clip  [+ optional (B, T) aux logits]
    """

    def __init__(self, mcfg: ModelConfig = None):
        super().__init__()
        mcfg = mcfg or cfg.model

        # ── 1. EfficientNet Backbone ──────────────────────────
        self.backbone = timm.create_model(
            mcfg.backbone,
            pretrained=(mcfg.pretrained == "imagenet"),
            features_only=mcfg.use_multi_scale,   # returns stage feature maps
            out_indices=mcfg.multi_scale_stages if mcfg.use_multi_scale else None,
        )

        if mcfg.use_gradient_checkpointing:
            # timm doesn't expose checkpointing directly; wrap manually
            self.backbone.set_grad_checkpointing(True)

        # Detect feature dimensions
        if mcfg.use_multi_scale:
            dummy = torch.zeros(1, 3, mcfg.face_size, mcfg.face_size)
            with torch.no_grad():
                stage_out = self.backbone(dummy)
            stage_dims = [s.shape[1] for s in stage_out]
            ms_out_dim = 512
            self.ms_fusion = MultiScaleFusion(stage_dims, ms_out_dim)
            spatial_dim = ms_out_dim
        else:
            dummy = torch.zeros(1, 3, mcfg.face_size, mcfg.face_size)
            with torch.no_grad():
                feat = self.backbone(dummy)
            spatial_dim = feat.shape[-1]   # timm global pool output
            self.ms_fusion = None

        # ── 2. Optional Frequency Branch ──────────────────────
        self.use_freq = getattr(mcfg, "use_freq_branch", True)
        freq_dim = 128
        if self.use_freq:
            self.freq_branch = FrequencyBranch(out_dim=freq_dim)
            spatial_dim += freq_dim

        # ── 3. Feature Projection before GRU ──────────────────
        proj_dim = mcfg.gru_hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ── 4. Bidirectional GRU ───────────────────────────────
        gru_out = mcfg.gru_hidden_dim * (2 if mcfg.gru_bidirectional else 1)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=mcfg.gru_hidden_dim,
            num_layers=mcfg.gru_num_layers,
            batch_first=True,
            dropout=mcfg.gru_dropout if mcfg.gru_num_layers > 1 else 0,
            bidirectional=mcfg.gru_bidirectional,
        )

        # ── 5. Temporal Attention ──────────────────────────────
        self.use_attn = mcfg.use_temporal_attention
        if self.use_attn:
            self.temporal_attn = TemporalAttention(
                hidden_dim=gru_out,
                num_heads=mcfg.attention_heads,
                dropout=0.1,
            )

        # ── 6. Aggregation ────────────────────────────────────
        # Mean + Max pooling concatenated → 2 × gru_out
        agg_dim = gru_out * 2

        # ── 7. Classification Head ────────────────────────────
        self.head = ClassificationHead(
            in_dim=agg_dim,
            hidden_dims=mcfg.classifier_dims,
            dropout=mcfg.classifier_dropout,
        )

        # ── 8. Auxiliary Frame-level Head ─────────────────────
        self.use_aux = mcfg.use_aux_loss
        if self.use_aux:
            self.aux_head = nn.Linear(gru_out, 1)

        # ── Weight Init ───────────────────────────────────────
        self._init_new_weights()

        # Track backbone param names for differential LR
        self._backbone_param_names = {n for n, _ in self.backbone.named_parameters()}

    # ── Initialisation ────────────────────────────────────────
    def _init_new_weights(self):
        for m in [self.proj, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ── Freeze / Unfreeze Backbone ────────────────────────────
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ── Frame Feature Extraction ──────────────────────────────
    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*T, C, H, W)
        returns: (B*T, spatial_dim)
        """
        if self.ms_fusion is not None:
            stages = self.backbone(x)           # list of (B*T, C_i, H_i, W_i)
            spatial = self.ms_fusion(stages)    # (B*T, ms_out_dim)
        else:
            spatial = self.backbone(x)          # (B*T, feat_dim) — global pooled

        if self.use_freq:
            freq = self.freq_branch(x)          # (B*T, freq_dim)
            spatial = torch.cat([spatial, freq], dim=-1)

        return spatial

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            clip_logits:  (B,)
            frame_logits: (B, T) or None
        """
        B, T, C, H, W = x.shape

        # ── Spatial features per frame ─────────────────────
        x_flat = x.view(B * T, C, H, W)
        feats = self._extract_frame_features(x_flat)    # (B*T, spatial_dim)
        feats = feats.view(B, T, -1)                    # (B, T, spatial_dim)

        # ── Project ────────────────────────────────────────
        feats = self.proj(feats.view(B * T, -1))        # (B*T, proj_dim)
        feats = feats.view(B, T, -1)                    # (B, T, proj_dim)

        # ── GRU ────────────────────────────────────────────
        gru_out, _ = self.gru(feats)                    # (B, T, gru_out)

        # ── Temporal Attention ─────────────────────────────
        if self.use_attn:
            gru_out = self.temporal_attn(gru_out)       # (B, T, gru_out)

        # ── Auxiliary frame-level logits ───────────────────
        frame_logits = None
        if self.use_aux and self.training:
            frame_logits = self.aux_head(gru_out).squeeze(-1)  # (B, T)

        # ── Mean + Max pooling over time ───────────────────
        mean_pool = gru_out.mean(dim=1)                 # (B, gru_out)
        max_pool = gru_out.max(dim=1).values            # (B, gru_out)
        agg = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2*gru_out)

        # ── Classification ─────────────────────────────────
        clip_logits = self.head(agg)                    # (B,)

        return clip_logits, frame_logits

    # ── Inference convenience ─────────────────────────────────
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fake probability per clip — no grad."""
        self.eval()
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)

    # ── Parameter groups for differential LR ─────────────────
    def get_param_groups(self, lr: float, backbone_multiplier: float = 0.1):
        backbone_params = list(self.backbone.parameters())
        other_params = [
            p for n, p in self.named_parameters()
            if n.split(".")[0] != "backbone"
        ]
        return [
            {"params": backbone_params, "lr": lr * backbone_multiplier},
            {"params": other_params,    "lr": lr},
        ]

    # ── Model summary ─────────────────────────────────────────
    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss — down-weights easy negatives."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class DeepfakeLoss(nn.Module):
    """
    Combined loss:
      L = clip_loss + aux_weight * frame_loss
    """

    def __init__(
        self,
        loss_type: str = "bce",
        label_smoothing: float = 0.05,
        aux_weight: float = 0.3,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.label_smoothing = label_smoothing

        pos_weight = None
        if class_weights:
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]])

        if loss_type == "focal":
            self.clip_loss_fn = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        else:
            self.clip_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Frame-level always uses plain BCE (already regularised enough)
        self.frame_loss_fn = nn.BCEWithLogitsLoss()

    def _smooth(self, targets: torch.Tensor) -> torch.Tensor:
        return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

    def forward(
        self,
        clip_logits: torch.Tensor,
        frame_logits: Optional[torch.Tensor],
        targets: torch.Tensor,          # (B,)  float 0/1
    ) -> Tuple[torch.Tensor, dict]:
        smooth_targets = self._smooth(targets)
        clip_loss = self.clip_loss_fn(clip_logits, smooth_targets)

        aux_loss = torch.zeros(1, device=clip_logits.device)
        if frame_logits is not None:
            # Broadcast clip label to every frame
            frame_targets = targets.unsqueeze(1).expand_as(frame_logits)
            aux_loss = self.frame_loss_fn(frame_logits, frame_targets)

        total = clip_loss + self.aux_weight * aux_loss
        return total, {"clip_loss": clip_loss.item(), "aux_loss": aux_loss.item()}


# ──────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = DeepfakeDetector()
    counts = model.param_count()
    print(f"Parameters — total: {counts['total']:,}  trainable: {counts['trainable']:,}")

    # Dummy forward pass
    B, T = 2, 16
    x = torch.randn(B, T, 3, 224, 224)
    clip_logits, frame_logits = model(x)
    print(f"clip_logits:  {clip_logits.shape}")
    print(f"frame_logits: {frame_logits.shape if frame_logits is not None else 'N/A'}")

    # Loss
    loss_fn = DeepfakeLoss()
    targets = torch.randint(0, 2, (B,)).float()
    loss, info = loss_fn(clip_logits, frame_logits, targets)
    print(f"loss: {loss.item():.4f}  {info}")