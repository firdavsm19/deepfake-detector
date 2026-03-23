%%writefile /content/model.py
# ============================================================
#  model.py — Wav2Vec2 backbone + AASIST classification head
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import config


# ─────────────────────────────────────────────────────────────
#  Graph Attention (simplified AASIST-style)
# ─────────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Single-head graph attention over a sequence of node features.
    Each frame is a node; attention weights are learned pairwise.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.a   = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky   = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, in_dim)
        returns : (B, T, out_dim)
        """
        h = self.W(x)                              # (B, T, out_dim)
        B, T, D = h.shape

        # Build all (i, j) pairs efficiently
        hi = h.unsqueeze(2).expand(-1, -1, T, -1)  # (B, T, T, D)
        hj = h.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, T, D)
        e  = self.leaky(self.a(torch.cat([hi, hj], dim=-1)).squeeze(-1))  # (B, T, T)

        alpha = F.softmax(e, dim=-1)               # (B, T, T)
        alpha = self.dropout(alpha)

        out = torch.bmm(alpha, h)                  # (B, T, out_dim)
        return F.elu(out)


# ─────────────────────────────────────────────────────────────
#  AASIST-style backend
# ─────────────────────────────────────────────────────────────
class AASISTBackend(nn.Module):
    """
    Spectro-temporal graph attention backend.

    Input  : projected features  (B, T, proj_dim)
    Output : logits              (B, 2)
    """

    def __init__(self, in_dim: int = 128):
        super().__init__()

        # Graph attention layers
        self.gat1 = GraphAttentionLayer(in_dim, 128)
        self.gat2 = GraphAttentionLayer(128, 64)

        # Temporal attention pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Classifier
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, in_dim)"""
        x = self.gat1(x)                            # (B, T, 128)
        x = self.gat2(x)                            # (B, T, 64)

        # Weighted pooling over time
        w = torch.softmax(self.pool_attn(x), dim=1) # (B, T, 1)
        x = (w * x).sum(dim=1)                      # (B, 64)

        return self.head(x)                          # (B, 2)


# ─────────────────────────────────────────────────────────────
#  Full model
# ─────────────────────────────────────────────────────────────
class VoiceDetector(nn.Module):
    """
    Wav2Vec2 (frozen or fine-tuned) → projection → AASIST backend.
    """

    def __init__(self):
        super().__init__()

        # Pre-trained Wav2Vec2 backbone
        self.backbone  = Wav2Vec2Model.from_pretrained(config.BACKBONE)
        self.proj      = nn.Sequential(
            nn.Linear(config.BACKBONE_DIM, config.PROJ_DIM),
            nn.LayerNorm(config.PROJ_DIM),
            nn.GELU(),
        )
        self.backend   = AASISTBackend(in_dim=config.PROJ_DIM)

        if config.FREEZE_BACKBONE:
            self._freeze_backbone()

    # ── forward ───────────────────────────────────────────────
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav  : (B, T)  raw waveform, 16 kHz, normalised to [-1, 1]
        returns logits (B, 2)
        """
        if config.FREEZE_BACKBONE:
            with torch.no_grad():
                feats = self.backbone(wav).last_hidden_state  # (B, T', 768)
        else:
            feats = self.backbone(wav).last_hidden_state

        x = self.proj(feats)      # (B, T', proj_dim)
        return self.backend(x)    # (B, 2)

    # ── helpers ───────────────────────────────────────────────
    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("🔒 Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔓 Backbone unfrozen.")

    def trainable_params(self) -> list:
        if config.FREEZE_BACKBONE:
            return list(self.proj.parameters()) + list(self.backend.parameters())
        return list(self.parameters())

    def count_params(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Quick sanity-check ────────────────────────────────────────
if __name__ == "__main__":
    model = VoiceDetector().to(config.DEVICE)
    info  = model.count_params()
    print(f"Total params    : {info['total']:,}")
    print(f"Trainable params: {info['trainable']:,}")

    dummy = torch.randn(2, config.MAX_LEN).to(config.DEVICE)
    out   = model(dummy)
    print(f"Output shape    : {out.shape}")   # (2, 2)