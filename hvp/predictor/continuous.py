"""Single-stage continuous next-frame predictor (the BLT cascade's per-stage worker).

Given a stage's continuous latent ``(B, C, T, H, W)``, predict the next-frame latent with
full within-frame self-attention + factorized 3D RoPE (motion modeling), causal across
time. Heads: ``l2`` (deterministic MSE) or ``gaussian`` (per-dim NLL). Optional additive
cross-stage conditioning input (the cascade uses cross-attention; this is the fallback)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp.attention.blocks import SelfAttnBlock
from hvp.attention.masks import temporal_causal_mask
from hvp.attention.rope import Rotary3D
from hvp.predictor.heads import head_mean, head_nll


class ContinuousStagePredictor(nn.Module):
    def __init__(self, in_ch: int, T: int, H: int, W: int, *, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 3, head: str = "l2", qk_norm: bool = True, cond_dim: int = 0):
        super().__init__()
        self.in_ch, self.T, self.H, self.W = in_ch, T, H, W
        self.head_kind = head
        self.proj_in = nn.Linear(in_ch, dim)
        self.spatial = nn.Parameter(torch.zeros(H * W, dim))
        self.cond_proj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None
        rope = Rotary3D(dim // n_heads)
        self.blocks = nn.ModuleList(SelfAttnBlock(dim, n_heads, rope, qk_norm=qk_norm) for _ in range(n_layers))
        self.norm = nn.LayerNorm(dim)
        out = in_ch if head == "l2" else 2 * in_ch
        self.head = nn.Linear(dim, out)
        self.register_buffer("mask", temporal_causal_mask(T, H, W), persistent=False)

    def forward(self, feat: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, T, H, W = feat.shape
        x = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x = self.proj_in(x) + self.spatial.repeat(T, 1)[None]
        if cond is not None and self.cond_proj is not None:
            x = x + self.cond_proj(cond.reshape(B, T * H * W, -1))
        for blk in self.blocks:
            x = blk(x, (T, H, W), self.mask)
        x = self.head(self.norm(x))
        return x.reshape(B, T, H, W, -1)


def _target(feat: torch.Tensor) -> torch.Tensor:
    """(B,C,T,H,W) -> (B,T-1,H,W,C): position t supervises frame t+1."""
    return feat[:, :, 1:].permute(0, 2, 3, 4, 1)


def predictive_loss(pred: torch.Tensor, feat: torch.Tensor, head: str = "l2",
                    n_components: int = 4, shift: int = 1) -> torch.Tensor:
    """Supervise position t's prediction against frame t+shift. shift=1 = next-frame (default);
    shift>1 = multi-step-ahead (the self-rollout term predicts shift=k+2 after k feedback steps)."""
    tgt = feat[:, :, shift:].permute(0, 2, 3, 4, 1)
    return head_nll(pred[:, :-shift], tgt, head, feat.shape[1], n_components)


@torch.no_grad()
def variance_explained(pred: torch.Tensor, feat: torch.Tensor, changed_frac: float = 0.5,
                       head: str = "l2", n_components: int = 4) -> dict:
    """VE = 1 - MSE(pred, next)/MSE(persistence, next). Reported overall and on the most
    'changed' positions (where persistence is a hard baseline). Uses the head's point estimate."""
    C = feat.shape[1]
    tgt = _target(feat)                       # (B,T-1,H,W,C)
    cur = feat[:, :, :-1].permute(0, 2, 3, 4, 1)
    mu = head_mean(pred[:, :-1], head, C, n_components)
    # per-position squared errors, reduced over channels
    err_pred = (mu - tgt).pow(2).mean(dim=-1)         # (B,T-1,H,W)
    err_persist = (cur - tgt).pow(2).mean(dim=-1)
    change = (tgt - cur).pow(2).mean(dim=-1)          # how much each position moves

    def ve(mask: torch.Tensor) -> float:
        ep = err_pred[mask].sum()
        e0 = err_persist[mask].sum()
        return float(1.0 - ep / (e0 + 1e-12)) if e0 > 0 else 0.0

    flat_change = change.flatten()
    thr = torch.quantile(flat_change, 1.0 - changed_frac) if flat_change.numel() else change.new_tensor(0.0)
    changed = change >= thr
    allmask = torch.ones_like(changed, dtype=torch.bool)
    return {
        "ve_all": ve(allmask),
        "ve_changed": ve(changed),
        "mse_pred": float(err_pred.mean()),
        "mse_persist": float(err_persist.mean()),
        "frac_changed": float(changed.float().mean()),
    }
