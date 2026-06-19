"""Single-step hierarchical BLT cascade (the dissertation contribution).

One forward pass, coarse->fine: predict TOP, its hidden conditions MID via cross-attention
(leak-free RF-overlap mask), MID's hidden conditions BOT. Each stage is a causal-temporal
+ 3D-RoPE transformer predicting the next-frame latent (teacher-forced t->t+1). Parent
hidden is detached by default (each stage shaped by its own loss; the cascade only carries
information forward, not collapse gradients)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp.attention.masks import temporal_causal_mask
from hvp.attention.rope import Rotary3D
from hvp.attention.sdpa import SelfAttention, _rms_norm
from hvp.config import TopologyConfig
from hvp.predictor.masks import CascadeMasks


class CrossAttention(nn.Module):
    """Child tokens (query) attend parent hidden tokens (key/value) under a bool mask."""

    def __init__(self, dim: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim, self.qk_norm = n_heads, dim // n_heads, qk_norm
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, parent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, Nc, _ = x.shape
        Np = parent.shape[1]
        q = self.q(x).reshape(B, Nc, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(parent).reshape(B, Np, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        if self.qk_norm:
            q, k = _rms_norm(q), _rms_norm(k)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        return self.out(o.transpose(1, 2).reshape(B, Nc, -1))


class CascadeStageBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: Rotary3D, has_parent: bool,
                 mlp_ratio: int = 4, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, rope, qk_norm=qk_norm)
        self.has_parent = has_parent
        if has_parent:
            self.norm_c = nn.LayerNorm(dim)
            self.cross = CrossAttention(dim, n_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x, grid, self_mask, parent=None, cross_mask=None):
        x = x + self.attn(self.norm1(x), grid, self_mask)
        if self.has_parent and parent is not None:
            x = x + self.cross(self.norm_c(x), parent, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CascadeStagePredictor(nn.Module):
    def __init__(self, in_ch: int, thw: Tuple[int, int, int], *, dim: int, n_heads: int,
                 n_layers: int, head: str, has_parent: bool, qk_norm: bool = True):
        super().__init__()
        T, H, W = thw
        self.thw, self.in_ch, self.head_kind = thw, in_ch, head
        self.proj_in = nn.Linear(in_ch, dim)
        self.spatial = nn.Parameter(torch.zeros(H * W, dim))
        rope = Rotary3D(dim // n_heads)
        self.blocks = nn.ModuleList(
            CascadeStageBlock(dim, n_heads, rope, has_parent, qk_norm=qk_norm) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, in_ch if head == "l2" else 2 * in_ch)
        self.register_buffer("self_mask", temporal_causal_mask(T, H, W), persistent=False)

    def forward(self, feat: torch.Tensor, parent: Optional[torch.Tensor], cross_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        x = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x = self.proj_in(x) + self.spatial.repeat(T, 1)[None]
        for blk in self.blocks:
            x = blk(x, (T, H, W), self.self_mask, parent=parent, cross_mask=cross_mask)
        hidden = self.norm(x)
        pred = self.head(hidden).reshape(B, T, H, W, -1)
        return pred, hidden


class HierarchicalCascade(nn.Module):
    """Per-stage predictors (fine->coarse list) wired into a coarse->fine single-step cascade."""

    def __init__(self, topo: TopologyConfig, pcfg) -> None:
        super().__init__()
        self.N = topo.n_stages()
        self.detach_parent = pcfg.detach_parent
        spatial, temporal = topo.stage_spatial(), topo.stage_temporal()
        thw = [(temporal[s], spatial[s], spatial[s]) for s in range(self.N)]
        self.masks = CascadeMasks(thw)
        self.stages = nn.ModuleList(
            CascadeStagePredictor(
                topo.stage_latent_ch[s], thw[s], dim=pcfg.dim, n_heads=pcfg.n_heads,
                n_layers=pcfg.n_layers, head=pcfg.head, has_parent=(s < self.N - 1),
                qk_norm=pcfg.qk_norm,
            )
            for s in range(self.N)
        )

    def forward(self, z: List[torch.Tensor], use_cascade: bool = True) -> List[torch.Tensor]:
        """z fine->coarse [bot, mid, top]; returns per-stage predictions (B,T,H,W,out)."""
        preds: List[Optional[torch.Tensor]] = [None] * self.N
        parent_hidden = None
        for s in range(self.N - 1, -1, -1):              # top -> mid -> bot
            cond = parent_hidden if (use_cascade and s < self.N - 1) else None
            cross_mask = self.masks.cross(s) if cond is not None else None
            pred, hidden = self.stages[s](z[s], cond, cross_mask)
            preds[s] = pred
            parent_hidden = hidden.detach() if self.detach_parent else hidden
        return preds  # type: ignore
