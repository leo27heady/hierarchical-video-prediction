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

from hvp.attention.masks import temporal_causal_mask, windowed_causal_mask
from hvp.attention.rope import Rotary3D
from hvp.attention.sdpa import SelfAttention, _rms_norm
from hvp.config import TopologyConfig
from hvp.predictor.heads import head_out_dim
from hvp.predictor.masks import CascadeMasks


class CrossAttention(nn.Module):
    """Child tokens (query, dim) attend parent hidden tokens (key/value, parent_dim) under a bool mask.
    parent_dim may differ from dim (per-stage capacity) — the kv projection handles the mismatch."""

    def __init__(self, dim: int, parent_dim: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim, self.qk_norm = n_heads, dim // n_heads, qk_norm
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(parent_dim, 2 * dim, bias=False)
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
    def __init__(self, dim: int, n_heads: int, rope: Rotary3D, has_parent: bool, parent_dim: int,
                 mlp_ratio: int = 4, qk_norm: bool = True, dropout: float = 0.0, s_window: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, rope, qk_norm=qk_norm, s_window=s_window)
        self.has_parent = has_parent
        if has_parent:
            self.norm_c = nn.LayerNorm(dim)
            self.cross = CrossAttention(dim, parent_dim, n_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x, grid, self_mask, parent=None, cross_mask=None):
        x = x + self.attn(self.norm1(x), grid, self_mask)
        if self.has_parent and parent is not None:
            x = x + self.cross(self.norm_c(x), parent, cross_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CascadeStagePredictor(nn.Module):
    def __init__(self, in_ch: int, thw: Tuple[int, int, int], *, dim: int, n_heads: int,
                 n_layers: int, head: str, has_parent: bool, parent_dim: int,
                 mlp_ratio: int = 4, qk_norm: bool = True, dropout: float = 0.0, rope_base: float = 10000.0,
                 t_window: int = 0, s_window: int = 0, head_components: int = 4,
                 grad_checkpoint: bool = False):
        super().__init__()
        T, H, W = thw
        self.thw, self.in_ch, self.head_kind, self.head_components = thw, in_ch, head, head_components
        self.grad_checkpoint = grad_checkpoint
        self.proj_in = nn.Linear(in_ch, dim)
        self.spatial = nn.Parameter(torch.zeros(H * W, dim))
        rope = Rotary3D(dim // n_heads, base=rope_base)
        self.blocks = nn.ModuleList(
            CascadeStageBlock(dim, n_heads, rope, has_parent, parent_dim, mlp_ratio=mlp_ratio,
                              qk_norm=qk_norm, dropout=dropout, s_window=s_window)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(dim)
        if head == "warp":
            # Explicit-motion head: predict a per-cell 2D flow and WARP the current latent (so detail is
            # resampled, not regressed -> stays sharp), plus a small residual for disocclusion / content
            # change. Zero-init both heads so training STARTS at identity-warp = persistence, then learns
            # motion. The warp samples the (detached) current latent, so it adds no encoder gradient.
            self.flow_head = nn.Linear(dim, 2)
            self.residual_head = nn.Linear(dim, in_ch)
            self.flow_scale = 1.5
            for m in (self.flow_head, self.residual_head):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
            self.register_buffer("base_grid", torch.stack([xs, ys], dim=-1), persistent=False)  # (H,W,2) (x,y)
        else:
            self.head = nn.Linear(dim, head_out_dim(head, in_ch, head_components))
        # Block-windowed self-attention (s_window>0): SelfAttention partitions into w x w spatial blocks
        # and attends within each, so the mask is the per-block (T*w*w)^2 causal mask (spatial handled by
        # partitioning -> s_window=0 in the mask). Otherwise the full-grid temporal-causal mask.
        if s_window and 0 < s_window < H and H % s_window == 0:
            self.register_buffer("self_mask", windowed_causal_mask(T, s_window, s_window, t_window, 0), persistent=False)
        else:
            self.register_buffer("self_mask", windowed_causal_mask(T, H, W, t_window, 0), persistent=False)

    def forward(self, feat: torch.Tensor, parent: Optional[torch.Tensor], cross_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = feat.shape
        x = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x = self.proj_in(x) + self.spatial.repeat(T, 1)[None]
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, (T, H, W), self.self_mask, parent, cross_mask, use_reentrant=False)
            else:
                x = blk(x, (T, H, W), self.self_mask, parent=parent, cross_mask=cross_mask)
        hidden = self.norm(x)
        if self.head_kind == "warp":
            h = hidden.reshape(B, T, H, W, -1)
            flow = torch.tanh(self.flow_head(h)) * self.flow_scale          # (B,T,H,W,2) grid offsets
            grid = self.base_grid[None] + flow.reshape(B * T, H, W, 2)
            feat_bt = feat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)    # current latent, per frame
            warped = F.grid_sample(feat_bt, grid, mode="bilinear", padding_mode="border", align_corners=False)
            warped = warped.reshape(B, T, C, H, W).permute(0, 1, 3, 4, 2)    # (B,T,H,W,C)
            pred = warped + self.residual_head(h)
        else:
            pred = self.head(hidden).reshape(B, T, H, W, -1)
        return pred, hidden


class HierarchicalCascade(nn.Module):
    """Per-stage predictors (fine->coarse list) wired into a coarse->fine single-step cascade."""

    def __init__(self, topo: TopologyConfig, pcfg) -> None:
        super().__init__()
        self.N = topo.n_stages()
        self.detach_parent = pcfg.detach_parent
        self.default_cascade = pcfg.use_cascade and pcfg.conditioning != "none"
        dims, heads, layers = pcfg.resolved(self.N)
        tw, sw = pcfg.windows(self.N)
        spatial, temporal = topo.stage_spatial(), topo.stage_temporal()
        thw = [(temporal[s], spatial[s], spatial[s]) for s in range(self.N)]
        self.masks = CascadeMasks(thw)
        self.stages = nn.ModuleList(
            CascadeStagePredictor(
                topo.stage_latent_ch[s], thw[s], dim=dims[s], n_heads=heads[s], n_layers=layers[s],
                head=pcfg.head, has_parent=(s < self.N - 1),
                parent_dim=dims[s + 1] if s < self.N - 1 else dims[s],
                mlp_ratio=pcfg.mlp_ratio, qk_norm=pcfg.qk_norm, dropout=pcfg.dropout,
                rope_base=pcfg.rope_base, t_window=tw[s], s_window=sw[s],
                head_components=pcfg.head_components, grad_checkpoint=pcfg.grad_checkpoint,
            )
            for s in range(self.N)
        )

    def forward(self, z: List[torch.Tensor], use_cascade: Optional[bool] = None) -> List[torch.Tensor]:
        """z fine->coarse [bot, mid, top]; returns per-stage predictions (B,T,H,W,out)."""
        use_cascade = self.default_cascade if use_cascade is None else use_cascade
        preds: List[Optional[torch.Tensor]] = [None] * self.N
        parent_hidden = None
        for s in range(self.N - 1, -1, -1):              # top -> mid -> bot
            cond = parent_hidden if (use_cascade and s < self.N - 1) else None
            cross_mask = self.masks.cross(s) if cond is not None else None
            pred, hidden = self.stages[s](z[s], cond, cross_mask)
            preds[s] = pred
            parent_hidden = hidden.detach() if self.detach_parent else hidden
        return preds  # type: ignore
