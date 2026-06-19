"""Pre-norm transformer block (self-attention + MLP)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hvp.attention.rope import Rotary3D
from hvp.attention.sdpa import SelfAttention


class SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: Optional[Rotary3D] = None,
                 mlp_ratio: int = 4, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, rope, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, grid, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), grid, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x
