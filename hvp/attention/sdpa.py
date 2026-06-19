"""Self-attention via PyTorch SDPA (FlashAttention on CUDA).

Mask handling: a boolean ``True=allowed`` mask is passed straight to SDPA (it converts
internally). qk_norm (parameter-free RMSNorm on q,k before RoPE) bounds attention-logit
magnitude under AMP — the repo's known NaN mitigation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp.attention.rope import Rotary3D


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: Optional[Rotary3D] = None, qk_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qk_norm = qk_norm
        self.rope = rope
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, grid, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        T, H, W = grid
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # (B, heads, N, head_dim)
        if self.qk_norm:
            q, k = _rms_norm(q), _rms_norm(k)
        if self.rope is not None:
            q, k = self.rope.apply(q, k, T, H, W)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, N, -1)
        return self.out(o)
