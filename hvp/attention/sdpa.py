"""Self-attention via PyTorch SDPA (FlashAttention on CUDA).

Mask handling: a boolean ``True=allowed`` mask is passed straight to SDPA (it converts
internally). qk_norm (parameter-free RMSNorm on q,k before RoPE) bounds attention-logit
magnitude under AMP — the repo's known NaN mitigation.

Optional **block-windowed** spatial attention (``s_window``): partition the (T,H,W) grid into
non-overlapping ``s_window x s_window`` spatial blocks and attend within each block (across causal
time). This is TRUE windowing — the score matrix is per-block ``(T*w*w)^2`` instead of the global
``(T*H*W)^2`` — so it actually cuts memory (a dense windowed *mask* does not). Cross-window / global
context is supplied by the hierarchy's cross-attention (coarse->fine), so BOT can stay local.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp.attention.rope import Rotary3D


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: Optional[Rotary3D] = None, qk_norm: bool = True,
                 s_window: int = 0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qk_norm = qk_norm
        self.rope = rope
        self.s_window = s_window          # 0 = global; w>0 = non-overlapping w x w spatial blocks
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def _attn_core(self, x: torch.Tensor, gT: int, gH: int, gW: int,
                   attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Attention over a flat sequence whose grid is (gT,gH,gW). x: (B', gT*gH*gW, C)."""
        Bp, M, _ = x.shape
        qkv = self.qkv(x).reshape(Bp, M, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # (B', heads, M, head_dim)
        if self.qk_norm:
            q, k = _rms_norm(q), _rms_norm(k)
        if self.rope is not None:
            q, k = self.rope.apply(q, k, gT, gH, gW)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out(o.transpose(1, 2).reshape(Bp, M, -1))

    def _windowed(self, x: torch.Tensor, T: int, H: int, W: int, w: int,
                  block_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Non-overlapping w x w spatial blocks; attend within each (across causal time)."""
        B, C = x.shape[0], x.shape[-1]
        Hb, Wb = H // w, W // w
        x = x.reshape(B, T, Hb, w, Wb, w, C)
        x = x.permute(0, 2, 4, 1, 3, 5, 6).reshape(B * Hb * Wb, T * w * w, C)   # (B*nWin, T*w*w, C)
        o = self._attn_core(x, T, w, w, block_mask)                             # block grid (T,w,w)
        o = o.reshape(B, Hb, Wb, T, w, w, C).permute(0, 3, 1, 4, 2, 5, 6)
        return o.reshape(B, T * H * W, C)

    def forward(self, x: torch.Tensor, grid, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T, H, W = grid
        w = self.s_window
        if w and 0 < w < H and H % w == 0 and W % w == 0:
            return self._windowed(x, T, H, W, w, attn_mask)   # attn_mask is the (T*w*w)^2 block mask
        return self._attn_core(x, T, H, W, attn_mask)
