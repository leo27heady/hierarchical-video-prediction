"""Factorized 3D RoPE (t, h, w). Head dim is split into three contiguous chunks; each
chunk is rotated by its own axis position so attention sees *relative* (t,h,w) offsets —
the motion-modeling lever (content can move across cells). cos/sin are built in fp32 and
cached per (T,H,W); rotation is applied per-chunk (a global rotate_half would mix axes)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def default_split(head_dim: int) -> Tuple[int, int, int]:
    """Time-heavy split into three even chunks summing to head_dim."""
    base = (head_dim // 3) // 2 * 2
    d_h = d_w = base
    d_t = head_dim - 2 * base
    if d_t % 2 != 0:  # keep all chunks even
        d_t -= 1
        d_w += 1
    return d_t, d_h, d_w


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Rotary3D(nn.Module):
    def __init__(self, head_dim: int, dims: Tuple[int, int, int] | None = None, base: float = 10000.0):
        super().__init__()
        self.dims = dims or default_split(head_dim)
        assert sum(self.dims) == head_dim and all(d % 2 == 0 for d in self.dims), (head_dim, self.dims)
        self.base = base
        self._cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _axis(self, name: str, d: int, pos: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if d == 0:
            z = torch.zeros(pos.shape[0], 0, device=device)
            return z, z
        inv = 1.0 / (self.base ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
        ang = pos.float()[:, None] * inv[None, :]            # (N, d/2)
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1)      # (N, d)
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
        return cos, sin

    def cos_sin(self, T: int, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (T, H, W, str(device))
        if key not in self._cache:
            t = torch.arange(T, device=device).repeat_interleave(H * W)
            h = torch.arange(H, device=device).repeat_interleave(W).repeat(T)
            w = torch.arange(W, device=device).repeat(T * H)
            cs = [self._axis(n, d, p, device) for n, d, p in
                  zip("thw", self.dims, (t, h, w))]
            cos = torch.cat([c for c, _ in cs], dim=-1)      # (N, head_dim)
            sin = torch.cat([s for _, s in cs], dim=-1)
            self._cache[key] = (cos, sin)
        return self._cache[key]

    def apply(self, q: torch.Tensor, k: torch.Tensor, T: int, H: int, W: int):
        """q,k: (B, heads, N, head_dim). Returns rotated (q, k)."""
        cos, sin = self.cos_sin(T, H, W, q.device)
        cos = cos.to(q.dtype)[None, None]                    # (1,1,N,D)
        sin = sin.to(q.dtype)[None, None]
        d_t, d_h, d_w = self.dims

        def rot(x):
            parts, off = [], 0
            full_cos = cos
            full_sin = sin
            # per-chunk rotate_half so axes don't mix
            xs = x.split([d_t, d_h, d_w], dim=-1)
            cs = full_cos.split([d_t, d_h, d_w], dim=-1)
            ss = full_sin.split([d_t, d_h, d_w], dim=-1)
            for xi, ci, si in zip(xs, cs, ss):
                parts.append(xi * ci + _rotate_half(xi) * si)
            return torch.cat(parts, dim=-1)

        return rot(q), rot(k)
