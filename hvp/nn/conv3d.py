"""Causal 3D conv primitives (transplanted from SEED-Voken improved_video_model.py).

Causality is enforced by an asymmetric LEFT-only time pad of ``K-1`` (nothing on the
right), so frame ``t`` only sees frames ``<= t``. Spatial padding stays symmetric.
A temporal stride-2 conv maps ``T -> (T-1)//2 + 1`` (the ``1 + s*t`` convention), and
``Upsampler3d`` inverts it (``T -> 2T``, then drop the first ``s-1`` frames -> ``2T-1``...
i.e. ``3 -> 5 -> 9`` on decode).

Norm: standard ``nn.GroupNorm`` (autocast keeps it in fp32) rather than the custom
FrameWiseGroupNorm that gave AMP instability in repo2 (WF-VAE result).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FrameNorm3d(nn.Module):
    """Per-frame GroupNorm for causal video: normalize each frame independently over
    (C-group, H, W) so statistics never mix across time (a plain GroupNorm over the
    5-D tensor would pool over T and silently break causality). GroupNorm2d is kept in
    fp32 by autocast, so this is also AMP-stable (the WF-VAE concern)."""

    def __init__(self, channels: int, max_groups: int = 8) -> None:
        super().__init__()
        groups = max(1, math.gcd(channels, max_groups))
        self.gn = nn.GroupNorm(groups, channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.transpose(1, 2).reshape(b * t, c, h, w)
        x = self.gn(x)
        return x.reshape(b, t, c, h, w).transpose(1, 2)


def group_norm(channels: int, max_groups: int = 8) -> FrameNorm3d:
    """Factory kept under the old name; returns the per-frame (causal) norm."""
    return FrameNorm3d(channels, max_groups)


class CausalConv3d(nn.Module):
    """Conv3d with causal (left-only) temporal padding + symmetric spatial padding."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        k = _triple(kernel_size)
        s = _triple(stride)
        h_pad, w_pad = k[1] // 2, k[2] // 2
        t_pad = k[0] - 1
        # F.pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
        self.pad = (w_pad, w_pad, h_pad, h_pad, t_pad, 0)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.pad, mode="constant"))


class ResBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = group_norm(in_ch)
        self.conv1 = CausalConv3d(in_ch, out_ch, 3, bias=False)
        self.norm2 = group_norm(out_ch)
        self.conv2 = CausalConv3d(out_ch, out_ch, 3, bias=False)
        self.skip = (
            CausalConv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return h + self.skip(x)


def depth_to_space3d(x: torch.Tensor, block_size: Tuple[int, int, int]) -> torch.Tensor:
    """Depth-to-space (DCR) over (T,H,W). Splits ``block_t*block_h*block_w`` from channels."""
    c, t, h, w = x.shape[-4:]
    bt, bh, bw = block_size
    s = bt * bh * bw
    if c % s != 0:
        raise ValueError(f"C={c} not divisible by block product {s}")
    outer = x.shape[:-4]
    x = x.view(-1, bt, bh, bw, c // s, t, h, w)
    x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)
    return x.contiguous().view(*outer, c // s, t * bt, h * bh, w * bw)


class Upsampler3d(nn.Module):
    """Causal upsample by ``block_size``. Temporal block>1 drops the first ``s-1`` frames
    so ``T -> 2T -> 2T-1`` (the inverse of the causal stride-2 encoder: 3->5, 5->9)."""

    def __init__(self, dim: int, block_size: Tuple[int, int, int]) -> None:
        super().__init__()
        self.block_size = tuple(block_size)
        dim_out = dim
        for b in self.block_size:
            dim_out *= b
        self.conv = CausalConv3d(dim, dim_out, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = depth_to_space3d(self.conv(x), self.block_size)
        if self.block_size[0] > 1:
            if out.shape[2] > 2:
                out = torch.cat([out[:, :, [0]], out[:, :, 2:]], dim=2)
            else:
                out = out[:, :, [0]]
        return out
