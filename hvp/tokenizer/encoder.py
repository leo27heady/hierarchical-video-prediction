"""Hierarchical causal encoder: video -> per-stage continuous latents (NO quantization).

32x32xT9 -> BOT 16x16xT9 -> MID 4x4xT5 -> TOP 1x1xT3 via the bigstep schedule
(``ch_mult``, ``temporal_downsample``). Each tap gets a thin latent head producing a
continuous latent of ``stage_latent_ch`` channels. Stages are returned fine->coarse
(bot, mid, top).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from hvp.config import TopologyConfig
from hvp.nn.conv3d import CausalConv3d, ResBlock3d, group_norm, swish


class _LatentHead(nn.Module):
    def __init__(self, in_ch: int, latent_ch: int) -> None:
        super().__init__()
        self.norm = group_norm(in_ch)
        self.proj = CausalConv3d(in_ch, latent_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(swish(self.norm(x)))


class HierEncoder(nn.Module):
    def __init__(self, topo: TopologyConfig) -> None:
        super().__init__()
        self.topo = topo
        n_levels = len(topo.ch_mult)
        n_down = topo.n_down()
        self.tap_after = tuple(topo.tap_after)

        self.conv_in = CausalConv3d(topo.image_c, topo.level_width(0), 3)

        self.levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = topo.level_width(0)
        for i in range(n_levels):
            blocks = nn.ModuleList()
            cur_in = prev
            for _ in range(topo.num_res_blocks):
                blocks.append(ResBlock3d(cur_in, topo.level_width(i)))
                cur_in = topo.level_width(i)
            self.levels.append(blocks)
            if i < n_down:
                t_stride = 2 if topo.temporal_downsample[i] else 1
                self.downs.append(
                    CausalConv3d(topo.level_width(i), topo.level_width(i), 3,
                                 stride=(t_stride, 2, 2))
                )
            prev = topo.level_width(i)

        # one latent head per tapped stage (fine->coarse)
        self.heads = nn.ModuleList(
            _LatentHead(topo.level_width(tap), lc)
            for tap, lc in zip(topo.tap_after, topo.stage_latent_ch)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.conv_in(x)
        tap_set = set(self.tap_after)
        taps: dict[int, torch.Tensor] = {}
        for i, blocks in enumerate(self.levels):
            for blk in blocks:
                h = blk(h)
            if i < len(self.downs):
                h = self.downs[i](h)
                if i in tap_set:
                    taps[i] = h
        return [head(taps[tap]) for head, tap in zip(self.heads, self.tap_after)]
