"""Hierarchical causal decoder: per-stage latents -> pixels, with deep supervision.

Top-down fuse: start from the coarsest latent, upsample (causal temporal+spatial) and
add the next finer latent, down to the bot grid, then render to pixels. For deep
supervision, each depth ``d`` (top-only, top+mid, top+mid+bot) is fused over its active
coarse stages and rendered at full resolution — so every stage is grounded by recon.

Upsample block sizes are derived from the topology's per-stage spatial/temporal sizes.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from hvp.config import TopologyConfig
from hvp.nn.conv3d import CausalConv3d, ResBlock3d, Upsampler3d, group_norm, swish


class HierDecoder(nn.Module):
    def __init__(self, topo: TopologyConfig) -> None:
        super().__init__()
        self.topo = topo
        self.N = topo.n_stages()
        w = topo.decode_width
        spatial = topo.stage_spatial()   # (bot, mid, top) = index 0..N-1 (fine->coarse)
        temporal = topo.stage_temporal()

        # per-stage projection latent_ch -> decode_width
        self.proj = nn.ModuleList(
            CausalConv3d(lc, w, 1) for lc in topo.stage_latent_ch
        )

        # ups[s] maps grid s -> grid s-1 (coarse stage s down toward finer s-1), s in 1..N-1
        self.ups = nn.ModuleList()
        for s in range(1, self.N):
            spatial_ratio = spatial[s - 1] // spatial[s]
            t_block = 2 if temporal[s - 1] != temporal[s] else 1
            self.ups.append(Upsampler3d(w, (t_block, spatial_ratio, spatial_ratio)))

        # render: bot grid -> pixels
        render_ratio = topo.image_size // spatial[0]
        self.render_up = Upsampler3d(w, (1, render_ratio, render_ratio))
        self.render_res = ResBlock3d(w, w)
        self.render_norm = group_norm(w)
        self.render_out = CausalConv3d(w, topo.image_c, 3)

    def _render(self, h: torch.Tensor) -> torch.Tensor:
        h = self.render_up(h)
        h = self.render_res(h)
        h = self.render_out(swish(self.render_norm(h)))
        return h

    def decode_depth(self, z: List[torch.Tensor], depth: int) -> torch.Tensor:
        """Fuse the ``depth`` coarsest stages, render at full resolution.

        depth=1 -> top only; depth=2 -> top+mid; depth=N -> all.
        """
        N = self.N
        h = self.proj[N - 1](z[N - 1])                 # coarsest (top)
        finest_active = N - depth                       # stage index where fuse stops
        for s in range(N - 1, finest_active, -1):       # fuse down to finest_active
            h = self.ups[s - 1](h)
            h = h + self.proj[s - 1](z[s - 1])
        for s in range(finest_active, 0, -1):           # upsample remainder to bot grid
            h = self.ups[s - 1](h)
        return self._render(h)

    def forward(self, z: List[torch.Tensor], max_depth: int) -> List[torch.Tensor]:
        return [self.decode_depth(z, d) for d in range(1, max_depth + 1)]
