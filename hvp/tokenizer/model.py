"""HierTokenizer — continuous (VQ-free) hierarchical video autoencoder.

encode(x) -> per-stage continuous latents (fine->coarse).
forward(x, max_depth) -> deep-supervised recons + latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from hvp.config import Config, TopologyConfig
from hvp.tokenizer.decoder import HierDecoder
from hvp.tokenizer.encoder import HierEncoder


@dataclass
class TokenizerOut:
    recons: List[torch.Tensor]   # one per active depth (coarse->fine); recons[-1] is the full decode
    z: List[torch.Tensor]        # per-stage latents, fine->coarse (bot, mid, top)

    @property
    def recon(self) -> torch.Tensor:
        return self.recons[-1]


class HierTokenizer(nn.Module):
    def __init__(self, topo: TopologyConfig) -> None:
        super().__init__()
        self.topo = topo
        self.topology = topo  # alias used by trainer/probes
        self.encoder = HierEncoder(topo)
        self.decoder = HierDecoder(topo)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def encode_grad(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, max_depth: Optional[int] = None) -> TokenizerOut:
        z = self.encoder(x)
        depth = self.topo.n_stages() if max_depth is None else max(1, min(max_depth, self.topo.n_stages()))
        recons = self.decoder(z, depth)
        return TokenizerOut(recons=recons, z=z)


def build_tokenizer(cfg: Config) -> HierTokenizer:
    return HierTokenizer(cfg.topology)
