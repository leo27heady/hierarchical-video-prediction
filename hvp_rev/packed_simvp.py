"""Packed hierarchical state boundary around a frame-to-frame sequence predictor."""
from __future__ import annotations

import torch
import torch.nn as nn

from hvp_rev.faithful_nets import LAT
from hvp_rev.fixed_region_revformer import FixedRegionReconstructor


class HierarchicalPackedTransition(nn.Module):
    """Use a fixed-region state as the recurrent interface of a pixel predictor.

    ``predictor`` follows the OpenSTL convention ``(B,T,C,H,W)->(B,T,C,H,W)``.
    The state itself is ``(B,T,24,16)``: 16 coarse cells followed by eight
    immutable fine-region tokens.
    """

    def __init__(self, ae: nn.Module, adapter: FixedRegionReconstructor,
                 predictor: nn.Module):
        super().__init__()
        self.ae = ae
        self.adapter = adapter
        self.predictor = predictor

    def pack_latents(self, zc: torch.Tensor, zf: torch.Tensor) -> torch.Tensor:
        coarse, regions, _ = self.adapter.pack(zc, zf)
        coarse_tokens = coarse.flatten(3).transpose(2, 3)
        return torch.cat([coarse_tokens, regions], dim=2)

    def unpack_latents(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 4 or state.shape[2:] != (24, LAT):
            raise ValueError(f"expected packed state (B,T,24,{LAT}), got {tuple(state.shape)}")
        b, t = state.shape[:2]
        zc = state[:, :, :16].transpose(2, 3).reshape(b, t, LAT, 4, 4)
        regions = state[:, :, 16:]
        zf = self.adapter.unpack(zc, regions)
        return zc, zf

    def pack_frames(self, frames01: torch.Tensor) -> torch.Tensor:
        b, t = frames01.shape[:2]
        zc, zf = self.ae.encode((frames01 * 2 - 1).reshape(b * t, 1, 64, 64))
        return self.pack_latents(
            zc.reshape(b, t, LAT, 4, 4), zf.reshape(b, t, LAT, 8, 8)
        )

    def unpack_frames(self, state: torch.Tensor) -> torch.Tensor:
        zc, zf = self.unpack_latents(state)
        b, t = zc.shape[:2]
        coarse = self.ae.decode_coarse(zc.reshape(b * t, LAT, 4, 4))
        regions = state[:, :, 16:]
        decoded = coarse.reshape(b, t, 1, 64, 64)
        decoded = decoded + self.adapter.pixel_blend * self.adapter.pixel_residual(regions)
        return (decoded + 1) * 0.5

    def predict_pixels(self, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.unpack_frames(state).contiguous())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.pack_frames(self.predict_pixels(state))

    @property
    def values_per_frame(self) -> int:
        return 24 * LAT
