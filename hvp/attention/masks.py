"""Rule-based attention masks (build-once, cacheable; True = allowed)."""

from __future__ import annotations

import torch


def temporal_causal_mask(T: int, H: int, W: int, device=None) -> torch.Tensor:
    """(N, N) bool, True=allowed. Full attention within a frame, causal across frames
    (token at frame t_i may attend tokens at frame t_j <= t_i)."""
    t = torch.arange(T, device=device).repeat_interleave(H * W)
    return t[:, None] >= t[None, :]
