"""Rule-based attention masks (build-once, cacheable; True = allowed)."""

from __future__ import annotations

import torch


def temporal_causal_mask(T: int, H: int, W: int, device=None) -> torch.Tensor:
    """(N, N) bool, True=allowed. Full attention within a frame, causal across frames
    (token at frame t_i may attend tokens at frame t_j <= t_i)."""
    t = torch.arange(T, device=device).repeat_interleave(H * W)
    return t[:, None] >= t[None, :]


def windowed_causal_mask(T: int, H: int, W: int, t_window: int = 0,
                         s_window: int = 0, device=None) -> torch.Tensor:
    """(N, N) bool, True=allowed — a causal mask with optional temporal + spatial windows.

    Lets a stage's *self*-attention be deliberately starved so it must lean on the parent
    (cross-attention) for the rest — the hierarchy-as-load-bearing test.

    - ``t_window``: how many frames back a query may attend, INCLUDING itself.
      ``0`` (default) = full causal (every frame ``<= t_i``); ``1`` = self-frame only
      (no temporal context → motion must come from the parent); ``k`` = frames ``[t_i-k+1 .. t_i]``.
    - ``s_window``: side of the square spatial neighbourhood a query may attend (within the
      allowed frames). ``0`` = global spatial; ``k`` = a ``(2*(k//2)+1)`` window centred on
      the query cell (e.g. ``8`` → a 9×9 local window).

    The query's own cell is always allowed, so no row is ever fully masked (SDPA-safe).
    ``windowed_causal_mask(T,H,W,0,0) == temporal_causal_mask(T,H,W)``.
    """
    t = torch.arange(T, device=device).repeat_interleave(H * W)
    hh = torch.arange(H, device=device).repeat_interleave(W).repeat(T)
    ww = torch.arange(W, device=device).repeat(H * T)
    dt = t[:, None] - t[None, :]
    allowed = dt >= 0
    if t_window and t_window > 0:
        allowed = allowed & (dt < t_window)
    if s_window and s_window > 0:
        r = s_window // 2
        dh = (hh[:, None] - hh[None, :]).abs()
        dw = (ww[:, None] - ww[None, :]).abs()
        allowed = allowed & (dh <= r) & (dw <= r)
    return allowed
