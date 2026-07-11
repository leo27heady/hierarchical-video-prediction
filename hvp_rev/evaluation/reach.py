"""Geometry-derived context reach and pair-specific causal bands."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContextGeometry:
    name: str
    coarse_slots: int = 0
    coarse_stride: int = 1
    fine_recent: int = 0
    fine_history_slots: int = 0
    fine_history_stride: int = 4
    top_slots: int = 0
    top_stride: int = 1

    @property
    def reach(self) -> int:
        coarse = self.coarse_slots * self.coarse_stride
        top = self.top_slots * self.top_stride
        if self.fine_history_slots:
            # Matches the legacy anchor rule: recent frames plus increasingly old stride slots.
            hist = self.fine_recent + 2 + self.fine_history_stride * (self.fine_history_slots - 1)
        else:
            hist = 0
        return max(coarse, top, self.fine_recent, hist)


def pair_specific_reach_mask(
    dt_since_visible: np.ndarray,
    longer: ContextGeometry,
    shorter: ContextGeometry,
    *,
    answerable: np.ndarray | None = None,
    foreground_nonempty: np.ndarray | None = None,
) -> np.ndarray:
    """Records visible to ``longer`` but outside ``shorter``'s addressable past.

    Arm order is checked rather than trusted, preventing a silently empty or reversed band.
    """
    if longer.reach <= shorter.reach:
        raise ValueError(
            f"{longer.name} reach ({longer.reach}) must exceed {shorter.name} ({shorter.reach})"
        )
    dt = np.asarray(dt_since_visible)
    mask = (dt > shorter.reach) & (dt <= longer.reach) & np.isfinite(dt)
    if answerable is not None:
        mask &= np.asarray(answerable, dtype=bool)
    if foreground_nonempty is not None:
        mask &= np.asarray(foreground_nonempty, dtype=bool)
    return mask
