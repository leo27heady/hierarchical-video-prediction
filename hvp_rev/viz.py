"""Save decoded prediction sample grids as PNGs, so we REVIEW images (not just scalars — scalars hid the RHCP fade and
the EBT mean-blur). Each row = one clip; columns = labeled panels (e.g. last-context | prediction | GT)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _to_u8(t):
    """(B,1,H,W) or (B,H,W) tensor/array in [-1,1] or [0,1] -> (B,H,W) uint8."""
    import torch
    if hasattr(t, "detach"):
        t = t.detach().float().cpu().numpy()
    t = np.asarray(t)
    if t.ndim == 4:
        t = t[:, 0]
    if t.min() < -0.01:                      # [-1,1] -> [0,1]
        t = (t + 1) * 0.5
    return (np.clip(t, 0, 1) * 255).astype(np.uint8)


def save_samples(path: str, panels: List[Tuple[str, "any"]], n: int = 8, gap: int = 2):
    """panels: list of (label, batch of frames). Saves an n-row x len(panels)-col grid PNG. Labels are printed to
    stdout (PIL text is optional). Returns the path."""
    from PIL import Image
    imgs = [(_lbl, _to_u8(x)[:n]) for _lbl, x in panels]
    n = min(n, min(x.shape[0] for _, x in imgs))
    H, W = imgs[0][1].shape[1:]
    cols, rows = len(imgs), n
    canvas = np.full((rows * (H + gap) - gap, cols * (W + gap) - gap), 40, np.uint8)
    for c, (_, x) in enumerate(imgs):
        for r in range(rows):
            y0, x0 = r * (H + gap), c * (W + gap)
            canvas[y0:y0 + H, x0:x0 + W] = x[r]
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(canvas).save(path)
    print(f"  [viz] saved {path}  columns: {' | '.join(l for l, _ in imgs)}  ({rows} rows)")
    return path
