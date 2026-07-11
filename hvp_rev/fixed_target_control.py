"""B1 causal control: strided context with a fixed per-frame target interface.

This module is separate from the live legacy trainer until c2 finishes.  It reuses the proven
network implementation but decouples context sampling from target sampling and exposes the buffer
update required by rollout.
"""
from __future__ import annotations

import numpy as np
import torch

from hvp_rev.faithful_nets import LAT
from hvp_rev.rhvd import WIN, WindowDiT


class FixedTargetWindowDiT(WindowDiT):
    """WindowDiT with strided coarse context and four per-frame coarse targets."""

    def __init__(self, *args, target_stride_c: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if target_stride_c != 1:
            raise ValueError("the current fixed-target control requires per-frame coarse targets")
        self.target_stride_c = target_stride_c
        self.n_tgt_c = self.win
        self.expected_seq = (
            self.n_ctx_t * 4
            + self.n_ctx_c * 16
            + (self.n_hist_f + self.n_ctx_f) * 64
            + self.n_tgt_t * 4
            + self.n_tgt_c * 16
            + self.win * 64
        )


def fixed_target_window_batch(
    data,
    batch_size: int,
    *,
    n_ctx_c: int,
    n_ctx_f: int,
    context_stride_c: int,
    train: bool = True,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Construct B1 batches with strided context but contiguous coarse/fine targets."""
    if WIN % context_stride_c:
        raise ValueError("context stride must divide the rollout window")
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pool = data.train_idx if train else data.val_idx
    reach = max(n_ctx_c * context_stride_c, n_ctx_f)
    ctx_c = torch.empty(batch_size, n_ctx_c, LAT, 4, 4)
    ctx_f = torch.empty(batch_size, n_ctx_f, LAT, 8, 8)
    tgt_c = torch.empty(batch_size, WIN, LAT, 4, 4)
    tgt_f = torch.empty(batch_size, WIN, LAT, 8, 8)
    actions = torch.empty(batch_size, WIN, dtype=torch.long)
    for row in range(batch_size):
        clip = int(pool[rng.randint(len(pool))])
        target = int(rng.randint(reach, data.T - WIN + 1))
        ctx_c[row] = data.zc[
            clip,
            target - n_ctx_c * context_stride_c + context_stride_c - 1:target:context_stride_c,
        ]
        ctx_f[row] = data.zf[clip, target - n_ctx_f:target]
        tgt_c[row] = data.zc[clip, target:target + WIN]
        tgt_f[row] = data.zf[clip, target:target + WIN]
        actions[row] = torch.from_numpy(data.acts[clip, target:target + WIN].astype(np.int64))
    return {"ctx_c": ctx_c, "ctx_f": ctx_f, "tgt_c": tgt_c, "tgt_f": tgt_f, "actions": actions}


def update_strided_context_buffer(
    buffer: torch.Tensor,
    predicted_per_frame: torch.Tensor,
    *,
    context_stride_c: int,
    keep_slots: int,
) -> torch.Tensor:
    """Append only block-end predictions to a strided context ring buffer."""
    if predicted_per_frame.shape[1] != WIN:
        raise ValueError(f"expected {WIN} per-frame coarse predictions")
    anchors = predicted_per_frame[:, context_stride_c - 1::context_stride_c]
    if anchors.shape[1] != WIN // context_stride_c:
        raise AssertionError("coarse anchor phasing is inconsistent")
    return torch.cat([buffer, anchors], dim=1)[:, -keep_slots:]
