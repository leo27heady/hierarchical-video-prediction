"""Complete, immutable rollout record writer for post-c2 integration."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from hvp_rev.evaluation.frame_metrics import evaluate_frames
from hvp_rev.provenance import ArtifactStore


def write_complete_rollout(
    store: ArtifactStore,
    *,
    pred: np.ndarray,
    true: np.ndarray,
    clip: np.ndarray,
    horizon: np.ndarray,
    threshold: float,
    labels: Mapping[str, np.ndarray] | None = None,
) -> str:
    """Compute and atomically save aligned frame metrics plus geometry labels."""
    frame = evaluate_frames(pred, true, threshold=threshold)
    records = len(frame["mse_full"])
    clip, horizon = np.asarray(clip), np.asarray(horizon)
    if clip.shape != (records,) or horizon.shape != (records,):
        raise ValueError("clip/horizon keys do not align with flattened frame records")
    arrays = {key: value for key, value in frame.items() if key != "record_shape"}
    arrays.update(clip=clip, h=horizon, record_shape=frame["record_shape"])
    for key, value in (labels or {}).items():
        value = np.asarray(value)
        if value.shape != (records,):
            raise ValueError(f"label {key} is not record-aligned")
        arrays[key] = value
    path = store.save_npz("metrics/rollout_complete.npz", **arrays)
    return str(path)
