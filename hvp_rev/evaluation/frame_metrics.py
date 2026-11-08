"""False-positive-sensitive frame metrics with explicit populations.

All functions return one value per leading record.  Empty-GT frames are represented explicitly;
they never silently switch a foreground metric into full-frame MSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def _records(x: np.ndarray) -> np.ndarray:
    """Normalize (..., H, W) or (..., 1, H, W) to (R, H, W)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim >= 3 and x.shape[-3] == 1:
        x = np.squeeze(x, axis=-3)
    if x.ndim < 2:
        raise ValueError(f"expected image records, got shape {x.shape}")
    return x.reshape(-1, x.shape[-2], x.shape[-1])


def _safe_div(num: np.ndarray, den: np.ndarray, *, empty: float = np.nan) -> np.ndarray:
    out = np.full(num.shape, empty, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out


@dataclass(frozen=True)
class _Component:
    mask: np.ndarray
    centroid: tuple[float, float]


def _components(mask: np.ndarray, min_pixels: int) -> list[_Component]:
    labels, n = ndimage.label(mask)
    result: list[_Component] = []
    for label in range(1, n + 1):
        component = labels == label
        if int(component.sum()) < min_pixels:
            continue
        ys, xs = np.nonzero(component)
        result.append(_Component(component, (float(ys.mean()), float(xs.mean()))))
    return result


def _component_record(pred_mask: np.ndarray, gt_mask: np.ndarray, min_pixels: int) -> tuple:
    pred = _components(pred_mask, min_pixels)
    gt = _components(gt_mask, min_pixels)
    n_pred, n_gt = len(pred), len(gt)
    missed, extra = n_gt, n_pred
    mean_iou = np.nan
    mean_centroid = np.nan
    if pred and gt:
        iou = np.zeros((n_gt, n_pred), dtype=np.float64)
        distance = np.zeros_like(iou)
        for gi, g in enumerate(gt):
            for pi, p in enumerate(pred):
                inter = np.logical_and(g.mask, p.mask).sum()
                union = np.logical_or(g.mask, p.mask).sum()
                iou[gi, pi] = inter / max(union, 1)
                distance[gi, pi] = np.hypot(
                    g.centroid[0] - p.centroid[0], g.centroid[1] - p.centroid[1]
                )
        gi, pi = linear_sum_assignment(1.0 - iou)
        # A zero-overlap assignment is not a detected component match.  It remains a miss+extra.
        keep = iou[gi, pi] > 0
        gi, pi = gi[keep], pi[keep]
        missed = n_gt - len(gi)
        extra = n_pred - len(pi)
        if len(gi):
            mean_iou = float(iou[gi, pi].mean())
            mean_centroid = float(distance[gi, pi].mean())
    elif not pred and not gt:
        missed = extra = 0
        mean_iou = 1.0
        mean_centroid = 0.0
    return n_gt, n_pred, missed, extra, mean_iou, mean_centroid


def evaluate_frames(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    threshold: float = 0.1,
    min_component_pixels: int = 4,
) -> dict[str, np.ndarray]:
    """Return record-level metrics for equally shaped prediction and truth arrays.

    The output population can be restored to the caller's leading shape with ``record_shape``.
    ``mse_gt_fg_nonempty`` is NaN on empty-GT records by design.  Background, full-frame and
    union-mask errors remain defined and expose hallucinations on those frames.
    """
    pred_arr, true_arr = np.asarray(pred), np.asarray(true)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"prediction/truth shapes differ: {pred_arr.shape} != {true_arr.shape}")
    record_shape = pred_arr.shape[:-3] if pred_arr.ndim >= 3 and pred_arr.shape[-3] == 1 \
        else pred_arr.shape[:-2]
    p, g = _records(pred_arr), _records(true_arr)
    p_mask, g_mask = p > threshold, g > threshold
    union = p_mask | g_mask
    se = (p - g) ** 2

    g_count = g_mask.sum(axis=(1, 2)).astype(np.float32)
    bg_count = (~g_mask).sum(axis=(1, 2)).astype(np.float32)
    union_count = union.sum(axis=(1, 2)).astype(np.float32)
    pred_count = p_mask.sum(axis=(1, 2)).astype(np.float32)
    intersection = (p_mask & g_mask).sum(axis=(1, 2)).astype(np.float32)

    mse_full = se.mean(axis=(1, 2)).astype(np.float32)
    mse_fg = _safe_div((se * g_mask).sum(axis=(1, 2)), g_count)
    mse_bg = _safe_div((se * (~g_mask)).sum(axis=(1, 2)), bg_count, empty=0.0)
    mse_union = _safe_div((se * union).sum(axis=(1, 2)), union_count, empty=0.0)
    laplacian = np.stack([ndimage.laplace(frame) for frame in p])
    fg_sharpness = _safe_div((laplacian ** 2 * g_mask).sum(axis=(1, 2)), g_count)

    # Empty-class conventions are explicit and accompanied by fg_nonempty/pred_fg_nonempty.
    precision = _safe_div(intersection, pred_count, empty=1.0)
    recall = _safe_div(intersection, g_count, empty=1.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, empty=0.0)

    comp = np.asarray([
        _component_record(pm, gm, min_component_pixels) for pm, gm in zip(p_mask, g_mask)
    ], dtype=np.float64)

    return {
        "record_shape": np.asarray(record_shape, dtype=np.int64),
        "fg_nonempty": g_count > 0,
        "pred_fg_nonempty": pred_count > 0,
        "mse_full": mse_full,
        "mse_gt_fg_nonempty": mse_fg.astype(np.float32),
        "mse_gt_bg": mse_bg.astype(np.float32),
        "mse_union": mse_union.astype(np.float32),
        "fg_laplacian_energy_nonempty": fg_sharpness.astype(np.float32),
        "fg_precision": precision.astype(np.float32),
        "fg_recall": recall.astype(np.float32),
        "fg_f1": f1.astype(np.float32),
        "gt_component_count": comp[:, 0].astype(np.int16),
        "pred_component_count": comp[:, 1].astype(np.int16),
        "missed_components": comp[:, 2].astype(np.int16),
        "extra_components": comp[:, 3].astype(np.int16),
        "matched_component_iou": comp[:, 4].astype(np.float32),
        "matched_centroid_l2": comp[:, 5].astype(np.float32),
    }
