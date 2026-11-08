"""Persistent-object scoring with one-to-one assignment and explicit coverage.

Predicted components do not acquire identity from horizontal order.  Instead, every frame is
matched jointly to generator-provided persistent objects.  Missed and extra objects are retained
as outcomes rather than removed through finite-value intersection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: Hashable
    centroid_yx: tuple[float, float]
    bbox_yxyx: tuple[int, int, int, int]
    class_id: int | None = None
    visible: bool = True


@dataclass(frozen=True)
class Detection:
    centroid_yx: tuple[float, float]
    bbox_yxyx: tuple[int, int, int, int]
    class_id: int | None = None
    class_confidence: float | None = None


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0, iy1, ix1 = max(ay0, by0), max(ax0, bx0), min(ay1, by1), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    aa = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    ba = max(0, by1 - by0) * max(0, bx1 - bx0)
    return float(inter / max(aa + ba - inter, 1))


def score_objects(
    detections: Sequence[Detection],
    ground_truth: Sequence[GroundTruthObject],
    *,
    frame_shape: tuple[int, int],
    distance_weight: float = 0.5,
    iou_weight: float = 0.5,
    max_assignment_cost: float = 0.65,
) -> dict:
    """Score one frame without deleting misses, extras, or identity failures.

    Trajectory error is intention-to-score: every visible GT object contributes, with a missed
    object receiving the frame diagonal.  ``matched_trajectory_error`` is also supplied as a
    diagnostic and must not replace the intention-to-score endpoint.
    """
    gt = [g for g in ground_truth if g.visible]
    det = list(detections)
    diagonal = float(np.hypot(*frame_shape))
    cost = np.empty((len(gt), len(det)), dtype=np.float64)
    distances = np.empty_like(cost)
    ious = np.empty_like(cost)
    for gi, g in enumerate(gt):
        for di, d in enumerate(det):
            distances[gi, di] = np.hypot(
                g.centroid_yx[0] - d.centroid_yx[0], g.centroid_yx[1] - d.centroid_yx[1]
            )
            ious[gi, di] = _bbox_iou(g.bbox_yxyx, d.bbox_yxyx)
            cost[gi, di] = (
                distance_weight * min(distances[gi, di] / diagonal, 1.0)
                + iou_weight * (1.0 - ious[gi, di])
            )

    matches: list[tuple[int, int]] = []
    if len(gt) and len(det):
        rows, cols = linear_sum_assignment(cost)
        matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= max_assignment_cost]
    matched_gt = {g for g, _ in matches}
    matched_det = {d for _, d in matches}
    missed = len(gt) - len(matched_gt)
    extra = len(det) - len(matched_det)

    matched_dist = [float(distances[g, d]) for g, d in matches]
    trajectory_all = matched_dist + [diagonal] * missed
    identity_values: list[float] = []
    identity_scorable = 0
    assignment = []
    for gi, di in matches:
        g, d = gt[gi], det[di]
        correct = None
        if g.class_id is not None and d.class_id is not None:
            correct = bool(g.class_id == d.class_id)
            identity_scorable += 1
            identity_values.append(float(correct))
        assignment.append({
            "object_id": g.object_id,
            "detection_index": di,
            "distance": float(distances[gi, di]),
            "bbox_iou": float(ious[gi, di]),
            "cost": float(cost[gi, di]),
            "identity_correct": correct,
        })

    # Missed labelled objects are identity failures in the primary intention-to-score accuracy.
    labelled_gt = sum(g.class_id is not None for g in gt)
    identity_correct = float(sum(identity_values))
    return {
        "n_gt": len(gt),
        "n_pred": len(det),
        "n_matched": len(matches),
        "missed": missed,
        "extra": extra,
        "count_error": abs(len(det) - len(gt)),
        "coverage": len(matches) / len(gt) if gt else 1.0,
        "matched_trajectory_error": float(np.mean(matched_dist)) if matched_dist else np.nan,
        "trajectory_error": float(np.mean(trajectory_all)) if trajectory_all else 0.0,
        "identity_scorable": identity_scorable,
        "identity_coverage": identity_scorable / labelled_gt if labelled_gt else np.nan,
        "identity_accuracy_matched": float(np.mean(identity_values)) if identity_values else np.nan,
        "identity_accuracy": identity_correct / labelled_gt if labelled_gt else np.nan,
        "assignments": assignment,
        "unmatched_object_ids": [gt[i].object_id for i in range(len(gt)) if i not in matched_gt],
        "unmatched_detection_indices": [i for i in range(len(det)) if i not in matched_det],
    }
