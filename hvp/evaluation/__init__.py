"""Construct-valid decoded-frame, persistent-object and reach evaluation.

All frame metrics are returned per record. Empty ground-truth foreground is represented
explicitly instead of silently changing the estimand. Persistent-object scoring uses one-to-one
Hungarian assignment and retains misses, extras and coverage.
"""
from hvp_rev.evaluation import (
    ContextGeometry,
    Detection,
    GroundTruthObject,
    evaluate_frames,
    pair_specific_reach_mask,
    score_objects,
)

__all__ = [
    "ContextGeometry",
    "Detection",
    "GroundTruthObject",
    "evaluate_frames",
    "pair_specific_reach_mask",
    "score_objects",
]
