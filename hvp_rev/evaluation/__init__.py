"""Construct-valid, record-level evaluation utilities.

The legacy evaluators remain readable for historical artifacts.  New studies should import from
this package so foreground fidelity, false positives, object coverage, and reach are not collapsed
into one ambiguous score.
"""

from .frame_metrics import evaluate_frames
from .object_metrics import Detection, GroundTruthObject, score_objects
from .reach import ContextGeometry, pair_specific_reach_mask

__all__ = [
    "ContextGeometry",
    "Detection",
    "GroundTruthObject",
    "evaluate_frames",
    "pair_specific_reach_mask",
    "score_objects",
]
