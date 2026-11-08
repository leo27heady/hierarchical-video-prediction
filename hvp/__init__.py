"""Stable public API for the hierarchical-video-prediction research package.

The implementation used by the frozen experiments remains in :mod:`hvp_rev`.  This package
exposes the construct-valid, provenance-aware subset intended for reuse. Importing :mod:`hvp`
does not load datasets, initialize a GPU, or launch an experiment.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .context import ArmSpec, ContextGeometry, TOKENS_PER_FRAME
from .evaluation import Detection, GroundTruthObject, evaluate_frames, pair_specific_reach_mask
from .evaluation import score_objects
from .statistics import (
    cluster_bootstrap_mean,
    head_package_interaction,
    paired_effect,
    simultaneous_cluster_intervals,
)

try:
    __version__ = version("hierarchical-video-prediction")
except PackageNotFoundError:  # editable source tree without package metadata
    __version__ = "0+unknown"

__all__ = [
    "ArmSpec",
    "ContextGeometry",
    "Detection",
    "GroundTruthObject",
    "TOKENS_PER_FRAME",
    "__version__",
    "cluster_bootstrap_mean",
    "evaluate_frames",
    "head_package_interaction",
    "paired_effect",
    "pair_specific_reach_mask",
    "score_objects",
    "simultaneous_cluster_intervals",
]
