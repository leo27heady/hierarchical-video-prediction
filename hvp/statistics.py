"""Cluster-aware paired effects and simultaneous inference.

The public API deliberately omits frame-IID convenience intervals. Training/evaluation records
from one clip are correlated and must be resampled at clip level.
"""
from hvp_rev.analysis import (
    cluster_bootstrap_mean,
    head_package_interaction,
    paired_effect,
    simultaneous_cluster_intervals,
)

__all__ = [
    "cluster_bootstrap_mean",
    "head_package_interaction",
    "paired_effect",
    "simultaneous_cluster_intervals",
]
