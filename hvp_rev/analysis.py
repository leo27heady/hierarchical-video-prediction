"""Cluster-aware effects and factorial contrasts for record-level evaluation."""
from __future__ import annotations

import numpy as np


def _aligned(*arrays: np.ndarray) -> list[np.ndarray]:
    values = [np.asarray(array) for array in arrays]
    if not values or any(value.shape != values[0].shape for value in values[1:]):
        raise ValueError("analysis arrays are not aligned")
    return values


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    values, clusters = _aligned(values, clusters)
    finite = np.isfinite(values)
    values, clusters = values[finite].astype(np.float64), clusters[finite]
    unique = np.unique(clusters)
    if not len(values) or not len(unique):
        raise ValueError("no finite clustered observations")
    sums = np.asarray([values[clusters == cluster].sum() for cluster in unique])
    counts = np.asarray([(clusters == cluster).sum() for cluster in unique], dtype=np.float64)
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(unique), (n_boot, len(unique)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "records": int(len(values)),
        "clusters": int(len(unique)),
    }


def paired_effect(
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    *,
    lower_is_better: bool = True,
    **bootstrap,
) -> dict:
    baseline, candidate, clusters = _aligned(baseline, candidate, clusters)
    delta = baseline - candidate if lower_is_better else candidate - baseline
    return cluster_bootstrap_mean(delta, clusters, **bootstrap)


def head_package_interaction(
    diffusion_temporal: np.ndarray,
    diffusion_spatial: np.ndarray,
    regression_temporal: np.ndarray,
    regression_spatial: np.ndarray,
    clusters: np.ndarray,
    **bootstrap,
) -> dict:
    """Difference-in-differences in temporal-package advantage (lower error is better).

    Positive package effects mean temporal is better.  A negative interaction means regression
    shifts the contrast toward the spatial package relative to diffusion.
    """
    dt, ds, rt, rs, clusters = _aligned(
        diffusion_temporal, diffusion_spatial, regression_temporal, regression_spatial, clusters
    )
    diffusion_effect = ds - dt
    regression_effect = rs - rt
    interaction = regression_effect - diffusion_effect
    result = cluster_bootstrap_mean(interaction, clusters, **bootstrap)
    result.update({
        "diffusion_temporal_advantage": float(diffusion_effect.mean()),
        "regression_temporal_advantage": float(regression_effect.mean()),
        "convention": "negative interaction = regression shifts toward spatial packaging",
    })
    return result
