"""Phase-C replicated equivalence summary (CPU-only).

The Phase-C null is an equivalence claim, not a failure-to-reject claim.  For each training seed,
this module computes paired clip-cluster bootstrap intervals and declares equivalence only when
the complete 95% interval lies within a pre-registered practical margin.  Training seeds remain
visible as separate estimates; frames are never pooled across seeds as independent observations.

Default margins, fixed before Phase C launch:
  fg-MSE      0.005 absolute
  identity    0.02  absolute accuracy (2 percentage points)
  trajectory  1.0   pixel on a 64x64 frame
  count       0.05  digit/frame (one extra/missing digit per 20 frames)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


LOG = Path("logs_rhcp")


def cluster_ci(delta, clips, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    unique = np.unique(clips)
    sums = np.asarray([delta[clips == c].sum() for c in unique])
    counts = np.asarray([(clips == c).sum() for c in unique], dtype=np.float64)
    pick = rng.randint(0, len(unique), (n_boot, len(unique)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aligned_pair(flat_path, hier_path, field, mask_fn=None):
    with np.load(flat_path) as f, np.load(hier_path) as h:
        for key in ("h", "clip"):
            if not np.array_equal(f[key], h[key]):
                raise ValueError(f"unaligned Phase-C records: {flat_path} vs {hier_path} ({key})")
        mask = np.isfinite(f[field]) & np.isfinite(h[field])
        if mask_fn is not None:
            mask &= mask_fn(f, h)
        return f[field][mask].astype(np.float64), h[field][mask].astype(np.float64), f["clip"][mask]


def metric_result(seed, name, margin, paths, field, higher_better=False, mask_fn=None):
    flat, hier, clips = aligned_pair(paths[0], paths[1], field, mask_fn)
    if not len(flat):
        return dict(seed=seed, name=name, n=0, mean=np.nan, lo=np.nan, hi=np.nan,
                    margin=margin, verdict="UNSCORABLE")
    # Positive always means the hierarchical arm is better.
    delta = hier - flat if higher_better else flat - hier
    lo, hi = cluster_ci(delta, clips, seed=1000 + seed)
    if lo >= -margin and hi <= margin:
        verdict = "EQUIVALENT"
    elif lo > 0:
        verdict = "HIER_BETTER"
    elif hi < 0:
        verdict = "FLAT_BETTER"
    else:
        verdict = "INCONCLUSIVE"
    return dict(seed=seed, name=name, n=len(delta), mean=float(delta.mean()), lo=lo, hi=hi,
                margin=margin, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fg_margin", type=float, default=0.005)
    ap.add_argument("--identity_margin", type=float, default=0.02)
    ap.add_argument("--trajectory_margin", type=float, default=1.0)
    ap.add_argument("--count_margin", type=float, default=0.05)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    for seed in seeds:
        std = (LOG / f"rhvd_eval_mmL-flat_s{seed}.npz", LOG / f"rhvd_eval_mmL-hier_s{seed}.npz")
        fac = (LOG / f"phasec_mmL-flat_s{seed}.npz", LOG / f"phasec_mmL-hier_s{seed}.npz")
        missing = [str(p) for p in (*std, *fac) if not p.exists()]
        if missing:
            raise FileNotFoundError("missing Phase-C artifacts: " + ", ".join(missing))
        results.extend([
            metric_result(seed, "fg_mse", args.fg_margin, std, "fg"),
            metric_result(seed, "identity", args.identity_margin, fac, "id_match", True,
                          lambda f, h: (f["id_scored"] > 0.5) & (h["id_scored"] > 0.5)),
            metric_result(seed, "trajectory", args.trajectory_margin, fac, "traj_err"),
            metric_result(seed, "count", args.count_margin, fac, "count_dev"),
        ])

    print("PHASE-C REPLICATED EQUIVALENCE SUMMARY")
    print("effect convention: positive = hierarchical arm better")
    for r in results:
        print(f"  {r['name']:10s} s{r['seed']} n={r['n']:5d} effect={r['mean']:+.5f} "
              f"95%CI=[{r['lo']:+.5f},{r['hi']:+.5f}] margin=+/-{r['margin']:.5f} "
              f"{r['verdict']}")
    for name in ("fg_mse", "identity", "trajectory", "count"):
        rows = [r for r in results if r["name"] == name]
        means = [r["mean"] for r in rows]
        all_eq = all(r["verdict"] == "EQUIVALENT" for r in rows)
        print(f"  {name:10s} seed effects={means} -> "
              f"{'REPLICATED_EQUIVALENCE' if all_eq else 'NOT_REPLICATED_EQUIVALENCE'}")
    all_eq = all(r["verdict"] == "EQUIVALENT" for r in results)
    print("PRIMARY VERDICT:", "REPLICATED_EQUIVALENCE" if all_eq else
          "NO EQUIVALENCE CLAIM; inspect signed effects and pre-registered alternative branch")


if __name__ == "__main__":
    main()
