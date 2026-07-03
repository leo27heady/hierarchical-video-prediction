"""Regime tuning + two-sided verification for testbed operating point B (HGRID_DESIGN.md v2 §1).

Modes:
  --tune            simulate trajectories over a p_cont sweep (no rendering) and report, per value:
                    median off-screen re-entry gap, effective loop period, warmup world coverage,
                    beyond-48 re-entry frames per clip in the rollout span. Recommends a p_cont.
  --verify PREFIX   load PREFIX.npy's sibling _positions.npy (exact trajectories) and run the
                    pre-registered two-sided verification:
                      (1) median re-entry gap in [GAP_LO, GAP_HI] = [192, 280]
                      (2) beyond-48 re-entry frames per clip in rollout span: median >= 10
                      (3) warmup coverage (fraction of world seen in frames 0..191): reported
                      (4) p_hold sanity vs the actions file
                    Exits nonzero on FAIL of (1) or (2).

  venv/Scripts/python.exe scripts/verify_regime.py --tune --world 768 --frames 480 --speed 4
  venv/Scripts/python.exe scripts/verify_regime.py --verify data/hgrid_train_1200x480.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_looping import pan_trajectory, V  # noqa: E402

WARM = 192          # GT warmup frames (one ballistic loop)
FLAT_WIN = 48       # largest affordable flat window (the capability threshold)
GAP_LO, GAP_HI = 192, 280


def column_visibility(pos, Ww):
    """(T,) positions -> (T, Ww) bool: world column c visible at t iff (c - pos[t]) mod Ww < V."""
    c = np.arange(Ww)[None, :]
    return ((c - pos[:, None]) % Ww) < V


def clip_stats(pos, Ww, T):
    """Per-clip regime stats from one trajectory."""
    vis = column_visibility(pos, Ww)                        # (T, Ww)
    # --- re-entry gaps: per column, gaps between consecutive visibility episodes
    gaps = []
    for c in range(0, Ww, 4):                               # stride 4 over columns (speed, unbiased)
        v = vis[:, c]
        d = np.diff(v.astype(np.int8))
        exits = np.where(d == -1)[0]                        # last visible at t
        entries = np.where(d == 1)[0] + 1                   # visible again at t
        for e in exits:
            nxt = entries[entries > e]
            if len(nxt):
                gaps.append(int(nxt[0] - e))
    # --- beyond-48 re-entry frames in the rollout span: frame t >= WARM where some column visible
    #     at t was last visible > FLAT_WIN frames ago (and HAS been seen before)
    last_seen = np.full(Ww, -10**9, np.int64)
    reentry_frames = 0
    dt_samples = []
    for t in range(T):
        vc = np.where(vis[t])[0]
        if t >= WARM:
            dt = t - last_seen[vc]
            fresh = dt[(dt > FLAT_WIN) & (dt < 10**8)]
            if len(fresh):
                reentry_frames += 1
                dt_samples.append(int(np.median(fresh)))
        last_seen[vc] = t
    # --- warmup coverage
    cov = float(vis[:WARM].any(0).mean())
    return gaps, reentry_frames, cov, dt_samples


def analyze(positions, Ww, label, log=print):
    N, T = positions.shape
    all_gaps, re_frames, covs, dts = [], [], [], []
    for i in range(N):
        g, rf, cov, dt = clip_stats(positions[i], Ww, T)
        all_gaps += g; re_frames.append(rf); covs.append(cov); dts += dt
    all_gaps = np.array(all_gaps); re_frames = np.array(re_frames); covs = np.array(covs)
    med_gap = float(np.median(all_gaps)) if len(all_gaps) else float("nan")
    log(f"[{label}] clips={N} T={T} world={Ww}")
    log(f"  re-entry gap: median {med_gap:.0f}  p25/p75 {np.percentile(all_gaps,25):.0f}/{np.percentile(all_gaps,75):.0f}"
        f"  (n={len(all_gaps)})")
    log(f"  beyond-{FLAT_WIN} re-entry frames per clip in rollout span [{WARM}..{T}): "
        f"median {np.median(re_frames):.0f}  p10 {np.percentile(re_frames,10):.0f}  min {re_frames.min()}  "
        f"zero-clips {(re_frames==0).mean()*100:.1f}%")
    log(f"  warmup coverage (frames 0..{WARM}): median {np.median(covs)*100:.1f}%  p10 {np.percentile(covs,10)*100:.1f}%")
    if dts:
        log(f"  dt-at-re-entry: median {np.median(dts):.0f}  p75 {np.percentile(dts,75):.0f}  max {max(dts)}")
    return med_gap, float(np.median(re_frames)), float(np.median(covs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--verify", default=None, help="clips .npy path (positions sibling required)")
    ap.add_argument("--world", type=int, default=768)
    ap.add_argument("--frames", type=int, default=480)
    ap.add_argument("--speed", type=int, default=4)
    ap.add_argument("--p_hold", type=float, default=0.1)
    ap.add_argument("--n_sim", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.tune:
        print(f"p_cont sweep @ world={args.world} T={args.frames} speed={args.speed} p_hold={args.p_hold}")
        print(f"criteria: median gap in [{GAP_LO},{GAP_HI}]; median beyond-{FLAT_WIN} frames/clip >= 10")
        best = None
        for p_cont in (0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995):
            rng = np.random.RandomState(args.seed)
            pos = np.stack([pan_trajectory(args.frames, args.world, rng, p_hold=args.p_hold,
                                           speed=args.speed, p_cont=p_cont)[0]
                            for _ in range(args.n_sim)])
            med_gap, med_re, cov = analyze(pos, args.world, f"p_cont={p_cont}")
            ok = GAP_LO <= med_gap <= GAP_HI and med_re >= 10
            print(f"  -> {'CANDIDATE' if ok else 'reject'}")
            if ok and best is None:
                best = p_cont
        print(f"\nRECOMMENDED p_cont: {best}")
        return

    if args.verify:
        pos_path = args.verify.replace(".npy", "_positions.npy")
        act_path = args.verify.replace(".npy", "_actions.npy")
        positions = np.load(pos_path)
        actions = np.load(act_path)
        Ww = args.world
        med_gap, med_re, cov = analyze(positions, Ww, os.path.basename(args.verify))
        p_hold_meas = float((actions == 0).mean())
        print(f"  measured p_hold: {p_hold_meas:.3f}")
        ok1 = GAP_LO <= med_gap <= GAP_HI
        ok2 = med_re >= 10
        print(f"\nVERIFICATION: (1) median gap {med_gap:.0f} in [{GAP_LO},{GAP_HI}]: {'PASS' if ok1 else 'FAIL'}"
              f"   (2) beyond-{FLAT_WIN} frames/clip median {med_re:.0f} >= 10: {'PASS' if ok2 else 'FAIL'}"
              f"   (3) warmup coverage {cov*100:.1f}% (reported)")
        sys.exit(0 if (ok1 and ok2) else 1)

    ap.error("choose --tune or --verify")


if __name__ == "__main__":
    main()
