"""Step 1 (trustworthy, action-aware) analysis of the looping testbed.

The 6-agent review + design red-team established: the prior "all denoisers blur -> task/scale-inherent" verdict was
CONFOUNDED (no denoiser consumed the recorded pan ACTION). But fixing that exposes a deeper question the red-team named:
given the action, KNOWN content is a deterministic warp (regression wins) and the only per-frame uncertainty is the
newly-revealed strip -- which is either DETERMINED (content still in context) or UNKNOWABLE (it left the short context
window; loop period 64 >> n_ctx 6). Is there ANY hard-but-computable regime?

mode=geom : PURE-GEOMETRY memory-horizon probe (NO training, ~seconds). From the recorded actions alone, as a function
            of context length n_ctx, what fraction of the target frame's FOREGROUND (sprite) columns are RECOVERABLE from
            context (their world-column was seen by some context frame) vs NOVEL (never seen -> unknowable)? This answers
            the determined-or-unknowable question for free and quantifies the memory horizon before any net is built.

  venv/Scripts/python.exe -m hvp_rev.step1 --mode geom
"""
from __future__ import annotations

import argparse

import numpy as np

V = 64      # viewport (gen_looping)


def rel_positions(acts, speed):
    """Relative left-edge world position per frame (arbitrary common origin), from cumulative actions. act[t] moves
    frame t-1 -> t by act[t]*speed, so rel = cumsum(act*speed); only differences/mod-W matter."""
    return np.cumsum(acts.astype(np.int64) * speed)


def geom(clip_path, action_path, n_ctxs, k=4, thr=0.5, n_samples=6000, seed=0, W=128, SPEED=2):
    clips = np.load(clip_path)
    acts = np.load(action_path)
    N, T = clips.shape[0], clips.shape[1]
    rng = np.random.RandomState(seed)
    maxc = max(n_ctxs)
    assert T - k > maxc, f"clip length {T} too short for max n_ctx {maxc}"
    res = {nc: {"novel_fg": [], "novel_strip": [], "fully_rec": []} for nc in n_ctxs}
    n_fg_cols = []
    strip_sizes = []
    for _ in range(n_samples):
        c = rng.randint(N)
        t = rng.randint(maxc, T - k)                     # context = [t-maxc .. t-1]; target = t+k-1
        rel = rel_positions(acts[c], SPEED)
        tgt = t + k - 1
        frame = clips[c, tgt].astype(np.float32) / 255.0
        fg_cols = np.where(frame.max(axis=0) > thr)[0]   # viewport columns containing sprite (bg maxes at 0.5)
        if len(fg_cols) == 0:
            continue
        n_fg_cols.append(len(fg_cols))
        tgt_world = (rel[tgt] + fg_cols) % W
        # strip = target columns whose world-col was NOT covered by the LAST context frame (t-1) = newly revealed
        last_world = set(((rel[t - 1] + np.arange(V)) % W).tolist())
        strip_mask = np.array([wc not in last_world for wc in tgt_world])
        strip_sizes.append(int(strip_mask.sum()))
        for nc in n_ctxs:
            covered = np.zeros(W, bool)
            for m in range(t - nc, t):                   # nc context frames
                covered[(rel[m] + np.arange(V)) % W] = True
            novel = ~covered[tgt_world]                  # fg columns never seen by any context frame
            res[nc]["novel_fg"].append(float(novel.mean()))
            res[nc]["fully_rec"].append(float(novel.sum() == 0))
            if strip_mask.any():                         # novelty restricted to the newly-revealed strip
                res[nc]["novel_strip"].append(float(novel[strip_mask].mean()))
    print(f"testbed geometry: viewport {V} / world {W} ({100*(W-V)//W}% off-screen), pan speed {SPEED}, "
          f"symmetric random walk (p_hold 0.2)")
    print(f"mean sprite columns / target frame: {np.mean(n_fg_cols):.1f}/64 ; "
          f"mean newly-revealed-strip width: {np.mean(strip_sizes):.1f} cols\n")
    print(f"{'n_ctx':>6} {'ctx_frames':>10} | {'novel_fg':>9} {'novel_strip':>12} {'fully_recov':>12}   (lower novel = more recoverable)")
    for nc in n_ctxs:
        nf = np.mean(res[nc]["novel_fg"])
        ns = np.mean(res[nc]["novel_strip"]) if res[nc]["novel_strip"] else float("nan")
        fr = np.mean(res[nc]["fully_rec"])
        print(f"{nc:>6} {nc:>10} | {nf:>9.3f} {ns:>12.3f} {fr:>12.3f}")
    print("\ninterpretation:")
    print("  - novel_fg   : fraction of TARGET sprite columns whose world-position was NEVER in any context frame")
    print("                 (information-theoretically UNKNOWABLE from context -> no denoiser/EBT can recover it).")
    print("  - novel_strip: same, restricted to the newly-revealed leading strip (the only place prediction is nontrivial).")
    print("  - fully_recov: fraction of target frames whose ENTIRE sprite content is recoverable from context.")
    print("  If novel_fg stays HIGH as n_ctx grows -> re-entry is unknowable at any tractable context (the slow symmetric")
    print("  random walk on a torus explores too little) -> determined-or-unknowable, EBT nulls -> testbed needs redesign.")
    print("  If novel_fg DROPS sharply with n_ctx -> a computable-with-memory regime exists -> memory/hierarchy has a job.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="geom", choices=["geom"])
    ap.add_argument("--clips", default="data/loop_train_3000x100.npy")
    ap.add_argument("--actions", default="data/loop_train_3000x100_actions.npy")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--world", type=int, default=128)
    ap.add_argument("--speed", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "geom":
        geom(args.clips, args.actions, n_ctxs=[6, 12, 24, 48, 72, 90], k=args.k, W=args.world, SPEED=args.speed)


if __name__ == "__main__":
    main()
