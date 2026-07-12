"""Purpose-built LONG-LOOPING video dataset with genuine OFF-SCREEN MEMORY — the testbed every prior null needed.

A 64x64 viewport pans horizontally over a WIDER torus world (width Ww) that holds a fixed background texture + K
sprites (MNIST digits) at FIXED world positions. As the viewport pans, a sprite exits one edge and — because the world
is a torus and the pan is a bounded random walk — must RE-ENTER at its correct world position later. So a model MUST
remember off-screen content (the coarse/hierarchy level's real job), and the sequence is exactly periodic under a full
loop (clean long-horizon consistency eval). Actions {left, hold, right} are recorded (action-conditioned).

Contrast with MMNIST / MMNIST-CIFAR: those have a static-per-clip background and NO off-screen memory (nothing exits and
must return), which is exactly why the hierarchy/clock nulled there. This dataset gives the hierarchy something to do.

  venv/Scripts/python.exe scripts/gen_looping.py --n 3000 --frames 100 --world 128 --digits 3 --out data/loop_train_3000x100.npy
"""
from __future__ import annotations

import argparse
import gzip
import os

import numpy as np

MNIST = "C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/train-images-idx3-ubyte.gz"
V = 64            # viewport size
DIGIT = 28


def load_mnist(path=MNIST):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32)


def make_background(Ww, Hh, rng, unique_amp=0.0, texture_unique=False):
    """Smooth low-frequency texture on the torus (seamless in x). Non-trivial but static in world coords.

    Default: 4 components with kx scaled by Ww/192 (per-viewport frequency invariance vs the
    loopmom reference) — NOTE this makes the bg exactly (Ww/scale)-periodic: positions are
    aliased (verified MSE 0.000000 between 192px-shifted crops at Ww=768).
    unique_amp > 0: + one kx=1 component (DC/level cue at viewport scale). MEASURED INEFFECTIVE:
    the frozen AE's GroupNorms strip DC brightness — absolute position stays linearly
    UNdecodable from the latents (R^2 -0.08, scripts/probe_position.py).
    texture_unique: REPLACES the scaled components with 6 incommensurate mid-frequency
    components (kx from odd pool, periods ~40-154px at Ww=768 ~= loopmom's 38-192px band):
    position is carried by the JOINT LOCAL PHASE PATTERN (survives conv+GroupNorm), and no
    common divisor with Ww/192 exists -> no positional aliasing."""
    x = np.arange(Ww)[None, :]
    y = np.arange(Hh)[:, None]
    bg = np.zeros((Hh, Ww), np.float32)
    if texture_unique == "loopmom_wide":
        # scaled-loopmom: UNSCALED low wavenumbers on the wide world — the exact recipe under
        # which slot retrieval demonstrably emerged (loopmom, T3), with k=1 FORCED so every world
        # is position-unique (loopmom itself only drew k=1 with p~=0.59; its 192px world + dense
        # digits masked that). Per-frame appearance differs from loopmom (smoother viewport bg)
        # -> the frozen AE is NOT reused; retrain on this dataset's train split.
        for kx in [1] + list(rng.randint(2, 6, 3)):
            ky = rng.randint(0, 4)
            ph = rng.rand() * 2 * np.pi
            amp = rng.uniform(0.15, 0.4)
            bg += amp * np.sin(2 * np.pi * int(kx) * x / Ww + ph) * np.cos(2 * np.pi * ky * y / Hh + ph)
    elif texture_unique:
        pool = np.array([5, 7, 9, 11, 13, 15, 17, 19])
        ks = rng.choice(pool, 6, replace=False)
        for kx in ks:
            ky = rng.randint(0, 4)
            ph = rng.rand() * 2 * np.pi
            amp = rng.uniform(0.15, 0.4)
            bg += amp * np.sin(2 * np.pi * int(kx) * x / Ww + ph) * np.cos(2 * np.pi * ky * y / Hh + ph)
    else:
        # scale wavenumbers with world size (reference = the 192px loopmom world) so the PER-VIEWPORT
        # background frequency distribution is world-size-invariant (integer scale keeps seamless wrap)
        kx_scale = max(1, round(Ww / 192))
        for _ in range(4):
            kx = rng.randint(1, 6) * kx_scale        # integer wavenumber -> seamless wrap
            ky = rng.randint(0, 4)
            ph = rng.rand() * 2 * np.pi
            amp = rng.uniform(0.15, 0.4)
            bg += amp * np.sin(2 * np.pi * kx * x / Ww + ph) * np.cos(2 * np.pi * ky * y / Hh + ph)
        if unique_amp > 0:
            # kx MUST be 1: any kx>1 leaves kx-fold positional degeneracy (verified: kx=2 left the
            # +384px crop byte-identical). Random phase per world provides the diversity instead.
            ph = rng.rand() * 2 * np.pi
            bg += unique_amp * np.sin(2 * np.pi * 1 * x / Ww + ph) * np.ones_like(y, np.float32)
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-6)
    return (0.15 + 0.35 * bg).astype(np.float32)     # dim background in [0.15, 0.5]


def build_world(Ww, Hh, digits, rng, num_digits, unique_amp=0.0, texture_unique=False,
                return_metadata=False):
    """Torus world image (Hh, Ww) in [0,1]: background texture + sprites at fixed world positions (max-composite)."""
    world = make_background(Ww, Hh, rng, unique_amp=unique_amp, texture_unique=texture_unique)
    metadata = []
    for object_id in range(num_digits):
        glyph_index = int(rng.randint(len(digits)))
        d = digits[glyph_index] / 255.0
        wx = rng.randint(0, Ww)                        # torus x (wraps)
        wy = rng.randint(0, Hh - DIGIT)
        metadata.append((object_id, glyph_index, int(wx), int(wy)))
        for dy in range(DIGIT):
            xs = (wx + np.arange(DIGIT)) % Ww          # wrap in x
            world[wy + dy, xs] = np.maximum(world[wy + dy, xs], d[dy])
    return (world, metadata) if return_metadata else world


def pan_trajectory(T, Ww, rng, p_hold=0.2, speed=2, p_cont=None):
    """Viewport left-edge trajectory over the torus; returns positions (T,) and actions (T,) in {-1 left,0 hold,+1 right}.

    p_cont=None -> original iid SYMMETRIC random walk (equal L/R). This diffuses too slowly to cycle the torus in a
    tractable clip, so off-screen content is either still in a short context (determined) or never revisited (unknowable)
    -> no memory-horizon regime (verified by the step1 geometry probe: 0.9-col newly-revealed strip).

    p_cont in (0,1] -> MOMENTUM walk: with prob p_hold hold; else continue the current heading w.p. p_cont, reverse w.p.
    1-p_cont. Directed motion CYCLES the torus (period ~Ww/speed frames) so content genuinely exits and RE-ENTERS -> a
    long-context model can recall it, a short-context one cannot = a real hard-but-computable-WITH-MEMORY regime."""
    pos = np.zeros(T, np.int32)
    act = np.zeros(T, np.int32)
    x = rng.randint(0, Ww)
    d = 1 if rng.rand() < 0.5 else -1
    for t in range(T):
        r = rng.rand()
        if p_cont is None:
            a = 0 if r < p_hold else (1 if r < p_hold + (1 - p_hold) / 2 else -1)
        else:
            if r < p_hold:
                a = 0
            else:
                if rng.rand() >= p_cont:
                    d = -d
                a = d
        act[t] = a
        x = (x + a * speed) % Ww
        pos[t] = x
    return pos, act


def render_clip(world, pos, Hh):
    """Extract the 64-wide viewport at each frame (wrapping in x). world (Hh,Ww) -> (T,64,64)."""
    Ww = world.shape[1]
    T = len(pos)
    out = np.zeros((T, Hh, V), np.float32)
    idx = (pos[:, None] + np.arange(V)[None, :]) % Ww   # (T, 64) wrapped column indices
    for t in range(T):
        out[t] = world[:, idx[t]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--world", type=int, default=128, help="torus world width (>64 so content goes off-screen)")
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--p_cont", type=float, default=None, help="momentum: prob of continuing heading (None=iid walk)")
    ap.add_argument("--p_hold", type=float, default=0.2)
    ap.add_argument("--unique_amp", type=float, default=0.0,
                    help=">0: add one unscaled kx in {1..3} bg component (breaks positional aliasing)")
    ap.add_argument("--texture_unique", nargs="?", const=True, default=False,
                    help="position-unique bg: flag alone = 6 incommensurate mid-freq components; "
                         "'loopmom_wide' = unscaled k=1..5 with k=1 forced (scaled-loopmom recipe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_actions", default=None)
    args = ap.parse_args()
    assert args.world > V, "world must be wider than the 64px viewport for off-screen content"
    digits = load_mnist()
    rng = np.random.RandomState(args.seed)
    clips = np.zeros((args.n, args.frames, V, V), np.uint8)
    actions = np.zeros((args.n, args.frames), np.int8)
    positions = np.zeros((args.n, args.frames), np.int32)   # viewport left edge (world coords)
    for i in range(args.n):
        world = build_world(args.world, V, digits, rng, args.digits, unique_amp=args.unique_amp,
                            texture_unique=args.texture_unique)
        pos, act = pan_trajectory(args.frames, args.world, rng, p_hold=args.p_hold, speed=args.speed, p_cont=args.p_cont)
        clip = render_clip(world, pos, V)
        clips[i] = np.clip(clip * 255, 0, 255).astype(np.uint8)
        actions[i] = act
        positions[i] = pos
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, clips)                            # (N, T, 64, 64) uint8, load with time_axis=1
    ap_out = args.out_actions or args.out.replace(".npy", "_actions.npy")
    np.save(ap_out, actions)
    pp_out = args.out.replace(".npy", "_positions.npy")
    np.save(pp_out, positions)                          # exact trajectories -> exact re-entry/warmup-seen labels
    off = args.world - V
    dens = args.digits * (V + DIGIT) / args.world       # E[digits overlapping a viewport frame]
    fg_meas = float((clips[: min(64, args.n)].astype(np.float32) / 255.0 > 0.55).mean())
    print(f"saved {args.out} {clips.shape}; actions {ap_out}; positions {pp_out}; seed={args.seed}")
    print(f"world width {args.world}, viewport {V} -> {off}px ({100*off/args.world:.0f}%) off-screen at any time; "
          f"ballistic loop ~ {args.world//max(1,args.speed)} frames (memory horizon)")
    print(f"digit density: analytic {dens:.2f}/frame; measured fg(>0.55) fraction {fg_meas:.4f}")


if __name__ == "__main__":
    main()
