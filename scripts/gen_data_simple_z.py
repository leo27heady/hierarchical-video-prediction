"""Render the SIMPLEST prediction-debug dataset: few cubes, Z-only (in-plane), SLOW rotation.

The default datasets rotate 15-45 degrees PER FRAME (transformation[2] in shapekit/rotation.py is
per-step), so consecutive frames differ a lot -> single-step prediction is hard. Here we use a small
per-frame angle (slow, near-persistence + a little rotation) and 1-2 cubes, so a working predictor
should produce a SHARP next frame. Z-only (no shape_rotation_axes) = effectively 2D apparent motion.

Run:  <SEED-Voken venv python>  scripts/gen_data_simple_z.py
"""

import argparse
import os
import random
import sys

SEED_ROOT = r"C:/Users/leoni/Documents/projects/SEED-Voken"
sys.path.insert(0, SEED_ROOT + "/src")
sys.path.insert(0, SEED_ROOT + "/simple-shape-dataset-toolbox")

from Open_MAGVIT2.data.shape_video import ShapeVideoDataset  # noqa: E402

CACHE = r"C:/Users/leoni/Documents/data/shape_cache"


def gen(mode: str, n: int, seed: int, a, size=64, seq_len=9, cubes=(1, 2)) -> str:
    """Z-only (in-plane) rotating-shape clips. `a` = (angle_min, angle_max) degrees PER FRAME.
    Per-frame angle is the motion-magnitude knob: <=~16 deg -> single-step prediction stays sharp;
    >=~30 deg -> the predictor mean-regresses to a blob (see docs/PREDICTION_FINDINGS.md)."""
    random.seed(seed)
    cfg = dict(
        mode=mode, sequence_length=seq_len, size=size, dataset_size=n,
        shape_scene_type="DIM_3", shape_min_cubes=cubes[0], shape_max_cubes=cubes[1],
        shape_angle_min=a[0], shape_angle_max=a[1],     # no shape_rotation_axes -> Z-only (in-plane)
        shape_cache_dir=CACHE, cache=True,
    )
    ds = ShapeVideoDataset(cfg)
    h = os.path.basename(ds.cache_dir)
    print(f"GENERATED z mode={mode} n={n} size={size} T={seq_len} angle={a} cubes={cubes} hash={h}", flush=True)
    return h


PRESETS = {  # name -> (angle_min, angle_max, cubes_max)
    "simple": (4, 10, 2),     # slow, 1-2 cubes
    "moderate": (8, 16, 3),   # the working recipe (recognizable/sharp single-step)
    "fast": (15, 45, 6),      # hard (mean-regression blob)
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="simple")
    ap.add_argument("--seq-len", type=int, default=9, help="frames per clip (T=17 = more temporal context)")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=100)
    args = ap.parse_args()
    amin, amax, cmax = PRESETS[args.preset]
    th = gen("train", args.n_train, seed=20, a=(amin, amax), seq_len=args.seq_len, cubes=(1, cmax))
    vh = gen("val", args.n_val, seed=21, a=(amin, amax), seq_len=args.seq_len, cubes=(1, cmax))
    print(f"\nWIRE INTO DataConfig ({args.preset}, T={args.seq_len}):  train_hash={th}  val_hash={vh}", flush=True)
