"""Render the SO(3) (multi-axis) rotating-shape dataset at 128x128 — for the 2->8->32 pyramid.

Identical render config to gen_data_so3.py except size=128 (so BOT 32 / MID 8 / TOP 2 after the
/4-per-stage encoder). 128 is part of the cache hash, so this produces NEW caches that do not
collide with the 64px ones. Driven by the SEED-Voken venv (pyvista + shapekit).

Run:  <SEED-Voken venv python>  scripts/gen_data_so3_128.py [--smoke]
Prints the cache hashes to wire into hvp DataConfig.
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
SIZE = 128


def gen(mode: str, n: int, seed: int, seq_len: int = 9) -> str:
    random.seed(seed)
    cfg = dict(
        mode=mode, sequence_length=seq_len, size=SIZE, dataset_size=n,
        shape_scene_type="DIM_3", shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=15, shape_angle_max=45,
        shape_rotation_axes="xyz",            # full SO(3)
        shape_cache_dir=CACHE, cache=True,
    )
    ds = ShapeVideoDataset(cfg)
    h = os.path.basename(ds.cache_dir)
    print(f"GENERATED so3@128 mode={mode} n={n} T={seq_len} hash={h}", flush=True)
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="render only a couple clips to time it")
    args = ap.parse_args()
    if args.smoke:
        h = gen("train", 2, seed=10)
        print(f"\nSMOKE OK hash={h}", flush=True)
    else:
        th = gen("train", 500, seed=10)
        vh = gen("val", 200, seed=11)
        lh = gen("val_long", 100, seed=12, seq_len=33)
        print(f"\nWIRE INTO DataConfig (so3@128):  train_hash={th}  val_hash={vh}  val_long_hash={lh}", flush=True)
