"""One-time data prep: render 64x64 rotating-shape clips into the shared cache.

Driven by the SEED-Voken venv (which has pyvista + shapekit). Generates an independently
rendered train and val split (mode is part of the cache hash -> a true cross-split).
Run:  <SEED-Voken venv python>  scripts/gen_data_64.py
Prints the two cache hashes to wire into hvp DataConfig.
"""

import os
import random
import sys

SEED_ROOT = r"C:/Users/leoni/Documents/projects/SEED-Voken"
sys.path.insert(0, SEED_ROOT + "/src")
sys.path.insert(0, SEED_ROOT + "/simple-shape-dataset-toolbox")

from Open_MAGVIT2.data.shape_video import ShapeVideoDataset  # noqa: E402

CACHE = r"C:/Users/leoni/Documents/data/shape_cache"


def gen(mode: str, n: int, seed: int) -> str:
    random.seed(seed)
    cfg = dict(
        mode=mode,
        sequence_length=9,
        size=64,                 # resolution
        dataset_size=n,          # clip count
        shape_scene_type="DIM_3",
        shape_min_cubes=2,
        shape_max_cubes=6,
        shape_angle_min=15,
        shape_angle_max=45,
        shape_cache_dir=CACHE,
        cache=True,
    )
    ds = ShapeVideoDataset(cfg)
    h = os.path.basename(ds.cache_dir)
    print(f"GENERATED mode={mode} n={n} hash={h} dir={ds.cache_dir}", flush=True)
    return h


if __name__ == "__main__":
    th = gen("train", 600, seed=0)
    vh = gen("val", 200, seed=1)
    print(f"\nWIRE INTO DataConfig:  train_hash={th}  val_hash={vh}", flush=True)
