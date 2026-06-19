"""Render the HARDER dataset: full SO(3) (multi-axis) rotation instead of Z-only.

A 3D shape rotating about x+y+z produces self-occlusion, foreshortening, and parts
appearing/disappearing — genuine depth/axis multimodality that a 2D constant-angular-velocity
rotation-warp baseline CANNOT predict. This is the dataset that makes the hierarchy / windowing /
distributional-head hypotheses actually testable (copying & 2D-warp stop being free wins).

Driven by the SEED-Voken venv (pyvista + shapekit). Prints the cache hashes to wire into hvp.
Run:  <SEED-Voken venv python>  scripts/gen_data_so3.py
"""

import os
import random
import sys

SEED_ROOT = r"C:/Users/leoni/Documents/projects/SEED-Voken"
sys.path.insert(0, SEED_ROOT + "/src")
sys.path.insert(0, SEED_ROOT + "/simple-shape-dataset-toolbox")

from Open_MAGVIT2.data.shape_video import ShapeVideoDataset  # noqa: E402

CACHE = r"C:/Users/leoni/Documents/data/shape_cache"


def gen(mode: str, n: int, seed: int, seq_len: int = 9) -> str:
    random.seed(seed)
    cfg = dict(
        mode=mode, sequence_length=seq_len, size=64, dataset_size=n,
        shape_scene_type="DIM_3", shape_min_cubes=2, shape_max_cubes=6,
        shape_angle_min=15, shape_angle_max=45,
        shape_rotation_axes="xyz",            # <-- the only change vs gen_data_64: full SO(3)
        shape_cache_dir=CACHE, cache=True,
    )
    ds = ShapeVideoDataset(cfg)
    h = os.path.basename(ds.cache_dir)
    print(f"GENERATED so3 mode={mode} n={n} T={seq_len} hash={h}", flush=True)
    return h


if __name__ == "__main__":
    th = gen("train", 500, seed=10)
    vh = gen("val", 200, seed=11)
    lh = gen("val_long", 100, seed=12, seq_len=33)
    print(f"\nWIRE INTO DataConfig (so3):  train_hash={th}  val_hash={vh}  val_long_hash={lh}", flush=True)
