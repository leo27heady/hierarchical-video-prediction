"""Generate a hash-pinned loopmom confirmation set without opening it for model selection.

This uses the exact legacy loopmom pixel/RNG path while additionally recording viewport positions,
source glyph indices, world coordinates, configuration, purpose and artifact hashes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import (
    atomic_save_npy,
    atomic_save_npz,
    atomic_write_json,
    sha256_file,
)
from scripts.gen_looping import DIGIT, MNIST, V, build_world, load_mnist, pan_trajectory, render_clip


def _sibling(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--dirty-hash", required=True)
    parser.add_argument(
        "--purpose",
        choices=(
            "confirmation_holdout",
            "robustness_holdout",
            "codec_robustness_holdout",
            "retrieval_reference_holdout",
            "development_fixture",
        ),
        required=True,
    )
    parser.add_argument("--mnist", default=MNIST)
    parser.add_argument("--source-split", choices=("train",), default="train")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--world", type=int, default=192)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--speed", type=int, default=4)
    parser.add_argument("--p-cont", type=float, default=0.9)
    parser.add_argument("--p-hold", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.out)
    if out.suffix != ".npy":
        raise ValueError("--out must end in .npy")
    actions_path = _sibling(out, "_actions.npy")
    positions_path = _sibling(out, "_positions.npy")
    objects_path = _sibling(out, "_objects.npz")
    manifest_path = _sibling(out, ".manifest.json")
    outputs = (out, actions_path, positions_path, objects_path, manifest_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"confirmation generation refuses overwrite: {existing}")
    if args.world <= V:
        raise ValueError("world must be wider than the viewport")

    mnist_path = Path(args.mnist)
    digits = load_mnist(str(mnist_path))
    rng = np.random.RandomState(args.seed)
    clips = np.zeros((args.n, args.frames, V, V), np.uint8)
    actions = np.zeros((args.n, args.frames), np.int8)
    positions = np.zeros((args.n, args.frames), np.int32)
    object_id = np.zeros((args.n, args.digits), np.int16)
    glyph_index = np.zeros((args.n, args.digits), np.int32)
    world_x = np.zeros((args.n, args.digits), np.int32)
    world_y = np.zeros((args.n, args.digits), np.int16)

    for clip in range(args.n):
        world, metadata = build_world(
            args.world, V, digits, rng, args.digits, return_metadata=True
        )
        pos, act = pan_trajectory(
            args.frames, args.world, rng, p_hold=args.p_hold, speed=args.speed,
            p_cont=args.p_cont,
        )
        rendered = render_clip(world, pos, V)
        clips[clip] = np.clip(rendered * 255, 0, 255).astype(np.uint8)
        actions[clip] = act
        positions[clip] = pos
        rows = np.asarray(metadata, dtype=np.int64)
        object_id[clip] = rows[:, 0]
        glyph_index[clip] = rows[:, 1]
        world_x[clip] = rows[:, 2]
        world_y[clip] = rows[:, 3]
        if (clip + 1) % 100 == 0:
            print(f"generated {clip + 1}/{args.n}", flush=True)

    if not set(np.unique(actions)).issubset({-1, 0, 1}):
        raise AssertionError("invalid action outside {-1,0,1}")
    if positions.min() < 0 or positions.max() >= args.world:
        raise AssertionError("viewport position outside torus")

    atomic_save_npy(out, clips)
    atomic_save_npy(actions_path, actions)
    atomic_save_npy(positions_path, positions)
    atomic_save_npz(
        objects_path,
        object_id=object_id,
        glyph_index=glyph_index,
        world_x=world_x,
        world_y=world_y,
        glyph_size=np.asarray(DIGIT, dtype=np.int16),
    )

    artifact_paths = {
        "clips": out,
        "actions": actions_path,
        "positions": positions_path,
        "objects": objects_path,
    }
    artifact_info = {
        key: {
            "path": _display_path(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in artifact_paths.items()
    }
    dataset_id = f"loopmom-{args.purpose}-{args.seed}-{artifact_info['clips']['sha256'][:12]}"
    manifest = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "purpose": args.purpose,
        "generator": {
            "path": "scripts/generate_loopmom_confirmation.py",
            "legacy_pixel_path": "scripts/gen_looping.py",
            "commit": args.generator_commit,
            "dirty_hash": args.dirty_hash,
            "rng": "numpy.random.RandomState/MT19937",
            "seed": args.seed,
        },
        "configuration": {
            "n": args.n,
            "frames": args.frames,
            "viewport": V,
            "world": args.world,
            "digits": args.digits,
            "speed": args.speed,
            "p_cont": args.p_cont,
            "p_hold": args.p_hold,
        },
        "source_glyph_pool": {
            "split": args.source_split,
            "file_name": mnist_path.name,
            "bytes": mnist_path.stat().st_size,
            "sha256": sha256_file(mnist_path),
            "indices_artifact": artifact_info["objects"]["path"],
        },
        "artifacts": artifact_info,
        "access_policy": {
            "training": "forbidden",
            "monitoring_grid": "forbidden",
            "locked_evaluation": "explicit purpose-matched switch and access log required",
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(f"wrote {manifest_path} dataset_id={dataset_id}")
    print("No frames were displayed or selected.")


if __name__ == "__main__":
    main()
