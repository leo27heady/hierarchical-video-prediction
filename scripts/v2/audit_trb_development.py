"""Audit and purpose-lock the legacy LoopMom array as TRB development data.

The new Article 4 campaign uses only the original 3,000-clip training array for
training and validation.  The old file named ``loopmom_test`` is intentionally
excluded because it has already been used during model development.  A new test
set must be generated and sealed only after the experimental protocol is locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _transition_matches(previous: np.ndarray, current: np.ndarray, action: int, speed: int) -> bool:
    if action == 0:
        return bool(np.array_equal(previous, current))
    if action == 1:
        return bool(np.array_equal(previous[:, speed:], current[:, :-speed]))
    if action == -1:
        return bool(np.array_equal(previous[:, :-speed], current[:, speed:]))
    return False


def _reentry_diagnostics(actions: np.ndarray, *, world: int, viewport: int, speed: int) -> dict:
    recent = 6
    evaluated = 0
    answerable_novel = 0
    any_novel = 0
    fractions: list[float] = []
    columns = np.arange(viewport)
    for clip_actions in actions:
        positions = np.cumsum(clip_actions.astype(np.int64) * speed) % world
        seen = np.zeros(world, dtype=bool)
        for frame, position in enumerate(positions):
            visible = (position + columns) % world
            if frame >= recent:
                recent_seen = np.zeros(world, dtype=bool)
                for old_position in positions[frame - recent:frame]:
                    recent_seen[(old_position + columns) % world] = True
                novel = ~recent_seen[visible]
                if novel.any():
                    any_novel += 1
                    answerable = seen[visible][novel]
                    fraction = float(answerable.mean())
                    fractions.append(fraction)
                    if answerable.any():
                        answerable_novel += 1
                evaluated += 1
            seen[visible] = True
    return {
        "recent_window_frames": recent,
        "evaluated_frames": evaluated,
        "frames_with_recently_novel_columns": any_novel,
        "frames_with_answerable_reentry_columns": answerable_novel,
        "answerable_reentry_frame_rate": answerable_novel / max(evaluated, 1),
        "mean_answerable_fraction_given_novel": float(np.mean(fractions)) if fractions else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-lineage", required=True)
    parser.add_argument("--split-seed", type=int, default=260901)
    parser.add_argument("--validation-clips", type=int, default=300)
    parser.add_argument("--transition-samples", type=int, default=4096)
    parser.add_argument("--world", type=int, default=192)
    parser.add_argument("--viewport", type=int, default=64)
    parser.add_argument("--speed", type=int, default=4)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    clips = np.load(args.clips, mmap_mode="r")
    actions = np.load(args.actions, mmap_mode="r")
    if clips.ndim != 4 or clips.shape[2:] != (args.viewport, args.viewport):
        raise ValueError(f"clips must have shape [N,T,{args.viewport},{args.viewport}]")
    if clips.dtype != np.uint8:
        raise ValueError(f"clips must be uint8, got {clips.dtype}")
    if actions.shape != clips.shape[:2]:
        raise ValueError(f"actions shape {actions.shape} does not match clips {clips.shape[:2]}")
    if actions.dtype != np.int8:
        raise ValueError(f"actions must be int8, got {actions.dtype}")
    values, counts = np.unique(actions, return_counts=True)
    if set(values.tolist()) != {-1, 0, 1}:
        raise ValueError(f"unexpected action vocabulary: {values.tolist()}")
    if not 0 < args.validation_clips < clips.shape[0]:
        raise ValueError("validation-clips must leave non-empty train and validation sets")

    rng = np.random.default_rng(args.split_seed)
    permutation = rng.permutation(clips.shape[0])
    validation_indices = np.sort(permutation[: args.validation_clips])
    train_indices = np.sort(permutation[args.validation_clips :])
    if np.intersect1d(train_indices, validation_indices).size:
        raise AssertionError("train/validation clip overlap")

    sample_rng = np.random.default_rng(260901)
    failures = 0
    checked = min(args.transition_samples, clips.shape[0] * (clips.shape[1] - 1))
    for _ in range(checked):
        clip_index = int(sample_rng.integers(0, clips.shape[0]))
        frame = int(sample_rng.integers(1, clips.shape[1]))
        if not _transition_matches(
            clips[clip_index, frame - 1],
            clips[clip_index, frame],
            int(actions[clip_index, frame]),
            args.speed,
        ):
            failures += 1
    if failures:
        raise ValueError(f"{failures}/{checked} sampled viewport transitions violate the action")

    diagnostic_clip_indices = permutation[: min(256, clips.shape[0])]
    reentry = _reentry_diagnostics(
        np.asarray(actions[diagnostic_clip_indices]),
        world=args.world,
        viewport=args.viewport,
        speed=args.speed,
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "trb-v2-development",
        "display_name": "Toroidal Re-entry Benchmark (TRB)",
        "legacy_name": "LoopMom",
        "purpose": "development_only",
        "test_policy": {
            "included": False,
            "reason": "a fresh sealed test set is generated only after protocol lock",
            "legacy_loopmom_test_allowed": False,
        },
        "configuration": {
            "world_width": args.world,
            "viewport": args.viewport,
            "speed": args.speed,
            "action_vocabulary": [-1, 0, 1],
            "clips": int(clips.shape[0]),
            "frames_per_clip": int(clips.shape[1]),
        },
        "generator_provenance": {
            "lineage": args.generator_lineage,
            "generation_seed": "not recorded for this legacy artifact",
            "limitation": (
                "pixel/action hashes make the existing artifact immutable, but do not recover "
                "the missing original generation seed"
            ),
        },
        "artifacts": {
            "clips": {"path": str(args.clips.resolve()), "sha256": sha256_file(args.clips)},
            "actions": {"path": str(args.actions.resolve()), "sha256": sha256_file(args.actions)},
        },
        "split": {
            "method": "fixed seeded clip permutation",
            "seed": args.split_seed,
            "train_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
            "train_count": int(train_indices.size),
            "validation_count": int(validation_indices.size),
        },
        "audit": {
            "shape": list(clips.shape),
            "clip_dtype": str(clips.dtype),
            "action_dtype": str(actions.dtype),
            "action_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
            "sampled_transition_checks": checked,
            "transition_failures": failures,
            "reentry_geometry": reentry,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(args.output),
        "train": int(train_indices.size),
        "validation": int(validation_indices.size),
        "transition_checks": checked,
        "reentry": reentry,
    }, indent=2))


if __name__ == "__main__":
    main()
