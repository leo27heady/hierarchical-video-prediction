"""Structural/distribution/collision gate for the untouched loopmom holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def _clip_hashes(array: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(clip).tobytes()).hexdigest() for clip in array}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--legacy-train", default="data/loopmom_train_3000x100.npy")
    parser.add_argument("--legacy-test", default="data/loopmom_test_500x200.npy")
    parser.add_argument("--comparison-manifest", action="append", default=[],
                        help="additional locked dataset manifest to audit for exact clip collisions")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    failures = []
    artifacts = manifest["artifacts"]
    for name, artifact in artifacts.items():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            failures.append(f"{name} artifact missing/hash mismatch")

    clips = np.load(artifacts["clips"]["path"], mmap_mode="r")
    actions = np.load(artifacts["actions"]["path"])
    positions = np.load(artifacts["positions"]["path"])
    expected_shape = (
        config["dataset"]["clips"], config["dataset"]["frames"],
        config["dataset"]["viewport"], config["dataset"]["viewport"],
    )
    if clips.shape != expected_shape or clips.dtype != np.uint8:
        failures.append(f"clip schema mismatch: {clips.shape}/{clips.dtype}")
    if actions.shape != expected_shape[:2] or positions.shape != expected_shape[:2]:
        failures.append("action/position shape mismatch")
    if not set(np.unique(actions)).issubset({-1, 0, 1}):
        failures.append("invalid action value")

    world, speed = config["dataset"]["world"], config["dataset"]["speed"]
    expected_delta = (actions[:, 1:].astype(np.int64) * speed) % world
    actual_delta = (positions[:, 1:].astype(np.int64) - positions[:, :-1]) % world
    if not np.array_equal(expected_delta, actual_delta):
        failures.append("positions are inconsistent with actions")
    if positions.min() < 0 or positions.max() >= world:
        failures.append("position outside torus")

    with np.load(artifacts["objects"]["path"]) as objects:
        digits = int(config["dataset"]["digits"])
        expected_objects_shape = (config["dataset"]["clips"], digits)
        for key in ("object_id", "glyph_index", "world_x", "world_y"):
            if key not in objects or objects[key].shape != expected_objects_shape:
                failures.append(f"object metadata schema mismatch: {key}")
        if "object_id" in objects and objects["object_id"].shape == expected_objects_shape:
            expected_ids = np.broadcast_to(np.arange(digits), expected_objects_shape)
            if not np.array_equal(objects["object_id"], expected_ids):
                failures.append("persistent object IDs are not canonical within clips")
        if "glyph_index" in objects and np.any(objects["glyph_index"] < 0):
            failures.append("negative source glyph index")
        if "world_x" in objects and (objects["world_x"].min() < 0 or objects["world_x"].max() >= world):
            failures.append("object world_x outside torus")
        if "world_y" in objects and (objects["world_y"].min() < 0 or objects["world_y"].max() >= expected_shape[-2]):
            failures.append("object world_y outside viewport height")

    p_hold = float((actions == 0).mean())
    nonzero = actions != 0
    continuations = []
    for row, mask in zip(actions, nonzero):
        directions = row[mask]
        if len(directions) > 1:
            continuations.extend((directions[1:] == directions[:-1]).tolist())
    p_continue = float(np.mean(continuations))
    if not 0.08 <= p_hold <= 0.12:
        failures.append(f"measured p_hold outside frozen tolerance: {p_hold:.4f}")
    if not 0.87 <= p_continue <= 0.93:
        failures.append(f"measured p_cont outside frozen tolerance: {p_continue:.4f}")

    confirmation_hashes = _clip_hashes(clips)
    if len(confirmation_hashes) != len(clips):
        failures.append("duplicate clips inside confirmation set")
    collision_counts = {}
    for label, path in (("legacy_train", args.legacy_train), ("legacy_test", args.legacy_test)):
        legacy = np.load(path, mmap_mode="r")
        collisions = len(confirmation_hashes & _clip_hashes(legacy))
        collision_counts[label] = collisions
        if collisions:
            failures.append(f"{collisions} exact clip collisions with {label}")
    for manifest_arg in args.comparison_manifest:
        other_manifest = json.loads(Path(manifest_arg).read_text(encoding="utf-8"))
        label = other_manifest.get("dataset_id", Path(manifest_arg).stem)
        other = np.load(other_manifest["artifacts"]["clips"]["path"], mmap_mode="r")
        collisions = len(confirmation_hashes & _clip_hashes(other))
        collision_counts[label] = collisions
        if collisions:
            failures.append(f"{collisions} exact clip collisions with {label}")

    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": sha256_file(args.manifest),
        "shape": list(clips.shape),
        "dtype": str(clips.dtype),
        "p_hold": p_hold,
        "p_continue_nonzero": p_continue,
        "position_action_consistency": not any("positions are inconsistent" in f for f in failures),
        "object_metadata_valid": not any("object" in f for f in failures),
        "unique_confirmation_clips": len(confirmation_hashes),
        "collisions": collision_counts,
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
