"""Validate official split isolation, persistent metadata and exact-clip disjointness for E6."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hvp_rev.data.moving_mnist import SCHEMA_VERSION, load_idx
from hvp_rev.provenance import (
    atomic_write_json,
    log_holdout_access,
    sha256_file,
    sha256_value,
)


def _manifest(path: str, purpose: str, split: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("data_schema") != SCHEMA_VERSION:
        raise ValueError(f"{path} is not {SCHEMA_VERSION}")
    if value.get("purpose") != purpose:
        raise ValueError(f"{path} purpose {value.get('purpose')!r} != {purpose!r}")
    if value["source_glyph_pool"]["split"] != split:
        raise ValueError(f"{path} uses the wrong official glyph split")
    for artifact in value["artifacts"].values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {artifact['path']}")
    return value


def _clip_hashes(path: str) -> set[str]:
    clips = np.load(path, mmap_mode="r")
    return {
        hashlib.sha256(np.ascontiguousarray(clips[index]).tobytes()).hexdigest()
        for index in range(len(clips))
    }


def _validate_bundle(manifest: dict, source_count: int) -> dict:
    clips = np.load(manifest["artifacts"]["clips"]["path"], mmap_mode="r")
    actions = np.load(manifest["artifacts"]["actions"]["path"], mmap_mode="r")
    with np.load(manifest["artifacts"]["metadata"]["path"]) as source:
        meta = {key: source[key] for key in source.files}
    expected = manifest["configuration"]
    if clips.shape != (
        expected["n"], expected["frames"], expected["image_size"], expected["image_size"]
    ):
        raise ValueError("clip shape differs from manifest configuration")
    if actions.shape != clips.shape[:2] or np.any(actions):
        raise ValueError("E6 actions must be identically zero and aligned")
    n, t = clips.shape[:2]
    objects = expected["objects"]
    shapes = {
        "object_id": (n, objects),
        "glyph_index": (n, objects),
        "class_id": (n, objects),
        "trajectory_yx": (n, objects, t, 2),
        "bbox_yxyx": (n, objects, t, 4),
        "overlap": (n, t, objects, objects),
    }
    for key, shape in shapes.items():
        if meta[key].shape != shape:
            raise ValueError(f"{key} shape {meta[key].shape} != {shape}")
    if len(np.unique(meta["object_id"])) != n * objects:
        raise ValueError("persistent object IDs are not unique within the dataset")
    if meta["glyph_index"].min() < 0 or meta["glyph_index"].max() >= source_count:
        raise ValueError("source glyph index escapes the declared official split")
    boxes = meta["bbox_yxyx"]
    if np.any(boxes[..., 2:] - boxes[..., :2] != 28):
        raise ValueError("generator boxes are not exactly 28x28")
    if not np.array_equal(boxes[..., :2], meta["trajectory_yx"]):
        raise ValueError("trajectory top-left coordinates differ from boxes")
    if np.any(meta["overlap"] != np.swapaxes(meta["overlap"], 2, 3)):
        raise ValueError("overlap matrix is not symmetric")
    unique_hashes = _clip_hashes(manifest["artifacts"]["clips"]["path"])
    if len(unique_hashes) != n:
        raise ValueError("dataset contains duplicate exact clips")
    return {
        "shape": list(clips.shape),
        "objects": int(n * objects),
        "overlap_frame_fraction": float(meta["overlap"].any(axis=(2, 3)).mean()),
        "clip_sha256_count": len(unique_hashes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--holdout-manifest", default="")
    parser.add_argument("--holdout-access-log", default="artifacts/phasec_v2/access.jsonl")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()

    out, marker = Path(args.out), Path(args.marker)
    if out.exists() or marker.exists():
        raise FileExistsError("dataset validation report/marker is immutable")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sources = config["official_mnist"]
    train_images, train_labels = load_idx(sources["train_images"], sources["train_labels"])
    test_images, test_labels = load_idx(sources["test_images"], sources["test_labels"])
    if len(train_images) != 60000 or len(train_labels) != 60000:
        raise ValueError("official MNIST train split must contain 60,000 glyphs")
    if len(test_images) != 10000 or len(test_labels) != 10000:
        raise ValueError("official MNIST test split must contain 10,000 glyphs")

    training = _manifest(args.training_manifest, "training", "train")
    development = _manifest(args.development_manifest, "development", "test")
    values = {
        "training": _validate_bundle(training, len(train_images)),
        "development": _validate_bundle(development, len(test_images)),
    }
    hashes = {
        "training": _clip_hashes(training["artifacts"]["clips"]["path"]),
        "development": _clip_hashes(development["artifacts"]["clips"]["path"]),
    }
    collisions = {
        "training_development": len(hashes["training"] & hashes["development"]),
    }
    manifests = {
        "training": {
            "dataset_id": training["dataset_id"],
            "sha256": sha256_file(args.training_manifest),
        },
        "development": {
            "dataset_id": development["dataset_id"],
            "sha256": sha256_file(args.development_manifest),
        },
    }
    if args.holdout_manifest:
        holdout_manifest_hash = sha256_file(args.holdout_manifest)
        log_holdout_access(
            args.holdout_manifest,
            args.holdout_access_log,
            run_spec_hash=sha256_value({
                "operation": "phase-c-v2-dataset-integrity",
                "config_sha256": sha256_file(args.config),
                "holdout_manifest_sha256": holdout_manifest_hash,
            })[:16],
            allowed_purpose="phase_c_v2_holdout",
            operation="Phase C v2 protected dataset integrity validation",
        )
        holdout = _manifest(args.holdout_manifest, "phase_c_v2_holdout", "test")
        values["holdout"] = _validate_bundle(holdout, len(test_images))
        hashes["holdout"] = _clip_hashes(holdout["artifacts"]["clips"]["path"])
        collisions.update({
            "training_holdout": len(hashes["training"] & hashes["holdout"]),
            "development_holdout": len(hashes["development"] & hashes["holdout"]),
        })
        manifests["holdout"] = {
            "dataset_id": holdout["dataset_id"],
            "sha256": holdout_manifest_hash,
        }
    if any(collisions.values()):
        raise ValueError(f"exact rendered-clip collision: {collisions}")
    report = {
        "schema_version": 1,
        "status": (
            "PHASEC_DATASETS_FINAL_PASS" if args.holdout_manifest
            else "PHASEC_TRAIN_DEVELOPMENT_DATA_PASS"
        ),
        "official_sources": {
            key: {"path": path, "sha256": sha256_file(path)}
            for key, path in sources.items() if key != "mirror"
        },
        "manifests": manifests,
        "datasets": values,
        "exact_clip_collisions": collisions,
        "config_sha256": sha256_file(args.config),
    }
    atomic_write_json(out, report)
    atomic_write_json(marker, {
        "schema_version": 1,
        "status": report["status"],
        "subject": out.as_posix(),
        "subject_sha256": sha256_file(out),
        "source_commit": args.source_commit,
    })
    print(f"{report['status']} {out}", flush=True)


if __name__ == "__main__":
    main()
