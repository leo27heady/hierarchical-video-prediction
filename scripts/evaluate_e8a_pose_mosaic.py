"""Evaluate the frozen E8a pose-indexed raw-memory references.

The implementation is intentionally independent of the neural rollout path. It uses only
ground-truth warm-up pixels, saved viewport positions and a training-only row-intensity prior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev import wb
from hvp_rev.evaluation.frame_metrics import evaluate_frames
from hvp_rev.provenance import (
    ArtifactStore,
    RunSpec,
    atomic_write_json,
    log_holdout_access,
    sha256_file,
)


COMPLETE_KEYS = (
    "fg_nonempty",
    "pred_fg_nonempty",
    "mse_full",
    "mse_gt_fg_nonempty",
    "mse_gt_bg",
    "mse_union",
    "fg_laplacian_energy_nonempty",
    "fg_precision",
    "fg_recall",
    "fg_f1",
    "gt_component_count",
    "pred_component_count",
    "missed_components",
    "extra_components",
    "matched_component_iou",
    "matched_centroid_l2",
)


def compute_row_prior(path: str | Path, *, chunk_clips: int = 16) -> np.ndarray:
    """Training-only mean intensity for each image row, in [0, 1]."""
    clips = np.load(path, mmap_mode="r")
    if clips.ndim != 4 or clips.shape[-2:] != (64, 64) or clips.dtype != np.uint8:
        raise ValueError(f"unexpected prior-source array: shape={clips.shape}, dtype={clips.dtype}")
    total = np.zeros(64, dtype=np.uint64)
    count = 0
    for start in range(0, len(clips), chunk_clips):
        block = np.asarray(clips[start:start + chunk_clips])
        total += block.sum(axis=(0, 1, 3), dtype=np.uint64)
        count += int(block.shape[0] * block.shape[1] * block.shape[3])
    if count <= 0:
        raise ValueError("empty training array")
    return (total.astype(np.float64) / (count * 255.0)).astype(np.float32)


def build_pose_mosaic(
    frames_u8: np.ndarray,
    positions: np.ndarray,
    *,
    world: int,
    warm: int,
    span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a static world-coordinate mosaic from the final ``span`` warm-up frames."""
    frames_u8 = np.asarray(frames_u8)
    positions = np.asarray(positions)
    if frames_u8.ndim != 3 or frames_u8.shape[1:] != (64, 64):
        raise ValueError(f"expected (T,64,64) frames, got {frames_u8.shape}")
    if span <= 0 or span > warm or warm > frames_u8.shape[0]:
        raise ValueError(f"invalid warm/span pair: warm={warm}, span={span}")
    if positions.shape[0] < warm:
        raise ValueError("positions do not cover warm-up")
    mosaic = np.zeros((64, world), dtype=np.uint8)
    observed = np.zeros(world, dtype=bool)
    viewport_columns = np.arange(64, dtype=np.int64)
    for t in range(warm - span, warm):
        world_columns = (int(positions[t]) + viewport_columns) % world
        frame = frames_u8[t]
        for view_x, world_x in enumerate(world_columns):
            column = frame[:, view_x]
            if observed[world_x]:
                if not np.array_equal(mosaic[:, world_x], column):
                    raise ValueError(
                        f"static-world consistency failure at t={t}, world_x={world_x}"
                    )
            else:
                mosaic[:, world_x] = column
                observed[world_x] = True
    return mosaic, observed


def query_pose_mosaic(
    mosaic: np.ndarray,
    observed: np.ndarray,
    positions: np.ndarray,
    *,
    row_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Render crops at the requested poses and return prediction plus observed-column mask."""
    positions = np.asarray(positions)
    row_prior = np.asarray(row_prior, dtype=np.float32)
    if row_prior.shape != (64,):
        raise ValueError(f"row prior must have shape (64,), got {row_prior.shape}")
    world = mosaic.shape[1]
    predictions = np.empty((len(positions), 64, 64), dtype=np.float32)
    coverage = np.empty((len(positions), 64), dtype=bool)
    viewport_columns = np.arange(64, dtype=np.int64)
    for i, position in enumerate(positions):
        indices = (int(position) + viewport_columns) % world
        known = observed[indices]
        crop = mosaic[:, indices].astype(np.float32) / 255.0
        if not known.all():
            crop[:, ~known] = row_prior[:, None]
        predictions[i] = crop
        coverage[i] = known
    return predictions, coverage


def _legacy_fg_mse(pred: np.ndarray, true: np.ndarray, *, threshold: float) -> np.ndarray:
    mask = true > threshold
    squared = (pred - true) ** 2
    count = mask.sum(axis=(1, 2))
    foreground = (squared * mask).sum(axis=(1, 2)) / np.maximum(count, 1)
    full = squared.mean(axis=(1, 2))
    return np.where(count > 0, foreground, full).astype(np.float32)


def _masked_mse(
    pred: np.ndarray, true: np.ndarray, columns: np.ndarray, *, invert: bool = False
) -> np.ndarray:
    use = ~columns if invert else columns
    mask = np.broadcast_to(use[:, None, :], pred.shape)
    count = mask.sum(axis=(1, 2))
    total = (((pred - true) ** 2) * mask).sum(axis=(1, 2))
    out = np.full(len(pred), np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def geometric_labels(
    frames_u8: np.ndarray,
    positions: np.ndarray,
    *,
    world: int,
    warm: int,
    threshold: float,
    n_short: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-NumPy equivalent of RHVD hardness and dt/seen labeling."""
    frames_u8 = np.asarray(frames_u8)
    positions = np.asarray(positions)
    horizons = frames_u8.shape[0] - warm
    # RHVD appends Python floats and therefore serializes these labels as float64. Preserve that
    # dtype and arithmetic so the classical and neural artifacts pair exactly, not approximately.
    hard = np.zeros(horizons, dtype=np.float64)
    dt_out = np.zeros(horizons, dtype=np.float64)
    seen_out = np.ones(horizons, dtype=np.float64)
    viewport = np.arange(64, dtype=np.int64)
    last_seen = np.full(world, -(10**9), dtype=np.int64)
    for t in range(warm):
        last_seen[(int(positions[t]) + viewport) % world] = t
    for offset, t in enumerate(range(warm, frames_u8.shape[0])):
        foreground = np.where(frames_u8[t].max(axis=0) > threshold * 255.0)[0]
        if len(foreground):
            target_world = (int(positions[t]) + foreground) % world
            short_coverage = np.zeros(world, dtype=bool)
            for prior_t in range(t - n_short, t):
                short_coverage[(int(positions[prior_t]) + viewport) % world] = True
            hard[offset] = float((~short_coverage[target_world]).mean())
            delays = t - last_seen[target_world]
            novel = delays > n_short
            if novel.any():
                novel_delays = delays[novel]
                finite = novel_delays < 10**8
                seen_out[offset] = float(finite.mean())
                dt_out[offset] = (
                    float(np.median(novel_delays[finite])) if finite.any() else 1e9
                )
        last_seen[(int(positions[t]) + viewport) % world] = t
    return hard, dt_out, seen_out


def evaluate_variant(
    clips_u8: np.ndarray,
    positions: np.ndarray,
    clip_indices: Iterable[int],
    *,
    row_prior: np.ndarray,
    world: int,
    warm: int,
    span: int,
    threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    clip_indices = tuple(int(index) for index in clip_indices)
    records: dict[str, list] = {
        "h": [],
        "hard": [],
        "fg": [],
        "sharp": [],
        "pers_fg": [],
        "clip": [],
        "dt": [],
        "seen": [],
        "current_crop_observed_fraction": [],
        "stored_world_coverage": [],
        "observed_pixel_mse": [],
        "unobserved_pixel_mse": [],
        "stored_pixel_payload_bytes": [],
        "allocated_mosaic_bytes": [],
        "stored_pose_bytes": [],
        "query_pose_bytes": [],
        "pose_bytes": [],
        "mask_bytes": [],
    }
    records.update({key: [] for key in COMPLETE_KEYS})
    stored_columns: list[int] = []
    for clip_index in clip_indices:
        frames = np.asarray(clips_u8[clip_index])
        clip_positions = np.asarray(positions[clip_index])
        mosaic, observed = build_pose_mosaic(
            frames, clip_positions, world=world, warm=warm, span=span
        )
        predicted, column_mask = query_pose_mosaic(
            mosaic, observed, clip_positions[warm:], row_prior=row_prior
        )
        true = frames[warm:].astype(np.float32) / 255.0
        persistence = np.broadcast_to(
            frames[warm - 1].astype(np.float32) / 255.0, true.shape
        )
        hard, delay, seen = geometric_labels(
            frames, clip_positions, world=world, warm=warm, threshold=threshold
        )
        metrics = evaluate_frames(
            predicted[:, None], true[:, None], threshold=threshold
        )
        legacy = _legacy_fg_mse(predicted, true, threshold=threshold)
        persistence_fg = _legacy_fg_mse(persistence, true, threshold=threshold)
        observed_mse = _masked_mse(predicted, true, column_mask)
        unobserved_mse = _masked_mse(predicted, true, column_mask, invert=True)
        n_future = len(true)
        unique_columns = int(observed.sum())
        stored_columns.append(unique_columns)
        records["h"].extend(range(1, n_future + 1))
        records["hard"].extend(hard)
        records["fg"].extend(legacy)
        records["sharp"].extend(metrics["fg_laplacian_energy_nonempty"])
        records["pers_fg"].extend(persistence_fg)
        records["clip"].extend([int(clip_index)] * n_future)
        records["dt"].extend(delay)
        records["seen"].extend(seen)
        records["current_crop_observed_fraction"].extend(column_mask.mean(axis=1))
        records["stored_world_coverage"].extend([unique_columns / world] * n_future)
        records["observed_pixel_mse"].extend(observed_mse)
        records["unobserved_pixel_mse"].extend(unobserved_mse)
        records["stored_pixel_payload_bytes"].extend([unique_columns * 64] * n_future)
        records["allocated_mosaic_bytes"].extend([int(mosaic.nbytes)] * n_future)
        stored_pose_bytes = span * np.dtype(np.int32).itemsize
        query_pose_bytes = n_future * np.dtype(np.int32).itemsize
        records["stored_pose_bytes"].extend([stored_pose_bytes] * n_future)
        records["query_pose_bytes"].extend([query_pose_bytes] * n_future)
        records["pose_bytes"].extend(
            [stored_pose_bytes + query_pose_bytes] * n_future
        )
        records["mask_bytes"].extend([int(observed.nbytes)] * n_future)
        for key in COMPLETE_KEYS:
            records[key].extend(metrics[key])
    result = {key: np.asarray(value) for key, value in records.items()}
    if len(result["fg"]) != len(clip_indices) * (clips_u8.shape[1] - warm):
        raise AssertionError("unexpected record count")
    if not np.isfinite(result["fg"]).all():
        raise ValueError("non-finite primary metric")
    summary = {
        "records": int(len(result["fg"])),
        "foreground_mse": float(result["fg"].mean()),
        "full_mse": float(result["mse_full"].mean()),
        "background_mse": float(result["mse_gt_bg"].mean()),
        "union_mse": float(result["mse_union"].mean()),
        "mean_current_crop_observed_fraction": float(
            result["current_crop_observed_fraction"].mean()
        ),
        "median_stored_world_columns": float(np.median(stored_columns)),
        "minimum_stored_world_columns": int(np.min(stored_columns)),
        "maximum_stored_world_columns": int(np.max(stored_columns)),
    }
    return result, summary


def _validate_manifest_header(manifest_path: Path, config: dict) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != config["dataset"]["purpose"]:
        raise ValueError("E8a dataset purpose mismatch")
    expected = config["dataset"]
    actual = manifest["configuration"]
    for key in ("n", "frames", "world", "digits", "speed", "p_cont", "p_hold"):
        expected_key = "clips" if key == "n" else key
        if actual[key] != expected[expected_key]:
            raise ValueError(f"dataset configuration mismatch for {key}")
    return manifest


def _validate_manifest_artifacts(manifest: dict) -> None:
    """Hash protected bytes only after the purpose-matched access record exists."""
    for artifact in manifest["artifacts"].values():
        path = Path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"dataset artifact hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/pose_indexed_retrieval.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--access-log", default="artifacts/e8a_pose_mosaic/access.jsonl")
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-required", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest_path = Path(args.manifest)
    manifest = _validate_manifest_header(manifest_path, config)
    prior_source = Path(config["memory"]["unknown_fill"]["training_array"])
    if sha256_file(prior_source) != config["memory"]["unknown_fill"]["training_array_sha256"]:
        raise ValueError("training-only prior source hash mismatch")
    variant_spans = [
        int(row["warmup_frames_stored"]) for row in config["memory"]["variants"]
    ]
    spec = RunSpec(
        study=config["study"],
        arm="pose-mosaic",
        seed=0,
        dataset_hash=manifest["artifacts"]["clips"]["sha256"],
        ae_hash="none-raw-pixel-reference",
        arm_config={
            "variants": config["memory"]["variants"],
            "algorithm": {
                "fusion_rule": config["memory"]["fusion_rule"],
                "query_rule": config["memory"]["query_rule"],
                "unknown_fill": config["memory"]["unknown_fill"],
            },
            "dataset_id": manifest["dataset_id"],
        },
        train_examples=0,
        optimizer_updates=0,
        batch_size=0,
        precision="uint8-memory-fp32-metrics",
        nfe=0,
        warmup_frames=int(config["evaluation"]["warmup_frames"]),
        rollout_frames=int(config["evaluation"]["rollout_frames"]),
        evaluation_clips=int(config["evaluation"]["evaluation_clips"]),
        source_commit=args.source_commit,
        metric_version=config["evaluation"]["metric_version"],
        optimizer={},
        ema={},
        sampler={"name": "exact-pose-query", "stochastic": False},
        tag="privileged-pose-indexed-raw-memory-reference",
    )
    log_holdout_access(
        manifest_path,
        args.access_log,
        run_spec_hash=spec.identity,
        allowed_purpose=config["dataset"]["purpose"],
        operation="E8a pose-indexed reference evaluation",
    )
    _validate_manifest_artifacts(manifest)
    store = ArtifactStore(args.artifact_root, spec)
    run_name = f"e8a-{manifest['dataset_id']}-pose-mosaic"
    wb.init(
        run_name,
        config={"run_spec": spec.resolved(), "design": config},
        project=args.wandb_project,
        run_id=f"e8a-{spec.identity}",
        tags=["loopmom", "e8a", "pose-indexed", "privileged-reference", "cpu"],
        resume="never",
        group=args.wandb_group,
        job_type="pose-mosaic-reference",
        required=args.wandb_required,
    )
    active = Path("artifacts/active_experiment.json")
    atomic_write_json(active, {
        "schema_version": 1,
        "status": "running",
        "study": config["study"],
        "arm": "pose-mosaic",
        "seed": 0,
        "optimizer_updates": 0,
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_hash": spec.identity,
        "run_dir": store.path.as_posix(),
        "wandb_url": wb.active_url(),
        "authorized_gpu_hours": 0.0,
    })
    try:
        row_prior = compute_row_prior(prior_source)
        store.save_npz(
            "prior/row_intensity_prior.npz",
            row_prior=row_prior,
            source_sha256=np.asarray(sha256_file(prior_source)),
        )
        clips_path = Path(manifest["artifacts"]["clips"]["path"])
        positions_path = Path(manifest["artifacts"]["positions"]["path"])
        clips = np.load(clips_path, mmap_mode="r")
        positions = np.load(positions_path, mmap_mode="r")
        start = int(config["dataset"]["evaluation_pool"].split("indices ")[1].split("..")[0])
        count = int(config["evaluation"]["evaluation_clips"])
        clip_indices = tuple(range(start, start + count))
        summaries = {}
        for variant, span in zip(config["memory"]["variants"], variant_spans):
            result, summary = evaluate_variant(
                clips,
                positions,
                clip_indices,
                row_prior=row_prior,
                world=int(config["dataset"]["world"]),
                warm=int(config["evaluation"]["warmup_frames"]),
                span=span,
                threshold=float(config["evaluation"]["metric_threshold"]),
            )
            variant_id = variant["id"]
            store.save_npz(f"metrics/{variant_id}.npz", **result)
            summaries[variant_id] = summary
            wb.log({
                f"{variant_id}/foreground_mse": summary["foreground_mse"],
                f"{variant_id}/full_mse": summary["full_mse"],
                f"{variant_id}/current_crop_coverage":
                    summary["mean_current_crop_observed_fraction"],
                f"{variant_id}/median_stored_world_columns":
                    summary["median_stored_world_columns"],
            }, step=span)
        store.write_json("summary.json", {
            "schema_version": 1,
            "dataset_id": manifest["dataset_id"],
            "run_spec_hash": spec.identity,
            "prior_source": prior_source.as_posix(),
            "prior_source_sha256": sha256_file(prior_source),
            "variants": summaries,
            "wandb_url": wb.active_url(),
        })
        store.finalize()
        wb.summary({
            "run_spec_hash": spec.identity,
            "dataset_id": manifest["dataset_id"],
            "variants_complete": len(summaries),
            "records_per_variant": int(
                config["evaluation"]["evaluation_clips"]
                * config["evaluation"]["rollout_frames"]
            ),
        })
        wb.artifact(
            name=f"e8a-{spec.identity}",
            artifact_type="pose-mosaic-reference",
            paths=[
                str(store.path / "run_spec.json"),
                str(store.path / "prior/row_intensity_prior.npz"),
                *[
                    str(store.path / f"metrics/{variant['id']}.npz")
                    for variant in config["memory"]["variants"]
                ],
                str(store.path / "summary.json"),
                str(store.path / "manifest.json"),
                str(manifest_path),
            ],
            metadata={
                "run_spec_hash": spec.identity,
                "dataset_id": manifest["dataset_id"],
                "privileged_reference": True,
                "token_matched": False,
            },
            aliases=["latest", "pose-mosaic"],
        )
        atomic_write_json(active, {
            "schema_version": 1,
            "status": "complete",
            "study": config["study"],
            "arm": "pose-mosaic",
            "seed": 0,
            "optimizer_updates": 0,
            "pid": os.getpid(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": spec.identity,
            "run_dir": store.path.as_posix(),
            "wandb_url": wb.active_url(),
            "authorized_gpu_hours": 0.0,
        })
        print(json.dumps(summaries, indent=2, sort_keys=True))
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
