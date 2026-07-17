"""Independent terminal inventory, alignment, budget and provenance validation for E6."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hvp_rev.provenance import atomic_write_json, sha256_file


def _verify_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    complete = run_dir / "COMPLETE"
    if not manifest_path.is_file() or not complete.is_file():
        raise ValueError(f"incomplete immutable run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"run artifact hash mismatch: {path}")


def _discover(base: Path, study: str, arms: list[str], seeds: list[int]) -> dict:
    found = {}
    root = base / study
    for spec_path in root.rglob("run_spec.json") if root.exists() else []:
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        key = (spec["arm"], int(spec["seed"]))
        if key in found:
            raise ValueError(f"duplicate completed {study} cell: {key}")
        _verify_manifest(run_dir)
        found[key] = (run_dir, spec)
    expected = {(arm, seed) for arm in arms for seed in seeds}
    if set(found) != expected:
        raise ValueError(f"{study} inventory {set(found)} != {expected}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--gpu-ledger", required=True)
    parser.add_argument("--access-log", required=True)
    parser.add_argument("--wandb-audit", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    holdout = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    wandb = json.loads(Path(args.wandb_audit).read_text(encoding="utf-8"))
    if config["status"] != "FROZEN":
        raise ValueError("terminal E6 config is not frozen")
    if holdout["purpose"] != "phase_c_v2_holdout":
        raise ValueError("E6 terminal manifest has the wrong purpose")
    if results["dataset_id"] != holdout["dataset_id"] or len(results["primary"]) != 15:
        raise ValueError("E6 results do not contain the registered protected family")
    if results["decision"]["verdict"] not in {
        "HIERARCHY_ADVANTAGE", "FLAT_ADVANTAGE", "MIXED_BOUNDARY"
    }:
        raise ValueError("E6 result lacks a valid complete registered verdict")
    if wandb.get("status") != "WANDB_GROUP_AUDIT_PASS":
        raise ValueError("E6 W&B group audit did not pass")

    arms = list(config["training"]["arms"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    training = _discover(Path(args.training_root), "phase-c-v2-predictors", arms, seeds)
    evaluation = _discover(Path(args.run_root), "phase-c-v2-evaluation", arms, seeds)
    expected_keys = None
    evaluation_hashes = {}
    for key, (run_dir, spec) in evaluation.items():
        if spec["arm_config"]["dataset_id"] != holdout["dataset_id"]:
            raise ValueError(f"E6 evaluation uses the wrong holdout: {key}")
        path = run_dir / "metrics" / "phasec_v2_records.npz"
        with np.load(path) as archive:
            if len(archive["clip"]) != config["evaluation"]["holdout_clips"] * config["evaluation"]["rollout_frames"]:
                raise ValueError(f"E6 evaluation record count differs: {key}")
            keys = np.stack([archive["clip"], archive["h"]], axis=1)
            if len(np.unique(keys, axis=0)) != len(keys):
                raise ValueError(f"E6 evaluation duplicates paired keys: {key}")
            order = np.lexsort((archive["h"], archive["clip"]))
            keys = keys[order]
            if expected_keys is None:
                expected_keys = keys
            elif not np.array_equal(expected_keys, keys):
                raise ValueError(f"E6 paired geometry differs: {key}")
            for required in (
                "mse_gt_fg_nonempty", "mse_union", "object_trajectory_error",
                "object_identity_accuracy", "object_coverage", "object_missed", "object_extra",
            ):
                if required not in archive.files or not np.isfinite(archive[required]).all():
                    raise ValueError(f"E6 endpoint missing/nonfinite: {key}/{required}")
        evaluation_hashes[f"{key[0]}-s{key[1]}"] = sha256_file(path)

    rows = [
        json.loads(line) for line in Path(args.gpu_ledger).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_hours = float(sum(float(row["gpu_hours"]) for row in rows))
    if total_hours > config["budget"]["authorized_max_gpu_hours"]:
        raise ValueError("E6 mechanical GPU ledger exceeds 40 hours")
    for row in rows:
        if float(row["gpu_hours"]) > float(row["cell_authorized_max_gpu_hours"]):
            raise ValueError(f"E6 cell exceeded its allocation: {row}")
    successful = [row for row in rows if int(row["exit_code"]) == 0]
    failed = [row for row in rows if int(row["exit_code"]) != 0]
    phase_counts = {}
    for row in successful:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
    expected_phases = {
        "codec-training": 1,
        "classifier-training": 1,
        "metric-validation": 1,
        "predictor-training": 6,
        "protected-evaluation": 6,
    }
    if phase_counts != expected_phases:
        raise ValueError(f"E6 GPU phase inventory {phase_counts} != {expected_phases}")

    access = [
        json.loads(line) for line in Path(args.access_log).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_access = [
        row for row in access if row["dataset_id"] == holdout["dataset_id"]
    ]
    if len(dataset_access) != 7:
        raise ValueError("E6 requires one integrity access plus six registered evaluation accesses")
    if len({row["run_spec_hash"] for row in dataset_access}) != 7:
        raise ValueError("E6 protected access RunSpec identities are not unique")

    report = {
        "schema_version": 1,
        "status": "PHASEC_V2_COMPLETION_PASS",
        "dataset_id": holdout["dataset_id"],
        "verdict": results["decision"]["verdict"],
        "training_cells": len(training),
        "evaluation_cells": len(evaluation),
        "paired_records_per_cell": int(len(expected_keys)),
        "protected_access_entries": len(dataset_access),
        "gpu_hours": total_hours,
        "gpu_phase_counts": phase_counts,
        "charged_failed_cells": len(failed),
        "wandb_run_count": wandb["run_count"],
        "evaluation_artifacts": evaluation_hashes,
        "hashes": {
            "config": sha256_file(args.config),
            "manifest": sha256_file(args.manifest),
            "results": sha256_file(args.results),
            "gpu_ledger": sha256_file(args.gpu_ledger),
            "access_log": sha256_file(args.access_log),
            "wandb_audit": sha256_file(args.wandb_audit),
        },
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
