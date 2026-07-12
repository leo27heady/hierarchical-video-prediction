"""Fail-closed acceptance audit for a completed E2 confirmation tranche."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file, sha256_value


IDENTITY_KEYS = ("clip", "h", "hard", "dt", "seen", "fg_nonempty")
FINITE_KEYS = (
    "clip", "h", "hard", "dt", "seen", "fg", "sharp", "pers_fg", "fg_nonempty",
    "pred_fg_nonempty", "mse_full", "mse_gt_bg", "mse_union",
    "fg_precision", "fg_recall", "fg_f1",
    "gt_component_count", "pred_component_count", "missed_components", "extra_components",
)
CONDITIONAL_NONEMPTY_KEYS = ("mse_gt_fg_nonempty", "fg_laplacian_energy_nonempty")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_key(run: dict) -> tuple[str, int, int]:
    return run["arm"], int(run["seed"]), int(run["steps"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/e2_untouched_confirmation.json")
    parser.add_argument("--manifest", default="data/confirmation/loopmom_confirmation_500x200.manifest.json")
    parser.add_argument("--results", default="artifacts/confirmation_e2/results.json")
    parser.add_argument("--run-root", default="artifacts/runs/confirmation-e2")
    parser.add_argument("--ledger", default="artifacts/gpu_hours.jsonl")
    parser.add_argument("--access-log", default="artifacts/confirmation_e2/access.jsonl")
    parser.add_argument("--out", default="artifacts/confirmation_e2/completion_validation.json")
    args = parser.parse_args()

    paths = {name: Path(value) for name, value in vars(args).items() if name != "out"}
    failures: list[str] = []
    caveats: list[str] = []
    config = _read_json(paths["config"])
    dataset_manifest = _read_json(paths["manifest"])
    results = _read_json(paths["results"])
    manifest_sha = sha256_file(paths["manifest"])
    dataset_sha = dataset_manifest["artifacts"]["clips"]["sha256"]

    if config.get("status") != "FROZEN":
        failures.append("E2 configuration is not frozen")
    if results.get("config_sha256") != sha256_file(paths["config"]):
        failures.append("analysis is not bound to the frozen config")
    if results.get("dataset_manifest_sha256") != manifest_sha:
        failures.append("analysis is not bound to the frozen dataset manifest")

    expected = {_run_key(run): run for run in config["runs"]}
    discovered: dict[tuple[str, int, int], tuple[Path, dict]] = {}
    for spec_path in paths["run_root"].rglob("run_spec.json"):
        run_dir = spec_path.parent
        spec = _read_json(spec_path)
        if spec.get("study") != "confirmation-e2" or spec.get("dataset_hash") != dataset_sha:
            continue
        key = (spec["arm"], int(spec["seed"]), int(spec["optimizer_updates"]))
        if key in discovered:
            failures.append(f"duplicate completed RunSpec key: {key}")
        discovered[key] = (run_dir, spec)

    if set(discovered) != set(expected):
        failures.append(
            f"completed RunSpec keys differ: missing={sorted(set(expected)-set(discovered))}, "
            f"extra={sorted(set(discovered)-set(expected))}"
        )

    reference_identity: dict[str, np.ndarray] | None = None
    run_rows = []
    successful_identities = []
    for key in sorted(set(expected) & set(discovered)):
        run = expected[key]
        run_dir, spec = discovered[key]
        identity = sha256_value(spec)[:16]
        successful_identities.append(identity)
        row_failures = []
        if run_dir.name != identity:
            row_failures.append("RunSpec directory identity mismatch")
        if spec["arm_config"].get("checkpoint_sha256") != run["checkpoint_sha256"]:
            row_failures.append("checkpoint hash differs from frozen config")
        if spec.get("batch_size") != run["batch"]:
            row_failures.append("batch size differs from frozen config")
        if not (run_dir / "COMPLETE").is_file():
            row_failures.append("COMPLETE marker missing")
        elif (run_dir / "COMPLETE").read_text(encoding="utf-8").strip() != identity:
            row_failures.append("COMPLETE marker identity mismatch")

        artifact_manifest_path = run_dir / "manifest.json"
        if not artifact_manifest_path.is_file():
            row_failures.append("artifact manifest missing")
        else:
            artifact_manifest = _read_json(artifact_manifest_path)
            if artifact_manifest.get("run_spec_hash") != identity:
                row_failures.append("artifact manifest identity mismatch")
            for artifact in artifact_manifest.get("artifacts", []):
                artifact_path = run_dir / artifact["path"]
                if not artifact_path.is_file():
                    row_failures.append(f"missing artifact {artifact['path']}")
                    continue
                if artifact_path.stat().st_size != artifact["bytes"]:
                    row_failures.append(f"byte count changed for {artifact['path']}")
                if sha256_file(artifact_path) != artifact["sha256"]:
                    row_failures.append(f"hash changed for {artifact['path']}")

        evaluation_path = run_dir / "evaluation.json"
        if evaluation_path.is_file():
            evaluation = _read_json(evaluation_path)
            if evaluation.get("confirmation_manifest_sha256") != manifest_sha:
                row_failures.append("evaluation manifest hash mismatch")
        else:
            row_failures.append("evaluation record missing")

        metric_path = run_dir / "metrics" / "rollout_complete.npz"
        if metric_path.is_file():
            with np.load(metric_path) as arrays:
                missing = sorted(
                    set(FINITE_KEYS + CONDITIONAL_NONEMPTY_KEYS + IDENTITY_KEYS) - set(arrays.files)
                )
                if missing:
                    row_failures.append(f"metric keys missing: {missing}")
                else:
                    expected_records = int(config["evaluation"]["evaluation_clips"]) * int(
                        config["evaluation"]["rollout_frames"]
                    )
                    lengths = {name: len(arrays[name]) for name in arrays.files}
                    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != expected_records:
                        row_failures.append(f"record counts differ: {sorted(set(lengths.values()))}")
                    for name in FINITE_KEYS:
                        if not np.isfinite(arrays[name]).all():
                            row_failures.append(f"non-finite required metric: {name}")
                    nonempty = arrays["fg_nonempty"].astype(bool)
                    for name in CONDITIONAL_NONEMPTY_KEYS:
                        if not np.isfinite(arrays[name][nonempty]).all():
                            row_failures.append(f"non-finite non-empty metric on its population: {name}")
                        if np.isfinite(arrays[name][~nonempty]).any():
                            row_failures.append(f"non-empty-only metric is defined on empty frames: {name}")
                    current = {name: arrays[name].copy() for name in IDENTITY_KEYS}
                    if reference_identity is None:
                        reference_identity = current
                    else:
                        for name in IDENTITY_KEYS:
                            if not np.array_equal(reference_identity[name], current[name]):
                                row_failures.append(f"paired identity differs: {name}")
        else:
            row_failures.append("metric NPZ missing")

        failures.extend(f"{key}: {message}" for message in row_failures)
        run_rows.append({"key": list(key), "run_spec_hash": identity, "status": "PASS" if not row_failures else "FAIL"})

    ledger = [json.loads(line) for line in paths["ledger"].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ledger_keys = [(row["arm"], int(row["seed"]), int(row["steps"])) for row in ledger]
    if Counter(ledger_keys) != Counter({key: 1 for key in expected}):
        failures.append("GPU ledger does not contain exactly one row per frozen run")
    if any(int(row.get("exit_code", 1)) != 0 for row in ledger):
        failures.append("GPU ledger contains a failed completed evaluation")
    successful_gpu_hours = float(sum(float(row["gpu_hours"]) for row in ledger))

    accesses = [json.loads(line) for line in paths["access_log"].read_text(encoding="utf-8").splitlines() if line.strip()]
    access_counts = Counter(row["run_spec_hash"] for row in accesses)
    missing_accesses = sorted(identity for identity in successful_identities if access_counts[identity] != 1)
    if missing_accesses:
        failures.append(f"successful RunSpec access count is not exactly one: {missing_accesses}")
    extra_accesses = [row for row in accesses if row["run_spec_hash"] not in set(successful_identities)]
    if extra_accesses:
        caveats.append(
            f"{len(extra_accesses)} additional frozen-protocol access(es) had no completed artifact; "
            "the aborted launch must be disclosed and charged conservatively"
        )

    if len(results.get("primary", [])) != 48:
        failures.append("analysis does not contain the expected 48 selected-pair rows")
    if len(results.get("frontier", {}).get("contrasts", {})) != 12:
        failures.append("analysis does not contain the expected 12 frontier contrasts")
    if results.get("frontier", {}).get("method") != "studentized max-statistic clip-cluster bootstrap":
        failures.append("frontier multiplicity method changed")

    report = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "config_sha256": sha256_file(paths["config"]),
        "dataset_manifest_sha256": manifest_sha,
        "analysis_sha256": sha256_file(paths["results"]),
        "expected_runs": len(expected),
        "completed_runs": len(discovered),
        "successful_gpu_hours": successful_gpu_hours,
        "access_records": len(accesses),
        "additional_aborted_accesses": extra_accesses,
        "runs": run_rows,
        "caveats": caveats,
        "failures": failures,
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
