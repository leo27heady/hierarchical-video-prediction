"""Independent immutable-artifact, budget, pairing, and decision validator for E7c."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--gpu-ledger", required=True)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if results["config_sha256"] != sha256_file(args.config):
        raise ValueError("E7c results are not bound to the frozen config")
    if results["dataset_manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("E7c results are not bound to the frozen dataset")

    expected = {
        (arm["arm"], int(v["nfe"]), int(v["sampler_seed"]))
        for arm in config["arms"] for v in config["variants"]
    }
    found = {}
    dataset_hash = manifest["artifacts"]["clips"]["sha256"]
    for spec_path in Path(args.run_root).rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "robustness-e7c" or spec.get("dataset_hash") != dataset_hash:
            continue
        key = (spec["arm"], int(spec["sampler"]["nfe"]), int(spec["sampler"]["sampler_seed"]))
        if key in found:
            raise ValueError(f"duplicate completed E7c RunSpec: {key}")
        if spec["metric_version"] != config["evaluation"]["metric_version"]:
            raise ValueError(f"metric version changed: {key}")
        if spec["warmup_frames"] != 48 or spec["rollout_frames"] != 152 or spec["evaluation_clips"] != 100:
            raise ValueError(f"evaluation geometry changed: {key}")
        metric_path = run_dir / "metrics/rollout_complete.npz"
        manifest_path = run_dir / "manifest.json"
        evaluation_path = run_dir / "evaluation.json"
        if not all(path.is_file() for path in (metric_path, manifest_path, evaluation_path)):
            raise FileNotFoundError(f"incomplete immutable artifacts: {key}")
        with np.load(metric_path) as data:
            for required in ("clip", "h", "fg", "mse_full", "mse_union", "fg_precision",
                             "missed_components", "extra_components"):
                if required not in data or len(data[required]) != 15200:
                    raise ValueError(f"missing/misaligned complete metric {required}: {key}")
        found[key] = metric_path
    if set(found) != expected:
        raise ValueError(f"E7c completed-cell mismatch: missing={sorted(expected-set(found))}")

    ledger_rows = [json.loads(line) for line in Path(args.gpu_ledger).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    accepted = [row for row in ledger_rows if row.get("study") == "robustness-e7c" and row.get("exit_code") == 0]
    ledger_keys = {(row["arm"], int(row["nfe"]), int(row["sampler_seed"])) for row in accepted}
    if ledger_keys != expected:
        raise ValueError("E7c GPU-ledger inventory differs from completed RunSpecs")
    total = sum(float(row["gpu_hours"]) for row in ledger_rows if row.get("study") == "robustness-e7c")
    if total > float(config["budget"]["authorized_max_gpu_hours"]):
        raise ValueError(f"E7c exceeded its GPU authorization: {total:.4f}")

    primary = results["primary"]
    if len(primary) != 20:
        raise ValueError("E7c primary family must contain exactly 20 effects")
    positive = all(float(row["mean"]) > 0 for row in primary)
    clean = all(float(row["lo"]) > 0 for row in primary)
    expected_verdict = ("ROBUST_CI_CLEAN" if positive and clean else
                        "DIRECTIONALLY_ROBUST" if positive else "SAMPLER_OR_NFE_SENSITIVE")
    if results["decision"]["verdict"] != expected_verdict:
        raise ValueError("E7c terminal decision does not follow the frozen rule")
    for key, path in found.items():
        stored = results["input_artifacts"].get("::".join(map(str, key)))
        if stored != sha256_file(path):
            raise ValueError(f"E7c analysis input hash mismatch: {key}")

    report = {
        "schema_version": 1,
        "status": "COMPLETION_VALIDATION_PASS",
        "study": "robustness-e7c",
        "dataset_id": manifest["dataset_id"],
        "evaluations": len(found),
        "primary_effects": len(primary),
        "gpu_hours": total,
        "authorized_gpu_hours": float(config["budget"]["authorized_max_gpu_hours"]),
        "verdict": expected_verdict,
        "results_sha256": sha256_file(args.results),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
