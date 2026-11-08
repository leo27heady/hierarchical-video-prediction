"""Independent E8a artifacts, access, alignment, budget and decision validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def _jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _verify_store(run_dir: Path) -> None:
    complete = run_dir / "COMPLETE"
    manifest_path = run_dir / "manifest.json"
    if complete.read_text(encoding="utf-8").strip() != run_dir.name:
        raise ValueError(f"COMPLETE marker mismatch: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_spec_hash") != run_dir.name:
        raise ValueError(f"artifact manifest RunSpec mismatch: {run_dir}")
    for row in manifest.get("artifacts", []):
        path = run_dir / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"immutable artifact mismatch: {path}")


def _canonical_alignment(path: Path) -> dict[str, np.ndarray]:
    keys = ("clip", "h", "hard", "dt", "seen", "fg_nonempty")
    with np.load(path) as values:
        missing = [key for key in keys if key not in values]
        if missing:
            raise ValueError(f"E8a alignment keys missing from {path}: {missing}")
        arrays = {key: np.asarray(values[key]) for key in keys}
    pairs = np.stack([arrays["clip"], arrays["h"]], axis=1)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError(f"E8a duplicate paired record key: {path}")
    order = np.lexsort((arrays["h"], arrays["clip"]))
    return {key: value[order] for key, value in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--gpu-ledger", required=True)
    parser.add_argument("--access-log", required=True)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    config_hash = sha256_file(args.config)
    manifest_hash = sha256_file(args.manifest)
    if (
        results.get("config_sha256") != config_hash
        or results.get("dataset_manifest_sha256") != manifest_hash
        or results.get("dataset_id") != dataset.get("dataset_id")
    ):
        raise ValueError("E8a results are not bound to the frozen config and dataset")
    dataset_hash = dataset["artifacts"]["clips"]["sha256"]
    expected_arms = {row["arm"] for row in config["neural_reference_runs"]}
    expected_mosaics = {row["id"] for row in config["memory"]["variants"]}
    neural = {}
    mosaic_dir = None
    source_commits = set()
    access_specs = set()
    for spec_path in Path(args.run_root).rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "reference-e8a" or spec.get("dataset_hash") != dataset_hash:
            continue
        _verify_store(run_dir)
        source_commits.add(spec["source_commit"])
        access_specs.add(run_dir.name)
        if spec["arm"] == "pose-mosaic":
            if mosaic_dir is not None:
                raise ValueError("duplicate completed E8a pose-mosaic run")
            if (
                spec["warmup_frames"], spec["rollout_frames"], spec["evaluation_clips"],
                spec["nfe"], spec["metric_version"]
            ) != (48, 152, 100, 0, "complete-v2-thr0.5"):
                raise ValueError("E8a pose-mosaic evaluation geometry differs")
            for variant in expected_mosaics:
                path = run_dir / f"metrics/{variant}.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path) as values:
                    for key in (
                        "clip", "h", "fg", "mse_full", "mse_union",
                        "current_crop_observed_fraction", "stored_world_coverage",
                        "stored_pose_bytes", "query_pose_bytes", "pose_bytes",
                    ):
                        if key not in values or len(values[key]) != 15200:
                            raise ValueError(f"missing/misaligned E8a mosaic metric {key}")
            mosaic_dir = run_dir
        else:
            arm = spec["arm"]
            if arm in neural:
                raise ValueError(f"duplicate completed E8a neural evaluation: {arm}")
            expected = next(row for row in config["neural_reference_runs"] if row["arm"] == arm)
            if (
                spec["ae_hash"] != config["codec"]["sha256"]
                or spec["arm_config"]["checkpoint_sha256"] != expected["checkpoint_sha256"]
            ):
                raise ValueError(f"E8a neural artifact identity differs: {arm}")
            if (
                spec["warmup_frames"], spec["rollout_frames"], spec["evaluation_clips"],
                spec["nfe"], spec["metric_version"], spec["sampler"]["sampler_seed"]
            ) != (48, 152, 100, 16, "complete-v2-thr0.5", 0):
                raise ValueError(f"E8a neural evaluation geometry differs: {arm}")
            metric_path = run_dir / "metrics/rollout_complete.npz"
            if not metric_path.is_file():
                raise FileNotFoundError(metric_path)
            with np.load(metric_path) as values:
                for key in ("clip", "h", "fg", "mse_full", "mse_union", "fg_precision"):
                    if key not in values or len(values[key]) != 15200:
                        raise ValueError(f"missing/misaligned E8a neural metric {key}: {arm}")
            neural[arm] = metric_path
    if set(neural) != expected_arms or mosaic_dir is None:
        raise ValueError(
            f"E8a inventory mismatch: missing neural={sorted(expected_arms-set(neural))}; "
            f"mosaic_complete={mosaic_dir is not None}"
        )
    if len(source_commits) != 1:
        raise ValueError(f"E8a RunSpecs use multiple source commits: {source_commits}")
    source_commit = next(iter(source_commits))

    access_rows = _jsonl(args.access_log)
    logged_specs = {row.get("run_spec_hash") for row in access_rows}
    if logged_specs != access_specs or len(access_rows) < 8:
        raise ValueError("E8a access log does not cover every completed RunSpec")
    ledger = _jsonl(args.gpu_ledger)
    if any(row.get("source_commit") != source_commit for row in ledger):
        raise ValueError("E8a GPU ledger contains a different source commit")
    total = sum(float(row["gpu_hours"]) for row in ledger)
    if total > float(config["budget"]["authorized_max_gpu_hours"]):
        raise ValueError(f"E8a exceeded GPU authorization: {total:.4f}")
    successful_arms = {
        row["arm"] for row in ledger
        if row.get("phase") == "neural-evaluation" and row.get("exit_code") == 0
    }
    if successful_arms != expected_arms:
        raise ValueError("E8a GPU ledger lacks the exact seven successful neural evaluations")

    primary = results["primary"]
    if len(primary) != 6:
        raise ValueError("E8a primary family must contain exactly six effects")
    neural_effect_ids = set(config["decision"]["neural_reference_effect_ids"])
    neural_rows = [row for row in primary if row["effect"] in neural_effect_ids]
    if len(neural_rows) != len(neural_effect_ids):
        raise ValueError("E8a neural-reference decision inventory is incomplete")
    all_positive = all(row["mean"] > 0 for row in neural_rows)
    all_clean = all(row["lo"] > 0 for row in neural_rows)
    expected_verdict = (
        "RAW_ACCESS_CEILING_NOT_ESTABLISHED" if not all_positive else
        "RAW_ACCESS_CEILING_CI_CLEAN" if all_clean else
        "RAW_ACCESS_DIRECTIONAL_REFERENCE"
    )
    if results["decision"]["verdict"] != expected_verdict:
        raise ValueError("E8a verdict does not follow the frozen rule")
    if set(results["decision"].get("neural_reference_effect_ids", [])) != neural_effect_ids:
        raise ValueError("E8a decision used a different neural-reference family")
    expected_populations = set(config["evaluation"]["secondary_populations"])
    actual_populations = set(results.get("secondary_populations", {}))
    if actual_populations != expected_populations:
        raise ValueError("E8a secondary-population inventory differs from the frozen config")
    for population_id, population in results["secondary_populations"].items():
        if (
            population.get("records", 0) <= 0
            or population.get("clips", 0) <= 0
            or len(population.get("effects", [])) != 6
            or population.get("rule")
            != config["evaluation"]["secondary_populations"][population_id]
        ):
            raise ValueError(f"invalid E8a secondary population: {population_id}")
    method_paths = dict(neural)
    for variant in expected_mosaics:
        method_paths[variant] = mosaic_dir / f"metrics/{variant}.npz"
    reference_alignment = _canonical_alignment(
        method_paths["pose-mosaic-k48"]
    )
    for method, path in method_paths.items():
        alignment = _canonical_alignment(path)
        for key, expected in reference_alignment.items():
            if not np.array_equal(alignment[key], expected):
                raise ValueError(f"E8a paired record key differs for {method}: {key}")
        if results["input_artifacts"].get(method) != sha256_file(path):
            raise ValueError(f"E8a analysis input hash mismatch: {method}")

    report = {
        "schema_version": 1,
        "status": "COMPLETION_VALIDATION_PASS",
        "study": "reference-e8a",
        "dataset_id": dataset["dataset_id"],
        "mosaic_variants": len(expected_mosaics),
        "neural_evaluations": len(neural),
        "primary_effects": len(primary),
        "gpu_hours": total,
        "authorized_gpu_hours": float(config["budget"]["authorized_max_gpu_hours"]),
        "source_commit": source_commit,
        "verdict": expected_verdict,
        "results_sha256": sha256_file(args.results),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
