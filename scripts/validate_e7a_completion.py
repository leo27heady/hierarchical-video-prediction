"""Independent E7a artifact, pairing, floor, access-log, budget and decision validator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--alternate-ae", required=True)
    parser.add_argument("--floor", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--gpu-ledger", required=True)
    parser.add_argument("--access-log", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--dispatch", required=True)
    parser.add_argument("--training-root", default="artifacts/training")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    floor = json.loads(Path(args.floor).read_text(encoding="utf-8"))
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    execution = json.loads(Path(args.execution).read_text(encoding="utf-8-sig"))
    dispatch = json.loads(Path(args.dispatch).read_text(encoding="utf-8"))
    config_hash, manifest_hash = sha256_file(args.config), sha256_file(args.manifest)
    alt_hash = sha256_file(args.alternate_ae)
    original_hash = config["codecs"]["original"]["sha256"]
    if results["config_sha256"] != config_hash or results["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("E7a results are not bound to the frozen config and dataset")
    if results["codec_hashes"] != {"original": original_hash, "alternate": alt_hash}:
        raise ValueError("E7a results codec identities differ from completed codecs")
    if floor.get("status") != "CODEC_FLOOR_PASS" or floor.get("config_sha256") != config_hash:
        raise ValueError("E7a codec floor did not pass the frozen config")
    if floor.get("original_codec_sha256") != original_hash or floor.get("alternate_codec_sha256") != alt_hash:
        raise ValueError("E7a codec floor was measured on different codec artifacts")
    if execution.get("config_sha256") != config_hash or execution.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("E7a execution record is not bound to the frozen config and dataset")
    source_commit = execution.get("source_commit")
    if (
        dispatch.get("status") != "E7A_EVALUATION_DISPATCH_PASS"
        or dispatch.get("config_sha256") != config_hash
        or dispatch.get("dataset_manifest_sha256") != manifest_hash
        or dispatch.get("dataset_id") != manifest["dataset_id"]
        or dispatch.get("source_commit") != source_commit
        or len(dispatch.get("rows", [])) != 14
    ):
        raise ValueError("E7a evaluation dispatch gate is missing or not bound to the execution")
    ae_spec = json.loads((Path(args.alternate_ae).parent / "run_spec.json").read_text(encoding="utf-8"))
    if ae_spec.get("source_commit") != source_commit:
        raise ValueError("alternate codec source commit differs from the execution record")

    expected_arms = {row["arm"] for row in config["predictor_runs"]}
    trained: dict[str, tuple[Path, dict]] = {}
    for spec_path in Path(args.training_root).rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "e7a-alternate-codec-predictors" or spec.get("ae_hash") != alt_hash:
            continue
        arm = spec["arm"]
        if arm in trained:
            raise ValueError(f"duplicate alternate-codec predictor: {arm}")
        if (spec["seed"], spec["optimizer_updates"], spec["batch_size"], spec["precision"],
                spec["metric_version"], spec["evaluation_clips"], spec["rollout_frames"], spec["nfe"]) != (
                    0, 30000, 16, "bf16", "checkpoint-only-v1", 0, 0, 0
                ):
            raise ValueError(f"alternate-codec predictor regime differs: {arm}")
        if spec.get("source_commit") != source_commit:
            raise ValueError(f"alternate predictor source commit differs: {arm}")
        checkpoint = run_dir / "checkpoint.pt"
        if not checkpoint.is_file() or not (run_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"incomplete alternate predictor artifact: {arm}")
        trained[arm] = (checkpoint, spec)
    if set(trained) != expected_arms:
        raise ValueError(f"alternate predictor inventory mismatch: missing={sorted(expected_arms-set(trained))}")

    codec_hashes = {"original": original_hash, "alternate": alt_hash}
    codec_by_hash = {value: key for key, value in codec_hashes.items()}
    dataset_hash = manifest["artifacts"]["clips"]["sha256"]
    evaluations: dict[tuple[str, str], tuple[Path, dict]] = {}
    access_specs = set()
    for spec_path in Path(args.run_root).rglob("run_spec.json"):
        run_dir = spec_path.parent
        if not (run_dir / "COMPLETE").is_file():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if spec.get("study") != "robustness-e7a" or spec.get("dataset_hash") != dataset_hash:
            continue
        codec = codec_by_hash.get(spec.get("ae_hash"))
        if codec is None:
            raise ValueError(f"unexpected codec in E7a evaluation: {spec_path}")
        key = (codec, spec["arm"])
        if key in evaluations:
            raise ValueError(f"duplicate E7a evaluation: {key}")
        if (spec["warmup_frames"], spec["rollout_frames"], spec["evaluation_clips"], spec["nfe"],
                spec["metric_version"], spec["sampler"]["sampler_seed"]) != (
                    48, 152, 100, 16, "complete-v2-thr0.5", 0
                ):
            raise ValueError(f"E7a evaluation geometry differs: {key}")
        if spec.get("source_commit") != source_commit:
            raise ValueError(f"E7a evaluation source commit differs: {key}")
        checkpoint_hash = spec["arm_config"]["checkpoint_sha256"]
        if codec == "alternate":
            expected_checkpoint_hash = sha256_file(trained[spec["arm"]][0])
        else:
            expected_checkpoint_hash = next(
                row["sha256"] for row in config["original_predictor_checkpoints"]
                if row["arm"] == spec["arm"]
            )
        if checkpoint_hash != expected_checkpoint_hash:
            raise ValueError(f"E7a evaluation used wrong predictor checkpoint: {key}")
        metric_path = run_dir / "metrics/rollout_complete.npz"
        for path in (metric_path, run_dir / "manifest.json", run_dir / "evaluation.json"):
            if not path.is_file():
                raise FileNotFoundError(f"missing E7a evaluation artifact: {path}")
        with np.load(metric_path) as values:
            for required in ("clip", "h", "fg", "mse_full", "mse_union", "fg_precision",
                             "missed_components", "extra_components"):
                if required not in values or len(values[required]) != 15200:
                    raise ValueError(f"missing/misaligned E7a metric {required}: {key}")
        evaluations[key] = (metric_path, spec)
        access_specs.add(spec_path.parent.name)
    expected_evaluations = {(codec, arm) for codec in codec_hashes for arm in expected_arms}
    dispatch_by_key = {
        (row["codec"], row["arm"]): row for row in dispatch["rows"]
    }
    if set(dispatch_by_key) != expected_evaluations:
        raise ValueError("E7a evaluation dispatch inventory differs from the frozen matrix")
    if set(evaluations) != expected_evaluations:
        raise ValueError(f"E7a evaluation inventory mismatch: missing={sorted(expected_evaluations-set(evaluations))}")
    for key, (_, spec) in evaluations.items():
        row = dispatch_by_key[key]
        if (
            row["ae_sha256"] != spec["ae_hash"]
            or row["checkpoint_sha256"] != spec["arm_config"]["checkpoint_sha256"]
        ):
            raise ValueError(f"E7a evaluation differs from its dispatch gate: {key}")

    access_rows = jsonl(args.access_log)
    logged_specs = {row.get("run_spec_hash") for row in access_rows}
    if access_specs != logged_specs or len(access_rows) < 14:
        raise ValueError("E7a access log must cover exactly the completed RunSpecs; retries may repeat them")

    ledger = jsonl(args.gpu_ledger)
    if any(row.get("source_commit") != source_commit for row in ledger if row.get("study") == "robustness-e7a"):
        raise ValueError("E7a GPU ledger contains a different source commit")
    total = sum(float(row["gpu_hours"]) for row in ledger if row.get("study") == "robustness-e7a")
    if total > float(config["budget"]["authorized_max_gpu_hours"]):
        raise ValueError(f"E7a exceeded GPU authorization: {total:.4f}")
    successful = [row for row in ledger if row.get("study") == "robustness-e7a" and row.get("exit_code") == 0]
    codec_phase_total = sum(
        float(row["gpu_hours"]) for row in ledger
        if row.get("study") == "robustness-e7a" and row.get("phase") in {"codec-training", "codec-floor"}
    )
    if codec_phase_total > float(config["budget"]["codec_training_max_gpu_hours"]):
        raise ValueError("E7a codec training plus floor exceeded their shared allocation")
    if sum(row.get("phase") == "codec-training" for row in successful) != 1:
        raise ValueError("E7a GPU ledger lacks exactly one successful codec training")
    if sum(row.get("phase") == "codec-floor" for row in successful) != 1:
        raise ValueError("E7a GPU ledger lacks exactly one successful codec floor")
    trained_ledger = {row.get("arm") for row in successful if row.get("phase") == "predictor-training"}
    if trained_ledger != expected_arms:
        raise ValueError("E7a predictor-training ledger inventory differs")
    eval_ledger = {(row.get("codec"), row.get("arm")) for row in successful if row.get("phase") == "evaluation"}
    if eval_ledger != expected_evaluations:
        raise ValueError("E7a evaluation ledger inventory differs")

    primary = results["primary"]
    if len(primary) != 8:
        raise ValueError("E7a primary family must contain exactly eight effects")
    original_rows = [row for row in primary if row["codec"] == "original"]
    alternate_rows = [row for row in primary if row["codec"] == "alternate"]
    controls_positive = all(row["mean"] > 0 for row in original_rows)
    alt_positive = all(row["mean"] > 0 for row in alternate_rows)
    alt_clean = all(row["lo"] > 0 for row in alternate_rows)
    expected_verdict = ("HOLDOUT_CONTROL_FAILURE" if not controls_positive else
                        "CODEC_SENSITIVE" if not alt_positive else
                        "CODEC_ROBUST_CI_CLEAN" if alt_clean else
                        "CODEC_DIRECTIONALLY_ROBUST")
    if results["decision"]["verdict"] != expected_verdict:
        raise ValueError("E7a verdict does not follow the frozen decision rule")
    for key, (path, _) in evaluations.items():
        if results["input_artifacts"].get("::".join(key)) != sha256_file(path):
            raise ValueError(f"E7a analysis input hash mismatch: {key}")

    report = {
        "schema_version": 1,
        "status": "COMPLETION_VALIDATION_PASS",
        "study": "robustness-e7a",
        "dataset_id": manifest["dataset_id"],
        "codec_floor": floor["status"],
        "predictor_trainings": len(trained),
        "evaluations": len(evaluations),
        "primary_effects": len(primary),
        "gpu_hours": total,
        "authorized_gpu_hours": float(config["budget"]["authorized_max_gpu_hours"]),
        "verdict": expected_verdict,
        "dispatch_sha256": sha256_file(args.dispatch),
        "results_sha256": sha256_file(args.results),
    }
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
