"""Resolve and validate E7a's exact fourteen evaluation command vectors without opening data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvp_rev.provenance import atomic_write_json, sha256_file


def _completed_alternate_predictors(
    config: dict,
    *,
    alternate_hash: str,
    source_commit: str,
    training_root: Path,
) -> dict[str, Path]:
    expected = {row["arm"] for row in config["predictor_runs"]}
    matches: dict[str, Path] = {}
    for spec_path in training_root.rglob("run_spec.json"):
        run_dir = spec_path.parent
        complete = run_dir / "COMPLETE"
        checkpoint = run_dir / "checkpoint.pt"
        manifest_path = run_dir / "manifest.json"
        if not (complete.is_file() and checkpoint.is_file() and manifest_path.is_file()):
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if (
            spec.get("study") != "e7a-alternate-codec-predictors"
            or spec.get("ae_hash") != alternate_hash
            or spec.get("source_commit") != source_commit
        ):
            continue
        arm = spec.get("arm")
        if arm not in expected:
            continue
        if arm in matches:
            raise ValueError(f"duplicate completed alternate predictor: {arm}")
        if complete.read_text(encoding="utf-8").strip() != run_dir.name:
            raise ValueError(f"alternate predictor COMPLETE marker mismatch: {arm}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_row = next(
            (row for row in manifest.get("artifacts", []) if row.get("path") == "checkpoint.pt"),
            None,
        )
        if (
            checkpoint_row is None
            or checkpoint_row.get("bytes") != checkpoint.stat().st_size
            or checkpoint_row.get("sha256") != sha256_file(checkpoint)
        ):
            raise ValueError(f"alternate predictor checkpoint manifest mismatch: {arm}")
        matches[arm] = checkpoint
    if set(matches) != expected:
        raise ValueError(
            f"alternate predictor inventory mismatch: missing={sorted(expected - set(matches))}"
        )
    return matches


def _option_value(argv: list[str], option: str) -> str:
    indices = [index for index, value in enumerate(argv) if value == option]
    if len(indices) != 1:
        raise ValueError(f"{option} must occur exactly once in E7a argv")
    index = indices[0]
    if index + 1 >= len(argv) or not argv[index + 1] or argv[index + 1].startswith("--"):
        raise ValueError(f"{option} requires a non-empty E7a argv value")
    return argv[index + 1]


def resolve_dispatch(
    *,
    config_path: Path,
    manifest_path: Path,
    alternate_ae: Path,
    execution_path: Path,
    training_root: Path,
    access_log: str,
    run_log: str,
    artifact_root: str = "artifacts/runs",
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8-sig"))
    if config.get("status") != "FROZEN" or config.get("study") != "robustness-e7a":
        raise ValueError("E7a dispatch requires the frozen robustness-e7a config")
    if dataset_manifest.get("purpose") != "codec_robustness_holdout":
        raise ValueError("E7a dispatch received the wrong dataset purpose")
    config_hash = sha256_file(config_path)
    if execution.get("config_sha256") != config_hash:
        raise ValueError("E7a execution record differs from the frozen config")
    if execution.get("dataset_id") != dataset_manifest.get("dataset_id"):
        raise ValueError("E7a execution record differs from the frozen dataset")
    source_commit = execution.get("source_commit")
    if not source_commit:
        raise ValueError("E7a execution record lacks its source commit")

    original_ae = Path(config["codecs"]["original"]["path"])
    if (
        not original_ae.is_file()
        or sha256_file(original_ae) != config["codecs"]["original"]["sha256"]
    ):
        raise ValueError("original E7a codec is missing or hash-mismatched")
    if not alternate_ae.is_file():
        raise FileNotFoundError(alternate_ae)
    alternate_hash = sha256_file(alternate_ae)
    alternate_spec_path = alternate_ae.parent / "run_spec.json"
    if not alternate_spec_path.is_file():
        raise FileNotFoundError(alternate_spec_path)
    alternate_spec = json.loads(alternate_spec_path.read_text(encoding="utf-8"))
    if alternate_spec.get("source_commit") != source_commit:
        raise ValueError("alternate E7a codec source commit differs from the execution record")

    alternate_checkpoints = _completed_alternate_predictors(
        config,
        alternate_hash=alternate_hash,
        source_commit=source_commit,
        training_root=training_root,
    )
    original_checkpoints: dict[str, Path] = {}
    for row in config["original_predictor_checkpoints"]:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"original E7a checkpoint missing/hash-mismatched: {row['arm']}")
        original_checkpoints[row["arm"]] = path

    group = (
        f"{config['evaluation']['wandb_group_prefix']}-"
        f"{dataset_manifest['dataset_id']}"
    )
    test_path = dataset_manifest["artifacts"]["clips"]["path"]
    codecs = [
        (
            "original",
            original_ae,
            config["codecs"]["original"]["sha256"],
            original_checkpoints,
        ),
        ("alternate", alternate_ae, alternate_hash, alternate_checkpoints),
    ]
    rows: list[dict] = []
    for codec_id, ae_path, ae_hash, checkpoints in codecs:
        for run in config["predictor_runs"]:
            arm = run["arm"]
            checkpoint = checkpoints[arm]
            checkpoint_hash = sha256_file(checkpoint)
            expected_hours = float(run["reference_eval_hours_nfe16"])
            max_hours = 1.5 * expected_hours
            argv = [
                "--arm", arm,
                "--seed", str(run["seed"]),
                "--steps", str(run["steps"]),
                "--batch", str(run["batch"]),
                "--bf16",
                "--eval_only",
                "--complete_metrics",
                "--metric_threshold", str(config["evaluation"]["metric_threshold"]),
                "--nfe", str(config["evaluation"]["nfe"]),
                "--sampler_seed", str(config["evaluation"]["sampler_seed"]),
                "--test_path", test_path,
                "--warm", str(config["evaluation"]["warmup_frames"]),
                "--n_val", "250",
                "--n_eval_clips", str(config["evaluation"]["evaluation_clips"]),
                "--ae_path", ae_path.as_posix(),
                "--checkpoint_path", checkpoint.as_posix(),
                "--codec_manifest", manifest_path.as_posix(),
                "--codec_access_log", access_log,
                "--artifact_root", artifact_root,
                "--source_commit", source_commit,
                "--wandb_project", config["evaluation"]["wandb_project"],
                "--wandb_group", group,
                "--wandb_required",
                "--run_log", run_log,
                "--authorized_gpu_hours", format(max_hours, ".12g"),
            ]
            if _option_value(argv, "--ae_path") != ae_path.as_posix():
                raise AssertionError("E7a AE path changed while constructing argv")
            if _option_value(argv, "--checkpoint_path") != checkpoint.as_posix():
                raise AssertionError("E7a checkpoint path changed while constructing argv")
            rows.append(
                {
                    "codec": codec_id,
                    "arm": arm,
                    "ae_path": ae_path.as_posix(),
                    "ae_sha256": ae_hash,
                    "checkpoint_path": checkpoint.as_posix(),
                    "checkpoint_sha256": checkpoint_hash,
                    "expected_gpu_hours": expected_hours,
                    "max_gpu_hours": max_hours,
                    "argv": argv,
                }
            )

    expected_keys = {
        (codec_id, run["arm"])
        for codec_id in ("original", "alternate")
        for run in config["predictor_runs"]
    }
    actual_keys = {(row["codec"], row["arm"]) for row in rows}
    if len(rows) != 14 or actual_keys != expected_keys:
        raise ValueError("E7a dispatch does not contain the exact frozen fourteen-cell matrix")
    for row in rows:
        if _option_value(row["argv"], "--ae_path") != row["ae_path"]:
            raise ValueError(f"E7a dispatch AE argv mismatch: {row['codec']}/{row['arm']}")
        if _option_value(row["argv"], "--checkpoint_path") != row["checkpoint_path"]:
            raise ValueError(
                f"E7a dispatch checkpoint argv mismatch: {row['codec']}/{row['arm']}"
            )

    return {
        "schema_version": 1,
        "status": "E7A_EVALUATION_DISPATCH_PASS",
        # Bind the report deterministically to the original execution. Re-running the CPU
        # resolver before resume must reproduce byte-identical command vectors and attestations.
        "created_utc": execution.get("created_utc"),
        "config_sha256": config_hash,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "dataset_id": dataset_manifest["dataset_id"],
        "source_commit": source_commit,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--alternate-ae", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--training-root", default="artifacts/training")
    parser.add_argument("--access-log", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--artifact-root", default="artifacts/runs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = resolve_dispatch(
        config_path=Path(args.config),
        manifest_path=Path(args.manifest),
        alternate_ae=Path(args.alternate_ae),
        execution_path=Path(args.execution),
        training_root=Path(args.training_root),
        access_log=args.access_log,
        run_log=args.run_log,
        artifact_root=args.artifact_root,
    )
    atomic_write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
