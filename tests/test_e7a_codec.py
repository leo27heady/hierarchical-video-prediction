import argparse
import json
from pathlib import Path

import numpy as np

from hvp_rev.provenance import assert_dataset_purpose, sha256_file
from hvp_rev.rhvd_train_v2 import resolved_spec
from scripts.analyze_e7a_codec_robustness import registered_verdict
from scripts.resolve_e7a_evaluation_dispatch import _portable_path, resolve_dispatch


def test_codec_robustness_purpose_is_explicitly_matchable(tmp_path):
    manifest = tmp_path / "codec.manifest.json"
    manifest.write_text(json.dumps({"purpose": "codec_robustness_holdout"}), encoding="utf-8")
    value = assert_dataset_purpose(
        manifest, allowed={"codec_robustness_holdout"}, operation="E7a evaluation"
    )
    assert value["purpose"] == "codec_robustness_holdout"


def test_train_only_runspec_has_no_evaluation_estimand(tmp_path):
    train = tmp_path / "train.npy"
    test = tmp_path / "test.npy"
    np.save(train, np.zeros((2, 10, 64, 64), dtype=np.uint8))
    np.save(test, np.zeros((2, 20, 64, 64), dtype=np.uint8))
    np.save(tmp_path / "train_actions.npy", np.zeros((2, 10), dtype=np.int8))
    np.save(tmp_path / "test_actions.npy", np.zeros((2, 20), dtype=np.int8))
    ae = tmp_path / "ae.pt"
    ae.write_bytes(b"fixture")
    args = argparse.Namespace(
        train_path=str(train), test_path=str(test), arm="hier-min", study="fixture",
        seed=0, steps=30000, batch=16, bf16=True, nfe=16, warm=48,
        n_eval_clips=100, source_commit="fixture", lr=3e-4, warmup=2000,
        provenance_tag="fixture", ae_path=str(ae), train_only=True,
    )
    spec = resolved_spec(args)
    assert spec.metric_version == "checkpoint-only-v1"
    assert spec.nfe == 0
    assert spec.rollout_frames == 0
    assert spec.evaluation_clips == 0
    assert spec.sampler["rollout"] == "not-run"


def test_frozen_e7a_matrix_and_budget_are_nonadaptive():
    with open("experiments/configs/alternate_codec_robustness.json", encoding="utf-8") as handle:
        config = json.load(handle)
    assert config["status"] == "FROZEN"
    assert len(config["predictor_runs"]) == 7
    assert len(config["contrasts"]) == 4
    components = (
        config["budget"]["codec_training_max_gpu_hours"] +
        sum(row["max_gpu_hours"] for row in config["predictor_runs"]) +
        config["budget"]["evaluation_allocated_gpu_hours"]
    )
    assert components <= config["budget"]["authorized_max_gpu_hours"] == 36.0
    assert "No codec retry" in config["decision"]["no_adaptive_extension"]


def test_e7a_registered_decision_precedence():
    def rows(original, alternate_lows):
        return ([{"codec": "original", "mean": value, "lo": value - 0.1} for value in original] +
                [{"codec": "alternate", "mean": mean, "lo": lo}
                 for mean, lo in alternate_lows])

    assert registered_verdict(rows([-0.01, 1, 1, 1], [(1, 1)] * 4)) == "HOLDOUT_CONTROL_FAILURE"
    assert registered_verdict(rows([1] * 4, [(1, 1), (1, 1), (-0.01, -0.1), (1, 1)])) == "CODEC_SENSITIVE"
    assert registered_verdict(rows([1] * 4, [(1, 0.1)] * 4)) == "CODEC_ROBUST_CI_CLEAN"
    assert registered_verdict(rows([1] * 4, [(1, 0.1), (1, 0), (1, 0.1), (1, 0.1)])) == "CODEC_DIRECTIONALLY_ROBUST"


def test_e7a_exact_fourteen_cell_dispatch_has_nonempty_hash_bound_paths(tmp_path):
    arms = [
        "hier-min", "flat-min", "hier-long", "flat-budget",
        "strided-flat", "hier-c2", "flat-c2",
    ]
    original_ae = tmp_path / "original-ae.pt"
    alternate_dir = tmp_path / "alternate"
    alternate_dir.mkdir()
    alternate_ae = alternate_dir / "checkpoint.pt"
    original_ae.write_bytes(b"original codec")
    alternate_ae.write_bytes(b"alternate codec")
    source_commit = "fixture-source"
    (alternate_dir / "run_spec.json").write_text(
        json.dumps({"source_commit": source_commit}), encoding="utf-8"
    )
    original_rows = []
    training_root = tmp_path / "training"
    for index, arm in enumerate(arms):
        original_checkpoint = tmp_path / f"original-{arm}.pt"
        original_checkpoint.write_bytes(f"original {arm}".encode())
        original_rows.append(
            {
                "arm": arm,
                "path": original_checkpoint.as_posix(),
                "sha256": sha256_file(original_checkpoint),
            }
        )
        run_dir = training_root / arm / "seed-0" / f"run-{index}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "checkpoint.pt"
        checkpoint.write_bytes(f"alternate {arm}".encode())
        (run_dir / "run_spec.json").write_text(
            json.dumps(
                {
                    "study": "e7a-alternate-codec-predictors",
                    "arm": arm,
                    "ae_hash": sha256_file(alternate_ae),
                    "source_commit": source_commit,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "COMPLETE").write_text(run_dir.name + "\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "path": "checkpoint.pt",
                            "bytes": checkpoint.stat().st_size,
                            "sha256": sha256_file(checkpoint),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "dataset.manifest.json"
    execution_path = tmp_path / "execution.json"
    config = {
        "status": "FROZEN",
        "study": "robustness-e7a",
        "codecs": {
            "original": {
                "path": original_ae.as_posix(),
                "sha256": sha256_file(original_ae),
            }
        },
        "predictor_runs": [
            {
                "arm": arm,
                "seed": 0,
                "steps": 30000,
                "batch": 16,
                "reference_eval_hours_nfe16": 0.1 + index / 1000,
            }
            for index, arm in enumerate(arms)
        ],
        "original_predictor_checkpoints": original_rows,
        "evaluation": {
            "wandb_group_prefix": "fixture-e7a",
            "wandb_project": "fixture",
            "metric_threshold": 0.5,
            "nfe": 16,
            "sampler_seed": 0,
            "warmup_frames": 48,
            "evaluation_clips": 100,
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "purpose": "codec_robustness_holdout",
                "dataset_id": "fixture-dataset",
                "artifacts": {"clips": {"path": "locked-do-not-open.npy"}},
            }
        ),
        encoding="utf-8",
    )
    execution_path.write_text(
        json.dumps(
            {
                "config_sha256": sha256_file(config_path),
                "dataset_id": "fixture-dataset",
                "source_commit": source_commit,
            }
        ),
        encoding="utf-8",
    )

    report = resolve_dispatch(
        config_path=config_path,
        manifest_path=manifest_path,
        alternate_ae=alternate_ae,
        execution_path=execution_path,
        training_root=training_root,
        access_log="access.jsonl",
        run_log="run.log",
    )
    assert report["status"] == "E7A_EVALUATION_DISPATCH_PASS"
    assert len(report["rows"]) == 14
    assert {(row["codec"], row["arm"]) for row in report["rows"]} == {
        (codec, arm) for codec in ("original", "alternate") for arm in arms
    }
    for row in report["rows"]:
        assert row["ae_path"] and row["checkpoint_path"]
        assert sha256_file(row["ae_path"]) == row["ae_sha256"]
        assert sha256_file(row["checkpoint_path"]) == row["checkpoint_sha256"]
        ae_index = row["argv"].index("--ae_path")
        checkpoint_index = row["argv"].index("--checkpoint_path")
        assert row["argv"][ae_index + 1] == row["ae_path"]
        assert row["argv"][checkpoint_index + 1] == row["checkpoint_path"]

    controller = Path("scripts/run_e7a_codec_robustness.ps1").read_text(encoding="utf-8")
    assert "--ae_path $codec.ae" not in controller
    assert "hvp_rev.rhvd @evalArgs" in controller


def test_e7a_dispatch_paths_are_repo_relative_when_portable():
    path = Path("artifacts") / "fixture" / "checkpoint.pt"
    assert _portable_path(path) == path.as_posix()
