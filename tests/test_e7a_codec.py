import argparse
import json

import numpy as np

from hvp_rev.provenance import assert_dataset_purpose
from hvp_rev.rhvd_train_v2 import resolved_spec
from scripts.analyze_e7a_codec_robustness import registered_verdict


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
