import pytest

from hvp_rev.comparison_v2 import validate_comparison


def _meta():
    return {
        "dataset_hash": "data",
        "ae_hash": "ae",
        "source_split": "development",
        "metric_version": "complete-v1",
        "warmup_frames": 48,
        "rollout_frames": 152,
        "evaluation_clips": 100,
        "nfe": 16,
        "sampler": "ddim",
        "precision": "bf16",
        "context_tokens": 512,
        "target_tokens": 320,
        "target_slots": {"coarse": 4, "fine": 4},
        "loss_weights": [0, 1, 1],
        "parameters": 12_000_000,
        "train_examples": 480_000,
        "optimizer_updates": 30_000,
        "batch_size": 16,
        "total_nfe": 16 * 38,
    }


def test_comparison_reports_exact_matching_definitions():
    report = validate_comparison(_meta(), _meta())
    report.require()
    assert "context-token-matched" in report.matching_definitions
    assert "target-interface-matched" in report.matching_definitions
    assert "total-nfe-matched" in report.matching_definitions


def test_dataset_mismatch_fails_closed_even_when_shapes_match():
    left, right = _meta(), _meta()
    right["dataset_hash"] = "other"
    report = validate_comparison(left, right)
    assert not report.compatible
    with pytest.raises(ValueError, match="dataset_hash"):
        report.require()
