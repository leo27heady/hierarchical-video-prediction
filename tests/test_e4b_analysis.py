import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_e4b_factorial import _aligned, training_seed_summary


def _run(with_labels=True):
    run = {
        "clip": np.array([0, 0, 1, 1]),
        "h": np.array([1, 2, 1, 2]),
        "hard": np.array([0.0, 1.0, 0.0, 1.0]),
        "fg": np.array([0.1, 0.2, 0.3, 0.4]),
    }
    if with_labels:
        run.update({"dt": np.array([1, 2, 1, 2]), "seen": np.ones(4)})
    return run


def test_training_seed_summary_uses_five_seed_t_interval():
    result = training_seed_summary([0.01, 0.02, 0.03, 0.04, 0.05])
    assert result["mean"] == pytest.approx(0.03)
    assert result["sample_sd"] == pytest.approx(np.std([0.01, 0.02, 0.03, 0.04, 0.05], ddof=1))
    assert result["df"] == 4
    assert result["lo"] < 0.03 < result["hi"]


def test_training_seed_summary_refuses_posthoc_partial_seed_sets():
    with pytest.raises(ValueError, match="five finite effects"):
        training_seed_summary([0.01, 0.02, 0.03, 0.04])


def test_e4b_alignment_accepts_legacy_optional_label_absence():
    _aligned(_run(), _run(False), _run(), _run(False))


def test_e4b_frozen_config_is_exact_terminal_replacement_design():
    config = json.loads(
        Path("experiments/configs/regression_factorial_seeds3_4.json").read_text(encoding="utf-8")
    )
    runs = config["new_training_runs"]
    expected = {
        (arm, seed)
        for seed in (3, 4)
        for arm in ("hier-long", "strided-flat", "reg-long", "strided-flat-reg")
    }
    assert {(run["arm"], run["seed"]) for run in runs} == expected
    assert len(runs) == 8
    assert sum(run["max_gpu_hours"] for run in runs) == 36.0
    assert config["budget"]["protected_reserve_allowed"] is False
    assert "E8c" in config["budget"]["replaces"]
    assert "no seeds 5+" in config["training_seed_analysis"]["terminal_rule"]
