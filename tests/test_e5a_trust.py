import numpy as np

from scripts.analyze_e5a_trust_replication import _decision, _interval


def _row(seed, directional, clean=False):
    return {"seed": seed, "directional_boundary": directional, "ci_clean_boundary": clean}


def test_e5a_decision_tree_is_terminal_and_outcome_independent():
    seed0 = _row(0, True, True)
    cases = [
        ([seed0, _row(1, True, True), _row(2, True, True)], "REPLICATED",
         "FULL_E5B_DESIGN_MAY_BE_FROZEN_AFTER_REVIEW"),
        ([seed0, _row(1, True), _row(2, True)], "DIRECTIONALLY_REPLICATED",
         "REDUCED_E5B_ONLY_AFTER_NEW_FREEZE"),
        ([seed0, _row(1, True), _row(2, False)], "HETEROGENEOUS_PARTIAL",
         "FULL_E5B_NOT_AUTHORIZED"),
        ([seed0, _row(1, False), _row(2, False)], "NOT_REPLICATED",
         "FULL_E5B_NOT_AUTHORIZED"),
    ]
    for rows, verdict, gate in cases:
        decision = _decision(rows)
        assert decision["verdict"] == verdict
        assert decision["e5b_gate"] == gate
        assert "beyond 2" in decision["terminal_rule"]


def test_e5a_interval_resamples_clips_as_clusters():
    delta = np.asarray([1.0, 1.0, -1.0, -1.0])
    clips = np.asarray([10, 10, 11, 11])
    observed = _interval(delta, clips, n_boot=1000, seed=3)
    assert observed["mean"] == 0.0
    assert observed["clips"] == 2
    assert observed["records"] == 4
    assert observed["lo"] == -1.0
    assert observed["hi"] == 1.0
