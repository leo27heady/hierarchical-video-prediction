import numpy as np

from hvp_rev.analysis import head_package_interaction, paired_effect


def test_paired_effect_bootstraps_clips_and_uses_explicit_sign():
    clips = np.repeat(np.arange(4), 3)
    baseline = np.ones(12)
    candidate = np.full(12, 0.75)
    result = paired_effect(baseline, candidate, clips, n_boot=500, seed=1)
    assert result["mean"] == 0.25
    assert result["lo"] == result["hi"] == 0.25
    assert result["clusters"] == 4


def test_factorial_reports_shift_without_calling_it_equivalence():
    clips = np.repeat(np.arange(5), 2)
    dt = np.full(10, 0.10)
    ds = np.full(10, 0.12)  # temporal advantage +.02 under diffusion
    rt = np.full(10, 0.10)
    rs = np.full(10, 0.09)  # temporal advantage -.01 under regression
    result = head_package_interaction(dt, ds, rt, rs, clips, n_boot=500, seed=2)
    assert np.isclose(result["diffusion_temporal_advantage"], 0.02)
    assert np.isclose(result["regression_temporal_advantage"], -0.01)
    assert np.isclose(result["mean"], -0.03)
