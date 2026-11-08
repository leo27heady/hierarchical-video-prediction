import pytest

from hvp_rev.rhvd import rollout_crn_seed


def test_sampler_seed_zero_is_legacy_identical():
    for chunk in (0, 1, 6):
        for step in (0, 1, 37):
            assert rollout_crn_seed(0, chunk, step) == 100000 + 1000 * chunk + step


def test_registered_sampler_streams_are_distinct_and_arm_independent():
    values = {rollout_crn_seed(seed, 3, 17) for seed in (0, 1, 2)}
    assert len(values) == 3
    assert rollout_crn_seed(2, 3, 17) == rollout_crn_seed(2, 3, 17)


def test_negative_rollout_seed_coordinates_fail_closed():
    with pytest.raises(ValueError):
        rollout_crn_seed(-1, 0, 0)
