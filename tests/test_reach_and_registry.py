import numpy as np

from hvp_rev.arm_registry import FIXED_TARGET_CONTROLS, ArmSpec
from hvp_rev.evaluation.reach import ContextGeometry, pair_specific_reach_mask


def test_pair_specific_band_is_derived_from_arm_geometry():
    long = ContextGeometry("hier-min", coarse_slots=48, coarse_stride=1, fine_recent=2)
    short = ContextGeometry("flat-min", fine_recent=14)
    dt = np.asarray([10, 14, 15, 48, 49, np.inf])
    mask = pair_specific_reach_mask(dt, long, short)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_answerability_and_nonempty_are_co_primary_filters():
    long = ContextGeometry("long", coarse_slots=24, coarse_stride=2, fine_recent=2)
    short = ContextGeometry("short", fine_recent=8)
    dt = np.asarray([9, 20, 48])
    mask = pair_specific_reach_mask(
        dt, long, short,
        answerable=np.asarray([False, False, True]),
        foreground_nonempty=np.asarray([True, True, False]),
    )
    assert not mask.any()


def test_fixed_target_control_holds_interface_and_context_budget():
    hier = FIXED_TARGET_CONTROLS["hier-s2-fixed-target"]
    flat = FIXED_TARGET_CONTROLS["flat-short-fixed-target"]
    assert hier.context_tokens == flat.context_tokens == 512
    assert hier.target_slots == flat.target_slots == {"top": 0, "coarse": 4, "fine": 4}
    assert hier.target_tokens == flat.target_tokens
    assert hier.context_reach == 48
    assert flat.context_reach == 8


def test_invalid_stride_is_rejected():
    try:
        ArmSpec("bad", 1, 1, coarse_context_stride=3)
    except ValueError as error:
        assert "divisor" in str(error)
    else:
        raise AssertionError("invalid stride accepted")
