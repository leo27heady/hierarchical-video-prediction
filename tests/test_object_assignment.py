import numpy as np

from hvp_rev.evaluation.object_metrics import Detection, GroundTruthObject, score_objects


def _gt(object_id, y, x, cls):
    return GroundTruthObject(object_id, (y, x), (y - 3, x - 3, y + 3, x + 3), cls)


def _det(y, x, cls):
    return Detection((y, x), (y - 3, x - 3, y + 3, x + 3), cls, 0.99)


def test_permutation_and_crossing_do_not_change_identity():
    gt = [_gt("persistent-a", 20, 45, 3), _gt("persistent-b", 20, 15, 8)]
    # Detection list is deliberately opposite the GT list and opposite the previous left/right order.
    pred = [_det(20, 15, 8), _det(20, 45, 3)]
    result = score_objects(pred, gt, frame_shape=(64, 64))
    assert result["n_matched"] == 2
    assert result["identity_accuracy"] == 1.0
    assert result["trajectory_error"] == 0.0
    assert {x["object_id"] for x in result["assignments"]} == {"persistent-a", "persistent-b"}


def test_miss_and_extra_are_not_removed_as_nan():
    gt = [_gt(1, 12, 12, 1), _gt(2, 45, 45, 2)]
    pred = [_det(12, 12, 1), _det(12, 45, 7)]
    result = score_objects(pred, gt, frame_shape=(64, 64))
    assert result["missed"] == 1
    assert result["extra"] == 1
    assert result["coverage"] == 0.5
    assert result["identity_accuracy"] == 0.5  # one correct, one missed labelled GT object
    assert np.isfinite(result["trajectory_error"])
    assert result["trajectory_error"] > result["matched_trajectory_error"]


def test_no_detections_receive_explicit_penalty_and_zero_coverage():
    result = score_objects([], [_gt("a", 10, 10, 4)], frame_shape=(64, 64))
    assert result["coverage"] == 0.0
    assert result["missed"] == 1
    assert result["extra"] == 0
    assert result["identity_accuracy"] == 0.0
    assert np.isclose(result["trajectory_error"], np.hypot(64, 64))
