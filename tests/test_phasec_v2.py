import numpy as np
import torch

from hvp_rev.evaluation import Detection, GroundTruthObject, score_objects
from hvp_rev.phasec_v2 import classify_components, extract_components


class _AlwaysThree(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x), 10, device=x.device)
        logits[:, 3] = 12
        return logits


def _gt(identifier, y, x, cls=3):
    return GroundTruthObject(identifier, (y, x), (y - 5, x - 5, y + 5, x + 5), cls)


def _det(y, x, cls=3):
    return Detection((y, x), (y - 5, x - 5, y + 5, x + 5), cls, 0.99)


def test_overlap_then_separation_keeps_persistent_intention_population():
    overlap = score_objects(
        [_det(30, 30)], [_gt("a", 28, 28), _gt("b", 32, 32)],
        frame_shape=(64, 64),
    )
    separated = score_objects(
        [_det(20, 15), _det(20, 45)], [_gt("a", 20, 15), _gt("b", 20, 45)],
        frame_shape=(64, 64),
    )
    assert overlap["n_gt"] == separated["n_gt"] == 2
    assert overlap["missed"] == 1
    assert overlap["identity_accuracy"] == 0.5
    assert separated["coverage"] == 1.0


def test_wrong_location_is_miss_plus_extra_and_wrong_class_is_identity_error():
    wrong_location = score_objects(
        [_det(58, 58)], [_gt("object", 6, 6)], frame_shape=(64, 64),
        max_assignment_cost=0.2,
    )
    assert wrong_location["missed"] == 1 and wrong_location["extra"] == 1
    wrong_class = score_objects(
        [_det(20, 20, cls=7)], [_gt("object", 20, 20, cls=3)], frame_shape=(64, 64),
    )
    assert wrong_class["coverage"] == 1.0
    assert wrong_class["identity_accuracy"] == 0.0


def test_assignment_tie_is_repeatable_and_does_not_drop_records():
    gt = [_gt("a", 20, 20), _gt("b", 20, 40)]
    detections = [_det(20, 30), _det(20, 30)]
    first = score_objects(detections, gt, frame_shape=(64, 64))
    second = score_objects(detections, gt, frame_shape=(64, 64))
    assert first["assignments"] == second["assignments"]
    assert first["n_gt"] == 2 and first["n_pred"] == 2


def test_component_extraction_keeps_extras_and_empty_prediction_explicit():
    frame = np.zeros((64, 64), np.float32)
    frame[5:15, 5:15] = 0.9
    frame[40:52, 42:54] = 0.8
    components = extract_components(frame, threshold=0.3, min_component_pixels=30)
    assert len(components) == 2
    detections = classify_components(
        _AlwaysThree(), components, device="cpu", confidence_threshold=0.0,
    )
    result = score_objects(detections, [_gt("a", 10, 10)], frame_shape=(64, 64))
    assert result["n_pred"] == 2 and result["extra"] == 1
    empty = score_objects([], [_gt("a", 10, 10)], frame_shape=(64, 64))
    assert empty["coverage"] == 0.0
    assert np.isfinite(empty["trajectory_error"])


def test_abstention_is_an_intention_to_score_identity_error():
    frame = np.zeros((64, 64), np.float32)
    frame[20:40, 20:40] = 1
    components = extract_components(frame, threshold=0.3, min_component_pixels=30)
    detections = classify_components(
        _AlwaysThree(), components, device="cpu", confidence_threshold=1.1,
    )
    result = score_objects(detections, [_gt("a", 30, 30)], frame_shape=(64, 64))
    assert result["coverage"] == 1.0
    assert result["identity_coverage"] == 0.0
    assert result["identity_accuracy"] == 0.0
