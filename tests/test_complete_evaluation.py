import numpy as np

from hvp_rev.evaluation.frame_metrics import evaluate_frames


def test_empty_foreground_is_not_full_frame_fallback():
    gt = np.zeros((2, 1, 16, 16), np.float32)
    pred = gt.copy()
    pred[0, 0, 3:7, 3:7] = 1.0
    metrics = evaluate_frames(pred, gt, threshold=0.5)
    assert metrics["fg_nonempty"].tolist() == [False, False]
    assert np.isnan(metrics["mse_gt_fg_nonempty"]).all()
    assert metrics["mse_full"][0] > 0
    assert metrics["mse_gt_bg"][0] > 0
    assert metrics["mse_union"][0] == 1.0
    assert metrics["extra_components"].tolist() == [1, 0]


def test_misses_and_extras_are_both_visible():
    gt = np.zeros((1, 1, 20, 20), np.float32)
    pred = np.zeros_like(gt)
    gt[0, 0, 2:7, 2:7] = 1
    gt[0, 0, 12:17, 12:17] = 1
    pred[0, 0, 2:7, 2:7] = 1
    pred[0, 0, 2:7, 13:18] = 1
    metrics = evaluate_frames(pred, gt, threshold=0.5)
    assert metrics["gt_component_count"][0] == 2
    assert metrics["pred_component_count"][0] == 2
    assert metrics["missed_components"][0] == 1
    assert metrics["extra_components"][0] == 1
    assert metrics["fg_precision"][0] < 1
    assert metrics["fg_recall"][0] < 1


def test_metric_is_record_level():
    gt = np.zeros((2, 3, 1, 8, 8), np.float32)
    pred = gt.copy()
    gt[:, :, :, 2:6, 2:6] = 1
    pred[1, 2] = 1 - gt[1, 2]
    metrics = evaluate_frames(pred, gt, threshold=0.5)
    assert metrics["record_shape"].tolist() == [2, 3]
    assert metrics["mse_full"].shape == (6,)
    assert metrics["mse_full"][-1] > metrics["mse_full"][:-1].max()
    assert metrics["fg_laplacian_energy_nonempty"].shape == (6,)
