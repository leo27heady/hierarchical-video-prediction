import numpy as np

from scripts.analyze_cwvae_black_baseline import black_frame_metrics, summarize_window


def test_black_metrics_are_exact_for_black_and_white_frames():
    gt = np.stack(
        [
            np.zeros((8, 8), dtype=np.float32),
            np.ones((8, 8), dtype=np.float32),
        ]
    )[None]
    ssim, psnr = black_frame_metrics(gt)
    assert ssim.shape == psnr.shape == (1, 2)
    assert ssim[0, 0] == 1.0
    assert psnr[0, 0] == 100.0
    assert psnr[0, 1] == 0.0


def test_window_summary_uses_clip_cluster_dispersion():
    values = np.asarray([[1.0, 3.0], [3.0, 5.0]])
    result = summarize_window(values, 0, 2)
    assert result["records"] == 4
    assert result["mean"] == 3.0
    assert np.isclose(result["clip_mean_sd"], np.sqrt(2.0))
