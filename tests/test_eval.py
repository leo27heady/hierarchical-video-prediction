import pytest
import torch

from hvp.eval.quality import LPIPSMetric, changed_mask, changed_region_metrics, fvd


def _moving_clip(b=2, t=4, h=16, w=16):
    """A clip where a bright square moves (changes) on a static background."""
    base = -torch.ones(b, 3, t, h, w)
    for ti in range(t):
        base[:, :, ti, 4:8, ti:ti + 4] = 1.0  # square slides along W
    return base


def test_changed_mask_static_vs_moving():
    clip = _moving_clip()
    gt_next = clip[:, :, 1:]
    cur = clip[:, :, :-1]
    m = changed_mask(gt_next, cur)
    assert 0.0 < float(m.float().mean()) < 0.5            # some pixels move, most static
    # a static pair -> ~no changed pixels
    static = changed_mask(cur, cur)
    assert float(static.float().mean()) < 1e-6


def test_changed_region_perfect_prediction():
    clip = _moving_clip()
    gt_next, cur = clip[:, :, 1:], clip[:, :, :-1]
    res = changed_region_metrics(gt_next, gt_next, cur)   # perfect pred == gt_next
    assert res["changed_psnr"] > 50
    assert res["changed_ssim"] > 0.99
    assert 0.0 < res["changed_frac"] < 0.5


def test_changed_region_persistence_is_penalized():
    clip = _moving_clip()
    gt_next, cur = clip[:, :, 1:], clip[:, :, :-1]
    persist = changed_region_metrics(cur, gt_next, cur)   # predict current (persistence)
    perfect = changed_region_metrics(gt_next, gt_next, cur)
    assert persist["changed_psnr"] < perfect["changed_psnr"]   # persistence loses on moving pixels


def test_lpips_runs_and_identical_is_small():
    try:
        m = LPIPSMetric()
        clip = _moving_clip(h=32, w=32)
        d_same = m(clip, clip)
        d_diff = m(clip, clip + 0.5 * torch.randn_like(clip))
    except Exception as e:  # network/download issues
        pytest.skip(f"lpips unavailable: {e}")
    assert d_same < 1e-3
    assert d_diff > d_same


def test_fvd_identical_is_near_zero():
    try:
        x = _moving_clip(b=6, t=9, h=32, w=32)
        same = fvd(x, x)
        diff = fvd(x, (x + 0.6 * torch.randn_like(x)).clamp(-1, 1))
    except Exception as e:
        pytest.skip(f"fvd backbone unavailable: {e}")
    assert same < 1.0
    assert diff > same
