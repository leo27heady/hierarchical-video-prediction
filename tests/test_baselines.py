import torch

from hvp.eval.baselines import _rotate, linear_extrap, motion_baselines, rotation_extrap
from hvp.eval.quality import changed_region_metrics


def _rotating_clip(b=2, t=12, h=32, w=32, theta=0.15):
    """A bright off-centre patch rigidly rotating about the centre at constant angular
    velocity — the exact regime the rotation baseline assumes."""
    f0 = -torch.ones(b, 3, h, w)
    f0[:, :, 6:12, 18:26] = 1.0                       # off-centre square so rotation moves it
    frames = [_rotate(f0, f0.new_full((b,), k * theta)) for k in range(t)]
    return torch.stack(frames, dim=2)                 # (b,3,t,h,w)


def test_baseline_shapes():
    seed = torch.rand(2, 3, 9, 32, 32) * 2 - 1            # realistic [-1, 1] input
    out = motion_baselines(seed, n=6)
    for k in ("persist", "linear", "rotation"):
        assert out[k].shape == (2, 3, 6, 32, 32), (k, out[k].shape)
        assert torch.isfinite(out[k]).all()
        assert out[k].min() >= -1.0 - 1e-5 and out[k].max() <= 1.0 + 1e-5


def test_rotation_baseline_beats_persistence_on_moving_pixels():
    clip = _rotating_clip(theta=0.15)
    seed_len, n = 9, 3
    seed = clip[:, :, :seed_len]
    gt = clip[:, :, seed_len:seed_len + n]
    prev = clip[:, :, seed_len - 1: seed_len - 1 + n]

    rot = rotation_extrap(seed, n)
    persist = seed[:, :, -1:].repeat(1, 1, n, 1, 1)

    rot_chg = changed_region_metrics(rot, gt, prev)["changed_psnr"]
    persist_chg = changed_region_metrics(persist, gt, prev)["changed_psnr"]
    # on a cleanly rotating clip the constant-angular-velocity null should clearly win
    assert rot_chg > persist_chg + 2.0, (rot_chg, persist_chg)


def test_linear_extrap_is_constant_velocity():
    seed = torch.zeros(1, 3, 9, 8, 8)
    seed[:, :, -2] = 0.2
    seed[:, :, -1] = 0.4                               # velocity = +0.2
    out = linear_extrap(seed, n=2)
    assert torch.allclose(out[:, :, 0], torch.full_like(out[:, :, 0], 0.6))
    assert torch.allclose(out[:, :, 1], torch.full_like(out[:, :, 1], 0.8))
