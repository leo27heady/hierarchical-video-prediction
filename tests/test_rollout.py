import torch

from hvp.config import PredictorConfig, TopologyConfig
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.rollout import free_run, teacher_forced_at
from hvp.tokenizer import HierTokenizer


def _build():
    topo = TopologyConfig(image_size=32, ch_mult=(1, 2, 2, 2, 4, 4),
                          temporal_downsample=(False, False, True, False, True), tap_after=(0, 2, 4))
    tok = HierTokenizer(topo).eval()
    cas = HierarchicalCascade(topo, PredictorConfig(dim=32, n_heads=4, n_layers=1)).eval()
    return tok, cas, topo


def test_free_run_shapes_both_modes():
    tok, cas, topo = _build()
    seed = torch.randn(2, 3, 9, 32, 32)
    for mode in ("pixel", "latent"):
        out = free_run(tok, cas, seed, n_steps=6, mode=mode)
        assert out.shape == (2, 3, 6, 32, 32), (mode, out.shape)
        assert torch.isfinite(out).all()


def test_step0_matches_teacher_forced():
    """The first rollout step uses the GT seed window, so it must equal TF at horizon 0
    (and both feedback modes must agree on step 0)."""
    tok, cas, topo = _build()
    gt = torch.randn(2, 3, 12, 32, 32)
    seed = gt[:, :, :9]
    px = free_run(tok, cas, seed, n_steps=1, mode="pixel")[:, :, 0]
    lat = free_run(tok, cas, seed, n_steps=1, mode="latent")[:, :, 0]
    tf = teacher_forced_at(tok, cas, gt, seed_len=9, horizon=0)
    assert torch.allclose(px, tf, atol=1e-5)
    assert torch.allclose(lat, tf, atol=1e-5)


def test_rollout_long_horizon_no_nan():
    tok, cas, topo = _build()
    seed = torch.randn(1, 3, 9, 32, 32)
    for mode in ("pixel", "latent"):
        out = free_run(tok, cas, seed, n_steps=20, mode=mode)
        assert torch.isfinite(out).all(), mode


def test_modes_diverge_after_several_steps():
    tok, cas, topo = _build()
    seed = torch.randn(1, 3, 9, 32, 32)
    px = free_run(tok, cas, seed, n_steps=5, mode="pixel")
    lat = free_run(tok, cas, seed, n_steps=5, mode="latent")
    assert not torch.allclose(px[:, :, -1], lat[:, :, -1], atol=1e-4)


def test_gmm_cascade_rollout_point_vs_sampled():
    """A GMM-head cascade rolls out; the point (mode) rollout is deterministic, the sampled
    rollout is stochastic (best-of-N path) — both finite."""
    topo = TopologyConfig(image_size=32, ch_mult=(1, 2, 2, 2, 4, 4),
                          temporal_downsample=(False, False, True, False, True), tap_after=(0, 2, 4))
    cas = HierarchicalCascade(topo, PredictorConfig(dim=32, n_heads=4, n_layers=1,
                                                    head="gmm", head_components=3)).eval()
    tok = HierTokenizer(topo).eval()
    seed = torch.randn(2, 3, 9, 32, 32)
    torch.manual_seed(0); a = free_run(tok, cas, seed, n_steps=4, mode="pixel", sample=False)
    torch.manual_seed(0); b = free_run(tok, cas, seed, n_steps=4, mode="pixel", sample=False)
    s1 = free_run(tok, cas, seed, n_steps=4, mode="pixel", sample=True)
    s2 = free_run(tok, cas, seed, n_steps=4, mode="pixel", sample=True)
    assert a.shape == (2, 3, 4, 32, 32) and torch.isfinite(a).all()
    assert torch.allclose(a, b)                                   # mode rollout deterministic
    assert torch.isfinite(s1).all() and not torch.allclose(s1, s2)  # sampled rollout stochastic
