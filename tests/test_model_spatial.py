"""CPU-safe unit tests for the spatial DLH.

The previous file executed training and loaded an external dataset during pytest collection.
Long real-data and GPU memory probes now belong in explicit smoke scripts, not the unit suite.
"""
import torch
import torch.nn.functional as F

from hvp_dlh.model_spatial import SpatialDLH, SpatialDLHConfig


def _small_model():
    cfg = SpatialDLHConfig(
        stoch_dim=2,
        det_dim=8,
        temporal_gru=8,
        head_hidden=16,
        batch_size=1,
        seq_len=2,
        device="cpu",
    )
    return cfg, SpatialDLH(cfg)


def test_shapes_and_force_tick():
    torch.manual_seed(0)
    cfg, model = _small_model()
    obs = torch.rand(1, 2, 1, 64, 64)
    features = model._bottom_up(obs)
    assert features[0].shape == (1, 2, cfg.det_dim, 16, 16)
    decoded, _, info = model._step(
        model._init_state(1, "cpu"), [features[n][:, 0] for n in range(cfg.levels)],
        mode="posterior", force_tick=True,
    )
    assert decoded.shape == (1, 1, 64, 64)
    assert all(torch.all(e == 1) for e in info["e"])
    assert torch.isclose(info["K"].mean(), torch.tensor(float(cfg.levels)))


def test_synthetic_loss_has_finite_gradients():
    torch.manual_seed(1)
    _, model = _small_model()
    loss, _ = model.loss(torch.rand(1, 2, 1, 64, 64), beta=0.1, force_tick=True)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for name, p in model.named_parameters()
                 if not name.startswith("pf_head.0") and p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_nested_constraint_dilation_expands_support():
    parent = torch.zeros(1, 1, 16, 16)
    parent[0, 0, 8, 8] = 1
    dilated = F.max_pool2d(parent, 3, stride=1, padding=1)
    assert int(parent.sum()) == 1
    assert int(dilated.sum()) == 9
