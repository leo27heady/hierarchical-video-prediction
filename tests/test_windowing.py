import torch

from hvp.attention.masks import temporal_causal_mask, windowed_causal_mask
from hvp.config import PredictorConfig, TopologyConfig
from hvp.predictor.cascade import HierarchicalCascade


def test_window_zero_equals_causal():
    a = windowed_causal_mask(5, 4, 4, 0, 0)
    b = temporal_causal_mask(5, 4, 4)
    assert torch.equal(a, b)


def test_self_frame_only_window():
    """t_window=1 -> a query attends only its own frame's cells, never earlier frames."""
    T, H, W = 4, 3, 3
    m = windowed_causal_mask(T, H, W, t_window=1, s_window=0)
    t = torch.arange(T).repeat_interleave(H * W)
    same_frame = t[:, None] == t[None, :]
    assert torch.equal(m, same_frame)                      # exactly the within-frame block-diagonal


def test_spatial_window_limits_neighbourhood_and_no_empty_rows():
    T, H, W = 2, 8, 8
    m = windowed_causal_mask(T, H, W, t_window=1, s_window=4)   # ~5x5 (r=2) local, self-frame
    assert m.diagonal().all()                               # every query attends itself (SDPA-safe)
    assert (m.sum(dim=1) > 0).all()                         # no fully-masked row
    # a far-apart pair in the same frame must be disallowed
    q = (0 * H + 0) * W + 0                                 # cell (t0,h0,w0)
    k = (0 * H + 7) * W + 7                                 # cell (t0,h7,w7) — outside the window
    assert not m[q, k]
    # full-attention reference allows it
    assert windowed_causal_mask(T, H, W, 0, 0)[q, k]


def test_cascade_builds_with_windows_and_runs():
    topo = TopologyConfig()
    pcfg = PredictorConfig(dim=64, n_heads=4, n_layers=1,
                           stage_t_window=(1, 1, 0), stage_s_window=(8, 0, 0))  # bot, mid, top
    cas = HierarchicalCascade(topo, pcfg).eval()
    # bot (finest) must be self-frame-only + spatially windowed; top unrestricted
    assert torch.equal(cas.stages[0].self_mask, windowed_causal_mask(9, 16, 16, 1, 8))
    assert torch.equal(cas.stages[2].self_mask, windowed_causal_mask(3, 1, 1, 0, 0))
    z = [torch.randn(1, c, t, s, s) for c, t, s in
         zip(topo.stage_latent_ch, topo.stage_temporal(), topo.stage_spatial())]
    preds = cas(z)
    assert all(torch.isfinite(p).all() for p in preds)
