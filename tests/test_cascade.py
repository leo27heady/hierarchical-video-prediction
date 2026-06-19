import torch

from hvp.config import PredictorConfig, TopologyConfig
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.masks import causal_receptive_groups, cross_rf_mask


def test_causal_receptive_groups_matches_user_spec():
    # top(T3) <- mid(T5): t1'->[m0], t2'->[m0,m1,m2], t3'->[m2,m3,m4]
    assert causal_receptive_groups(5) == [[0], [0, 1, 2], [2, 3, 4]]
    # mid(T5) <- bot(T9)
    assert causal_receptive_groups(9) == [[0], [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]]


def test_cross_mask_no_future_leak():
    """Every allowed (child, parent) connection must satisfy max(RF(parent_frame)) <= child_frame."""
    # mid(child T5,4x4) <- top(parent T3,1x1)
    for (ct, ch, cw, pt, ph, pw) in [(5, 4, 4, 3, 1, 1), (9, 16, 16, 5, 4, 4)]:
        m = cross_rf_mask(ct, ch, cw, pt, ph, pw)
        rf_max = [max(g) for g in causal_receptive_groups(ct)]
        nz = m.nonzero(as_tuple=False)
        for ci, pi in nz.tolist():
            tc = ci // (ch * cw)
            tp = pi // (ph * pw)
            assert rf_max[tp] <= tc, (tc, tp, rf_max[tp])


def test_cross_mask_no_empty_rows():
    m = cross_rf_mask(9, 16, 16, 5, 4, 4)
    assert (m.sum(dim=1) > 0).all()


def test_cross_mask_spatial_covering():
    # bot cell (h=5,w=5) at 16x16 -> mid cell (1,1) at 4x4 (ratio 4)
    m = cross_rf_mask(9, 16, 16, 5, 4, 4)
    tc = 4  # a frame where some parent is allowed
    ci = (tc * 16 + 5) * 16 + 5
    allowed_parent_cells = {(pi % (4 * 4)) for pi in m[ci].nonzero(as_tuple=False).flatten().tolist()}
    assert allowed_parent_cells == {1 * 4 + 1}  # only the covering parent cell


def _cascade():
    topo = TopologyConfig()
    pcfg = PredictorConfig(dim=64, n_heads=4, n_layers=2)
    return HierarchicalCascade(topo, pcfg), topo


def _zs(topo, b=2):
    sp, tp = topo.stage_spatial(), topo.stage_temporal()
    return [torch.randn(b, topo.stage_latent_ch[s], tp[s], sp[s], sp[s]) for s in range(topo.n_stages())]


def test_cascade_shapes():
    cas, topo = _cascade()
    z = _zs(topo)
    preds = cas(z)
    sp, tp = topo.stage_spatial(), topo.stage_temporal()
    for s in range(topo.n_stages()):
        assert preds[s].shape == (2, tp[s], sp[s], sp[s], topo.stage_latent_ch[s])


def test_conditioning_flows_down_not_up():
    """Perturbing TOP changes BOT/MID predictions; perturbing BOT leaves TOP/MID unchanged."""
    torch.manual_seed(0)
    cas, topo = _cascade()
    cas.eval()
    z = _zs(topo)
    with torch.no_grad():
        base = cas(z)
        z_top = [t.clone() for t in z]; z_top[2] += 3.0
        out_top = cas(z_top)
        z_bot = [t.clone() for t in z]; z_bot[0] += 3.0
        out_bot = cas(z_bot)
    # top perturbation propagates down to bot (cascade conditioning)
    assert not torch.allclose(out_top[0], base[0], atol=1e-4)
    # bot perturbation does NOT propagate up to top or mid
    assert torch.allclose(out_bot[2], base[2], atol=1e-5)
    assert torch.allclose(out_bot[1], base[1], atol=1e-5)


def test_cascade_off_ignores_parent():
    cas, topo = _cascade()
    cas.eval()
    z = _zs(topo)
    with torch.no_grad():
        z_top = [t.clone() for t in z]; z_top[2] += 3.0
        a = cas(z, use_cascade=False)
        b = cas(z_top, use_cascade=False)
    # with cascade off, top perturbation must not affect bot
    assert torch.allclose(a[0], b[0], atol=1e-5)
