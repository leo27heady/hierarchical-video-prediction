import torch

from hvp.predictor.continuous import (
    ContinuousStagePredictor,
    predictive_loss,
    variance_explained,
)


def test_predictor_shapes_l2_and_gaussian():
    for head, mult in (("l2", 1), ("gaussian", 2)):
        p = ContinuousStagePredictor(4, 5, 4, 4, dim=32, n_heads=4, n_layers=2, head=head)
        out = p(torch.randn(2, 4, 5, 4, 4))
        assert out.shape == (2, 5, 4, 4, 4 * mult)


def test_predictor_is_causal():
    """Perturbing the last input frame must not change earlier-frame predictions."""
    torch.manual_seed(0)
    p = ContinuousStagePredictor(4, 5, 3, 3, dim=32, n_heads=4, n_layers=2).eval()
    x = torch.randn(2, 4, 5, 3, 3)
    with torch.no_grad():
        a = p(x)
        x2 = x.clone(); x2[:, :, -1] += 5.0
        b = p(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_persistence_ve_is_zero():
    x = torch.randn(3, 4, 5, 4, 4)
    cur = x[:, :, :-1].permute(0, 2, 3, 4, 1)
    # feed persistence as the "prediction" (pred at t == current frame t)
    pred = torch.zeros(3, 5, 4, 4, 4)
    pred[:, :-1] = cur
    ve = variance_explained(pred, x)
    assert abs(ve["ve_all"]) < 1e-4


def test_learns_motion_beats_persistence():
    """On rolling content (each frame = previous rolled by one cell), a motion-aware
    predictor beats persistence on changed cells — validates the whole probe pipeline."""
    torch.manual_seed(0)
    B, C, T, H, W = 64, 4, 6, 4, 4
    base = torch.randn(B, C, 1, H, W)
    frames = [torch.roll(base, shifts=t, dims=4) for t in range(T)]
    feat = torch.cat(frames, dim=2)  # (B,C,T,H,W), content rolls along W
    split = 48
    tr, va = feat[:split], feat[split:]
    p = ContinuousStagePredictor(C, T, H, W, dim=64, n_heads=4, n_layers=3)
    opt = torch.optim.Adam(p.parameters(), lr=3e-3)
    for _ in range(400):
        out = p(tr)
        loss = predictive_loss(out, tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    p.eval()
    with torch.no_grad():
        ve = variance_explained(p(va), va)
    assert ve["ve_changed"] > 0.3, ve
