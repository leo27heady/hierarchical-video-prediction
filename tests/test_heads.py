import math

import torch

from hvp.predictor.heads import head_mean, head_mode, head_nll, head_out_dim, head_sample


def test_out_dims():
    assert head_out_dim("l2", 8) == 8
    assert head_out_dim("gaussian", 8) == 16
    assert head_out_dim("gmm", 8, 4) == 4 * (2 * 8 + 1)


def test_shapes_and_finite():
    B, T, H, W, C, K = 2, 3, 4, 4, 5, 3
    pred = torch.randn(B, T, H, W, head_out_dim("gmm", C, K))
    for fn in (head_mean, head_mode, head_sample):
        out = fn(pred, "gmm", C, K)
        assert out.shape == (B, T, H, W, C)
        assert torch.isfinite(out).all()
    tgt = torch.randn(B, T, H, W, C)
    assert torch.isfinite(head_nll(pred, tgt, "gmm", C, K))


def test_nll_drops_as_mean_approaches_target():
    C, K = 2, 2
    tgt = torch.zeros(1, C)
    far = torch.zeros(1, head_out_dim("gmm", C, K)); far[:, K:].fill_(5.0)   # means far from 0
    near = torch.zeros(1, head_out_dim("gmm", C, K))                          # means at 0 = target
    assert head_nll(near, tgt, "gmm", C, K) < head_nll(far, tgt, "gmm", C, K)


def _fit(C, K, tgt, steps=1500):
    out = head_out_dim("gmm", C, K)
    p = torch.nn.Parameter(1.0 * torch.randn(1, out))      # wide init breaks component symmetry
    opt = torch.optim.Adam([p], lr=0.05)
    loss = None
    for _ in range(steps):
        opt.zero_grad()
        loss = head_nll(p.expand(tgt.shape[0], out), tgt, "gmm", C, K)
        loss.backward(); opt.step()
    return p.detach(), float(loss.detach())


def test_gmm_beats_single_gaussian_on_bimodal_target():
    """The whole point: on a genuinely bimodal future a 2-component mixture models both modes;
    a single component (K=1) must straddle them (worse NLL) and its mean lands in the middle."""
    torch.manual_seed(0)
    C, K = 1, 2
    n = 2000
    tgt = torch.where(torch.rand(n, 1) < 0.5, torch.full((n, 1), 1.0), torch.full((n, 1), -1.0))  # ±1

    pg, nll_g = _fit(C, 1, tgt)        # single component (== a single Gaussian)
    pm, nll_m = _fit(C, K, tgt)        # 2-component mixture, same NLL convention
    assert nll_m < nll_g - 0.2, (nll_m, nll_g)             # mixture fits the two modes better

    # best-of-2 sampling beats the mixture mean (which sits ~0, between the modes)
    predN = pm.expand(n, head_out_dim("gmm", C, K))
    mean_pred = head_mean(predN, "gmm", C, K)
    assert mean_pred.abs().mean() < 0.3                    # mean collapses to the empty middle
    s1 = head_sample(predN, "gmm", C, K)
    s2 = head_sample(predN, "gmm", C, K)
    best = torch.minimum((s1 - tgt).abs(), (s2 - tgt).abs()).mean()
    mean_err = (mean_pred - tgt).abs().mean()
    assert best < mean_err                                  # best-of-N >> the blurry mean
