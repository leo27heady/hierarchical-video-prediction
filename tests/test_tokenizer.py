import torch

from hvp.config import TopologyConfig
from hvp.tokenizer import HierTokenizer


def _topo():
    # small 32x32 topology for fast CPU tests (architecture is resolution-agnostic)
    return TopologyConfig(
        image_size=32, ch_mult=(1, 2, 2, 2, 4, 4),
        temporal_downsample=(False, False, True, False, True), tap_after=(0, 2, 4),
    )


def _x(b=2):
    return torch.randn(b, 3, 9, 32, 32)


def test_encoder_latent_shapes():
    topo = _topo()
    enc = HierTokenizer(topo).encoder
    z = enc(_x())
    spatial, temporal = topo.stage_spatial(), topo.stage_temporal()
    for s, (lc, sp, tt) in enumerate(zip(topo.stage_latent_ch, spatial, temporal)):
        assert z[s].shape == (2, lc, tt, sp, sp), (s, z[s].shape)


def test_decoder_recon_shapes_and_frame_count_roundtrip():
    topo = _topo()
    tok = HierTokenizer(topo)
    out = tok(_x(), max_depth=3)
    assert len(out.recons) == 3
    for r in out.recons:
        # every depth renders to full resolution with the original frame count (9 -> ... -> 9)
        assert r.shape == (2, 3, 9, 32, 32), r.shape
    assert out.recon.shape == (2, 3, 9, 32, 32)


def test_encoder_bot_stage_is_causal():
    """Perturbing only the LAST input frame must not change earlier bot-latent frames."""
    topo = _topo()
    enc = HierTokenizer(topo).encoder.eval()
    x = _x()
    with torch.no_grad():
        z = enc(x)
        x2 = x.clone()
        x2[:, :, -1] += 5.0
        z2 = enc(x2)
    bot = z[0]  # (B, C, 9, 16, 16), 1:1 temporal with input
    assert torch.allclose(bot[:, :, :8], z2[0][:, :, :8], atol=1e-5)
    assert not torch.allclose(bot[:, :, 8], z2[0][:, :, 8], atol=1e-4)


def test_default_64_config_roundtrip():
    """The 64x64 default topology builds and renders at full resolution / frame count."""
    topo = TopologyConfig()  # default = 64x64
    assert topo.stage_spatial() == (16, 4, 1)
    assert topo.stage_temporal() == (9, 5, 3)
    tok = HierTokenizer(topo).eval()
    with torch.no_grad():
        out = tok(torch.randn(1, 3, 9, 64, 64), max_depth=3)
    assert out.recon.shape == (1, 3, 9, 64, 64)


def test_overfit_one_batch_decreases_recon():
    """G-recon smoke: the autoencoder can reduce recon MSE on a fixed batch."""
    torch.manual_seed(0)
    topo = _topo()
    tok = HierTokenizer(topo)
    x = torch.randn(2, 3, 9, 32, 32) * 0.3
    opt = torch.optim.Adam(tok.parameters(), lr=2e-3)
    first = last = None
    for step in range(40):
        out = tok(x, max_depth=3)
        loss = sum(torch.nn.functional.mse_loss(r, x) for r in out.recons) / len(out.recons)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < 0.6 * first, (first, last)
