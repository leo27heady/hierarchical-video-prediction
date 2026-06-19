import torch

from hvp.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from hvp.config import Config, DataConfig, PredictorConfig, TopologyConfig
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import HierTokenizer


def _small_cfg():
    topo = TopologyConfig(image_size=32, ch_mult=(1, 2, 2, 2, 4, 4),
                          temporal_downsample=(False, False, True, False, True), tap_after=(0, 2, 4))
    return Config(topology=topo, predictor=PredictorConfig(dim=32, n_heads=4, n_layers=1),
                  data=DataConfig(image_size=32))


def _build(cfg):
    tok = HierTokenizer(cfg.topology)
    cas = HierarchicalCascade(cfg.topology, cfg.predictor)
    opt = torch.optim.AdamW(list(tok.parameters()) + list(cas.parameters()), lr=1e-3)
    return tok, cas, opt


def test_seed_everything_reproducible():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_checkpoint_roundtrip_restores_everything(tmp_path):
    cfg = _small_cfg()
    tok, cas, opt = _build(cfg)
    x = torch.randn(1, 3, 9, 32, 32)
    for _ in range(2):  # populate params + optimizer state
        out = tok(x, max_depth=3)
        preds = cas([zi.detach() for zi in out.z])
        loss = sum(p.pow(2).mean() for p in preds) + out.recon.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    path = save_checkpoint(tmp_path / "ckpt.pt", tok=tok, cascade=cas, opt=opt, step=7, cfg=cfg)

    tok2, cas2, opt2 = _build(cfg)
    # before load: params differ
    assert not torch.equal(next(tok.parameters()), next(tok2.parameters()))
    step = load_checkpoint(path, tok=tok2, cascade=cas2, opt=opt2)
    assert step == 7
    for a, b in zip(tok.parameters(), tok2.parameters()):
        assert torch.equal(a, b)
    for a, b in zip(cas.parameters(), cas2.parameters()):
        assert torch.equal(a, b)
    assert len(opt2.state) > 0  # optimizer momentum restored


def test_backward_compatible_keys(tmp_path):
    """eval/probe scripts read blob['model'|'cascade'|'topo'] — keep them present."""
    cfg = _small_cfg()
    tok, cas, opt = _build(cfg)
    path = save_checkpoint(tmp_path / "c.pt", tok=tok, cascade=cas, opt=opt, step=0, cfg=cfg)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model", "cascade", "topo", "optimizer", "step", "rng", "config"} <= set(blob)
