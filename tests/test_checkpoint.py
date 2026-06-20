from dataclasses import replace

import pytest
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


# --------------------------------------------------------------------------- #
# Periodic / durable step-stamped checkpointing (symptom S2).
# A run that never reaches max_steps must still leave a recoverable stepNNNNNN.pt,
# not just last.pt (which the next run overwrites).
# --------------------------------------------------------------------------- #
def _cfg_with_run(tmp_path, **run_kw):
    cfg = _small_cfg()
    return replace(cfg, run=replace(cfg.run, out_dir=str(tmp_path), **run_kw))


class _Interrupted(Exception):
    """Stands in for a Ctrl-C / SLURM-kill that aborts a run before max_steps."""


def _fake_loaders(topo):
    """A one-clip in-memory loader; itertools.cycle in the trainer replays it forever."""
    x = torch.randn(1, topo.image_c, topo.sequence_length, topo.image_size, topo.image_size)
    return lambda cfg: ([{"video": x}], [])


def _fake_loss(interrupt_after):
    """Cheap finite loss (the save logic, not the objective, is under test); raises after
    `interrupt_after` calls to simulate a kill mid-run."""
    calls = {"n": 0}

    def fake(tok, cascade, x, cfg, step):
        calls["n"] += 1
        if calls["n"] > interrupt_after:
            raise _Interrupted()
        total = torch.zeros((), requires_grad=True) + 1.0
        m = {"loss": 1.0, "recon": 0.0, "psnr_full": 0.0, "active_depth": 0, "lambda": 0.0}
        return total, m, None, None

    return fake


def test_periodic_save_is_stamped_and_survives_interrupt(tmp_path, monkeypatch):
    """save_every>0 must emit step-stamped copies *during* the loop, so a run killed before
    max_steps still leaves a recoverable ckpt. Regression: the periodic save used to write
    only last.pt, so an interrupted under-progressed run kept zero durable checkpoints."""
    from hvp import train as T

    cfg = _cfg_with_run(tmp_path, save_every=2, max_checkpoints_keep=5)
    monkeypatch.setattr(T, "_loaders", _fake_loaders(cfg.topology))
    monkeypatch.setattr(T, "compute_oneshot_loss", _fake_loss(interrupt_after=3))

    with pytest.raises(_Interrupted):  # killed at step 3, long before max_steps=50
        T.fit_oneshot(cfg, max_steps=50)

    out = tmp_path / "oneshot"
    stamped = sorted(p.name for p in out.glob("step*.pt"))
    assert stamped, "interrupted run before max_steps left NO stepNNNNNN.pt (only last.pt)"
    assert "step000002.pt" in stamped  # the periodic (not final) save is itself stamped
    assert (out / "last.pt").exists()


def test_wallclock_timer_save_independent_of_save_every(tmp_path, monkeypatch):
    """save_every_min checkpoints on wall-clock time even with save_every=0, so a run too slow
    to cross a step boundary still leaves a durable copy."""
    from hvp import train as T

    class _FakeClock:  # advances 120 s per reading -> any positive interval is due each step
        def __init__(self):
            self.t = 1000.0

        def time(self):
            self.t += 120.0
            return self.t

    cfg = _cfg_with_run(tmp_path, save_every=0, save_every_min=1)  # 60 s interval
    monkeypatch.setattr(T, "_loaders", _fake_loaders(cfg.topology))
    monkeypatch.setattr(T, "compute_oneshot_loss", _fake_loss(interrupt_after=3))
    monkeypatch.setattr(T, "time", _FakeClock())

    with pytest.raises(_Interrupted):
        T.fit_oneshot(cfg, max_steps=50)

    out = tmp_path / "oneshot"
    stamped = sorted(p.name for p in out.glob("step*.pt"))
    assert stamped, "wall-clock timer left no stepNNNNNN.pt with save_every=0"
    assert "step000001.pt" in stamped  # timer fired on the first post-init step


def test_max_checkpoints_keep_prunes_old_stamped(tmp_path):
    """max_checkpoints_keep rotates stamped copies (each ~265 MB) so disk doesn't fill;
    last.pt is the always-present rolling pointer and is never pruned."""
    from hvp.train import _save

    cfg = _cfg_with_run(tmp_path, max_checkpoints_keep=2)
    tok, cas, opt = _build(cfg)
    for s in (0, 1, 2, 3):
        _save(tok, cas, opt, s, cfg, stamped=True)

    out = tmp_path / "oneshot"
    stamped = sorted(p.name for p in out.glob("step*.pt"))
    assert stamped == ["step000002.pt", "step000003.pt"]  # only the 2 newest survive
    assert (out / "last.pt").exists()
