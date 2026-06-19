import math

import pytest
import torch

from hvp.config import (Config, OptimConfig, PredictorConfig, SchedConfig,
                        TopologyConfig, VAEConfig)
from hvp.objectives import compute_oneshot_loss, compute_rollout_loss, lambda_rollout_at
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.continuous import predictive_loss
from hvp.predictor.rollout import unroll_latent
from hvp.tokenizer import build_tokenizer


def _cfg(**sched):
    topo = TopologyConfig(image_size=32, ch_mult=(1, 2, 2, 2, 4, 4),
                          temporal_downsample=(False, False, True, False, True), tap_after=(0, 2, 4))
    pred = PredictorConfig(dim=32, n_heads=4, n_layers=1, head="gmm", head_components=2,
                           pred_warmup_steps=0, sched=SchedConfig(**sched))
    return Config(topology=topo, predictor=pred, vae=VAEConfig(progressive=False),
                  optim=OptimConfig(batch_size=2))


def _build(cfg):
    torch.manual_seed(0)
    tok = build_tokenizer(cfg)
    cas = HierarchicalCascade(cfg.topology, cfg.predictor)
    x = torch.randn(2, 3, 9, 32, 32).clamp(-1, 1)
    return tok, cas, x


# ---- SchedConfig -----------------------------------------------------------

def test_schedconfig_defaults_are_noop():
    sc = SchedConfig()
    assert sc.rollout_enabled is False and sc.lambda_rollout == 0.0 and sc.rollout_steps == 1


def test_schedconfig_validate_rejects_bad_steps():
    with pytest.raises(ValueError):
        SchedConfig(rollout_steps=0).validate()


def test_lambda_rollout_ramp():
    pred = PredictorConfig(pred_warmup_steps=100,
                           sched=SchedConfig(rollout_enabled=True, lambda_rollout=0.1, rollout_warmup_steps=100))
    assert lambda_rollout_at(50, pred) == 0.0          # before pred warmup ends
    assert lambda_rollout_at(100, pred) == 0.0         # ramp just starting
    assert lambda_rollout_at(150, pred) == pytest.approx(0.05)
    assert lambda_rollout_at(200, pred) == pytest.approx(0.1)
    assert lambda_rollout_at(99999, pred) == pytest.approx(0.1)


# ---- byte-identity when off ------------------------------------------------

def test_rollout_off_is_byte_identical():
    """Default (rollout off) and rollout-on-with-lambda-0 must give the EXACT same total loss."""
    cfg_off = _cfg()
    cfg_zero = _cfg(rollout_enabled=True, lambda_rollout=0.0, rollout_warmup_steps=0)
    tok, cas, x = _build(cfg_off)
    t_off, _, _, _ = compute_oneshot_loss(tok, cas, x, cfg_off, step=500)
    t_zero, m_zero, _, _ = compute_oneshot_loss(tok, cas, x, cfg_zero, step=500)
    assert torch.equal(t_off, t_zero)                  # the no-op term changes nothing
    assert "sf_tf_ar_proxy" in m_zero                  # but the proxy is still logged


def test_rollout_on_changes_the_loss():
    cfg = _cfg(rollout_enabled=True, lambda_rollout=0.5, rollout_warmup_steps=0)
    tok, cas, x = _build(cfg)
    t_on, m, _, _ = compute_oneshot_loss(tok, cas, x, cfg, step=500)
    t_off, _, _, _ = compute_oneshot_loss(tok, cas, x, _cfg(), step=500)
    assert not torch.equal(t_on, t_off)
    assert m["lambda_rollout"] == pytest.approx(0.5)


# ---- the detached-target discipline (no encoder collapse channel) ----------

def test_rollout_term_never_touches_the_encoder():
    cfg = _cfg(rollout_enabled=True, lambda_rollout=1.0, rollout_warmup_steps=0)
    tok, cas, x = _build(cfg)
    z_det = [zi.detach() for zi in tok(x).z]
    roll = compute_rollout_loss(cas, z_det, cfg.predictor)
    roll.backward()
    enc_grads = [p.grad for p in tok.encoder.parameters() if p.grad is not None]
    assert not enc_grads                                # zero encoder gradient from the rollout term
    assert any(p.grad is not None for p in cas.parameters())  # but the cascade IS trained


# ---- unroll_latent grad truncation -----------------------------------------

def test_unroll_latent_detaches_feedback():
    cfg = _cfg()
    tok, cas, x = _build(cfg)
    z0 = [zi.detach().requires_grad_(True) for zi in tok(x).z]
    seq = unroll_latent(cas, z0, n_steps=2, detach_feedback=True)
    assert len(seq) == 2 and all(torch.isfinite(p).all() for p in seq[-1])
    seq[-1][0].sum().backward()
    assert all(z.grad is None for z in z0)              # truncated: no grad back to the start

    z0b = [zi.detach().requires_grad_(True) for zi in tok(x).z]
    seq_b = unroll_latent(cas, z0b, n_steps=2, detach_feedback=False)
    seq_b[-1][0].sum().backward()
    assert any(z.grad is not None for z in z0b)         # un-truncated: grad flows through


# ---- shift correctness -----------------------------------------------------

def test_predictive_loss_shift_targets():
    feat = torch.randn(1, 4, 3, 2, 2)                   # (B,C,T,H,W), T=3
    pred = torch.zeros(1, 3, 2, 2, 4)                   # l2 head, perfect-zero pred
    # make position t exactly equal feat[t+2] so a shift-2 MSE is ~0 but shift-1 is not
    feat[:, :, 0] = 0; feat[:, :, 2] = 0                # frames 0,2 = 0
    p0 = predictive_loss(pred, feat, "l2", shift=2)     # supervises pos0 -> frame2 (both 0) => 0
    p1 = predictive_loss(pred, feat, "l2", shift=1)     # supervises pos0 -> frame1 (nonzero) => >0
    assert float(p0) < 1e-6 < float(p1)
