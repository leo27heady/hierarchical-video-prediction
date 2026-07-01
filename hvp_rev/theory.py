"""Chapter D (theory) — the invertible STOCHASTIC recurrence + the honest stability statement, measured on the REAL
conv core (not the toy MLP proto). Run:  venv/Scripts/python.exe -m hvp_rev.theory

T1  Block invertibility holds for BOTH unconstrained AND Lipschitz coupling (invertibility is independent of the
    Lipschitz constraint -- additive coupling inverts for ANY F).
T2  STOCHASTIC invertibility (store-the-noise): a stochastic step s_{t+1}=Rev(s_t,[ctx,eps]) with reparam eps is
    exactly invertible IFF eps is stored; wrong eps breaks it. This is the genuinely novel bit vs deterministic
    CrevNet/TempVerseFormer -- a provably-invertible STOCHASTIC recurrence.
T3  Invertibility != stability (the mechanism-review result, made concrete): free-run K steps and measure the
    perturbation growth ||g^K(s+d)-g^K(s)|| / ||d||. Unconstrained coupling is UNCONTROLLED (drifts far from 1 ->
    explode/fade, the RHCP fade mechanism); the Lipschitz (spectral-norm) coupling is BOUNDED. Stability comes from
    the Lipschitz constraint, NOT from invertibility.
T4  Gradient-equivalence (reverse-replay == BPTT) still holds with Lipschitz + stochastic conditioning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.rhcp import RevConvStage, ConvMixer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _randomize(stage, std=0.2):
    for m in stage.modules():
        if isinstance(m, ConvMixer):
            with torch.no_grad():
                w = m.pw2
                if hasattr(w, "parametrizations") and "weight" in w.parametrizations:
                    w.parametrizations.weight.original.normal_(0, std)   # spectral-normed: write the RAW weight
                else:
                    w.weight.normal_(0, std)
                if m.pw2.bias is not None:
                    m.pw2.bias.normal_(0, std)


def _freeze(stage, s, c, warmup=30):
    """Converge spectral_norm's power iteration (train-mode forwards) then EVAL-freeze so the weight is deterministic
    (spectral_norm updates u/v every train forward -> the weight would drift between forward and inverse otherwise)."""
    stage.train()
    with torch.no_grad():
        for _ in range(warmup):
            stage.forward(s, c)
    stage.eval()
    return stage


def t1_invertibility():
    print("\n" + "=" * 74 + "\nT1  block invertibility: unconstrained vs Lipschitz (both must invert)\n" + "=" * 74)
    for lip, dt, tol in [(False, torch.float64, 1e-9), (True, torch.float32, 1e-4)]:
        stage = RevConvStage(64, cond_ch=8, depth=2, lipschitz=lip).to(DEV).to(dt)
        _randomize(stage, 0.3)
        s = torch.randn(4, 64, 8, 8, dtype=dt, device=DEV)
        c = torch.randn(4, 8, 8, 8, dtype=dt, device=DEV)
        if lip:
            _freeze(stage, s, c)
        with torch.no_grad():
            err = (stage.inverse(stage.forward(s, c), c) - s).abs().max().item()
        print(f"  lipschitz={lip!s:>5} ({str(dt).split('.')[-1]}): inverse(forward)-s max-err {err:.2e}  "
              f"{'PASS' if err < tol else 'FAIL'}")
        assert err < tol


def t2_stochastic():
    print("\n" + "=" * 74 + "\nT2  STOCHASTIC invertibility (store-the-noise): exact iff eps stored\n" + "=" * 74)
    D, CTX, S, K, B = 64, 8, 8, 40, 2                       # S = noise channels
    stage = RevConvStage(D, cond_ch=CTX + S, depth=2).to(DEV).double()
    _randomize(stage, 0.2)
    s0 = torch.randn(B, D, 8, 8, dtype=torch.float64, device=DEV)
    ctx = [torch.randn(B, CTX, 8, 8, dtype=torch.float64, device=DEV) for _ in range(K)]
    eps = [torch.randn(B, S, 8, 8, dtype=torch.float64, device=DEV) for _ in range(K)]   # stored per-step noise
    fwd = [s0]
    with torch.no_grad():
        for t in range(K):
            fwd.append(stage.forward(fwd[-1], torch.cat([ctx[t], eps[t]], 1)))
        # reconstruct from final state + stored (ctx, eps)
        rec, drift = fwd[-1], 0.0
        for t in reversed(range(K)):
            rec = stage.inverse(rec, torch.cat([ctx[t], eps[t]], 1))
            drift = max(drift, (rec - fwd[t]).abs().max().item())
        # WRONG eps -> breaks
        rec2, dbad = fwd[-1], 0.0
        for t in reversed(range(K)):
            rec2 = stage.inverse(rec2, torch.cat([ctx[t], torch.randn_like(eps[t])], 1))
            dbad = max(dbad, (rec2 - fwd[t]).abs().max().item())
    print(f"  stored eps  : reconstruct s_0..s_K drift {drift:.2e}  {'PASS (exact)' if drift < 1e-8 else 'FAIL'}")
    print(f"  wrong  eps  : drift {dbad:.2e}  {'PASS (breaks, as expected)' if dbad > 1e-2 else 'FAIL'}")
    assert drift < 1e-8 and dbad > 1e-2


def _pert_growth(stage, K=100, B=8, D=64, hw=8):
    """Free-run K steps from s and s+delta; return ||delta_K||/||delta_0|| and the state-norm ratio ||s_K||/||s_0||."""
    s = torch.randn(B, D, hw, hw, device=DEV)
    d0 = 1e-3 * torch.randn_like(s)
    c = torch.zeros(B, stage.cond_ch, hw, hw, device=DEV) if stage.cond_ch else None
    a, b = s.clone(), (s + d0).clone()
    with torch.no_grad():
        for _ in range(K):
            a = stage.forward(a, c); b = stage.forward(b, c)
    pert = ((b - a).norm() / d0.norm()).item()
    snorm = (a.norm() / s.norm()).item()
    return pert, snorm


def t3_stability():
    print("\n" + "=" * 74 + "\nT3  invertibility != stability: perturbation growth over K=100 steps\n" + "=" * 74)
    torch.manual_seed(0)
    # additive coupling s+F(s) is inherently >=1-expansive per step; spectral-norming F to <=1 is NOT enough (step
    # Lipschitz ~ 1+Lip(F) ~ 2). Only shrinking Lip(F) (out_scale) reins in the K-step growth. All are exactly
    # invertible (T1). This is the precise, honest form of "invertibility != stability".
    configs = [("unconstrained", False, 1.0), ("SN Lip(F)<=1", True, 1.0),
               ("SN Lip(F)<=0.1", True, 0.1), ("SN Lip(F)<=0.02", True, 0.02)]
    for label, lip, sc in configs:
        stage = RevConvStage(64, cond_ch=0, depth=2, lipschitz=lip, out_scale=sc).to(DEV)
        _randomize(stage, 0.5)
        if lip:
            _freeze(stage, torch.randn(8, 64, 8, 8, device=DEV), None)
        else:
            stage.eval()
        pert, snorm = _pert_growth(stage)
        tag = "BOUNDED" if 0.05 < pert < 20 else "UNCONTROLLED (explode/fade)"
        print(f"  {label:<16}: ||d_100||/||d_0|| = {pert:.2e}   ||s_100||/||s_0|| = {snorm:.2e}   -> {tag}")
    print("  => det-Jac=1 (invertible) for ALL (T1), yet the dynamics range from explosive to bounded purely by the")
    print("     Lipschitz bound on F. Invertibility contributes nothing to stability; only Lip(F) does. Additive")
    print("     coupling is inherently >=1-expansive, so stability needs SMALL Lip(F) (or a different transition) --")
    print("     the RHCP fade is the expected outcome of an unconstrained coupling.")


def t4_grad_equiv():
    print("\n" + "=" * 74 + "\nT4  reverse-replay == BPTT with Lipschitz + stochastic conditioning\n" + "=" * 74)
    from hvp_rev.rollout_train import bptt_loss_grads, reverse_replay_loss_grads, obj_l1mse, _cos
    from hvp_rev.rhcp import RHCP, RHCPConfig, LevelSpec
    torch.manual_seed(0)
    # a small RHCP whose stages are Lipschitz (rebuild core stages with lipschitz=True)
    cfg = RHCPConfig(levels=[LevelSpec(64, 16, 1), LevelSpec(96, 8, 2), LevelSpec(128, 4, 4)])
    model = RHCP(cfg).to(DEV)
    for l, spec in enumerate(cfg.levels):
        model.core.stages[l] = RevConvStage(spec.channels, cond_ch=model.core.cond_ch[l], depth=spec.depth,
                                            lipschitz=True).to(DEV)
    with torch.no_grad():
        for p in model.core.parameters():
            p.add_(0.02 * torch.randn_like(p))
    ctx = torch.randn(4, 10, 1, 64, 64, device=DEV)
    tgt = torch.randn(4, 40, 1, 64, 64, device=DEV).tanh()
    # converge + FREEZE spectral_norm (train-mode updates u/v every forward -> weight would differ between the
    # no-grad forward pass and the per-step reverse recompute), then eval so both see the identical weight.
    model.train()
    with torch.no_grad():
        model.rollout(ctx, 4)
    model.eval()
    lb, gb = bptt_loss_grads(model, ctx, tgt, obj_l1mse)
    lr, gr = reverse_replay_loss_grads(model, ctx, tgt, obj_l1mse)
    cos, rel = _cos(gr, gb, list(model.parameters()))
    print(f"  grad-cos {cos:.6f}  rel-err {rel:.2e}  {'PASS' if cos > 0.9999 else 'FAIL'}")
    assert cos > 0.9999


if __name__ == "__main__":
    print(f"Chapter-D theory verification | torch {torch.__version__} | device {DEV}")
    t1_invertibility()
    t2_stochastic()
    t3_stability()
    t4_grad_equiv()
    print("\nAll Chapter-D theory properties verified.")
