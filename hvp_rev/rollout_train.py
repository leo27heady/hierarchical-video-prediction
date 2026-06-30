"""Free-running rollout training for RHCP with constant-memory reverse-replay backprop (P0 fixes 1-3).

The make-or-break trainer: roll K steps free-running (the model conditions on its OWN predictions), and backprop the
WHOLE chain by reverse-replay -- reconstruct each past state via core.inverse_step (O(1)-in-K activation memory, with
O(sqrt K) re-anchors to bound fp32 drift) -- so a long unroll fits where vanilla BPTT OOMs. Both the short (K=10) and
long (K=40) Panel-C arms route through THIS step, so reversibility is a fixed enabler, not a varied factor.

Numerics policy (red-team locked): rollout trains in fp32 (no AMP); inverse runs in fp32; re-anchor every floor(sqrt K).
Gradients are exact vs autograd-BPTT up to fp reconstruction drift -- gate: grad-cos > 0.9999 at K=40 on the REAL RHCP.

  venv/Scripts/python.exe -m hvp_rev.rollout_train --gate           # gradient-equivalence gate (training-free)
"""
from __future__ import annotations

import argparse
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.rhcp import RHCP, RHCPConfig, LevelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- objectives
_LPIPS = None


def _lpips_fn(device=DEV):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    return _LPIPS


def obj_l1mse(y, tgt):
    return F.l1_loss(y, tgt) + F.mse_loss(y, tgt)


def obj_l1lpips(y, tgt, lam=0.1):
    """L1 + lam*LPIPS. y,tgt in [-1,1], (B,1,H,W). Distribution-matching-ish: avoids the mean-fade of pure MSE (E1)."""
    l1 = F.l1_loss(y, tgt)
    lp = _lpips_fn(y.device)(y.repeat(1, 3, 1, 1), tgt.repeat(1, 3, 1, 1)).mean()
    return l1 + lam * lp


OBJECTIVES = {"l1mse": obj_l1mse, "l1lpips": obj_l1lpips}


# ----------------------------------------------------------------------------- the two backward paths
def bptt_loss_grads(model: RHCP, context, targets, objective: Callable):
    """Reference: full-graph BPTT. Activation memory O(K). Returns (loss, {param: grad})."""
    K = targets.shape[1]
    s = model.init_states(context)
    loss = 0.0
    for t in range(K):
        s = model.core.step(s, t)
        loss = loss + objective(model.decoder(s[0]), targets[:, t]) / K   # MEAN over frames: K-invariant grad scale
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return float(loss.detach()), {id(p): (g if g is not None else torch.zeros_like(p)) for p, g in zip(params, grads)}


def reverse_replay_loss_grads(model: RHCP, context, targets, objective: Callable,
                              anchor_every: Optional[int] = None, accumulate_into_grad: bool = False):
    """Constant-memory reverse-replay. Reconstruct states via inverse_step, recompute one step at a time.
    Returns (loss, {param: grad}). If accumulate_into_grad, also adds grads into p.grad (for the training loop)."""
    K = targets.shape[1]
    core, dec, enc = model.core, model.decoder, model.encoder
    cd_params = list(core.parameters()) + list(dec.parameters())
    enc_params = list(enc.parameters())
    if anchor_every is None:
        anchor_every = max(1, int(math.sqrt(K)))

    # ---- forward, no graph: keep final state + sqrt(K) anchors (NOT activations) ----
    with torch.no_grad():
        s0 = enc(context)
        anchors = {0: [x.clone() for x in s0]}
        s = s0
        for t in range(K):
            s = core.step(s, t)
            if (t + 1) % anchor_every == 0:
                anchors[t + 1] = [x.clone() for x in s]
        sT = s

    pgrads = {id(p): torch.zeros_like(p) for p in cd_params}
    adj = [torch.zeros_like(x) for x in sT]          # dL/d s_{t+1}
    cur = sT
    total_loss = 0.0
    for t in reversed(range(K)):
        # reconstruct the pre-step state s_t (exact at an anchor, else via inverse in fp32)
        if t in anchors:
            s_pre = [x.clone() for x in anchors[t]]
        else:
            with torch.no_grad():
                s_pre = core.inverse_step(cur, t)
        leaves = [x.detach().requires_grad_(True) for x in s_pre]
        s_next = core.step(leaves, t)
        y = dec(s_next[0])
        loss_t = objective(y, targets[:, t]) / K       # MEAN over frames: K-invariant grad scale (matches BPTT path)
        surrogate = loss_t + sum((a * sn).sum() for a, sn in zip(adj, s_next))
        g = torch.autograd.grad(surrogate, leaves + cd_params, allow_unused=True)
        new_adj = g[:len(leaves)]
        for p, gp in zip(cd_params, g[len(leaves):]):
            if gp is not None:
                pgrads[id(p)] = pgrads[id(p)] + gp
        adj = [a if a is not None else torch.zeros_like(l) for a, l in zip(new_adj, leaves)]
        total_loss += float(loss_t.detach())
        cur = s_pre

    # ---- propagate dL/ds_0 (= adj) into the encoder ----
    s0g = enc(context)
    enc_surrogate = sum((a * s).sum() for a, s in zip(adj, s0g))
    enc_grads = torch.autograd.grad(enc_surrogate, enc_params, allow_unused=True)
    for p, gp in zip(enc_params, enc_grads):
        pgrads[id(p)] = gp if gp is not None else torch.zeros_like(p)

    if accumulate_into_grad:
        for p in model.parameters():
            g = pgrads[id(p)]
            p.grad = g if p.grad is None else (p.grad + g)
    return total_loss, pgrads


# ----------------------------------------------------------------------------- gradient-equivalence gate
def _cos(ga, gb, params):
    a = torch.cat([ga[id(p)].flatten() for p in params])
    b = torch.cat([gb[id(p)].flatten() for p in params])
    return F.cosine_similarity(a, b, dim=0).item(), ((a - b).norm() / (b.norm() + 1e-12)).item()


def gate(K=40, B=4, seed=0):
    print(f"\n=== reverse-replay gradient-equivalence gate (REAL RHCP, K={K}) ===")
    torch.manual_seed(seed)
    cfg = RHCPConfig(levels=[LevelSpec(64, 16, 1), LevelSpec(96, 8, 2), LevelSpec(128, 4, 4)])
    model = RHCP(cfg).to(DEV)
    # nudge the reversible core off identity init so the test is non-trivial
    with torch.no_grad():
        for p in model.core.parameters():
            p.add_(0.02 * torch.randn_like(p))
    context = torch.randn(B, 10, 1, 64, 64, device=DEV)
    targets = torch.randn(B, K, 1, 64, 64, device=DEV).tanh()
    for name, obj in [("l1mse", obj_l1mse), ("l1lpips", obj_l1lpips)]:
        l_b, g_b = bptt_loss_grads(model, context, targets, obj)
        l_r, g_r = reverse_replay_loss_grads(model, context, targets, obj)
        cos, rel = _cos(g_r, g_b, list(model.parameters()))
        dloss = abs(l_b - l_r)
        ok = cos > 0.9999
        print(f"  obj={name}: loss BPTT {l_b:.4f} / replay {l_r:.4f} (d={dloss:.2e}) | "
              f"grad-cos {cos:.6f} rel-err {rel:.2e}  {'PASS' if ok else 'FAIL'} (gate cos>0.9999)")
        assert ok, "reverse-replay gradients do not match BPTT"
    print("  GATE PASS: reverse-replay backprop is exact on the real RHCP (encoder+core+decoder).")


# ----------------------------------------------------------------------------- data (CW-VAE split, matched to baselines)
import numpy as np
from hvp_rev.eval_adapters import load_clips, CWVAE_TRAIN, CWVAE_TEST


class RolloutClips:
    """CW-VAE train split (N,T) -> (context n_in, targets K) windows in [-1,1]. Matched to the CW-VAE/DLH baselines."""

    def __init__(self, path=CWVAE_TRAIN, time_axis=1, n_in=10, K=40, val_frac=0.05):
        arr = load_clips(path, time_axis)                    # (N,T,1,H,W) in [0,1]
        self.x = torch.from_numpy(arr * 2 - 1)               # [-1,1]
        self.N, self.T = self.x.shape[0], self.x.shape[1]
        self.n_in, self.K = n_in, K
        self.n_val = max(8, int(val_frac * self.N))

    def batch(self, B, train=True, K=None):
        K = K or self.K
        hi = self.N - self.n_val if train else self.N
        lo = 0 if train else self.N - self.n_val
        idx = torch.randint(lo, hi, (B,))
        span = self.n_in + K
        starts = torch.randint(0, max(1, self.T - span + 1), (B,))
        ctx = torch.stack([self.x[idx[b], starts[b]:starts[b] + self.n_in] for b in range(B)]).to(DEV)
        tgt = torch.stack([self.x[idx[b], starts[b] + self.n_in:starts[b] + self.n_in + K] for b in range(B)]).to(DEV)
        return ctx, tgt


def train_rollout(K_train=40, objective="l1lpips", seed=0, steps=2000, batch=8, lr=5e-4,
                  n_in=10, frames_budget=None, log_every=50, out=None, backward="replay"):
    """Free-running latent-rollout training. backward='replay' = constant-memory reverse-replay (P3, both arms);
    'autograd' = plain BPTT (fast, for small-K P1). frames_budget overrides steps to equalise predicted-frames-seen
    across K (compute-match): steps = frames_budget // (batch*K_train). Objective is K-normalised (MEAN over frames)."""
    torch.manual_seed(seed)
    if frames_budget:
        steps = max(1, frames_budget // (batch * K_train))
    cfg = RHCPConfig(levels=[LevelSpec(64, 16, 1), LevelSpec(96, 8, 2), LevelSpec(128, 4, 4)])
    model = RHCP(cfg).to(DEV)
    data = RolloutClips(n_in=n_in, K=K_train)
    obj = OBJECTIVES[objective]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    print(f"train_rollout: K={K_train} obj={objective} seed={seed} steps={steps} batch={batch} backward={backward} "
          f"params={model.n_params()/1e6:.2f}M  (frames/step={batch*K_train})")
    run = 0.0
    for step in range(1, steps + 1):
        ctx, tgt = data.batch(batch, train=True)
        opt.zero_grad(set_to_none=True)
        if backward == "autograd":
            K = tgt.shape[1]
            s = model.init_states(ctx)
            loss = 0.0
            for t in range(K):
                s = model.core.step(s, t)
                loss = loss + obj(model.decoder(s[0]), tgt[:, t]) / K
            loss.backward()
            loss = float(loss.detach())
        else:
            loss, _ = reverse_replay_loss_grads(model, ctx, tgt, obj, accumulate_into_grad=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss
        if step % log_every == 0:
            print(f"[step {step:5d}] loss {run/log_every:.4f}  lr {sched.get_last_lr()[0]:.2e}")
            run = 0.0
    if out:
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg, "K_train": K_train, "seed": seed}, out)
        print(f"saved {out}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--K_train", type=int, default=40)
    ap.add_argument("--objective", default="l1lpips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--frames_budget", type=int, default=None)
    ap.add_argument("--backward", default="replay", choices=["replay", "autograd"])
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n_in", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"rollout_train | torch {torch.__version__} | device {DEV}")
    if args.gate:
        gate(K=args.K)
    elif args.smoke:
        train_rollout(K_train=12, objective="l1mse", steps=120, batch=8, log_every=20)
    elif args.train:
        train_rollout(K_train=args.K_train, objective=args.objective, seed=args.seed, lr=args.lr,
                      steps=args.steps, batch=args.batch, frames_budget=args.frames_budget,
                      backward=args.backward, n_in=args.n_in, out=args.out)
    else:
        gate(K=args.K)
