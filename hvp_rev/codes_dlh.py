"""Extract frozen DLH decoder-input codes (the 200-dim `c` that feeds `img_dec`) paired with frames, for
renderer training/eval. Teacher-forced (posterior, real frames) and free-run (open-loop prior rollout) variants.

DLH builds `c` inside `_step` and calls `self.img_dec(c_dec)`. We capture `c_dec` (the actual decoder input,
== `c` when decoder_det_carry is off) by temporarily wrapping `img_dec`. Mirrors hvp_rev/codes.py for CW-VAE."""
from __future__ import annotations
import torch, torch.nn as nn


class _Capture(nn.Module):
    def __init__(self, inner, sink): super().__init__(); self.inner, self.sink = inner, sink
    def forward(self, z): self.sink.append(z.detach()); return self.inner(z)


@torch.no_grad()
def teacher_forced_codes_dlh(model, obs):
    """obs (B,T,1,64,64) -> (codes (B*T,200), frames (B*T,1,64,64)). Posterior unroll, decoder-input code."""
    cfg = model.cfg; B, T = obs.shape[:2]
    feats = model._bottom_up(obs); state = model._init_state(B, obs.device)
    sink = []; real = model.img_dec; model.img_dec = _Capture(real, sink)
    try:
        for t in range(T):
            x_t = [feats[n][:, t] for n in range(cfg.levels)]
            _, state, _ = model._step(state, x_t, mode="posterior")        # appends one code per step
    finally:
        model.img_dec = real
    C = torch.stack(sink, 1).reshape(B * T, -1)                            # (B*T,200)
    return C, obs.reshape(B * T, 1, 64, 64)


@torch.no_grad()
def free_run_codes_dlh(model, obs, ctx, horizon):
    """obs (B,ctx+H,1,64,64) -> (codes (B*H,200), frames (B*H,1,64,64)) for the FREE-RUN future — the open-loop
    codes the renderer must actually handle at inference. Context uses the posterior, then prior free-run."""
    cfg = model.cfg; B = obs.shape[0]
    feats = model._bottom_up(obs[:, :ctx]); state = model._init_state(B, obs.device)
    for t in range(ctx):                                                   # warm the state on context (no capture)
        x_t = [feats[n][:, t] for n in range(cfg.levels)]
        _, state, _ = model._step(state, x_t, mode="posterior")
    sink = []; real = model.img_dec; model.img_dec = _Capture(real, sink)
    try:
        for _ in range(horizon):
            _, state, _ = model._step(state, None, mode="prior")           # appends one code per step
    finally:
        model.img_dec = real
    C = torch.stack(sink, 1).reshape(B * horizon, -1)                      # (B*H,200)
    return C, obs[:, ctx:ctx + horizon].reshape(B * horizon, 1, 64, 64)
