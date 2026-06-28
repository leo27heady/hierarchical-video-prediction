"""Extract frozen CW-VAE bottom codes [s,det] (220-dim) paired with frames, for renderer training/eval.
Teacher-forced (posterior, real frames) and free-run (prior rollout) variants."""
from __future__ import annotations
import torch


@torch.no_grad()
def teacher_forced_codes(m, obs, sample=True):
    """obs (B,T,1,64,64) -> (codes (B*T,220), frames (B*T,1,64,64)). Posterior bottom output."""
    enc = m.encoder(obs)
    bot, _, _, _ = m.hierarchical_unroll(enc, use_obs=True, sample=sample)
    B, T = obs.shape[:2]
    return bot.reshape(B * T, -1).detach(), obs.reshape(B * T, 1, 64, 64)


@torch.no_grad()
def free_run_codes(m, obs, ctx, sample=True):
    """obs (B,ctx+H,1,64,64) -> (codes (B*H,220), frames (B*H,1,64,64)) for the free-run future (the open-loop
    codes the renderer must handle at inference). Mirrors CWVAE.open_loop's two-phase unroll."""
    enc = m.encoder(obs); L = m.levels
    pre, post, cl = [], [], ctx
    for lvl in range(L):
        pre.append(enc[lvl][:, :cl]); post.append(torch.zeros_like(enc[lvl][:, cl:])); cl = cl // m.factor
    _, last, _, _ = m.hierarchical_unroll(pre, use_obs=True, sample=sample)
    bot, _, _, _ = m.hierarchical_unroll(post, use_obs=[False] * L, initial=last, sample=sample)
    B, H = bot.shape[:2]
    return bot.reshape(B * H, -1).detach(), obs[:, ctx:ctx + H].reshape(B * H, 1, 64, 64)
