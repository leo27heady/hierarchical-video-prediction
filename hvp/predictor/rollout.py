"""Autoregressive rollout (P1) — turn the single-step cascade into a free-running predictor.

The model predicts frame t+1 from a fixed-size window of T_seed frames, so rollout slides a
fixed window (O(n_steps), no KV-cache needed). Two feedback modes:

- ``pixel``  (honest / decode-consistent): decode the predicted latents to a frame, append it
  to the pixel window, RE-ENCODE, predict again — pays the tokenizer round-trip each step.
- ``latent`` (optimistic): the predicted next-step latent stack becomes the next context
  directly (no decode→re-encode); cheaper, no round-trip error, but optimistic.

`teacher_forced_at` gives the best-case single-step prediction at a given horizon (GT context),
for the TF→AR gap.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from hvp.predictor.heads import head_mode, head_sample


def _pred_to_latents(cascade, preds: List[torch.Tensor], sample: bool = False) -> List[torch.Tensor]:
    """(B,T,H,W,out) per stage -> (B,C,T,H,W) latents via each stage's head.
    sample=False -> the stable point prediction (mixture mode); sample=True -> a stochastic
    draw (for best-of-N / multimodal rollout). l2/gaussian are unaffected by the flag."""
    fn = head_sample if sample else head_mode
    out = []
    for s, p in enumerate(preds):
        st = cascade.stages[s]
        lat = fn(p, st.head_kind, st.in_ch, st.head_components)
        out.append(lat.permute(0, 4, 1, 2, 3).contiguous())
    return out


@torch.no_grad()
def free_run(tok, cascade, seed: torch.Tensor, n_steps: int, mode: str = "pixel",
             sample: bool = False) -> torch.Tensor:
    """seed: (B,3,T_seed,H,W). Returns predicted frames (B,3,n_steps,H,W) for the frames
    immediately following the seed window."""
    tok.eval(); cascade.eval()
    N = tok.topology.n_stages()
    outs: List[torch.Tensor] = []

    if mode == "pixel":
        window = seed.clone()
        for _ in range(n_steps):
            z = tok.encoder(window)
            z_pred = _pred_to_latents(cascade, cascade(z), sample=sample)
            recon = tok.decoder.decode_depth(z_pred, N)        # (B,3,T_seed,H,W)
            nxt = recon[:, :, -1]                               # predicted next frame
            outs.append(nxt)
            window = torch.cat([window[:, :, 1:], nxt[:, :, None]], dim=2)
    elif mode == "latent":
        z = tok.encoder(seed)
        for _ in range(n_steps):
            z_pred = _pred_to_latents(cascade, cascade(z), sample=sample)
            recon = tok.decoder.decode_depth(z_pred, N)
            outs.append(recon[:, :, -1])
            z = z_pred                                          # predicted shifted stack = next context
    else:
        raise ValueError(f"unknown rollout mode {mode!r}")

    return torch.stack(outs, dim=2).clamp(-1, 1)


@torch.no_grad()
def teacher_forced_at(tok, cascade, gt: torch.Tensor, seed_len: int, horizon: int) -> torch.Tensor:
    """Best-case single-step prediction of frame ``seed_len+horizon`` given the GROUND-TRUTH
    window ending just before it. gt: (B,3,T,H,W). Returns (B,3,H,W)."""
    tok.eval(); cascade.eval()
    N = tok.topology.n_stages()
    window = gt[:, :, horizon: horizon + seed_len]              # GT frames [h .. h+seed_len-1]
    z = tok.encoder(window)
    z_pred = _pred_to_latents(cascade, cascade(z))
    recon = tok.decoder.decode_depth(z_pred, N)
    return recon[:, :, -1].clamp(-1, 1)
