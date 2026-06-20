"""Predictive uncertainty / entropy maps for the single-step distributional predictor.

For each held-out clip we draw ``n_samples`` next-frame predictions from the cascade's
distributional (GMM/gaussian) head, decode each to pixels, and measure the per-pixel spread
(std across samples). That spread IS the model's predictive uncertainty. We render it as a
black-and-white map where **WHITE = certain** (low spread) and **BLACK = uncertain** (high spread).

Only meaningful for a distributional head: an ``l2`` head samples deterministically (zero
spread -> an all-white map), which we detect and warn about.

Notes on the decode:
- ``preds = cascade(z_gt)`` predicts the *next*-frame latent at every stage (teacher-forced t->t+1),
  so position ``t`` of a decoded sample is the prediction of frame ``t+1``.
- By default we decode the FULL predicted hierarchy (the model's actual single-step output).
  ``teacher_coarse=True`` instead keeps the predicted BOT (finest) latent but supplies the
  ground-truth coarser latents, which removes the cross-rate temporal inconsistency of decoding
  all predicted stages jointly (the finest stage is temporally 1:1, the coarse ones are not).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from hvp.predictor.rollout import _pred_to_latents


def _mix_with_teacher(z_pred: List[torch.Tensor], z_gt: List[torch.Tensor],
                      teacher_coarse: bool) -> List[torch.Tensor]:
    """Keep predicted finest (stage 0); optionally replace coarser stages with ground-truth latents."""
    if not teacher_coarse:
        return z_pred
    return [z_pred[0]] + [z_gt[s] for s in range(1, len(z_pred))]


@torch.no_grad()
def predict_with_uncertainty(tok, cascade, x: torch.Tensor, n_samples: int = 16,
                             teacher_coarse: bool = False
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (B,3,T,H,W) in [-1,1].
    Returns:
      mode_pred : (B,3,T,H,W)  the point (mixture-mode) prediction, decoded.
      std_pix   : (B,T,H,W)    per-pixel std across samples (mean over colour channels).
    Position t of both is the prediction of input frame t+1 (compare to x[:, :, 1:])."""
    tok.eval(); cascade.eval()
    N = tok.topology.n_stages()
    z_gt = [zi.float() for zi in tok.encoder(x)]
    preds = [p.float() for p in cascade(z_gt)]

    z_mode = _mix_with_teacher(_pred_to_latents(cascade, preds, sample=False), z_gt, teacher_coarse)
    mode_pred = tok.decoder.decode_depth(z_mode, N).float().clamp(-1, 1)

    if n_samples <= 0:
        return mode_pred, torch.zeros(x.shape[0], x.shape[2], x.shape[3], x.shape[4], device=x.device)

    acc_sum = torch.zeros_like(mode_pred)
    acc_sq = torch.zeros_like(mode_pred)
    for _ in range(n_samples):
        z_s = _mix_with_teacher(_pred_to_latents(cascade, preds, sample=True), z_gt, teacher_coarse)
        rec = tok.decoder.decode_depth(z_s, N).float().clamp(-1, 1)
        acc_sum += rec
        acc_sq += rec * rec
    mean = acc_sum / n_samples
    var = (acc_sq / n_samples - mean * mean).clamp(min=0.0)
    std_pix = var.sqrt().mean(dim=1)                 # (B,T,H,W): mean std over channels
    return mode_pred, std_pix


def uncertainty_to_gray(std_pix: torch.Tensor, std_scale: Optional[float] = None) -> torch.Tensor:
    """(.., H, W) per-pixel std -> (.., H, W) grayscale in [0,1], 1=white=certain, 0=black=uncertain.

    If ``std_scale`` is given, std is normalized by it (fixed, comparable across frames/runs);
    otherwise it self-calibrates to the 99th percentile of the given tensor (good contrast for review)."""
    if std_scale is None:
        flat = std_pix.flatten().float()
        std_scale = float(torch.quantile(flat, 0.99)) if flat.numel() else 1.0
        std_scale = max(std_scale, 1e-6)
    norm = (std_pix / std_scale).clamp(0.0, 1.0)
    return 1.0 - norm                                # white = certain


def gray_to_strip(gray: torch.Tensor) -> np.ndarray:
    """(T,H,W) in [0,1] -> (H, T*W, 3) uint8 horizontal strip (replicated to 3 channels)."""
    g = (gray.detach().float().cpu().clamp(0, 1) * 255).byte().numpy()   # (T,H,W)
    T, H, W = g.shape
    row = np.concatenate([g[t] for t in range(T)], axis=1)               # (H, T*W)
    return np.repeat(row[:, :, None], 3, axis=2)                         # (H, T*W, 3)
