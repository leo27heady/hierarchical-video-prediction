"""Training objective: deep-supervised reconstruction + VICReg anti-collapse.

The encoder is shaped by reconstruction (+ a VICReg variance/covariance floor so the
coarse latents cannot dimension-collapse). A coarse-first curriculum ramps the active
decode depth (top -> top+mid -> top+mid+bot). The predictive term (detached target) is
added by the trainer on top and is intentionally NOT part of the encoder's recon loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from hvp.config import Config
from hvp.predictor.continuous import predictive_loss
from hvp.tokenizer import HierTokenizer, TokenizerOut


def active_depth_at(step: int, n_stages: int, cfg: Config) -> int:
    if not cfg.vae.progressive:
        return n_stages
    return min(n_stages, 1 + step // max(1, cfg.vae.progressive_steps_per_depth))


def _recon_term(pred: torch.Tensor, x: torch.Tensor, kind: str) -> torch.Tensor:
    return F.l1_loss(pred, x) if kind == "l1" else F.mse_loss(pred, x)


def psnr(pred: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(pred, x).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)  # range [-1,1] -> peak-to-peak 2


def vicreg_terms(z: torch.Tensor, gamma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-dim variance hinge + off-diagonal covariance penalty on a (B,C,T,H,W) latent."""
    c = z.shape[1]
    f = z.permute(0, 2, 3, 4, 1).reshape(-1, c)           # (N, C)
    f = f - f.mean(dim=0, keepdim=True)
    std = torch.sqrt(f.var(dim=0) + 1e-4)                  # (C,)
    var_hinge = torch.clamp(gamma - std, min=0.0).pow(2).mean()
    n = max(1, f.shape[0] - 1)
    cov = (f.T @ f) / n                                    # (C, C)
    off = cov - torch.diag(torch.diag(cov))
    cov_pen = off.pow(2).sum() / c
    return var_hinge, cov_pen


def compute_vae_loss(
    tokenizer: HierTokenizer, x: torch.Tensor, cfg: Config, step: int
) -> Tuple[torch.Tensor, Dict[str, float], TokenizerOut]:
    n = cfg.topology.n_stages()
    depth = active_depth_at(step, n, cfg)
    out = tokenizer(x, max_depth=depth)

    weights = cfg.vae.depth_loss_weights or tuple(1.0 for _ in range(len(out.recons)))
    recon = x.new_zeros(())
    for d, pred in enumerate(out.recons):
        recon = recon + weights[d] * _recon_term(pred, x, cfg.vae.recon_loss)
    recon = recon / max(len(out.recons), 1)

    var_total = x.new_zeros(())
    cov_total = x.new_zeros(())
    stage_std: List[float] = []
    for s, z in enumerate(out.z):
        vh, cp = vicreg_terms(z, cfg.vae.vicreg_gamma)
        scale = cfg.vae.vicreg_stage_scale[s] if s < len(cfg.vae.vicreg_stage_scale) else 1.0
        var_total = var_total + scale * vh
        cov_total = cov_total + scale * cp
        with torch.no_grad():
            stage_std.append(float(z.permute(0, 2, 3, 4, 1).reshape(-1, z.shape[1]).std(dim=0).mean()))

    total = (
        cfg.vae.recon_weight * recon
        + cfg.vae.vicreg_weight * var_total
        + cfg.vae.vicreg_cov_weight * cov_total
    )

    metrics = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "vicreg_var": float(var_total.detach()),
        "vicreg_cov": float(cov_total.detach()),
        "active_depth": depth,
        "psnr_full": psnr(out.recon.detach(), x),
    }
    for s, sd in enumerate(stage_std):
        metrics[f"std_s{s}"] = sd
    return total, metrics, out


def lambda_pred_at(step: int, pcfg) -> float:
    if pcfg.pred_warmup_steps <= 0:
        return pcfg.lambda_pred
    return pcfg.lambda_pred * min(1.0, step / pcfg.pred_warmup_steps)


def compute_oneshot_loss(tokenizer, cascade, x, cfg: Config, step: int):
    """Detached-target one-shot: encoder shaped by recon (+VICReg); the cascade predicts
    the encoder's own DETACHED latents (so prediction never collapses the encoder)."""
    recon_total, m, out = compute_vae_loss(tokenizer, x, cfg, step)
    z_det = [zi.detach() for zi in out.z]
    preds = cascade(z_det)
    lam = lambda_pred_at(step, cfg.predictor)
    pw = cfg.predictor.stage_pred_weight
    pred_loss = x.new_zeros(())
    for s, p in enumerate(preds):
        pl = predictive_loss(p, z_det[s], cfg.predictor.head, cfg.predictor.head_components)
        w = pw[s] if s < len(pw) else 1.0
        pred_loss = pred_loss + w * pl
        m[f"pred_s{s}"] = float(pl.detach())
    total = recon_total + lam * pred_loss
    m["pred_loss"] = float(pred_loss.detach())
    m["lambda"] = lam
    m["loss"] = float(total.detach())
    return total, m, out, preds
