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
from hvp.predictor.rollout import unroll_latent
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


def lambda_rollout_at(step: int, pcfg) -> float:
    """Ramp the P2 self-rollout weight 0->target over rollout_warmup_steps, STARTING only after the
    predictive warmup (so the encoder/cascade have settled before the harder term turns on)."""
    sc = pcfg.sched
    if not sc.rollout_enabled:
        return 0.0
    if sc.rollout_warmup_steps <= 0:
        return sc.lambda_rollout
    prog = (step - pcfg.pred_warmup_steps) / sc.rollout_warmup_steps
    return sc.lambda_rollout * min(1.0, max(0.0, prog))


def compute_rollout_loss(cascade, z_det, pcfg) -> torch.Tensor:
    """P2 self-rollout: the cascade consumes its OWN detached predictions for K steps and is supervised
    (head_nll) against the DETACHED encoder latents shifted k+2 ahead. Within-window targets only
    (K=1 needs no extra frames); positions whose shift falls outside the window are skipped."""
    sc = pcfg.sched
    pw, head, K = pcfg.stage_pred_weight, pcfg.head, pcfg.head_components
    seq = unroll_latent(cascade, z_det, sc.rollout_steps, sc.sample_rollout, sc.detach_feedback)
    loss = z_det[0].new_zeros(())
    for k, rp in enumerate(seq):
        shift = k + 2
        for s in range(len(z_det)):
            if z_det[s].shape[2] > shift:
                w = pw[s] if s < len(pw) else 1.0
                loss = loss + w * predictive_loss(rp[s], z_det[s], head, K, shift=shift)
    return loss


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

    # --- P2 anti-drift self-rollout term (off by default => byte-identical to before) ---
    sc = cfg.predictor.sched
    lam_roll = lambda_rollout_at(step, cfg.predictor)
    if sc.rollout_enabled:
        if lam_roll > 0:
            roll_loss = compute_rollout_loss(cascade, z_det, cfg.predictor)
            total = total + lam_roll * roll_loss
        else:                                          # warmup: log the proxy without building grad
            with torch.no_grad():
                roll_loss = compute_rollout_loss(cascade, z_det, cfg.predictor)
        # Live diagnostic (interpret COMPARATIVELY: P2.1 run vs the lambda_rollout=0 control, not by
        # absolute sign). `rollout_loss` is the clean train-what-you-test signal (it exactly matches the
        # deployed latent rollout). The TF-2 baseline below feeds the GT-shifted stack, which is itself
        # positionally off-distribution for the cascade, so the proxy's absolute sign is confounded — the
        # AUTHORITATIVE gate is the offline scripts/eval_rollout.py PIXEL changed_psnr@h / TF->AR gap.
        with torch.no_grad():
            pw, head, K = cfg.predictor.stage_pred_weight, cfg.predictor.head, cfg.predictor.head_components
            gt_next = [torch.cat([z[:, :, 1:], z[:, :, -1:]], dim=2) for z in z_det]  # GT intermediate stack
            tf_preds = cascade(gt_next)
            tf2 = total.new_zeros(())
            for s in range(len(z_det)):
                if z_det[s].shape[2] > 2:
                    w = pw[s] if s < len(pw) else 1.0
                    tf2 = tf2 + w * predictive_loss(tf_preds[s], z_det[s], head, K, shift=2)
        m["rollout_loss"] = float(roll_loss.detach())
        m["tf2_loss"] = float(tf2)
        m["sf_tf_ar_proxy"] = float(roll_loss.detach() - tf2)
    m["lambda_rollout"] = lam_roll
    m["loss"] = float(total.detach())
    return total, m, out, preds
