"""Per-stage prediction heads: deterministic L2, single Gaussian, or a Gaussian MIXTURE (GMM).

The L2 head regresses the conditional mean -> blur, and (the v1 bug) the `gaussian` path was
never sampled. This module centralises all three so the distributional head is wired END-TO-END:
the loss (NLL), the point estimate used by variance-explained, and the SAMPLED latent used by the
pixel decode / rollout all go through here.

A GMM head with K components outputs, per cell: K mixture logits + K*C means + K*C log-variances
(``head_out_dim`` = K*(2C+1)). Under genuine multimodality (e.g. SO(3) depth/axis ambiguity) the
mixture can place mass on each future mode instead of averaging them to mush.

All functions operate on the last axis: ``pred`` is (..., head_out_dim), ``tgt``/returns are (..., C).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def head_out_dim(kind: str, in_ch: int, n_components: int = 1) -> int:
    if kind in ("l2", "warp"):
        return in_ch
    if kind == "gaussian":
        return 2 * in_ch
    if kind == "gmm":
        return n_components * (2 * in_ch + 1)
    raise ValueError(f"unknown head {kind!r}")


def _split_gmm(pred: torch.Tensor, C: int, K: int):
    """(..., K*(2C+1)) -> logits (...,K), mu (...,K,C), logvar (...,K,C)."""
    logits = pred[..., :K]
    rest = pred[..., K:].reshape(*pred.shape[:-1], K, 2 * C)
    mu, logvar = rest[..., :C], rest[..., C:].clamp(-8.0, 8.0)
    return logits, mu, logvar


def head_nll(pred: torch.Tensor, tgt: torch.Tensor, kind: str, C: int, K: int = 1) -> torch.Tensor:
    """Training loss: MSE (l2) or negative log-likelihood (gaussian / gmm). Mean over all cells."""
    if kind in ("l2", "warp"):
        return F.mse_loss(pred[..., :C], tgt)
    if kind == "gaussian":
        mu, logvar = pred[..., :C], pred[..., C:].clamp(-8.0, 8.0)
        return 0.5 * (logvar + (tgt - mu) ** 2 / logvar.exp()).mean()
    if kind == "gmm":
        logits, mu, logvar = _split_gmm(pred, C, K)
        log_pi = F.log_softmax(logits, dim=-1)                       # (...,K)
        t = tgt.unsqueeze(-2)                                        # (...,1,C)
        log_n = -0.5 * (logvar + (t - mu) ** 2 / logvar.exp() + math.log(2 * math.pi)).sum(-1)  # (...,K)
        return -(torch.logsumexp(log_pi + log_n, dim=-1)).mean()
    raise ValueError(f"unknown head {kind!r}")


def head_mean(pred: torch.Tensor, kind: str, C: int, K: int = 1) -> torch.Tensor:
    """MMSE point estimate (mixture mean). Used by variance-explained."""
    if kind in ("l2", "gaussian", "warp"):
        return pred[..., :C]
    logits, mu, _ = _split_gmm(pred, C, K)
    pi = F.softmax(logits, dim=-1).unsqueeze(-1)                     # (...,K,1)
    return (pi * mu).sum(-2)


def head_mode(pred: torch.Tensor, kind: str, C: int, K: int = 1) -> torch.Tensor:
    """Most-likely component's mean — the stable point prediction for rollout/decode."""
    if kind in ("l2", "gaussian", "warp"):
        return pred[..., :C]
    logits, mu, _ = _split_gmm(pred, C, K)
    k = logits.argmax(-1, keepdim=True).unsqueeze(-1).expand(*logits.shape[:-1], 1, C)
    return torch.gather(mu, -2, k).squeeze(-2)


def head_sample(pred: torch.Tensor, kind: str, C: int, K: int = 1) -> torch.Tensor:
    """Draw one stochastic sample (for best-of-N / multimodal eval)."""
    if kind in ("l2", "warp"):
        return pred[..., :C]
    if kind == "gaussian":
        mu, logvar = pred[..., :C], pred[..., C:].clamp(-8.0, 8.0)
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
    logits, mu, logvar = _split_gmm(pred, C, K)
    pi = F.softmax(logits, dim=-1)
    k = torch.distributions.Categorical(probs=pi).sample().unsqueeze(-1).unsqueeze(-1).expand(*pi.shape[:-1], 1, C)
    muk = torch.gather(mu, -2, k).squeeze(-2)
    lvk = torch.gather(logvar, -2, k).squeeze(-2)
    return muk + (0.5 * lvk).exp() * torch.randn_like(muk)
