"""Image-quality metrics (torch, no extra deps) + frame-strip helpers for logging.

All image tensors are (.., C, H, W) or (B, C, T, H, W) in [-1, 1] unless noted.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(pred, x).item()
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)


def _gauss_window(ws: int = 11, sigma: float = 1.5) -> torch.Tensor:
    c = torch.arange(ws).float() - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_map(a: torch.Tensor, b: torch.Tensor, ws: int = 7) -> torch.Tensor:
    """Per-pixel SSIM map, mean over channels. a,b: (N,C,H,W) in [-1,1] -> (N,1,H,W)."""
    a = (a + 1) / 2
    b = (b + 1) / 2
    C = a.shape[1]
    w = _gauss_window(ws).to(a.device, a.dtype)[None, None].repeat(C, 1, 1, 1)
    pad = ws // 2
    conv = lambda t: F.conv2d(t, w, padding=pad, groups=C)
    mu_a, mu_b = conv(a), conv(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = conv(a * a) - mu_a2
    vb = conv(b * b) - mu_b2
    vab = conv(a * b) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * vab + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return s.mean(dim=1, keepdim=True)


def ssim(a: torch.Tensor, b: torch.Tensor, ws: int = 7) -> float:
    """Mean SSIM over a batch of images. a,b: (N,C,H,W) in [-1,1]."""
    return float(ssim_map(a, b, ws).mean())


def video_ssim(pred: torch.Tensor, x: torch.Tensor) -> float:
    """pred,x: (B,C,T,H,W) in [-1,1] -> mean per-frame SSIM."""
    B, C, T, H, W = pred.shape
    p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    g = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return ssim(p, g)


def frames_to_strip(v: torch.Tensor) -> np.ndarray:
    """(C,T,H,W) in [-1,1] -> (H, T*W, 3) uint8 horizontal strip."""
    v = ((v.detach().float().cpu() + 1) * 127.5).clamp(0, 255).byte().numpy()  # (C,T,H,W)
    C, T, H, W = v.shape
    return np.concatenate([v[:, t].transpose(1, 2, 0) for t in range(T)], axis=1)


def stacked_strip(rows: List[np.ndarray]) -> np.ndarray:
    """Stack labelled strip rows vertically (e.g. GT over Pred)."""
    return np.concatenate(rows, axis=0)
