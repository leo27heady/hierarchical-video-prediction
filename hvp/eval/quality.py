"""Perceptual / distributional / changed-region metrics (P0 eval spine).

- changed_region_*: PSNR/SSIM restricted to pixels that actually MOVE frame-to-frame,
  so the persistence baseline (trivially good on the static background) stops being a
  free win.
- LPIPSMetric: perceptual distance (catches the L2 blur that PSNR misses). Lazy-loaded.
- fvd: Fréchet Video Distance with a Kinetics-pretrained r3d_18 backbone (best-effort —
  noisy on small sample counts; reported, not gated).

All image tensors are in [-1, 1].
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from hvp.metrics import ssim_map


def _video_to_frames(v: torch.Tensor) -> torch.Tensor:
    """(B,C,T,H,W) -> (B*T,C,H,W)."""
    B, C, T, H, W = v.shape
    return v.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


def changed_mask(gt_next: torch.Tensor, cur: torch.Tensor, thresh: float = 0.1) -> torch.Tensor:
    """Per-pixel mask (B*T,1,H,W) where the frame changed by > thresh (L1 over channels)."""
    g, c = _video_to_frames(gt_next), _video_to_frames(cur)
    return (g - c).abs().mean(dim=1, keepdim=True) > thresh


def changed_region_metrics(pred: torch.Tensor, gt_next: torch.Tensor, cur: torch.Tensor,
                           thresh: float = 0.1) -> Dict[str, float]:
    """PSNR/SSIM over the moving region only. pred,gt_next,cur: (B,C,T,H,W) in [-1,1]."""
    p, g = _video_to_frames(pred), _video_to_frames(gt_next)
    m = changed_mask(gt_next, cur, thresh).float()                  # (N,1,H,W)
    denom = m.sum().clamp(min=1.0)
    sq = (p - g).pow(2).mean(dim=1, keepdim=True)                   # (N,1,H,W)
    mse = (sq * m).sum() / denom
    psnr = 99.0 if mse <= 1e-12 else 20 * math.log10(2.0) - 10 * math.log10(float(mse))
    smap = ssim_map(p, g)                                           # (N,1,H,W)
    sval = float((smap * m).sum() / denom)
    return {"changed_psnr": psnr, "changed_ssim": sval, "changed_frac": float(m.mean())}


class LPIPSMetric:
    """Lazy LPIPS (AlexNet). Call on (B,C,T,H,W) in [-1,1]; returns mean perceptual distance."""

    def __init__(self, net: str = "alex"):
        self.net = net
        self._model = None

    def _ensure(self, device):
        if self._model is None:
            import lpips
            self._model = lpips.LPIPS(net=self.net, verbose=False).to(device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        model = self._ensure(pred.device)
        p, g = _video_to_frames(pred), _video_to_frames(gt)
        return float(model(p, g).mean())


_FVD_MODEL = None
_KIN_MEAN = (0.43216, 0.394666, 0.37645)
_KIN_STD = (0.22803, 0.22145, 0.216989)


@torch.no_grad()
def _fvd_features(v: torch.Tensor, device) -> torch.Tensor:
    """r3d_18 penultimate features for clips v: (B,3,T,H,W) in [-1,1]."""
    global _FVD_MODEL
    if _FVD_MODEL is None:
        from torchvision.models.video import R3D_18_Weights, r3d_18
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(device).eval()
        m.fc = torch.nn.Identity()
        _FVD_MODEL = m
    x = (v + 1) / 2                                                # [-1,1] -> [0,1]
    B, C, T, H, W = x.shape
    x = F.interpolate(x.reshape(B * T, C, H, W), size=(112, 112), mode="bilinear", align_corners=False)
    x = x.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)        # (B,C,T,112,112)
    mean = torch.tensor(_KIN_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(_KIN_STD, device=device).view(1, 3, 1, 1, 1)
    x = (x - mean) / std
    return _FVD_MODEL(x)                                           # (B, 512)


def _frechet(f_real: torch.Tensor, f_fake: torch.Tensor) -> float:
    mu_r, mu_f = f_real.mean(0), f_fake.mean(0)
    fr = f_real - mu_r
    ff = f_fake - mu_f
    cov_r = fr.T @ fr / max(1, f_real.shape[0] - 1)
    cov_f = ff.T @ ff / max(1, f_fake.shape[0] - 1)
    diff = (mu_r - mu_f).pow(2).sum()
    # sqrt(cov_r @ cov_f) trace via eigen of the product (symmetrized for stability)
    prod = cov_r @ cov_f
    eig = torch.linalg.eigvals(prod).real.clamp(min=0)
    covmean = eig.sqrt().sum()
    return float(diff + cov_r.trace() + cov_f.trace() - 2 * covmean)


@torch.no_grad()
def fvd(real: torch.Tensor, fake: torch.Tensor, device: Optional[torch.device] = None) -> float:
    """Fréchet Video Distance (r3d_18). real,fake: (B,3,T,H,W) in [-1,1]. Best-effort:
    noisy when B is small relative to the 512-dim feature covariance."""
    device = device or real.device
    fr = _fvd_features(real.to(device), device).float()
    ff = _fvd_features(fake.to(device), device).float()
    return _frechet(fr, ff)
