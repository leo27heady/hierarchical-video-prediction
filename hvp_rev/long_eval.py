"""Long-horizon foreground evaluation for the RHCP drift panels (P0/P3).

The ONLY trustworthy long-horizon metrics on MMNIST (aggregate MSE/SSIM/PSNR are background-dominated -- all-black wins):
  - fg-MSE        : MSE on the foreground mask (gt>0.1), per frame.
  - fg-LPIPS      : AlexNet LPIPS on fg-masked frames, per frame.
  - centroid-L2   : L2 (in pixels) between intensity-weighted foreground centroids of pred vs gt -- POSITION drift,
                    which (unlike fg-pixel-MSE) does NOT saturate at the decorrelation floor by h~15.
  - digit-IoU     : IoU of thresholded (>0.1) pred vs gt masks -- shape/coverage agreement.
All per-(clip,frame); aggregated over horizon windows with a clip-bootstrap 95% CI.

pred/true are (N, T, 1, H, W) in [0,1] (use hvp_rev.eval_adapters.load_clips + a model rollout).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------------- per-frame metrics
def fg_mse(pred: np.ndarray, true: np.ndarray, thr: float = 0.1) -> np.ndarray:
    """(N,T,1,H,W) -> (N,T) MSE over foreground pixels (gt>thr); falls back to full-frame if mask empty."""
    m = (true > thr).astype(np.float32)
    se = (pred - true) ** 2 * m
    denom = m.sum(axis=(2, 3, 4))
    num = se.sum(axis=(2, 3, 4))
    full = ((pred - true) ** 2).mean(axis=(2, 3, 4))
    out = np.where(denom > 0, num / np.maximum(denom, 1e-6), full)
    return out


def _centroid(frame: np.ndarray) -> Tuple[float, float]:
    """Intensity-weighted centroid (row, col) of a single (H,W) frame; (-1,-1) if empty."""
    s = frame.sum()
    if s < 1e-6:
        return -1.0, -1.0
    H, W = frame.shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    return float((frame * rows).sum() / s), float((frame * cols).sum() / s)


def centroid_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """(N,T,1,H,W) -> (N,T) L2 pixel distance between fg centroids. Empty-pred penalized with the frame diagonal."""
    N, T = pred.shape[0], pred.shape[1]
    H, W = pred.shape[3], pred.shape[4]
    diag = float(np.hypot(H, W))
    out = np.zeros((N, T), np.float32)
    for i in range(N):
        for j in range(T):
            pr, pc = _centroid(pred[i, j, 0])
            gr, gc = _centroid(true[i, j, 0])
            if gr < 0:                       # no gt foreground -> skip (set 0)
                out[i, j] = 0.0
            elif pr < 0:                     # pred faded to empty -> max penalty
                out[i, j] = diag
            else:
                out[i, j] = float(np.hypot(pr - gr, pc - gc))
    return out


def digit_iou(pred: np.ndarray, true: np.ndarray, thr: float = 0.1) -> np.ndarray:
    """(N,T,1,H,W) -> (N,T) IoU of thresholded masks."""
    p = pred[:, :, 0] > thr
    g = true[:, :, 0] > thr
    inter = (p & g).sum(axis=(2, 3)).astype(np.float32)
    union = (p | g).sum(axis=(2, 3)).astype(np.float32)
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 1.0)


_LPIPS = None


def fg_lpips(pred: np.ndarray, true: np.ndarray, thr: float = 0.1, device: str = "cuda",
            batch: int = 256) -> np.ndarray:
    """(N,T,1,H,W) -> (N,T) AlexNet LPIPS on fg-masked frames (pred,true multiplied by gt mask, mapped to [-1,1])."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    N, T = pred.shape[0], pred.shape[1]
    m = (true > thr).astype(np.float32)
    pm = (pred * m)[:, :, 0]                  # (N,T,H,W)
    gm = (true * m)[:, :, 0]
    pm = np.repeat(pm.reshape(N * T, 1, *pm.shape[2:]), 3, axis=1)   # 1->3 ch
    gm = np.repeat(gm.reshape(N * T, 1, *gm.shape[2:]), 3, axis=1)
    out = np.zeros(N * T, np.float32)
    with torch.no_grad():
        for s in range(0, N * T, batch):
            a = torch.from_numpy(pm[s:s + batch] * 2 - 1).to(device)
            b = torch.from_numpy(gm[s:s + batch] * 2 - 1).to(device)
            out[s:s + batch] = _LPIPS(a, b).squeeze().cpu().numpy()
    return out.reshape(N, T)


# ----------------------------------------------------------------------------- windowed bootstrap CI
def bootstrap_ci(per_clip: np.ndarray, n_boot: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    """per_clip: (N,) clip-level means -> (mean, lo95, hi95) via clip-bootstrap."""
    rng = np.random.RandomState(seed)
    N = per_clip.shape[0]
    boots = np.array([per_clip[rng.randint(0, N, N)].mean() for _ in range(n_boot)])
    return float(per_clip.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def windowed(metric_nt: np.ndarray, windows: List[Tuple[int, int]]) -> Dict[str, Tuple[float, float, float]]:
    """metric_nt: (N,T) per-frame -> {label: (mean, lo, hi)} averaging frames in each [a,b) window per clip."""
    out = {}
    for a, b in windows:
        per_clip = metric_nt[:, a:b].mean(axis=1)
        out[f"+{a+1}-{b}"] = bootstrap_ci(per_clip)
    return out


def report(pred: np.ndarray, true: np.ndarray, windows=None, do_lpips=True, label="") -> Dict:
    """Full per-horizon report. pred/true (N,T,1,H,W) in [0,1] (already trimmed to the horizon, gt-aligned)."""
    T = pred.shape[1]
    if windows is None:
        windows = [(0, min(10, T)), (10, min(25, T)), (25, T)] if T > 25 else [(0, T)]
        windows = [w for w in windows if w[1] > w[0]]
    res = {"label": label, "T": T,
           "fg_mse": windowed(fg_mse(pred, true), windows),
           "centroid_l2": windowed(centroid_l2(pred, true), windows),
           "iou": windowed(digit_iou(pred, true), windows)}
    if do_lpips:
        try:
            res["fg_lpips"] = windowed(fg_lpips(pred, true), windows)
        except Exception as e:
            res["fg_lpips_error"] = str(e)
    return res


def print_report(res: Dict):
    print(f"\n[{res['label']}] horizon={res['T']}")
    for met in ["fg_mse", "centroid_l2", "iou", "fg_lpips"]:
        if met not in res:
            continue
        cells = "  ".join(f"{k} {v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for k, v in res[met].items())
        print(f"  {met:>11}: {cells}")
