"""Open-loop evaluation for DLH — best-of-N SSIM/PSNR PLUS the honest metrics the project learned to
require: foreground-MSE (digit/ball pixels) and trivial baselines (black, persist), so SSIM on
mostly-black frames is never read in isolation. Also returns a GT-vs-prediction strip + K trajectory."""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_m


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def _collect_clips(test, n_clips, total_len):
    """Return (n_clips, total_len, C, H, W) float tensor from a Dataset or IterableDataset."""
    clips = []
    if hasattr(test, "__len__"):                                    # map-style Dataset (MMNIST)
        for i in range(min(n_clips, len(test))):
            clips.append(test[i][:total_len])
    else:                                                           # IterableDataset (Moving Ball)
        it = iter(test)
        for _ in range(n_clips):
            clips.append(next(it)[:total_len])
    return torch.stack(clips, 0)


@torch.no_grad()
def evaluate(model, test, cfg, n_clips=16, n_samples=10, device="cuda"):
    model.eval()
    ctx, H = cfg.ctx_frames, cfg.eval_horizon
    clips = _collect_clips(test, n_clips, ctx + H).to(device)            # (N,ctx+H,C,h,w)
    gt = clips[:, ctx:].cpu().numpy()                                    # (N,H,C,h,w)
    Nc = clips.shape[0]

    # ---- CW-VAE/SVG best-of-N (oracle over samples, by mean SSIM on luminance) ----
    best = np.full(Nc, -1.0)
    ssim_best = np.zeros((Nc, H)); psnr_best = np.zeros((Nc, H))
    pred_keep = np.zeros_like(gt); K_keep = np.zeros((Nc, H))
    for _ in range(n_samples):
        pred, K = model.open_loop(clips, ctx, H)                        # (N,H,C,h,w), (N,H)
        p = pred.cpu().numpy(); Kn = K.cpu().numpy()
        for c in range(Nc):
            ss = np.array([ssim_m(gt[c, t, 0], p[c, t, 0], data_range=1.0) for t in range(H)])
            if ss.mean() > best[c]:
                best[c] = ss.mean(); ssim_best[c] = ss
                psnr_best[c] = np.array([_psnr(gt[c, t], p[c, t]) for t in range(H)])
                pred_keep[c] = p[c]; K_keep[c] = Kn[c]

    # ---- honest metrics ----
    fg = gt > 0.1
    fg_mse = float(((pred_keep - gt) ** 2)[fg].mean()) if fg.any() else 0.0
    black_fg = float((gt ** 2)[fg].mean()) if fg.any() else 0.0
    persist = np.repeat(clips[:, ctx - 1:ctx].cpu().numpy(), H, axis=1)
    persist_fg = float(((persist - gt) ** 2)[fg].mean()) if fg.any() else 0.0
    black_ssim = float(np.mean([[ssim_m(gt[c, t, 0], np.zeros_like(gt[c, t, 0]), data_range=1.0)
                                 for t in range(H)] for c in range(Nc)]))

    metrics = {
        "ssim": float(ssim_best.mean()), "psnr": float(psnr_best.mean()),
        "ssim_f1": float(ssim_best[:, 0].mean()), "ssim_fmid": float(ssim_best[:, H // 2].mean()),
        "ssim_flast": float(ssim_best[:, -1].mean()),
        "fg_mse": fg_mse, "fg_mse_black": black_fg, "fg_mse_persist": persist_fg,
        "ssim_black_baseline": black_ssim,
        "K_eval_mean": float(K_keep.mean()),
        "pred_mean_intensity": float(pred_keep.mean()), "gt_mean_intensity": float(gt.mean()),
        "ssim_curve": ssim_best.mean(0), "K_curve": K_keep.mean(0),
    }
    # ---- a GT-vs-pred strip at chosen horizons (for wandb image) ----
    idx = [i for i in [0, H // 8, H // 4, H // 2, 3 * H // 4, H - 1] if i < H]
    strip = []
    for c in range(min(3, Nc)):
        row = np.concatenate([np.concatenate([gt[c, t, 0], pred_keep[c, t, 0]], 0) for t in idx], 1)
        strip.append(row)
    metrics["strip"] = np.concatenate(strip, 0)                          # GT-over-pred, rows=clips
    model.train()
    return metrics
