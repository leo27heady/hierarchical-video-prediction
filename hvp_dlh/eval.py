"""Open-loop evaluation for DLH — decision metrics are foreground-MSE + foreground-LPIPS (perceptual),
NEVER aggregate SSIM alone (MMNIST is 93% background; a black frame scores SSIM 0.76). best-of-N is
selected by the DECISION metric (fg-MSE), not SSIM. LPIPS is computed ONCE on the kept sample with a
spatial map + foreground mask. Per-level KL and K (teacher-forced) are reported so the hierarchy is
tracked every eval. Trivial baselines (black, persist) are always reported."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_m

_LPIPS = None


def _get_lpips(device):
    """Lazy AlexNet-LPIPS with a SPATIAL distance map (B,1,H,W) so it can be foreground-masked."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", spatial=True).to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    return _LPIPS


@torch.no_grad()
def _lpips_maps(pred, gt, device, chunk=256):
    """pred,gt (M,1,64,64) in [0,1] -> spatial LPIPS map (M,1,64,64). Chunked to bound memory."""
    net = _get_lpips(device)
    out = []
    for i in range(0, pred.shape[0], chunk):
        p = (pred[i:i + chunk] * 2 - 1).repeat(1, 3, 1, 1)
        g = (gt[i:i + chunk] * 2 - 1).repeat(1, 3, 1, 1)
        d = net(p.to(device), g.to(device))                       # (b,1,64,64) (spatial=True)
        out.append(d.cpu())
    return torch.cat(out, 0)


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def _collect_clips(test, n_clips, total_len):
    clips = []
    if hasattr(test, "__len__"):
        for i in range(min(n_clips, len(test))):
            clips.append(test[i][:total_len])
    else:
        it = iter(test)
        for _ in range(n_clips):
            clips.append(next(it)[:total_len])
    return torch.stack(clips, 0)


@torch.no_grad()
def evaluate(model, test, cfg, n_clips=16, n_samples=10, device="cuda", seed=0, with_lpips=True):
    """One open-loop eval. best-of-N by per-clip fg-MSE (the decision metric). Returns per-frame curves
    + scalar metrics incl. foreground-LPIPS and teacher-forced per-level KL / K."""
    model.eval()
    torch.manual_seed(seed)                                       # reproducible best-of-N
    ctx, H = cfg.ctx_frames, cfg.eval_horizon
    clips = _collect_clips(test, n_clips, ctx + H).to(device)
    gt = clips[:, ctx:].cpu().numpy()                            # (N,H,C,h,w)
    Nc = clips.shape[0]
    fg = gt > 0.1                                                # foreground (digit/ball) mask

    # ---- best-of-N selected by per-clip foreground-MSE (the DECISION metric, not SSIM) ----
    best = np.full(Nc, np.inf)                                   # lower fg-MSE is better
    pred_keep = np.zeros_like(gt); K_keep = np.zeros((Nc, H))
    for _ in range(n_samples):
        pred, K = model.open_loop(clips, ctx, H)
        p = pred.cpu().numpy(); Kn = K.cpu().numpy()
        for c in range(Nc):
            m = ((p[c] - gt[c]) ** 2)[fg[c]].mean() if fg[c].any() else ((p[c] - gt[c]) ** 2).mean()
            if m < best[c]:
                best[c] = m; pred_keep[c] = p[c]; K_keep[c] = Kn[c]

    # ---- per-frame curves on the kept sample ----
    ssim_c = np.array([[ssim_m(gt[c, t, 0], pred_keep[c, t, 0], data_range=1.0) for t in range(H)] for c in range(Nc)])
    psnr_c = np.array([[_psnr(gt[c, t], pred_keep[c, t]) for t in range(H)] for c in range(Nc)])
    fgmse_c = np.array([[((pred_keep[c, t] - gt[c, t]) ** 2)[fg[c, t]].mean() if fg[c, t].any() else 0.0
                         for t in range(H)] for c in range(Nc)])
    persist = np.repeat(clips[:, ctx - 1:ctx].cpu().numpy(), H, axis=1)

    def fgm(arr):  # foreground-MSE scalar over all clips/frames
        return float(((arr - gt) ** 2)[fg].mean()) if fg.any() else 0.0

    m = {
        "fg_mse": float(fgmse_c.mean()), "fg_mse_black": fgm(np.zeros_like(gt)), "fg_mse_persist": fgm(persist),
        "ssim": float(ssim_c.mean()), "psnr": float(psnr_c.mean()),
        "ssim_black_baseline": float(np.mean([[ssim_m(gt[c, t, 0], np.zeros_like(gt[c, t, 0]), data_range=1.0)
                                               for t in range(H)] for c in range(Nc)])),
        "pred_mean_intensity": float(pred_keep.mean()), "gt_mean_intensity": float(gt.mean()),
        "K_eval_mean": float(K_keep.mean()),
        "fg_mse_curve": fgmse_c.mean(0), "ssim_curve": ssim_c.mean(0), "K_curve": K_keep.mean(0),
    }
    # decision horizons (in training range) + diagnostics
    for h in [25, 50, 100, 150]:
        if h <= H:
            m[f"fg_mse@{h}"] = float(fgmse_c[:, h - 1].mean())

    # ---- foreground-LPIPS (perceptual decision metric), computed ONCE on the kept sample ----
    if with_lpips:
        fg2d = fg[:, :, 0]                                                 # (N,H,64,64) drop channel for masking
        gtt = torch.from_numpy(gt.reshape(-1, 1, 64, 64)).float()
        def fg_lpips(arr):                                                 # spatial LPIPS map, fg-masked
            dm = _lpips_maps(torch.from_numpy(arr.reshape(-1, 1, 64, 64)).float(), gtt, device)
            dm = dm.numpy().reshape(Nc, H, 64, 64)
            return dm, float((dm * fg2d).sum() / max(fg2d.sum(), 1))
        dmap, m["lpips_fg"] = fg_lpips(pred_keep)
        m["lpips_full"] = float(dmap.mean())
        _, m["lpips_fg_black"] = fg_lpips(np.zeros_like(gt))               # sanity: must be worse than model
        _, m["lpips_fg_persist"] = fg_lpips(persist)
        m["lpips_fg_curve"] = np.array([(dmap[:, t] * fg2d[:, t]).sum() / max(fg2d[:, t].sum(), 1) for t in range(H)])

    # ---- hierarchy: teacher-forced per-level KL + K (so every eval tracks the clock) ----
    _, tm = model.loss(clips[:min(8, Nc)], beta=1.0)
    m["skl_per_level"] = tm["skl_per_level"].cpu().tolist()
    m["K_mean_tf"] = float(tm["K_mean"])
    m["e_per_level"] = tm["e_per_level"].cpu().tolist()

    # ---- GT-vs-pred strip ----
    idx = [i for i in [0, H // 8, H // 4, H // 2, 3 * H // 4, H - 1] if i < H]
    strip = [np.concatenate([np.concatenate([gt[c, t, 0], pred_keep[c, t, 0]], 0) for t in idx], 1)
             for c in range(min(3, Nc))]
    m["strip"] = np.concatenate(strip, 0)
    model.train()
    return m
