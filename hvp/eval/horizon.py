"""Per-horizon rollout evaluation (P1) — make prediction-horizon drift a *curve*.

Seeds each long clip with the first ``seed_len`` frames, free-runs the cascade to the end,
and reports per-horizon PSNR/SSIM/changed-region for: AR (pixel + latent feedback),
teacher-forced (best-case single-step with GT context), and persistence (frozen last seed
frame). Also the TF→AR gap, the horizon where AR stops beating persistence, and FVD.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from hvp.eval.quality import changed_region_metrics, fvd
from hvp.metrics import frames_to_strip, ssim, video_ssim
from hvp.predictor.rollout import free_run, teacher_forced_at


def _psnr_from_mse(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 20 * math.log10(2.0) - 10 * math.log10(mse)


class _Acc:
    def __init__(self, n: int):
        self.sse = np.zeros(n); self.cnt = np.zeros(n)      # exact MSE accumulation -> PSNR
        self.ssim = np.zeros(n); self.cpsnr = np.zeros(n)   # batch-weighted sums
        self.nb = np.zeros(n)

    def add(self, pred, gt, prev):
        n = pred.shape[2]
        for h in range(n):
            p, g = pred[:, :, h], gt[:, :, h]
            b = p.shape[0]
            self.sse[h] += float(((p - g) ** 2).sum()); self.cnt[h] += p.numel()
            self.ssim[h] += ssim(p, g) * b
            cr = changed_region_metrics(p[:, :, None], g[:, :, None], prev[:, :, h, None])
            self.cpsnr[h] += cr["changed_psnr"] * b
            self.nb[h] += b

    def psnr(self): return np.array([_psnr_from_mse(s / max(c, 1)) for s, c in zip(self.sse, self.cnt)])
    def ssim_(self): return self.ssim / np.maximum(self.nb, 1)
    def changed_psnr(self): return self.cpsnr / np.maximum(self.nb, 1)


@torch.no_grad()
def horizon_eval(tok, cascade, loader, cfg, device, modes=("pixel", "latent"),
                 max_batches: Optional[int] = None, lpips_metric=None,
                 strip_path: Optional[str] = None) -> Dict:
    seed_len = cfg.topology.sequence_length
    keys = list(modes) + ["tf", "persist"]
    acc: Dict[str, _Acc] = {}
    feats: Dict[str, List] = {**{m: [] for m in modes}, "gt": []}
    n_steps = None
    strip_rows = None

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        clip = batch["video"].to(device)
        T = clip.shape[2]
        n = T - seed_len
        if n_steps is None:
            n_steps = n
            acc = {k: _Acc(n) for k in keys}
        seed = clip[:, :, :seed_len]
        gt_future = clip[:, :, seed_len:]
        prev = clip[:, :, seed_len - 1: T - 1]
        persist = clip[:, :, seed_len - 1: seed_len].repeat(1, 1, n, 1, 1)

        preds = {m: free_run(tok, cascade, seed, n, m) for m in modes}
        preds["tf"] = torch.stack([teacher_forced_at(tok, cascade, clip, seed_len, h) for h in range(n)], dim=2)
        preds["persist"] = persist
        for k, p in preds.items():
            acc[k].add(p, gt_future, prev)
        for m in modes:
            feats[m].append(preds[m].cpu())
        feats["gt"].append(gt_future.cpu())
        if strip_path and strip_rows is None:
            hs = list(range(0, n, max(1, n // 8)))[:8]
            strip_rows = [("GT", frames_to_strip(gt_future[0][:, hs]))]
            for m in modes:
                strip_rows.append((m, frames_to_strip(preds[m][0][:, hs])))

    res: Dict = {"n_steps": n_steps, "seed_len": seed_len}
    for k in keys:
        res[f"psnr/{k}"] = acc[k].psnr().tolist()
        res[f"ssim/{k}"] = acc[k].ssim_().tolist()
        res[f"changed_psnr/{k}"] = acc[k].changed_psnr().tolist()

    # TF->AR gap (per horizon) + summaries for the primary (pixel) mode
    primary = modes[0]
    tf_psnr = np.array(res["psnr/tf"]); ar_psnr = np.array(res[f"psnr/{primary}"])
    persist_psnr = np.array(res["psnr/persist"])
    res["tf_ar_gap"] = (tf_psnr - ar_psnr).tolist()
    res["tf_ar_gap_mean"] = float((tf_psnr - ar_psnr).mean())
    beats = np.where(ar_psnr <= persist_psnr)[0]
    res["beats_persistence_until"] = int(beats[0]) if len(beats) else int(n_steps)
    res["ar_minus_persist_mean"] = float((ar_psnr - persist_psnr).mean())

    # FVD over rolled clips (best-effort)
    try:
        gt_all = torch.cat(feats["gt"], 0)
        for m in modes:
            res[f"fvd/{m}"] = fvd(gt_all, torch.cat(feats[m], 0), device=device)
    except Exception as e:
        res["fvd_error"] = str(e)

    if lpips_metric is not None:
        try:
            res[f"lpips/{primary}"] = lpips_metric(torch.cat(feats[primary], 0).to(device), torch.cat(feats["gt"], 0).to(device))
        except Exception as e:
            res["lpips_error"] = str(e)

    if strip_path and strip_rows is not None:
        _save_strip(strip_rows, strip_path)
        res["strip"] = strip_path
    return res


def _save_strip(rows, path):
    try:
        from pathlib import Path

        from PIL import Image
    except Exception:
        return
    img = np.concatenate([r for _, r in rows], axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(path)
