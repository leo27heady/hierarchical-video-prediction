"""Pixel-space single-step prediction eval (the anti-blob proof).

Encode held-out clips, run the cascade to predict next-frame latents, decode the predicted
latents to pixels, and compare predicted frame t to GT frame t+1 — against the persistence
baseline (copy frame t). A blob would lose to persistence and look like an average shape;
a working predictor beats persistence and tracks the rotation. Optionally saves a strip PNG.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hvp.config import Config, TopologyConfig
from hvp.data import dataset_from_cfg
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import HierTokenizer


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-12 else 20 * math.log10(2.0) - 10 * math.log10(mse)


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    topo = TopologyConfig(**blob["topo"])
    tok = HierTokenizer(topo).to(device).eval()
    tok.load_state_dict(blob["model"])
    from hvp.config import PredictorConfig
    cas = HierarchicalCascade(topo, PredictorConfig()).to(device).eval()
    cas.load_state_dict(blob["cascade"])
    return tok, cas, topo


@torch.no_grad()
def run(cfg: Config, ckpt: str, n_clips=32, strip_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cas, topo = load(ckpt, device)
    loader = DataLoader(dataset_from_cfg(cfg, "val", max_clips=n_clips), batch_size=8, num_workers=0)

    pred_psnrs, persist_psnrs = [], []
    strip_gt, strip_pred = [], []
    for batch in loader:
        x = batch["video"].to(device)                         # (B,3,9,32,32)
        z = tok.encoder(x)
        preds = cas(z, use_cascade=True)                      # per-stage (B,T,H,W,C)
        z_pred = [p.permute(0, 4, 1, 2, 3).contiguous() for p in preds]   # (B,C,T,H,W)
        recon_pred = tok.decoder.decode_depth(z_pred, topo.n_stages())     # (B,3,9,32,32)
        # bot is 1:1 temporal: predicted frame t ~ GT frame t+1, for t in 0..T-2
        p = recon_pred[:, :, :-1]
        gt_next = x[:, :, 1:]
        persist = x[:, :, :-1]
        pred_psnrs.append(_psnr(p, gt_next))
        persist_psnrs.append(_psnr(persist, gt_next))
        if len(strip_gt) == 0:
            strip_gt = gt_next[0].clamp(-1, 1).cpu()
            strip_pred = p[0].clamp(-1, 1).cpu()

    avg = lambda L: sum(L) / max(len(L), 1)
    pp, qq = avg(pred_psnrs), avg(persist_psnrs)
    print("=== single-step prediction (pixel space, held-out) ===")
    print(f"  predicted-frame PSNR : {pp:5.2f} dB")
    print(f"  persistence  PSNR    : {qq:5.2f} dB")
    print(f"  lift over persistence: {pp - qq:+5.2f} dB   -> {'BEATS persistence (no blob)' if pp > qq else 'LOSES (blob risk)'}")

    if strip_path is not None and len(strip_gt):
        _save_strip(strip_gt, strip_pred, strip_path)
        print(f"  strip saved: {strip_path}")
    return {"pred_psnr": pp, "persist_psnr": qq, "lift": pp - qq}


def _save_strip(gt, pred, path):
    try:
        from PIL import Image
    except Exception:
        return
    def to_row(v):  # v: (3,T,32,32) in [-1,1]
        v = ((v + 1) * 127.5).clamp(0, 255).byte().numpy()       # (3,T,32,32)
        T = v.shape[1]
        return np.concatenate([v[:, t].transpose(1, 2, 0) for t in range(T)], axis=1)
    img = np.concatenate([to_row(gt), to_row(pred)], axis=0)     # GT row over predicted row
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/oneshot/last.pt")
    ap.add_argument("--n-clips", type=int, default=32)
    ap.add_argument("--strip", default="logs/predict_strip.png")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    from hvp.config import apply_cli_overrides
    cfg = Config.from_dict(apply_cli_overrides({}, args.set))
    cfg.validate()
    run(cfg, args.ckpt, n_clips=args.n_clips, strip_path=args.strip)


if __name__ == "__main__":
    main()
