"""Generate entropy/uncertainty maps from a trained checkpoint.

Rebuilds the tokenizer + cascade from the SAVED config in the checkpoint (so GMM / big / 128
configs load correctly — unlike a hardcoded default), runs the single-step predictor on held-out
clips, and saves per-clip strips stacked vertically:
    row 1: GT next frames
    row 2: predicted next frames (mixture mode)
    row 3: B&W uncertainty  (WHITE = certain, BLACK = uncertain)

Uncertainty = per-pixel std across ``--samples`` stochastic draws from the distributional head.
Requires a distributional head (gmm/gaussian); an l2 checkpoint yields an all-white (degenerate) map.

Run:  venv\\Scripts\\python.exe scripts\\uncertainty_map.py --ckpt logs/so3_gmm/oneshot/last.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hvp.config import Config, PredictorConfig, TopologyConfig
from hvp.data import dataset_from_cfg
from hvp.eval.uncertainty import gray_to_strip, predict_with_uncertainty, uncertainty_to_gray
from hvp.metrics import frames_to_strip
from hvp.predictor.cascade import HierarchicalCascade
from hvp.tokenizer import HierTokenizer


def load_from_ckpt(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    if "config" in blob:
        cfg = Config.from_dict(blob["config"])
    else:  # legacy checkpoint: only topo saved -> default predictor (l2)
        cfg = Config(topology=TopologyConfig(**blob["topo"]), predictor=PredictorConfig())
    tok = HierTokenizer(cfg.topology).to(device).eval()
    tok.load_state_dict(blob["model"])
    cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).eval()
    cas.load_state_dict(blob["cascade"])
    return tok, cas, cfg


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-12 else 20 * math.log10(2.0) - 10 * math.log10(mse)


@torch.no_grad()
def run(ckpt, n_clips=8, samples=24, std_scale=None, teacher_coarse=False, out_dir="logs/uncertainty", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cas, cfg = load_from_ckpt(ckpt, device)
    head = cfg.predictor.head
    if head == "l2":
        print("[warn] checkpoint head is 'l2' (deterministic) -> uncertainty map will be all-white.")
    loader = DataLoader(dataset_from_cfg(cfg, "val", max_clips=n_clips), batch_size=min(8, n_clips), num_workers=0)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    saved, mean_unc, pp = 0, [], []
    for batch in loader:
        x = batch["video"].to(device)
        mode_pred, std = predict_with_uncertainty(tok, cas, x, n_samples=samples, teacher_coarse=teacher_coarse)
        pred, gt_next, std_v = mode_pred[:, :, :-1], x[:, :, 1:], std[:, :-1]
        pp.append(_psnr(pred, gt_next))
        mean_unc.append(float(std_v.mean()))
        gray = uncertainty_to_gray(std_v, std_scale=std_scale)          # (B,T-1,H,W), 1=white=certain
        for j in range(x.shape[0]):
            rows = [
                frames_to_strip(gt_next[j].clamp(-1, 1)),
                frames_to_strip(pred[j].clamp(-1, 1)),
                gray_to_strip(gray[j]),
            ]
            _save(np.concatenate(rows, axis=0), out / f"uncertainty_{saved:02d}.png")
            saved += 1
    print(f"[uncertainty] head={head} samples={samples} teacher_coarse={teacher_coarse}")
    print(f"  mean per-pixel std (uncertainty): {sum(mean_unc)/max(len(mean_unc),1):.4f}")
    print(f"  predicted-frame PSNR (mode)     : {sum(pp)/max(len(pp),1):.2f} dB")
    print(f"  saved {saved} maps to {out}/")


def _save(img: np.ndarray, path: Path, scale=3):
    try:
        from PIL import Image
    except Exception:
        return
    im = Image.fromarray(img)
    im = im.resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/so3_gmm/oneshot/last.pt")
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--std-scale", type=float, default=None, help="fixed normalization; default self-calibrates to 99th pct")
    ap.add_argument("--teacher-coarse", action="store_true", help="predicted BOT + GT coarse (isolate finest stage)")
    ap.add_argument("--out", default="logs/uncertainty")
    args = ap.parse_args()
    run(args.ckpt, n_clips=args.n_clips, samples=args.samples, std_scale=args.std_scale,
        teacher_coarse=args.teacher_coarse, out_dir=args.out)


if __name__ == "__main__":
    main()
