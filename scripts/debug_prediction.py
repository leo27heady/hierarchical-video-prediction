"""Diagnose WHY single-step predictions decode to blobs.

For a checkpoint, decode the next-frame prediction several ways and compare PSNR + a visual strip,
to separate the candidate causes:
  - recon         : decode GT latents (sanity; should be sharp = tokenizer is fine)
  - pred_mean     : all predicted stages, mixture MEAN (averages multimodal futures -> blob if multimodal)
  - pred_mode     : all predicted stages, mixture MODE (sharper than mean if multimodal)
  - pred_sample   : all predicted stages, one stochastic draw (sharp if the head is well-formed)
  - pred_bot_orac : predicted BOT (1:1) + GT next-frame coarse latents (isolates the finest stage;
                    sharp here but blob in pred_mean => the predicted COARSE stages / multi-rate
                    decode are the culprit, not BOT)

Rows of the saved strip (top->bottom): GT_next, recon, pred_mean, pred_mode, pred_sample, pred_bot_orac.

Run:  venv\\Scripts\\python.exe scripts\\debug_prediction.py --ckpt <ckpt> --out logs/debug_pred
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
from hvp.metrics import frames_to_strip
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.rollout import _pred_to_latents
from hvp.tokenizer import HierTokenizer


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-12 else 20 * math.log10(2.0) - 10 * math.log10(mse)


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Config.from_dict(blob["config"]) if "config" in blob else Config(
        topology=TopologyConfig(**blob["topo"]), predictor=PredictorConfig())
    tok = HierTokenizer(cfg.topology).to(device).eval(); tok.load_state_dict(blob["model"])
    cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).eval(); cas.load_state_dict(blob["cascade"])
    return tok, cas, cfg


@torch.no_grad()
def run(ckpt, n_clips=4, out_dir="logs/debug_pred", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cas, cfg = load(ckpt, device)
    n = cfg.topology.n_stages()
    print(f"[debug] ckpt={ckpt}  head={cfg.predictor.head}  image_size={cfg.topology.image_size}  "
          f"data train={cfg.data.train_hash}")
    loader = DataLoader(dataset_from_cfg(cfg, "val", max_clips=n_clips), batch_size=min(4, n_clips), num_workers=0)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dec = lambda zlist: tok.decoder.decode_depth(zlist, n).float().clamp(-1, 1)

    agg = {k: [] for k in ["recon", "oracle_full", "pred_mean", "pred_bot_cur", "pred_bot_orac"]}
    saved = 0
    for batch in loader:
        x = batch["video"].to(device)
        gt_next = x[:, :, 1:]
        z = [zi.float() for zi in tok.encoder(x)]
        preds = [p.float() for p in cas(z)]
        z_next = [torch.cat([zi[:, :, 1:], zi[:, :, -1:]], dim=2) for zi in z]   # GT next-frame latents

        pmean = _pred_to_latents(cas, preds, use_mean=True)
        variants = {
            "recon": dec(z),
            "oracle_full": dec(z_next),                                          # decode PERFECT next latents (ceiling)
            "pred_mean": dec(pmean),                                             # all stages predicted (mean)
            "pred_bot_cur": dec([pmean[0]] + [z[s] for s in range(1, n)]),       # pred BOT + CURRENT-time coarse (fix-1 probe)
            "pred_bot_orac": dec([pmean[0]] + z_next[1:]),                       # pred BOT + FUTURE-oracle coarse
        }
        # recon aligns to its own frames; the rest predict t->t+1 so compare [:-1] vs gt_next
        agg["recon"].append(_psnr(variants["recon"][:, :, 1:], gt_next))
        for k in ["oracle_full", "pred_mean", "pred_bot_cur", "pred_bot_orac"]:
            agg[k].append(_psnr(variants[k][:, :, :-1], gt_next))

        for j in range(x.shape[0]):
            rows = [frames_to_strip(gt_next[j]),
                    frames_to_strip(variants["recon"][j, :, 1:]),
                    frames_to_strip(variants["oracle_full"][j, :, :-1]),
                    frames_to_strip(variants["pred_mean"][j, :, :-1]),
                    frames_to_strip(variants["pred_bot_cur"][j, :, :-1]),
                    frames_to_strip(variants["pred_bot_orac"][j, :, :-1])]
            _save(np.concatenate(rows, axis=0), out / f"debug_{saved:02d}.png")
            saved += 1

    print("  rows: GT / recon / ORACLE(perfect next) / pred_mean(all) / pred_bot+CURRENT-coarse / pred_bot+oracle-coarse")
    for k, v in agg.items():
        print(f"  {k:16s} PSNR {sum(v)/max(len(v),1):6.2f} dB")
    print(f"  saved {saved} strips to {out}/")


def _save(img, path, scale=3):
    try:
        from PIL import Image
    except Exception:
        return
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--out", default="logs/debug_pred")
    args = ap.parse_args()
    run(args.ckpt, n_clips=args.n_clips, out_dir=args.out)


if __name__ == "__main__":
    main()
