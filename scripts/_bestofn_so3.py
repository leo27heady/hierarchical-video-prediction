"""Best-of-N evaluation: does the distributional (GMM) head's SAMPLING beat its single point
prediction? On genuinely multimodal SO(3) futures, sampling N rollouts and taking the oracle-best
(per clip+horizon) should beat the mode rollout — while an L2 head (identical samples) shows no gain.

Reports per-horizon changed-region PSNR for: mode rollout, best-of-N rollout, persistence, rotation.
Run on each checkpoint:  python scripts/_bestofn_so3.py --ckpt logs/so3_gmm/oneshot/last.pt --n 8
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from hvp.config import Config, TopologyConfig
from hvp.data import dataset_from_cfg
from hvp.eval.baselines import rotation_extrap
from hvp.eval.quality import changed_region_metrics
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.rollout import free_run
from hvp.tokenizer import build_tokenizer


def load(ckpt, device):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = Config.from_dict(blob["config"]) if "config" in blob else Config(topology=TopologyConfig(**blob["topo"]))
    tok = build_tokenizer(cfg).to(device).eval(); tok.load_state_dict(blob["model"])
    cas = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).eval(); cas.load_state_dict(blob["cascade"])
    return tok, cas, cfg


@torch.no_grad()
def run(ckpt, N, max_batches, hmax):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, cas, cfg = load(ckpt, dev)
    print(f"head={cfg.predictor.head} K={cfg.predictor.head_components}  ckpt={ckpt}")
    loader = DataLoader(dataset_from_cfg(cfg, "val_long"), batch_size=10, num_workers=0)
    seed_len = cfg.topology.sequence_length
    acc = None
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        clip = batch["video"].to(dev)
        n = min(hmax, clip.shape[2] - seed_len)
        if acc is None:
            acc = {k: np.zeros(n) for k in ("mode", "bestN", "persist", "rotation")}; nb = np.zeros(n)
        seed, gt = clip[:, :, :seed_len], clip[:, :, seed_len:seed_len + n]
        prev = clip[:, :, seed_len - 1: seed_len - 1 + n]
        mode = free_run(tok, cas, seed, n, "pixel", sample=False)
        samples = torch.stack([free_run(tok, cas, seed, n, "pixel", sample=True) for _ in range(N)])  # (N,B,3,n,H,W)
        err = ((samples - gt[None]) ** 2).mean(dim=(2, 4, 5))                 # (N,B,n)
        best_idx = err.argmin(0)                                             # (B,n)
        idx = best_idx[None, :, None, :, None, None].expand(1, *samples.shape[1:])
        best = samples.gather(0, idx)[0]                                     # (B,3,n,H,W)
        rot = rotation_extrap(seed, n)
        b = clip.shape[0]
        for h in range(n):
            acc["mode"][h] += changed_region_metrics(mode[:, :, h:h+1], gt[:, :, h:h+1], prev[:, :, h:h+1])["changed_psnr"] * b
            acc["bestN"][h] += changed_region_metrics(best[:, :, h:h+1], gt[:, :, h:h+1], prev[:, :, h:h+1])["changed_psnr"] * b
            acc["persist"][h] += changed_region_metrics(prev[:, :, h:h+1], gt[:, :, h:h+1], prev[:, :, h:h+1])["changed_psnr"] * b
            acc["rotation"][h] += changed_region_metrics(rot[:, :, h:h+1], gt[:, :, h:h+1], prev[:, :, h:h+1])["changed_psnr"] * b
            nb[h] += b
    print(f"\n  best-of-{N} changed-region PSNR (dB), SO(3) val_long:")
    print(f"  {'h':>3s} | {'mode':>6s} {'bestN':>6s} {'gain':>5s} | {'persist':>7s} {'rotation':>8s}")
    for h in range(len(nb)):
        m, bn = acc["mode"][h]/nb[h], acc["bestN"][h]/nb[h]
        print(f"  {h:>3d} | {m:6.2f} {bn:6.2f} {bn-m:+5.2f} | {acc['persist'][h]/nb[h]:7.2f} {acc['rotation'][h]/nb[h]:8.2f}")
    mm = np.array([acc['mode'][h]/nb[h] for h in range(len(nb))]).mean()
    bb = np.array([acc['bestN'][h]/nb[h] for h in range(len(nb))]).mean()
    print(f"\n  mean: mode {mm:.2f}  best-of-{N} {bb:.2f}  gain {bb-mm:+.2f} dB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=8, help="samples for best-of-N")
    ap.add_argument("--max-batches", type=int, default=6)
    ap.add_argument("--hmax", type=int, default=12)
    a = ap.parse_args()
    run(a.ckpt, a.n, a.max_batches, a.hmax)
