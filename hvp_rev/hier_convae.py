"""2-level hierarchical conv-AE (minimal MilliVid-style): a shared encoder taps a COARSE (4x4) and a FINE (8x8) latent,
each INDEPENDENTLY decodable to the full 64x64 frame (MilliVid's per-level-decodability property, minimal version).
Trained with random-level dropout. The coarse level is the low-entropy "layout/pan" code; the fine level is texture.

  venv/Scripts/python.exe -m hvp_rev.hier_convae --train --data data/loop_train_3000x100.npy --out logs_rhcp/hier_ae.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.convae import FramePool

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAT = 16


def _dn(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU())
def _up(ci, co): return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU())


class HierConvAE(nn.Module):
    def __init__(self, img_ch=1, w=64):
        super().__init__()
        self.enc = nn.Sequential(_dn(img_ch, w), _dn(w, w * 2), _dn(w * 2, w * 2))     # 64->32->16->8
        self.fine_lat = nn.Conv2d(w * 2, LAT, 1)                                        # 8x8
        self.coarse_enc = _dn(w * 2, w * 2)                                             # 8->4
        self.coarse_lat = nn.Conv2d(w * 2, LAT, 1)                                      # 4x4
        # decoders (each level -> full 64x64)
        self.fine_from = nn.Conv2d(LAT, w * 2, 1)
        self.fine_dec = nn.Sequential(_up(w * 2, w * 2), _up(w * 2, w), _up(w, w))      # 8->16->32->64
        self.fine_head = nn.Conv2d(w, img_ch, 3, padding=1)
        self.coarse_from = nn.Conv2d(LAT, w * 2, 1)
        self.coarse_dec = nn.Sequential(_up(w * 2, w * 2), _up(w * 2, w * 2), _up(w * 2, w), _up(w, w))  # 4->8->16->32->64
        self.coarse_head = nn.Conv2d(w, img_ch, 3, padding=1)

    def encode(self, x):
        f = self.enc(x)
        z_fine = torch.tanh(self.fine_lat(f))                       # (B,LAT,8,8)
        z_coarse = torch.tanh(self.coarse_lat(self.coarse_enc(f)))  # (B,LAT,4,4)
        return z_coarse, z_fine

    def decode_fine(self, z):
        return torch.tanh(self.fine_head(self.fine_dec(self.fine_from(z))))

    def decode_coarse(self, z):
        return torch.tanh(self.coarse_head(self.coarse_dec(self.coarse_from(z))))

    def forward(self, x):
        zc, zf = self.encode(x)
        return self.decode_coarse(zc), self.decode_fine(zf), zc, zf


def train(path, out, steps=7000, batch=64, lr=3e-4):
    torch.manual_seed(0)
    ae = HierConvAE().to(DEV)
    data = FramePool(path)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    print(f"HierConvAE train: {path} params={sum(p.numel() for p in ae.parameters())/1e6:.2f}M "
          f"coarse 4x4 / fine 8x8", flush=True)
    run = 0.0
    for s in range(1, steps + 1):
        x = data.batch(batch)
        rc, rf, _, _ = ae(x)
        # random-level dropout: supervise ONE level per step (coarse up-weighted so it isn't ignored)
        if torch.rand(()) < 0.5:
            loss = 4.0 * (F.l1_loss(rc, x) + F.mse_loss(rc, x))
        else:
            loss = F.l1_loss(rf, x) + F.mse_loss(rf, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += float(loss.detach())
        if s % 1000 == 0:
            with torch.no_grad():
                xv = data.batch(256, train=False)
                rc, rf, _, _ = ae(xv)
                vc = F.mse_loss(rc, xv).item(); vf = F.mse_loss(rf, xv).item()
            print(f"  [step {s:5d}] loss {run/1000:.4f}  val recon coarse {vc:.4f} fine {vf:.4f}", flush=True)
            run = 0.0
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": ae.state_dict()}, out)
    print(f"saved {out}")


def load(path):
    ae = HierConvAE().to(DEV)
    ae.load_state_dict(torch.load(path, map_location=DEV)["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="data/loop_train_3000x100.npy")
    ap.add_argument("--out", default="logs_rhcp/hier_ae.pt")
    ap.add_argument("--steps", type=int, default=7000)
    args = ap.parse_args()
    if args.train:
        train(args.data, args.out, steps=args.steps)
