"""Simple robust conv autoencoder -> a frozen latent space for the EBT Stage-0b experiment.

Plain strided-conv encoder/decoder (64x64x1 <-> 8x8x C_lat). No token pyramid, no fancy attention -- this reliably
reconstructs (unlike the buggy tokenizer.py) and gives a small latent the EBT can predict next-frame in.

  venv/Scripts/python.exe -m hvp_rev.convae --train --data data/loop_train_3000x100.npy --out logs_rhcp/convae_loop.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class ConvAE(nn.Module):
    def __init__(self, img_ch=1, lat_ch=16, w=64):
        super().__init__()
        self.lat_ch, self.lat_hw = lat_ch, 8
        def dn(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU())
        def up(ci, co): return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.SiLU())
        self.enc = nn.Sequential(dn(img_ch, w), dn(w, w * 2), dn(w * 2, w * 2))   # 64->32->16->8
        self.to_lat = nn.Conv2d(w * 2, lat_ch, 1)
        self.from_lat = nn.Conv2d(lat_ch, w * 2, 1)
        self.dec = nn.Sequential(up(w * 2, w * 2), up(w * 2, w), up(w, w))        # 8->16->32->64
        self.head = nn.Conv2d(w, img_ch, 3, padding=1)

    def encode(self, x):
        return torch.tanh(self.to_lat(self.enc(x)))          # (B, lat_ch, 8, 8) in [-1,1]

    def decode(self, z):
        return torch.tanh(self.head(self.dec(self.from_lat(z))))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class FramePool:
    def __init__(self, path, n_val=4000):
        a = np.load(path).reshape(-1, 64, 64).astype(np.float32) / 127.5 - 1.0
        self.x = torch.from_numpy(a).unsqueeze(1)
        self.M = self.x.shape[0]
        self.n_val = n_val

    def batch(self, B, train=True):
        hi = self.M - self.n_val if train else self.M
        lo = 0 if train else self.M - self.n_val
        return self.x[torch.randint(lo, hi, (B,))].to(DEV)


def train(path, out, steps=6000, batch=64, lr=3e-4):
    torch.manual_seed(0)
    ae = ConvAE().to(DEV)
    data = FramePool(path)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    print(f"convae train: {path} frames={data.M} params={sum(p.numel() for p in ae.parameters())/1e6:.2f}M "
          f"latent {ae.lat_ch}x8x8", flush=True)
    run = 0.0
    for s in range(1, steps + 1):
        x = data.batch(batch)
        r, _ = ae(x)
        loss = F.l1_loss(r, x) + F.mse_loss(r, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += float(loss.detach())
        if s % 500 == 0:
            with torch.no_grad():
                xv = data.batch(256, train=False)
                vr = F.mse_loss(ae(xv)[0], xv).item()
            print(f"  [step {s:5d}] train {run/500:.4f}  val-reconMSE {vr:.4f}", flush=True)
            run = 0.0
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": ae.state_dict()}, out)
    print(f"saved {out}")
    return ae


def load(path):
    ae = ConvAE().to(DEV)
    ae.load_state_dict(torch.load(path, map_location=DEV)["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="data/loop_train_3000x100.npy")
    ap.add_argument("--out", default="logs_rhcp/convae_loop.pt")
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    if args.train:
        train(args.data, args.out, steps=args.steps)
