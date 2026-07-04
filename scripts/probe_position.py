"""Relative-displacement decodability probe (v2 — the v1 absolute probe was invalid: bg phases
are random PER WORLD, so absolute position is unregressable across clips by design; what
slot-addressing needs is WITHIN-clip relative displacement between two frames of the same world).

For frame pairs (t1, t2) of the same clip: can Delta-p = (p2-p1) mod 768 be decoded from the two
frozen-AE coarse latents? Features [z1, z2, z1*z2] (768-d; the product term carries phase
comparison), ridge -> sin/cos(2*pi*k*dp/W) at k=1 (absolute displacement — what beyond-192px
addressing needs) and k=4 (the 192px-aliased harmonic).
Expected: hgrid k=4 good / k=1 ~0 (aliased bg); a FIXED dataset must show k=1 >> 0.

  venv/Scripts/python.exe scripts/probe_position.py [--paths a.npy b.npy ...]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hvp_rev import hier_convae                                   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PAIRS = 6000
W = 768


def encode_frames(ae, clips, ci, ti):
    feats = np.empty((len(ci), 256), np.float32)
    with torch.no_grad():
        for lo in range(0, len(ci), 256):
            x = torch.from_numpy(np.stack([clips[c, t] for c, t in zip(ci[lo:lo + 256], ti[lo:lo + 256])])
                                 .astype(np.float32) / 127.5 - 1.0).unsqueeze(1).to(DEV)
            zc, _ = ae.encode(x)
            feats[lo:lo + 256] = zc.reshape(x.shape[0], -1).cpu().numpy()
    return feats


def probe(ae, clips_path):
    clips = np.load(clips_path, mmap_mode="r")
    pos = np.load(clips_path.replace(".npy", "_positions.npy"))
    rng = np.random.RandomState(0)
    N, T = clips.shape[0], clips.shape[1]
    ci = rng.randint(0, N, N_PAIRS)
    t1 = rng.randint(0, T, N_PAIRS)
    t2 = rng.randint(0, T, N_PAIRS)
    z1 = encode_frames(ae, clips, ci, t1)
    z2 = encode_frames(ae, clips, ci, t2)
    X = np.concatenate([z1, z2, z1 * z2], 1)                      # (N, 768)
    dp = (pos[ci, t2].astype(np.int64) - pos[ci, t1]) % W
    ntr = N_PAIRS // 2
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xtr, Xte = (X[:ntr] - mu) / sd, (X[ntr:] - mu) / sd
    A = Xtr.T @ Xtr + 10.0 * np.eye(X.shape[1])
    out = {}
    for k in (1, 4):
        y = np.stack([np.sin(2 * np.pi * k * dp / W), np.cos(2 * np.pi * k * dp / W)], 1)
        w = np.linalg.solve(A, Xtr.T @ y[:ntr])
        pred = Xte @ w
        ss_res = ((y[ntr:] - pred) ** 2).sum()
        ss_tot = ((y[ntr:] - y[ntr:].mean(0)) ** 2).sum()
        out[k] = 1.0 - ss_res / ss_tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["data/hgrid_train_1200x480.npy",
                                                   "data/hgrid2_train_1200x480.npy"])
    args = ap.parse_args()
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    for path in args.paths:
        r = probe(ae, path)
        print(f"{path}: rel-displacement R^2  k=1(absolute) {r[1]:+.3f}   k=4(mod-192) {r[4]:+.3f}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
