"""3rd (TOP, 2x2) AE level as a FROZEN-TRUNK EXTENSION of the existing 2-level HierConvAE.

Design constraint (pre-registered, CW-style-table program 2026-07-10): the existing frozen AE
(logs_rhcp/hier_ae_mom.pt) must stay bit-identical -- every measured arm lives in its latent
space. So the top level is a separate branch trained ON TOP of the frozen trunk:

    x -> [frozen] enc (8x8 feats) -> [frozen] coarse_enc (4x4 feats) -> top_enc (2x2) -> top_lat
    zt (B,LAT,2,2), tanh, independently decodable to 64x64 (per-level decodability preserved).

Token cost: 2x2 = 4 tokens/frame (vs coarse 16, fine 64) -- the factor-2 spatial pyramid
2x2 / 4x4 / 8x8 for the depth-3 arm (hier3) at matched budget.

  # train the top branch (GPU, ~10-15 min; base AE frozen throughout):
  venv/Scripts/python.exe -m hvp_rev.hier3 --train --base logs_rhcp/hier_ae_mom.pt \
      --data data/loopmom_train_3000x100.npy --out logs_rhcp/hier_ae_mom_top.pt

  # smoke (CPU ok): construction + shapes + frozen-trunk assert
  venv/Scripts/python.exe -m hvp_rev.hier3 --smoke
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.convae import FramePool
from hvp_rev.hier_convae import LAT, HierConvAE, _dn, _up
from hvp_rev.hier_convae import load as load_base

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOP_HW = 2  # (B,LAT,2,2) -> 4 tokens/frame


class TopBranch(nn.Module):
    """Encoder tap 4x4 -> 2x2 top latent + independent decoder to 64x64 (per-level decodability)."""

    def __init__(self, img_ch=1, w=64):
        super().__init__()
        self.top_enc = _dn(w * 2, w * 2)                                             # 4 -> 2
        self.top_lat = nn.Conv2d(w * 2, LAT, 1)                                      # 2x2
        self.top_from = nn.Conv2d(LAT, w * 2, 1)
        self.top_dec = nn.Sequential(_up(w * 2, w * 2), _up(w * 2, w * 2),
                                     _up(w * 2, w * 2), _up(w * 2, w), _up(w, w))    # 2->64
        self.top_head = nn.Conv2d(w, img_ch, 3, padding=1)

    def encode_from_c4(self, c4):
        """c4 = frozen trunk's coarse_enc output (B,w*2,4,4) -> zt (B,LAT,2,2)."""
        return torch.tanh(self.top_lat(self.top_enc(c4)))

    def decode_top(self, zt):
        return torch.tanh(self.top_head(self.top_dec(self.top_from(zt))))


class TopExtAE(nn.Module):
    """Frozen 2-level base + trainable/frozen top branch. encode() stays 2-level (drop-in for the
    existing harness); encode3() adds zt. decode_fine/decode_coarse delegate to the base so
    decode01/metrics are bit-identical to the 2-level pipeline."""

    def __init__(self, base: HierConvAE, top: TopBranch):
        super().__init__()
        self.base, self.top = base, top
        for p in self.base.parameters():
            p.requires_grad_(False)

    def _c4(self, x):
        return self.base.coarse_enc(self.base.enc(x))

    def encode(self, x):                      # 2-level interface (ActionHierData uses this)
        return self.base.encode(x)

    def encode_top(self, x):
        return self.top.encode_from_c4(self._c4(x))

    def encode3(self, x):
        f = self.base.enc(x)
        zf = torch.tanh(self.base.fine_lat(f))
        c4 = self.base.coarse_enc(f)
        zc = torch.tanh(self.base.coarse_lat(c4))
        zt = self.top.encode_from_c4(c4)
        return zt, zc, zf

    def decode_fine(self, z):
        return self.base.decode_fine(z)

    def decode_coarse(self, z):
        return self.base.decode_coarse(z)

    def decode_top(self, z):
        return self.top.decode_top(z)


def load_top(base_path, top_path, dev=DEV):
    base = load_base(base_path)
    top = TopBranch().to(dev)
    st = torch.load(top_path, map_location=dev, weights_only=False)
    top.load_state_dict(st["top"])
    ae3 = TopExtAE(base, top).to(dev).eval()
    return ae3


def train_top(base_path, data_path, out, steps=7000, batch=64, lr=3e-4):
    """Train ONLY the top branch (trunk frozen). Mirrors hier_convae.train's recipe/loss."""
    torch.manual_seed(0)
    base = load_base(base_path)
    top = TopBranch().to(DEV)
    ae3 = TopExtAE(base, top).to(DEV)
    data = FramePool(data_path)
    opt = torch.optim.Adam(top.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    n_frozen = sum(p.numel() for p in base.parameters())
    n_train = sum(p.numel() for p in top.parameters())
    print(f"TopBranch train: {data_path} trainable={n_train/1e6:.2f}M (frozen base {n_frozen/1e6:.2f}M)"
          f" top 2x2", flush=True)
    run, t0 = 0.0, time.time()
    for s in range(1, steps + 1):
        x = data.batch(batch)
        with torch.no_grad():
            c4 = ae3._c4(x)
        rt = top.decode_top(top.encode_from_c4(c4))
        loss = F.l1_loss(rt, x) + F.mse_loss(rt, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += float(loss.detach())
        if s % 1000 == 0:
            with torch.no_grad():
                xv = data.batch(256, train=False)
                rtv = ae3.decode_top(ae3.encode_top(xv))
                mse = float(F.mse_loss(rtv, xv))
            print(f"  [{s}/{steps}] loss {run/1000:.4f}  val top-MSE {mse:.4f}  "
                  f"{(time.time()-t0)/s:.3f}s/step", flush=True)
            run = 0.0
    torch.save({"top": top.state_dict(), "base_path": base_path}, out)
    print(f"saved {out}")
    return ae3


def attach_top(data, ae3, stats=None, chunk_t=True):
    """Add standardized zt (N,T,LAT,2,2) + zt_stats to an existing ActionHierData instance.
    Mirrors the [STD-FIX] pattern exactly: train data computes stats from train_idx; test data
    MUST be passed the train data's .zt_stats. Returns the stats for chaining."""
    N, T = data.N, data.T
    zt = torch.empty(N, T, LAT, TOP_HW, TOP_HW)
    t0 = time.time()
    with torch.no_grad():
        for t in range(T):
            x = torch.from_numpy(data.clips_u8[:, t].astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
            zt[:, t] = ae3.encode_top(x.to(next(ae3.parameters()).device)).cpu()
    if stats is None:
        zt_t = zt[data.train_idx]
        stats = (zt_t.mean(dim=(0, 1, 3, 4)).view(LAT, 1, 1),
                 zt_t.std(dim=(0, 1, 3, 4)).clamp_min(1e-4).view(LAT, 1, 1))
    mu_t, sd_t = stats
    data.zt = (zt - mu_t) / sd_t
    data.zt_stats = tuple(s.clone() for s in stats)
    print(f"[hier3] attached zt{tuple(data.zt.shape)} (std-fixed, raw std {float(sd_t.mean()):.4f})"
          f" in {time.time()-t0:.1f}s", flush=True)
    return data.zt_stats


def _smoke():
    torch.manual_seed(0)
    base = HierConvAE()
    top = TopBranch()
    ae3 = TopExtAE(base, top)
    x = torch.randn(2, 1, 64, 64).clamp(-1, 1)
    zt, zc, zf = ae3.encode3(x)
    assert zt.shape == (2, LAT, 2, 2) and zc.shape == (2, LAT, 4, 4) and zf.shape == (2, LAT, 8, 8)
    rt = ae3.decode_top(zt)
    assert rt.shape == x.shape
    # frozen-trunk assert: base params must not require grad; top must
    assert not any(p.requires_grad for p in ae3.base.parameters())
    assert all(p.requires_grad for p in ae3.top.parameters())
    # encode() (2-level path) matches the raw base bit-exactly
    zc2, zf2 = ae3.encode(x)
    zc3, zf3 = base.encode(x)
    assert torch.equal(zc2, zc3) and torch.equal(zf2, zf3)
    n = sum(p.numel() for p in top.parameters())
    print(f"hier3 SMOKE OK (top branch {n/1e6:.2f}M params, zt {tuple(zt.shape)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--base", default="logs_rhcp/hier_ae_mom.pt")
    ap.add_argument("--data", default="data/loopmom_train_3000x100.npy")
    ap.add_argument("--out", default="logs_rhcp/hier_ae_mom_top.pt")
    ap.add_argument("--steps", type=int, default=7000)
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    elif args.train:
        train_top(args.base, args.data, args.out, steps=args.steps)
    else:
        ap.print_help()
