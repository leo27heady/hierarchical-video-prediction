"""Cheap diagnostics that GATE the hierarchical-EBT build (per the red-team). No EBT needed -- these are INFORMATION
questions answered with fast feed-forward regressors, decoded through the frozen hierarchical AE.

D2 (decisive ceiling test): does knowing the GROUND-TRUTH coarse latent of the target help predict the fine latent?
   A = regress fine_target from context;  B = regress fine_target from context + ORACLE coarse_target.
   B >> A  -> the coarse carries useful, hard-to-predict-but-usable info -> a good coarse predictor could help (GREEN).
   B ~ A   -> the coarse is useless; the fine level already extracts everything from context -> hierarchy NULL (ship Ch.D).
D1-lite: is the target COARSE latent even predictable from context (or is it the unknowable random-walk offset)?
   + is there a spatial-frequency band with an INTERMEDIATE residual (the hard-but-computable regime), vs
   determined-or-unknowable at every scale?
All stratified into easy (low latent-change) vs hard (high latent-change) transitions.

  venv/Scripts/python.exe -m hvp_rev.ebt_diag --ae logs_rhcp/hier_ae.pt --k 4
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.hier_convae import load as load_hier, LAT
from hvp_rev import long_eval as LE

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class HierLatents:
    def __init__(self, ae, clip_path, n_ctx=6, k=4, n_clips=2500, max_t=100):
        clips = np.load(clip_path)[:n_clips]
        N, T = clips.shape[0], min(max_t, clips.shape[1])
        x = torch.from_numpy(clips[:, :T].astype(np.float32) / 127.5 - 1.0).unsqueeze(2)
        zc, zf = [], []
        with torch.no_grad():
            for t in range(T):
                c, f = ae.encode(x[:, t].to(DEV))
                zc.append(c.cpu()); zf.append(f.cpu())
        self.zc = torch.stack(zc, 1)                # (N,T,LAT,4,4)
        self.zf = torch.stack(zf, 1)                # (N,T,LAT,8,8)
        self.N, self.T, self.n_ctx, self.k = N, T, n_ctx, k

    def batch(self, B, train=True):
        n_val = 300
        hi = self.N - n_val if train else self.N
        lo = 0 if train else self.N - n_val
        ci = torch.randint(lo, hi, (B,))
        ti = torch.randint(self.n_ctx, self.T - self.k, (B,))
        ctxf = torch.stack([self.zf[ci[b], ti[b] - self.n_ctx:ti[b]].reshape(-1, 8, 8) for b in range(B)])  # (B,n_ctx*LAT,8,8)
        tgtf = torch.stack([self.zf[ci[b], ti[b] + self.k - 1] for b in range(B)])                          # (B,LAT,8,8)
        tgtc = torch.stack([self.zc[ci[b], ti[b] + self.k - 1] for b in range(B)])                          # (B,LAT,4,4)
        lastf = torch.stack([self.zf[ci[b], ti[b] - 1] for b in range(B)])
        change = (tgtf - lastf).flatten(1).norm(dim=1)                                                       # hardness proxy
        return ctxf.to(DEV), tgtf.to(DEV), tgtc.to(DEV), change.to(DEV)


def _reg(cin, cout, w=128):
    return nn.Sequential(nn.Conv2d(cin, w, 3, padding=1), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(w, cout, 3, padding=1)).to(DEV)


def _train_reg(net, data, use_oracle, steps=3000, B=64):
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for s in range(steps):
        ctxf, tgtf, tgtc, _ = data.batch(B)
        inp = ctxf if not use_oracle else torch.cat([ctxf, F.interpolate(tgtc, size=(8, 8), mode="nearest")], 1)
        loss = F.smooth_l1_loss(net(inp), tgtf)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net


def _decoded_metrics(ae, zf_pred, zf_gt):
    with torch.no_grad():
        p = ae.decode_fine(zf_pred); g = ae.decode_fine(zf_gt)
    pp = ((p.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    gg = ((g.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    fg = float(LE.fg_mse(pp, gg).mean())
    kk = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=p.dtype, device=p.device)[None, None]
    sh = float(F.conv2d((p.clamp(-1, 1) + 1) * 0.5, kk, padding=1).var().item())
    return fg, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="logs_rhcp/hier_ae.pt")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_ctx", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    ae = load_hier(args.ae)
    data = HierLatents(ae, "data/loop_train_3000x100.npy", n_ctx=args.n_ctx, k=args.k)
    cin = args.n_ctx * LAT
    print(f"EBT diagnostics | n_ctx={args.n_ctx} k={args.k} | hier AE {args.ae}", flush=True)

    # ---- D2: oracle-coarse ceiling ----
    A = _train_reg(_reg(cin, LAT), data, use_oracle=False)
    Bnet = _train_reg(_reg(cin + LAT, LAT), data, use_oracle=True)
    ctxf, tgtf, tgtc, change = data.batch(1024, train=False)
    med = change.median()
    hard = change > med
    with torch.no_grad():
        pa = A(ctxf)
        pb = Bnet(torch.cat([ctxf, F.interpolate(tgtc, size=(8, 8), mode="nearest")], 1))
    fa, sa = _decoded_metrics(ae, pa, tgtf); fb, sb = _decoded_metrics(ae, pb, tgtf)
    fa_h, _ = _decoded_metrics(ae, pa[hard], tgtf[hard]); fb_h, _ = _decoded_metrics(ae, pb[hard], tgtf[hard])
    print(f"\n  D2 ORACLE-COARSE CEILING (fine fg-MSE):")
    print(f"    context-only        : all {fa:.4f}  hard {fa_h:.4f}  sharpness {sa:.4f}")
    print(f"    context + ORACLE coarse: all {fb:.4f}  hard {fb_h:.4f}  sharpness {sb:.4f}")
    gain_all = 100 * (fa - fb) / max(fa, 1e-6); gain_hard = 100 * (fa_h - fb_h) / max(fa_h, 1e-6)
    print(f"    -> oracle-coarse improves fine prediction by {gain_all:.1f}% (all) / {gain_hard:.1f}% (hard)")

    # ---- D1-lite: is the coarse itself predictable from context? ----
    Cnet = _reg(cin, LAT)                                   # predict target COARSE (4x4) from context
    opt = torch.optim.Adam(Cnet.parameters(), lr=3e-4)
    for s in range(2000):
        ctxf, _, tgtc, _ = data.batch(64)
        pred = F.avg_pool2d(Cnet(ctxf), 2)                 # 8x8 -> 4x4
        loss = F.smooth_l1_loss(pred, tgtc)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    ctxf, _, tgtc, _ = data.batch(1024, train=False)
    with torch.no_grad():
        pc = F.avg_pool2d(Cnet(ctxf), 2)
    coarse_res = float(((pc - tgtc) ** 2).mean() / (tgtc.var() + 1e-6))    # normalized residual (1=unpredictable,0=perfect)
    print(f"\n  D1 coarse predictability: normalized residual {coarse_res:.3f} "
          f"({'PREDICTABLE' if coarse_res < 0.5 else 'mostly UNKNOWABLE (random-walk offset)'})")

    green = (gain_hard > 10) and (coarse_res < 0.7)
    print(f"\n  DIAGNOSTIC VERDICT: oracle-coarse-helps-hard={gain_hard>10:.0f}  coarse-predictable={coarse_res<0.7} -> "
          f"{'GREEN: build the CONSISTENCY-COUPLED hierarchy (coarse carries usable, predictable info)' if green else 'NULL/RED: hierarchy has no hard-but-computable job here -> ship Ch.D (theory). Report to user.'}")


if __name__ == "__main__":
    main()
