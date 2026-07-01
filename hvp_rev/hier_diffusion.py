"""CONTROL experiment: the SAME hierarchical coarse-to-fine setup as hier_ebt.py, but with a DIFFUSION denoiser
(MilliVid's mechanism) instead of EBT. Isolates whether the mean-blur is EBT-specific (deterministic energy descent ->
mean) or task-inherent. Diffusion SAMPLES from the posterior (v-prediction, cosine schedule, DDIM), so on unresolvable
multimodal futures it should COMMIT to a sharp mode where EBT/regression blur to the mean.

Eval is fair to a sampler: sharpness (does it commit?) + Best-of-N ORACLE fg-MSE (closest of N samples to GT, since a
sampler produces a valid-but-random mode, not the mean). Compared to hier-regression, persistence, and the logged EBT
numbers (fg-MSE 0.0101 / sharp 0.0120 / GT sharp 0.0294).

  venv/Scripts/python.exe -m hvp_rev.hier_diffusion --ae logs_rhcp/hier_ae.pt --steps 8000
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.hier_convae import load as load_hier, LAT
from hvp_rev.hier_ebt import HierData, _metrics
from hvp_rev import long_eval as LE
from hvp_rev.viz import save_samples

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def alpha_sigma(t):
    """cosine schedule, variance-preserving. t in [0,1] (1=pure noise)."""
    a = torch.cos(t * math.pi / 2)
    s = torch.sin(t * math.pi / 2)
    return a, s


def _tembed(t, dim=64):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class Denoiser(nn.Module):
    """conv v-predictor over the latent grid; context + optional cond (clean coarse) + time embedding."""

    def __init__(self, ctx_ch, cond_ch=0, w=128):
        super().__init__()
        cin = LAT + ctx_ch + cond_ch
        self.inp = nn.Conv2d(cin, w, 3, padding=1)
        self.temb = nn.Linear(64, w)
        self.mid = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1),
                                 nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, w, 3, padding=1))
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(), nn.Conv2d(w, LAT, 3, padding=1))

    def forward(self, y_noisy, ctx, t, cond=None):
        parts = [y_noisy, ctx] + ([cond] if cond is not None else [])
        h = self.inp(torch.cat(parts, 1)) + self.temb(_tembed(t))[:, :, None, None]
        return self.out(h + self.mid(h))


def diffusion_loss(net, x0, ctx, cond=None):
    B = x0.shape[0]
    t = torch.rand(B, device=DEV)
    a, s = alpha_sigma(t)
    a, s = a[:, None, None, None], s[:, None, None, None]
    eps = torch.randn_like(x0)
    x_t = a * x0 + s * eps
    v = a * eps - s * x0
    return F.mse_loss(net(x_t, ctx, t, cond), v)


@torch.no_grad()
def ddim_sample(net, ctx, shape, n_steps=16, cond=None):
    y = torch.randn(ctx.shape[0], *shape, device=DEV)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=DEV)
    x_hat = y
    for i in range(n_steps):
        t = ts[i].expand(ctx.shape[0])
        a, s = alpha_sigma(ts[i]); an, sn = alpha_sigma(ts[i + 1])
        v = net(y, ctx, t, cond)
        x_hat = a * y - s * v
        eps = s * y + a * v
        y = an * x_hat + sn * eps
    return x_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="logs_rhcp/hier_ae.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n_ctx", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sample_steps", type=int, default=16)
    ap.add_argument("--bo_n", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0)
    ae = load_hier(args.ae)
    data = HierData(ae, "data/loop_train_3000x100.npy", n_ctx=args.n_ctx, k=args.k)
    dc = Denoiser(ctx_ch=LAT * args.n_ctx).to(DEV)                       # coarse denoiser
    df = Denoiser(ctx_ch=LAT * args.n_ctx, cond_ch=LAT).to(DEV)          # fine denoiser | clean coarse
    opt = torch.optim.Adam(list(dc.parameters()) + list(df.parameters()), lr=3e-4)
    print(f"HIER-DIFFUSION control | n_ctx={args.n_ctx} k={args.k} sample_steps={args.sample_steps} bo_n={args.bo_n}", flush=True)
    for s in range(1, args.steps + 1):
        cc, cf, tc, tf, _ = data.batch(args.batch)
        lc = diffusion_loss(dc, tc, cc)
        lf = diffusion_loss(df, tf, cf, cond=F.interpolate(tc, size=(8, 8), mode="nearest"))
        loss = lc + lf
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  [step {s}] diff loss {float(loss.detach()):.4f}", flush=True)

    # ---- eval: single-sample sharpness + Best-of-N oracle (the fair metric for a sampler) ----
    cc, cf, tc, tf, change, lastf, gtf = data.batch(512, train=False, return_frames=True)
    med = change.median(); hard = change > med
    # single sample
    yc = ddim_sample(dc, cc, (LAT, 4, 4), args.sample_steps)
    yf = ddim_sample(df, cf, (LAT, 8, 8), args.sample_steps, cond=F.interpolate(yc, size=(8, 8), mode="nearest"))
    fg1, sh1, dec1 = _metrics(ae, yf, tf)
    # Best-of-N oracle (closest of N samples to GT latent)
    best = None; best_err = None
    for _ in range(args.bo_n):
        yc = ddim_sample(dc, cc, (LAT, 4, 4), args.sample_steps)
        yfn = ddim_sample(df, cf, (LAT, 8, 8), args.sample_steps, cond=F.interpolate(yc, size=(8, 8), mode="nearest"))
        err = ((yfn - tf) ** 2).mean(dim=(1, 2, 3))
        if best is None:
            best, best_err = yfn, err
        else:
            m = err < best_err; best = torch.where(m[:, None, None, None], yfn, best); best_err = torch.minimum(err, best_err)
    fgo, sho, deco = _metrics(ae, best, tf)
    fgo_h, _, _ = _metrics(ae, best[hard], tf[hard], decode=False)
    print(f"\n  DIFFUSION (sampler) fine prediction:")
    print(f"    single sample : fg-MSE {fg1:.4f}  sharpness {sh1:.4f}")
    print(f"    Best-of-{args.bo_n} oracle: fg-MSE {fgo:.4f} (hard {fgo_h:.4f})  sharpness {sho:.4f}")
    print(f"    [compare] EBT: 0.0101 / sharp 0.0120  |  hier-regression: 0.0098 / sharp 0.0106  |  GT sharp 0.0294")
    verdict = "DIFFUSION COMMITS to sharper modes than EBT/regression -> the blur was EBT-specific (deterministic descent)" \
        if sh1 > 0.017 else "DIFFUSION also blurs -> the task/data is the limiter, not EBT"
    print(f"    => {verdict}")
    save_samples("logs_rhcp/hier_diffusion_samples.png",
                 [("last", lastf), ("diff-1samp", dec1), ("diff-bestN", deco), ("GT", gtf)], n=8)
    print("  => review logs_rhcp/hier_diffusion_samples.png (last | diffusion-1sample | diffusion-bestN | GT)")


if __name__ == "__main__":
    main()
