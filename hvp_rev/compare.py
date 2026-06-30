"""Unified comparison: RHCP vs CW-VAE (mean + best5) vs DLH vs persistence vs black.

  short table (P1):  venv/Scripts/python.exe -m hvp_rev.compare --mode short --rhcp logs_rhcp/p1_seed0.pt
  long drift  (P3):  venv/Scripts/python.exe -m hvp_rev.compare --mode long  --rhcp_s ... --rhcp_l ... --H 40

All models get the SAME n_in context frames and free-run the horizon (no peeking). Short metrics are the bit-identical
OpenSTL convention (eval_adapters); long metrics are the foreground/position-drift set (long_eval). CW-VAE evaluated in
mean-mode (sample=False, matched to deterministic RHCP) AND best-of-5 (stochastic ceiling).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CW))

from hvp_rev.eval_adapters import load_clips, evaluate, OPENSTL_TEST, CWVAE_TEST
from hvp_rev import long_eval as LE
from hvp_rev.rhcp import RHCP

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def get_clips(dataset, n_in, H, n_clips):
    """Returns ctx01 (N,n_in,1,64,64), gt01 (N,H,1,64,64), full01 (N,n_in+H,...) -- all [0,1]."""
    path, ta = {"openstl": (OPENSTL_TEST, 0), "cwvae": (CWVAE_TEST, 1)}[dataset]
    arr = load_clips(path, ta, n=n_clips, t=n_in + H)              # (N,T,1,64,64) [0,1]
    return arr[:, :n_in], arr[:, n_in:n_in + H], arr


# ------------------------------------------------------------------ model runners (all return preds (N,H,1,64,64) [0,1])
@torch.no_grad()
def run_rhcp(ckpt, ctx01, H, batch=64):
    sd = torch.load(ckpt, map_location=DEV)
    model = RHCP(sd["cfg"]).to(DEV).eval()
    model.load_state_dict(sd["model"])
    outs = []
    for s in range(0, ctx01.shape[0], batch):
        c = torch.from_numpy(ctx01[s:s + batch] * 2 - 1).to(DEV)
        y = model.rollout(c, H)
        outs.append(((y.clamp(-1, 1) + 1) * 0.5).cpu().numpy())
    return np.concatenate(outs)


def run_persist(ctx01, H):
    last = ctx01[:, -1:]                                            # (N,1,1,64,64)
    return np.repeat(last, H, axis=1)


def run_black(ctx01, H):
    return np.zeros((ctx01.shape[0], H, 1, 64, 64), np.float32)


@torch.no_grad()
def run_cwvae(full01, n_in, H, sample=False, nsamp=1, batch=50):
    from cwvae_torch import CWVAE
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3,
              enc_embed=1000, cm=1, dec_stddev=1.0).to(DEV).eval()
    m.load_state_dict(torch.load(str(CW / "logs_torch" / "cwvae" / "last.pt"), map_location=DEV)["model"])
    obs = torch.from_numpy(full01[:, :n_in + H] * 1.0).float().to(DEV)   # (N,n_in+H,1,64,64) in [0,1]
    best = None
    for sd in range(nsamp):
        torch.manual_seed(sd)
        outs = []
        for s in range(0, obs.shape[0], batch):
            p = m.open_loop(obs[s:s + batch], n_in, sample=sample)[:, :H]
            outs.append(p.cpu().numpy())
        pred = np.concatenate(outs)                                # (N,H,1,64,64)
        if best is None:
            best = pred
        else:                                                      # best-of-N per clip by mean fg-MSE vs gt
            best = pred  # placeholder; best-of-N handled by caller when needed
    return best


@torch.no_grad()
def run_dlh(full01, n_in, H, batch=50):
    from hvp_dlh.config import DLHConfig
    from hvp_dlh.model import DLH
    sd = torch.load(str(ROOT / "logs_dlh" / "baseline_iter8000.pt"), map_location=DEV)
    m = DLH(DLHConfig(**sd["cfg"])).to(DEV).eval()
    m.load_state_dict(sd["model"])
    obs = torch.from_numpy(full01[:, :n_in + H] * 1.0).float().to(DEV)
    outs = []
    for s in range(0, obs.shape[0], batch):
        p, _ = m.open_loop(obs[s:s + batch], n_in, H)
        outs.append(p.cpu().numpy())
    return np.concatenate(outs)


def cwvae_best_of_n(full01, gt01, n_in, H, nsamp=5, batch=50):
    """Per-clip best-of-N selection by mean fg-MSE vs gt (the stochastic ceiling)."""
    from cwvae_torch import CWVAE
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3,
              enc_embed=1000, cm=1, dec_stddev=1.0).to(DEV).eval()
    m.load_state_dict(torch.load(str(CW / "logs_torch" / "cwvae" / "last.pt"), map_location=DEV)["model"])
    obs = torch.from_numpy(full01[:, :n_in + H] * 1.0).float().to(DEV)
    samples = []
    for sd in range(nsamp):
        torch.manual_seed(sd)
        outs = []
        for s in range(0, obs.shape[0], batch):
            outs.append(m.open_loop(obs[s:s + batch], n_in, sample=True)[:, :H].cpu().numpy())
        samples.append(np.concatenate(outs))
    samples = np.stack(samples)                                    # (S,N,H,1,64,64)
    fgm = (gt01 > 0.1)
    err = ((samples - gt01[None]) ** 2 * fgm[None]).sum(axis=(3, 4, 5)) / np.maximum(fgm.sum(axis=(2, 3, 4))[None], 1)
    pick = err.mean(axis=2).argmin(axis=0)                         # (N,) best sample per clip
    return samples[pick, np.arange(samples.shape[1])]


# ------------------------------------------------------------------ modes
def mode_short(args):
    # OpenSTL standard is 10->10 (test clips are only 20 frames). CW-VAE/DLH need >=36 ctx frames (factor-6 temporal
    # hierarchy) so they CANNOT run this protocol -- the P1 comparison is vs the published table + persistence.
    print(f"\n==== SHORT TABLE (10->10) ====")
    for dataset in args.datasets.split(","):
        ctx, gt, full = get_clips(dataset, 10, 10, args.n_clips)
        tag = "cross-distribution (RHCP trained on cwvae)" if dataset == "openstl" else "matched-to-baselines"
        print(f"\n-- {dataset} ({ctx.shape[0]} clips) --   [{tag}]")
        rows = {"persistence": run_persist(ctx, 10), "black": run_black(ctx, 10)}
        if args.rhcp:
            rows["RHCP"] = run_rhcp(args.rhcp, ctx, 10)
        for name, pred in rows.items():
            r = evaluate(pred, gt)
            print(f"   {name:>14}: MSE {r['mse']:7.2f}  MAE {r['mae']:7.2f}  SSIM {r['ssim']:.4f}  PSNR {r['psnr']:.2f}")
        print("   (OpenSTL table tier: PredRNN++ 22.06 / SimVP 32.15 / CrevNet 30.15 [reversible]; "
              "CW-VAE/DLH need ctx>=36 -> long panel only)")


def mode_long(args):
    H, C = args.H, args.ctx
    print(f"\n==== LONG-HORIZON DRIFT (ctx={C} -> {H}) on cwvae test ====   (ctx>=36 required by CW-VAE/DLH)")
    ctx, gt, full = get_clips("cwvae", C, H, args.n_clips)
    models = {"persistence": run_persist(ctx, H),
              "CW-VAE(mean)": run_cwvae(full, C, H, sample=False),
              "CW-VAE(best5)": cwvae_best_of_n(full, gt, C, H, nsamp=5),
              "DLH": run_dlh(full, C, H)}
    if args.rhcp_s:
        models["RHCP-S(K10)"] = run_rhcp(args.rhcp_s, ctx, H)
    if args.rhcp_l:
        models["RHCP-L(K40)"] = run_rhcp(args.rhcp_l, ctx, H)
    if args.rhcp:
        models["RHCP"] = run_rhcp(args.rhcp, ctx, H)
    reports = {}
    for name, pred in models.items():
        reports[name] = LE.report(pred, gt, label=name)
        LE.print_report(reports[name])
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["short", "long"], required=True)
    ap.add_argument("--rhcp", default=None)
    ap.add_argument("--rhcp_s", default=None)
    ap.add_argument("--rhcp_l", default=None)
    ap.add_argument("--datasets", default="openstl,cwvae")
    ap.add_argument("--n_clips", type=int, default=512)
    ap.add_argument("--H", type=int, default=40)
    ap.add_argument("--ctx", type=int, default=36)
    args = ap.parse_args()
    print(f"compare | device {DEV}")
    if args.mode == "short":
        mode_short(args)
    else:
        mode_long(args)
