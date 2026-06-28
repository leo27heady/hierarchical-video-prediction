"""A0 — surface the long-horizon DLH-vs-CW-VAE verdict WITH a clip-bootstrap CI + fg-LPIPS.

The plan's central long-horizon claim lives in logs_dlh/compare_curves.json but as point
estimates only. This re-runs both models on matched clips and computes a 95% CI on the
DLH-CW-VAE fg-MSE difference per horizon window (bootstrap over clips), plus fg-LPIPS curves.

CAVEAT (stated honestly): only ONE trained checkpoint exists per model, so this captures
clip + rollout-seed variance, NOT training-seed variance (the dominant source). It is a
LOWER bound on uncertainty — if the gap is already inside this CI, it is certainly not
significant; if outside, training-seed variance still needs >=3 seeds to confirm.

Usage: venv/Scripts/python.exe scripts/a0_longhorizon_ci.py
"""
import sys, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CW))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CTX, HOR, CLIPS, NSAMP = 36, 150, 64, 5
WINDOWS = [(0, 25), (25, 50), (50, 100), (100, 150)]

test = np.load(str(CW / "data_mmnist" / "test.npy"))[:CLIPS, :CTX + HOR]
x = torch.from_numpy(test.astype(np.float32) / 255.0).unsqueeze(2).to(dev)
gt = x[:, CTX:CTX + HOR]                                  # (N,H,1,64,64)
gt_np = gt.cpu().numpy()
fgmask = gt_np > 0.1

import lpips
_lp = lpips.LPIPS(net='alex', spatial=True).to(dev).eval()

@torch.no_grad()
def fg_lpips_curve(pred):                                  # pred (N,H,1,64,64) tensor [0,1]
    """per-frame foreground-masked LPIPS, mean over clips -> (H,)."""
    out = np.zeros(HOR)
    for t in range(HOR):
        P = pred[:, t].repeat(1, 3, 1, 1) * 2 - 1
        G = gt[:, t].repeat(1, 3, 1, 1) * 2 - 1
        dmap = _lp(P, G)                                   # (N,1,h,w)
        dmap = F.interpolate(dmap, size=(64, 64), mode='bilinear', align_corners=False)
        m = torch.from_numpy(fgmask[:, t].astype(np.float32)).to(dev)
        out[t] = ((dmap * m).sum() / m.sum().clamp(min=1)).item()
    return out

def per_clip_fgmse(pred_np):                               # (N,H,1,64,64) -> (N,H) fg-MSE per clip/frame
    se = (pred_np - gt_np) ** 2
    out = np.zeros((CLIPS, HOR))
    for c in range(CLIPS):
        for t in range(HOR):
            mk = fgmask[c, t]
            out[c, t] = se[c, t][mk].mean() if mk.any() else 0.0
    return out

def run_dlh():
    from hvp_dlh.config import DLHConfig
    from hvp_dlh.model import DLH
    sd = torch.load(str(ROOT / "logs_dlh" / "baseline_iter8000.pt"), map_location=dev)
    m = DLH(DLHConfig(**sd["cfg"])).to(dev); m.load_state_dict(sd["model"]); m.eval()
    preds = []
    for s in range(NSAMP):
        torch.manual_seed(s)
        p, _ = m.open_loop(x, CTX, HOR); preds.append(p)
    return preds

def run_cwvae():
    from cwvae_torch import CWVAE
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3,
              enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
    m.load_state_dict(torch.load(str(CW / "logs_torch" / "cwvae" / "last.pt"), map_location=dev)["model"]); m.eval()
    preds = []
    for s in range(NSAMP):
        torch.manual_seed(s)
        p = m.open_loop(x, CTX, sample=True)[:, :HOR]; preds.append(p)
    return preds

def select(preds, mode):
    """preds: list of NSAMP (N,H,1,64,64). mode='n1' -> sample 0; 'best5' -> per-clip min mean-fg-MSE."""
    pcs = [per_clip_fgmse(p.cpu().numpy()) for p in preds]   # list of (N,H)
    if mode == 'n1':
        return preds[0], pcs[0]
    means = np.stack([pc.mean(1) for pc in pcs], 0)         # (NSAMP, N)
    pick = means.argmin(0)                                  # (N,)
    keep = torch.stack([preds[pick[c]][c] for c in range(CLIPS)], 0)
    keep_pc = np.stack([pcs[pick[c]][c] for c in range(CLIPS)], 0)
    return keep, keep_pc

def boot_ci(diff_per_clip, nboot=4000):                    # diff_per_clip (N,) -> (mean, lo, hi)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff_per_clip), size=(nboot, len(diff_per_clip)))
    bs = diff_per_clip[idx].mean(1)
    return float(diff_per_clip.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

print(f"running DLH + CW-VAE open-loop, ctx={CTX} hor={HOR} clips={CLIPS} samples={NSAMP} ...", flush=True)
dlh_preds = run_dlh(); cw_preds = run_cwvae()
res = {"config": dict(ctx=CTX, horizon=HOR, clips=CLIPS, nsamp=NSAMP), "windows": {}, "curves": {}}

for mode in ['n1', 'best5']:
    dlh_keep, dlh_pc = select(dlh_preds, mode)
    cw_keep, cw_pc = select(cw_preds, mode)
    res["curves"][mode] = {"dlh_fgmse": dlh_pc.mean(0).tolist(), "cw_fgmse": cw_pc.mean(0).tolist()}
    if mode == 'best5':
        res["curves"][mode]["dlh_fglpips"] = fg_lpips_curve(dlh_keep).tolist()
        res["curves"][mode]["cw_fglpips"] = fg_lpips_curve(cw_keep).tolist()
    res["windows"][mode] = {}
    print(f"\n=== mode={mode} :: fg-MSE per window (DLH - CW-VAE), 95% clip-bootstrap CI ===")
    print(f"{'window':>12}{'DLH':>9}{'CW-VAE':>9}{'diff':>9}{'95% CI':>20}{'sig?':>6}")
    for lo, hi in WINDOWS:
        dwin = dlh_pc[:, lo:hi].mean(1); cwin = cw_pc[:, lo:hi].mean(1)
        diff = dwin - cwin
        mean, clo, chi = boot_ci(diff)
        sig = "yes" if (clo > 0 or chi < 0) else "no"
        res["windows"][mode][f"{lo}-{hi}"] = {"dlh": float(dwin.mean()), "cw": float(cwin.mean()),
                                               "diff": mean, "ci": [clo, chi], "sig": sig}
        print(f"{f'+{lo}-{hi}':>12}{dwin.mean():>9.3f}{cwin.mean():>9.3f}{mean:>+9.3f}"
              f"{f'[{clo:+.3f},{chi:+.3f}]':>20}{sig:>6}  (neg = DLH better)")

# crossover frame (best5)
d = np.array(res["curves"]["best5"]["dlh_fgmse"]); c = np.array(res["curves"]["best5"]["cw_fgmse"])
cross = np.where(d < c)[0]
res["crossover_frame"] = int(cross[0]) if len(cross) else -1
print(f"\nDLH fg-MSE crosses below CW-VAE at frame +{res['crossover_frame']} (best5)")
print(f"fg-LPIPS @ +100-150: DLH {np.mean(res['curves']['best5']['dlh_fglpips'][100:150]):.3f} "
      f"CW-VAE {np.mean(res['curves']['best5']['cw_fglpips'][100:150]):.3f}")

outp = ROOT / "logs_dlh" / "a0_longhorizon_ci.json"
outp.write_text(json.dumps(res, indent=2)); print("wrote", outp)
