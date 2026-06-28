"""E2 — rollout-drift / noise-temperature probe on the EXISTING CW-VAE checkpoint (zero training).

Long free-run accumulates per-step stochastic noise -> drift. Test whether the injected prior-sampling
noise is a DRIFT DRIVER by sweeping a global noise temperature alpha on the free-run prior:
  z_t = mu_p + alpha * sigma_p * eps     (alpha=0 -> mean rollout; alpha=1 -> standard sampling)
Implemented by scaling torch.randn_like during open_loop (no model change).

If LOWER alpha FLATTENS / LOWERS the long-horizon fg-MSE drift curve (without collapsing), then a
learned per-step noise schedule (Diffusion-Forcing-style) is motivated -> greenlight the on-the-fly
rollout-objective direction. If alpha=1 is already best or low-alpha collapses, noise is not the drift
driver and that direction is deprioritized.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CTX, HOR, CLIPS = 36, 150, 64
WIN = [(0,25),(25,50),(50,100),(100,150)]
x = torch.from_numpy(np.load(str(CW/"data_mmnist"/"test.npy"))[:CLIPS, :CTX+HOR].astype(np.float32)/255.0).unsqueeze(2).to(dev)
gt = x[:, CTX:CTX+HOR]; gt_np = gt.cpu().numpy(); fg = gt_np > 0.1

m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
m.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"]); m.eval()

_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=dev).view(1,1,3,3)
def lapv(img): return F.conv2d(img.reshape(-1,1,64,64),_lap,padding=1).var(dim=(1,2,3)).mean().item()

def run(alpha):
    real = torch.randn_like
    torch.randn_like = (lambda t,*a,**k: torch.zeros_like(t)) if alpha == 0 else (lambda t,*a,**k: alpha*real(t))
    try:
        torch.manual_seed(0); pred = m.open_loop(x, CTX, sample=True)[:, :HOR]
    finally:
        torch.randn_like = real
    return pred

def fgmse_curve(pred):
    p = pred.cpu().numpy(); se = (p - gt_np)**2
    return np.array([[ se[c,t][fg[c,t]].mean() if fg[c,t].any() else 0.0 for t in range(HOR)] for c in range(CLIPS)])  # (C,HOR)

res = {"alpha": {}}
print(f"{'alpha':>6}" + "".join(f"{'fgMSE+'+str(a)+'-'+str(b):>14}" for a,b in WIN) + f"{'lap_var':>10}")
for alpha in [0.0, 0.1, 0.25, 0.5, 1.0]:
    pred = run(alpha); fc = fgmse_curve(pred)            # (C,HOR)
    wins = {f"{a}-{b}": float(fc[:, a:b].mean()) for a,b in WIN}
    lv = lapv(pred)
    res["alpha"][alpha] = {"fgmse_windows": wins, "fgmse_curve": fc.mean(0).tolist(), "lap_var": lv}
    print(f"{alpha:>6}" + "".join(f"{wins[f'{a}-{b}']:>14.3f}" for a,b in WIN) + f"{lv:>10.4f}")

# verdict: does any alpha<1 beat alpha=1 at +100-150 (long-horizon drift) ?
base = res["alpha"][1.0]["fgmse_windows"]["100-150"]
best_a = min(res["alpha"], key=lambda a: res["alpha"][a]["fgmse_windows"]["100-150"])
best_v = res["alpha"][best_a]["fgmse_windows"]["100-150"]
res["verdict"] = {"alpha1_longhorizon": base, "best_alpha": best_a, "best_longhorizon": best_v,
                  "noise_is_drift_driver": bool(best_a < 1.0 and best_v < base - 0.005)}
print(f"\nalpha=1 (+100-150) fg-MSE {base:.3f} | best alpha={best_a} -> {best_v:.3f}")
print("VERDICT:", "noise IS a drift driver -> on-the-fly noise-schedule / Diffusion-Forcing motivated"
      if res["verdict"]["noise_is_drift_driver"] else
      "reducing noise does NOT help long-horizon (or collapses) -> noise-schedule less promising")
(ROOT/"logs_dlh"/"e2_rollout_drift.json").write_text(json.dumps(res, indent=2))
print("wrote", ROOT/"logs_dlh"/"e2_rollout_drift.json")
