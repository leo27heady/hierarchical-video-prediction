"""E1 — clock-premise pre-check on the EXISTING CW-VAE checkpoint (zero training).

The adaptive-clock idea needs a grounding signal that SPIKES at events and is LOW during static
stretches. Entropy was already shown dead (rho~0). Here we test the better candidate: the per-level
one-step surprise g = KL(posterior || prior) that CW-VAE already computes. Does g track real frame
events (foreground change)? If not, the error-grounded clock is dead on CW-VAE too, for free.

Decision: if bottom-level Spearman(g, fg-change) is meaningfully > 0 (say > 0.2) AND g separates
event vs static frames beyond a bootstrap CI, the premise is ALIVE; else KILLED.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

T, CLIPS, FACTOR = 100, 64, 6
x = torch.from_numpy(np.load(str(CW/"data_mmnist"/"test.npy"))[:CLIPS, :T].astype(np.float32)/255.0).unsqueeze(2).to(dev)

m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
m.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"]); m.eval()

def kl(q, p):  # (B,Tl,20) -> (B,Tl) sum over dims
    return (torch.log(p["stddev"]/q["stddev"]) + (q["stddev"]**2 + (q["mean"]-p["mean"])**2)/(2*p["stddev"]**2) - 0.5).sum(-1)

with torch.no_grad():
    _, priors, posteriors = m(x)
g = [kl(posteriors[l], priors[l]).cpu().numpy() for l in range(3)]          # per level (B,Tl)
# prior differential entropy (the DEAD baseline from memory): 0.5*sum log(2 pi e sigma_p^2)
ent0 = (0.5*torch.log(2*np.pi*np.e*priors[0]["stddev"]**2)).sum(-1).detach().cpu().numpy()  # (B,T)

fg = (x > 0.1).float().cpu().numpy()                                         # (B,T,1,64,64)
chg = np.abs(np.diff(fg, axis=1)).mean(axis=(2,3,4))                         # (B,T-1) frame-to-frame fg change
chg = np.concatenate([np.zeros((CLIPS,1)), chg], axis=1)                     # (B,T), c_0=0

def boot_diff(a, b, n=4000):                                                 # mean(a)-mean(b) with 95% CI
    rng = np.random.default_rng(0)
    da = a[rng.integers(0,len(a),(n,len(a)))].mean(1); db = b[rng.integers(0,len(b),(n,len(b)))].mean(1)
    d = da-db; return float(a.mean()-b.mean()), float(np.percentile(d,2.5)), float(np.percentile(d,97.5))

res = {"levels": {}}
print(f"{'level':>6}{'Tl':>5}{'rho(g,fgchg)':>14}{'rho(entropy)':>14}{'g_event':>10}{'g_static':>10}{'sep_CI':>22}{'verdict':>10}")
for l in range(3):
    Tl = g[l].shape[1]
    if l == 0:
        cc = chg
    else:                                                                    # window fg-change to this level's rate
        merge = FACTOR**l; cc = np.zeros((CLIPS, Tl))
        for j in range(Tl):
            cc[:, j] = chg[:, j*merge:min((j+1)*merge, T)].mean(1) if j*merge < T else 0.0
    gl = g[l]
    rho, _ = spearmanr(gl[:,1:].ravel(), cc[:,1:].ravel())                    # skip t=0 (c_0=0 by construction)
    rho_e = spearmanr(ent0[:,1:].ravel(), chg[:,1:].ravel())[0] if l == 0 else np.nan
    # event vs static separation (per this level)
    flat_g = gl[:,1:].ravel(); flat_c = cc[:,1:].ravel()
    hi = flat_c >= np.percentile(flat_c, 75); lo = flat_c <= np.percentile(flat_c, 25)
    md, clo, chi = boot_diff(flat_g[hi], flat_g[lo])
    sep_sig = clo > 0
    verdict = "ALIVE" if (rho > 0.2 and sep_sig) else ("weak" if (rho > 0.1 and sep_sig) else "DEAD")
    res["levels"][l] = {"Tl": Tl, "rho_g_fgchg": float(rho), "rho_entropy": (None if np.isnan(rho_e) else float(rho_e)),
                        "g_event": float(flat_g[hi].mean()), "g_static": float(flat_g[lo].mean()),
                        "sep_diff": md, "sep_ci": [clo, chi], "verdict": verdict}
    print(f"{l:>6}{Tl:>5}{rho:>14.3f}{(rho_e if l==0 else np.nan):>14.3f}{flat_g[hi].mean():>10.2f}{flat_g[lo].mean():>10.2f}"
          f"{f'[{clo:+.2f},{chi:+.2f}]':>22}{verdict:>10}")

overall = res["levels"][0]["verdict"]
res["overall"] = overall
print(f"\nBottom-level g=KL(post||prior) vs fg-change: rho={res['levels'][0]['rho_g_fgchg']:.3f} "
      f"(entropy baseline rho={res['levels'][0]['rho_entropy']:.3f}).  PREMISE: {overall}")
print("(ALIVE = an error-grounded adaptive clock has a real signal to gate on; DEAD = like entropy, abandon the clock.)")
(ROOT/"logs_dlh"/"e1_clock_premise.json").write_text(json.dumps(res, indent=2))
print("wrote", ROOT/"logs_dlh"/"e1_clock_premise.json")
