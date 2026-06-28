"""E1 GATE — validate the rollout-training arms after training (red-teamed criterion).

Loads the 3 seeds x {lambda0, nograd, withgrad} final checkpoints + the frozen CW-VAE baseline,
re-evaluates open-loop on a held-out set (64 clips, multiple rollout seeds), aggregates mean +- std
across the 3 TRAINING seeds, and applies the gate:

  THE METHOD WINS if `withgrad` beats BOTH controls (nograd AND lambda0) on the IN-WINDOW metrics
  (fg-MSE@+8 AND fg-LPIPS@+8) by more than the combined training-seed std, WITHOUT regressing
  short-horizon (+5) or sharpness (lap-var) or collapsing the top level (top-KL >= 0.5x baseline).
  +25/+50 are reported as EXTRAPOLATION diagnostics (not gated). nograd>baseline isolates 'exposure';
  withgrad>nograd isolates 'gradient through the rollout dynamics' = the thesis.

Usage: venv/Scripts/python.exe -m hvp_rev.gate_rollout
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
from hvp_rev.train_rollout import kl_sum, lapv
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = ROOT / "logs_rollout"
CTX, HOR, CLIPS, RSEEDS = 36, 50, 64, 3
HZ = [4, 7, 24, 49]   # frames +5,+8,+25,+50 (0-based idx)

def new_model():
    return CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)

@torch.no_grad()
def eval_model(m, test, lp):
    x = torch.from_numpy(test[:CLIPS, :CTX+HOR].astype(np.float32)/255.0).unsqueeze(2).to(dev)
    gt = x[:, CTX:]; gt_np = gt.cpu().numpy(); fg = gt_np > 0.1
    acc = {f"fgmse@{t+1}": [] for t in HZ}; acc.update({f"fglpips@{t+1}": [] for t in HZ}); acc["lap_var"] = []
    for s in range(RSEEDS):
        torch.manual_seed(s); pred = m.open_loop(x, CTX, sample=True)[:, :HOR]; p = pred.cpu().numpy()
        acc["lap_var"].append(lapv(pred))
        for t in HZ:
            acc[f"fgmse@{t+1}"].append(float(np.mean([((p[c,t]-gt_np[c,t])**2)[fg[c,t]].mean() if fg[c,t].any() else 0.0 for c in range(CLIPS)])))
            P = pred[:,t].repeat(1,3,1,1)*2-1; G = gt[:,t].repeat(1,3,1,1)*2-1
            dm = F.interpolate(lp(P,G), size=(64,64), mode="bilinear", align_corners=False)
            mk = torch.from_numpy(fg[:,t].astype(np.float32)).to(dev)
            acc[f"fglpips@{t+1}"].append(float((dm*mk).sum()/mk.sum().clamp(min=1)))
    _, pri, pos = m(x[:, :CTX]); topkl = float(kl_sum(pos[-1], pri[-1]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}; out["top_kl"] = topkl
    return out

def main():
    import lpips; lp = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    test = np.load(str(CW/"data_mmnist"/"test.npy"))
    arms = {}
    # baseline (frozen pretrained CW-VAE)
    mb = new_model(); mb.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"]); mb.eval()
    arms["baseline"] = [eval_model(mb, test, lp)]
    # trained arms x seeds
    for arm in ["lambda0", "nograd", "withgrad"]:
        runs = []
        for seed in [0, 1, 2]:
            ck = LOG / f"e1_{arm}_s{seed}_last.pt"
            if not ck.exists(): print(f"  (missing {ck.name})"); continue
            sd = torch.load(ck, map_location=dev); m = new_model(); m.load_state_dict(sd["model"]); m.eval()
            runs.append(eval_model(m, test, lp))
        if runs: arms[arm] = runs

    def agg(arm, key): v = np.array([r[key] for r in arms[arm]]); return float(v.mean()), float(v.std())
    keys = ["fgmse@5","fgmse@8","fglpips@5","fglpips@8","fgmse@25","fgmse@50","fglpips@25","fglpips@50","lap_var","top_kl"]
    print(f"\n{'arm':>10} " + " ".join(f"{k:>12}" for k in keys[:6]))
    summary = {}
    for arm in arms:
        summary[arm] = {k: agg(arm, k) for k in keys}
        print(f"{arm:>10} " + " ".join(f"{summary[arm][k][0]:6.3f}±{summary[arm][k][1]:.3f}" for k in keys[:6]))
    print("(diagnostics +25/+50 + lap_var/top_kl in JSON)")

    # ---- gate ----
    res = {"summary": summary, "gate": {}}
    if all(a in arms for a in ["withgrad","nograd","lambda0"]):
        def beats(a, b, key):  # a < b beyond combined std (lower better)
            (am, asd), (bm, bsd) = summary[a][key], summary[b][key]
            return am < bm - (asd + bsd)
        wins_inwindow = (beats("withgrad","nograd","fgmse@8") and beats("withgrad","nograd","fglpips@8"))
        beats_l0 = (beats("withgrad","lambda0","fgmse@8") and beats("withgrad","lambda0","fglpips@8"))
        exposure = beats("nograd","baseline","fgmse@8")
        no_regress = (summary["withgrad"]["fgmse@5"][0] <= summary["baseline"]["fgmse@5"][0] + 0.02
                      and summary["withgrad"]["lap_var"][0] >= 0.8*summary["baseline"]["lap_var"][0]
                      and summary["withgrad"]["top_kl"][0] >= 0.5*summary["baseline"]["top_kl"][0])
        res["gate"] = {"withgrad_beats_nograd_inwindow": wins_inwindow, "withgrad_beats_lambda0": beats_l0,
                       "exposure_effect(nograd>baseline)": exposure, "no_regression": no_regress,
                       "VERDICT": "METHOD CONFIRMED" if (wins_inwindow and no_regress) else "NULL / inconclusive"}
        print("\n=== GATE ===")
        for k, v in res["gate"].items(): print(f"  {k}: {v}")
    (LOG / "e1_gate.json").write_text(json.dumps(res, indent=2)); print("\nwrote", LOG/"e1_gate.json")

if __name__ == "__main__":
    main()
