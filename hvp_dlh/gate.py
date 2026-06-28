"""Rigorous gate: compare checkpoints on the DECISION metrics (fg-MSE + fg-LPIPS) with mean±std over
multiple rollout seeds and many clips, at in-training-range horizons (+25/+50). Also reports the
hierarchy (top-level KL, K). A config 'beats' the control if it lowers fg-MSE AND fg-LPIPS at +50
beyond the combined std, without hierarchy regression. Usage:
  python -m hvp_dlh.gate --ckpts logs_dlh/s1_A0_tanh_std1_last.pt logs_dlh/s1_A_sig_std1_last.pt ... \
         --control s1_A0_tanh_std1 --seeds 3 --clips 48 --samples 4 --horizon 100
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_dlh.data import make_loaders
from hvp_dlh.eval import evaluate


def run_one(ckpt, seeds, clips, samples, horizon, dev):
    sd = torch.load(ckpt, map_location=dev)
    cfg = DLHConfig(**sd["cfg"]); cfg.eval_horizon = horizon; cfg.ctx_frames = 36
    m = DLH(cfg).to(dev); m.load_state_dict(sd["model"]); m.eval()
    _, test = make_loaders(cfg)
    acc = {}
    for s in range(seeds):
        ev = evaluate(m, test, cfg, n_clips=clips, n_samples=samples, device=dev, seed=s)
        rec = {"fg_mse@25": ev.get("fg_mse@25", np.nan), "fg_mse@50": ev.get("fg_mse@50", np.nan),
               "lpips@25": float(ev["lpips_fg_curve"][24]) if "lpips_fg_curve" in ev and horizon > 25 else np.nan,
               "lpips@50": float(ev["lpips_fg_curve"][49]) if "lpips_fg_curve" in ev and horizon > 50 else np.nan,
               "lpips_fg": ev["lpips_fg"], "fg_mse": ev["fg_mse"],
               "top_kl": ev["skl_per_level"][-1], "K_tf": ev["K_mean_tf"], "step": sd.get("step", 0)}
        for k, v in rec.items():
            acc.setdefault(k, []).append(v)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--control", default=None, help="run-name substring of the control (e.g. s1_A0)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out", default="logs_dlh/gate_step1.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for ck in args.ckpts:
        name = Path(ck).stem.replace("_last", "")
        print(f"evaluating {name} ...", flush=True)
        res[name] = run_one(ck, args.seeds, args.clips, args.samples, args.horizon, dev)
    Path(args.out).write_text(json.dumps(res, indent=2))

    keys = ["fg_mse@25", "fg_mse@50", "lpips@25", "lpips@50", "top_kl", "K_tf"]
    print(f"\n{'run':24}" + "".join(f"{k:>13}" for k in keys))
    for name, r in res.items():
        print(f"{name:24}" + "".join(f"  {r[k][0]:6.3f}±{r[k][1]:.3f}" for k in keys))

    if args.control:
        ctrl = next((r for n, r in res.items() if args.control in n), None)
        if ctrl:
            print(f"\n=== Δ vs control ({args.control}); winner = fg_mse@50 AND lpips@50 down beyond combined std, hierarchy not worse ===")
            for name, r in res.items():
                if args.control in name:
                    continue
                dfm = r["fg_mse@50"][0] - ctrl["fg_mse@50"][0]; sfm = r["fg_mse@50"][1] + ctrl["fg_mse@50"][1]
                dlp = r["lpips@50"][0] - ctrl["lpips@50"][0]; slp = r["lpips@50"][1] + ctrl["lpips@50"][1]
                win = (dfm < -sfm) and (dlp < -slp) and (r["top_kl"][0] >= 0.5 * ctrl["top_kl"][0] - 1e-6)
                print(f"  {name:22} Δfg_mse@50 {dfm:+.3f}(±{sfm:.3f})  Δlpips@50 {dlp:+.3f}(±{slp:.3f})  "
                      f"top_kl {r['top_kl'][0]:.3f}  => {'WIN' if win else 'no'}")
    print("\nGATE DONE ->", args.out)


if __name__ == "__main__":
    main()
