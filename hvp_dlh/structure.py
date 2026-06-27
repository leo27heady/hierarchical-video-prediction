"""Dynamic-structure diagnostics for a trained DLH checkpoint — the figures that show the mechanism:
per-level state-KL (the auto-collapse signature, paper App. C Table 4), the two decision KLs
(KL->change vs KL->static, which drive ticking), tick rate per level, the K(t) trajectory over an
open-loop rollout, and a GT-vs-prediction strip. Usage:
  python -m hvp_dlh.structure --ckpt logs_dlh/ball-2lvl-v3_last.pt --out logs_dlh/structure
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_dlh.data import make_loaders


@torch.no_grad()
def teacher_forced_stats(model, obs):
    """Run the posterior unroll and collect per-level KL, tick rate, and the two decision KLs."""
    cfg = model.cfg
    B, T = obs.shape[:2]
    feats = model._bottom_up(obs)
    state = model._init_state(B, obs.device)
    N = cfg.levels
    skl = np.zeros(N); tick = np.zeros(N); klc = np.zeros(N); kls = np.zeros(N); cnt = np.zeros(N)
    for t in range(T):
        x_t = [feats[n][:, t] for n in range(N)]
        _, state, info = model._step(state, x_t, mode="posterior")
        for n in range(N):
            skl[n] += info["skl_level"][n].mean().item()
            tick[n] += info["e"][n].mean().item()
            kc, ks = info["kl_dec"][n]
            if kc is not None:
                klc[n] += kc.mean().item(); kls[n] += ks.mean().item(); cnt[n] += 1
    skl /= T; tick /= T
    klc = np.where(cnt > 0, klc / np.maximum(cnt, 1), np.nan)
    kls = np.where(cnt > 0, kls / np.maximum(cnt, 1), np.nan)
    return skl, tick, klc, kls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="logs_dlh/structure")
    ap.add_argument("--clips", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sd = torch.load(args.ckpt, map_location=dev)
    cfg = DLHConfig(**sd["cfg"]); model = DLH(cfg).to(dev)
    model.load_state_dict(sd["model"]); model.eval()
    print(f"loaded {args.ckpt} (step {sd.get('step','?')}) | {cfg.levels}-level {cfg.dataset}")

    _, test = make_loaders(cfg)
    clips = []
    if hasattr(test, "__len__"):
        clips = [test[i][:cfg.ctx_frames + cfg.eval_horizon] for i in range(min(args.clips, len(test)))]
    else:
        it = iter(test); clips = [next(it)[:cfg.ctx_frames + cfg.eval_horizon] for _ in range(args.clips)]
    obs = torch.stack(clips, 0).to(dev)

    # ---- teacher-forced structure ----
    skl, tick, klc, kls = teacher_forced_stats(model, obs)
    print("\n== per-level structure (teacher-forced) ==")
    print(f"{'level':>6} {'state_KL':>10} {'tick_rate':>10} {'KL->change':>12} {'KL->static':>12}")
    for n in range(cfg.levels):
        print(f"{n:>6} {skl[n]:>10.3f} {tick[n]:>10.3f} {klc[n]:>12.3f} {kls[n]:>12.3f}"
              + ("   <- bottom (always ticks)" if n == 0 else ""))
    print("  (decision: tick when KL->change < KL->static; collapse signature = top levels' state_KL -> 0)")

    # ---- open-loop K(t) + rollout ----
    pred, K = model.open_loop(obs, cfg.ctx_frames, cfg.eval_horizon)
    Kt = K.float().mean(0).cpu().numpy()
    print(f"\n== open-loop == mean K over horizon {Kt.mean():.3f}  (range {Kt.min():.2f}-{Kt.max():.2f})")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # per-level KL bar (collapse signature)
        plt.figure(figsize=(4, 3)); plt.bar(range(cfg.levels), skl)
        plt.xlabel("level (0=bottom)"); plt.ylabel("state KL / frame"); plt.title("per-level KL (collapse signature)")
        plt.tight_layout(); plt.savefig(out / "per_level_kl.png", dpi=80); plt.close()
        # K(t)
        plt.figure(figsize=(5, 2.5)); plt.plot(range(1, len(Kt) + 1), Kt)
        plt.xlabel("predicted frame"); plt.ylabel("update depth K"); plt.title("adaptive clock K(t)")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out / "K_trajectory.png", dpi=80); plt.close()
        # rollout strip (GT over pred) at several horizons
        g = obs[:, cfg.ctx_frames:].cpu().numpy(); p = pred.cpu().numpy()
        H = g.shape[1]; idx = [i for i in [0, H // 8, H // 4, H // 2, 3 * H // 4, H - 1] if i < H]
        rows = []
        for c in range(min(4, obs.shape[0])):
            rows.append(np.concatenate([np.concatenate([g[c, t, 0], p[c, t, 0]], 0) for t in idx], 1))
        strip = np.concatenate(rows, 0)
        plt.figure(figsize=(len(idx) * 1.5, 4)); plt.imshow(np.clip(strip, 0, 1), cmap="gray", vmin=0, vmax=1)
        plt.axis("off"); plt.title("rows=clips; each col GT(top)/pred(bottom) @ frames " + str([i + 1 for i in idx]))
        plt.tight_layout(); plt.savefig(out / "rollout_strip.png", dpi=80); plt.close()
        print(f"saved figures to {out}/")
    except Exception as e:
        print("plot err", e)


if __name__ == "__main__":
    main()
