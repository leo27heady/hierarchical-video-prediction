"""Train DLH. Sequential, gated, fully logged to Weights & Biases (metrics, per-level KL + clock
usage, open-loop SSIM/PSNR + honest baselines, sample GT-vs-pred strips, K-trajectory plots).

Usage:
  python -m hvp_dlh.train --dataset moving_ball --levels 2 --iters 30000 --run-name ball-2lvl
  python -m hvp_dlh.train --dataset mmnist --levels 3 --batch-size 24 --iters 80000 --run-name mmnist-3lvl
"""
from __future__ import annotations

import argparse
import io
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_dlh.data import make_loaders
from hvp_dlh.eval import evaluate


def build_cfg():
    p = argparse.ArgumentParser()
    for f in fields(DLHConfig):                                   # expose every config field as a CLI flag
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(f.default, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(f.default, (tuple, list)):
            p.add_argument(flag, type=str, default=None)          # comma-separated
        else:
            p.add_argument(flag, type=type(f.default), default=None)
    args = p.parse_args()
    cfg = DLHConfig()
    for f in fields(DLHConfig):
        v = getattr(args, f.name)
        if v is None:
            continue
        if isinstance(f.default, (tuple, list)) and isinstance(v, str):
            elem = type(f.default[0]) if len(f.default) else float  # infer int vs float from the default
            v = tuple(elem(x) for x in v.split(","))
        setattr(cfg, f.name, v)
    return cfg


def infinite(loader):
    while True:
        for b in loader:
            yield b


def _curve_fig(y, title, xlabel="frame"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 2.5)); plt.plot(range(1, len(y) + 1), y); plt.title(title)
    plt.xlabel(xlabel); plt.grid(alpha=0.3); plt.tight_layout()
    return fig


def main():
    cfg = build_cfg()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    Path(cfg.logdir).mkdir(parents=True, exist_ok=True)

    import wandb
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, mode=cfg.wandb_mode, config=cfg.to_dict())

    model = DLH(cfg).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"DLH {cfg.levels}-level | params {nparams/1e6:.2f}M | dataset {cfg.dataset} | device {dev}")
    wandb.summary["params_M"] = nparams / 1e6
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    start_step = 0
    if cfg.auto_resume and not cfg.resume:                         # idempotent re-launch
        cand = Path(cfg.logdir) / f"{cfg.run_name}_last.pt"
        if cand.exists():
            cfg.resume = str(cand)
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {cfg.resume} at step {start_step}"
              + ("" if "opt" in ck else " (model-only; optimizer re-initialised)"), flush=True)

    train_loader, test = make_loaders(cfg)
    it = infinite(train_loader)

    for step in range(start_step + 1, cfg.total_iters + 1):
        beta = cfg.beta_max * min(1.0, step / max(1, cfg.beta_warmup_iters))
        force_tick = step <= cfg.force_tick_iters                  # dense warmup, then learned selection
        obs = next(it).to(dev)
        loss, met = model.loss(obs, beta=beta, force_tick=force_tick)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0:
            log = {"train/loss": met["loss"].item(), "train/nll": met["nll"].item(),
                   "train/state_kl": met["state_kl"].item(), "train/trans_kl": met["trans_kl"].item(),
                   "train/K_mean": met["K_mean"].item(), "train/beta": beta,
                   "train/force_tick": float(force_tick),
                   "train/grad_norm": float(gnorm), "step": step}
            for n, (e, sk) in enumerate(zip(met["e_per_level"].tolist(), met["skl_per_level"].tolist())):
                log[f"clock/e_L{n}"] = e            # fraction of steps level n ticks (adaptive clock usage)
                log[f"kl/state_L{n}"] = sk          # per-level state KL (the collapse signature)
            wandb.log(log)
            print(f"[{step:6d}] loss {met['loss'].item():8.1f} nll {met['nll'].item():7.1f} "
                  f"sKL {met['state_kl'].item():.3f} tKL {met['trans_kl'].item():.3f} "
                  f"K {met['K_mean'].item():.2f} e {[round(x,2) for x in met['e_per_level'].tolist()]} "
                  f"beta {beta:.2f}", flush=True)

        if step % cfg.eval_every == 0:
            ev = evaluate(model, test, cfg, n_clips=cfg.eval_clips, n_samples=cfg.eval_samples, device=dev)
            _arr = ("strip",) + tuple(k for k in ev if isinstance(ev[k], np.ndarray) or isinstance(ev[k], list))
            elog = {f"eval/{k}": v for k, v in ev.items() if k not in _arr}
            elog["step"] = step
            for n, (e, sk) in enumerate(zip(ev["e_per_level"], ev["skl_per_level"])):
                elog[f"eval/e_L{n}"] = e; elog[f"eval/kl_L{n}"] = sk
            elog["eval/strip"] = wandb.Image(np.clip(ev["strip"], 0, 1), caption="rows=clips; per col GT/pred")
            for cv, ttl in [("fg_mse_curve", "fg-MSE(t)"), ("lpips_fg_curve", "fg-LPIPS(t)"),
                            ("ssim_curve", "SSIM(t)"), ("K_curve", "K(t)")]:
                if cv in ev:
                    elog[f"eval/{cv}"] = wandb.Image(_curve_fig(ev[cv], ttl))
            wandb.log(elog)
            print(f"  EVAL[{step}] fg_MSE {ev['fg_mse']:.3f} (black {ev['fg_mse_black']:.3f} "
                  f"persist {ev['fg_mse_persist']:.3f}) | LPIPS_fg {ev.get('lpips_fg',-1):.3f} "
                  f"(black {ev.get('lpips_fg_black',-1):.3f}) | SSIM {ev['ssim']:.3f} | "
                  f"lapvar {ev.get('lapvar',-1):.4f}/{ev.get('lapvar_gt',-1):.4f} | "
                  f"KL/lvl {[round(x,2) for x in ev['skl_per_level']]} K_tf {ev['K_mean_tf']:.2f}", flush=True)

        if step % cfg.ckpt_every == 0 or step == cfg.total_iters:
            ck = Path(cfg.logdir) / f"{cfg.run_name}_last.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "cfg": cfg.to_dict(), "step": step}, ck)

    wandb.finish()


if __name__ == "__main__":
    main()
