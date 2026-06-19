"""One-shot trainer: co-train the VQ-free tokenizer (recon + VICReg) and the single-step
cascade (detached predictive target). Logs everything to wandb (scalars + a GT/Pred/Recon
image table), reports held-out PSNR/SSIM, a collapse canary (effective rank), per-stage
predictive VE with the cascade ON vs OFF, and pixel-space single-step prediction lift."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from hvp.checkpoint import load_checkpoint, save_checkpoint, seed_everything
from hvp.config import Config
from hvp.data import dataset_from_cfg
from hvp.metrics import frames_to_strip, psnr, video_ssim
from hvp.objectives import compute_oneshot_loss, compute_vae_loss
from hvp.predictor.cascade import HierarchicalCascade
from hvp.predictor.continuous import variance_explained
from hvp.predictor.rollout import _pred_to_latents
from hvp.tokenizer import build_tokenizer
from hvp.wandblog import WandbLogger


def device_of(cfg: Config) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective_rank(z: torch.Tensor) -> float:
    c = z.shape[1]
    f = z.permute(0, 2, 3, 4, 1).reshape(-1, c).float()
    f = f - f.mean(dim=0, keepdim=True)
    cov = (f.T @ f) / max(1, f.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = ev.sum()
    return float((s * s) / (ev.pow(2).sum() + 1e-12)) if s > 0 else 0.0


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def _loaders(cfg: Config):
    common = dict(batch_size=cfg.optim.batch_size, num_workers=cfg.data.num_workers, pin_memory=True)
    tr = DataLoader(dataset_from_cfg(cfg, "train"), shuffle=True, drop_last=True, **common)
    va = DataLoader(dataset_from_cfg(cfg, "val"), shuffle=False, drop_last=False, **common)
    return tr, va


def build_optimizer(params, oc):
    kw = dict(lr=oc.lr, betas=tuple(oc.betas), eps=oc.eps, weight_decay=oc.weight_decay)
    return torch.optim.AdamW(params, **kw) if oc.optimizer == "adamw" else torch.optim.Adam(params, **kw)


def lr_at(step: int, oc, total: int) -> float:
    if oc.warmup_steps > 0 and step < oc.warmup_steps:
        return oc.lr * (step + 1) / oc.warmup_steps
    if oc.lr_schedule in ("cosine", "warmup_cosine"):
        s0 = oc.warmup_steps if oc.lr_schedule == "warmup_cosine" else 0
        prog = min(1.0, max(0.0, (step - s0) / max(1, total - s0)))
        return oc.min_lr + (oc.lr - oc.min_lr) * 0.5 * (1 + math.cos(math.pi * prog))
    return oc.lr


@torch.no_grad()
def validate(tok, cascade, val_loader, cfg, device, max_batches=None) -> Tuple[Dict[str, float], List]:
    tok.eval(); cascade.eval()
    max_batches = cfg.run.val_max_batches if max_batches is None else max_batches
    n = tok.topology.n_stages()
    acc: Dict[str, list] = defaultdict(list)
    rows: List[Tuple[int, str, np.ndarray]] = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch["video"].to(device)
        _, m, out = compute_vae_loss(tok, x, cfg, step=10**9)
        acc["val/recon_psnr"].append(m["psnr_full"])
        acc["val/recon_ssim"].append(video_ssim(out.recon.float(), x))
        z = [zi.float() for zi in out.z]
        p_on = cascade(z, use_cascade=True)
        p_off = cascade(z, use_cascade=False)
        z_pred = _pred_to_latents(cascade, [p.float() for p in p_on])
        recon_pred = tok.decoder.decode_depth(z_pred, n).float()
        pred, gt_next, persist = recon_pred[:, :, :-1], x[:, :, 1:], x[:, :, :-1]
        acc["val/pred_psnr"].append(psnr(pred, gt_next))
        acc["val/persist_psnr"].append(psnr(persist, gt_next))
        acc["val/pred_ssim"].append(video_ssim(pred, gt_next))
        acc["val/persist_ssim"].append(video_ssim(persist, gt_next))
        hk, hc = cfg.predictor.head, cfg.predictor.head_components
        for s in range(n):
            acc[f"val/erank_s{s}"].append(effective_rank(z[s]))
            acc[f"val/ve_on_s{s}"].append(variance_explained(p_on[s].float(), z[s], head=hk, n_components=hc)["ve_changed"])
            acc[f"val/ve_off_s{s}"].append(variance_explained(p_off[s].float(), z[s], head=hk, n_components=hc)["ve_changed"])
        if not rows:
            for j in range(min(cfg.run.n_log_images, x.shape[0])):
                rows.append((j, "GT", frames_to_strip(x[j, :, 1:])))
                rows.append((j, "Recon", frames_to_strip(out.recon[j, :, 1:].float())))
                rows.append((j, "Pred", frames_to_strip(recon_pred[j, :, :-1])))
    tok.train(); cascade.train()
    res = {k: sum(v) / max(len(v), 1) for k, v in acc.items()}
    res["val/pred_psnr_lift"] = res["val/pred_psnr"] - res["val/persist_psnr"]
    res["val/pred_ssim_lift"] = res["val/pred_ssim"] - res["val/persist_ssim"]
    for s in range(n):
        res[f"val/cascade_lift_s{s}"] = res[f"val/ve_on_s{s}"] - res[f"val/ve_off_s{s}"]
    return res, rows


def fit_oneshot(cfg: Config, max_steps: int | None = None, log_every: int | None = None,
                resume: str | None = None) -> Dict:
    device = device_of(cfg)
    seed_everything(cfg.optim.seed)
    tok = build_tokenizer(cfg).to(device).train()
    cascade = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).train()
    n_params = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in cascade.parameters())
    tr, va = _loaders(cfg)
    params = list(tok.parameters()) + list(cascade.parameters())
    opt = build_optimizer(params, cfg.optim)
    steps = max_steps or cfg.optim.max_steps
    log_every = log_every or cfg.run.log_every
    use_amp = device.type == "cuda" and cfg.optim.amp == "bf16"
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, tok=tok, cascade=cascade, opt=opt, device=device) + 1
        print(f"[fit] resumed from {resume} at step {start_step}")
    logger = WandbLogger(cfg, cfg.run)
    print(f"[fit] device={device} params={n_params/1e6:.2f}M steps={start_step}->{steps} "
          f"bs={cfg.optim.batch_size} amp={'bf16' if use_amp else 'off'} wandb={logger.enabled}")

    data_iter = itertools.cycle(tr)
    t0 = time.time()
    last = None
    for step in range(start_step, steps):
        cur_lr = lr_at(step, cfg.optim, steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x = next(data_iter)["video"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total, m, _, _ = compute_oneshot_loss(tok, cascade, x, cfg, step)
        if not torch.isfinite(total):
            print(f"[fit] step {step}: non-finite loss, skipping")
            continue
        total.backward()
        gn = grad_norm(params)
        if not math.isfinite(gn):
            # bf16 has no GradScaler; a non-finite GRAD (loss can still be finite) would be
            # written into every weight by opt.step() and silently corrupt the whole run.
            # clip_grad_norm_ does NOT stop NaN, so guard here and skip the step.
            print(f"[fit] step {step}: non-finite grad (norm={gn}), skipping")
            opt.zero_grad(set_to_none=True)
            continue
        if cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
        opt.step()
        last = m
        if step % log_every == 0 or step == steps - 1:
            ips = (step + 1) / (time.time() - t0)
            scal = {f"train/{k}": v for k, v in m.items()}
            scal.update({"train/grad_norm": gn, "train/it_per_s": ips, "train/lr": cur_lr})
            logger.log(scal, step)
            ps = " ".join(f"p{s}={m.get(f'pred_s{s}',0):.3f}" for s in range(cfg.topology.n_stages()))
            roll = f" roll {m['rollout_loss']:.2f} lr {m['lambda_rollout']:.3f} sf-tf {m['sf_tf_ar_proxy']:+.3f}" if "sf_tf_ar_proxy" in m else ""
            print(f"[{step:5d}] loss {m['loss']:.3f} recon {m['recon']:.3f} psnr {m['psnr_full']:5.2f} "
                  f"d{m['active_depth']} lam {m['lambda']:.2f} gn {gn:5.1f} pred[{ps}]{roll}")
        if step > 0 and (step % cfg.run.val_every == 0):
            vmet, vrows = validate(tok, cascade, va, cfg, device)
            logger.log(vmet, step)
            logger.log_image_table(vrows, step)
            print(f"        [val@{step}] recon_psnr {vmet['val/recon_psnr']:.2f} ssim {vmet['val/recon_ssim']:.3f} "
                  f"| pred_lift {vmet['val/pred_psnr_lift']:+.2f}dB | mid_casc_lift {vmet['val/cascade_lift_s1']:+.3f}")
        if cfg.run.save_every > 0 and step > 0 and step % cfg.run.save_every == 0:
            _save(tok, cascade, opt, step, cfg)

    vmet, vrows = validate(tok, cascade, va, cfg, device)
    logger.log(vmet, steps)
    logger.log_image_table(vrows, steps)
    dt = time.time() - t0
    n = cfg.topology.n_stages()
    names = ["bot", "mid", "top"]
    print(f"[fit] done {steps} steps in {dt:.1f}s ({steps/dt:.1f} it/s)")
    print(f"  VAL recon: psnr {vmet['val/recon_psnr']:.2f} ssim {vmet['val/recon_ssim']:.3f}")
    print(f"  VAL pred : psnr {vmet['val/pred_psnr']:.2f} (persist {vmet['val/persist_psnr']:.2f}, "
          f"lift {vmet['val/pred_psnr_lift']:+.2f}dB) ssim {vmet['val/pred_ssim']:.3f} "
          f"(lift {vmet['val/pred_ssim_lift']:+.3f})")
    print(f"  {'stage':5s} {'ve_cascade':>10s} {'ve_indep':>9s} {'lift':>7s} {'erank':>6s}")
    for s in range(n):
        print(f"  {names[s]:5s} {vmet[f'val/ve_on_s{s}']:10.3f} {vmet[f'val/ve_off_s{s}']:9.3f} "
              f"{vmet[f'val/cascade_lift_s{s}']:+7.3f} {vmet[f'val/erank_s{s}']:6.1f}")
    ckpt = _save(tok, cascade, opt, steps - 1, cfg, stamped=True)
    logger.finish()
    return {"ckpt": ckpt, **vmet}


def _save(tok, cascade, opt, step, cfg, stamped: bool = False) -> str:
    out = Path(cfg.run.out_dir) / "oneshot"
    path = save_checkpoint(out / "last.pt", tok=tok, cascade=cascade, opt=opt, step=step, cfg=cfg)
    if stamped:
        # retain a step-stamped copy so a headline number is never silently overwritten by a later run
        save_checkpoint(out / f"step{step:06d}.pt", tok=tok, cascade=cascade, opt=opt, step=step, cfg=cfg)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    ap.add_argument("--set", nargs="*", default=[], help="dot-path overrides, e.g. optim.lr=1e-3")
    args = ap.parse_args()
    if args.config:
        cfg = Config.from_yaml(args.config, args.set)
    else:
        from hvp.config import apply_cli_overrides
        cfg = Config.from_dict(apply_cli_overrides({}, args.set))
    cfg.validate()
    fit_oneshot(cfg, max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
