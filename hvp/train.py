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
from hvp.eval.quality import changed_region_metrics
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
        # MMSE mixture MEAN for the pixel metric (the mode handicaps an MSE/PSNR metric for a GMM
        # head); clamp the decode (the raw conv decoder can overshoot [-1,1], which would unfairly
        # deflate pred vs the in-range persistence/GT baseline).
        z_pred = _pred_to_latents(cascade, [p.float() for p in p_on], use_mean=True)
        recon_pred = tok.decoder.decode_depth(z_pred, n).float().clamp(-1, 1)
        pred, gt_next, persist = recon_pred[:, :, :-1], x[:, :, 1:], x[:, :, :-1]
        acc["val/pred_psnr"].append(psnr(pred, gt_next))
        acc["val/persist_psnr"].append(psnr(persist, gt_next))
        acc["val/pred_ssim"].append(video_ssim(pred, gt_next))
        acc["val/persist_ssim"].append(video_ssim(persist, gt_next))
        # changed-region (moving-pixel) PSNR: full-frame PSNR is dominated by the static background
        # where persistence is trivially perfect; the moving region is the honest progress signal.
        acc["val/pred_changed_psnr"].append(changed_region_metrics(pred, gt_next, persist)["changed_psnr"])
        acc["val/persist_changed_psnr"].append(changed_region_metrics(persist, gt_next, persist)["changed_psnr"])
        # oracle ceiling: GT next-frame latents through the SAME multi-rate shifted decode. The
        # single-step pixel ceiling is NOT the recon PSNR (~30); the temporally-shifted stack decode
        # caps it below that, so (pred-persist)/(oracle-persist) is the honest fraction captured.
        z_next = [torch.cat([zi[:, :, 1:], zi[:, :, -1:]], dim=2) for zi in z]   # position t holds frame t+1
        recon_oracle = tok.decoder.decode_depth(z_next, n).float().clamp(-1, 1)[:, :, :-1]
        acc["val/pred_psnr_oracle"].append(psnr(recon_oracle, gt_next))
        # finest-stage attribution: predicted BOT (temporally 1:1) with ORACLE next-frame coarse
        # context isolates how well the predicted finest latent renders the next frame.
        if n > 1:
            recon_bot = tok.decoder.decode_depth([z_pred[0]] + z_next[1:], n).float().clamp(-1, 1)[:, :, :-1]
            acc["val/pred_psnr_bot"].append(psnr(recon_bot, gt_next))
            acc["val/pred_changed_psnr_bot"].append(changed_region_metrics(recon_bot, gt_next, persist)["changed_psnr"])
        hk, hc = cfg.predictor.head, cfg.predictor.head_components
        for s in range(n):
            acc[f"val/erank_s{s}"].append(effective_rank(z[s]))
            acc[f"val/ve_on_s{s}"].append(variance_explained(p_on[s].float(), z[s], head=hk, n_components=hc)["ve_changed"])
            acc[f"val/ve_off_s{s}"].append(variance_explained(p_off[s].float(), z[s], head=hk, n_components=hc)["ve_changed"])
        if not rows:
            unc_gray = None
            if cfg.predictor.head in ("gmm", "gaussian"):
                # entropy/uncertainty map (white=certain, black=uncertain) from the distributional head
                from hvp.eval.uncertainty import (gray_to_strip, predict_with_uncertainty,
                                                  uncertainty_to_gray)
                _, std_u = predict_with_uncertainty(tok, cascade, x[:cfg.run.n_log_images], n_samples=8)
                unc_gray = uncertainty_to_gray(std_u[:, :-1])
            for j in range(min(cfg.run.n_log_images, x.shape[0])):
                rows.append((j, "GT", frames_to_strip(x[j, :, 1:])))
                rows.append((j, "Recon", frames_to_strip(out.recon[j, :, 1:].float().clamp(-1, 1))))
                rows.append((j, "Pred", frames_to_strip(recon_pred[j, :, :-1])))
                if unc_gray is not None:
                    rows.append((j, "Uncertainty", gray_to_strip(unc_gray[j])))
    tok.train(); cascade.train()
    res = {k: sum(v) / max(len(v), 1) for k, v in acc.items()}
    res["val/pred_psnr_lift"] = res["val/pred_psnr"] - res["val/persist_psnr"]
    res["val/pred_ssim_lift"] = res["val/pred_ssim"] - res["val/persist_ssim"]
    res["val/pred_changed_lift"] = res["val/pred_changed_psnr"] - res["val/persist_changed_psnr"]
    denom = res.get("val/pred_psnr_oracle", 0.0) - res["val/persist_psnr"]
    if denom > 1e-6:
        res["val/pred_oracle_frac"] = (res["val/pred_psnr"] - res["val/persist_psnr"]) / denom
    if "val/pred_psnr_bot" in res:
        res["val/pred_psnr_bot_lift"] = res["val/pred_psnr_bot"] - res["val/persist_psnr"]
    for s in range(n):
        res[f"val/cascade_lift_s{s}"] = res[f"val/ve_on_s{s}"] - res[f"val/ve_off_s{s}"]
    # canary: a non-finite val metric means the model has silently corrupted — abort the run loudly.
    for k in ("val/recon_psnr", "val/pred_psnr"):
        if not math.isfinite(res.get(k, 0.0)):
            raise FloatingPointError(f"[validate] non-finite {k} — model corrupted, aborting")
    return res, rows


def fit_oneshot(cfg: Config, max_steps: int | None = None, log_every: int | None = None,
                resume: str | None = None) -> Dict:
    device = device_of(cfg)
    seed_everything(cfg.optim.seed)
    tok = build_tokenizer(cfg).to(device).train()
    cascade = HierarchicalCascade(cfg.topology, cfg.predictor).to(device).train()
    n_params = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in cascade.parameters())
    tr, va = _loaders(cfg)
    tok_params, cas_params = list(tok.parameters()), list(cascade.parameters())
    params = tok_params + cas_params
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

    def _infinite(loader):
        # Re-iterate each epoch so shuffle=True actually RESHUFFLES. itertools.cycle would cache
        # one epoch in RAM and replay a FIXED order forever — wrong (and memory-heavy at 128px)
        # for a long multi-epoch run.
        while True:
            for b in loader:
                yield b

    data_iter = _infinite(tr)
    accum = max(1, cfg.optim.accum_steps)          # microbatches per optimizer step (effective bs)
    t0 = time.time()
    last = None
    last_ckpt_t = t0
    consec_skips = 0
    for step in range(start_step, steps):
        cur_lr = lr_at(step, cfg.optim, steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        # gradient accumulation: sum grads over `accum` microbatches, then one optimizer step.
        m = None
        n_good = 0
        for _ in range(accum):
            x = next(data_iter)["video"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                total, m_i, _, _ = compute_oneshot_loss(tok, cascade, x, cfg, step)
            if not torch.isfinite(total):
                continue
            (total / accum).backward()
            m, n_good = m_i, n_good + 1
        if n_good == 0:
            consec_skips += 1
            print(f"[fit] step {step}: all {accum} microbatch(es) non-finite loss, skipping ({consec_skips})")
            if consec_skips >= 50:
                raise FloatingPointError(f"[fit] aborting: {consec_skips} consecutive non-finite steps")
            continue
        gn = grad_norm(params)
        if not math.isfinite(gn):
            # bf16 has no GradScaler; a non-finite GRAD (loss can still be finite) would be
            # written into every weight by opt.step() and silently corrupt the whole run.
            # clip_grad_norm_ does NOT stop NaN, so guard here and skip the step.
            consec_skips += 1
            print(f"[fit] step {step}: non-finite grad (norm={gn}), skipping ({consec_skips})")
            opt.zero_grad(set_to_none=True)
            if consec_skips >= 50:
                raise FloatingPointError(f"[fit] aborting: {consec_skips} consecutive non-finite steps")
            continue
        consec_skips = 0
        if cfg.optim.grad_clip > 0:
            # clip the tokenizer and cascade SEPARATELY: a large predictor/rollout gradient must not
            # throttle the recon (tokenizer) gradient through a shared global norm (P2.1 showed a big
            # cascade gradient regressing recon ~1.9 dB under joint clipping).
            torch.nn.utils.clip_grad_norm_(tok_params, cfg.optim.grad_clip)
            torch.nn.utils.clip_grad_norm_(cas_params, cfg.optim.grad_clip)
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
            _save_val_strips(vrows, cfg.run.out_dir, step)
            print(f"        [val@{step}] recon {vmet['val/recon_psnr']:.2f}dB ssim {vmet['val/recon_ssim']:.3f} "
                  f"| pred {vmet['val/pred_psnr']:.2f} (lift {vmet['val/pred_psnr_lift']:+.2f}, "
                  f"changed-lift {vmet.get('val/pred_changed_lift', 0):+.2f}, oracle-frac "
                  f"{vmet.get('val/pred_oracle_frac', 0):.2f}) | bot {vmet.get('val/pred_psnr_bot', 0):.2f}dB "
                  f"| mid_casc {vmet['val/cascade_lift_s1']:+.3f}")
        # Durable, step-stamped periodic checkpoints. A run that never reaches max_steps
        # (slow / under-progressed) must still leave a recoverable copy — last.pt alone is
        # overwritten by the next run (symptom S2). Save on a step boundary OR a wall-clock
        # timer (so a run too slow to cross a save_every boundary still checkpoints).
        due_step = cfg.run.save_every > 0 and step % cfg.run.save_every == 0
        due_time = (cfg.run.save_every_min > 0
                    and (time.time() - last_ckpt_t) >= cfg.run.save_every_min * 60.0)
        if step > 0 and (due_step or due_time):
            _save(tok, cascade, opt, step, cfg, stamped=True)
            last_ckpt_t = time.time()

    vmet, vrows = validate(tok, cascade, va, cfg, device)
    logger.log(vmet, steps)
    logger.log_image_table(vrows, steps)
    _save_val_strips(vrows, cfg.run.out_dir, steps)
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
        # retain a step-stamped copy so a headline number (or an interrupted run) is never
        # silently overwritten by a later run's last.pt
        save_checkpoint(out / f"step{step:06d}.pt", tok=tok, cascade=cascade, opt=opt, step=step, cfg=cfg)
        _prune_stamped(out, cfg.run.max_checkpoints_keep)
    return path


def _prune_stamped(out: Path, keep: int) -> None:
    """Keep only the `keep` newest stepNNNNNN.pt (each is ~265 MB); keep<=0 keeps all.
    The 6-digit zero-padded step makes lexicographic == numeric order, so newest sort last."""
    if keep <= 0:
        return
    stamped = sorted(out.glob("step*.pt"))
    for stale in stamped[:-keep]:
        stale.unlink(missing_ok=True)


def _save_val_strips(vrows, out_dir, step) -> None:
    """Persist the validation image rows (GT/Recon/Pred/Uncertainty) as PNGs on disk, so the
    entropy/uncertainty maps are reviewable as files independent of wandb. One stacked PNG per
    logged clip (rows top-to-bottom = GT, Recon, Pred, Uncertainty[white=certain/black=uncertain])."""
    try:
        from PIL import Image
    except Exception:
        return
    from collections import defaultdict
    by_idx: Dict[int, list] = defaultdict(list)
    for idx, _label, strip in vrows:
        by_idx[idx].append(strip)
    d = Path(out_dir) / "val_strips"
    d.mkdir(parents=True, exist_ok=True)
    for idx, strips in by_idx.items():
        w = max(s.shape[1] for s in strips)
        rows = [s if s.shape[1] == w else np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0))) for s in strips]
        img = np.concatenate(rows, axis=0)
        Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(
            d / f"step{step:06d}_img{idx:02d}.png")


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
