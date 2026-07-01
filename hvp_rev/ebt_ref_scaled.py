"""STAGE-1 FAIR-SCALE EBT RUN — hardens (or overturns) the small-scale EBT negative.

The 2026-07-02 forensic audit found every E(mean)<E(target) landscape probe ran on models trained
<= 2,000 steps = 3.3% of the paper's steps (inside its 10k warmup), 0.4% of its sample budget,
FFN 1x vs 4x, batch 32 vs 256, with the loss still falling at stop. This script removes the scale
variables while keeping the recipe THEIRS:

  --recipe s2   (default) the exact S2 recipe already replicated (ebt_s2_vae.sh semantics via
                RefEBT/RefTrainer from ebt_reference_control), scaled to the paper's xxs video
                config: d384 / 6 heads / 6 layers / FFN 4x / ctx 16 / eff-batch 128 (4 x micro 32
                + 16 replay rows per micro) / 20k steps / 5k warmup / cosine (min_lr_scale 10 over
                1M scheduling steps, their scheduler shape).
  --recipe s1   the recipe behind the paper's ACTUAL video generations (ebt_s1_vae.sh +
                ebt_ssv2_vae.sh inference), NEVER previously run through the control:
                detach-between-steps, create_graph=learning on EVERY step, loss = MEAN of per-step
                SmoothL1, NO replay, NO train langevin, mcmc_num_steps=2, LEARNABLE alpha in its
                own param group (lr = peak_lr * 90000 * alpha_init/30000, ratio-faithful), and the
                paper's inference: best-of-N by min energy, 4 steps on the final landscape,
                alpha_override = alpha/4 (their 7500/30000), langevin 1 at inference.
                --alpha_mode transplant uses their literal 30000; recalib (default) applies their
                own movement-calibration principle to OUR latent/grad scale. Run BOTH (cheap) --
                the calibration ambiguity is a known reviewer hole.

PROBES every --probe_every steps (the scoping-gate trend data):
  E(target) vs E(mean) vs E(randn) on the final landscape; cold-descent d_tgt + decoded fg-sharp;
  for s1 additionally the paper-recipe best-of-N min-energy sample's d_tgt + fg-sharp.
SCOPING GATE (per the review): if by 10k post-warmup steps there is NO E(target)<E(mean) flip AND
no sharpness trend, write the hardened negative; a flip/trend => scale-sensitivity finding.

  venv/Scripts/python.exe -u -m hvp_rev.ebt_ref_scaled --recipe s2 --steps 20000 --probe_steps 100
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from hvp_rev import hier_convae
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import ActionHierData, decode01, fg_sharp, save_ckpt
from hvp_rev.ebt_reference_control import (RefEBT, RefTrainer, probe, calibrate_alpha, _rms,
                                           TOKDIM)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPILL_SEC = 10.0


# ============================================================================================
# S1 recipe: detach between steps, create_graph EVERY step, per-step losses, learnable alpha.
# Mirrors EBT/model/vid/ebt.py forward default path (no no_mcmc_detach, no truncate_mcmc,
# langevin 0, no randomize flags -> mcmc_steps = [0, 1]).
# ============================================================================================
class RefEBTS1(RefEBT):
    def __init__(self, *a, **kw):
        kw.setdefault("langevin", 0.0)          # S1 trains WITHOUT langevin (script passes none)
        super().__init__(*a, **kw)
        self.alpha.requires_grad_(True)          # --mcmc_step_size_learnable

    def forward(self, real_original, actions, learning, replay_pred=None, no_randomness=True,
                n_steps=None, alpha_override=None, langevin_override=0.0):
        """S1 default path of EBT_VID.forward: detach+requires_grad each step (ebt.py:109),
        create_graph=learning on every step (ebt.py:129, truncate off), no train langevin.
        n_steps/alpha_override/langevin_override implement the paper's inference recipe
        (extra steps stay on the final landscape's time-embedding index, ebt.py:97 comment)."""
        B, S, _ = real_original.shape
        alpha = torch.clamp(self.alpha, min=1e-4)
        if alpha_override is not None:
            alpha = torch.tensor(float(alpha_override), device=real_original.device)
        real_down = self.encoder_down_projection(real_original)
        orig_pred = torch.randn_like(real_original)          # random_noise init
        if replay_pred is not None:
            orig_pred[B - replay_pred.shape[0]:] = replay_pred
        total = self.mcmc_num_steps if n_steps is None else int(n_steps)
        pred_list, energy_list = [], []
        with torch.set_grad_enabled(True):
            for i in range(total):
                mcmc_step = min(i, self.mcmc_num_steps - 1)   # extra infer steps: final landscape
                orig_pred = orig_pred.detach().requires_grad_().reshape(B, S, TOKDIM)
                pred_down = self.encoder_down_projection(orig_pred)
                pred_down = self._add_action(pred_down, actions)
                if langevin_override:
                    pred_down = pred_down + torch.randn_like(pred_down.detach()) * float(langevin_override)
                all_emb = torch.cat([real_down, pred_down], dim=1)
                energy = self.transformer(all_emb, start_pos=0, mcmc_step=mcmc_step).reshape(-1, 1)
                energy_list.append(energy)
                grad = torch.autograd.grad([energy.sum()], [orig_pred], create_graph=bool(learning))[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("NaN/Inf grad in MCMC")
                orig_pred = orig_pred - alpha * grad
                pred_list.append(orig_pred)
        return pred_list, energy_list


class RefTrainerS1:
    """forward_loss_wrapper, S1 branch: loss = mean over per-step recon losses (ebt.py:178-181)."""

    def __init__(self, model: RefEBTS1):
        self.m = model
        self.crit = nn.SmoothL1Loss(beta=1.0)

    def train_step(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        full = self.m._seq_from_ctx(ctx_c, ctx_f, tgt_c, tgt_f)
        real_in, gt = full[:, :-1], full[:, 1:]
        pred_list, _ = self.m(real_in, actions, learning=True, no_randomness=False)
        loss = sum(self.crit(p, gt) for p in pred_list) / len(pred_list)
        return loss


@torch.no_grad()
def s1_paper_inference_probe(m: RefEBTS1, test, ae, N=4, n_steps=4, seed=777):
    """The paper's actual video-generation recipe (ebt_ssv2_vae.sh): N samples, n_steps descent
    steps on the final landscape, alpha_override = alpha/4 (their 7500/30000 ratio), langevin 1
    at inference, pick per-element argmin final-frame energy. Returns (d_tgt, fg_sharp)."""
    b = test.batch(64, train=False, return_frames=True, seed=seed)
    full = m._seq_from_ctx(b["ctx_c"], b["ctx_f"], b["tgt_c"], b["tgt_f"])
    real_in, tgt_tok = full[:, :-1], full[:, -1]
    a_over = float(torch.clamp(m.alpha, min=1e-4)) / 4.0
    cands, energies = [], []
    for n in range(N):
        with torch.enable_grad():
            pred_list, _ = m(real_in, b["actions"], learning=False,
                             n_steps=n_steps, alpha_override=a_over, langevin_override=1.0)
        pred = pred_list[-1].detach()
        e = m.energy_of(real_in, pred, b["actions"], mcmc_step=m.mcmc_num_steps - 1)[:, -1, 0]
        cands.append(pred[:, -1])                 # target-frame token (B,1280)
        energies.append(e)
    E = torch.stack(energies, 0)                  # (N,B)
    pick = E.argmin(dim=0)                        # (B,)
    B = tgt_tok.shape[0]
    chosen = torch.stack([cands[pick[i]][i] for i in range(B)], 0)
    d_tgt = _rms(chosen - tgt_tok)
    _, zf = m._unflat(chosen)
    pred01 = decode01(ae, zf)
    gt01 = decode01(ae, b["tgt_f"])
    return d_tgt, fg_sharp(pred01, gt01)


def lr_at(s, peak, warmup, sched_total=1_000_000, min_scale=10.0):
    """Their scheduler shape: linear warmup then cosine from peak to peak/min_scale over
    max_scheduling_steps=1e6 (so a 20k run sits near-peak after warmup, faithfully)."""
    if s < warmup:
        return peak * s / max(1, warmup)
    prog = min(1.0, (s - warmup) / max(1, sched_total - warmup))
    return peak / min_scale + 0.5 * (peak - peak / min_scale) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=["s2", "s1"], default="s2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--micro", type=int, default=32, help="per-forward fresh batch (theirs: 32/device)")
    ap.add_argument("--accum", type=int, default=4, help="grad accumulation (eff fresh batch = micro*accum)")
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--ffn_mult", type=float, default=4.0)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 => auto-calibrate (movement principle)")
    ap.add_argument("--alpha_mode", choices=["recalib", "transplant"], default="recalib",
                    help="s1 only: recalib = their calibration principle on OUR scale; transplant = literal 30000")
    ap.add_argument("--probe_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--probe_steps", type=int, default=0, help=">0: timing/VRAM probe only, then exit")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    M._BF16 = bool(args.bf16)
    tag = f"{args.recipe}{args.tag}"

    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=args.ctx, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=args.ctx, k=4, n_val=250,
                          stats=data.stats)   # [STD-FIX] test standardized with TRAIN stats
    b0 = data.batch(1024, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean())
    sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    print(f"[standardize] post-datalayer coarse std {sc:.4f}  fine std {sf:.4f} (expect ~1)", flush=True)

    if args.recipe == "s2":
        m = RefEBT(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=args.ctx, k=4,
                   sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
                   randomize_step_scale=2.0, langevin=1.0, alpha=1.0, ffn_mult=args.ffn_mult).to(DEV)
        trainer = RefTrainer(m, use_replay=True, batch=args.micro)
        k_eval_steps = m.randomize_num_steps + 1
    else:
        m = RefEBTS1(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=args.ctx, k=4,
                     sc=sc, sf=sf, mcmc_num_steps=2, randomize_num_steps=0, randomize_num_steps_min=0,
                     randomize_step_scale=1.0, alpha=30000.0, ffn_mult=args.ffn_mult).to(DEV)
        trainer = RefTrainerS1(m)
        k_eval_steps = m.mcmc_num_steps

    # ---- alpha init ----
    bcal = data.batch(64, train=True)
    a_auto, gmag = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                   bcal["actions"], k_eval_steps)
    if args.recipe == "s2":
        a_init = a_auto if args.alpha == 0 else args.alpha
    else:
        a_init = (a_auto if args.alpha == 0 else args.alpha) if args.alpha_mode == "recalib" else 30000.0
    m.alpha.data.fill_(a_init)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    eff = args.micro * args.accum
    print(f"[model] {tag}: xxs-shape d{args.d}/{args.depth}L/{args.heads}H ffn{args.ffn_mult}x "
          f"ctx{args.ctx} params {n_params:.2f}M | alpha={a_init:.4g} ({args.alpha_mode if args.recipe=='s1' else 'S2-calib'}, "
          f"|gradE|={gmag:.4f}) | eff-batch {eff} ({args.micro}x{args.accum}) | steps {args.steps} "
          f"warmup {args.warmup} | probes every {args.probe_every}", flush=True)

    # ---- optimizer: alpha in its own param group (s1: lr = peak*90000*alpha_init/30000) ----
    base_params = [p for n, p in m.named_parameters() if n != "alpha" and p.requires_grad]
    groups = [{"params": base_params, "lr": args.lr, "weight_decay": 0.01}]
    if args.recipe == "s1":
        alpha_lr = args.lr * 90000.0 * (a_init / 30000.0)   # ratio-faithful to their 90000x on 30000
        groups.append({"params": [m.alpha], "lr": alpha_lr, "weight_decay": 0.0})
        print(f"[opt] alpha param group lr = {alpha_lr:.4g}", flush=True)
    opt = torch.optim.AdamW(groups)

    ckpt = f"logs_rhcp/ebt_scaled_{tag}.pt"
    start = 1
    if args.resume and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start = int(st["step"]) + 1
        print(f"[ckpt] resumed {ckpt} @ step {st['step']}", flush=True)

    # ---- timing probe mode ----
    if args.probe_steps > 0:
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        m.train(); t0 = time.time()
        for s in range(args.probe_steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                bt = data.batch(args.micro, train=True)
                loss = trainer.train_step(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"],
                                          bt["actions"]) / args.accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.probe_steps
        vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
        print(f"[TIMING] {dt:.3f} s/opt-step | peak VRAM {vram:.2f} GB | "
              f"{args.steps} steps => {args.steps*dt/3600:.1f} h", flush=True)
        return

    # ---- training with periodic probes ----
    pms = (m.mcmc_num_steps - 1) if args.recipe == "s1" else 0   # probe the FINAL landscape
    m.eval(); probe(m, trainer, test, ae, f"scaled_{tag}_step0", k_eval_steps, probe_mcmc_step=pms); m.train()
    run, t_last, gn = 0.0, time.time(), 0.0
    for s in range(start, args.steps + 1):
        lr = lr_at(s, args.lr, args.warmup)
        opt.param_groups[0]["lr"] = lr
        if args.recipe == "s1":
            opt.param_groups[1]["lr"] = lr / args.lr * (args.lr * 90000.0 * (a_init / 30000.0))
        opt.zero_grad(set_to_none=True)
        step_loss, bad = 0.0, False
        for _ in range(args.accum):
            bt = data.batch(args.micro, train=True)
            loss = trainer.train_step(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"],
                                      bt["actions"]) / args.accum
            if not torch.isfinite(loss):
                bad = True
                continue
            loss.backward()
            step_loss += float(loss.detach())
        if not bad:
            gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            opt.step()
            run += step_loss
        dt = time.time() - t_last; t_last = time.time()
        if dt > SPILL_SEC and s > start + 3:
            save_ckpt(ckpt, step=s, model=m.state_dict(), opt=opt.state_dict())
            raise RuntimeError(f"step {s} took {dt:.1f}s > {SPILL_SEC}s -> WDDM spill; aborting "
                               f"(ckpt saved). Shrink micro/accum and resume.")
        if s % 200 == 0:
            print(f"[train {s}/{args.steps}] sl1 {run/200:.4f} lr {lr:.2e} gnorm {gn:.3f} "
                  f"alpha {float(m.alpha.detach()):.4g} {dt:.2f}s/step", flush=True)
            run = 0.0
        if s % args.save_every == 0 or s == args.steps:
            save_ckpt(ckpt, step=s, model=m.state_dict(), opt=opt.state_dict())
        if s % args.probe_every == 0 or s == args.steps:
            m.eval()
            probe(m, trainer, test, ae, f"scaled_{tag}_step{s}", k_eval_steps, probe_mcmc_step=pms)
            if args.recipe == "s1":
                d_tgt, sh = s1_paper_inference_probe(m, test, ae)
                print(f"  [s1-paper-infer step{s}] best-of-4 min-energy: d_tgt {d_tgt:.3f} "
                      f"fg-sharp {sh:.4f} (GT~0.25) "
                      f"{'[COLLAPSED]' if sh < 0.08 else '[HAS STRUCTURE]'}", flush=True)
            m.train()
    print(f"\nSCALED {tag} DONE ({args.steps} steps, eff-batch {eff}). Scoping gate: check the "
          f"E(tgt)-vs-E(mean) and fg-sharp trends in the probe lines above, post-warmup.", flush=True)


if __name__ == "__main__":
    main()
