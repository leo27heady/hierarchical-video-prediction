"""MEMORY-PAYOFF GATE — the cheapest decisive experiment for the article's memory-horizon headline,
run BEFORE building any rollout machinery. Question: does long temporal context measurably lower
fg-MSE on RE-ENTRY frames (high novel-fg-at-6) vs a short context (n_ctx=6)?

ARMS (2026-07-02 red-team patch — the original two-arm design tested "long context helps", NOT the
headline mechanism "long COARSE context at a token budget flat can't afford"):
  nctx6     : short baseline, 6 coarse + 6 fine frames                  (560 tokens)
  coarseL   : L coarse frames + only the LAST 6 fine frames             (L*16 + 6*64 + 80 tokens)
              <- THE HEADLINE ARM: long memory lives in the CHEAP coarse level only
  bothL     : L coarse + L fine (--with_longboth; expensive ceiling arm, L*80+80 tokens)

Also patched per red-team: paired bootstrap CIs on the per-bin delta (the hard bin is small — a
bare-mean kill decision was statistically unpowered); eval widened to ~768 samples; the eval viz
frames now REUSE the scored samples (the original drew NEW samples for the grid, so the grid did
not show the frames that were scored); per-arm atomic checkpoints + auto-resume; WDDM-spill guard
(a slow step on this machine means silent shared-memory spill, not load — abort, don't crawl).

If the coarseL arm does NOT beat nctx6 on the hard re-entry bin (CI excluding 0, above the
~0.011 decoder-blur floor), the memory-horizon headline is dead -> re-scope the article to
mechanism-study + EBT negative + Ch. D. Run 2 seeds before invoking the KILL branch.

  venv/Scripts/python.exe -u -m hvp_rev.mem_gate --long 48 --steps 6000 --batch 16 --bf16 --seed 0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from hvp_rev import hier_convae, viz
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import (ActionHierData, Diffusion, decode01, fg_mse_px_frames, fg_sharp,
                                load_ckpt, save_ckpt)
from hvp_rev.faithful_nets import DiTDenoiser
from hvp_rev.step1 import rel_positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V = 64
LAT = 16
SPILL_SEC = 10.0  # a single step slower than this on a 4060 Ti = WDDM shared-memory spill


def novel_fg6(data, ci, tj, n_ctx=6, thr=0.5):
    """Fraction of the TARGET (tj+k-1) sprite columns whose world-col was NOT covered by the ACTUAL
    short context [tj-n_ctx .. tj-1]. This is the re-entry hardness FOR A 6-FRAME model, used as
    the shared stratification axis for ALL arms."""
    rel = rel_positions(data.acts[ci], data.speed)
    tgt = tj + data.k - 1
    frame = data.clips_u8[ci, tgt].astype(np.float32) / 255.0
    fg = np.where(frame.max(axis=0) > thr)[0]
    if len(fg) == 0:
        return 0.0
    tw = (rel[tgt] + fg) % data.W
    cov = np.zeros(data.W, bool)
    for m in range(tj - n_ctx, tj):
        cov[(rel[m] + np.arange(V)) % data.W] = True
    return float((~cov[tw]).mean())


def gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=True, seed=None, frames=False):
    """Sample tj in [maxc, T-k); coarse context = last n_ctx_c frames, fine context = last n_ctx_f
    frames (n_ctx_f <= n_ctx_c); target = tj+k-1. Shared seed => identical (clip,target) across
    arms => per-sample PAIRED deltas. Always returns novel_fg-at-6 for stratification."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pool = data.train_idx if train else data.val_idx
    ctx_c = torch.empty(B, n_ctx_c, LAT, 4, 4); ctx_f = torch.empty(B, n_ctx_f, LAT, 8, 8)
    tgt_c = torch.empty(B, LAT, 4, 4); tgt_f = torch.empty(B, LAT, 8, 8)
    acts = torch.empty(B, data.k, dtype=torch.long)
    nf6 = torch.empty(B)
    gt = torch.empty(B, 1, 64, 64) if frames else None
    lastf = torch.empty(B, 1, 64, 64) if frames else None
    for b in range(B):
        ci = int(pool[rng.randint(len(pool))])
        tj = int(rng.randint(maxc, data.T - data.k))
        ti = tj + data.k - 1
        ctx_c[b] = data.zc[ci, tj - n_ctx_c:tj]; ctx_f[b] = data.zf[ci, tj - n_ctx_f:tj]
        tgt_c[b] = data.zc[ci, ti]; tgt_f[b] = data.zf[ci, ti]
        acts[b] = torch.from_numpy(data.acts[ci, tj:tj + data.k])
        nf6[b] = novel_fg6(data, ci, tj)
        if frames:
            gt[b, 0] = torch.from_numpy(data.clips_u8[ci, ti].astype(np.float32) / 127.5 - 1.0)
            lastf[b, 0] = torch.from_numpy(data.clips_u8[ci, tj - 1].astype(np.float32) / 127.5 - 1.0)
    out = dict(ctx_c=ctx_c.to(DEV), ctx_f=ctx_f.to(DEV), tgt_c=tgt_c.to(DEV), tgt_f=tgt_f.to(DEV),
               actions=acts.to(DEV), nf6=nf6)
    if frames:
        out["gt"] = gt.to(DEV); out["last"] = lastf.to(DEV)
    return out


def train_arm(name, data, n_ctx_c, n_ctx_f, steps, B, maxc, seed, val_every=1000, save_every=1000):
    torch.manual_seed(seed); np.random.seed(seed)
    net = DiTDenoiser(n_ctx=n_ctx_c, k=data.k, d=256).to(DEV)
    diff = Diffusion(net)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    ckpt = f"logs_rhcp/mem_gate_{name}_s{seed}.pt"
    start = load_ckpt(ckpt, net, opt, sched)
    ntok = n_ctx_c * 16 + n_ctx_f * 64 + 80
    p = sum(x.numel() for x in net.parameters()) / 1e6
    print(f"-- arm {name}: n_ctx_c={n_ctx_c} n_ctx_f={n_ctx_f} ({p:.2f}M, {ntok} tokens) "
          f"steps {start}..{steps} B={B} seed={seed} --", flush=True)
    if start > steps:
        print(f"   [arm {name}] already trained (ckpt @ >= {steps}), skipping to eval", flush=True)
        return diff
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train(); run = 0.0; t_last = time.time()
    for s in range(start, steps + 1):
        bt = gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=True)
        loss = diff.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        run += float(loss.detach())
        dt = time.time() - t_last; t_last = time.time()
        if dt > SPILL_SEC and s > start + 3:  # [SPILL GUARD] never crawl silently on WDDM spill
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
            raise RuntimeError(f"[arm {name}] step {s} took {dt:.1f}s > {SPILL_SEC}s -> WDDM spill; "
                               f"aborting arm (checkpoint saved). Shrink B or tokens and resume.")
        if s % val_every == 0:
            vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
            print(f"   [{name} {s}/{steps}] v-mse {run/val_every:.4f}  {dt:.2f}s/step  peakVRAM {vram:.2f}GB",
                  flush=True)
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
    return diff


@torch.no_grad()
def eval_arm(name, diff, data, ae, n_ctx_c, n_ctx_f, maxc, N=8, nfe=16, nb=24, B=32):
    """Best-of-N pixel fg-MSE per sample + sharpness; frames for the viz grid REUSE the scored
    samples (red-team fix: the original drew a fresh sample set for the grid)."""
    diff.net.eval()
    per_fg, per_nf6, sharps, panels = [], [], [], None
    for bi in range(nb):
        bt = gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=False, seed=1000 + bi, frames=(bi == 0))
        gt01 = decode01(ae, bt["tgt_f"])
        all_frames, allfg = [], np.empty((N, B), np.float32)
        for n in range(N):
            f01 = decode01(ae, diff.sample(bt["ctx_c"], bt["ctx_f"], bt["actions"], n_steps=nfe)[1])
            all_frames.append(f01)
            allfg[n] = fg_mse_px_frames(f01, gt01)
        best_idx = allfg.argmin(axis=0)
        bestf = torch.stack([all_frames[best_idx[b]][b] for b in range(B)], 0)
        per_fg.append(allfg.min(axis=0)); per_nf6.append(bt["nf6"].numpy())
        sharps.append(fg_sharp(bestf, gt01))
        if bi == 0:
            panels = (bt.get("last"), bestf, gt01)
            viz.save_samples(f"logs_rhcp/mem_gate_{name}.png",
                             [("last", panels[0]), (f"pred(best{N})", panels[1]), ("GT", panels[2])], n=8)
    return np.concatenate(per_fg), np.concatenate(per_nf6), float(np.mean(sharps))


def paired_bootstrap_ci(delta, n_boot=10000, seed=0):
    """95% CI of the mean of per-sample paired deltas."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(delta), (n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


BINS = [(-0.01, 0.001, "easy (nf6=0)"), (0.001, 0.1, "mid (0,0.1]"), (0.1, 1.01, "HARD re-entry (>0.1)")]


def report(arms, base_name, args):
    fg0, nf6, sharp0 = arms[base_name]
    print(f"\n{'='*94}\n  MEMORY-PAYOFF GATE  (best-of-{args.N} fg-MSE, stratified by novel-fg-at-6; "
          f"paired bootstrap 95% CI)\n{'='*94}")
    print(f"  baseline arm {base_name}: fg-MSE ALL {fg0.mean():.4f} | sharp {sharp0:.4f}")
    for name, (fg, nf6_a, sharp) in arms.items():
        if name == base_name:
            continue
        assert np.allclose(nf6, nf6_a), "eval sets diverged between arms — pairing broken"
        print(f"\n  arm {name} (sharp {sharp:.4f}) vs {base_name}:  [delta = base - arm; >0 = arm better]")
        print(f"  {'bin':<24}{'n':>6}{'base':>10}{'arm':>10}{'delta':>10}{'95% CI':>22}{'verdict':>10}")
        for lo, hi, bname in BINS + [(-0.01, 1.01, "ALL")]:
            m = (nf6 > lo) & (nf6 <= hi)
            if m.sum() < 5:
                continue
            d = fg0[m] - fg[m]
            ci_lo, ci_hi = paired_bootstrap_ci(d)
            verdict = "WIN" if ci_lo > 0 else ("LOSS" if ci_hi < 0 else "null")
            print(f"  {bname:<24}{int(m.sum()):>6}{fg0[m].mean():>10.4f}{fg[m].mean():>10.4f}"
                  f"{d.mean():>+10.4f}   [{ci_lo:+.4f},{ci_hi:+.4f}]{verdict:>10}")
    print("\n  GATE G0a: the coarseL arm must WIN the HARD re-entry bin (CI excluding 0). If null/LOSS")
    print("  on seed 0, run --seed 1 before declaring the memory-horizon headline dead (2-seed rule).")
    print("  grids: logs_rhcp/mem_gate_<arm>.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", type=int, default=48, help="long-context arm length L (short arm fixed at 6)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--with_longboth", action="store_true", help="also run the L-coarse + L-fine ceiling arm")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    M._BF16 = bool(args.bf16)
    maxc = args.long
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=maxc, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=maxc, k=4, n_val=250,
                          stats=data.stats)  # [STD-FIX] test standardized with TRAIN stats
    steps = 200 if args.smoke else args.steps
    nb, B_eval = (2, 16) if args.smoke else (24, 32)

    arm_specs = [("nctx6", 6, 6), (f"coarse{args.long}", args.long, 6)]
    if args.with_longboth:
        arm_specs.append((f"both{args.long}", args.long, args.long))

    arms = {}
    for name, nc, nf in arm_specs:
        diff = train_arm(name, data, nc, nf, steps, args.batch, maxc, args.seed)
        fg, nf6, sharp = eval_arm(name, diff, test, ae, nc, nf, maxc, N=args.N, nfe=args.nfe,
                                  nb=nb, B=B_eval)
        arms[name] = (fg, nf6, sharp)
        del diff
        torch.cuda.empty_cache()

    report(arms, "nctx6", args)


if __name__ == "__main__":
    main()
