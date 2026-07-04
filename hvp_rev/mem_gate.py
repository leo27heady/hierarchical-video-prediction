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

from hvp_rev import hier_convae, viz, wb
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


def dt_seen(data, ci, tj, thr=0.5, n_short=6):
    """[H-grid] dt-since-last-visible + answerability of the target (tj+k-1), single-step twin of
    rhvd.DtLabeler: over the target's NOVEL sprite columns (not covered in the last n_short ctx
    frames), dt = median frames since each column was last inside the viewport (any frame < tj);
    seen = fraction ever visible before. (0.0, 1.0) when no fg / no novel columns."""
    rel = rel_positions(data.acts[ci], data.speed)
    ti = tj + data.k - 1
    frame = data.clips_u8[ci, ti].astype(np.float32) / 255.0
    fg = np.where(frame.max(axis=0) > thr)[0]
    if len(fg) == 0:
        return 0.0, 1.0
    tw = (rel[ti] + fg) % data.W
    last = np.full(data.W, -(10 ** 9), np.int64)
    for m in range(tj):
        last[(rel[m] + np.arange(V)) % data.W] = m
    dt = ti - last[tw]
    novel = dt > (ti - (tj - n_short))          # not covered by the last n_short context frames
    if not novel.any():
        return 0.0, 1.0
    dtn = dt[novel]
    finite = dtn < 10 ** 8
    if not finite.any():
        return 1e9, 0.0
    return float(np.median(dtn[finite])), float(finite.mean())


def gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=True, seed=None, frames=False, stride_c=1,
               labels=False):
    """Sample tj in [maxc, T-k); coarse context = last n_ctx_c frames AT TEMPORAL STRIDE stride_c
    (slots anchored at tj-1, tj-1-stride, ... — span n_ctx_c*stride_c frames; stride_c=1 = legacy
    contiguous), fine context = last n_ctx_f contiguous frames; target = tj+k-1. Shared seed =>
    identical (clip,target) across arms => per-sample PAIRED deltas (maxc must be the SHARED max
    span across arms of one run). Always returns novel_fg-at-6 for stratification; labels=True
    additionally returns dt/seen (eval only — dt is O(T*V) per sample)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pool = data.train_idx if train else data.val_idx
    span_c = n_ctx_c * stride_c
    assert maxc >= span_c, f"maxc {maxc} < coarse span {span_c}"
    ctx_c = torch.empty(B, n_ctx_c, LAT, 4, 4); ctx_f = torch.empty(B, n_ctx_f, LAT, 8, 8)
    tgt_c = torch.empty(B, LAT, 4, 4); tgt_f = torch.empty(B, LAT, 8, 8)
    acts = torch.empty(B, data.k, dtype=torch.long)
    nf6 = torch.empty(B)
    dts = torch.empty(B) if labels else None
    seen = torch.empty(B) if labels else None
    gt = torch.empty(B, 1, 64, 64) if frames else None
    lastf = torch.empty(B, 1, 64, 64) if frames else None
    for b in range(B):
        ci = int(pool[rng.randint(len(pool))])
        tj = int(rng.randint(maxc, data.T - data.k))
        ti = tj + data.k - 1
        # strided slots: frames tj-span+stride-1, ..., tj-1-stride, tj-1 (anchor = block end)
        ctx_c[b] = data.zc[ci, tj - span_c + stride_c - 1: tj: stride_c]
        ctx_f[b] = data.zf[ci, tj - n_ctx_f:tj]
        tgt_c[b] = data.zc[ci, ti]; tgt_f[b] = data.zf[ci, ti]
        acts[b] = torch.from_numpy(data.acts[ci, tj:tj + data.k])
        nf6[b] = novel_fg6(data, ci, tj)
        if labels:
            d_, s_ = dt_seen(data, ci, tj)
            dts[b] = d_; seen[b] = s_
        if frames:
            gt[b, 0] = torch.from_numpy(data.clips_u8[ci, ti].astype(np.float32) / 127.5 - 1.0)
            lastf[b, 0] = torch.from_numpy(data.clips_u8[ci, tj - 1].astype(np.float32) / 127.5 - 1.0)
    out = dict(ctx_c=ctx_c.to(DEV), ctx_f=ctx_f.to(DEV), tgt_c=tgt_c.to(DEV), tgt_f=tgt_f.to(DEV),
               actions=acts.to(DEV), nf6=nf6)
    if labels:
        out["dt"] = dts; out["seen"] = seen
    if frames:
        out["gt"] = gt.to(DEV); out["last"] = lastf.to(DEV)
    return out


def train_arm(name, data, n_ctx_c, n_ctx_f, steps, B, maxc, seed, val_every=1000, save_every=1000,
              stride_c=1):
    torch.manual_seed(seed); np.random.seed(seed)
    net = DiTDenoiser(n_ctx=n_ctx_c, k=data.k, d=256).to(DEV)
    diff = Diffusion(net)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    ckpt = f"logs_rhcp/mem_gate_{name}_s{seed}_t{steps}.pt"  # steps in path: a smoke ckpt must never
    #                                                          resume-poison the full run's schedule
    start = load_ckpt(ckpt, net, opt, sched)
    ntok = n_ctx_c * 16 + n_ctx_f * 64 + 80
    p = sum(x.numel() for x in net.parameters()) / 1e6
    print(f"-- arm {name}: n_ctx_c={n_ctx_c} (stride {stride_c}, span {n_ctx_c*stride_c}) "
          f"n_ctx_f={n_ctx_f} ({p:.2f}M, {ntok} tokens) "
          f"steps {start}..{steps} B={B} seed={seed} --", flush=True)
    if start > steps:
        print(f"   [arm {name}] already trained (ckpt @ >= {steps}), skipping to eval", flush=True)
        return diff
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train(); run = 0.0; t_last = time.time()
    for s in range(start, steps + 1):
        bt = gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=True, stride_c=stride_c)
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
            wb.log({f"{name}/v_mse": run / val_every, f"{name}/s_per_step": dt,
                    f"{name}/vram_gb": vram}, step=s)
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
    return diff


@torch.no_grad()
def eval_arm(name, diff, data, ae, n_ctx_c, n_ctx_f, maxc, N=8, nfe=16, nb=24, B=32, seed=0,
             stride_c=1):
    """Best-of-N pixel fg-MSE per sample + sharpness; frames for the viz grid REUSE the scored
    samples (red-team fix: the original drew a fresh sample set for the grid). Sampling noise is
    reseeded per eval batch (common random numbers across arms -> tighter paired CIs)."""
    diff.net.eval()
    per_fg, per_nf6, per_dt, per_seen, sharps, panels = [], [], [], [], [], None
    for bi in range(nb):
        torch.manual_seed(31337 + bi)   # CRN: identical DDIM noise draws across arms
        bt = gate_batch(data, B, n_ctx_c, n_ctx_f, maxc, train=False, seed=1000 + bi, frames=(bi == 0),
                        stride_c=stride_c, labels=True)
        gt01 = decode01(ae, bt["tgt_f"])
        all_frames, allfg = [], np.empty((N, B), np.float32)
        for n in range(N):
            f01 = decode01(ae, diff.sample(bt["ctx_c"], bt["ctx_f"], bt["actions"], n_steps=nfe)[1])
            all_frames.append(f01)
            allfg[n] = fg_mse_px_frames(f01, gt01)
        best_idx = allfg.argmin(axis=0)
        bestf = torch.stack([all_frames[best_idx[b]][b] for b in range(B)], 0)
        per_fg.append(allfg.min(axis=0)); per_nf6.append(bt["nf6"].numpy())
        per_dt.append(bt["dt"].numpy()); per_seen.append(bt["seen"].numpy())
        sharps.append(fg_sharp(bestf, gt01))
        if bi == 0:
            panels = (bt.get("last"), bestf, gt01)
            viz.save_samples(f"logs_rhcp/mem_gate_{name}_s{seed}.png",
                             [("last", panels[0]), (f"pred(best{N})", panels[1]), ("GT", panels[2])], n=8)
    return (np.concatenate(per_fg), np.concatenate(per_nf6), float(np.mean(sharps)),
            np.concatenate(per_dt), np.concatenate(per_seen))


def paired_bootstrap_ci(delta, n_boot=10000, seed=0):
    """95% CI of the mean of per-sample paired deltas."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(delta), (n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


BINS = [(-0.01, 0.001, "easy (nf6=0)"), (0.001, 0.1, "mid (0,0.1]"), (0.1, 1.01, "HARD re-entry (>0.1)")]


def report(arms, base_name, args):
    fg0, nf6, sharp0, dt0, seen0, _ = arms[base_name]
    print(f"\n{'='*94}\n  MEMORY-PAYOFF GATE  (best-of-{args.N} fg-MSE, stratified by novel-fg-at-6; "
          f"paired bootstrap 95% CI; eval_nb={args.eval_nb})\n{'='*94}")
    print(f"  baseline arm {base_name}: fg-MSE ALL {fg0.mean():.4f} | sharp {sharp0:.4f}")
    dt_rows = [(6, 48, "dt 7-48"), (48, 96, "dt 49-96"), (96, 192, "dt 97-192"), (192, 1e12, "dt >192")]
    for name, (fg, nf6_a, sharp, dt, seen, reach) in arms.items():
        if name == base_name:
            continue
        assert np.allclose(nf6, nf6_a), "eval sets diverged between arms — pairing broken"
        print(f"\n  arm {name} (sharp {sharp:.4f}, memory reach {reach}f) vs {base_name}:"
              f"  [delta = base - arm; >0 = arm better]")
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
        # [H-grid] dt-stratified HARD rows on answerable targets: where in the memory horizon does
        # the strided arm win/lose? (dt ~= how long the novel content was off-screen)
        print(f"  --- HARD (>0.1) x dt-since-visible, answerable (seen>=0.5) ---")
        for lo, hi, bname in dt_rows:
            m = (nf6 > 0.1) & (dt0 > lo) & (dt0 <= hi) & (seen0 >= 0.5)
            if m.sum() < 5:
                continue
            d = fg0[m] - fg[m]
            ci_lo, ci_hi = paired_bootstrap_ci(d)
            verdict = "WIN" if ci_lo > 0 else ("LOSS" if ci_hi < 0 else "null")
            print(f"  {bname:<24}{int(m.sum()):>6}{fg0[m].mean():>10.4f}{fg[m].mean():>10.4f}"
                  f"{d.mean():>+10.4f}   [{ci_lo:+.4f},{ci_hi:+.4f}]{verdict:>10}")
        # [REACHABILITY 2026-07-04] the effective loop (~196-213f: p_hold slows mean speed to
        # 3.6px/f) EXCEEDS a 192f span, so >half the dt>192 mass is unanswerable BY CONSTRUCTION
        # for a 48x4 arm — these rows read the gate where the arm's memory physically reaches.
        # Pre-registered criterion for the Path-B decision: REACHABLE-HARD (48<dt<=reach) CI>0.
        print(f"  --- HARD x REACHABILITY (arm memory reach = {reach} frames) ---")
        for m, bname in ((
            (nf6 > 0.1) & (dt0 > 48) & (dt0 <= reach) & (seen0 >= 0.5), f"REACHABLE 48<dt<={reach}"), (
            (nf6 > 0.1) & (dt0 > reach) & (seen0 >= 0.5), f"BEYOND dt>{reach}")):
            if m.sum() < 5:
                continue
            d = fg0[m] - fg[m]
            ci_lo, ci_hi = paired_bootstrap_ci(d)
            verdict = "WIN" if ci_lo > 0 else ("LOSS" if ci_hi < 0 else "null")
            print(f"  {bname:<24}{int(m.sum()):>6}{fg0[m].mean():>10.4f}{fg[m].mean():>10.4f}"
                  f"{d.mean():>+10.4f}   [{ci_lo:+.4f},{ci_hi:+.4f}]{verdict:>10}")
    # [DISAMBIG] pairwise tables between non-base arms, split into SHARED-reach and
    # DIFFERENTIAL-reach dt bands: if the higher-reach arm wins ONLY in the differential band,
    # the effect is reach-caused; a uniform win incl. the shared band says circuit-level
    # (learned-to-use-memory) difference.
    others = [(n, v) for n, v in arms.items() if n != base_name]
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            (na, va), (nb_, vb) = others[i], others[j]
            if va[5] < vb[5]:
                (na, va), (nb_, vb) = (nb_, vb), (na, va)          # a = higher reach
            fga, fgb, ra, rb = va[0], vb[0], va[5], vb[5]
            print(f"\n  [pairwise] {na} (reach {ra}f) vs {nb_} (reach {rb}f)"
                  f"  [delta = {nb_} - {na}; >0 = {na} better]")
            rows = [((nf6 > 0.1) & (dt0 > 48) & (dt0 <= rb) & (seen0 >= 0.5), f"SHARED 48<dt<={rb}"),
                    ((nf6 > 0.1) & (dt0 > rb) & (dt0 <= ra) & (seen0 >= 0.5), f"DIFFERENTIAL {rb}<dt<={ra}"),
                    ((nf6 > 0.1), "HARD all")]
            for m, bname in rows:
                if m.sum() < 5:
                    continue
                d = fgb[m] - fga[m]
                ci_lo, ci_hi = paired_bootstrap_ci(d)
                verdict = "WIN" if ci_lo > 0 else ("LOSS" if ci_hi < 0 else "null")
                print(f"  {bname:<24}{int(m.sum()):>6}{fgb[m].mean():>10.4f}{fga[m].mean():>10.4f}"
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
    # ---- H-grid strided PRE-CHECK (HGRID_DESIGN.md v2 §3, the kill-switch) ----
    ap.add_argument("--precheck", action="store_true",
                    help="arms = nctx6 vs strided long-coarse (L slots @ --stride); new-dataset args required")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--train_path", default="data/loopmom_train_3000x100.npy")
    ap.add_argument("--test_path", default="data/loopmom_test_500x200.npy")
    ap.add_argument("--world", type=int, default=192)
    ap.add_argument("--wspeed", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--ae_path", default="logs_rhcp/hier_ae_mom.pt")
    ap.add_argument("--with_contig", action="store_true",
                    help="precheck: add a contiguous coarse-48 positive-control arm (loopmom's winner)")
    ap.add_argument("--eval_nb", type=int, default=24,
                    help="eval batches (widened-power re-eval: 96 = 4x samples, ~1.7x tighter CIs; "
                         "batches extend the same CRN seed series so arms stay paired)")
    ap.add_argument("--disambig", action="store_true",
                    help="pre-committed R1/R2/R3 arm set: nctx6 + stride{long}x4 + stride48x4 + "
                         "contig{long}, one shared sample pool, pairwise reach-band tables")
    args = ap.parse_args()
    M._BF16 = bool(args.bf16)
    maxc = args.long * (args.stride if args.precheck else 1)   # SHARED tj pool = max span
    if not args.no_wandb:
        wb.init(f"memgate-{'precheck' if args.precheck else 'gate'}-s{args.seed}"
                + ("-smoke" if args.smoke else ""),
                config=vars(args), tags=["hgrid-precheck" if args.precheck else "memgate"])
    ae = hier_convae.load(args.ae_path)
    data = ActionHierData(ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
                          n_ctx=maxc, k=4, W=args.world, speed=args.wspeed)
    test = ActionHierData(ae, args.test_path, args.test_path.replace(".npy", "_actions.npy"),
                          n_ctx=maxc, k=4, n_val=args.n_val, W=args.world, speed=args.wspeed,
                          stats=data.stats)  # [STD-FIX] test standardized with TRAIN stats
    steps = 200 if args.smoke else args.steps
    nb, B_eval = (2, 16) if args.smoke else (args.eval_nb, 32)

    if args.precheck:
        # kill-switch arms: short baseline vs strided slow-clock coarse (span = L*stride frames).
        # NOTE the near-ballistic loop regime makes the HARD (novel-at-6) bin ~= beyond-window
        # re-entry (measured dt-at-re-entry median ~196 >> 48), so the HARD-bin verdict here IS
        # the beyond-window retrieval signal.
        # Arm names carry a DATASET-derived prefix: ckpt paths are name-derived, and both the
        # legacy loopmom campaign AND any earlier pre-check on a different dataset would
        # otherwise be silently resumed as the baseline (caught live twice, 2026-07-04).
        tagpc = os.path.basename(args.train_path).split("_")[0] + "pc"   # e.g. hgridpc / hgrid2pc
        arm_specs = [(f"{tagpc}-nctx6", 6, 6, 1),
                     (f"{tagpc}-stride{args.long}x{args.stride}", args.long, 6, args.stride)]
        if args.with_contig:
            # positive control: contiguous coarse-48 (stride 1, 48-frame lookback) — the exact
            # design that retrieved on loopmom. Disambiguates stride-mechanism vs dataset failure.
            arm_specs.append((f"{tagpc}-contig48", 48, 6, 1))
        if args.disambig:
            # [PRE-COMMITTED RULE, HGRID_DESIGN addendum 2026-07-04] joint shared-pool set:
            # baseline + both stride spans + contig56 (SAME tokens/slots as stride56, reach 59
            # vs 227 — isolates REACH causally). maxc = long*stride covers every arm's span.
            arm_specs = [(f"{tagpc}-nctx6", 6, 6, 1),
                         (f"{tagpc}-stride{args.long}x{args.stride}", args.long, 6, args.stride),
                         (f"{tagpc}-stride48x4", 48, 6, 4),
                         (f"{tagpc}-contig{args.long}", args.long, 6, 1)]
    else:
        arm_specs = [("nctx6", 6, 6, 1), (f"coarse{args.long}", args.long, 6, 1)]
        if args.with_longboth:
            arm_specs.append((f"both{args.long}", args.long, args.long, 1))

    arms = {}
    for name, nc, nf, sc in arm_specs:
        diff = train_arm(name, data, nc, nf, steps, args.batch, maxc, args.seed, stride_c=sc)
        fg, nf6, sharp, dt, seen = eval_arm(name, diff, test, ae, nc, nf, maxc, N=args.N,
                                            nfe=args.nfe, nb=nb, B=B_eval, seed=args.seed,
                                            stride_c=sc)
        arms[name] = (fg, nf6, sharp, dt, seen, nc * sc + data.k - 1)   # + reach in frames
        wb.summary({f"{name}/fg_mse_all": float(fg.mean()), f"{name}/sharp": sharp,
                    f"{name}/fg_mse_hard": float(fg[nf6 > 0.1].mean()) if (nf6 > 0.1).any() else 0.0})
        wb.image(f"{name}/eval_grid", f"logs_rhcp/mem_gate_{name}_s{args.seed}.png")
        del diff
        torch.cuda.empty_cache()

    report(arms, arm_specs[0][0], args)
    wb.finish()


if __name__ == "__main__":
    main()
