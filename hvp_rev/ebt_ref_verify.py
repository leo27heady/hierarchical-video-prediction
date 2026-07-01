"""DECISIVE VERIFICATION of the reference-EBT mean-collapse negative.

A prior agent built hvp_rev/ebt_reference_control.py (imports the GENUINE EBT repo model
model.ar_ebt_time_embed.EBTTimeConcat + model.replay_buffer.CausalReplayBuffer + a line-for-line
transcription of EBT_VID.forward MCMC descent) and found the reference EBT MEAN-COLLAPSES on our
loopmom latents: E(mean) < E(target) always, decoded predictions blank (fg-sharp ~0.01).

BEFORE accepting this direction-killing negative we rule out a subtle harness/config/wiring bug by
running FOUR decisive tests the prior agent never ran. We reuse the EXACT RefEBT/RefTrainer wrapper
(so any bug in it is exercised identically), and add:

  T1 CONTEXT-CONDITIONING: energy matrix E(ctx_i, tgt_j). Diagonal (each target under ITS OWN context)
     should be systematically LOWER than off-diagonal if the energy conditions on context. Also
     directly perturb the context tensor and measure dE/d(context).
  T2 LANDSCAPE REACHABILITY: warm-start the candidate AT the GT target, run LONG descent (up to 200
     steps) across a WIDE alpha sweep. Does descent-from-target STAY near target (target is a local
     min) or FLEE to the global mean (only basin is the mean)? Also sweep cold-init inference steps
     {2..128}: does MORE thinking ever reach the target (the paper's System-2 claim)?
  T3 POSITIVE CONTROL (the crux): a MINIMAL context-conditional COPY task -- target == last context
     frame's latent (identity). A functioning conditional energy MUST commit: E(target) lowest,
     decoded output == the copied frame, sharp. If it commits, the harness is faithful and our
     real-data collapse is genuine. If it ALSO collapses on the trivial copy, the harness has a
     conditioning/scale bug and the whole negative is suspect.
  T4 If T1-T3 point to a fixable issue, APPLY the fix and re-probe the real-data task.

    venv/Scripts/python.exe -u -m hvp_rev.ebt_ref_verify --train_steps 1500 --bf16
"""
from __future__ import annotations
import argparse, os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the EXACT reference wrapper + trainer + probe from the prior harness
from hvp_rev.ebt_reference_control import (
    RefEBT, RefTrainer, probe, calibrate_alpha, _rms,
    DEV, LAT, CDIM, FDIM, TOKDIM,
)
from hvp_rev import hier_convae, viz
from hvp_rev.step1_mech import ActionHierData, decode01, fg_sharp


def log(*a):
    print(*a, flush=True)


# ============================================================================================
# Shared: load data + AE + build a trained-or-fresh RefEBT on the REAL task.
# ============================================================================================
def build_real(args):
    import hvp_rev.step1_mech as M
    M._BF16 = bool(args.bf16)
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250)
    b0 = data.batch(1024, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean())
    sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    log(f"[standardize] coarse std {sc:.4f}  fine std {sf:.4f}")
    m = RefEBT(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=6, k=4,
               sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
               randomize_step_scale=2.0, langevin=args.langevin, alpha=1.0).to(DEV)
    k_eval = m.randomize_num_steps + 1
    bcal = data.batch(64, train=True)
    a_auto, gmag = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                   bcal["actions"], k_eval)
    m.alpha.data.fill_(a_auto if args.alpha == 0 else args.alpha)
    log(f"[model] REF-EBT params {sum(p.numel() for p in m.parameters())/1e6:.2f}M | "
        f"auto-alpha {a_auto:.3f} -> alpha {float(m.alpha):.3f} | eval-steps {k_eval}")
    return ae, data, test, m, k_eval


def quick_train(m, data, steps, batch, warm=200, tag="real"):
    """Same optimizer/recipe as ebt_reference_control.main train loop."""
    trainer = RefTrainer(m, use_replay=True, batch=batch)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    m.train()
    run = 0.0
    for s in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = 4e-4 * min(1.0, s / warm)
        bt = data.batch(batch, train=True)
        loss = trainer.train_step(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            run += float(loss.detach())
        if s % 300 == 0:
            log(f"  [{tag} train {s}/{steps}] smooth_l1 {run/300:.4f}")
            run = 0.0
    return trainer


# ============================================================================================
# T1  CONTEXT-CONDITIONING TEST
# ============================================================================================
@torch.no_grad()
def test_context_conditioning(m, test, tag):
    """Energy matrix E(context_i, target_j). If energy conditions on context, the DIAGONAL
    (each target under its own context) should be systematically LOWER than off-diagonal."""
    log(f"\n=== T1 CONTEXT-CONDITIONING [{tag}] ===")
    n = 8
    b = test.batch(n, train=False, return_frames=True, seed=123)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)     # (n, n_ctx+1, 1280)
    real_in = full[:, :-1]                     # (n, n_ctx, 1280) context frames per example
    tgt_tok = full[:, -1]                      # (n, 1280) target token per example
    gt_seq = full[:, 1:]                       # (n, n_ctx, 1280) gt for predicted slots

    # E(ctx_i, tgt_j): use context of example i, candidate-last = target of example j, action of i.
    E = np.zeros((n, n), np.float32)
    with torch.enable_grad():
        for i in range(n):
            ri = real_in[i:i+1]                                    # (1,n_ctx,1280)
            gi = gt_seq[i:i+1].clone()                             # (1,n_ctx,1280) gt slots of i
            ai = a[i:i+1]
            for j in range(n):
                cand = gi.clone()
                cand[:, -1] = tgt_tok[j:j+1]                       # swap in target j
                e = m.energy_of(ri, cand, ai, mcmc_step=0)         # (1,S,1)
                E[i, j] = float(e[0, -1].detach())                # last-slot energy
    diag = np.diag(E)
    off = E[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    # per-row: is the diagonal the argmin over j? (each target lowest under its OWN context)
    argmin_is_diag = (E.argmin(axis=1) == np.arange(n))
    gap = float(off.mean() - diag.mean())      # >0 means diagonal lower (good)
    row_gap = float((off.mean(axis=1) - diag).mean())
    log(f"  E(ctx_i,tgt_j) matrix (rows=context, cols=target):")
    for i in range(n):
        row = " ".join(f"{E[i,j]:7.2f}" for j in range(n))
        log(f"    ctx{i}: {row}   [diag={E[i,i]:7.2f}]")
    log(f"  diag mean {diag.mean():+.3f} | off-diag mean {off.mean():+.3f} | "
        f"gap(off-diag) {gap:+.3f}  row-wise {row_gap:+.3f}")
    log(f"  argmin-over-targets == own-context on {int(argmin_is_diag.sum())}/{n} rows "
        f"(random baseline {n*(1/n):.1f})")

    # direct dE/d(context): perturb the context tokens, measure gradient magnitude.
    ri = real_in[:1].clone().requires_grad_(True)
    gi = gt_seq[:1].clone()
    cand = gi.clone()
    cand[:, -1] = tgt_tok[:1]
    with torch.enable_grad():
        e = m.energy_of(ri, cand, a[:1], mcmc_step=0)
        g_ctx = torch.autograd.grad(e[:, -1].sum(), ri)[0]
    dE_dctx = _rms(g_ctx)
    # for scale reference: dE/d(candidate target)
    cand2 = gi.clone().requires_grad_(True)
    with torch.enable_grad():
        cand2b = cand2.clone()
        e2 = m.energy_of(ri.detach(), cand2, a[:1], mcmc_step=0)
        g_cand = torch.autograd.grad(e2[:, -1].sum(), cand2)[0]
    dE_dcand = _rms(g_cand[:, -1])
    log(f"  |dE/d(context)| rms {dE_dctx:.5f}  vs  |dE/d(candidate target)| rms {dE_dcand:.5f}  "
        f"ratio {dE_dctx/max(dE_dcand,1e-9):.3f}")
    conditions = (gap > 0.05) and (argmin_is_diag.sum() >= n * 0.5)
    log(f"  --> ENERGY {'CONDITIONS' if conditions else 'DOES NOT CONDITION'} ON CONTEXT "
        f"(gap>{0.05} and >=50% rows diag-argmin)")
    return dict(gap=gap, row_gap=row_gap, diag_argmin=int(argmin_is_diag.sum()), n=n,
                dE_dctx=dE_dctx, dE_dcand=dE_dcand, conditions=conditions)


# ============================================================================================
# T2  LANDSCAPE REACHABILITY  (warm-start-at-target long descent + cold step sweep)
# ============================================================================================
def test_reachability(m, test, tag, alphas=(0.3, 1, 3, 10, 30, 100, 300, 1000, 3000),
                      long_steps=200):
    log(f"\n=== T2 LANDSCAPE REACHABILITY [{tag}] ===")
    b = test.batch(64, train=False, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in = full[:, :-1]
    gt_seq = full[:, 1:]                                  # (B,S,1280)
    tgt_tok = full[:, -1]                                 # (B,1280)
    mean_tok = tgt_tok.mean(0, keepdim=True).expand_as(tgt_tok).contiguous()
    d_mean_tgt = _rms(mean_tok - tgt_tok)                 # distance mean<->target (fixed ref)

    real_down = m.encoder_down_projection(real_in)
    saved_lang = float(m.langevin_std); m.langevin_std.data.fill_(0.0)   # deterministic descent

    def descend_from(init_last, alpha, nsteps):
        """Descend ONLY the last (target) slot from init_last; other slots pinned to gt."""
        orig = gt_seq.clone()
        orig[:, -1] = init_last
        orig = orig.detach()
        for _ in range(nsteps):
            orig = orig.detach().requires_grad_(True)
            pred_down = m.encoder_down_projection(orig)
            pred_down = m._add_action(pred_down, a)
            all_emb = torch.cat([real_down.detach(), pred_down], dim=1)
            with torch.enable_grad():
                e = m.transformer(all_emb, start_pos=0, mcmc_step=0)
                g = torch.autograd.grad(e.sum(), orig)[0]
            orig = orig - alpha * g
        return orig[:, -1].detach()

    log("  WARM-START AT TARGET, long descent (does it STAY at target or FLEE to mean?):")
    log(f"    (ref: d(mean,target) = {d_mean_tgt:.3f}; start d_tgt=0)")
    warm_res = {}
    for al in alphas:
        with torch.enable_grad():
            final = descend_from(tgt_tok, float(al), long_steps)
        d_tgt = _rms(final - tgt_tok)
        d_mean = _rms(final - mean_tok)
        warm_res[al] = (d_tgt, d_mean)
        verdict = "STAYS@tgt" if d_tgt < 0.5 * d_mean_tgt else ("->MEAN" if d_mean < d_tgt else "wandered")
        log(f"    alpha {al:>6}: after {long_steps} steps  d_tgt {d_tgt:6.3f}  d_mean {d_mean:6.3f}  [{verdict}]")

    log("  COLD-INIT (randn) inference-step sweep (does MORE thinking reach target? System-2 claim):")
    best_alpha = float(m.alpha)
    for al_lbl, al in (("model-alpha", best_alpha), ("alpha=10", 10.0)):
        log(f"    [{al_lbl}={al:.3f}]")
        for ns in (2, 4, 8, 16, 32, 64, 128):
            torch.manual_seed(0)
            init = torch.randn_like(tgt_tok)
            with torch.enable_grad():
                final = descend_from(init, al, ns)
            d_tgt = _rms(final - tgt_tok)
            d_mean = _rms(final - mean_tok)
            log(f"      steps {ns:>3}: d_tgt {d_tgt:6.3f}  d_mean {d_mean:6.3f}  "
                f"{'[closer to TARGET]' if d_tgt < d_mean else '[closer to MEAN]'}")

    m.langevin_std.data.fill_(saved_lang)
    # verdict: is the target a local min under ANY alpha?
    target_is_localmin = any(dt < 0.5 * d_mean_tgt for dt, _ in warm_res.values())
    log(f"  --> target is a LOCAL MIN under some alpha: {target_is_localmin}  "
        f"(else only-basin=mean)")
    return dict(warm_res=warm_res, d_mean_tgt=d_mean_tgt, target_is_localmin=target_is_localmin)


# ============================================================================================
# T3  POSITIVE CONTROL -- minimal COPY task (target == last context frame's latent)
# ============================================================================================
class CopyData:
    """Wraps ActionHierData but REPLACES the target with the LAST CONTEXT FRAME's latent
    (identity/copy). A functioning conditional energy MUST commit: predict == last ctx frame."""
    def __init__(self, base: ActionHierData, shift=False):
        self.base = base
        self.shift = shift          # if True, target = last ctx frame shifted (still deterministic)

    def batch(self, B, train=True, return_frames=False, seed=None):
        b = self.base.batch(B, train=train, return_frames=return_frames, seed=seed)
        # last context frame latent == ctx[:, -1]
        last_c = b["ctx_c"][:, -1].clone()
        last_f = b["ctx_f"][:, -1].clone()
        if self.shift:
            # deterministic shift: roll the fine latent spatially by 1 col (a trivial deterministic map)
            last_f = torch.roll(last_f, shifts=1, dims=-1)
            last_c = torch.roll(last_c, shifts=1, dims=-1)
        b["tgt_c"] = last_c
        b["tgt_f"] = last_f
        return b


@torch.no_grad()
def probe_copy(m, cdata, ae, tag, k_eval):
    """Probe on the copy task: E(target=last-ctx) vs E(mean) vs E(randn); cold-init prediction
    distance to target; decoded sharpness AND whether decoded pred matches the LAST CONTEXT frame."""
    b = cdata.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in = full[:, :-1]
    gt_seq = full[:, 1:]
    tgt_tok = full[:, -1]
    mean_tok = tgt_tok.mean(0, keepdim=True).expand_as(tgt_tok).contiguous()
    randn_tok = torch.randn_like(tgt_tok)

    def E_last(cand_last):
        cand = gt_seq.clone(); cand[:, -1] = cand_last
        with torch.enable_grad():
            e = m.energy_of(real_in, cand, a, mcmc_step=0)
        return float(e[:, -1].mean())

    Et, Em, Er = E_last(tgt_tok), E_last(mean_tok), E_last(randn_tok)
    lo = min([("target", Et), ("mean", Em), ("randn", Er)], key=lambda x: x[1])[0]

    saved = float(m.langevin_std); m.langevin_std.data.fill_(0.0)
    with torch.enable_grad():
        pred_list, _ = m(real_in, a, learning=False, no_randomness=True)
    m.langevin_std.data.fill_(saved)
    pred_last = pred_list[-1][:, -1].detach()
    d_tgt = _rms(pred_last - tgt_tok)
    d_mean = _rms(mean_tok - tgt_tok)

    _, pred_zf = m._unflat(pred_last)
    pred01 = decode01(ae, pred_zf)
    gt01 = decode01(ae, tf)
    last01 = decode01(ae, cf[:, -1])       # decoded LAST CONTEXT frame (== copy target)
    sh = fg_sharp(pred01, gt01)
    # copy-match: fg-MSE between decoded prediction and the LAST CONTEXT frame (the thing to copy)
    from hvp_rev.step1_mech import fg_mse_px_frames
    copy_fgmse = float(fg_mse_px_frames(pred01, last01).mean())
    committed = (lo == "target") and (sh >= 0.08) and (d_tgt < d_mean)
    log(f"  [{tag}] E: tgt {Et:+.3f} | mean {Em:+.3f} | randn {Er:+.3f} -> LOWEST={lo} "
        f"{'[GOOD]' if lo=='target' else '[BAD]'} | cold d_tgt {d_tgt:.3f} (mean {d_mean:.3f}) | "
        f"fg-sharp {sh:.4f} (GT~0.25) | copy-fgMSE {copy_fgmse:.4f} "
        f"-> {'COMMITTED' if committed else 'COLLAPSED'}")
    os.makedirs("logs_rhcp", exist_ok=True)
    viz.save_samples(f"logs_rhcp/ebt_verify_{tag}.png",
                     [("LAST-CTX(copy tgt)", last01 * 2 - 1), ("GT", gt01 * 2 - 1),
                      ("REF-EBT", pred01 * 2 - 1)], n=8)
    return dict(lo=lo, Et=Et, Em=Em, Er=Er, d_tgt=d_tgt, d_mean=d_mean, sharp=sh,
                copy_fgmse=copy_fgmse, committed=committed)


def test_positive_control(args, ae, data, test, shift=False):
    label = "shift" if shift else "copy"
    log(f"\n=== T3 POSITIVE CONTROL: {label.upper()} TASK (target == last ctx frame"
        f"{' shifted' if shift else ''}) ===")
    cdata = CopyData(data, shift=shift)
    ctest = CopyData(test, shift=shift)
    b0 = cdata.batch(512, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean())
    sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    m = RefEBT(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=6, k=4,
               sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
               randomize_step_scale=2.0, langevin=args.langevin, alpha=1.0).to(DEV)
    k_eval = m.randomize_num_steps + 1
    bcal = cdata.batch(64, train=True)
    a_auto, gmag = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                   bcal["actions"], k_eval)
    m.alpha.data.fill_(a_auto if args.alpha == 0 else args.alpha)
    log(f"  [copy-model] alpha {float(m.alpha):.3f} | init |gradE| {gmag:.4f}")
    m.eval(); r0 = probe_copy(m, ctest, ae, f"{label}_step0", k_eval); m.train()
    quick_train(m, cdata, args.copy_steps, args.batch, tag=label)
    m.eval(); rF = probe_copy(m, ctest, ae, f"{label}_final", k_eval)
    # also run the context-conditioning test on the copy-trained model
    r_ctx = test_context_conditioning(m, ctest, f"{label}_trained")
    m.train()
    return dict(step0=r0, final=rF, ctx=r_ctx, model=m, ctest=ctest, k_eval=k_eval)


# ============================================================================================
# T4  Apply-fix probes on the REAL task (only if T1-T3 point to a fix).
#     We test the most likely fixes discovered structurally: (a) inference-only many-step descent
#     at a swept alpha (already covered by T2), (b) energy-argmin selection over N cold samples.
# ============================================================================================
@torch.no_grad()
def test_energy_argmin_selection(m, test, ae, tag, N=16, steps=8):
    """Even if the mean is the single lowest-energy basin, does energy-argmin OVER N noisy cold
    samples ever pick a sharper, closer-to-target candidate than the raw descent mean? (a cheap
    'fix' that some EBT setups rely on.)"""
    log(f"\n=== T4b ENERGY-ARGMIN over N={N} cold samples [{tag}] ===")
    b = test.batch(48, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in = full[:, :-1]
    tgt_tok = full[:, -1]
    real_down = m.encoder_down_projection(real_in)
    gt_seq = full[:, 1:]
    B = tgt_tok.shape[0]

    def one_sample():
        orig = torch.randn_like(gt_seq)
        for _ in range(steps):
            orig = orig.detach().requires_grad_(True)
            pd = m._add_action(m.encoder_down_projection(orig), a)
            all_emb = torch.cat([real_down, pd], dim=1)
            with torch.enable_grad():
                e = m.transformer(all_emb, start_pos=0, mcmc_step=0)
                g = torch.autograd.grad(e.sum(), orig)[0]
            orig = orig - float(m.alpha) * g
        last = orig[:, -1].detach()
        cand = gt_seq.clone(); cand[:, -1] = last
        with torch.enable_grad():
            e = m.energy_of(real_in, cand, a, mcmc_step=0)
        return last, e[:, -1, 0].detach()      # (B,1280),(B,)

    saved = float(m.langevin_std); m.langevin_std.data.fill_(float(m.langevin_std) or 1.0)
    cands, energies = [], []
    with torch.enable_grad():
        for _ in range(N):
            l, e = one_sample()
            cands.append(l); energies.append(e)
    m.langevin_std.data.fill_(saved)
    C = torch.stack(cands, 0)          # (N,B,1280)
    E = torch.stack(energies, 0)       # (N,B)
    amin = E.argmin(0)                 # (B,)
    picked = torch.stack([C[amin[b], b] for b in range(B)], 0)
    _, zf = m._unflat(picked)
    pred01 = decode01(ae, zf); gt01 = decode01(ae, tf)
    sh = fg_sharp(pred01, gt01)
    d_tgt = _rms(picked - tgt_tok)
    log(f"  energy-argmin pick: d_tgt {d_tgt:.3f} | fg-sharp {sh:.4f} (GT~0.25) "
        f"{'[HAS STRUCTURE]' if sh>=0.08 else '[still COLLAPSED]'}")
    viz.save_samples(f"logs_rhcp/ebt_verify_{tag}_argmin.png",
                     [("GT", gt01 * 2 - 1), ("REF-EBT argmin", pred01 * 2 - 1)], n=8)
    return dict(d_tgt=d_tgt, sharp=sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_steps", type=int, default=1500, help="real-task training before probes")
    ap.add_argument("--copy_steps", type=int, default=1500, help="copy-task training steps")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 => auto-calibrate")
    ap.add_argument("--langevin", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--skip_real_train", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    ae, data, test, m, k_eval = build_real(args)

    # ---- T3 first: the crux positive control (COPY). If harness is sound it MUST commit here. ----
    pc = test_positive_control(args, ae, data, test, shift=False)
    # optional: a slightly harder deterministic map (shift) to confirm it's not degenerate-copy only
    pc_shift = test_positive_control(args, ae, data, test, shift=True)

    # ---- Real-task model: train briefly, then T1 + T2 + T4 ----
    if not args.skip_real_train:
        log(f"\n=== TRAIN REAL-TASK REF-EBT ({args.train_steps} steps) ===")
        m.eval(); probe(m, None, test, ae, "verify_real_step0", k_eval); m.train()
        quick_train(m, data, args.train_steps, args.batch, tag="real")
    m.eval()
    r_real0 = probe(m, None, test, ae, "verify_real_final", k_eval)
    t1 = test_context_conditioning(m, test, "real_trained")
    t2 = test_reachability(m, test, "real_trained")
    t4b = test_energy_argmin_selection(m, test, ae, "real_trained", N=16, steps=8)

    # ============================================================================
    # DECISIVE VERDICT
    # ============================================================================
    log("\n" + "=" * 92)
    log("=== DECISIVE VERDICT ===")
    log("=" * 92)

    copy_committed = pc["final"]["committed"]
    shift_committed = pc_shift["final"]["committed"]
    log(f"\n=== DOES THE ENERGY CONDITION ON CONTEXT? ===")
    log(f"  REAL task:  off-vs-diag energy gap {t1['gap']:+.3f} (row-wise {t1['row_gap']:+.3f}); "
        f"diag-argmin {t1['diag_argmin']}/{t1['n']} rows; |dE/dctx|={t1['dE_dctx']:.4f} "
        f"vs |dE/dcand|={t1['dE_dcand']:.4f}")
    log(f"  COPY task:  off-vs-diag energy gap {pc['ctx']['gap']:+.3f}; "
        f"diag-argmin {pc['ctx']['diag_argmin']}/{pc['ctx']['n']} rows")
    log(f"  --> {'YES energy conditions on context' if (t1['conditions'] or pc['ctx']['conditions']) else 'NO / WEAK context conditioning'}")

    log(f"\n=== IS THE TARGET A LOCAL MIN, OR ONLY THE MEAN IS? ===")
    log(f"  warm-start-at-target long descent: target_is_local_min = {t2['target_is_localmin']}")
    for al, (dt, dm) in t2["warm_res"].items():
        log(f"    alpha {al:>6}: d_tgt {dt:.3f} d_mean {dm:.3f}")

    log(f"\n=== DID MORE MCMC STEPS / ALPHA SWEEP EVER COMMIT? ===")
    log(f"  (see T2 cold-step sweep + T4b energy-argmin: argmin fg-sharp {t4b['sharp']:.4f}, "
        f"d_tgt {t4b['d_tgt']:.3f}) -- structure if fg-sharp>=0.08")

    log(f"\n=== POSITIVE CONTROL (copy task): COMMIT or COLLAPSE? ===")
    log(f"  COPY  step0: {'COMMIT' if pc['step0']['committed'] else 'collapse'} "
        f"(sharp {pc['step0']['sharp']:.4f}) -> final: "
        f"{'COMMIT' if copy_committed else 'COLLAPSE'} "
        f"(E-lowest={pc['final']['lo']}, sharp {pc['final']['sharp']:.4f}, "
        f"copy-fgMSE {pc['final']['copy_fgmse']:.4f}, d_tgt {pc['final']['d_tgt']:.3f} "
        f"vs mean {pc['final']['d_mean']:.3f}) [PNG logs_rhcp/ebt_verify_copy_final.png]")
    log(f"  SHIFT final: {'COMMIT' if shift_committed else 'COLLAPSE'} "
        f"(E-lowest={pc_shift['final']['lo']}, sharp {pc_shift['final']['sharp']:.4f}, "
        f"copy-fgMSE {pc_shift['final']['copy_fgmse']:.4f})")

    log(f"\n=== REAL-DATA PROBE (for reference) ===")
    lo, sh = r_real0[0], r_real0[1]
    log(f"  real final: E-lowest={lo} sharp {sh:.4f} (collapse reproduced = {lo!='target' or sh<0.08})")

    log(f"\n=== FINAL: bug / fixable / genuine? ===")
    if copy_committed or shift_committed:
        log("  The reference harness COMMITS on the positive-control conditional task "
            "(copy/shift). => the wrapper CAN produce sharp, context-conditional output when it "
            "should. Therefore the real-data mean-collapse is GENUINE at our scale (task is "
            "near-deterministic given action, yet the energy still prefers the global mean): "
            "the negative STANDS. Confidence: HIGH.")
    else:
        log("  The reference harness ALSO COLLAPSES on the trivial copy positive-control. This is "
            "NOT a fundamental EBT limit -- a working conditional energy MUST solve copy. => there "
            "is a CONDITIONING/WIRING/SCALE BUG in the harness; the whole negative is SUSPECT. "
            "Confidence: HIGH that the prior negative is unsafe until the harness passes copy.")
    log("=" * 92)


if __name__ == "__main__":
    main()
