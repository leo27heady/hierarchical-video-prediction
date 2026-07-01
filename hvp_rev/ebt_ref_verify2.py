"""FOLLOW-UP: separate 'energy is wrong' from 'descent can't optimize'.

The main verify run proved: on COPY/SHIFT positive controls the ENERGY LANDSCAPE commits (E(target)
lowest, conditions on context, argmin-diag 8/8) -- but cold-init MCMC descent still lands at the mean
(blank decode). This isolates the failure to REACHABILITY of the target basin by gradient descent,
NOT to a conditioning/wiring bug.

Here we confirm that split directly on a freshly copy-trained model:
  (A) COLD randn-init descent          -> expect blank/mean (reproduce)
  (B) WARM init = last-context frame    -> the copy target IS the last ctx frame; start there
  (C) WARM init = target + small noise  -> tiny basin-of-attraction probe
For each, decode + fg-sharp + fg-MSE-to-copy-target + energy. If (B)/(C) commit while (A) fails =>
the energy is RIGHT, the descent/init is the lever (fixable), NOT a harness bug.

    venv/Scripts/python.exe -u -m hvp_rev.ebt_ref_verify2 --copy_steps 2000 --batch 32 --bf16
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from hvp_rev.ebt_reference_control import RefEBT, RefTrainer, calibrate_alpha, _rms, DEV
from hvp_rev import hier_convae, viz
from hvp_rev.step1_mech import ActionHierData, decode01, fg_sharp, fg_mse_px_frames
from hvp_rev.ebt_ref_verify import build_real, quick_train, CopyData


def log(*a): print(*a, flush=True)


@torch.no_grad()
def descend_and_report(m, real_in, gt_seq, a, tgt_tok, init_last, ae, tf, cf, nsteps, tag):
    real_down = m.encoder_down_projection(real_in)
    orig = gt_seq.clone(); orig[:, -1] = init_last; orig = orig.detach()
    with torch.enable_grad():
        for _ in range(nsteps):
            orig = orig.detach().requires_grad_(True)
            pd = m._add_action(m.encoder_down_projection(orig), a)
            all_emb = torch.cat([real_down.detach(), pd], dim=1)
            e = m.transformer(all_emb, start_pos=0, mcmc_step=0)
            g = torch.autograd.grad(e.sum(), orig)[0]
            orig = orig - float(m.alpha) * g
    last = orig[:, -1].detach()
    _, zf = m._unflat(last)
    pred01 = decode01(ae, zf); gt01 = decode01(ae, tf)
    last01 = decode01(ae, cf[:, -1])
    sh = fg_sharp(pred01, gt01)
    copy_fgmse = float(fg_mse_px_frames(pred01, last01).mean())
    d_tgt = _rms(last - tgt_tok)
    log(f"  [{tag}] nsteps {nsteps}: d_tgt {d_tgt:.3f} | fg-sharp {sh:.4f} (GT~0.25) | "
        f"copy-fgMSE {copy_fgmse:.4f} {'[COMMIT/sharp]' if sh>=0.08 else '[collapsed]'}")
    return pred01, last01, gt01, sh, copy_fgmse, d_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy_steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--langevin", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    ae, data, test, _m, _k = build_real(args)
    cdata = CopyData(data, shift=False)
    ctest = CopyData(test, shift=False)
    b0 = cdata.batch(512, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean()); sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    m = RefEBT(d=args.d, n_layers=args.depth, n_heads=args.heads, n_frames_ctx=6, k=4,
               sc=sc, sf=sf, mcmc_num_steps=1, randomize_num_steps=2, randomize_num_steps_min=2,
               randomize_step_scale=2.0, langevin=args.langevin, alpha=1.0).to(DEV)
    bcal = cdata.batch(64, train=True)
    a_auto, _ = calibrate_alpha(m, bcal["ctx_c"], bcal["ctx_f"], bcal["tgt_c"], bcal["tgt_f"],
                                bcal["actions"], 3)
    m.alpha.data.fill_(a_auto if args.alpha == 0 else args.alpha)
    log(f"[copy-model] alpha {float(m.alpha):.3f}")
    quick_train(m, cdata, args.copy_steps, args.batch, tag="copy2")

    m.eval()
    b = ctest.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    full = m._seq_from_ctx(cc, cf, tc, tf)
    real_in, gt_seq, tgt_tok = full[:, :-1], full[:, 1:], full[:, -1]
    # copy target == last context frame's standardized token
    last_ctx_tok = m._flat_std(cc[:, -1], cf[:, -1])
    m.langevin_std.data.fill_(0.0)   # deterministic descent for the probe

    log("\n=== SPLIT: energy-correct vs descent-reachable (copy-trained model) ===")
    # (A) cold randn init
    torch.manual_seed(1)
    descend_and_report(m, real_in, gt_seq, a, tgt_tok, torch.randn_like(tgt_tok), ae, tf, cf, 8,
                       "A cold-randn n8")
    descend_and_report(m, real_in, gt_seq, a, tgt_tok, torch.randn_like(tgt_tok), ae, tf, cf, 64,
                       "A cold-randn n64")
    # (B) warm init = last context frame (the copy source) -- most_recent_embedding init
    pB, lB, gB, *_ = descend_and_report(m, real_in, gt_seq, a, tgt_tok, last_ctx_tok, ae, tf, cf, 8,
                                        "B warm=last-ctx n8")
    descend_and_report(m, real_in, gt_seq, a, tgt_tok, last_ctx_tok, ae, tf, cf, 0,
                       "B warm=last-ctx n0(no-descent)")
    # (C) warm init = target + small noise (basin probe)
    descend_and_report(m, real_in, gt_seq, a, tgt_tok, tgt_tok + 0.3 * torch.randn_like(tgt_tok),
                       ae, tf, cf, 8, "C tgt+0.3noise n8")

    viz.save_samples("logs_rhcp/ebt_verify2_warmstart.png",
                     [("last-ctx(copy tgt)", lB * 2 - 1), ("GT", gB * 2 - 1),
                      ("warm-init pred", pB * 2 - 1)], n=8)
    log("\nDONE -> logs_rhcp/ebt_verify2_warmstart.png")


if __name__ == "__main__":
    main()
