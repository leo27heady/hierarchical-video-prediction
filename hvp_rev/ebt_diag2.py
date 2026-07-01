"""EBT MEAN-COLLAPSE diagnostic. The mechanism harness shows EBT predicting a near-uniform (mean/blank)
latent while diffusion commits to sharp digits. This instruments WHY: energy-descent dynamics, the
landscape shape, train/eval step-count mismatch, warm-vs-cold init, alpha scale, and grad magnitudes.

  venv/Scripts/python.exe -u -m hvp_rev.ebt_diag2 --steps 800
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae, viz
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import ActionHierData, EBT, decode01, fg_sharp
from hvp_rev.faithful_nets import EnergyTransformer, LAT

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rms(t):
    return float(t.pow(2).mean().sqrt())


@torch.no_grad()
def landscape(ebt, cc, cf, a, tc, tf):
    """Energy at the TARGET vs degenerate candidates. If E(target) is not the lowest, the landscape
    itself is broken (energy doesn't prefer the true future). step=0 embedding used for all."""
    B = cc.shape[0]
    step = torch.zeros(B, dtype=torch.long, device=DEV)
    mean_c = tc.mean(0, keepdim=True).expand_as(tc).contiguous()
    mean_f = tf.mean(0, keepdim=True).expand_as(tf).contiguous()

    def E(yc, yf):
        return float(ebt.net.energy(cc, cf, yc, yf, step, a).mean())
    return dict(target=E(tc, tf), mean=E(mean_c, mean_f),
                randn=E(torch.randn_like(tc), torch.randn_like(tf)),
                zeros=E(torch.zeros_like(tc), torch.zeros_like(tf)))


def descend_logged(ebt, cc, cf, a, n_steps, tc, tf, warm=False):
    """Replicates eval descent but logs per-step E, grad-rms, effective step, and distance of the
    running iterate y to the TARGET vs to the MEAN latent. Returns (yf_final, trajectory rows)."""
    B = cc.shape[0]
    mean_f = tf.mean(0, keepdim=True).expand_as(tf)
    if warm:                                            # start NEAR the target (tests local basin)
        yc = (tc + 0.3 * torch.randn_like(tc)); yf = (tf + 0.3 * torch.randn_like(tf))
    else:                                               # cold random init (what eval actually uses)
        yc = torch.randn(B, LAT, 4, 4, device=DEV); yf = torch.randn(B, LAT, 8, 8, device=DEV)
    y0f = yf.clone()
    al = float(ebt.alpha.detach())
    rows = []
    with torch.enable_grad():
        for s in range(n_steps):
            yc = yc.detach().requires_grad_(True); yf = yf.detach().requires_grad_(True)
            step = torch.full((B,), min(s, ebt.net.step_embedding.num_embeddings - 1),
                              dtype=torch.long, device=DEV)
            E = ebt.net.energy(cc, cf, yc, yf, step, a)
            gc, gf = torch.autograd.grad(E.sum(), [yc, yf])
            gmag = _rms(gf)
            yc = (yc - ebt.alpha.detach() * gc).detach()
            yf = (yf - ebt.alpha.detach() * gf).detach()
            if ebt.langevin > 0:
                yc = yc + ebt.langevin * torch.randn_like(yc)
                yf = yf + ebt.langevin * torch.randn_like(yf)
            rows.append(dict(step=s, E=float(E.mean()), grms=gmag, eff=al * gmag,
                             d_tgt=_rms(yf - tf), d_mean=_rms(yf - mean_f), moved=_rms(yf - y0f)))
    return yf.detach(), rows


def probe(ebt, test, ae, tag):
    ebt.net.eval()
    b = test.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    L = landscape(ebt, cc, cf, a, tc, tf)
    lo = min(L, key=L.get)
    print(f"\n===== PROBE @ {tag}  (alpha={float(ebt.alpha.detach()):.3f}) =====")
    print(f"  landscape E:  target {L['target']:+.3f} | mean {L['mean']:+.3f} | randn {L['randn']:+.3f} "
          f"| zeros {L['zeros']:+.3f}   -> LOWEST = {lo}  {'[GOOD: target]' if lo=='target' else '[BAD: not target]'}")
    # distance references
    mean_f = tf.mean(0, keepdim=True).expand_as(tf)
    print(f"  ref dists (fine): randn->tgt {_rms(torch.randn_like(tf)-tf):.3f} | mean->tgt "
          f"{_rms(mean_f-tf):.3f} | tgt std {_rms(tf-tf.mean()):.3f}")
    # cold-init descent, sweep n_steps -> where does it land (target vs mean)?
    print(f"  COLD-init descent (final fine-latent distance):  {'nstep':>6} {'d_tgt':>7} {'d_mean':>7}")
    for ns in [1, 2, 4, 8, 16]:
        yf, rows = descend_logged(ebt, cc, cf, a, ns, tc, tf, warm=False)
        r = rows[-1]
        print(f"                                                  {ns:>6} {r['d_tgt']:>7.3f} {r['d_mean']:>7.3f}")
    # full trajectory at ns=8 (the eval setting) to see E + movement per step
    _, rows = descend_logged(ebt, cc, cf, a, 8, tc, tf, warm=False)
    print(f"  COLD ns=8 trajectory:  step |    E    | grad-rms | eff-step | d_tgt | d_mean | moved")
    for r in rows:
        print(f"                         {r['step']:>4} | {r['E']:+7.2f} | {r['grms']:>8.4f} | {r['eff']:>8.4f} "
              f"| {r['d_tgt']:.3f} | {r['d_mean']:.3f} | {r['moved']:.3f}")
    # warm-init descent: does the landscape at least HOLD near the target?
    yfw, roww = descend_logged(ebt, cc, cf, a, 8, tc, tf, warm=True)
    print(f"  WARM-init (tgt+0.3n) ns=8: d_tgt {roww[-1]['d_tgt']:.3f} (start ~0.30)  "
          f"-> {'HOLDS near target' if roww[-1]['d_tgt']<0.4 else 'DRIFTS AWAY from target'}")
    # decode the cold ns=8 prediction, measure sharpness + save a grid
    yf8, _ = descend_logged(ebt, cc, cf, a, 8, tc, tf, warm=False)
    pred01 = decode01(ae, yf8); gt01 = decode01(ae, tf)
    sh = fg_sharp(pred01, gt01)
    print(f"  decoded COLD ns=8 pred: fg-sharp {sh:.4f} (GT-roundtrip ~0.25) "
          f"-> {'COLLAPSED (blank)' if sh<0.08 else 'has structure'}")
    viz.save_samples(f"logs_rhcp/ebt_diag_{tag}.png",
                     [("GT", gt01 * 2 - 1), ("EBT cold ns8", pred01 * 2 - 1)], n=8)
    ebt.net.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--alpha0", type=float, default=0.5)
    ap.add_argument("--bf16", action="store_true", help="bf16 training for speed (probes stay fp32)")
    args = ap.parse_args()
    torch.manual_seed(0)
    M._BF16 = bool(args.bf16)                            # training goes bf16; probe() calls energy directly (fp32)

    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250)
    b0 = test.batch(512, train=False)
    print(f"[latent scale] target fine: std {_rms(b0['tgt_f']-b0['tgt_f'].mean()):.3f}  "
          f"abs-mean {float(b0['tgt_f'].abs().mean()):.3f}  vs randn-init std ~1.0  "
          f"(if init >> target scale, cold descent must travel far)")

    ebt = EBT(EnergyTransformer(n_ctx=6, k=4, d=256).to(DEV), alpha0=args.alpha0)
    opt = torch.optim.AdamW([{"params": ebt.net.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
                             {"params": [ebt.log_alpha], "lr": 5e-2, "weight_decay": 0.0}])
    print(f"[train] fresh EBT, {args.steps} steps, B={args.batch}, bf16={args.bf16}")
    probe(ebt, test, ae, "step0")
    run = 0.0
    gnorm_last = 0.0
    for s in range(1, args.steps + 1):
        bt = data.batch(args.batch, train=True)
        loss = ebt.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            gnorm_last = float(torch.nn.utils.clip_grad_norm_(ebt.net.parameters(), 1.0))
            torch.nn.utils.clip_grad_norm_([ebt.log_alpha], 1.0)
            opt.step()
            run += float(loss.detach())
        if s % 200 == 0:
            print(f"\n[train {s}/{args.steps}] sl1 {run/200:.4f}  alpha {float(ebt.alpha.detach()):.4f}  "
                  f"grad-norm(net) {gnorm_last:.3f}  log_alpha.grad "
                  f"{float(ebt.log_alpha.grad) if ebt.log_alpha.grad is not None else 0:.4f}")
            run = 0.0
            probe(ebt, test, ae, f"step{s}")
    print("\nDIAG DONE -> review logs_rhcp/ebt_diag_step*.png (GT | EBT cold ns8)")


if __name__ == "__main__":
    main()
