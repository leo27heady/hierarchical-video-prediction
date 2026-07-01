"""FIXED EBT — applies the root-cause fixes found by the collapse investigation and re-runs the probe
to VERIFY the fix (E(target) < E(mean); cold-init descent reaches the target; decoded digits sharp).

Fixes vs the broken hvp_rev/step1_mech.py EBT:
  1. LATENT STANDARDIZATION: operate in unit-std space (z/std) so cold randn(std 1) init matches the
     target scale (real EBT does this via x0.18215). Un-standardize before decode.
  2. FIXED (non-learnable) ALPHA, calibrated so alpha*|grad E|*n_steps ~ init->target distance
     (real EBT S2: alpha fixed, not learned). Per-element [alpha/2, 2alpha] randomization at train.
  3. COLD randn init at BOTH train and eval (no replay-buffer warm-start) -> removes the mispaired-
     target mean-predictor bug AND the train/eval init mismatch (matches real EBT inference).
  4. EVAL LANGEVIN = 0 (real EBT infer_langevin_dynamics_noise=0); small train langevin on unit scale.
  5. ALIGNED train/eval steps: train randint(2,K+1), eval K -> step-embeddings 0..K-1 are all trained.

  venv/Scripts/python.exe -u -m hvp_rev.ebt_fixed --steps 1500 --alpha 0 --bf16   # alpha 0 => auto-calibrate
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae, viz
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import ActionHierData, decode01, fg_sharp
from hvp_rev.faithful_nets import EnergyTransformer, LAT, double_backward_attention

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _rms(t):
    return float(t.pow(2).mean().sqrt())


class FixedEBT:
    def __init__(self, net, alpha, sc, sf, train_lang=0.05, k_eval=4):
        self.net = net
        self.alpha = float(alpha)                     # FIXED, not learnable
        self.sc, self.sf = float(sc), float(sf)       # per-level std (standardization)
        self.train_lang = train_lang
        self.k_eval = k_eval
        self.max_steps = k_eval                        # train uses up to k_eval steps (aligned)

    # standardize / un-standardize
    def stdc(self, z): return z / self.sc
    def stdf(self, z): return z / self.sf
    def unstdf(self, z): return z * self.sf

    def descend(self, cc, cf, actions, n_steps, learning, langevin, per_elem=False):
        """cc,cf are STANDARDIZED context. Cold randn init. Returns standardized (yc,yf)."""
        B = cc.shape[0]
        yc = torch.randn(B, LAT, 4, 4, device=DEV)
        yf = torch.randn(B, LAT, 8, 8, device=DEV)
        yc = yc.requires_grad_(True); yf = yf.requires_grad_(True)
        if per_elem:
            scale = torch.empty(B, device=DEV).uniform_(0.5, 2.0).view(-1, 1, 1, 1)
            a = self.alpha * scale
        else:
            a = self.alpha
        for s in range(n_steps):
            cg = bool(learning and s == n_steps - 1)
            step = torch.full((B,), min(s, self.net.step_embedding.num_embeddings - 1),
                              dtype=torch.long, device=DEV)
            if cg:
                with double_backward_attention():
                    with M._ac():
                        E = self.net.energy(cc, cf, yc, yf, step, actions)
            else:
                with M._ac():
                    E = self.net.energy(cc, cf, yc, yf, step, actions)
            gc, gf = torch.autograd.grad(E.sum(), [yc, yf], create_graph=cg)
            yc = yc - a * gc
            yf = yf - a * gf
            if langevin > 0:
                yc = yc + langevin * torch.randn_like(yc)
                yf = yf + langevin * torch.randn_like(yf)
            if not cg:
                yc = yc.detach().requires_grad_(True)
                yf = yf.detach().requires_grad_(True)
        return yc, yf

    def loss(self, cc, cf, tc, tf, actions):
        cc, cf, tc, tf = self.stdc(cc), self.stdf(cf), self.stdc(tc), self.stdf(tf)
        n = int(np.random.randint(2, self.max_steps + 1))
        yc, yf = self.descend(cc, cf, actions, n, learning=True, langevin=self.train_lang, per_elem=True)
        return F.smooth_l1_loss(yc, tc) + F.smooth_l1_loss(yf, tf)

    @torch.no_grad()
    def sample_f(self, cc, cf, actions, n_steps):
        """cold-init eval descent, langevin=0; returns UN-standardized fine latent for decode."""
        with torch.enable_grad():
            yc, yf = self.descend(self.stdc(cc), self.stdf(cf), actions, n_steps,
                                  learning=False, langevin=0.0)
        return self.unstdf(yf.detach())


def calibrate_alpha(fe, cc, cf, actions, k_eval):
    """Set alpha so alpha*|grad E|*k_eval ~ (cold-init -> target distance ~ sqrt(2) in unit space)."""
    cc, cf = fe.stdc(cc), fe.stdf(cf)
    B = cc.shape[0]
    yc = torch.randn(B, LAT, 4, 4, device=DEV).requires_grad_(True)
    yf = torch.randn(B, LAT, 8, 8, device=DEV).requires_grad_(True)
    step = torch.zeros(B, dtype=torch.long, device=DEV)
    E = fe.net.energy(cc, cf, yc, yf, step, actions)
    gc, gf = torch.autograd.grad(E.sum(), [yc, yf])
    g = 0.5 * (_rms(gc) + _rms(gf))
    target_move = 1.4                                  # ~sqrt(2): unit randn init to unit target
    alpha = target_move / (max(k_eval, 1) * max(g, 1e-6))
    return alpha, g


@torch.no_grad()
def probe(fe, test, ae, tag):
    b = test.batch(64, train=False, return_frames=True, seed=777)
    cc, cf, a, tc, tf = b["ctx_c"], b["ctx_f"], b["actions"], b["tgt_c"], b["tgt_f"]
    ccs, cfs, tcs, tfs = fe.stdc(cc), fe.stdf(cf), fe.stdc(tc), fe.stdf(tf)
    step = torch.zeros(cc.shape[0], dtype=torch.long, device=DEV)
    mean_c = tcs.mean(0, keepdim=True).expand_as(tcs).contiguous()
    mean_f = tfs.mean(0, keepdim=True).expand_as(tfs).contiguous()

    def E(yc, yf):
        return float(fe.net.energy(ccs, cfs, yc, yf, step, a).mean())
    Et, Em, Er = E(tcs, tfs), E(mean_c, mean_f), E(torch.randn_like(tcs), torch.randn_like(tfs))
    lo = min([("target", Et), ("mean", Em), ("randn", Er)], key=lambda x: x[1])[0]
    yf = fe.sample_f(cc, cf, a, fe.k_eval)
    d_tgt = _rms(fe.stdf(fe.unstdf(yf) if False else yf) / fe.sf - tfs) if False else _rms((yf / fe.sf) - tfs)
    pred01 = decode01(ae, yf); gt01 = decode01(ae, tf)
    sh = fg_sharp(pred01, gt01)
    print(f"  [{tag}] landscape E: tgt {Et:+.2f} | mean {Em:+.2f} | randn {Er:+.2f} -> LOWEST={lo} "
          f"{'[GOOD]' if lo=='target' else '[BAD]'}  | cold ns{fe.k_eval} d_tgt {d_tgt:.3f} "
          f"(mean->tgt {_rms(mean_f-tfs):.3f}) | fg-sharp {sh:.4f} (GT~0.25) "
          f"{'[COLLAPSED]' if sh<0.08 else '[HAS STRUCTURE]'}", flush=True)
    viz.save_samples(f"logs_rhcp/ebt_fixed_{tag}.png", [("GT", gt01 * 2 - 1), ("EBT", pred01 * 2 - 1)], n=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 => auto-calibrate")
    ap.add_argument("--k_eval", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    M._BF16 = bool(args.bf16)
    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250)
    b0 = data.batch(1024, train=True)
    sc = _rms(b0["tgt_c"] - b0["tgt_c"].mean())
    sf = _rms(b0["tgt_f"] - b0["tgt_f"].mean())
    print(f"[standardize] coarse std {sc:.3f}  fine std {sf:.3f}  -> operating in unit space", flush=True)

    net = EnergyTransformer(n_ctx=6, k=4, d=256).to(DEV)
    fe = FixedEBT(net, alpha=1.0, sc=sc, sf=sf, k_eval=args.k_eval)
    bcal = data.batch(64, train=True)
    a_auto, g = calibrate_alpha(fe, bcal["ctx_c"], bcal["ctx_f"], bcal["actions"], args.k_eval)
    fe.alpha = a_auto if args.alpha == 0 else args.alpha
    print(f"[alpha] init |grad E|={g:.4f}  auto-alpha={a_auto:.3f}  -> using alpha={fe.alpha:.3f} "
          f"(FIXED, not learnable)", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"[train] FIXED EBT: standardized, alpha={fe.alpha:.3f} fixed, cold-init train+eval, "
          f"eval-langevin 0, steps train 2-{args.k_eval}/eval {args.k_eval}", flush=True)
    net.eval(); probe(fe, test, ae, "step0"); net.train()
    run = 0.0
    for s in range(1, args.steps + 1):
        bt = data.batch(args.batch, train=True)
        loss = fe.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)); opt.step()
            run += float(loss.detach())
        if s % 300 == 0:
            print(f"\n[train {s}/{args.steps}] sl1 {run/300:.4f}  grad-norm {gn:.3f}", flush=True)
            run = 0.0
            net.eval(); probe(fe, test, ae, f"step{s}"); net.train()
    print("\nFIXED-EBT DONE -> review logs_rhcp/ebt_fixed_step*.png (GT | EBT)", flush=True)


if __name__ == "__main__":
    main()
