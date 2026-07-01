"""Stage 1: the EBT VALUE-PROP gate (the honest go/no-go before any MilliVid build).

Two tests of EBT's actual claims, on genuinely HARD frames of the looping dataset:
  (A) SYSTEM-2: does MORE inference energy-descent steps ("think longer") reduce error -- and MORE so on hard frames
      (pan-direction changes) than easy frames? This is EBT's headline, unclaimed-for-video.
  (B) MULTIMODALITY vs HORIZON: predict k frames ahead (k=1..10). As k grows the future diverges; does EBT keep its
      prediction SHARP (commit to a mode) while regression blurs to the mean?
UNCONDITIONED (no action given) so the future is genuinely multimodal. Verdict PASS -> proceed to minimal MilliVid+EBT;
NULL -> fall back to the verified reversibility/theory paper.

  venv/Scripts/python.exe -m hvp_rev.ebt_stage1 --ae logs_rhcp/convae_loop.pt --steps 6000 --k 4
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.convae import load as load_ae
from hvp_rev.ebt_video import LatentEnergy, mcmc, LAT, HW
from hvp_rev import long_eval as LE

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Stage1Data:
    """Encode looping clips; sample (n_ctx context latents -> target latent k frames ahead) UNCONDITIONED, with a
    per-sample HARDNESS label: hard = the pan direction at the target differs from the last context action."""

    def __init__(self, ae, clip_path, act_path, n_ctx=2, k=4, n_clips=2500, max_t=100, n_val=300):
        clips = np.load(clip_path)[:n_clips]
        acts = np.load(act_path)[:n_clips].astype(np.int64)
        N, T = clips.shape[0], min(max_t, clips.shape[1])
        x = torch.from_numpy(clips[:, :T].astype(np.float32) / 127.5 - 1.0).unsqueeze(2)
        with torch.no_grad():
            self.z = torch.stack([ae.encode(x[:, t].to(DEV)).cpu() for t in range(T)], 1)   # (N,T,LAT,8,8)
        self.act = torch.from_numpy(acts[:, :T])
        self.N, self.T, self.n_ctx, self.k = N, T, n_ctx, k
        self.n_val = n_val

    def batch(self, B, train=True):
        hi = self.N - self.n_val if train else self.N
        lo = 0 if train else self.N - self.n_val
        ci = torch.randint(lo, hi, (B,))
        ti = torch.randint(self.n_ctx, self.T - self.k, (B,))
        ctx = torch.stack([self.z[ci[b], ti[b] - self.n_ctx:ti[b]] for b in range(B)])       # (B,n_ctx,LAT,8,8)
        nxt = torch.stack([self.z[ci[b], ti[b] + self.k - 1] for b in range(B)])             # k-ahead target
        # hardness: does the pan direction change across the prediction window vs the last context action?
        last_ctx_a = torch.stack([self.act[ci[b], ti[b] - 1] for b in range(B)])
        win_a = torch.stack([self.act[ci[b], ti[b]:ti[b] + self.k].float().mean() for b in range(B)])
        hard = (win_a.sign() != last_ctx_a.sign()) | (last_ctx_a == 0)                        # direction change / hold
        return ctx.to(DEV), nxt.to(DEV), hard.to(DEV)


def _fg_sharp(ae, zpred, zgt):
    with torch.no_grad():
        p = ae.decode(zpred); g = ae.decode(zgt)
    pp = ((p.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    gg = ((g.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
    fg = float(LE.fg_mse(pp, gg).mean())
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=p.dtype, device=p.device)[None, None]
    sh = float(F.conv2d((p.clamp(-1, 1) + 1) * 0.5, k, padding=1).var().item())
    return fg, sh


class ReplayBuffer:
    """FIFO of past FINAL predictions. Re-initializing part of each batch's descent from these simulates much longer
    descent trajectories than n_mcmc alone -> shapes the energy landscape far from the target, which is what makes
    'think longer' at inference actually help (the S2 landscape-regularization the S1 recipe lacks)."""

    def __init__(self, size=1024):
        self.buf, self.size = [], size

    def sample(self, n):
        if len(self.buf) < n:
            return None
        idx = torch.randint(0, len(self.buf), (n,))
        return torch.stack([self.buf[i] for i in idx])

    def add(self, batch):
        for b in batch.detach().cpu():
            self.buf.append(b)
        self.buf = self.buf[-self.size:]


def bo_predict(model, ctx, n_steps, best_of_n=4, langevin=0.05):
    """EBT inference = energy DESCENT (needs grad w.r.t. y) + Best-of-N argmin-energy. NOT under no_grad."""
    best_y, best_E = None, None
    for _ in range(best_of_n):
        y, _, _ = mcmc(model, ctx, None, n_steps, langevin=langevin, train=False)   # mcmc handles grad internally
        with torch.no_grad():
            E = model.energy(ctx, None, y).mean().item()
        if best_E is None or E < best_E:
            best_y, best_E = y.detach(), E
    return best_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="logs_rhcp/convae_loop.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_mcmc", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--recipe", default="s2", choices=["s1", "s2"], help="s2 = thinking recipe (replay buffer + no-detach + randomized steps)")
    args = ap.parse_args()
    torch.manual_seed(0)
    ae = load_ae(args.ae)
    data = Stage1Data(ae, "data/loop_train_3000x100.npy", "data/loop_train_3000x100_actions.npy", k=args.k)

    ebt = LatentEnergy(use_action=False).to(DEV)
    base = nn.Sequential(nn.Conv2d(LAT * 2, 128, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, LAT, 3, padding=1)).to(DEV)
    oe = torch.optim.Adam([{"params": [p for n, p in ebt.named_parameters() if n != "log_alpha"], "lr": 3e-4},
                           {"params": [ebt.log_alpha], "lr": 3e-2, "weight_decay": 0.0}])
    ob = torch.optim.Adam(base.parameters(), lr=3e-4)
    print(f"Stage-1 EBT value-prop gate | UNCONDITIONED k={args.k}-ahead | recipe={args.recipe} | steps={args.steps}", flush=True)
    rb = ReplayBuffer(1024) if args.recipe == "s2" else None
    for s in range(1, args.steps + 1):
        ctx, nxt, _ = data.batch(args.batch)
        B = ctx.shape[0]
        if args.recipe == "s2":
            # replay-buffer warm start (half the batch) + no-detach + final loss + langevin + randomized alpha & steps
            y0 = torch.randn(B, LAT, HW, HW, device=DEV)
            rs = rb.sample(B // 2)
            if rs is not None:
                y0[: B // 2] = rs.to(DEV)
            nsteps = args.n_mcmc + int(torch.randint(0, 3, (1,)))          # randomized num-steps
            yN, _, le = mcmc(ebt, ctx, None, nsteps, target=nxt, mode="s2", langevin=0.1, randomize=True,
                             train=True, y_init=y0)
            rb.add(yN)
        else:
            _, _, le = mcmc(ebt, ctx, None, args.n_mcmc, target=nxt, mode="s1", train=True)
        oe.zero_grad(set_to_none=True); le.backward(); torch.nn.utils.clip_grad_norm_(ebt.parameters(), 1.0); oe.step()
        lb = F.smooth_l1_loss(base(ctx.reshape(B, -1, HW, HW)), nxt)
        ob.zero_grad(set_to_none=True); lb.backward(); ob.step()
        if s % 1000 == 0:
            print(f"  [step {s}] EBT {float(le.detach()):.4f} base {float(lb.detach()):.4f} alpha {float(ebt.alpha()):.1f}", flush=True)

    # ---- (A) System-2: error vs inference steps, split easy/hard ----
    ctx, nxt, hard = data.batch(1024, train=False)
    print(f"\n  (A) SYSTEM-2  (hard frames = {int(hard.sum())}/{len(hard)}):  fg-MSE by #inference energy-steps")
    print(f"      {'steps':>6} | {'all':>8} | {'easy':>8} | {'hard':>8}")
    e0 = h0 = None
    for ns in [1, 2, 4, 8, 16, 32]:
        y = bo_predict(ebt, ctx, ns, best_of_n=4)
        fa, _ = _fg_sharp(ae, y, nxt)
        fe, _ = _fg_sharp(ae, y[~hard], nxt[~hard])
        fh, _ = _fg_sharp(ae, y[hard], nxt[hard])
        print(f"      {ns:>6} | {fa:>8.4f} | {fe:>8.4f} | {fh:>8.4f}")
        if ns == 2: e0, h0 = fe, fh
        if ns == 32:
            sys2_all = (h0 - fh) / max(h0, 1e-6)
            print(f"      -> think-longer (2->32 steps) reduces HARD-frame error by {100*sys2_all:.1f}% "
                  f"(easy by {100*(e0-fe)/max(e0,1e-6):.1f}%)")

    # ---- (B) multimodality vs horizon: EBT vs regression sharpness (GT-relative) ----
    print(f"\n  (B) SHARPNESS vs regression (higher=commits to a mode; regression blurs to mean):")
    y = bo_predict(ebt, ctx, 16, best_of_n=8)
    with torch.no_grad():
        bp = base(ctx.reshape(ctx.shape[0], -1, HW, HW))
    _, e_sh = _fg_sharp(ae, y, nxt)
    _, b_sh = _fg_sharp(ae, bp, nxt)
    _, g_sh = _fg_sharp(ae, nxt, nxt)
    print(f"      EBT {e_sh:.4f}  regression {b_sh:.4f}  GT {g_sh:.4f}")
    win_sys2 = (h0 is not None) and (h0 - fh) > 0.05 * h0
    win_sharp = e_sh > b_sh * 1.15
    print(f"\n  STAGE-1 VERDICT: think-longer-helps-hard={win_sys2}  sharper-than-regression={win_sharp}  -> "
          f"{'PASS: EBT value prop shows -> proceed to minimal MilliVid+EBT' if (win_sys2 or win_sharp) else 'NULL: EBT shows no advantage here -> fall back to the verified reversibility/theory paper'}")


if __name__ == "__main__":
    main()
