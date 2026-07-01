"""Stage 0b: EBT next-frame prediction on FROZEN conv-AE latents of the looping dataset.

Tests, on REAL data: (1) does the EBT energy objective produce good next-frame predictions vs a matched single-step
regression baseline, (2) does energy converge at inference, (3) does mean-collapse / multimodality bite. Action-
conditioned (the looping pan action makes next-frame well-posed); an UNconditioned variant is the multimodality probe.

  venv/Scripts/python.exe -m hvp_rev.ebt_video --ae logs_rhcp/convae_loop.pt --steps 8000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.convae import load as load_ae
from hvp_rev import long_eval as LE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAT, HW = 16, 8


# ------------------------------------------------------------------ data: (context latents, action) -> next latent
class LatentPairs:
    """Encode looping clips to latents once; sample (n_ctx context frames + action, next frame) latent pairs."""

    def __init__(self, ae, clip_path, act_path, n_ctx=2, n_clips=2500, max_t=100, n_val=256):
        clips = np.load(clip_path)[:n_clips]                 # (N,T,64,64) uint8
        acts = np.load(act_path)[:n_clips]                   # (N,T) int
        N, T = clips.shape[0], min(max_t, clips.shape[1])
        x = torch.from_numpy(clips[:, :T].astype(np.float32) / 127.5 - 1.0).unsqueeze(2)   # (N,T,1,64,64)
        with torch.no_grad():
            z = torch.stack([ae.encode(x[:, t].to(DEV)).cpu() for t in range(T)], 1)       # (N,T,LAT,8,8)
        self.z = z                                           # latents
        self.act = torch.from_numpy(acts[:, :T].astype(np.int64)) + 1   # {-1,0,1} -> {0,1,2} for nn.Embedding
        self.N, self.T, self.n_ctx = N, T, n_ctx
        self.n_val = n_val
        self.ae = ae

    def batch(self, B, train=True):
        hi = self.N - self.n_val if train else self.N
        lo = 0 if train else self.N - self.n_val
        ci = torch.randint(lo, hi, (B,))
        ti = torch.randint(self.n_ctx, self.T, (B,))         # predict frame ti from [ti-n_ctx..ti-1]
        ctx = torch.stack([self.z[ci[b], ti[b] - self.n_ctx:ti[b]] for b in range(B)])     # (B,n_ctx,LAT,8,8)
        nxt = torch.stack([self.z[ci[b], ti[b]] for b in range(B)])                        # (B,LAT,8,8)
        act = torch.stack([self.act[ci[b], ti[b]] for b in range(B)])                      # (B,)
        return ctx.to(DEV), act.to(DEV), nxt.to(DEV)


# ------------------------------------------------------------------ energy over latents
class LatentEnergy(nn.Module):
    """E(context latents, action, candidate) -> scalar. Conv over the 8x8 latent grid; action added as a bias plane."""

    def __init__(self, lat=LAT, n_ctx=2, n_act=3, w=128, use_action=True):
        super().__init__()
        self.use_action = use_action
        self.act_emb = nn.Embedding(n_act, lat)              # maps action -> a latent-plane bias
        cin = lat * (n_ctx + 1) + (lat if use_action else 0)
        self.net = nn.Sequential(
            nn.Conv2d(cin, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU())
        self.head = nn.Linear(w, 1)
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def alpha(self):
        return self.log_alpha.exp().clamp(min=1e-4, max=200.0)   # wide ceiling; alpha gets its own high-LR group

    def energy(self, ctx, act, y):
        B = ctx.shape[0]
        parts = [ctx.reshape(B, -1, HW, HW), y]
        if self.use_action:
            parts.append(self.act_emb(act)[:, :, None, None].expand(-1, -1, HW, HW))
        h = self.net(torch.cat(parts, 1))
        return self.head(h.mean(dim=(2, 3)))                 # (B,1)


def mcmc(model, ctx, act, n_steps, target=None, mode="s1", langevin=0.0, randomize=False, train=False, scale=2.0,
         y_init=None):
    """Faithful EBT descent (verified vs the real repo): y<-y-alpha*grad E; per-element multiplicative randomized
    alpha; NO raw-grad clamp; SmoothL1 loss. mode='s1' = detach between steps + loss at EVERY step (the paper's
    stable-first recipe); 's2' = no detach + loss on final step only. y_init warm-starts descent (replay buffer)."""
    y = torch.randn(ctx.shape[0], LAT, HW, HW, device=ctx.device) if y_init is None else y_init.clone()
    a0 = model.alpha()
    traj, losses = [], []
    for _ in range(n_steps):
        y = y.requires_grad_(True)
        E = model.energy(ctx, act, y)
        g = torch.autograd.grad(E.sum(), y, create_graph=train)[0]
        a = (a0 / scale + torch.rand_like(y) * (a0 * scale - a0 / scale)) if randomize else a0
        y = y - a * g
        if langevin > 0:
            y = y + langevin * torch.randn_like(y)
        traj.append(float(E.mean().detach()))
        if train and target is not None and mode == "s1":
            losses.append(F.smooth_l1_loss(y, target))
            y = y.detach()                                   # S1: detach between steps
        elif not train:
            y = y.detach()
    loss = None
    if train and target is not None:
        loss = torch.stack(losses).mean() if mode == "s1" else F.smooth_l1_loss(y, target)
    return y, traj, loss


def predict(model, ctx, act, n_steps, langevin=0.0, best_of_n=1):
    best_y, best_E, best_traj = None, None, None
    for _ in range(best_of_n):
        y, traj, _ = mcmc(model, ctx, act, n_steps, langevin=langevin, train=False)
        with torch.no_grad():
            E = model.energy(ctx, act, y).mean().item()
        if best_E is None or E < best_E:
            best_y, best_E, best_traj = y, E, traj
    return best_y, best_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="logs_rhcp/convae_loop.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_mcmc", type=int, default=3)
    ap.add_argument("--infer_steps", type=int, default=8)
    ap.add_argument("--no_action", action="store_true", help="unconditioned = multimodality probe")
    args = ap.parse_args()
    torch.manual_seed(0)
    ae = load_ae(args.ae)
    data = LatentPairs(ae, "data/loop_train_3000x100.npy", "data/loop_train_3000x100_actions.npy")
    use_act = not args.no_action
    ebt = LatentEnergy(use_action=use_act).to(DEV)
    base = nn.Sequential(nn.Conv2d(LAT * 2 + (LAT if use_act else 0), 128, 3, padding=1), nn.SiLU(),
                         nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, LAT, 3, padding=1)).to(DEV)
    # dedicated high-LR, zero-wd param group for alpha ("by far the most important EBT hyperparameter")
    alpha_p = [ebt.log_alpha]
    rest_p = [p for n, p in ebt.named_parameters() if n != "log_alpha"]
    oe = torch.optim.Adam([{"params": rest_p, "lr": 3e-4}, {"params": alpha_p, "lr": 3e-2, "weight_decay": 0.0}])
    ob = torch.optim.Adam(base.parameters(), lr=3e-4)
    print(f"EBT-video Stage-0b (S1 recipe: SmoothL1, per-step loss+detach, per-elem alpha, alpha-LR 3e-2) | "
          f"action={'on' if use_act else 'OFF (multimodality probe)'} n_mcmc={args.n_mcmc} steps={args.steps}", flush=True)

    def base_pred(ctx, act):
        B = ctx.shape[0]
        parts = [ctx.reshape(B, -1, HW, HW)]
        if use_act:
            parts.append(ebt.act_emb(act)[:, :, None, None].expand(-1, -1, HW, HW).detach())
        return base(torch.cat(parts, 1))

    run_e = run_b = 0.0
    nan = False
    for s in range(1, args.steps + 1):
        ctx, act, nxt = data.batch(args.batch)
        _, _, le = mcmc(ebt, ctx, act, args.n_mcmc, target=nxt, mode="s1", train=True)   # S1: stable-first
        oe.zero_grad(set_to_none=True); le.backward(); torch.nn.utils.clip_grad_norm_(ebt.parameters(), 1.0); oe.step()
        lb = F.smooth_l1_loss(base_pred(ctx, act), nxt)
        ob.zero_grad(set_to_none=True); lb.backward(); ob.step()
        run_e += float(le.detach()); run_b += float(lb.detach())
        if not torch.isfinite(le):
            print(f"  [step {s}] EBT loss NON-FINITE -> unstable (would fall back to lower alpha-LR)"); nan = True; break
        if s % 1000 == 0:
            print(f"  [step {s:5d}] EBT SmoothL1 {run_e/1000:.4f}  baseline {run_b/1000:.4f}  alpha {float(ebt.alpha()):.2f}", flush=True)
            run_e = run_b = 0.0
    if nan:
        return

    # eval: decode predictions -> foreground metrics vs GT next frame
    ctx, act, nxt = data.batch(256, train=False)
    yhat, traj = predict(ebt, ctx, act, args.infer_steps, langevin=0.05, best_of_n=4)
    with torch.no_grad():
        pe = ae.decode(yhat); pb = ae.decode(base_pred(ctx, act)); gt = ae.decode(nxt)
    to01 = lambda t: ((t.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None].transpose(0, 1, 2, 3, 4)
    def fg(pred):
        p = ((pred.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]      # (N,1,1,64,64)
        g = ((gt.clamp(-1, 1) + 1) * 0.5).cpu().numpy()[:, None]
        return float(LE.fg_mse(p, g).mean()), float(LE.centroid_l2(p, g).mean())
    e_fg, e_c = fg(pe); b_fg, b_c = fg(pb)

    def lapvar(img):   # sharpness: variance of the Laplacian (mean-seeking blur -> LOW; committing to a mode -> HIGH)
        x = ((img.clamp(-1, 1) + 1) * 0.5)
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)[None, None]
        return float(F.conv2d(x, k, padding=1).var().item())
    e_sh, b_sh, g_sh = lapvar(pe), lapvar(pb), lapvar(gt)
    mono = all(traj[i + 1] <= traj[i] + 1e-4 for i in range(len(traj) - 1))
    print(f"\n  energy @infer: {[round(t,2) for t in traj]} -> {'converges' if mono else 'NOT monotone'}")
    print(f"  decoded next-frame  EBT: fg-MSE {e_fg:.4f} centroid {e_c:.2f}px sharpness {e_sh:.4f}")
    print(f"                 baseline: fg-MSE {b_fg:.4f} centroid {b_c:.2f}px sharpness {b_sh:.4f}   (GT sharpness {g_sh:.4f})")
    if use_act:
        print(f"  => action-conditioned (deterministic): EBT {'competitive' if e_fg <= b_fg*1.25 else 'lags'} regression on fg-MSE")
    else:
        print(f"  => MULTIMODALITY PROBE: EBT sharpness {e_sh:.4f} vs regression {b_sh:.4f} (GT {g_sh:.4f}) -> "
              f"{'EBT COMMITS to a sharp mode (beats mean-blur)' if e_sh > b_sh*1.3 else 'EBT also blurs to the mean (the documented limitation)'}")


if __name__ == "__main__":
    main()
