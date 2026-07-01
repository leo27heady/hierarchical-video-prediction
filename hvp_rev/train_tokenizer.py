"""Train the learned token-pyramid AE (Chapter C) on the looping testbed frames. Per-frame; random-level dropout.

  venv/Scripts/python.exe -m hvp_rev.train_tokenizer --variant adaptive --steps 8000 --out logs_rhcp/ae_adaptive.pt
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev.tokenizer import TokenPyramidAE, TokCfg

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = "data/loop_train_3000x100.npy"


class Frames:
    """All frames of the looping train set as a flat pool in [-1,1]."""
    def __init__(self, path=TRAIN, n_val=2000):
        a = np.load(path)                       # (N,T,64,64) uint8
        a = a.reshape(-1, 64, 64).astype(np.float32) / 127.5 - 1.0
        self.x = torch.from_numpy(a).unsqueeze(1)   # (M,1,64,64) [-1,1]
        self.M = self.x.shape[0]
        self.n_val = n_val

    def batch(self, B, train=True):
        hi = self.M - self.n_val if train else self.M
        lo = 0 if train else self.M - self.n_val
        idx = torch.randint(lo, hi, (B,))
        return self.x[idx].to(DEV)


@torch.no_grad()
def eval_recon(ae, data, n=512, B=128):
    ae.eval()
    L = len(ae.grids)
    errs = [0.0] * L
    done = 0
    while done < n:
        x = data.batch(min(B, n - done), train=False)
        recons, _ = ae(x)
        for l in range(L):
            errs[l] += float(((recons[l] - x) ** 2).mean(dim=(1, 2, 3)).sum())
        done += x.shape[0]
    ae.train()
    return [e / done for e in errs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="adaptive", choices=["adaptive", "cascaded", "flat"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = TokCfg(variant=args.variant)
    ae = TokenPyramidAE(cfg).to(DEV)
    data = Frames()
    L = len(ae.grids)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb, name=f"ae-{args.variant}", config=vars(args), reinit=True)
    print(f"train_tokenizer variant={args.variant} steps={args.steps} params={ae.n_params()/1e6:.2f}M "
          f"frames={data.M} levels={ae.grids}", flush=True)
    run = 0.0
    for step in range(1, args.steps + 1):
        x = data.batch(args.batch, train=True)
        lat = ae.encode(x)
        lvl = int(torch.randint(0, L, (1,)))         # random-level dropout: supervise ONE level
        recon = ae.decode_level(lat, lvl)
        w = 4 ** lvl                                  # up-weight coarse (MilliVid)
        loss = w * (F.l1_loss(recon, x) + F.mse_loss(recon, x))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0); opt.step(); sched.step()
        run += float(loss.detach())
        if step % 100 == 0:
            if wb: wb.log({"train/loss": run / 100, "lr": sched.get_last_lr()[0]}, step=step)
            run = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            e = eval_recon(ae, data)
            msg = "  ".join(f"L{l}(g{ae.grids[l]}) {e[l]:.2f}" for l in range(L))
            print(f"[step {step:5d}] per-level recon-MSE(->64): {msg}", flush=True)
            if wb: wb.log({f"val/recon_L{l}": e[l] for l in range(L)}, step=step)
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({"model": ae.state_dict(), "cfg": cfg}, args.out)
        print(f"saved {args.out}")
    if wb: wb.finish()


if __name__ == "__main__":
    main()
