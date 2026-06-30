"""D1 trainer for RHCP on Moving-MNIST (the OpenSTL tab:mmnist target).

Smoke (proves the end-to-end pipeline trains; ~1-2 min on the 4060 Ti):
    venv/Scripts/python.exe -m hvp_rev.rhcp_train --smoke

Full from-scratch D1 (1-2 day run; HONEST expectation = MSE high-30s/low-40s, BELOW the table -- see
docs/RHCP_ARCHITECTURE.md sec 7; table-range needs the warm-start in STEP 7):
    venv/Scripts/python.exe -m hvp_rev.rhcp_train --steps 100000 --batch 16 --n-in 10 --n-out 10

This trains the SELF-CONTAINED RHCP (PyramidEncoder + reversible clock core + PixelDecoder) on local clip files so it
runs on THIS branch without the cwvae backbone. For a table-comparable number, port the model into
research/cwvae-backbone:hvp/e2e/train.py (on-the-fly train gen + fixed mnist_test_seq.npy val + the verified
hvp/eval/openstl_metrics.py). MSE here uses the OpenSTL convention (mean over N,T, SUM over C,H,W on [0,1]).
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev.rhcp import RHCP, RHCPConfig, LevelSpec

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _find_clips(min_T: int) -> str:
    cands = sorted(glob.glob("data/mmnist_long_test_*x*_d2_s64.npy"))
    best = None
    for f in cands:
        try:
            T = np.load(f, mmap_mode="r").shape[0]
        except Exception:
            continue
        if T >= min_T and "pturn" not in f:
            # prefer the largest N at sufficient T
            N = np.load(f, mmap_mode="r").shape[1]
            if best is None or N > best[1]:
                best = (f, N)
    if best is None:
        raise FileNotFoundError(f"no data/mmnist_long_test_*.npy with T>={min_T}")
    return best[0]


class ClipData:
    """(T, N, H, W) uint8 -> tensors (B, T, 1, H, W) in [-1, 1]. Holds out the last `n_val` clips for eval."""

    def __init__(self, path: str, n_val: int = 256):
        arr = np.load(path)                       # (T, N, H, W)
        arr = np.transpose(arr, (1, 0, 2, 3))     # (N, T, H, W)
        self.x = torch.from_numpy(arr).float().div_(127.5).sub_(1.0).unsqueeze(2)  # (N,T,1,H,W) in [-1,1]
        self.N, self.T = self.x.shape[0], self.x.shape[1]
        self.n_val = min(n_val, self.N // 4)
        self.path = path

    def batch(self, B, n_in, n_out, train=True):
        hi = self.N - self.n_val if train else self.N
        lo = 0 if train else self.N - self.n_val
        idx = torch.randint(lo, hi, (B,))
        clip = self.x[idx]                        # (B,T,1,H,W)
        return clip[:, :n_in].to(DEV), clip[:, n_in:n_in + n_out].to(DEV)


def openstl_mse(pred01, true01):
    """OpenSTL convention: mean over (N,T) then SUM over (C,H,W). pred/true in [0,1], (N,T,C,H,W)."""
    return ((pred01 - true01) ** 2).mean(dim=(0, 1)).sum().item()


def to01(x):
    return (x.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def evaluate(model, data, n_in, n_out, n_clips=256, B=64):
    model.eval()
    preds, trues = [], []
    done = 0
    while done < n_clips:
        b = min(B, n_clips - done)
        x, y = data.batch(b, n_in, n_out, train=False)
        yh = model.rollout(x, n_out)
        preds.append(to01(yh).cpu()); trues.append(to01(y).cpu())
        done += b
    pred = torch.cat(preds).permute(0, 1, 2, 3, 4)   # (N,T,C,H,W)
    true = torch.cat(trues)
    mse = openstl_mse(pred, true)
    mae = (pred - true).abs().mean(dim=(0, 1)).sum().item()
    model.train()
    return {"mse": mse, "mae": mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick pipeline check (~300 steps)")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-in", type=int, default=10)
    ap.add_argument("--n-out", type=int, default=10)
    ap.add_argument("--levels", default="16:64:1,8:96:2,4:128:4", help="spatial:channels:tick comma-list (bot->top)")
    ap.add_argument("--deep-sup", type=float, default=0.0, help="latent deep-sup weight (0=pixel-only smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.log_every = 300, 300, 25

    levels = []
    for tok in args.levels.split(","):
        sp, ch, tk = (int(v) for v in tok.split(":"))
        levels.append(LevelSpec(channels=ch, spatial=sp, tick=tk))
    cfg = RHCPConfig(levels=levels)

    need_T = args.n_in + args.n_out
    data = ClipData(_find_clips(need_T))
    print(f"data: {data.path}  N={data.N} T={data.T}  (val held-out {data.n_val})  device={DEV}")

    model = RHCP(cfg).to(DEV)
    print(f"RHCP params: {model.n_params()/1e6:.2f}M  levels={[(s.spatial,s.channels,s.tick) for s in levels]}  "
          f"protocol {args.n_in}->{args.n_out}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    t0 = time.perf_counter()
    run_loss = 0.0
    for step in range(1, args.steps + 1):
        x, y = data.batch(args.batch, args.n_in, args.n_out, train=True)
        yh = model.rollout(x, args.n_out)
        loss = F.l1_loss(yh, y) + F.mse_loss(yh, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += float(loss.detach())
        if step % args.log_every == 0:
            sps = step / (time.perf_counter() - t0)
            print(f"[step {step:6d}] loss {run_loss/args.log_every:.4f}  {sps:.1f} it/s  lr {sched.get_last_lr()[0]:.2e}")
            run_loss = 0.0
        if step % args.eval_every == 0:
            res = evaluate(model, data, args.n_in, args.n_out)
            print(f"  >> eval @ {step}: OpenSTL-MSE {res['mse']:.2f}  MAE {res['mae']:.2f}  "
                  f"(table tier: CrevNet 30.15 / SimVP 32.15; honest from-scratch floor high-30s/low-40s)")
    print("done.")


if __name__ == "__main__":
    main()
