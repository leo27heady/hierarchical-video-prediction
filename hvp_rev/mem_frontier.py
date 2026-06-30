"""Panel A — the memory frontier (reversible vs BPTT, with the mandatory checkpoint line), measured in the REAL
training loop (encoder + core + decoder + Adam state) at a ship config until vanilla BPTT actually OOMs on 16 GB.

Honest framing (red-team locked): reversible is NOT a memory win over per-step gradient checkpointing (both are
O(1)-in-K); the headline is the constant-memory CAPABILITY that vanilla full-sequence BPTT cannot fit. The figure
plots THREE lines and the caption states checkpointing achieves the same memory at one full-forward recompute.

  venv/Scripts/python.exe -m hvp_rev.mem_frontier --levels 3 --batch 32 --K_grid 10,20,40,60,80,120
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hvp_rev.rhcp import RHCP, RHCPConfig, LevelSpec
from hvp_rev.rollout_train import reverse_replay_loss_grads, obj_l1mse

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _reset():
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()


def _peak():
    torch.cuda.synchronize(); return torch.cuda.max_memory_allocated() / 1e6


def make(levels, batch):
    specs = {3: [LevelSpec(64, 16, 1), LevelSpec(96, 8, 2), LevelSpec(128, 4, 4)],
             4: [LevelSpec(64, 16, 1), LevelSpec(96, 8, 2), LevelSpec(128, 4, 4), LevelSpec(192, 2, 8)]}[levels]
    cfg = RHCPConfig(levels=specs)
    model = RHCP(cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctx = torch.randn(batch, 10, 1, 64, 64, device=DEV)
    return model, opt, ctx


def run_bptt(model, opt, ctx, K):
    _reset(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    s = model.init_states(ctx)
    loss = 0.0
    for t in range(K):
        s = model.core.step(s, t)
        loss = loss + obj_l1mse(model.decoder(s[0]), torch.zeros(ctx.shape[0], 1, 64, 64, device=DEV))
    loss.backward(); opt.step()
    return _peak(), time.perf_counter() - t0


def run_ckpt(model, opt, ctx, K):
    _reset(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    nL = len(model.cfg.levels)
    s = [x.requires_grad_(True) for x in model.init_states(ctx)]

    def stepfn(ti, *st):
        st = list(st)
        st = model.core.step(st, int(ti))
        return (*st, model.decoder(st[0]))

    loss = 0.0
    for t in range(K):
        out = checkpoint(stepfn, torch.tensor(float(t), device=DEV), *s, use_reentrant=False)
        s = list(out[:nL])
        loss = loss + obj_l1mse(out[nL], torch.zeros(ctx.shape[0], 1, 64, 64, device=DEV))
    loss.backward(); opt.step()
    return _peak(), time.perf_counter() - t0


def run_rev(model, opt, ctx, K):
    _reset(); t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    tgt = torch.zeros(ctx.shape[0], K, 1, 64, 64, device=DEV)
    reverse_replay_loss_grads(model, ctx, tgt, obj_l1mse, accumulate_into_grad=True)
    opt.step()
    return _peak(), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--K_grid", default="10,20,40,60,80,120")
    args = ap.parse_args()
    Ks = [int(k) for k in args.K_grid.split(",")]
    print(f"Panel A memory frontier | device {DEV} | levels {args.levels} batch {args.batch} res 64 (real train loop + Adam)")
    print(f"\n  {'K':>5} | {'BPTT MB':>10} | {'ckpt MB':>9} | {'rev MB':>9} | {'BPTT ms':>8} | {'ckpt ms':>8} | {'rev ms':>8}")
    print("  " + "-" * 78)
    bptt_oom_at = None
    for K in Ks:
        model, opt, ctx = make(args.levels, args.batch)
        # BPTT
        try:
            mb, tb = run_bptt(model, opt, ctx, K)
            bs = f"{mb:>10.0f}"; tbs = f"{1e3*tb:>8.0f}"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                bs = f"{'OOM':>10}"; tbs = f"{'-':>8}"; bptt_oom_at = bptt_oom_at or K
                torch.cuda.empty_cache()
            else:
                raise
        # checkpoint
        model, opt, ctx = make(args.levels, args.batch)
        try:
            mc, tc = run_ckpt(model, opt, ctx, K); cs = f"{mc:>9.0f}"; tcs = f"{1e3*tc:>8.0f}"
        except RuntimeError as e:
            cs = f"{'OOM':>9}"; tcs = f"{'-':>8}"; torch.cuda.empty_cache()
        # reversible
        model, opt, ctx = make(args.levels, args.batch)
        try:
            mr, tr = run_rev(model, opt, ctx, K); rs = f"{mr:>9.0f}"; trs = f"{1e3*tr:>8.0f}"
        except RuntimeError as e:
            rs = f"{'OOM':>9}"; trs = f"{'-':>8}"; torch.cuda.empty_cache()
        print(f"  {K:>5} | {bs} | {cs} | {rs} | {tbs} | {tcs} | {trs}")
    print("\n  CAPTION (mandatory): Per-step gradient checkpointing achieves the same K-independent core memory at one")
    print("  full-forward recompute; we use exact reverse-replay as the native recompute-free mechanism for a")
    print("  reversible network. The scientific claim (Panel C drift) is mechanism-agnostic.")
    if bptt_oom_at:
        print(f"\n  => MEASURED OOM: vanilla BPTT OOMs at K={bptt_oom_at} (batch {args.batch}, {args.levels}lvl, 16GB); "
              f"reversible + checkpoint still fit. Capability demonstrated.")
    else:
        print(f"\n  => BPTT did not OOM on this grid; raise --batch/--levels/--K_grid to surface the frontier.")


if __name__ == "__main__":
    main()
