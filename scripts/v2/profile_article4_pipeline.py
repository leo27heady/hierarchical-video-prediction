"""Profile the controlled Article 4 model/data path without creating a training run.

This is deliberately separate from the legacy ``hvp_rev.rhvd`` CLI.  It lets the
Path B campaign compare candidate *experimental instrument* sizes before the
training protocol is frozen, while exercising the real TRB latent data path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_rev import hier_convae
from hvp_rev import rhvd as legacy
from hvp_rev.step1_mech import ActionHierData
from hvp_hca.performance import enable_mixed_precision_rmsnorm


def _sync() -> None:
    if legacy.DEV == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--ae-path", required=True)
    parser.add_argument("--n-ctx-coarse", type=int, default=0)
    parser.add_argument("--n-ctx-fine", type=int, required=True)
    parser.add_argument("--stride-coarse", type=int, default=1)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-profile-steps", type=int, default=25)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fused-rmsnorm", action="store_true")
    parser.add_argument("--world", type=int, default=192)
    parser.add_argument("--speed", type=int, default=4)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if args.width % args.heads:
        parser.error("--width must be divisible by --heads")
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("--steps must be positive and --warmup-steps non-negative")

    legacy.M._BF16 = bool(args.bf16)
    ae = hier_convae.load(args.ae_path)
    encode_started = time.perf_counter()
    data = ActionHierData(
        ae,
        args.train_path,
        str(Path(args.train_path).with_name(Path(args.train_path).stem + "_actions.npy")),
        n_ctx=6,
        k=4,
        W=args.world,
        speed=args.speed,
    )
    encode_seconds = time.perf_counter() - encode_started

    net = legacy.WindowDiT(
        n_ctx_c=args.n_ctx_coarse,
        n_ctx_f=args.n_ctx_fine,
        mode="diff",
        stride_c=args.stride_coarse,
        d=args.width,
        depth=args.depth,
        heads=args.heads,
    ).to(legacy.DEV)
    rmsnorm_replacements = enable_mixed_precision_rmsnorm(net) if args.fused_rmsnorm else 0
    wrapper = legacy.WindowDiffusion(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4)

    _sync()
    batch_started = time.perf_counter()
    for _ in range(args.batch_profile_steps):
        legacy.window_batch(
            data,
            args.batch,
            args.n_ctx_coarse,
            args.n_ctx_fine,
            train=True,
            stride_c=args.stride_coarse,
        )
    _sync()
    batch_seconds = time.perf_counter() - batch_started

    def update() -> None:
        batch = legacy.window_batch(
            data,
            args.batch,
            args.n_ctx_coarse,
            args.n_ctx_fine,
            train=True,
            stride_c=args.stride_coarse,
        )
        loss = wrapper.loss(
            batch["ctx_c"],
            batch["ctx_f"],
            batch["tgt_c"],
            batch["tgt_f"],
            batch["actions"],
            ctx_t=batch.get("ctx_t"),
            tgt_t=batch.get("tgt_t"),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup_steps):
        update()
    _sync()
    if legacy.DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        update()
    _sync()
    elapsed = time.perf_counter() - started

    context_tokens = args.n_ctx_coarse * 16 + args.n_ctx_fine * 64
    result = {
        "device": str(legacy.DEV),
        "bf16": bool(args.bf16),
        "mixed_precision_rmsnorm_replacements": rmsnorm_replacements,
        "context_tokens": context_tokens,
        "n_ctx_coarse": args.n_ctx_coarse,
        "n_ctx_fine": args.n_ctx_fine,
        "stride_coarse": args.stride_coarse,
        "width": args.width,
        "depth": args.depth,
        "heads": args.heads,
        "batch": args.batch,
        "profile_steps": args.steps,
        "encode_seconds": encode_seconds,
        "batch_assembly_seconds_per_batch": batch_seconds / args.batch_profile_steps,
        "seconds_per_update": elapsed / args.steps,
        "estimated_hours_12000_updates": elapsed / args.steps * 12000 / 3600,
        "estimated_hours_30000_updates": elapsed / args.steps * 30000 / 3600,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
        ),
        "parameter_count": sum(parameter.numel() for parameter in net.parameters()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
