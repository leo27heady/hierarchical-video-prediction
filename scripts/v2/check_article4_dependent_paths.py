"""GPU correctness gate for the adaptive A4-LVL and A4-OVR model/data paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.allocation import overlap_1792_plan, three_level_1792_plan
from hvp_hca.data import PackedWindowBatcher, load_trb_development
from hvp_hca.performance import enable_mixed_precision_rmsnorm
from hvp_rev import hier3, hier_convae
from hvp_rev import rhvd as legacy


def exercise(data: object, plan: object, *, width: int, depth: int, heads: int) -> dict:
    batcher = PackedWindowBatcher(
        data, plan, batch_size=4, seed=0, device=legacy.DEV, context_anchor=96
    )
    batch = batcher.sample(split="train", seed=9901)
    torch.manual_seed(71)
    if legacy.DEV == "cuda":
        torch.cuda.manual_seed_all(71)
        torch.cuda.reset_peak_memory_stats()
    net = legacy.WindowDiT(
        n_ctx_t=len(plan.top_ages),
        n_ctx_c=len(plan.coarse_ages),
        n_ctx_f=plan.reach,
        fine_token_budget=plan.fine_tokens,
        mode="diff",
        d=width,
        depth=depth,
        heads=heads,
        temporal_embedding_positions=100,
        temporal_context_span=96,
        top_context_only=bool(plan.top_ages),
    ).to(legacy.DEV)
    replacements = enable_mixed_precision_rmsnorm(net)
    wrapper = legacy.WindowDiffusion(net)
    ema = legacy.EMA(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    for step in (1, 2):
        loss = legacy.training_update(
            wrapper, net, ema, optimizer, batch, step, lr=3e-4, warmup=2000
        )
        losses.append(float(loss.detach()))
    if not all(np.isfinite(losses)):
        raise AssertionError(f"dependent-path loss is non-finite: {losses}")
    return {
        "policy": plan.policy,
        "overlap": plan.overlap,
        "fine_representation": plan.fine_representation,
        "levels": 1 + int(bool(plan.coarse_ages)) + int(bool(plan.top_ages)),
        "context_tokens": plan.token_count,
        "reach": plan.reach,
        "expected_attention_sequence": net.expected_seq,
        "parameters": sum(parameter.numel() for parameter in net.parameters()),
        "mixed_precision_rmsnorm_replacements": replacements,
        "losses": losses,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/v2/manifests/trb_v2_development.manifest.json"),
    )
    parser.add_argument(
        "--base-ae",
        default="C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom.pt",
    )
    parser.add_argument(
        "--top-ae",
        default="C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom_top.pt",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    legacy.M._BF16 = True
    base = hier_convae.load(args.base_ae)
    loaded = load_trb_development(base, args.manifest)
    ae3 = hier3.load_top(args.base_ae, args.top_ae)
    hier3.attach_top(loaded.data, ae3)
    plans = [
        three_level_1792_plan("balanced_recent"),
        overlap_1792_plan("balanced_recent", "exclusive"),
        overlap_1792_plan("balanced_recent", "residual"),
    ]
    results = [exercise(loaded.data, plan, width=256, depth=6, heads=8) for plan in plans]
    if {result["context_tokens"] for result in results} != {1792}:
        raise AssertionError("dependent paths violated equal token budget")
    payload = {"schema_version": 1, "device": legacy.DEV, "passed": True, "paths": results}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
