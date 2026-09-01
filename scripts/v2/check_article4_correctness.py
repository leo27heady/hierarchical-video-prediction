"""Run the executable A4-COR data, leakage, replay, and tiny-overfit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.allocation import frontier_plan
from hvp_hca.data import PackedWindowBatcher, load_manifest, load_trb_development, sha256_file
from hvp_hca.performance import enable_mixed_precision_rmsnorm
from hvp_rev import hier_convae
from hvp_rev import rhvd as legacy


def audit_action_alignment(manifest: dict[str, Any], *, samples: int = 4096) -> dict[str, Any]:
    clips = np.load(manifest["artifacts"]["clips"]["path"], mmap_mode="r")
    actions = np.load(manifest["artifacts"]["actions"]["path"], mmap_mode="r")
    speed = int(manifest["configuration"]["speed"])
    vocabulary = set(int(value) for value in np.unique(actions))
    expected_vocabulary = set(int(value) for value in manifest["configuration"]["action_vocabulary"])
    if vocabulary != expected_vocabulary:
        raise AssertionError(f"action vocabulary {vocabulary} != {expected_vocabulary}")
    rng = np.random.default_rng(260_901)
    checked_pixels = mismatched_pixels = 0
    for _ in range(samples):
        clip = int(rng.integers(clips.shape[0]))
        frame = int(rng.integers(1, clips.shape[1]))
        action = int(actions[clip, frame])
        previous, current = clips[clip, frame - 1], clips[clip, frame]
        if action > 0:
            left, right = current[:, :-speed], previous[:, speed:]
        elif action < 0:
            left, right = current[:, speed:], previous[:, :-speed]
        else:
            left, right = current, previous
        checked_pixels += left.size
        mismatched_pixels += int(np.count_nonzero(left != right))
    if mismatched_pixels:
        raise AssertionError(
            f"{mismatched_pixels}/{checked_pixels} overlapping pixels violate action alignment"
        )
    return {
        "sampled_transitions": samples,
        "checked_overlap_pixels": checked_pixels,
        "mismatched_overlap_pixels": mismatched_pixels,
        "action_vocabulary": sorted(vocabulary),
    }


def seeded_loss(wrapper: legacy.WindowDiffusion, batch: dict[str, Any], seed: int) -> float:
    devices = [torch.cuda.current_device()] if legacy.DEV == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if legacy.DEV == "cuda":
            torch.cuda.manual_seed_all(seed)
        loss = wrapper.loss(
            batch["ctx_c"], batch["ctx_f"], batch["tgt_c"], batch["tgt_f"],
            batch["actions"], ctx_c_frame_ids=batch["ctx_c_frame_ids"],
            ctx_f_ids=batch["ctx_f_ids"], ctx_f_frame_ids=batch["ctx_f_frame_ids"],
            ctx_f_packed=batch["ctx_f_packed"],
        )
    return float(loss.detach())


def tiny_overfit(data: Any, policy: str, *, updates: int) -> dict[str, Any]:
    plan = frontier_plan(1792, policy)
    batcher = PackedWindowBatcher(
        data, plan, batch_size=4, seed=0, device=legacy.DEV, context_anchor=96
    )
    batch = batcher.sample(split="train", seed=4400)
    torch.manual_seed(700)
    if legacy.DEV == "cuda":
        torch.cuda.manual_seed_all(700)
    net = legacy.WindowDiT(
        n_ctx_c=len(plan.coarse_ages), n_ctx_f=plan.reach,
        fine_token_budget=plan.fine_tokens, mode="diff", d=64, depth=2, heads=4,
        temporal_embedding_positions=100,
        temporal_context_span=96,
    ).to(legacy.DEV)
    enable_mixed_precision_rmsnorm(net)
    wrapper = legacy.WindowDiffusion(net)
    ema = legacy.EMA(net, decay=0.99)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    evaluation_seeds = (901, 902, 903, 904)

    replay_first = seeded_loss(wrapper, batch, evaluation_seeds[0])
    replay_second = seeded_loss(wrapper, batch, evaluation_seeds[0])
    if replay_first != replay_second:
        raise AssertionError(f"deterministic loss replay failed: {replay_first} != {replay_second}")
    curve = [{"step": 0, "mean_fixed_noise_loss": float(np.mean([
        seeded_loss(wrapper, batch, seed) for seed in evaluation_seeds
    ]))}]
    net.train()
    for step in range(1, updates + 1):
        legacy.training_update(
            wrapper, net, ema, optimizer, batch, step, lr=1e-3, warmup=20
        )
        if step % 50 == 0 or step == updates:
            net.eval()
            curve.append({"step": step, "mean_fixed_noise_loss": float(np.mean([
                seeded_loss(wrapper, batch, seed) for seed in evaluation_seeds
            ]))})
            net.train()
    ratio = curve[-1]["mean_fixed_noise_loss"] / curve[0]["mean_fixed_noise_loss"]
    if not np.isfinite(ratio) or ratio >= 0.90:
        raise AssertionError(f"{policy} tiny overfit failed: final/initial loss={ratio:.4f}")
    return {
        "policy": policy,
        "updates": updates,
        "parameters": sum(parameter.numel() for parameter in net.parameters()),
        "expected_attention_sequence": net.expected_seq,
        "deterministic_replay_loss": replay_first,
        "curve": curve,
        "final_to_initial_loss_ratio": ratio,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("experiments/v2/manifests/trb_v2_development.manifest.json"),
    )
    parser.add_argument(
        "--ae", type=Path,
        default=Path("C:/Users/leoni/Documents/projects/hierarchical-video-prediction/logs_rhcp/hier_ae_mom.pt"),
    )
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.updates < 50:
        parser.error("--updates must be at least 50")

    legacy.M._BF16 = True
    manifest, manifest_hash = load_manifest(args.manifest)
    result: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "autoencoder_sha256": sha256_file(args.ae),
        "device": legacy.DEV,
        "action_alignment": audit_action_alignment(manifest),
    }
    ae = hier_convae.load(str(args.ae))
    loaded = load_trb_development(ae, args.manifest)
    result["split"] = {
        "train_clips": len(loaded.data.train_idx),
        "validation_clips": len(loaded.data.val_idx),
        "overlap": int(np.intersect1d(loaded.data.train_idx, loaded.data.val_idx).size),
    }
    result["tiny_overfit"] = [
        tiny_overfit(loaded.data, "flat_recent", updates=args.updates),
        tiny_overfit(loaded.data, "balanced_recent", updates=args.updates),
    ]
    if len({item["parameters"] for item in result["tiny_overfit"]}) != 1:
        raise AssertionError("flat and hierarchy tiny-overfit models have unequal capacity")
    result["passed"] = True
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
