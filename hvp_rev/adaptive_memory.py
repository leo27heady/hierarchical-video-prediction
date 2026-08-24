"""Activation- and state-memory accounting for the integration feasibility study.

This is development instrumentation, not frozen article evidence.  It deliberately separates:

* the 1,152 context tokens consumed by the predictor;
* the buffers retained by the current adaptive prototype; and
* the packed state that a release implementation would have to store to claim context-memory
  parity with ``hier-long``.

The CUDA profile unrolls the actual WindowDiT regression predictor through multiple four-frame
windows with ordinary BPTT.  Fixed and selected packages have the same 1,472-token transformer
sequence, so any material difference would indicate implementation overhead rather than a token
budget effect.  Reversible-core measurements are produced by ``hvp_rev.rhcp_validate`` and remain
a separate mechanism result; this script does not imply that WindowDiT itself is reversible.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from hvp_rev import rhvd
from hvp_rev.adaptive_context import (
    CELLS,
    CONTEXT_FRAMES,
    CONTEXT_TOKENS,
    SelectionConfig,
    build_model,
    initialize_from_checkpoint,
    select_spacetime_cells,
)
from hvp_rev.faithful_nets import LAT
import hvp_rev.step1_mech as mechanism


def context_payload_accounting(element_bytes: int = 4) -> dict:
    """Return exact per-sample payload accounting for fixed, prototype and packed states."""
    if element_bytes not in {2, 4, 8}:
        raise ValueError("element_bytes must be 2, 4 or 8")
    coarse_tokens = CONTEXT_FRAMES * 16
    fine_budget = CONTEXT_FRAMES * 8
    dense_fine_tokens = CONTEXT_FRAMES * CELLS
    fixed_tokens = coarse_tokens + 6 * CELLS
    coefficient_bytes = LAT * element_bytes
    # The feasibility rollout keeps full fine latents and float32 frames so that its selector can
    # be recomputed after every predicted window.  This is honest prototype storage, not the
    # intended packed implementation.
    prototype_latent_bytes = (coarse_tokens + dense_fine_tokens) * coefficient_bytes
    prototype_routing_frame_bytes = CONTEXT_FRAMES * 64 * 64 * 4
    # Current tensor interface uses two int64 arrays.  A compact implementation needs at most one
    # byte each for frame and cell ids because both lie in [0,63].
    packed_metadata_int64 = fine_budget * 2 * 8
    packed_metadata_uint8 = fine_budget * 2
    return {
        "latent_channels_per_token": LAT,
        "element_bytes": element_bytes,
        "fixed_hier_long": {
            "context_tokens": fixed_tokens,
            "latent_payload_bytes": fixed_tokens * coefficient_bytes,
        },
        "adaptive_logical_package": {
            "context_tokens": CONTEXT_TOKENS,
            "latent_payload_bytes": CONTEXT_TOKENS * coefficient_bytes,
            "metadata_bytes_current_int64_interface": packed_metadata_int64,
            "metadata_bytes_compact_uint8": packed_metadata_uint8,
        },
        "adaptive_current_rollout_prototype": {
            "retained_latent_cell_slots": coarse_tokens + dense_fine_tokens,
            "latent_payload_bytes": prototype_latent_bytes,
            "routing_frames_float32_bytes": prototype_routing_frame_bytes,
            "total_payload_bytes_excluding_tensor_overhead": (
                prototype_latent_bytes + prototype_routing_frame_bytes
            ),
        },
        "warning": (
            "The current adaptive rollout proves routing/quality feasibility only. A packed causal "
            "buffer is required before claiming an implemented context-state memory saving."
        ),
    }


def _ids(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = SelectionConfig(
        mode="variance", ema_alpha=0.5, layout="recent4",
        dense_recent_frames=4, min_older_cells=2,
    )
    generator = torch.Generator().manual_seed(12345)
    frames = torch.rand((batch, CONTEXT_FRAMES, 64, 64), generator=generator)
    frame_ids, cell_ids = select_spacetime_cells(frames, cfg)
    return frame_ids.to(device), cell_ids.to(device)


def _profile_one(
    variant: str,
    windows: int,
    batch: int,
    checkpoint: Path,
    bf16: bool,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an activation-memory profile")
    if variant not in {"fixed", "adaptive"}:
        raise ValueError("variant must be fixed or adaptive")
    device = torch.device("cuda")
    torch.manual_seed(1234)
    mechanism._BF16 = bf16
    selected = variant == "adaptive"
    net = build_model(selected, layout="recent4", mode="reg")
    initialize_from_checkpoint(net, checkpoint)
    net.train()
    net.zero_grad(set_to_none=True)

    coarse = torch.randn(batch, CONTEXT_FRAMES, LAT, 4, 4, device=device)
    fine_frames = CONTEXT_FRAMES if selected else 6
    fine = torch.randn(batch, fine_frames, LAT, 8, 8, device=device)
    yc = torch.zeros(batch, rhvd.WIN, LAT, 4, 4, device=device)
    yf = torch.zeros(batch, rhvd.WIN, LAT, 8, 8, device=device)
    timestep = torch.zeros(batch, device=device)
    frame_ids, cell_ids = _ids(batch, device) if selected else (None, None)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    loss = torch.zeros((), device=device)
    for step in range(windows):
        actions = torch.randint(-1, 2, (batch, rhvd.WIN), device=device)
        pred_c, pred_f = net(
            yc, yf, coarse, fine, timestep, actions,
            ctx_f_ids=cell_ids, ctx_f_frame_ids=frame_ids,
        )
        loss = loss + pred_c.float().square().mean() + pred_f.float().square().mean()
        coarse = torch.cat([coarse, pred_c], dim=1)[:, -CONTEXT_FRAMES:]
        keep_fine = CONTEXT_FRAMES if selected else 6
        fine = torch.cat([fine, pred_f], dim=1)[:, -keep_fine:]
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    report = {
        "variant": variant,
        "windows": windows,
        "predicted_frames": windows * rhvd.WIN,
        "batch": batch,
        "bf16_autocast": bf16,
        "transformer_sequence_tokens": net.expected_seq,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "seconds": seconds,
        "milliseconds_per_predicted_frame": 1000 * seconds / (windows * rhvd.WIN),
        "loss": float(loss.detach()),
    }
    del net, coarse, fine, yc, yf, timestep, frame_ids, cell_ids, loss
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs_rhcp/rhvd_reg-long_s0_t30000.pt")
    parser.add_argument("--windows", default="1,2,4,8")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--output",
        default="D:/hierarchical-video-prediction-storage/feasibility/memory_profile.json",
    )
    parser.add_argument("--acknowledge-feasibility", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_feasibility:
        raise RuntimeError("pass --acknowledge-feasibility for development-only profiling")
    checkpoint = Path(args.checkpoint)
    horizons = [int(value) for value in args.windows.split(",")]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("--windows must contain positive comma-separated integers")
    runs = []
    for windows in horizons:
        for variant in ("fixed", "adaptive"):
            try:
                result = _profile_one(variant, windows, args.batch, checkpoint, args.bf16)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                result = {
                    "variant": variant,
                    "windows": windows,
                    "predicted_frames": windows * rhvd.WIN,
                    "batch": args.batch,
                    "status": "CUDA_OOM",
                    "error": str(error),
                }
            runs.append(result)
            print(json.dumps(result, indent=2), flush=True)
    allocated = [
        row for row in runs if "incremental_peak_allocated_bytes" in row
    ]
    slopes = {}
    for variant in ("fixed", "adaptive"):
        rows = [row for row in allocated if row["variant"] == variant]
        if len(rows) >= 2:
            x = np.asarray([row["predicted_frames"] for row in rows], dtype=np.float64)
            y = np.asarray([row["incremental_peak_allocated_bytes"] for row in rows], dtype=np.float64)
            slopes[variant] = {
                "linear_slope_bytes_per_predicted_frame": float(np.polyfit(x, y, 1)[0]),
                "interpretation": "empirical local BPTT slope over the profiled horizons",
            }
    report = {
        "schema_version": 1,
        "study": "adaptive-hierarchy-integration-memory-feasibility",
        "evidence_class": "development-feasibility-not-frozen-article-evidence",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checkpoint": str(checkpoint.resolve()),
        "ordinary_bptt": runs,
        "bptt_slopes": slopes,
        "context_state_payload": context_payload_accounting(4),
        "scope": (
            "WindowDiT BPTT and RHCP reverse replay are separately measured mechanisms. "
            "These rows do not establish an end-to-end reversible WindowDiT pipeline."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
