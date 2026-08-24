"""Matched CUDA training-memory profiles for the feasibility architecture and OpenSTL.

Gradient checkpointing is not used.  The reversible and vanilla rows use identical
weights and arithmetic; only activation retention differs.  OpenSTL models are
loaded read-only from the sibling checkout through direct module loading so its
dirty worktree is never imported as a package or modified.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hvp_rev import hier_convae
from hvp_rev.faithful_nets import LAT
from hvp_rev.fixed_region_revformer import FixedRegionRevPredictor, count_context_tokens


def direct_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reset_cuda() -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def finish_peak(baseline: int) -> float:
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 1e6


def profile_revformer(ae_path: str, steps: int, batch: int, reversible: bool,
                      dim: int, depth: int) -> dict:
    device = torch.device("cuda")
    ae = hier_convae.load(ae_path)
    model = FixedRegionRevPredictor(dim=dim, heads=4, depth=depth,
                                    max_frames=10, max_steps=max(steps, 1)).to(device)
    zc = torch.randn(batch, 10, LAT, 4, 4, device=device)
    zf = torch.randn(batch, 10, LAT, 8, 8, device=device)
    target = torch.randn(batch, 10, 1, 64, 64, device=device).clamp(-1, 1)
    baseline = reset_cuda()
    pred_c, _, _ = model(zc, zf, steps=steps, reversible=reversible)
    pred = ae.decode_coarse(pred_c.reshape(-1, LAT, 4, 4)).reshape(batch, 10, 1, 64, 64)
    loss = F.mse_loss(pred, target)
    loss.backward()
    peak = finish_peak(baseline)
    result = {
        "model": "fixed-region-revformer" if reversible else "fixed-region-vanilla-bptt",
        "batch": batch, "frames": 10, "temporal_steps": steps,
        "incremental_peak_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_tokens": count_context_tokens(10),
    }
    del model, ae, zc, zf, target, pred_c, pred, loss
    reset_cuda()
    return result


def profile_simvp(openstl: Path, batch: int, model_type: str = "gSTA") -> dict:
    sys.path.insert(0, str(openstl))
    module = direct_module("openstl_simvp_direct", openstl / "openstl/models/simvp_model.py")
    model = module.SimVP_Model((10, 1, 64, 64), hid_S=64, hid_T=512, N_T=8,
                               N_S=4, model_type=model_type, drop_path=0).cuda()
    x = torch.randn(batch, 10, 1, 64, 64, device="cuda")
    target = torch.randn_like(x)
    baseline = reset_cuda()
    pred = model(x)
    loss = F.mse_loss(pred, target)
    loss.backward()
    peak = finish_peak(baseline)
    result = {
        "model": f"OpenSTL-SimVP-{model_type}",
        "batch": batch, "frames": 10, "temporal_steps": 10,
        "incremental_peak_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model, x, target, pred, loss
    reset_cuda()
    return result


def patchify(frames: torch.Tensor, patch: int = 4) -> torch.Tensor:
    b, t, c, h, w = frames.shape
    return frames.reshape(b, t, c, h // patch, patch, w // patch, patch).permute(
        0, 1, 3, 5, 2, 4, 6
    ).reshape(b, t, h // patch, w // patch, c * patch * patch)


def profile_convlstm(openstl: Path, batch: int) -> dict:
    sys.path.insert(0, str(openstl))
    module = direct_module("openstl_convlstm_direct", openstl / "openstl/models/convlstm_model.py")
    config = SimpleNamespace(
        in_shape=(10, 1, 64, 64), patch_size=4, filter_size=5, stride=1,
        layer_norm=0, pre_seq_length=10, aft_seq_length=10, reverse_scheduled_sampling=0,
    )
    model = module.ConvLSTM_Model(4, [128] * 4, config).cuda()
    raw = torch.randn(batch, 20, 1, 64, 64, device="cuda")
    frames = patchify(raw)
    mask = torch.zeros(batch, 9, 16, 16, 16, device="cuda")
    baseline = reset_cuda()
    pred, loss = model(frames, mask, return_loss=True)
    loss.backward()
    peak = finish_peak(baseline)
    result = {
        "model": "OpenSTL-ConvLSTM",
        "batch": batch, "frames": 20, "temporal_steps": 19,
        "incremental_peak_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model, raw, frames, mask, pred, loss
    reset_cuda()
    return result


def profile_e3dlstm(openstl: Path, batch: int) -> dict:
    sys.path.insert(0, str(openstl))
    module = direct_module("openstl_e3dlstm_direct", openstl / "openstl/models/e3dlstm_model.py")
    config = SimpleNamespace(
        in_shape=(10, 1, 64, 64), patch_size=4, filter_size=5, stride=1,
        layer_norm=0, pre_seq_length=10, aft_seq_length=10, reverse_scheduled_sampling=0,
    )
    model = module.E3DLSTM_Model(4, [128] * 4, config).cuda()
    raw = torch.randn(batch, 20, 1, 64, 64, device="cuda")
    frames = patchify(raw)
    mask = torch.zeros(batch, 9, 16, 16, 16, device="cuda")
    baseline = reset_cuda()
    pred, loss = model(frames, mask, return_loss=True)
    loss.backward()
    peak = finish_peak(baseline)
    result = {
        "model": "OpenSTL-E3DLSTM",
        "batch": batch, "frames": 20, "temporal_steps": 19,
        "incremental_peak_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model, raw, frames, mask, pred, loss
    reset_cuda()
    return result


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")
    rows = []
    for steps in args.steps:
        rows.append(profile_revformer(args.ae_path, steps, args.batch, True, args.dim, args.depth))
        rows.append(profile_revformer(args.ae_path, steps, args.batch, False, args.dim, args.depth))
    openstl = Path(args.openstl)
    rows.append(profile_simvp(openstl, args.batch, "gSTA"))
    rows.append(profile_simvp(openstl, args.batch, "incepu"))
    rows.append(profile_convlstm(openstl, args.batch))
    rows.append(profile_e3dlstm(openstl, args.batch))
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "measurement": "max_memory_allocated minus allocation immediately before forward; decimal MB",
        "gradient_checkpointing": False,
        "rows": rows,
    }
    print(json.dumps(result, indent=2), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-path", default="logs_rhcp/hier_ae_mmnist.pt")
    parser.add_argument("--openstl", default=r"C:\Users\leoni\Documents\projects\OpenSTL")
    parser.add_argument("--output", default=r"D:\hierarchical-video-prediction-storage\feasibility\fixed_region_memory_profile.json")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 10, 16, 32])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    main(parser.parse_args())
