"""Profile the packed OpenSTL regression transition under reversible long rollout."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae
from hvp_rev.fixed_region_revformer import FixedRegionReconstructor
from hvp_rev.packed_simvp import HierarchicalPackedTransition
from hvp_rev.reversible_rollout import ReversibleSecondOrderRollout


def load_simvp(root: Path, checkpoint: Path):
    sys.path.insert(0, str(root))
    source = root / "openstl/models/simvp_model.py"
    spec = importlib.util.spec_from_file_location("simvp_integrated_profile_direct", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SimVP_Model((10, 1, 64, 64), hid_S=64, hid_T=512, N_T=8,
                               N_S=4, model_type="gSTA", drop_path=0).cuda()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded["state_dict"] if "state_dict" in loaded else loaded["model"]
    if any(key.startswith("model.") for key in state):
        state = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    model.load_state_dict(state)
    return model


def reset_peak() -> int:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    return baseline


def build(args):
    ae = hier_convae.load(args.ae_path)
    adapter = FixedRegionReconstructor().cuda()
    adapter.load_state_dict(torch.load(args.adapter_checkpoint, map_location="cuda",
                                       weights_only=False)["model"])
    adapter.eval()
    for module in (ae, adapter):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    simvp = load_simvp(Path(args.openstl), Path(args.simvp_checkpoint))
    transition = HierarchicalPackedTransition(ae, adapter, simvp).cuda()
    # Reverse replay requires the transition to be identical during inverse and
    # backward recomputation.  OpenSTL's gSTA schedule keeps a 0.01 DropPath in
    # its first block even when the top-level drop_path argument is zero, so the
    # profile uses deterministic evaluation mode while retaining gradients.
    transition.eval()
    return transition


def initial_state(transition, test_path: str, batch: int):
    raw = np.load(test_path, mmap_mode="r")
    sequences = raw[:, :batch].transpose(1, 0, 2, 3) if raw.shape[0] == 20 else raw[:batch]
    frames = torch.from_numpy(np.array(sequences[:, :10], dtype=np.float32, copy=True)).cuda().unsqueeze(2) / 255
    with torch.no_grad():
        return transition.pack_frames(frames).detach()


def profile(transition, state: torch.Tensor, chunks: int, reversible: bool):
    transition.zero_grad(set_to_none=True)
    state = state.detach().clone().requires_grad_(True)
    target = torch.zeros(state.shape[0], 10, 1, 64, 64, device="cuda")
    baseline = reset_peak()
    current = state
    if chunks > 1:
        rollout = ReversibleSecondOrderRollout(transition)
        _, current = rollout(current, chunks - 1, reversible=reversible)
    prediction = transition.predict_pixels(current)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / 1e6
    grad = next(parameter.grad for parameter in transition.predictor.parameters()
                if parameter.grad is not None)
    return {
        "mode": "reversible" if reversible else "vanilla_bptt",
        "chunks": chunks,
        "predicted_frames": chunks * 10,
        "incremental_peak_mb": peak,
        "loss": float(loss.detach()),
        "first_parameter_grad_norm": float(grad.norm()),
    }


@torch.no_grad()
def drift(transition, state: torch.Tensor, chunks: int):
    rollout = ReversibleSecondOrderRollout(transition)
    previous = torch.zeros_like(state)
    final = rollout(state, chunks, previous=previous, reversible=False)
    recovered = rollout.inverse(*final, steps=chunks)
    return {
        "chunks": chunks,
        "previous_max_abs_drift": float((recovered[0] - previous).abs().max()),
        "current_max_abs_drift": float((recovered[1] - state).abs().max()),
    }


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    transition = build(args)
    state = initial_state(transition, args.test_path, args.batch)
    rows = []
    for chunks in args.chunks:
        rows.append(profile(transition, state, chunks, True))
        rows.append(profile(transition, state, chunks, False))
    drifts = [drift(transition, state, chunks) for chunks in args.chunks]
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "gradient_checkpointing": False,
        "state_shape": list(state.shape),
        "packed_values_per_frame": transition.values_per_frame,
        "measurement": "max_memory_allocated minus allocation immediately before forward; decimal MB",
        "rows": rows,
        "inverse_drift": drifts,
    }
    print(json.dumps(result, indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-path", default="logs_rhcp/hier_ae_mmnist.pt")
    parser.add_argument("--adapter-checkpoint", default=r"D:\hierarchical-video-prediction-storage\feasibility\fixed_region_pixel_adapter_mmnist_s0.pt")
    parser.add_argument("--openstl", default=r"C:\Users\leoni\Documents\projects\OpenSTL")
    parser.add_argument("--simvp-checkpoint", default=r"D:\hierarchical-video-prediction-storage\feasibility\simvp_packed_pixel_finetune_s0_best.pt")
    parser.add_argument("--test-path", default=r"C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\mnist_test_seq.npy")
    parser.add_argument("--output", default=r"D:\hierarchical-video-prediction-storage\feasibility\integrated_reversible_openstl_memory.json")
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--batch", type=int, default=1)
    main(parser.parse_args())
