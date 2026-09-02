"""Paired CUDA-memory profile for one trained integrated-model checkpoint.

All cases reload identical weights.  Hierarchy changes the retained context;
reversible/vanilla changes only internal activation storage; static/dynamic at a
fixed quota is reported as equal-memory allocation unless measurement says
otherwise.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
import torch.nn.functional as F


CASES = (
    ("flat_vanilla", "flat", "dense", False),
    ("hier_static_vanilla", "hierarchical", "static", False),
    ("hier_dynamic_vanilla", "hierarchical", "hybrid", False),
    ("hier_static_reversible", "hierarchical", "static", True),
    ("full_dynamic_reversible", "hierarchical", "hybrid", True),
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config(path: Path, dataset: str) -> dict:
    from openstl.datasets import dataset_parameters
    values = dict(dataset_parameters[dataset])
    values.update({
        key: value for key, value in runpy.run_path(str(path)).items()
        if not key.startswith("__")
    })
    return values


def checkpoint_state(path: Path) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = loaded.get("state_dict", loaded.get("model", loaded))
    if any(key.startswith("model.") for key in state):
        state = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    return state


def synchronize() -> None:
    torch.cuda.synchronize()


def clean_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    synchronize()


def train_iteration(model, inputs, targets, optimizer, precision: str) -> float:
    optimizer.zero_grad(set_to_none=True)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    enabled = precision in {"bf16", "fp16"}
    with torch.autocast("cuda", dtype=dtype, enabled=enabled):
        prediction = model(inputs)
        loss = F.mse_loss(prediction, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def profile_case(model_class, model_config, state, inputs, targets, case, arguments) -> dict:
    name, hierarchy, routing, reversible = case
    model = model_class(**model_config).cuda()
    model.load_state_dict(state)
    model.hierarchy_mode = hierarchy
    model.set_routing_mode(routing)
    model.set_reversible_core(reversible)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(arguments.warmup):
        train_iteration(model, inputs, targets, optimizer, arguments.precision)
    clean_memory()
    peaks_allocated, peaks_reserved, durations, losses = [], [], [], []
    for _ in range(arguments.iterations):
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses.append(train_iteration(model, inputs, targets, optimizer, arguments.precision))
        synchronize()
        durations.append(time.perf_counter() - started)
        peaks_allocated.append((torch.cuda.max_memory_allocated() - baseline_allocated) / 1e6)
        peaks_reserved.append((torch.cuda.max_memory_reserved() - baseline_reserved) / 1e6)
    contract = model.memory_contract(
        batch_size=arguments.batch, dtype_bytes=2 if arguments.precision != "fp32" else 4
    )
    result = {
        "case": name,
        "hierarchy_mode": hierarchy,
        "routing_mode": routing,
        "reversible_core": reversible,
        "memory_contract": contract,
        "incremental_peak_allocated_mb": max(peaks_allocated),
        "incremental_peak_reserved_mb": max(peaks_reserved),
        "mean_seconds_per_step": sum(durations) / len(durations),
        "mean_loss": sum(losses) / len(losses),
        "iterations": arguments.iterations,
        "warmup": arguments.warmup,
    }
    del optimizer, model
    clean_memory()
    return result


def profile_core(model, inputs, routing: str, reversible: bool, arguments) -> dict:
    model.eval()
    with torch.no_grad():
        packed = model.encode_and_pack(inputs, routing)
    tokens = packed.tokens.detach()
    durations, peaks = [], []
    for _ in range(arguments.warmup + arguments.iterations):
        model.core.zero_grad(set_to_none=True)
        value = tokens.clone().requires_grad_(True)
        synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        dtype = torch.bfloat16 if arguments.precision == "bf16" else torch.float16
        enabled = arguments.precision in {"bf16", "fp16"}
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            output = model.core(value, reversible=reversible)
            loss = output.square().mean()
        loss.backward()
        synchronize()
        if _ >= arguments.warmup:
            durations.append(time.perf_counter() - started)
            peaks.append((torch.cuda.max_memory_allocated() - baseline) / 1e6)
    return {
        "mode": "reversible" if reversible else "vanilla",
        "tokens": tokens.shape[1],
        "incremental_peak_allocated_mb": max(peaks),
        "mean_seconds": sum(durations) / len(durations),
    }


def main(arguments) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")
    sys.path.insert(0, str(arguments.openstl.resolve()))
    from openstl.models.hvp_integrated_model import HVPIntegrated_Model
    model_config = config(arguments.config.resolve(), arguments.dataset)
    state = checkpoint_state(arguments.checkpoint.resolve())
    shape = tuple(model_config["in_shape"])
    generator = torch.Generator(device="cuda").manual_seed(arguments.seed)
    inputs = torch.rand((arguments.batch, *shape), generator=generator, device="cuda")
    targets = torch.rand(
        (arguments.batch, int(model_config["aft_seq_length"]), shape[1], shape[2], shape[3]),
        generator=generator, device="cuda",
    )
    rows = []
    for case in CASES:
        try:
            row = profile_case(
                HVPIntegrated_Model, model_config, state, inputs, targets, case, arguments
            )
        except torch.cuda.OutOfMemoryError as error:
            clean_memory()
            row = {"case": case[0], "oom": True, "error": repr(error)}
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    parity_model = HVPIntegrated_Model(**model_config).cuda().eval()
    parity_model.load_state_dict(state)
    parity_model.hierarchy_mode = "hierarchical"
    parity_model.set_routing_mode("hybrid")
    first = inputs[:1].detach().clone().requires_grad_(True)
    second = first.detach().clone().requires_grad_(True)
    output_reversible = parity_model(first, reversible_core=True)
    output_vanilla = parity_model(second, reversible_core=False)
    output_reversible.square().mean().backward()
    reversible_gradient = first.grad.detach().clone()
    parity_model.zero_grad(set_to_none=True)
    output_vanilla.square().mean().backward()
    parity = {
        "output_max_abs": float((output_reversible - output_vanilla).abs().max()),
        "input_gradient_max_abs": float((reversible_gradient - second.grad).abs().max()),
    }
    core_rows = [
        profile_core(parity_model, inputs[:1], "hybrid", False, arguments),
        profile_core(parity_model, inputs[:1], "hybrid", True, arguments),
    ]
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dataset": arguments.dataset,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": sha256(arguments.checkpoint.resolve()),
        "precision": arguments.precision,
        "batch": arguments.batch,
        "measurement": "peak CUDA allocated/reserved minus immediately pre-forward baseline; decimal MB",
        "cases": rows,
        "forward_gradient_parity": parity,
        "core_only": core_rows,
    }
    atomic_json(arguments.output.resolve(), result)
    print(json.dumps(result, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--openstl", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--dataset", choices=("mmnist", "taxibj"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    result.add_argument("--seed", type=int, default=260902)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
