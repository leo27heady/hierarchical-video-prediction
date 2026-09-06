"""Fixed-weight synthetic CUDA benchmark. Version 2; not a quality experiment.

AdamW lr=0 includes optimizer work without changing checkpoint weights. Each
repeat starts with empty optimizer state; training peaks include its allocation.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time
import torch
import torch.nn.functional as F

MODES = ("eager", "checkpoint", "reversible")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def autocast(precision, device="cuda"):
    return torch.autocast(device, dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                          enabled=precision != "fp32")


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def tensor_bytes(tensors, unique=False):
    seen, total = set(), 0
    for value in tensors:
        if value is None:
            continue
        if unique:
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
        else:
            total += value.numel() * value.element_size()
    return total


def set_mode(model, mode):
    model.set_reversible_core(mode == "reversible")
    model.core.activation_checkpointing = mode == "checkpoint"


def saved_tensor_accounting(model, inputs, precision):
    """Diagnostic forward only, outside timing; exclude parameter/input storage."""
    excluded = {p.untyped_storage().data_ptr() for p in model.parameters()}
    excluded.add(inputs.untyped_storage().data_ptr())
    seen, logical = {}, 0
    def pack(value):
        nonlocal logical
        storage = value.untyped_storage()
        if storage.data_ptr() not in excluded:
            logical += value.numel() * value.element_size()
            seen[storage.data_ptr()] = storage.nbytes()
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        with torch.enable_grad(), autocast(precision, inputs.device.type):
            output = model(inputs)
    del output
    return {"forward_saved_logical_bytes": logical, "forward_saved_unique_storage_bytes": sum(seen.values()),
            "excludes": "original parameters and input storage; includes nonparameter saved tensors and casts",
            "scope": "forward autograd save requests, not allocator peak or backward recomputation peak"}


def packed_accounting(packed):
    tensors = [packed.tokens, packed.coarse_grid, packed.route_indices, packed.route_scores]
    tensors += [v for v in vars(packed.metadata).values() if isinstance(v, torch.Tensor)]
    tensors += [v for v in packed.diagnostics.values() if isinstance(v, torch.Tensor)]
    return {"logical_bytes": tensor_bytes(tensors), "unique_storage_bytes": tensor_bytes(tensors, True),
            "tokens_dtype": str(packed.tokens.dtype), "coarse_grid_dtype": str(packed.coarse_grid.dtype),
            "includes": "tokens, coarse bypass, metadata, route indices/scores, diagnostics"}


def measure(module, forward, inputs, targets, args, training):
    parameters = list(module.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.0)
    rows = []
    for repeat in range(args.warmup + args.iterations):
        module.zero_grad(set_to_none=True)
        if inputs.is_leaf:
            inputs.grad = None
        optimizer.state.clear()
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        clean_memory()
        base_a, base_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.set_grad_enabled(training), autocast(args.precision):
            output = forward(inputs)
            loss = F.mse_loss(output.float(), targets.float())
        if training:
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                raise RuntimeError("FP16 overflow skipped optimizer work; this is not a valid profile repeat")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite profile loss")
        row = {"seconds": elapsed, "loss": float(loss.detach()),
               "absolute_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "absolute_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "baseline_allocated_bytes": base_a, "baseline_reserved_bytes": base_r,
               "gradient_bytes": tensor_bytes(p.grad for p in parameters),
               "optimizer_bytes": tensor_bytes(v for state in optimizer.state.values()
                                                for v in state.values() if isinstance(v, torch.Tensor))}
        row["incremental_peak_allocated_bytes"] = row["absolute_peak_allocated_bytes"] - base_a
        row["incremental_peak_reserved_bytes"] = row["absolute_peak_reserved_bytes"] - base_r
        if repeat >= args.warmup:
            rows.append(row)
        del output, loss
    module.zero_grad(set_to_none=True)
    del optimizer
    return {"training": training, "batch": inputs.shape[0], "input_dtype": str(inputs.dtype),
            "parameter_bytes": tensor_bytes(parameters), "repeats": rows,
            "median_seconds": statistics.median(r["seconds"] for r in rows),
            "min_seconds": min(r["seconds"] for r in rows),
            "max_seconds": max(r["seconds"] for r in rows)}


def parity(model, inputs, targets, precision):
    records = {}
    for mode in MODES:
        set_mode(model, mode)
        model.zero_grad(set_to_none=True)
        value = inputs.detach().clone().requires_grad_(True)
        with autocast(precision, inputs.device.type):
            prediction = model(value)
            loss = F.mse_loss(prediction.float(), targets.float())
        loss.backward()
        records[mode] = (prediction.detach().cpu().clone(), value.grad.detach().cpu().clone(),
                         {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                          if p.grad is not None})
        del prediction, value, loss
    atol, rtol = ((3e-5, 4e-4) if precision == "fp32" else (1e-2, 8e-2))
    reference = records["eager"]
    for mode in MODES[1:]:
        current = records[mode]
        torch.testing.assert_close(current[0], reference[0], atol=atol, rtol=rtol)
        torch.testing.assert_close(current[1], reference[1], atol=atol, rtol=rtol)
        assert current[2].keys() == reference[2].keys()
        for name, grad in current[2].items():
            assert torch.isfinite(grad).all(), name
            torch.testing.assert_close(grad, reference[2][name], atol=atol, rtol=rtol, msg=name)
    for prefix in ("encoder.", "router.", "core."):
        magnitude = sum(float(g.abs().sum()) for n, g in reference[2].items() if n.startswith(prefix))
        if magnitude <= 1e-10:
            raise RuntimeError(f"degenerate parity probe: no meaningful {prefix} gradient")
    model.zero_grad(set_to_none=True)
    return {"passed": True, "modes": MODES, "precision": precision, "batch": inputs.shape[0],
            "atol": atol, "rtol": rtol, "compared": "outputs, input and active named parameter gradients"}


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")
    if min(args.batch, args.iterations) < 1 or args.warmup < 0:
        raise ValueError("positive batch/iterations and nonnegative warmup required")
    sys.path.insert(0, str(args.openstl.resolve()))
    from openstl.models.hvp_integrated_model import HVPIntegrated_Model
    from openstl.datasets import dataset_parameters
    values = dict(dataset_parameters[args.dataset])
    values.update({k: v for k, v in runpy.run_path(str(args.config)).items() if not k.startswith("__")})
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = loaded.get("state_dict", loaded.get("model", loaded))
    if any(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    torch.manual_seed(args.seed)
    shape = tuple(values["in_shape"])
    inputs = torch.rand((args.batch, *shape), device="cuda")
    targets = torch.rand((args.batch, int(values["aft_seq_length"]), *shape[1:]), device="cuda")
    model = HVPIntegrated_Model(**values).cuda().eval()
    model.load_state_dict(state)
    model.hierarchy_mode = "hierarchical"
    model.set_routing_mode("hybrid")
    parity_result = parity(model, inputs, targets, args.precision)
    del model
    clean_memory()
    rows, core_rows = [], []
    for hierarchy, route in (("flat", "dense"), ("hierarchical", "static"), ("hierarchical", "hybrid")):
        for mode in MODES:
            model = HVPIntegrated_Model(**values).cuda().eval()
            model.load_state_dict(state)
            model.hierarchy_mode = hierarchy
            model.set_routing_mode(route)
            set_mode(model, mode)
            with torch.no_grad(), autocast(args.precision):
                packed = model.encode_and_pack(inputs, route)
            accounting = packed_accounting(packed)
            del packed
            for training in (False, True):
                model.train(training)
                saved = saved_tensor_accounting(model, inputs, args.precision) if training else None
                row = measure(model, model, inputs, targets, args, training)
                row.update(hierarchy=hierarchy, routing=route, execution_mode=mode,
                           scope="whole_model", packed_state=accounting, saved_tensors=saved)
                rows.append(row)
            if hierarchy == "hierarchical" and route == "hybrid":
                model.eval()
                with torch.no_grad(), autocast(args.precision):
                    packed = model.encode_and_pack(inputs, route)
                tokens = packed.tokens.detach().clone().requires_grad_(True)
                del packed
                for training in (False, True):
                    row = measure(model.core, lambda x: model.core(x, reversible=mode == "reversible"),
                                  tokens, torch.zeros_like(tokens), args, training)
                    row.update(scope="core_operation_with_whole_model_resident", execution_mode=mode,
                               tokens=tokens.shape[1])
                    core_rows.append(row)
                del tokens
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), state[name], atol=0, rtol=0, msg=name)
            del model
            clean_memory()
    atomic_json(args.output.resolve(), {
        "schema_version": 2, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint), "profiler_sha256": sha256(Path(__file__)),
        "dataset": args.dataset, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "precision": args.precision, "batch": args.batch, "warmup": args.warmup,
        "measurement": "bytes; fixed weights; synthetic data; cold AdamW state each repeat; lr=0",
        "saved_activations": "included in training peaks; not separately attributed by allocator",
        "parity": parity_result, "cases": rows, "core_only": core_rows})


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("openstl", "config", "checkpoint", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--dataset", choices=("mmnist", "taxibj"), required=True)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    result.add_argument("--seed", type=int, default=260902)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
