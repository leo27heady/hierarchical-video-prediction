"""Adapt the frozen OpenSTL SimVP checkpoint to the 24-token hierarchical input boundary."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae
from hvp_rev.fixed_region_revformer import FixedRegionReconstructor


def load_simvp(root: Path, checkpoint: Path, device):
    sys.path.insert(0, str(root))
    source = root / "openstl/models/simvp_model.py"
    spec = importlib.util.spec_from_file_location("simvp_packed_finetune_direct", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SimVP_Model((10, 1, 64, 64), hid_S=64, hid_T=512, N_T=8,
                               N_S=4, model_type="gSTA", drop_path=0).to(device)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data["state_dict"] if "state_dict" in checkpoint_data else checkpoint_data["model"]
    if any(key.startswith("model.") for key in state):
        state = {key[6:]: value for key, value in state.items() if key.startswith("model.")}
    model.load_state_dict(state)
    return model


def normalize(raw: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.array(raw, dtype=np.float32, copy=True)).to(device).unsqueeze(2) / 255.0


@torch.no_grad()
def reconstruct_context(ae, adapter, frames01: torch.Tensor) -> torch.Tensor:
    b, t = frames01.shape[:2]
    zc, zf = ae.encode((frames01 * 2 - 1).reshape(b * t, 1, 64, 64))
    zc = zc.reshape(b, t, 16, 4, 4)
    zf = zf.reshape(b, t, 16, 8, 8)
    predicted_zf, regions, _ = adapter(zc, zf)
    coarse = ae.decode_coarse(zc.reshape(b * t, 16, 4, 4))
    reconstructed = coarse.reshape(b, t, 1, 64, 64)
    reconstructed = reconstructed + adapter.pixel_blend * adapter.pixel_residual(regions)
    return ((reconstructed.clamp(-1, 1) + 1) * 0.5).contiguous()


@torch.no_grad()
def evaluate(model, ae, adapter, sequences: np.ndarray, device, batch: int,
             limit: int | None = None) -> dict:
    model.eval()
    n = len(sequences) if limit is None else min(limit, len(sequences))
    squared = absolute = 0.0
    frames = 0
    for begin in range(0, n, batch):
        sequence = normalize(sequences[begin:min(begin + batch, n)], device)
        context = reconstruct_context(ae, adapter, sequence[:, :10])
        target = sequence[:, 10:20]
        prediction = model(context).clamp(0, 1)
        squared += float((prediction - target).square().sum())
        absolute += float((prediction - target).abs().sum())
        frames += prediction.shape[0] * prediction.shape[1]
    return {"mse": squared / frames, "mae": absolute / frames, "clips": n}


def atomic_checkpoint(path: Path, model, optimizer, update: int, best: float, args, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "update": update, "best_mse": best, "args": vars(args), "history": history,
    }, temporary)
    os.replace(temporary, path)


def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.train_path, mmap_mode="r")
    split = len(data) - args.dev_clips
    train_indices = np.arange(split)
    development = data[split:]
    raw_test = np.load(args.test_path, mmap_mode="r")
    test = raw_test.transpose(1, 0, 2, 3) if raw_test.shape[0] == 20 else raw_test

    ae = hier_convae.load(args.ae_path)
    adapter = FixedRegionReconstructor(args.temperature).to(device)
    adapter_state = torch.load(args.adapter_checkpoint, map_location=device, weights_only=False)["model"]
    adapter.load_state_dict(adapter_state)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    model = load_simvp(Path(args.openstl), Path(args.simvp_checkpoint), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates, eta_min=args.lr * 0.05,
    )
    rng = np.random.RandomState(args.seed + 73)
    output = Path(args.output)
    best_output = output.with_name(output.stem + "_best.pt")
    best = float("inf")
    history = []
    started = time.time()
    initial = evaluate(model, ae, adapter, development, device, args.eval_batch,
                       args.eval_clips)
    history.append({"update": 0, **initial, "elapsed_hours": 0.0})
    best = initial["mse"]
    print("development " + json.dumps(history[-1]), flush=True)
    atomic_checkpoint(best_output, model, optimizer, 0, best, args, history)
    running = 0.0
    for update in range(1, args.updates + 1):
        chosen = rng.choice(train_indices, size=args.batch, replace=False)
        sequence = normalize(data[chosen], device)
        context = reconstruct_context(ae, adapter, sequence[:, :10])
        target = sequence[:, 10:20]
        model.train()
        prediction = model(context)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())
        if update % args.log_every == 0:
            print(f"update={update} loss={running / args.log_every:.7f}", flush=True)
            running = 0.0
        if update % args.eval_every == 0 or update == args.updates:
            metrics = evaluate(model, ae, adapter, development, device, args.eval_batch,
                               args.eval_clips)
            record = {"update": update, **metrics, "elapsed_hours": (time.time() - started) / 3600}
            history.append(record)
            print("development " + json.dumps(record), flush=True)
            if metrics["mse"] < best:
                best = metrics["mse"]
                atomic_checkpoint(best_output, model, optimizer, update, best, args, history)
            atomic_checkpoint(output, model, optimizer, update, best, args, history)

    best_checkpoint = torch.load(best_output, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(model, ae, adapter, test, device, args.eval_batch,
                            args.test_clips or None)
    result = {
        **test_metrics,
        "selected_update": best_checkpoint["update"],
        "selection_population": (
            "last 500 generated training clips; canonical test evaluated only after selection "
            "within this fine-tuning run (public development benchmark, not protected evidence)"
        ),
        "packed_values_per_frame": 384,
        "raw_values_per_frame": 4096,
        "raw_to_packed_value_ratio": 4096 / 384,
        "official_convlstm_s_200ep_mse": 29.80,
        "local_raw_simvp_mse": 26.688837,
        "passes_convlstm_gate": test_metrics["mse"] <= 29.80,
        "passes_raw_simvp_gate": test_metrics["mse"] <= 26.688837,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/mmnist_bridge_train_10000x20.npy")
    parser.add_argument("--test-path", default=r"C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\mnist_test_seq.npy")
    parser.add_argument("--ae-path", default="logs_rhcp/hier_ae_mmnist.pt")
    parser.add_argument("--adapter-checkpoint", default=r"D:\hierarchical-video-prediction-storage\feasibility\fixed_region_pixel_adapter_mmnist_s0.pt")
    parser.add_argument("--openstl", default=r"C:\Users\leoni\Documents\projects\OpenSTL")
    parser.add_argument("--simvp-checkpoint", default=r"C:\Users\leoni\Documents\projects\OpenSTL\work_dirs\mmnist_simvp_gsta_anchor\checkpoints\best.ckpt")
    parser.add_argument("--output", default=r"D:\hierarchical-video-prediction-storage\feasibility\simvp_packed_finetune_s0.pt")
    parser.add_argument("--metrics-out", default=r"D:\hierarchical-video-prediction-storage\feasibility\simvp_packed_finetune_s0_metrics.json")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-clips", type=int, default=500)
    parser.add_argument("--dev-clips", type=int, default=500)
    parser.add_argument("--test-clips", type=int, default=0,
                        help="0 evaluates all canonical test clips")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
