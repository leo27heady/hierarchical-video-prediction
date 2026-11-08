"""Train/evaluate the fixed-region reversible feasibility model on Moving-MNIST.

The canonical comparison uses the public OpenSTL 10->10 test sequence.  Its MSE
definition is reproduced exactly: mean over sample/time and sum over C/H/W.
This script never modifies the sibling OpenSTL checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Release-local path defaults; CLI flags remain available for every path.
REPO_ROOT = Path(__file__).resolve().parents[1]
OPENSTL_ROOT = Path(os.environ.get("HVP_OPENSTL_ROOT", str(REPO_ROOT.parent / "OpenSTL")))
STORAGE_ROOT = Path(os.environ.get("HVP_STORAGE_ROOT", str(REPO_ROOT / "hierarchical-video-prediction-storage"))) / "feasibility"

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae
from hvp_rev.fixed_region_revformer import (
    FixedRegionRevPredictor,
    count_context_tokens,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequences(path: str, expected_frames: int = 20) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"expected a four-dimensional Moving-MNIST array, got {data.shape}")
    if data.shape[0] == expected_frames and data.shape[1] != expected_frames:
        data = data.transpose(1, 0, 2, 3)
    if data.shape[1] < expected_frames or data.shape[-2:] != (64, 64):
        raise ValueError(f"expected (N,>={expected_frames},64,64), got {data.shape}")
    return data


def frames_to_tensor(raw: np.ndarray, device: torch.device) -> torch.Tensor:
    copied = np.array(raw, dtype=np.float32, copy=True)
    return torch.from_numpy(copied).to(device).unsqueeze(2) / 127.5 - 1.0


def encode(ae, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, t = frames.shape[:2]
    with torch.no_grad():
        zc, zf = ae.encode(frames.reshape(b * t, 1, 64, 64))
    return zc.reshape(b, t, *zc.shape[1:]), zf.reshape(b, t, *zf.shape[1:])


def decode(ae, latent: torch.Tensor, level: str = "coarse") -> torch.Tensor:
    b, t = latent.shape[:2]
    flat = latent.reshape(b * t, *latent.shape[2:])
    decoded = ae.decode_coarse(flat) if level == "coarse" else ae.decode_fine(flat)
    return decoded.reshape(b, t, 1, 64, 64)


def sample_batch(data: np.ndarray, indices: np.ndarray, batch: int, context: int,
                 shift: int, rng: np.random.RandomState, device: torch.device):
    chosen = rng.choice(indices, size=batch, replace=len(indices) < batch)
    max_start = data.shape[1] - context - shift
    start = int(rng.randint(max_start + 1)) if max_start else 0
    x = frames_to_tensor(data[chosen, start:start + context], device)
    y = frames_to_tensor(data[chosen, start + shift:start + shift + context], device)
    return x, y


@torch.no_grad()
def evaluate(model, ae, data: np.ndarray, device: torch.device, context: int = 10,
             shift: int = 10, batch: int = 16, limit: int | None = None) -> dict[str, float]:
    model.eval()
    n = len(data) if limit is None else min(limit, len(data))
    total_sq = 0.0
    total_abs = 0.0
    total_frames = 0
    for begin in range(0, n, batch):
        raw = data[begin:min(begin + batch, n)]
        x = frames_to_tensor(raw[:, :context], device)
        y = frames_to_tensor(raw[:, shift:shift + context], device)
        zc, zf = encode(ae, x)
        pred_c, _, _ = model(zc, zf, steps=shift, reversible=True)
        pred01 = ((decode(ae, pred_c, level="coarse").clamp(-1, 1) + 1) * 0.5)
        true01 = (y + 1) * 0.5
        total_sq += float((pred01 - true01).square().sum())
        total_abs += float((pred01 - true01).abs().sum())
        total_frames += pred01.shape[0] * pred01.shape[1]
    return {
        "mse": total_sq / total_frames,
        "mae": total_abs / total_frames,
        "mean_pixel_mse": total_sq / (total_frames * 64 * 64),
        "clips": n,
    }


def save_checkpoint(path: Path, model, optimizer, update: int, args, best_mse: float,
                    history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "best_mse": best_mse,
        "args": vars(args),
        "history": history,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, temporary)
    os.replace(temporary, path)


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_sequences(args.train_path)
    split = len(data) - args.val_clips
    if split <= 0:
        raise ValueError("val_clips leaves no training data")
    train_indices = np.arange(split)
    val_data = data[split:]
    ae = hier_convae.load(args.ae_path)
    model = FixedRegionRevPredictor(
        dim=args.dim, heads=args.heads, depth=args.depth,
        max_frames=args.context, max_steps=args.shift, temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.updates, eta_min=args.lr * 0.05)
    rng = np.random.RandomState(args.seed + 991)
    start_update = 0
    best_mse = float("inf")
    history: list[dict] = []
    output = Path(args.output)
    if args.resume and output.exists():
        checkpoint = torch.load(output, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        best_mse = float(checkpoint["best_mse"])
        history = list(checkpoint.get("history", []))
        print(f"resumed {output} at update {start_update}", flush=True)

    print(json.dumps({
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "tokens_per_frame": model.tokens_per_frame,
        "context_tokens": count_context_tokens(args.context),
        "protocol": f"Moving-MNIST {args.context}->{args.context}, displacement {args.shift}",
        "quality_gate": {"ConvLSTM-S_200ep_mse": 29.80, "SimVP-gSTA_local_mse": 26.688837},
    }, indent=2), flush=True)
    started = time.time()
    running = 0.0
    for update in range(start_update + 1, args.updates + 1):
        if update <= args.identity_updates:
            shift = 0
        else:
            elapsed = update - args.identity_updates - 1
            maximum = min(args.shift, 1 + elapsed // args.curriculum_interval)
            shift = args.shift if maximum == args.shift and rng.rand() < 0.5 else int(rng.randint(1, maximum + 1))
        x, y = sample_batch(data, train_indices, args.batch, args.context, shift, rng, device)
        zc, zf = encode(ae, x)
        target_c, target_f = encode(ae, y)
        pred_c, pred_f, weights = model(zc, zf, steps=shift, reversible=args.reversible)
        pred_frames = decode(ae, pred_c, level="coarse")
        pixel_loss = F.mse_loss(pred_frames, y)
        latent_loss = F.smooth_l1_loss(pred_c, target_c)
        entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=-1).mean()
        loss = pixel_loss + args.latent_weight * latent_loss - args.entropy_weight * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())

        if update % args.log_every == 0:
            elapsed_s = time.time() - started
            print(
                f"update={update} shift={shift} loss={running / args.log_every:.6f} "
                f"pixel={float(pixel_loss.detach()):.6f} latent={float(latent_loss.detach()):.6f} "
                f"{elapsed_s / max(1, update - start_update):.3f}s/update",
                flush=True,
            )
            running = 0.0
        if update % args.eval_every == 0 or update == args.updates:
            metrics = evaluate(model, ae, val_data, device, args.context, args.shift,
                               args.eval_batch, args.eval_clips)
            record = {"update": update, **metrics, "elapsed_hours": (time.time() - started) / 3600}
            history.append(record)
            print("validation " + json.dumps(record), flush=True)
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                save_checkpoint(output.with_name(output.stem + "_best.pt"), model, optimizer,
                                update, args, best_mse, history)
            save_checkpoint(output, model, optimizer, update, args, best_mse, history)


def run_evaluation(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    saved = checkpoint["args"]
    ae = hier_convae.load(args.ae_path)
    model = FixedRegionRevPredictor(
        dim=saved["dim"], heads=saved["heads"], depth=saved["depth"],
        max_frames=saved["context"], max_steps=saved["shift"], temperature=saved["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    data = load_sequences(args.test_path)
    metrics = evaluate(model, ae, data, device, saved["context"], saved["shift"],
                       args.eval_batch, args.eval_clips or None)
    metrics.update({
        "checkpoint": args.output,
        "update": checkpoint["update"],
        "conv_lstm_200ep_mse": 29.80,
        "simvp_gsta_local_mse": 26.688837,
        "passes_convlstm_gate": metrics["mse"] <= 29.80,
        "passes_simvp_gate": metrics["mse"] <= 26.688837,
    })
    print(json.dumps(metrics, indent=2), flush=True)
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train", "eval"), default="train")
    ap.add_argument("--train-path", default="data/mmnist_bridge_train_10000x20.npy")
    ap.add_argument("--test-path", default=str(OPENSTL_ROOT / "data/moving_mnist/mnist_test_seq.npy"))
    ap.add_argument("--ae-path", default="logs_rhcp/hier_ae_mmnist.pt")
    ap.add_argument("--output", default=str(STORAGE_ROOT / "fixed_region_revformer_mmnist_s0.pt"))
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--identity-updates", type=int, default=300)
    ap.add_argument("--curriculum-interval", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-clips", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--val-clips", type=int, default=500)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--shift", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--latent-weight", type=float, default=0.1)
    ap.add_argument("--entropy-weight", type=float, default=0.001)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reversible", action=argparse.BooleanOptionalAction, default=True)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "train":
        train(arguments)
    else:
        run_evaluation(arguments)
