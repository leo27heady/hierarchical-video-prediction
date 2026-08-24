"""Train the fixed-region spatial adapter and evaluate it through frozen OpenSTL SimVP."""
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


def load_simvp(root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(root))
    path = root / "openstl/models/simvp_model.py"
    spec = importlib.util.spec_from_file_location("simvp_fixed_region_direct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SimVP_Model(
        (10, 1, 64, 64), hid_S=64, hid_T=512, N_T=8, N_S=4,
        model_type="gSTA", drop_path=0,
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({key[6:]: value for key, value in state.items() if key.startswith("model.")})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def normalize(raw: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.array(raw, dtype=np.float32, copy=True)).to(device).unsqueeze(2) / 127.5 - 1


def encode(ae, frames: torch.Tensor):
    b, t = frames.shape[:2]
    with torch.no_grad():
        zc, zf = ae.encode(frames.reshape(b * t, 1, 64, 64))
    return zc.reshape(b, t, *zc.shape[1:]), zf.reshape(b, t, *zf.shape[1:])


def decode_level(ae, latent: torch.Tensor, level: str) -> torch.Tensor:
    zf = latent
    b, t = zf.shape[:2]
    flat = zf.reshape(b * t, *zf.shape[2:])
    decoded = ae.decode_fine(flat) if level == "fine" else ae.decode_coarse(flat)
    return decoded.reshape(b, t, 1, 64, 64)


def blended_reconstruction(adapter, ae, zc, predicted_zf, regions):
    coarse = decode_level(ae, zc, "coarse")
    return coarse + adapter.pixel_blend * adapter.pixel_residual(regions)


@torch.no_grad()
def downstream_metrics(adapter, ae, simvp, test: np.ndarray, device: torch.device,
                       clips: int, batch: int) -> dict:
    n = min(clips, test.shape[0])
    squared = absolute = reconstruction = 0.0
    frames = 0
    for begin in range(0, n, batch):
        sequence = normalize(test[begin:min(begin + batch, n)], device)
        context, target = sequence[:, :10], sequence[:, 10:20]
        zc, zf = encode(ae, context)
        reconstructed_zf, regions, _ = adapter(zc, zf)
        reconstructed = blended_reconstruction(adapter, ae, zc, reconstructed_zf, regions).clamp(-1, 1)
        prediction = simvp(((reconstructed + 1) * 0.5).contiguous()).clamp(0, 1)
        truth = (target + 1) * 0.5
        squared += float((prediction - truth).square().sum())
        absolute += float((prediction - truth).abs().sum())
        reconstruction += float((((reconstructed + 1) * 0.5 - (context + 1) * 0.5).square().sum()))
        frames += context.shape[0] * 10
    return {
        "mse": squared / frames,
        "mae": absolute / frames,
        "input_reconstruction_mse": reconstruction / frames,
        "clips": n,
    }


def save(path: Path, adapter, optimizer, update: int, best: float, args, history: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": adapter.state_dict(), "optimizer": optimizer.state_dict(),
        "update": update, "best_mse": best, "args": vars(args), "history": history,
    }, temporary)
    os.replace(temporary, path)


def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = np.load(args.train_path, mmap_mode="r")
    raw_test = np.load(args.test_path, mmap_mode="r")
    test = raw_test.transpose(1, 0, 2, 3) if raw_test.shape[0] == 20 else raw_test
    ae = hier_convae.load(args.ae_path)
    adapter = FixedRegionReconstructor(args.temperature).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.updates, pct_start=0.05,
    )
    root = Path(args.openstl)
    simvp = load_simvp(root, Path(args.simvp_checkpoint), device)
    rng = np.random.RandomState(args.seed + 31)
    best = float("inf")
    history: list[dict] = []
    started = time.time()
    running = 0.0
    for update in range(1, args.updates + 1):
        clips = rng.randint(0, train.shape[0], size=args.batch)
        times = rng.randint(0, train.shape[1], size=args.batch)
        raw = train[clips, times][:, None]
        frame = normalize(raw, device)
        zc, zf = encode(ae, frame)
        predicted_zf, regions, _ = adapter(zc, zf)
        reconstruction = blended_reconstruction(adapter, ae, zc, predicted_zf, regions)
        pixel = F.mse_loss(reconstruction, frame)
        latent = F.smooth_l1_loss(predicted_zf, zf)
        loss = pixel + args.latent_weight * latent
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += float(loss.detach())
        if update % args.log_every == 0:
            print(
                f"update={update} loss={running / args.log_every:.6f} "
                f"pixel={float(pixel.detach()):.6f} latent={float(latent.detach()):.6f} "
                f"pixel_blend={float(adapter.pixel_blend.detach()):.4f}",
                flush=True,
            )
            running = 0.0
        if update % args.eval_every == 0 or update == args.updates:
            metrics = downstream_metrics(adapter, ae, simvp, test, device,
                                         args.eval_clips, args.eval_batch)
            record = {"update": update, **metrics, "elapsed_hours": (time.time() - started) / 3600}
            history.append(record)
            print("validation " + json.dumps(record), flush=True)
            output = Path(args.output)
            if metrics["mse"] < best:
                best = metrics["mse"]
                save(output.with_name(output.stem + "_best.pt"), adapter, optimizer,
                     update, best, args, history)
            save(output, adapter, optimizer, update, best, args, history)
    final = downstream_metrics(adapter, ae, simvp, test, device, args.final_clips, args.eval_batch)
    result = {
        **final,
        "best_development_mse": best,
        "context_tokens_per_frame": 24,
        "stored_values_per_frame": adapter.values_per_frame,
        "raw_values_per_frame": 64 * 64,
        "raw_to_packed_value_ratio": (64 * 64) / adapter.values_per_frame,
        "official_convlstm_s_200ep_mse": 29.80,
        "local_raw_simvp_mse": 26.688837,
        "passes_convlstm_gate": final["mse"] <= 29.80,
        "passes_raw_simvp_gate": final["mse"] <= 26.688837,
    }
    result_path = Path(args.metrics_out)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/mmnist_bridge_train_10000x20.npy")
    parser.add_argument("--test-path", default=r"C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\mnist_test_seq.npy")
    parser.add_argument("--ae-path", default="logs_rhcp/hier_ae_mmnist.pt")
    parser.add_argument("--openstl", default=r"C:\Users\leoni\Documents\projects\OpenSTL")
    parser.add_argument("--simvp-checkpoint", default=r"C:\Users\leoni\Documents\projects\OpenSTL\work_dirs\mmnist_simvp_gsta_anchor\checkpoints\best.ckpt")
    parser.add_argument("--output", default=r"D:\hierarchical-video-prediction-storage\feasibility\fixed_region_adapter_mmnist_s0.pt")
    parser.add_argument("--metrics-out", default=r"D:\hierarchical-video-prediction-storage\feasibility\fixed_region_adapter_mmnist_s0_metrics.json")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=10)
    parser.add_argument("--eval-clips", type=int, default=200)
    parser.add_argument("--final-clips", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
