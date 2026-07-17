"""Content-addressed, W&B-required wrapper for the legacy hierarchical codec kernel."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from hvp_rev import hier_convae, wb
from hvp_rev.provenance import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    environment_fingerprint,
    sha256_file,
    sha256_value,
)


def resolved_spec(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 1,
        "study": args.study,
        "architecture": "HierConvAE-2level-4x4-8x8-LAT16-w64",
        "seed": int(args.seed),
        "training_data": Path(args.data).as_posix(),
        "training_data_sha256": sha256_file(args.data),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "optimizer": "Adam",
        "scheduler": {"name": "OneCycleLR", "pct_start": 0.05},
        "objective": "random-level L1+MSE; coarse multiplier 4.0",
        "source_commit": args.source_commit,
        "environment": environment_fingerprint(),
        "tag": args.provenance_tag,
    }


def _manifest(run_dir: Path, identity: str) -> dict:
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "COMPLETE"):
        files.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": 1,
        "run_spec_hash": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="e7a-alternate-codec")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", default="data/loopmom_train_3000x100.npy")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--artifact_root", default="artifacts/autoencoders")
    parser.add_argument("--source_commit", required=True)
    parser.add_argument("--provenance_tag", default="e7a-codec-seed-intervention")
    parser.add_argument("--wandb_project", default="hgrid")
    parser.add_argument("--wandb_group", required=True)
    parser.add_argument("--dataset_tag", default="loopmom")
    parser.add_argument("--allow_seed_zero", action="store_true",
                        help="explicitly permit a fresh seed-0 codec outside the E7a intervention")
    parser.add_argument("--run_log", default="")
    parser.add_argument("--authorized_gpu_hours", type=float, required=True)
    args = parser.parse_args()

    if hier_convae.DEV != "cuda":
        raise RuntimeError("scientific E7a codec training requires CUDA")
    if args.seed == 0 and not args.allow_seed_zero:
        raise ValueError("E7a must train a non-legacy codec seed")
    spec = resolved_spec(args)
    identity = sha256_value(spec)[:16]
    run_dir = Path(args.artifact_root) / args.study / f"seed-{args.seed}" / identity
    complete = run_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError(f"completed codec run is immutable: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "run_spec.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text(encoding="utf-8"))
        if canonical_json(old) != canonical_json(spec):
            raise FileExistsError(f"codec RunSpec collision: {run_dir}")
    else:
        atomic_write_json(spec_path, spec)

    run_name = f"{args.study}-seed{args.seed}-{identity}"
    wb.init(
        run_name,
        config={"run_spec": spec, "cli": vars(args)},
        project=args.wandb_project,
        run_id=run_name,
        tags=[args.dataset_tag, "codec-training", args.provenance_tag],
        resume="never",
        group=args.wandb_group,
        job_type="codec-training",
        required=True,
    )
    active = Path("artifacts/active_experiment.json")
    started = time.time()
    atomic_write_json(active, {
        "schema_version": 1, "status": "running", "study": args.study,
        "arm": "hier-convae", "seed": args.seed, "optimizer_updates": args.steps,
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
        "run_log": args.run_log, "wandb_url": wb.active_url(),
        "authorized_gpu_hours": args.authorized_gpu_hours,
    })

    def progress(row: dict) -> None:
        wb.log({
            "train/loss": row["train_loss"],
            "val/coarse_mse": row["val_coarse_mse"],
            "val/fine_mse": row["val_fine_mse"],
            "train/lr": row["lr"],
            "health/elapsed_hours": row["elapsed_hours"],
            "health/vram_gb": row["peak_vram_gb"],
        }, step=row["step"])

    try:
        checkpoint = run_dir / "checkpoint.pt"
        if checkpoint.exists():
            raise FileExistsError(f"partial codec output requires an audit before retry: {checkpoint}")
        torch.cuda.reset_peak_memory_stats()
        hier_convae.train(
            args.data, str(checkpoint), steps=args.steps, batch=args.batch, lr=args.lr,
            seed=args.seed, progress_callback=progress,
        )
        elapsed = (time.time() - started) / 3600
        training_path = run_dir / "training.json"
        atomic_write_json(training_path, {
            "schema_version": 1,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_gpu_hours": elapsed,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "wandb_url": wb.active_url(),
        })
        manifest_path = run_dir / "manifest.json"
        atomic_write_json(manifest_path, _manifest(run_dir, identity))
        wb.summary({
            "training/complete": True,
            "training/gpu_hours": elapsed,
            "training/checkpoint_sha256": sha256_file(checkpoint),
        })
        wb.artifact(
            f"e7a-codec-{identity}", "trained-codec",
            [str(spec_path), str(checkpoint), str(training_path), str(manifest_path)],
            metadata={"run_spec_hash": identity, "seed": args.seed}, aliases=["latest", f"seed-{args.seed}"],
        )
        atomic_write_bytes(complete, f"{identity}\n".encode())
        atomic_write_json(active, {
            "schema_version": 1, "status": "complete", "study": args.study,
            "arm": "hier-convae", "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        print(f"E7A_CODEC_DONE {identity} {checkpoint}", flush=True)
    except Exception:
        atomic_write_json(active, {
            "schema_version": 1, "status": "failed", "study": args.study,
            "arm": "hier-convae", "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "failed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        raise
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
