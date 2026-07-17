"""Train the E6 digit classifier from official MNIST-train glyphs with immutable provenance."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hvp_rev import wb
from hvp_rev.digit_classifier import load_idx, train_digit_classifier
from hvp_rev.phasec_v2 import extract_components
from hvp_rev.provenance import (
    atomic_write_bytes,
    atomic_write_json,
    environment_fingerprint,
    sha256_file,
    sha256_value,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact-root", default="artifacts/phasec_v2/classifier")
    parser.add_argument("--wandb-project", default="hgrid")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--authorized-gpu-hours", type=float, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("E6 classifier training requires CUDA in the scientific controller")
    spec = {
        "schema_version": 1,
        "study": "phase-c-persistent-object-v2",
        "role": "identity-classifier",
        "architecture": "DigitCNN-v1",
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "source_split": "official-MNIST-train",
        "input_distribution": (
            "uniform mixture of canonical 28x28 glyphs and deterministic production-component "
            "crops from the same official-train glyphs"
        ),
        "production_crop": {"threshold": 0.3, "min_component_pixels": 4},
        "images_sha256": sha256_file(args.images),
        "labels_sha256": sha256_file(args.labels),
        "source_commit": args.source_commit,
        "environment": environment_fingerprint(),
    }
    identity = sha256_value(spec)[:16]
    run_dir = Path(args.artifact_root) / identity
    complete = run_dir / "COMPLETE"
    if complete.exists():
        raise FileExistsError(f"completed classifier is immutable: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_dir / "run_spec.json", spec)

    run_id = f"phasec-v2-classifier-{identity}"
    wb.init(
        run_id,
        config={"run_spec": spec, "cli": vars(args)},
        project=args.wandb_project,
        run_id=run_id,
        group=args.wandb_group,
        tags=["moving-mnist", "phase-c-v2", "classifier", "official-train-only"],
        job_type="classifier-training",
        required=True,
        resume="never",
    )
    active = Path("artifacts/active_experiment.json")
    started = time.time()
    atomic_write_json(active, {
        "schema_version": 1, "status": "running", "study": "phase-c-persistent-object-v2",
        "arm": "digit-classifier", "seed": args.seed, "optimizer_updates": args.steps,
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
        "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
    })

    def progress(row: dict) -> None:
        wb.log({"train/loss": row["train_loss"]}, step=row["step"])

    try:
        images, labels = load_idx(args.images, args.labels)
        production_crops = []
        for frame in images[:, 0].numpy():
            components = extract_components(frame, threshold=0.3, min_component_pixels=4)
            production_crops.append(
                max(components, key=lambda item: item.pixels).crop if components else frame
            )
        augmented = torch.from_numpy(np.asarray(production_crops, np.float32))[:, None]
        training_images = torch.cat([images, augmented], dim=0)
        training_labels = torch.cat([labels, labels], dim=0)
        model = train_digit_classifier(
            training_images, training_labels, seed=args.seed, steps=args.steps, batch_size=args.batch,
            learning_rate=args.lr, device="cuda", progress_callback=progress,
        )
        checkpoint = run_dir / "checkpoint.pt"
        torch.save(model.cpu().state_dict(), checkpoint)
        elapsed = (time.time() - started) / 3600
        report = {
            "schema_version": 1,
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_gpu_hours": elapsed,
            "wandb_url": wb.active_url(),
        }
        atomic_write_json(run_dir / "training.json", report)
        manifest = {
            "schema_version": 1,
            "run_spec_hash": identity,
            "artifacts": [
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(run_dir.iterdir()) if path.is_file()
            ],
        }
        atomic_write_json(run_dir / "manifest.json", manifest)
        wb.summary({"training/complete": True, "training/gpu_hours": elapsed})
        wb.artifact(
            f"phasec-v2-classifier-{identity}",
            "digit-classifier",
            [str(path) for path in run_dir.iterdir() if path.is_file()],
            metadata={"run_spec_hash": identity},
            aliases=["latest"],
        )
        atomic_write_bytes(complete, f"{identity}\n".encode())
        atomic_write_json(active, {
            "schema_version": 1, "status": "complete", "study": "phase-c-persistent-object-v2",
            "arm": "digit-classifier", "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        print(f"PHASEC_V2_CLASSIFIER_DONE {identity} {checkpoint}", flush=True)
    except Exception:
        atomic_write_json(active, {
            "schema_version": 1, "status": "failed", "study": "phase-c-persistent-object-v2",
            "arm": "digit-classifier", "seed": args.seed, "optimizer_updates": args.steps,
            "pid": os.getpid(), "failed_utc": datetime.now(timezone.utc).isoformat(),
            "run_spec_hash": identity, "run_dir": run_dir.as_posix(),
            "wandb_url": wb.active_url(), "authorized_gpu_hours": args.authorized_gpu_hours,
        })
        raise
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
