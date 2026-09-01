"""Config-driven, manifest-locked trainer for Article 4 A4-COR calibration fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_run import load_article4_run_spec, validate_authorization
from hvp_hca.data import PackedWindowBatcher, load_trb_development, sha256_file
from hvp_hca.performance import enable_mixed_precision_rmsnorm
from hvp_rev import hier_convae
from hvp_rev import rhvd as legacy
from hvp_rev.step1_mech import decode01


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_jsonl_after_step(path: Path, maximum_step: int) -> None:
    """Remove records newer than a rolling checkpoint before deterministic replay."""

    if not path.exists():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", -1)) <= maximum_step:
            retained.append(json.dumps(record, sort_keys=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(("\n".join(retained) + "\n") if retained else "", encoding="utf-8")
    os.replace(temporary, path)


def git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def forward_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "ctx_c_frame_ids": batch["ctx_c_frame_ids"],
        "ctx_f_ids": batch["ctx_f_ids"],
        "ctx_f_frame_ids": batch["ctx_f_frame_ids"],
        "ctx_f_packed": batch["ctx_f_packed"],
    }


@torch.no_grad()
def validate(
    *,
    ema: legacy.EMA,
    ae: torch.nn.Module,
    batcher: PackedWindowBatcher,
    validation_batches: int,
    nfe: int,
    step: int,
) -> dict[str, float]:
    wrapper = legacy.WindowDiffusion(ema.shadow)
    loss_sum = latent_sum = pixel_sum = fg_sum = h4_pixel_sum = h4_fg_sum = 0.0
    sample_count = frame_count = 0
    devices = [torch.cuda.current_device()] if legacy.DEV == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(910_000)
        if legacy.DEV == "cuda":
            torch.cuda.manual_seed_all(910_000)
        for batch_index in range(validation_batches):
            batch = batcher.sample(
                split="validation",
                seed=710_000 + batch_index,
                include_frames=True,
            )
            kwargs = forward_kwargs(batch)
            loss = wrapper.loss(
                batch["ctx_c"],
                batch["ctx_f"],
                batch["tgt_c"],
                batch["tgt_f"],
                batch["actions"],
                **kwargs,
            )
            _, prediction = wrapper.sample(
                batch["ctx_c"],
                batch["ctx_f"],
                batch["actions"],
                n_steps=nfe,
                **kwargs,
            )
            target = batch["tgt_f"]
            batch_size, horizon = prediction.shape[:2]
            latent = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            decoded = decode01(ae, prediction.reshape(-1, *prediction.shape[2:]))
            decoded = decoded.reshape(batch_size, horizon, 1, 64, 64)
            ground_truth = batch["gt_frames_01"]
            squared = (decoded - ground_truth).square()
            foreground = ground_truth > 0.5
            foreground_mse = (squared * foreground).sum() / foreground.sum().clamp_min(1)
            h4_squared = squared[:, -1]
            h4_foreground = foreground[:, -1]
            h4_foreground_mse = (
                (h4_squared * h4_foreground).sum() / h4_foreground.sum().clamp_min(1)
            )
            loss_sum += float(loss) * batch_size
            latent_sum += float(latent) * batch_size
            pixel_sum += float(squared.mean()) * batch_size
            fg_sum += float(foreground_mse) * batch_size
            h4_pixel_sum += float(h4_squared.mean()) * batch_size
            h4_fg_sum += float(h4_foreground_mse) * batch_size
            sample_count += batch_size
            frame_count += batch_size * horizon
    return {
        "step": float(step),
        "validation_samples": float(sample_count),
        "validation_frames": float(frame_count),
        "diffusion_loss": loss_sum / sample_count,
        "latent_mse": latent_sum / sample_count,
        "pixel_mse": pixel_sum / sample_count,
        "foreground_mse": fg_sum / sample_count,
        "horizon4_pixel_mse": h4_pixel_sum / sample_count,
        "horizon4_foreground_mse": h4_fg_sum / sample_count,
    }


def checkpoint_payload(
    *,
    spec_hash: str,
    step: int,
    model: torch.nn.Module,
    ema: legacy.EMA,
    optimizer: torch.optim.Optimizer | None,
    batcher: PackedWindowBatcher | None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config_hash": spec_hash,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
    }
    if metrics is not None:
        payload["validation"] = metrics
    if optimizer is not None and batcher is not None:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.random.get_rng_state(),
                "numpy_batch_rng_state": batcher.rng.bit_generator.state,
                "cuda_rng_state": torch.cuda.get_rng_state_all() if legacy.DEV == "cuda" else None,
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="execute two updates in an isolated smoke directory")
    parser.add_argument("--skip-source-hash", action="store_true", help="smoke/debug only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    spec = load_article4_run_spec(args.config)
    authorization = validate_authorization(spec, args.authorization)
    commit, dirty = git_state(root)
    if dirty and not args.smoke:
        raise RuntimeError("refusing scientific training from a dirty worktree")
    manifest_path = (root / spec.dataset_manifest).resolve()
    if sha256_file(manifest_path) != spec.dataset_manifest_sha256:
        raise ValueError("dataset manifest hash does not match run config")
    if sha256_file(spec.ae_path) != spec.ae_sha256:
        raise ValueError("autoencoder hash does not match run config")

    effective_updates = 2 if args.smoke else spec.train_updates
    output_root = root / ("artifacts/v2/smoke" if args.smoke else spec.output_root)
    run_dir = output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = root / "artifacts/v2/registry/article4_runs.jsonl"
    metrics_path = run_dir / "validation_metrics.jsonl"
    rolling_path = run_dir / "rolling_resume.pt"
    status_path = run_dir / "status.json"

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": legacy.DEV,
        "gpu": torch.cuda.get_device_name(0) if legacy.DEV == "cuda" else None,
    }
    run_manifest = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "config": spec.canonical_dict(),
        "config_hash": spec.content_hash,
        "config_file_sha256": sha256_file(args.config),
        "authorization_file_sha256": sha256_file(args.authorization),
        "authorization_purpose": authorization["purpose"],
        "git_commit": commit,
        "git_dirty_at_launch": dirty,
        "smoke": args.smoke,
        "environment": environment,
        "started_utc": utc_now(),
    }
    atomic_json(run_dir / "run_manifest.json", run_manifest)
    append_jsonl(registry, {"event": "started", **run_manifest})
    atomic_json(status_path, {"status": "loading_data", "step": 0, "updated_utc": utc_now()})

    legacy.M._BF16 = True
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    if legacy.DEV == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
    ae = hier_convae.load(spec.ae_path)
    loaded = load_trb_development(
        ae,
        manifest_path,
        verify_source_hashes=not args.skip_source_hash,
    )
    plan = spec.plan
    train_batcher = PackedWindowBatcher(
        loaded.data,
        plan,
        batch_size=spec.batch_size,
        seed=spec.seed,
        device=legacy.DEV,
        context_anchor=spec.context_anchor,
    )
    validation_batcher = PackedWindowBatcher(
        loaded.data,
        plan,
        batch_size=spec.validation_batch_size,
        seed=800_000,
        device=legacy.DEV,
        context_anchor=spec.context_anchor,
    )
    net = legacy.WindowDiT(
        n_ctx_c=len(plan.coarse_ages),
        n_ctx_f=plan.reach,
        fine_token_budget=plan.fine_tokens,
        mode="diff",
        d=spec.width,
        depth=spec.depth,
        heads=spec.heads,
        temporal_embedding_positions=spec.temporal_embedding_positions,
        temporal_context_span=spec.context_anchor,
    ).to(legacy.DEV)
    replacements = enable_mixed_precision_rmsnorm(net) if spec.mixed_precision_rmsnorm else 0
    wrapper = legacy.WindowDiffusion(net)
    ema = legacy.EMA(net, decay=spec.ema_decay)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )

    start_step = 1
    if rolling_path.exists() and not args.smoke:
        # RNG states are CPU ByteTensors even for a CUDA run. Loading the whole payload directly
        # onto CUDA makes torch.random.set_rng_state reject its state; model/optimizer loaders move
        # their tensors to the parameter device themselves.
        saved = torch.load(rolling_path, map_location="cpu", weights_only=False)
        if saved["config_hash"] != spec.content_hash:
            raise ValueError("rolling checkpoint belongs to a different config")
        net.load_state_dict(saved["model"])
        ema.shadow.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["torch_rng_state"])
        train_batcher.rng.bit_generator.state = saved["numpy_batch_rng_state"]
        if legacy.DEV == "cuda" and saved["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        restored_step = int(saved["step"])
        truncate_jsonl_after_step(run_dir / "training_metrics.jsonl", restored_step)
        truncate_jsonl_after_step(metrics_path, restored_step)
        start_step = restored_step + 1

    state_bytes = train_batcher.state_bytes_per_sample()
    run_manifest.update(
        {
            "dataset_manifest_sha256": loaded.manifest_sha256,
            "allocation": {
                "policy": plan.policy,
                "overlap": plan.overlap,
                "reach": plan.reach,
                "coarse_times": len(plan.coarse_ages),
                "fine_tokens": plan.fine_tokens,
                "context_tokens": plan.token_count,
                "common_context_anchor": spec.context_anchor,
            },
            "state_bytes_per_sample_bf16": state_bytes,
            "model_parameters": sum(parameter.numel() for parameter in net.parameters()),
            "expected_attention_sequence": net.expected_seq,
            "model_first_target_position": net.base,
            "mixed_precision_rmsnorm_replacements": replacements,
            "effective_updates": effective_updates,
        }
    )
    atomic_json(run_dir / "run_manifest.json", run_manifest)
    atomic_json(
        status_path,
        {"status": "running", "step": start_step - 1, "updated_utc": utc_now()},
    )

    if start_step > effective_updates:
        atomic_json(
            status_path,
            {
                "status": "complete",
                "experiment_id": spec.experiment_id,
                "config_hash": spec.content_hash,
                "step": effective_updates,
                "resume_noop": True,
                "updated_utc": utc_now(),
            },
        )
        print(f"{spec.experiment_id} already complete through {effective_updates}")
        return 0
    if legacy.DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train()
    rolling_loss = 0.0
    rolling_count = 0
    rolling_started = time.perf_counter()
    run_started = time.perf_counter()
    for step in range(start_step, effective_updates + 1):
        batch = train_batcher.sample(split="train")
        loss = legacy.training_update(
            wrapper,
            net,
            ema,
            optimizer,
            batch,
            step,
            lr=spec.learning_rate,
            warmup=spec.warmup_updates,
        )
        rolling_loss += float(loss.detach())
        rolling_count += 1
        if step % spec.log_every == 0 or step == effective_updates:
            elapsed = time.perf_counter() - rolling_started
            record = {
                "event": "training",
                "step": step,
                "mean_loss": rolling_loss / rolling_count,
                "seconds_per_update": elapsed / rolling_count,
                "lr": optimizer.param_groups[0]["lr"],
                "utc": utc_now(),
            }
            append_jsonl(run_dir / "training_metrics.jsonl", record)
            print(json.dumps(record), flush=True)
            rolling_loss = 0.0
            rolling_count = 0
            rolling_started = time.perf_counter()

        if step % spec.checkpoint_every == 0 or step == effective_updates:
            atomic_torch_save(
                rolling_path,
                checkpoint_payload(
                    spec_hash=spec.content_hash,
                    step=step,
                    model=net,
                    ema=ema,
                    optimizer=optimizer,
                    batcher=train_batcher,
                ),
            )
            atomic_json(status_path, {"status": "running", "step": step, "updated_utc": utc_now()})

        should_validate = step % spec.validation_every == 0 or step == effective_updates
        if should_validate:
            net.eval()
            validation_started = time.perf_counter()
            metrics = validate(
                ema=ema,
                ae=ae,
                batcher=validation_batcher,
                validation_batches=(1 if args.smoke else spec.validation_batches),
                nfe=(2 if args.smoke else spec.validation_nfe),
                step=step,
            )
            metrics.update(
                {
                    "event": "validation",
                    "validation_seconds": time.perf_counter() - validation_started,
                    "utc": utc_now(),
                }
            )
            append_jsonl(metrics_path, metrics)
            atomic_torch_save(
                run_dir / f"milestone_step{step:06d}.pt",
                checkpoint_payload(
                    spec_hash=spec.content_hash,
                    step=step,
                    model=net,
                    ema=ema,
                    optimizer=None,
                    batcher=None,
                    metrics=metrics,
                ),
            )
            print(json.dumps(metrics), flush=True)
            net.train()

    if legacy.DEV == "cuda":
        torch.cuda.synchronize()
    summary = {
        "status": "complete",
        "experiment_id": spec.experiment_id,
        "config_hash": spec.content_hash,
        "step": effective_updates,
        "runtime_seconds": time.perf_counter() - run_started,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 2**30 if legacy.DEV == "cuda" else 0.0
        ),
        "completed_utc": utc_now(),
    }
    atomic_json(status_path, summary)
    append_jsonl(registry, {"event": "complete", **summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
