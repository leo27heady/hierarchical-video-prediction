"""Train one hash-authorized Article 4 primary run with crash-safe local/W&B evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hvp_hca.article4_campaign import (
    Article4CampaignRunSpec,
    file_sha256,
    load_campaign_run_spec,
    validate_campaign_authorization,
)
from hvp_hca.data import PackedWindowBatcher, load_trb_development
from hvp_hca.performance import enable_mixed_precision_rmsnorm
from hvp_hca.tracking import WandbMirror
from hvp_rev import hier3, hier_convae, viz
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


def clean_future_evidence(run_dir: Path, maximum_step: int) -> None:
    pattern = re.compile(r"(?:step|panel_step)(\d{6})")
    for directory in (run_dir / "panels", run_dir / "panel_data", run_dir / "milestones"):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            match = pattern.search(path.name)
            if path.is_file() and match and int(match.group(1)) > maximum_step:
                path.unlink()


def git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    return commit, dirty


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {"pid": -1}
            if process_alive(int(existing.get("pid", -1))):
                raise RuntimeError(f"run already active under PID {existing.get('pid')}")
            stale = self.path.with_name(f"run.lock.stale.{int(time.time())}.json")
            os.replace(self.path, stale)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = json.dumps(
            {"pid": os.getpid(), "host": socket.gethostname(), "created_utc": utc_now()},
            sort_keys=True,
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.close(descriptor)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.path.exists():
            self.path.unlink()


def forward_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "ctx_t": batch.get("ctx_t"),
        "ctx_c_frame_ids": batch["ctx_c_frame_ids"],
        "ctx_f_ids": batch["ctx_f_ids"],
        "ctx_f_frame_ids": batch["ctx_f_frame_ids"],
        "ctx_f_packed": batch["ctx_f_packed"],
        "ctx_f_reference_packed": batch.get("ctx_f_reference_packed"),
    }


def global_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = (-2, -1)
    mu_x = prediction.mean(dim=dimensions)
    mu_y = target.mean(dim=dimensions)
    x_centered = prediction - mu_x[..., None, None]
    y_centered = target - mu_y[..., None, None]
    var_x = x_centered.square().mean(dim=dimensions)
    var_y = y_centered.square().mean(dim=dimensions)
    covariance = (x_centered * y_centered).mean(dim=dimensions)
    return (
        (2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)
        / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    )


@torch.no_grad()
def validate(
    *,
    ema: legacy.EMA,
    ae: torch.nn.Module,
    batcher: PackedWindowBatcher,
    validation_batches: int,
    nfe: int,
    step: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    wrapper = legacy.WindowDiffusion(ema.shadow)
    totals: dict[str, float] = {}
    sample_count = 0
    panel: dict[str, Any] | None = None
    devices = [torch.cuda.current_device()] if legacy.DEV == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(910_000)
        if legacy.DEV == "cuda":
            torch.cuda.manual_seed_all(910_000)
        for batch_index in range(validation_batches):
            batch = batcher.sample(
                split="validation", seed=710_000 + batch_index, include_frames=True
            )
            kwargs = forward_kwargs(batch)
            loss = wrapper.loss(
                batch["ctx_c"], batch["ctx_f"], batch["tgt_c"], batch["tgt_f"],
                batch["actions"], **kwargs,
            )
            sampled = wrapper.sample(
                batch["ctx_c"], batch["ctx_f"], batch["actions"], n_steps=nfe, **kwargs
            )
            prediction = sampled[1]
            target_latent = batch["tgt_f"]
            batch_size, horizon = prediction.shape[:2]
            decoded = decode01(ae, prediction.reshape(-1, *prediction.shape[2:]))
            decoded = decoded.reshape(batch_size, horizon, 1, 64, 64)
            target = batch["gt_frames_01"]
            squared = (decoded - target).square()
            absolute = (decoded - target).abs()
            foreground = target > 0.5
            background = ~foreground
            values = {
                "diffusion_loss": float(loss),
                "latent_mse": float(F.mse_loss(prediction.float(), target_latent.float())),
                "pixel_mse": float(squared.mean()),
                "pixel_mae": float(absolute.mean()),
                "foreground_mse": float(
                    (squared * foreground).sum() / foreground.sum().clamp_min(1)
                ),
                "background_mse": float(
                    (squared * background).sum() / background.sum().clamp_min(1)
                ),
                "psnr": float(-10.0 * torch.log10(squared.mean().clamp_min(1e-12))),
                "ssim_global": float(global_ssim(decoded, target).mean()),
            }
            for horizon_index in range(horizon):
                h_squared = squared[:, horizon_index]
                h_absolute = absolute[:, horizon_index]
                h_foreground = foreground[:, horizon_index]
                h_ssim = global_ssim(decoded[:, horizon_index], target[:, horizon_index])
                prefix = f"horizon{horizon_index + 1}"
                values[f"{prefix}_pixel_mse"] = float(h_squared.mean())
                values[f"{prefix}_pixel_mae"] = float(h_absolute.mean())
                values[f"{prefix}_foreground_mse"] = float(
                    (h_squared * h_foreground).sum() / h_foreground.sum().clamp_min(1)
                )
                values[f"{prefix}_psnr"] = float(
                    -10.0 * torch.log10(h_squared.mean().clamp_min(1e-12))
                )
                values[f"{prefix}_ssim_global"] = float(h_ssim.mean())
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            sample_count += batch_size
            if panel is None:
                panel = {
                    "sampled_clips": batch["sampled_clips"],
                    "sampled_starts": batch["sampled_starts"],
                    "last": batch["last_context_frame_01"].detach().cpu(),
                    "prediction": decoded.detach().cpu(),
                    "target": target.detach().cpu(),
                    "absolute_error": absolute.detach().cpu(),
                }
    metrics = {key: value / sample_count for key, value in totals.items()}
    metrics.update(
        {
            "step": float(step),
            "validation_samples": float(sample_count),
            "validation_frames": float(sample_count * 4),
        }
    )
    if panel is None:
        raise AssertionError("validation produced no panel")
    return metrics, panel


def save_panel(run_dir: Path, step: int, panel: dict[str, Any]) -> tuple[Path, Path]:
    png = run_dir / "panels" / f"panel_step{step:06d}.png"
    raw = run_dir / "panel_data" / f"panel_step{step:06d}.npz"
    columns = [("last context", panel["last"])]
    for horizon in range(4):
        columns.extend(
            [
                (f"prediction h{horizon + 1}", panel["prediction"][:, horizon]),
                (f"target h{horizon + 1}", panel["target"][:, horizon]),
                (f"absolute error h{horizon + 1}", panel["absolute_error"][:, horizon]),
            ]
        )
    viz.save_samples(str(png), columns, n=min(8, len(panel["sampled_clips"])))
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw,
        sampled_clips=np.asarray(panel["sampled_clips"], dtype=np.int64),
        sampled_starts=np.asarray(panel["sampled_starts"], dtype=np.int64),
        last=panel["last"].numpy().astype(np.float16),
        prediction=panel["prediction"].numpy().astype(np.float16),
        target=panel["target"].numpy().astype(np.float16),
        absolute_error=panel["absolute_error"].numpy().astype(np.float16),
    )
    return png, raw


def checkpoint_payload(
    *,
    spec: Article4CampaignRunSpec,
    campaign_hash: str,
    source_commit: str,
    step: int,
    model: torch.nn.Module,
    ema: legacy.EMA,
    optimizer: torch.optim.Optimizer | None,
    batcher: PackedWindowBatcher | None,
    validation: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "config_hash": spec.content_hash,
        "campaign_sha256": campaign_hash,
        "source_commit": source_commit,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow.state_dict(),
    }
    if validation is not None:
        payload["validation"] = validation
    if optimizer is not None and batcher is not None:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "torch_rng_state": torch.random.get_rng_state(),
                "numpy_batch_rng_state": batcher.rng.bit_generator.state,
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if legacy.DEV == "cuda" else None
                ),
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--resolution", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true", help="smoke only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = load_campaign_run_spec(args.config)
    campaign, campaign_hash, resolution_hash = validate_campaign_authorization(
        spec, args.campaign, resolution_path=args.resolution
    )
    commit, dirty = git_state(root)
    if commit != args.source_commit:
        raise RuntimeError(f"source commit changed: {commit} != {args.source_commit}")
    if dirty and not args.smoke:
        raise RuntimeError("refusing scientific training from a dirty worktree")
    manifest_path = (root / spec.dataset_manifest).resolve()
    if file_sha256(manifest_path) != spec.dataset_manifest_sha256:
        raise ValueError("dataset manifest hash mismatch")
    if file_sha256(spec.ae_path) != spec.ae_sha256:
        raise ValueError("base autoencoder hash mismatch")
    if spec.top_ae_path and file_sha256(spec.top_ae_path) != spec.top_ae_sha256:
        raise ValueError("top autoencoder hash mismatch")

    if args.smoke_updates <= 0:
        parser.error("--smoke-updates must be positive")
    effective_updates = args.smoke_updates if args.smoke else spec.train_updates
    output_root = root / ("artifacts/v2/smoke/article4_campaign" if args.smoke else spec.output_root)
    run_dir = output_root / f"{spec.experiment_id}__{spec.content_hash[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = root / "artifacts/v2/article4_primary_30k/registry/runs.jsonl"
    training_path = run_dir / "training_metrics.jsonl"
    validation_path = run_dir / "validation_metrics.jsonl"
    rolling_path = run_dir / "rolling_resume.pt"
    status_path = run_dir / "status.json"
    attempts_path = run_dir / "attempts.jsonl"
    if attempts_path.exists():
        prior_attempts = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        attempt = 1 + max(
            (int(record.get("attempt", 0)) for record in prior_attempts), default=0
        )
    else:
        attempt = 1

    with RunLock(run_dir / "run.lock"):
        append_jsonl(
            attempts_path,
            {"event": "attempt_started", "attempt": attempt, "pid": os.getpid(), "utc": utc_now()},
        )
        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": legacy.DEV,
            "gpu": torch.cuda.get_device_name(0) if legacy.DEV == "cuda" else None,
            "gpu_capability": (
                list(torch.cuda.get_device_capability(0)) if legacy.DEV == "cuda" else None
            ),
            "host": socket.gethostname(),
        }
        run_manifest = {
            "schema_version": 2,
            "campaign_id": spec.campaign_id,
            "campaign_sha256": campaign_hash,
            "dependency_resolution_sha256": resolution_hash,
            "experiment_id": spec.experiment_id,
            "config": spec.canonical_dict(),
            "config_hash": spec.content_hash,
            "allocation_hash": spec.allocation_hash,
            "config_file_sha256": file_sha256(args.config),
            "git_commit": commit,
            "git_dirty_at_launch": dirty,
            "smoke": args.smoke,
            "environment": environment,
            "attempt": attempt,
            "started_utc": utc_now(),
            "primary_checkpoint": 30_000,
            "selection_policy": "final 30k checkpoint; intermediate validation is diagnostic",
        }
        atomic_json(run_dir / "run_manifest.json", run_manifest)
        append_jsonl(registry, {"event": "started", **run_manifest})
        atomic_json(status_path, {"status": "loading_data", "step": 0, "updated_utc": utc_now()})

        tracker = WandbMirror(
            run_dir=run_dir,
            project=spec.wandb_project,
            group=spec.wandb_group,
            experiment_id=spec.experiment_id,
            config_hash=spec.content_hash,
            config={
                **spec.canonical_dict(),
                "config_hash": spec.content_hash,
                "allocation_hash": spec.allocation_hash,
                "campaign_sha256": campaign_hash,
                "source_commit": commit,
            },
            tags=["article4", spec.family.lower(), spec.policy, spec.overlap, f"seed-{spec.seed}"],
            disabled=args.no_wandb,
        )
        try:
            legacy.M._BF16 = True
            torch.manual_seed(spec.seed)
            np.random.seed(spec.seed)
            if legacy.DEV == "cuda":
                torch.cuda.manual_seed_all(spec.seed)
            if spec.top_ae_path:
                ae = hier3.load_top(spec.ae_path, spec.top_ae_path)
            else:
                ae = hier_convae.load(spec.ae_path)
            loaded = load_trb_development(
                ae, manifest_path, verify_source_hashes=not args.skip_source_hash
            )
            if spec.top_ae_path:
                hier3.attach_top(loaded.data, ae)
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
                n_ctx_t=len(plan.top_ages),
                n_ctx_c=len(plan.coarse_ages),
                n_ctx_f=plan.reach,
                fine_token_budget=plan.fine_tokens,
                mode="diff",
                d=spec.width,
                depth=spec.depth,
                heads=spec.heads,
                temporal_embedding_positions=spec.temporal_embedding_positions,
                temporal_context_span=spec.context_anchor,
                top_context_only=bool(plan.top_ages),
            ).to(legacy.DEV)
            replacements = (
                enable_mixed_precision_rmsnorm(net) if spec.mixed_precision_rmsnorm else 0
            )
            wrapper = legacy.WindowDiffusion(net)
            ema = legacy.EMA(net, decay=spec.ema_decay)
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
            )

            start_step = 1
            if rolling_path.exists():
                saved = torch.load(rolling_path, map_location="cpu", weights_only=False)
                for key, expected in (
                    ("config_hash", spec.content_hash),
                    ("campaign_sha256", campaign_hash),
                    ("source_commit", commit),
                ):
                    if saved.get(key) != expected:
                        raise ValueError(f"rolling checkpoint {key} mismatch")
                net.load_state_dict(saved["model"])
                ema.shadow.load_state_dict(saved["ema"])
                optimizer.load_state_dict(saved["optimizer"])
                torch.random.set_rng_state(saved["torch_rng_state"])
                train_batcher.rng.bit_generator.state = saved["numpy_batch_rng_state"]
                if legacy.DEV == "cuda" and saved["cuda_rng_state"] is not None:
                    torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
                restored = int(saved["step"])
                truncate_jsonl_after_step(training_path, restored)
                truncate_jsonl_after_step(validation_path, restored)
                clean_future_evidence(run_dir, restored)
                start_step = restored + 1

            state_bytes = train_batcher.state_bytes_per_sample()
            run_manifest.update(
                {
                    "dataset_manifest_sha256": loaded.manifest_sha256,
                    "allocation": {
                        "policy": plan.policy,
                        "overlap": plan.overlap,
                        "fine_representation": plan.fine_representation,
                        "levels": spec.levels,
                        "reach": plan.reach,
                        "top_times": len(plan.top_ages),
                        "coarse_times": len(plan.coarse_ages),
                        "fine_tokens": plan.fine_tokens,
                        "context_tokens": plan.token_count,
                    },
                    "state_bytes_per_sample_bf16": state_bytes,
                    "latent_standardization": {
                        "coarse_mean": loaded.data.stats[0].reshape(-1).tolist(),
                        "coarse_std": loaded.data.stats[1].reshape(-1).tolist(),
                        "fine_mean": loaded.data.stats[2].reshape(-1).tolist(),
                        "fine_std": loaded.data.stats[3].reshape(-1).tolist(),
                        "top_mean": (
                            loaded.data.zt_stats[0].reshape(-1).tolist()
                            if hasattr(loaded.data, "zt_stats")
                            else None
                        ),
                        "top_std": (
                            loaded.data.zt_stats[1].reshape(-1).tolist()
                            if hasattr(loaded.data, "zt_stats")
                            else None
                        ),
                        "computed_from": "development training split only",
                    },
                    "model_parameters": sum(parameter.numel() for parameter in net.parameters()),
                    "expected_attention_sequence": net.expected_seq,
                    "mixed_precision_rmsnorm_replacements": replacements,
                    "effective_updates": effective_updates,
                    "wandb_url": tracker.url,
                }
            )
            atomic_json(run_dir / "run_manifest.json", run_manifest)
            atomic_json(
                status_path,
                {"status": "running", "step": start_step - 1, "attempt": attempt, "updated_utc": utc_now()},
            )
            if start_step > effective_updates:
                atomic_json(
                    status_path,
                    {"status": "complete", "step": effective_updates, "resume_noop": True, "updated_utc": utc_now()},
                )
                return 0

            if legacy.DEV == "cuda":
                torch.cuda.reset_peak_memory_stats()
            net.train()
            rolling_loss = rolling_grad = 0.0
            rolling_count = 0
            rolling_started = time.perf_counter()
            run_started = time.perf_counter()
            last_completed_step = start_step - 1
            for step in range(start_step, effective_updates + 1):
                for group in optimizer.param_groups:
                    group["lr"] = spec.learning_rate * min(1.0, step / spec.warmup_updates)
                batch = train_batcher.sample(split="train")
                loss = wrapper.loss(
                    batch["ctx_c"], batch["ctx_f"], batch["tgt_c"], batch["tgt_f"],
                    batch["actions"], **forward_kwargs(batch),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                ema.update(net)
                last_completed_step = step
                rolling_loss += float(loss.detach())
                rolling_grad += float(grad_norm.detach())
                rolling_count += 1
                if step % spec.log_every == 0 or step == effective_updates:
                    elapsed = time.perf_counter() - rolling_started
                    seconds_per_update = elapsed / rolling_count
                    record = {
                        "event": "training",
                        "step": step,
                        "mean_loss": rolling_loss / rolling_count,
                        "mean_grad_norm_pre_clip": rolling_grad / rolling_count,
                        "seconds_per_update": seconds_per_update,
                        "samples_per_second": spec.batch_size / seconds_per_update,
                        "context_tokens_per_second": (
                            spec.batch_size * plan.token_count / seconds_per_update
                        ),
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds_this_attempt": time.perf_counter() - run_started,
                        "eta_seconds": seconds_per_update * (effective_updates - step),
                        "cuda_allocated_gib": (
                            torch.cuda.memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
                        ),
                        "cuda_reserved_gib": (
                            torch.cuda.memory_reserved() / 2**30 if legacy.DEV == "cuda" else 0.0
                        ),
                        "cuda_peak_allocated_gib": (
                            torch.cuda.max_memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
                        ),
                        "attempt": attempt,
                        "utc": utc_now(),
                    }
                    append_jsonl(training_path, record)
                    tracker.log({f"train/{key}": value for key, value in record.items() if isinstance(value, (int, float))}, step=step)
                    print(json.dumps(record), flush=True)
                    rolling_loss = rolling_grad = 0.0
                    rolling_count = 0
                    rolling_started = time.perf_counter()

                if step % spec.checkpoint_every == 0 or step == effective_updates:
                    atomic_torch_save(
                        rolling_path,
                        checkpoint_payload(
                            spec=spec,
                            campaign_hash=campaign_hash,
                            source_commit=commit,
                            step=step,
                            model=net,
                            ema=ema,
                            optimizer=optimizer,
                            batcher=train_batcher,
                        ),
                    )
                    atomic_json(
                        status_path,
                        {"status": "running", "step": step, "attempt": attempt, "updated_utc": utc_now()},
                    )

                if step % spec.validation_every == 0 or step == effective_updates:
                    net.eval()
                    validation_started = time.perf_counter()
                    metrics, panel = validate(
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
                    append_jsonl(validation_path, metrics)
                    tracker.log(
                        {f"validation/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))},
                        step=step,
                    )
                    if step % spec.image_every == 0 or step == effective_updates:
                        panel_png, _ = save_panel(run_dir, step, panel)
                        tracker.image(
                            "validation/fixed_prediction_panel",
                            panel_png,
                            step=step,
                            caption=f"{spec.experiment_id} fixed validation panel at step {step}",
                        )
                    if step % spec.milestone_every == 0 or step == effective_updates:
                        milestone = run_dir / "milestones" / f"milestone_step{step:06d}.pt"
                        atomic_torch_save(
                            milestone,
                            checkpoint_payload(
                                spec=spec,
                                campaign_hash=campaign_hash,
                                source_commit=commit,
                                step=step,
                                model=net,
                                ema=ema,
                                optimizer=None,
                                batcher=None,
                                validation=metrics,
                            ),
                        )
                    print(json.dumps(metrics), flush=True)
                    net.train()

            if legacy.DEV == "cuda":
                torch.cuda.synchronize()
            final_milestone = run_dir / "milestones" / "milestone_step030000.pt"
            if not args.smoke and not final_milestone.exists():
                raise RuntimeError("final immutable 30k milestone is missing")
            summary = {
                "status": "complete",
                "experiment_id": spec.experiment_id,
                "config_hash": spec.content_hash,
                "step": effective_updates,
                "runtime_seconds_this_attempt": time.perf_counter() - run_started,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 2**30 if legacy.DEV == "cuda" else 0.0
                ),
                "peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 2**30 if legacy.DEV == "cuda" else 0.0
                ),
                "final_milestone_sha256": (
                    file_sha256(final_milestone) if final_milestone.exists() else None
                ),
                "training_metrics_sha256": file_sha256(training_path),
                "validation_metrics_sha256": file_sha256(validation_path),
                "wandb_url": tracker.url,
                "completed_utc": utc_now(),
            }
            atomic_json(status_path, summary)
            append_jsonl(registry, {"event": "complete", **summary})
            append_jsonl(attempts_path, {"event": "attempt_complete", "attempt": attempt, "step": effective_updates, "utc": utc_now()})
            tracker.finish(
                summary=summary,
                artifact_paths=(run_dir / "run_manifest.json", training_path, validation_path, status_path),
            )
            if rolling_path.exists() and not args.smoke:
                rolling_path.unlink()
            print(json.dumps(summary, indent=2), flush=True)
            return 0
        except BaseException as error:
            interrupted = isinstance(error, KeyboardInterrupt)
            failure = {
                "status": "interrupted" if interrupted else "failed",
                "experiment_id": spec.experiment_id,
                "config_hash": spec.content_hash,
                "step": locals().get("last_completed_step", 0),
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": utc_now(),
                "rolling_checkpoint_exists": rolling_path.exists(),
            }
            atomic_json(status_path, failure)
            append_jsonl(registry, {"event": failure["status"], **failure})
            append_jsonl(attempts_path, {"event": f"attempt_{failure['status']}", **failure})
            tracker.finish(summary=failure)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
