"""One resumable OpenSTL-native integrated-model training worker.

Local JSONL/checkpoints are authoritative.  W&B is a best-effort mirror: a
network or authentication failure is recorded but never discards the local run.
The controller launches one worker at a time from an immutable campaign entry.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys
import subprocess
import time
from typing import Any, Dict

import numpy as np
import torch


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(root: Path) -> dict:
    return {
        "root": str(root),
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"], text=True
        ).strip(),
        "clean": not bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()),
    }


class JsonlEvidence:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> None:
        payload = {"timestamp": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def scalar_metrics(metrics: Dict[str, Any]) -> dict:
    result = {}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[str(name)] = float(value.detach().cpu())
        elif isinstance(value, (int, float)):
            result[str(name)] = float(value)
    return result


def frame_rgb(frame: torch.Tensor) -> np.ndarray:
    value = frame.detach().float().clamp(0, 1).cpu().numpy()
    if value.shape[0] == 1:
        value = np.repeat(value, 3, axis=0)
    elif value.shape[0] == 2:
        value = np.concatenate((value, np.zeros_like(value[:1])), axis=0)
    else:
        value = value[:3]
    return np.rint(value.transpose(1, 2, 0) * 255).astype(np.uint8)


def save_panel(panel: dict, path: Path) -> None:
    from PIL import Image, ImageDraw

    inputs = panel["inputs"][0]
    targets = panel["targets"][0]
    predictions = panel["predictions"][0]
    horizons = sorted({0, len(targets) - 1})
    tiles = [("last input", frame_rgb(inputs[-1]))]
    for horizon in horizons:
        target = frame_rgb(targets[horizon])
        prediction = frame_rgb(predictions[horizon])
        error = np.abs(target.astype(np.int16) - prediction.astype(np.int16)).clip(0, 255).astype(np.uint8)
        tiles.extend(
            [(f"target t+{horizon + 1}", target),
             (f"prediction t+{horizon + 1}", prediction),
             (f"error t+{horizon + 1}", error)]
        )
    height, width = tiles[0][1].shape[:2]
    label = 22
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * (height + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(tiles):
        row, column = divmod(index, columns)
        left, top = column * width, row * (height + label)
        canvas.paste(Image.fromarray(image), (left, top + label))
        draw.text((left + 2, top + 3), name, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, path)


class ResilientWandbMirror:
    def __init__(self, *, enabled: bool, mode: str, project: str, group: str,
                 run_id: str, config: dict, evidence: JsonlEvidence, output: Path):
        self.run = None
        self.module = None
        self.evidence = evidence
        if not enabled:
            evidence.write({"event": "wandb_disabled"})
            return
        try:
            import wandb
            self.module = wandb
            self.run = wandb.init(
                project=project,
                group=group,
                id=run_id,
                name=run_id,
                resume="allow",
                mode=mode,
                dir=str(output / "wandb"),
                config=config,
                settings=wandb.Settings(start_method="thread"),
            )
            evidence.write({"event": "wandb_initialized", "mode": mode, "project": project})
        except Exception as error:  # local run remains authoritative
            evidence.write({"event": "wandb_initialization_failed", "error": repr(error)})

    def log(self, values: dict, step: int, image_path: Path | None = None) -> None:
        if self.run is None:
            return
        try:
            payload = dict(values)
            if image_path is not None and image_path.exists():
                payload["validation_panel"] = self.module.Image(str(image_path))
            self.run.log(payload, step=step)
        except Exception as error:
            self.evidence.write({"event": "wandb_log_failed", "step": step, "error": repr(error)})
            self.run = None

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            with suppress(Exception):
                self.run.finish(exit_code=exit_code)


def load_config(path: Path) -> dict:
    values = runpy.run_path(str(path))
    return {key: value for key, value in values.items() if not key.startswith("__")}


def import_openstl(root: Path):
    sys.path.insert(0, str(root))
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from openstl.datasets import dataset_parameters, load_data
    from openstl.methods import method_maps
    from openstl.utils import default_parser
    return Trainer, seed_everything, Callback, ModelCheckpoint, dataset_parameters, load_data, method_maps, default_parser


def cuda_metrics() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "cuda_reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }


def main(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = JsonlEvidence(output / "metrics.jsonl")
    config_file = arguments.config.resolve()
    openstl_root = arguments.openstl.resolve()
    (
        Trainer, seed_everything, Callback, ModelCheckpoint, dataset_parameters,
        load_data, method_maps, default_parser,
    ) = import_openstl(openstl_root)

    config = dict(default_parser())
    config.update(dataset_parameters[arguments.dataset])
    config.update(load_config(config_file))
    config.update({
        "dataname": arguments.dataset,
        "seed": arguments.seed,
        "data_root": str(arguments.data_root.resolve()),
        "res_dir": str(output.parent),
        "ex_name": output.name,
        "save_dir": str(output),
        "gpus": [0],
        "dist": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": arguments.num_workers,
        "evidence_log_every": arguments.log_every,
    })
    if arguments.max_epochs is not None:
        config["epoch"] = arguments.max_epochs
    if arguments.batch_size is not None:
        config["batch_size"] = arguments.batch_size
        config["val_batch_size"] = arguments.batch_size
    atomic_json(output / "resolved_config.json", config)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    seed_everything(arguments.seed, workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
        torch.cuda.reset_peak_memory_stats()

    data_kwargs = {
        key: value for key, value in config.items()
        if key not in {"dataname", "batch_size", "val_batch_size", "num_workers", "data_root", "dist"}
    }
    train_loader, validation_loader, test_loader = load_data(
        arguments.dataset,
        config["batch_size"],
        config["val_batch_size"],
        config["num_workers"],
        config["data_root"],
        dist=False,
        **data_kwargs,
    )
    method = method_maps["hvpintegrated"](
        steps_per_epoch=len(train_loader),
        test_mean=getattr(test_loader.dataset, "mean", 0),
        test_std=getattr(test_loader.dataset, "std", 1),
        **config,
    )
    parameters = sum(parameter.numel() for parameter in method.model.parameters())
    trainable = sum(parameter.numel() for parameter in method.model.parameters() if parameter.requires_grad)
    planned_steps = int(config["epoch"]) * len(train_loader)
    contract = method.model.memory_contract(batch_size=config["batch_size"], dtype_bytes=2 if "bf16" in arguments.precision else 4)
    provenance = {
        "event": "worker_start",
        "run_id": arguments.run_id,
        "config_file": str(config_file),
        "config_sha256": file_sha256(config_file),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "train_batches": len(train_loader),
        "validation_batches": len(validation_loader),
        "test_batches": len(test_loader),
        "planned_optimizer_steps": planned_steps,
        "memory_contract": contract,
        "repositories": {
            "hvp": git_provenance(Path(__file__).resolve().parents[2]),
            "openstl": git_provenance(openstl_root),
        },
    }
    evidence.write(provenance)
    atomic_json(output / "run_provenance.json", provenance)

    mirror = ResilientWandbMirror(
        enabled=not arguments.no_wandb,
        mode=arguments.wandb_mode,
        project=arguments.wandb_project,
        group=arguments.wandb_group,
        run_id=arguments.run_id,
        config={**config, "run_id": arguments.run_id, "repositories": provenance["repositories"]},
        evidence=evidence,
        output=output,
    )

    class EvidenceCallback(Callback):
        def __init__(self):
            self.last_step_time = None
            self.last_logged_step = None

        def on_train_start(self, trainer, pl_module):
            self.last_step_time = time.perf_counter()
            self.last_logged_step = int(trainer.global_step)
            evidence.write({"event": "train_start", **cuda_metrics()})

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            if step == 0 or step % arguments.log_every:
                return
            now = time.perf_counter()
            duration = now - self.last_step_time if self.last_step_time is not None else float("nan")
            step_delta = step - self.last_logged_step if self.last_logged_step is not None else arguments.log_every
            self.last_step_time = now
            self.last_logged_step = step
            values = scalar_metrics(trainer.callback_metrics)
            values.update(cuda_metrics())
            values.update({
                "event": "train_progress",
                "step": step,
                "epoch": int(trainer.current_epoch),
                "batch": int(batch_idx),
                "seconds_per_log_interval": duration,
                "updates_per_second": step_delta / duration if duration > 0 else float("nan"),
                "samples_per_second": (
                    step_delta * int(config["batch_size"]) / duration if duration > 0 else float("nan")
                ),
                "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
            })
            evidence.write(values)
            mirror.log({key: value for key, value in values.items() if isinstance(value, (int, float))}, step)

        def on_validation_epoch_end(self, trainer, pl_module):
            step = int(trainer.global_step)
            values = scalar_metrics(trainer.callback_metrics)
            values.update(cuda_metrics())
            values.update({"event": "validation", "step": step, "epoch": int(trainer.current_epoch)})
            panel_path = None
            panel = getattr(pl_module, "validation_panel", None)
            if panel is not None:
                panel_path = output / "panels" / f"validation_epoch_{trainer.current_epoch:04d}.png"
                save_panel(panel, panel_path)
                values["panel"] = str(panel_path)
                np.save(
                    output / "panels" / f"routes_epoch_{trainer.current_epoch:04d}.npy",
                    panel["route_indices"].numpy(),
                )
            evidence.write(values)
            mirror.log(
                {key: value for key, value in values.items() if isinstance(value, (int, float))},
                step,
                panel_path,
            )

        def on_exception(self, trainer, pl_module, exception):
            evidence.write({"event": "trainer_exception", "error": repr(exception), **cuda_metrics()})

    class AtomicRollingCheckpoint(Callback):
        def _save(self, trainer):
            path = output / "checkpoints" / "last.ckpt"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".ckpt.tmp")
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, path)
            evidence.write({"event": "rolling_checkpoint", "step": int(trainer.global_step), "path": str(path)})

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step and trainer.global_step % arguments.checkpoint_every == 0:
                self._save(trainer)

        def on_train_epoch_end(self, trainer, pl_module):
            # A max_steps stop can occur midway through an epoch.  Its exact
            # batch checkpoint was already written by on_train_batch_end; do
            # not overwrite it with an epoch-end loop state that advances past
            # the unprocessed remainder.
            if not trainer.should_stop:
                self._save(trainer)

        def on_fit_end(self, trainer, pl_module):
            # The scientific schedule is update-count based. Preserve the exact
            # terminal optimizer state even if max_steps lands mid-epoch after a
            # resumed run skipped a dataloader remainder.
            self._save(trainer)

        def on_exception(self, trainer, pl_module, exception):
            with suppress(Exception):
                self._save(trainer)

    best = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        filename="best-{epoch:03d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
    )
    callbacks = [EvidenceCallback(), AtomicRollingCheckpoint(), best]
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=arguments.max_steps if arguments.max_steps is not None else planned_steps,
        precision=arguments.precision if torch.cuda.is_available() else "32-true",
        callbacks=callbacks,
        logger=False,
        deterministic=True,
        benchmark=False,
        gradient_clip_val=float(config.get("clip_grad", 1.0)),
        log_every_n_steps=max(1, arguments.log_every),
        enable_checkpointing=True,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        default_root_dir=str(output),
    )
    resume = output / "checkpoints" / "last.ckpt"
    ckpt_path = str(resume) if arguments.resume and resume.exists() else None
    started = time.time()
    try:
        trainer.fit(method, train_dataloaders=train_loader, val_dataloaders=validation_loader, ckpt_path=ckpt_path)
        best_path = best.best_model_path or str(resume)
        evidence.write({"event": "test_start", "checkpoint": best_path})
        trainer.test(method, dataloaders=test_loader, ckpt_path=best_path)
        panel = getattr(method, "test_panel", None)
        test_panel_path = output / "panels" / "locked_test.png"
        if panel is not None:
            save_panel(panel, test_panel_path)
            mirror.log({}, int(trainer.global_step), test_panel_path)
        result = {
            "status": "completed",
            "run_id": arguments.run_id,
            "best_checkpoint": best_path,
            "best_validation_loss": float(best.best_model_score) if best.best_model_score is not None else None,
            "test_metrics": getattr(method, "streaming_test_metrics", {}),
            "elapsed_hours": (time.time() - started) / 3600,
            "parameters": parameters,
            "memory_contract": contract,
            **cuda_metrics(),
        }
        atomic_json(output / "result.json", result)
        evidence.write({"event": "worker_complete", **result})
        mirror.finish(0)
        return 0
    except BaseException as error:
        evidence.write({"event": "worker_failed", "error": repr(error), **cuda_metrics()})
        atomic_json(output / "failure.json", {"status": "failed", "error": repr(error), "run_id": arguments.run_id})
        mirror.finish(1)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--openstl", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--dataset", choices=("mmnist", "taxibj"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--max-epochs", type=int)
    result.add_argument("--max-steps", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--precision", default="bf16-mixed")
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument("--checkpoint-every", type=int, default=1000)
    result.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--wandb-project", default="hierarchical-video-prediction-integrated-v2")
    result.add_argument("--wandb-group", default="integrated-openstl-confirmation-2026-09")
    result.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    result.add_argument("--no-wandb", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
