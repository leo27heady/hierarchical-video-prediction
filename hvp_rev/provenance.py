"""Immutable run identity, artifact manifests, holdout access, and RNG-complete checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCIENTIFIC_FIELDS = (
    "dataset_hash", "ae_hash", "arm", "arm_config", "seed", "train_examples",
    "optimizer_updates", "batch_size", "precision", "nfe", "warmup_frames",
    "rollout_frames", "evaluation_clips", "source_commit", "metric_version",
    "optimizer", "ema", "sampler",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def environment_fingerprint() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


@dataclass(frozen=True)
class RunSpec:
    study: str
    arm: str
    seed: int
    dataset_hash: str
    ae_hash: str
    arm_config: Mapping[str, Any]
    train_examples: int
    optimizer_updates: int
    batch_size: int
    precision: str
    nfe: int
    warmup_frames: int
    rollout_frames: int
    evaluation_clips: int
    source_commit: str
    metric_version: str = "complete-v1"
    optimizer: Mapping[str, Any] = field(default_factory=dict)
    ema: Mapping[str, Any] = field(default_factory=dict)
    sampler: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=environment_fingerprint)
    tag: str = ""

    def resolved(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> str:
        return sha256_value(self.resolved())[:16]

    def assert_compatible(self, other: "RunSpec") -> None:
        left, right = self.resolved(), other.resolved()
        mismatches = [k for k in SCIENTIFIC_FIELDS if left[k] != right[k]]
        if mismatches:
            details = ", ".join(f"{k}: {left[k]!r} != {right[k]!r}" for k in mismatches)
            raise ValueError(f"incompatible run specifications: {details}")


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_save_npy(path: str | Path, array: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(tmp_name, array)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


class ArtifactStore:
    """A content-addressed run directory that refuses completed-run overwrite."""

    def __init__(self, root: str | Path, spec: RunSpec, *, create: bool = True):
        self.spec = spec
        self.path = Path(root) / spec.study / spec.arm / f"seed-{spec.seed}" / spec.identity
        self.complete_marker = self.path / "COMPLETE"
        if self.complete_marker.exists():
            raise FileExistsError(f"completed run is immutable: {self.path}")
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
            spec_path = self.path / "run_spec.json"
            if spec_path.exists():
                old = json.loads(spec_path.read_text(encoding="utf-8"))
                # JSON round-trips tuples as lists. Compare the canonical serialized
                # representation so an interrupted run can reopen its own identical
                # content-addressed directory without weakening collision detection.
                if canonical_json(old) != canonical_json(spec.resolved()):
                    raise FileExistsError(f"run identity collision with different spec: {self.path}")
            else:
                atomic_write_json(spec_path, spec.resolved())

    def save_npz(self, relative: str | Path, **arrays: np.ndarray) -> Path:
        target = self.path / relative
        if target.exists():
            raise FileExistsError(f"artifact already exists: {target}")
        return atomic_save_npz(target, **arrays)

    def write_json(self, relative: str | Path, value: Any) -> Path:
        target = self.path / relative
        if target.exists():
            raise FileExistsError(f"artifact already exists: {target}")
        return atomic_write_json(target, value)

    def finalize(self) -> Path:
        artifacts = []
        for file in sorted(p for p in self.path.rglob("*") if p.is_file() and p.name != "COMPLETE"):
            artifacts.append({
                "path": file.relative_to(self.path).as_posix(),
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            })
        atomic_write_json(self.path / "manifest.json", {
            "run_spec_hash": self.spec.identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts,
        })
        return atomic_write_bytes(self.complete_marker, f"{self.spec.identity}\n".encode())


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    spec: RunSpec,
    step: int,
    model: Mapping[str, Any],
    optimizer: Mapping[str, Any],
    ema: Mapping[str, Any],
    scheduler: Mapping[str, Any] | None = None,
    scaler: Mapping[str, Any] | None = None,
    sampler_state: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "run_spec": spec.resolved(),
        "run_spec_hash": spec.identity,
        "step": int(step),
        "model": model,
        "optimizer": optimizer,
        "ema": ema,
        "scheduler": scheduler,
        "scaler": scaler,
        "sampler_state": sampler_state,
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def load_training_checkpoint(path: str | Path, spec: RunSpec, *, restore_rng: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 2:
        raise ValueError("legacy checkpoint lacks RNG-complete provenance; use the legacy loader explicitly")
    saved = RunSpec(**payload["run_spec"])
    spec.assert_compatible(saved)
    if payload.get("run_spec_hash") != saved.identity:
        raise ValueError("checkpoint run-spec hash is corrupt")
    if restore_rng:
        restore_rng_state(payload["rng"])
    return payload


def assert_dataset_purpose(manifest_path: str | Path, *, allowed: set[str], operation: str) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    purpose = manifest.get("purpose")
    if purpose not in allowed:
        raise PermissionError(f"dataset purpose {purpose!r} is forbidden for {operation}; allowed={allowed}")
    return manifest


def log_holdout_access(manifest_path: str | Path, access_log: str | Path, *, run_spec_hash: str) -> None:
    manifest = assert_dataset_purpose(
        manifest_path, allowed={"confirmation_holdout"}, operation="confirmation evaluation"
    )
    entry = canonical_json({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest_hash": sha256_file(manifest_path),
        "dataset_id": manifest.get("dataset_id"),
        "run_spec_hash": run_spec_hash,
    }) + "\n"
    access_log = Path(access_log)
    access_log.parent.mkdir(parents=True, exist_ok=True)
    with access_log.open("a", encoding="utf-8") as handle:
        handle.write(entry)
