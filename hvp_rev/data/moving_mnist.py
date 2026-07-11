"""Moving-MNIST generator with persistent identities and official split provenance."""
from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hvp_rev.provenance import atomic_save_npz, atomic_write_bytes, atomic_write_json, sha256_file


SCHEMA_VERSION = "moving-mnist-persistent-v1"
DIGIT_SIZE = 28


@dataclass(frozen=True)
class MovingMNISTBundle:
    frames: np.ndarray
    object_id: np.ndarray
    glyph_index: np.ndarray
    class_id: np.ndarray
    trajectory_yx: np.ndarray
    bbox_yxyx: np.ndarray
    overlap: np.ndarray
    source_split: str
    seed: int

    def validate(self) -> None:
        if self.frames.ndim != 4:
            raise ValueError(f"frames must be (N,T,H,W), got {self.frames.shape}")
        n, t, h, w = self.frames.shape
        if self.object_id.ndim != 2 or self.object_id.shape[0] != n:
            raise ValueError("object_id must be (N,K)")
        k = self.object_id.shape[1]
        expected = {
            "glyph_index": (n, k),
            "class_id": (n, k),
            "trajectory_yx": (n, k, t, 2),
            "bbox_yxyx": (n, k, t, 4),
            "overlap": (n, t, k, k),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape {getattr(self, name).shape} != {shape}")
        if h < DIGIT_SIZE or w < DIGIT_SIZE:
            raise ValueError("canvas is smaller than a source glyph")
        if self.source_split not in {"train", "test", "synthetic-fixture"}:
            raise ValueError(f"unrecognized source split {self.source_split!r}")


def load_idx(images_path: str | Path, labels_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(images_path, "rb") as handle:
        images = np.frombuffer(handle.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open(labels_path, "rb") as handle:
        labels = np.frombuffer(handle.read(), np.uint8, offset=8)
    if len(images) != len(labels):
        raise ValueError("MNIST image/label counts differ")
    return images, labels


def _trajectory(frames: int, canvas: int, rng: np.random.RandomState, step: float) -> np.ndarray:
    x, y = rng.rand(), rng.rand()
    theta = rng.rand() * 2 * np.pi
    vx, vy = np.cos(theta), np.sin(theta)
    sx, sy = np.empty(frames), np.empty(frames)
    for frame in range(frames):
        x += vx * step
        y += vy * step
        if x <= 0 or x >= 1:
            x = np.clip(x, 0, 1)
            vx = -vx
        if y <= 0 or y >= 1:
            y = np.clip(y, 0, 1)
            vy = -vy
        sx[frame], sy[frame] = x, y
    return np.stack([(canvas * sy).astype(np.int16), (canvas * sx).astype(np.int16)], axis=-1)


def generate_moving_mnist(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    n_clips: int,
    frames: int,
    image_size: int = 64,
    objects: int = 2,
    seed: int = 0,
    step: float = 0.1,
    source_split: str,
) -> MovingMNISTBundle:
    """Generate max-composited clips and persistent per-object metadata."""
    images, labels = np.asarray(images), np.asarray(labels)
    if images.shape[1:] != (DIGIT_SIZE, DIGIT_SIZE) or labels.shape != (len(images),):
        raise ValueError("source arrays must be MNIST-shaped images and labels")
    if image_size < DIGIT_SIZE:
        raise ValueError("image_size must be at least 28")
    rng = np.random.RandomState(seed)
    clips = np.zeros((n_clips, frames, image_size, image_size), dtype=np.uint8)
    object_id = np.arange(n_clips * objects, dtype=np.int64).reshape(n_clips, objects)
    glyph_index = np.empty((n_clips, objects), dtype=np.int64)
    class_id = np.empty((n_clips, objects), dtype=np.int16)
    trajectory = np.empty((n_clips, objects, frames, 2), dtype=np.int16)
    bbox = np.empty((n_clips, objects, frames, 4), dtype=np.int16)
    overlap = np.zeros((n_clips, frames, objects, objects), dtype=bool)
    canvas = image_size - DIGIT_SIZE

    for clip in range(n_clips):
        object_masks = np.zeros((objects, frames, image_size, image_size), dtype=bool)
        for obj in range(objects):
            glyph = int(rng.randint(len(images)))
            glyph_index[clip, obj] = glyph
            class_id[clip, obj] = int(labels[glyph])
            path = _trajectory(frames, canvas, rng, step)
            trajectory[clip, obj] = path
            source = images[glyph]
            for frame, (y, x) in enumerate(path):
                y, x = int(y), int(x)
                bbox[clip, obj, frame] = (y, x, y + DIGIT_SIZE, x + DIGIT_SIZE)
                region = clips[clip, frame, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE]
                clips[clip, frame, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE] = np.maximum(region, source)
                object_masks[obj, frame, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE] = source > 0
        for frame in range(frames):
            for first in range(objects):
                for second in range(first + 1, objects):
                    value = bool(np.logical_and(object_masks[first, frame], object_masks[second, frame]).any())
                    overlap[clip, frame, first, second] = value
                    overlap[clip, frame, second, first] = value

    bundle = MovingMNISTBundle(
        frames=clips,
        object_id=object_id,
        glyph_index=glyph_index,
        class_id=class_id,
        trajectory_yx=trajectory,
        bbox_yxyx=bbox,
        overlap=overlap,
        source_split=source_split,
        seed=seed,
    )
    bundle.validate()
    return bundle


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    atomic_write_bytes(path, buffer.getvalue())


def save_moving_mnist(
    bundle: MovingMNISTBundle,
    output: str | Path,
    *,
    purpose: str,
    generator_commit: str,
    source_images_path: str | Path | None = None,
    source_labels_path: str | Path | None = None,
) -> dict:
    """Save frames, metadata and a hash-pinned manifest; refuses overwrite."""
    bundle.validate()
    if purpose not in {"development", "confirmation_holdout"}:
        raise ValueError("purpose must be development or confirmation_holdout")
    output = Path(output)
    metadata_path = output.with_suffix(".metadata.npz")
    manifest_path = output.with_suffix(".manifest.json")
    existing = [p for p in (output, metadata_path, manifest_path) if p.exists()]
    if existing:
        raise FileExistsError("refusing dataset overwrite: " + ", ".join(map(str, existing)))
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(output, bundle.frames)
    atomic_save_npz(
        metadata_path,
        object_id=bundle.object_id,
        glyph_index=bundle.glyph_index,
        class_id=bundle.class_id,
        trajectory_yx=bundle.trajectory_yx,
        bbox_yxyx=bundle.bbox_yxyx,
        overlap=bundle.overlap,
        source_split=np.asarray(bundle.source_split),
        seed=np.asarray(bundle.seed),
        schema_version=np.asarray(SCHEMA_VERSION),
    )
    source = {}
    for key, value in (("images", source_images_path), ("labels", source_labels_path)):
        if value is not None:
            source[key] = {"path": str(value), "sha256": sha256_file(value)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"moving-mnist-{bundle.source_split}-s{bundle.seed}",
        "purpose": purpose,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_commit": generator_commit,
        "seed": bundle.seed,
        "source_split": bundle.source_split,
        "source": source,
        "shape": list(bundle.frames.shape),
        "files": {
            "frames": {"path": output.name, "sha256": sha256_file(output)},
            "metadata": {"path": metadata_path.name, "sha256": sha256_file(metadata_path)},
        },
    }
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest
