"""Build the frozen pre-access classifier-domain fixture from public/development data only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from hvp_rev import hier_convae
from hvp_rev.data.moving_mnist import load_idx
from hvp_rev.phasec_v2 import extract_components
from hvp_rev.provenance import atomic_save_npz, atomic_write_json, sha256_file


def _nearest_production_crop(
    frame: np.ndarray,
    expected_center: tuple[float, float],
) -> np.ndarray | None:
    components = extract_components(frame, threshold=0.3, min_component_pixels=30)
    if not components:
        return None
    return min(
        components,
        key=lambda item: np.hypot(
            item.detection.centroid_yx[0] - expected_center[0],
            item.detection.centroid_yx[1] - expected_center[1],
        ),
    ).crop


def _canvas_crop(image: np.ndarray, y: int, x: int) -> np.ndarray | None:
    canvas = np.zeros((64, 64), np.float32)
    canvas[y:y + 28, x:x + 28] = image.astype(np.float32) / 255.0
    return _nearest_production_crop(canvas, (y + 14, x + 14))


def _development_crops(
    clips: np.ndarray,
    metadata: dict[str, np.ndarray],
    *,
    want_overlap: bool,
    limit: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = metadata["overlap"].any(axis=(2, 3))
    if want_overlap:
        eligible = overlap
    else:
        # The registered rendered-component gate is the isolated domain. Pixel masks can be
        # disjoint while the other 28x28 box still enters the target crop, which is an ambiguous
        # multi-digit sample and belongs in the separate overlap diagnostic.
        boxes = metadata["bbox_yxyx"]
        eligible = np.ones(overlap.shape, dtype=bool)
        for first in range(boxes.shape[1]):
            for second in range(first + 1, boxes.shape[1]):
                a, b = boxes[:, first], boxes[:, second]
                separated = (
                    (a[..., 2] + 2 <= b[..., 0]) | (b[..., 2] + 2 <= a[..., 0])
                    | (a[..., 3] + 2 <= b[..., 1]) | (b[..., 3] + 2 <= a[..., 1])
                )
                eligible &= separated
    candidate = np.argwhere(eligible)
    rng.shuffle(candidate)
    crops, labels = [], []
    for clip, frame in candidate:
        objects = np.arange(metadata["class_id"].shape[1])
        rng.shuffle(objects)
        for obj in objects:
            box = tuple(int(v) for v in metadata["bbox_yxyx"][clip, obj, frame])
            y0, x0, y1, x1 = box
            crop = _nearest_production_crop(
                clips[clip, frame].astype(np.float32) / 255.0,
                ((y0 + y1) / 2, (x0 + x1) / 2),
            )
            if crop is None:
                continue
            crops.append(crop)
            labels.append(int(metadata["class_id"][clip, obj]))
            if len(crops) >= limit:
                return np.asarray(crops, np.float32), np.asarray(labels, np.int64)
    if not crops:
        raise ValueError(f"no {'overlap' if want_overlap else 'non-overlap'} development crops")
    return np.asarray(crops, np.float32), np.asarray(labels, np.int64)


@torch.no_grad()
def _ae_crops(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    ae_path: str,
    indices: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    ae = hier_convae.load(ae_path)
    device = next(ae.parameters()).device
    canvases, boxes = [], []
    for index in indices:
        y, x = int(rng.randint(0, 37)), int(rng.randint(0, 37))
        canvas = np.zeros((64, 64), np.float32)
        canvas[y:y + 28, x:x + 28] = images[index].astype(np.float32) / 255.0
        canvases.append(canvas)
        boxes.append((y, x, y + 28, x + 28))
    reconstructed = []
    for start in range(0, len(canvases), 256):
        batch = torch.from_numpy(np.asarray(canvases[start:start + 256]))[:, None].to(device)
        _, fine = ae.encode(batch * 2 - 1)
        output = (ae.decode_fine(fine).clamp(-1, 1) + 1) * 0.5
        reconstructed.extend(output[:, 0].cpu().numpy())
    crops, retained_labels = [], []
    for frame, box, label in zip(reconstructed, boxes, labels[indices]):
        y0, x0, y1, x1 = box
        crop = _nearest_production_crop(frame, ((y0 + y1) / 2, (x0 + x1) / 2))
        if crop is not None:
            crops.append(crop)
            retained_labels.append(int(label))
    return np.asarray(crops, np.float32), np.asarray(retained_labels, np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-images", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--development-metadata", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    report_path = Path(args.report)
    if output.exists() or report_path.exists():
        raise FileExistsError("classifier-domain fixture/report is immutable")
    rng = np.random.RandomState(args.seed)
    test_images, test_labels = load_idx(args.test_images, args.test_labels)
    if len(test_images) != 10000:
        raise ValueError("official MNIST test split must contain 10,000 glyphs")
    selected = rng.choice(len(test_images), size=args.limit, replace=False)
    clean = test_images[selected].astype(np.float32) / 255.0
    clean_labels = test_labels[selected].astype(np.int64)

    translated, translated_labels = [], []
    for index in selected:
        crop = _canvas_crop(
            test_images[index], int(rng.randint(0, 37)), int(rng.randint(0, 37))
        )
        if crop is not None:
            translated.append(crop)
            translated_labels.append(int(test_labels[index]))
    translated = np.asarray(translated, np.float32)
    translated_labels = np.asarray(translated_labels, np.int64)

    clips = np.load(args.development, mmap_mode="r")
    with np.load(args.development_metadata) as source:
        metadata = {key: source[key] for key in source.files}
    rendered, rendered_labels = _development_crops(
        clips, metadata, want_overlap=False, limit=args.limit, rng=rng,
    )
    overlap, overlap_labels = _development_crops(
        clips, metadata, want_overlap=True, limit=args.limit, rng=rng,
    )
    ae_reconstruction, ae_labels = _ae_crops(
        test_images, test_labels, ae_path=args.ae, indices=selected, rng=rng,
    )

    # Registered deliberately-degraded domain: lower-resolution, attenuated AE crops plus
    # seeded Gaussian sensor noise. It is development-only and never tuned after holdout access.
    tensor = torch.from_numpy(ae_reconstruction)[:, None]
    degraded = F.interpolate(tensor, size=(20, 20), mode="bilinear", align_corners=False)
    degraded = F.interpolate(degraded, size=(28, 28), mode="bilinear", align_corners=False)[:, 0]
    degraded = degraded.numpy() * 0.82
    degraded += rng.normal(0, 0.025, degraded.shape).astype(np.float32)
    degraded = np.clip(degraded, 0, 1)

    arrays = {
        "clean_test_images": clean,
        "clean_test_labels": clean_labels,
        "translated_images": translated,
        "translated_labels": translated_labels,
        "rendered_component_images": rendered,
        "rendered_component_labels": rendered_labels,
        "overlap_images": overlap,
        "overlap_labels": overlap_labels,
        "ae_reconstruction_images": ae_reconstruction,
        "ae_reconstruction_labels": ae_labels,
        "generated_degraded_images": degraded,
        "generated_degraded_labels": ae_labels,
    }
    atomic_save_npz(output, **arrays)
    report = {
        "schema_version": 1,
        "status": "CLASSIFIER_DOMAINS_BUILT",
        "seed": args.seed,
        "fixture": output.as_posix(),
        "fixture_sha256": sha256_file(output),
        "sources": {
            "test_images_sha256": sha256_file(args.test_images),
            "test_labels_sha256": sha256_file(args.test_labels),
            "development_sha256": sha256_file(args.development),
            "development_metadata_sha256": sha256_file(args.development_metadata),
            "ae_sha256": sha256_file(args.ae),
        },
        "counts": {
            name.removesuffix("_images"): int(len(value))
            for name, value in arrays.items() if name.endswith("_images")
        },
        "degraded_domain": {
            "downsample": "28->20->28 bilinear",
            "attenuation": 0.82,
            "gaussian_noise_std": 0.025,
        },
    }
    atomic_write_json(report_path, report)
    print(f"PHASEC_V2_CLASSIFIER_DOMAINS_DONE {output} {report['fixture_sha256']}", flush=True)


if __name__ == "__main__":
    main()
