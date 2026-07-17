"""Validated persistent-object primitives for the E6 long Moving-MNIST boundary study."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from hvp_rev.evaluation import Detection, GroundTruthObject, evaluate_frames, score_objects


@dataclass(frozen=True)
class DetectedComponent:
    detection: Detection
    crop: np.ndarray
    pixels: int


def _classifier_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    output_size: int = 28,
    margin: int = 2,
) -> np.ndarray:
    """Return a square, zero-padded crop without using horizontal-order information."""
    y0, x0, y1, x1 = bbox
    height, width = y1 - y0, x1 - x0
    side = max(height, width, output_size) + 2 * margin
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    sy0, sx0 = int(np.floor(cy - side / 2)), int(np.floor(cx - side / 2))
    square = np.zeros((side, side), dtype=np.float32)
    iy0, ix0 = max(0, sy0), max(0, sx0)
    iy1, ix1 = min(frame.shape[0], sy0 + side), min(frame.shape[1], sx0 + side)
    square[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = frame[iy0:iy1, ix0:ix1]
    tensor = torch.from_numpy(square)[None, None]
    return F.interpolate(
        tensor, size=(output_size, output_size), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def extract_components(
    frame01: np.ndarray,
    *,
    threshold: float,
    min_component_pixels: int,
) -> list[DetectedComponent]:
    """Detect all sufficiently large connected components; output order is non-semantic."""
    frame = np.asarray(frame01, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError(f"frame must be (H,W), got {frame.shape}")
    labels, count = ndimage.label(frame > threshold)
    components: list[DetectedComponent] = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        pixels = int(len(ys))
        if pixels < min_component_pixels:
            continue
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
        weights = frame[ys, xs]
        total = float(weights.sum())
        if total > 0:
            centroid = (float((ys * weights).sum() / total), float((xs * weights).sum() / total))
        else:
            centroid = (float(ys.mean()), float(xs.mean()))
        crop = _classifier_crop(frame, bbox)
        components.append(
            DetectedComponent(
                Detection(centroid_yx=centroid, bbox_yxyx=bbox),
                crop=crop,
                pixels=pixels,
            )
        )
    return components


@torch.no_grad()
def classify_components(
    classifier: torch.nn.Module,
    components: Sequence[DetectedComponent],
    *,
    device: str,
    confidence_threshold: float,
) -> list[Detection]:
    """Attach class predictions; abstentions remain matched but identity-unscorable."""
    if not components:
        return []
    images = torch.from_numpy(np.stack([component.crop for component in components]))[:, None]
    probabilities = classifier.to(device).eval()(images.to(device)).softmax(1)
    confidence, classes = probabilities.max(1)
    detections = []
    for component, class_id, conf in zip(components, classes.cpu(), confidence.cpu()):
        value = float(conf)
        detections.append(
            Detection(
                centroid_yx=component.detection.centroid_yx,
                bbox_yxyx=component.detection.bbox_yxyx,
                class_id=int(class_id) if value >= confidence_threshold else None,
                class_confidence=value,
            )
        )
    return detections


def metadata_objects(
    *,
    object_id: np.ndarray,
    class_id: np.ndarray,
    bbox_yxyx: np.ndarray,
    clip: int,
    frame: int,
) -> list[GroundTruthObject]:
    """Build persistent GT objects; bbox centers define the registered trajectory coordinate."""
    result = []
    for index in range(object_id.shape[1]):
        box = tuple(int(value) for value in bbox_yxyx[clip, index, frame])
        y0, x0, y1, x1 = box
        result.append(
            GroundTruthObject(
                object_id=int(object_id[clip, index]),
                centroid_yx=((y0 + y1) / 2, (x0 + x1) / 2),
                bbox_yxyx=box,
                class_id=int(class_id[clip, index]),
                visible=True,
            )
        )
    return result


def evaluate_prediction_frame(
    pred01: np.ndarray,
    true01: np.ndarray,
    *,
    ground_truth: Sequence[GroundTruthObject],
    classifier: torch.nn.Module,
    device: str,
    detection_threshold: float,
    min_component_pixels: int,
    confidence_threshold: float,
    distance_weight: float,
    iou_weight: float,
    max_assignment_cost: float,
) -> dict:
    """Return false-positive-sensitive frame and intention-to-score object endpoints."""
    components = extract_components(
        pred01,
        threshold=detection_threshold,
        min_component_pixels=min_component_pixels,
    )
    detections = classify_components(
        classifier,
        components,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    objects = score_objects(
        detections,
        ground_truth,
        frame_shape=tuple(int(value) for value in pred01.shape),
        distance_weight=distance_weight,
        iou_weight=iou_weight,
        max_assignment_cost=max_assignment_cost,
    )
    frame = evaluate_frames(
        pred01[None, None],
        true01[None, None],
        threshold=detection_threshold,
        min_component_pixels=min_component_pixels,
    )
    frame_values = {
        key: np.asarray(value).reshape(-1)[0]
        for key, value in frame.items()
        if key != "record_shape"
    }
    return {"frame": frame_values, "objects": objects, "detections": detections}
