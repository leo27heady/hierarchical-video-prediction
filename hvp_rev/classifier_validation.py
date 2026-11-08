"""Domain-stratified validation for Phase C digit classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ValidationDomain:
    name: str
    images: np.ndarray
    labels: np.ndarray
    minimum_accuracy: float | None


def _expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total, result = len(confidence), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence > lo) & (confidence <= hi if hi < 1 else confidence <= hi)
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result) if total else float("nan")


@torch.no_grad()
def validate_classifier(
    model: torch.nn.Module,
    domains: Mapping[str, ValidationDomain],
    *,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """Validate clean and shifted domains; overlap may be diagnostic with no pass threshold."""
    model = model.to(device).eval()
    reports, overall_pass = {}, True
    for name, domain in domains.items():
        images = np.asarray(domain.images, dtype=np.float32)
        labels = np.asarray(domain.labels, dtype=np.int64)
        if images.ndim == 3:
            images = images[:, None]
        if images.shape[1:] != (1, 28, 28) or labels.shape != (len(images),):
            raise ValueError(f"invalid classifier domain {name}: {images.shape}, {labels.shape}")
        pred, conf = [], []
        for start in range(0, len(images), batch_size):
            logits = model(torch.from_numpy(images[start:start + batch_size]).to(device))
            prob = logits.softmax(1)
            confidence, prediction = prob.max(1)
            pred.append(prediction.cpu().numpy())
            conf.append(confidence.cpu().numpy())
        prediction = np.concatenate(pred) if pred else np.empty(0, np.int64)
        confidence = np.concatenate(conf) if conf else np.empty(0, np.float32)
        correct = prediction == labels
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (labels, prediction), 1)
        accuracy = float(correct.mean()) if len(correct) else float("nan")
        passed = domain.minimum_accuracy is None or accuracy >= domain.minimum_accuracy
        overall_pass &= passed
        reports[name] = {
            "n": len(images),
            "accuracy": accuracy,
            "minimum_accuracy": domain.minimum_accuracy,
            "passed": passed,
            "mean_confidence": float(confidence.mean()) if len(confidence) else float("nan"),
            "ece_10bin": _expected_calibration_error(confidence, correct),
            "confusion": confusion.tolist(),
        }
    return {"passed": overall_pass, "domains": reports}
