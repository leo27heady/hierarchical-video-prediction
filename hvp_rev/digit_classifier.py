"""Seeded, path-configurable digit classifier used only after domain validation."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_idx(images_path: str | Path, labels_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    with gzip.open(images_path, "rb") as handle:
        images = np.frombuffer(handle.read(), np.uint8, offset=16).reshape(-1, 28, 28).copy()
    with gzip.open(labels_path, "rb") as handle:
        labels = np.frombuffer(handle.read(), np.uint8, offset=8).astype(np.int64)
    if len(images) != len(labels):
        raise ValueError("classifier images and labels differ in length")
    return torch.from_numpy(images).float().unsqueeze(1) / 255.0, torch.from_numpy(labels)


def train_digit_classifier(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    steps: int = 3000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    progress_callback: Callable[[dict], None] | None = None,
) -> DigitCNN:
    torch.manual_seed(seed)
    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    model.train()
    running = 0.0
    running_count = 0
    for step in range(1, steps + 1):
        index = torch.randint(0, len(images), (batch_size,), generator=generator)
        loss = F.cross_entropy(model(images[index].to(device)), labels[index].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.detach())
        running_count += 1
        if progress_callback is not None and (step % 250 == 0 or step == steps):
            progress_callback({"step": step, "train_loss": running / running_count})
            running = 0.0
            running_count = 0
    return model.eval()
