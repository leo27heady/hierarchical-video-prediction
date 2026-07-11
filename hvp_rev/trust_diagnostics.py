"""CPU-testable interventions and summaries for addressability versus rollout trust."""
from __future__ import annotations

import numpy as np
import torch


def corrupt_latent(latent: torch.Tensor, sigma: float, *, seed: int) -> torch.Tensor:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    generator = torch.Generator(device=latent.device).manual_seed(seed)
    noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
    return latent + sigma * noise


def refresh_mask(horizon: int, every: int | None) -> np.ndarray:
    """True at 1-indexed rollout frames where a GT memory refresh is injected."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if every is None:
        return np.zeros(horizon, dtype=bool)
    if every < 1:
        raise ValueError("refresh interval must be positive")
    frames = np.arange(1, horizon + 1)
    return frames % every == 0


def trust_penalty(addressable_error: np.ndarray, rollout_error: np.ndarray) -> np.ndarray:
    """Positive means self-feeding is worse than the addressable/teacher-forced condition."""
    addressable_error, rollout_error = np.asarray(addressable_error), np.asarray(rollout_error)
    if addressable_error.shape != rollout_error.shape:
        raise ValueError("trust conditions are unaligned")
    return rollout_error - addressable_error
