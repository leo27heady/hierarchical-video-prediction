"""Training-free motion baselines for the prediction-horizon gate.

Persistence (a frozen last frame) is a ZERO-MOTION baseline — trivially strong on a
static background and trivially beaten on the moving region by anything that moves at
all. On constant-angular-velocity rotation data, the honest null is a *motion* model:
assume the apparent motion seen in the seed continues unchanged. We provide two, both
parameter-free and computed from the seed window only:

- ``linear_extrap``  — constant velocity in pixel space: ``f_T + (h+1)*(f_T - f_{T-1})``.
  Weak on rotation (linear pixel extrapolation ghosts a rotating edge) but a fair null.
- ``rotation_extrap`` — constant angular velocity: estimate the in-plane rotation that
  best maps ``f_{T-1} -> f_T`` (a coarse angle search), then predict horizon ``h`` by
  rotating ``f_T`` by ``(h+1)*theta`` about the image centre. This is the strong null the
  review flagged — the learned predictor must beat it on the moving region to claim it
  is modelling motion rather than "sophisticated copy with rotation".

All tensors are ``(B, C, T, H, W)`` in ``[-1, 1]``; predictions are ``(B, C, n, H, W)``.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _rotate(img: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate ``img`` (B,C,H,W) about its centre by per-sample ``angle`` (B,) radians.

    Edge ('border') padding replicates the (uniform) background, so a centred shape on a
    flat background rotates cleanly. Sign convention is internal-consistent: the same
    helper is used to estimate the angle and to extrapolate it, so multiples compose.
    """
    b, c, h, w = img.shape
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = img.new_zeros(b, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False, padding_mode="border", mode="bilinear")


@torch.no_grad()
def _estimate_angle(f_prev: torch.Tensor, f_cur: torch.Tensor,
                    max_angle: float = 0.8, n_search: int = 65) -> torch.Tensor:
    """Per-clip in-plane rotation (radians) best mapping ``f_prev -> f_cur`` (B,C,H,W).
    Coarse linear search over [-max_angle, max_angle], then a fine refine around the best."""
    B = f_prev.shape[0]

    def search(centre: torch.Tensor, half: float, n: int) -> torch.Tensor:
        offsets = torch.linspace(-half, half, n, device=f_prev.device)
        best_a = centre.clone()
        best_e = f_prev.new_full((B,), float("inf"))
        for off in offsets:
            a = centre + off
            err = (_rotate(f_prev, a) - f_cur).pow(2).flatten(1).mean(dim=1)
            better = err < best_e
            best_e = torch.where(better, err, best_e)
            best_a = torch.where(better, a, best_a)
        return best_a

    zero = f_prev.new_zeros(B)
    coarse = search(zero, max_angle, n_search)
    step = 2 * max_angle / (n_search - 1)
    return search(coarse, step, 9)        # refine within one coarse bin


@torch.no_grad()
def persistence(seed: torch.Tensor, n: int) -> torch.Tensor:
    return seed[:, :, -1:].repeat(1, 1, n, 1, 1)


@torch.no_grad()
def linear_extrap(seed: torch.Tensor, n: int) -> torch.Tensor:
    """Constant pixel velocity from the last two seed frames, extrapolated h steps."""
    f_cur, f_prev = seed[:, :, -1], seed[:, :, -2]
    vel = f_cur - f_prev
    steps = torch.arange(1, n + 1, device=seed.device).view(1, 1, n, 1, 1)
    return (f_cur[:, :, None] + steps * vel[:, :, None]).clamp(-1, 1)


@torch.no_grad()
def rotation_extrap(seed: torch.Tensor, n: int,
                    max_angle: float = 0.8, n_search: int = 65) -> torch.Tensor:
    """Constant angular velocity: rotate the last seed frame by (h+1)*theta."""
    f_cur, f_prev = seed[:, :, -1], seed[:, :, -2]
    theta = _estimate_angle(f_prev, f_cur, max_angle, n_search)        # (B,)
    outs = [_rotate(f_cur, (h + 1) * theta) for h in range(n)]
    return torch.stack(outs, dim=2).clamp(-1, 1)


@torch.no_grad()
def motion_baselines(seed: torch.Tensor, n: int) -> Dict[str, torch.Tensor]:
    """All training-free baselines as a name -> (B,C,n,H,W) dict."""
    return {
        "persist": persistence(seed, n),
        "linear": linear_extrap(seed, n),
        "rotation": rotation_extrap(seed, n),
    }
