"""Reversible second-order rollout for arbitrary fixed-width video transitions.

For a transition ``F`` and state pair ``(p, q)`` the update is

    p' = q
    q' = p + F(q).

It is inverted by ``q = p'`` and ``p = q' - F(p')``.  Starting from
``p=0`` makes the first prediction exactly ``F(q)``; no quality-changing
approximation is introduced at the one-step OpenSTL comparison boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _ReversibleRolloutFunction(Function):
    @staticmethod
    def forward(ctx, previous: torch.Tensor, current: torch.Tensor,
                steps: int, transition: nn.Module):
        with torch.no_grad():
            for _ in range(steps):
                previous, current = current, previous + transition(current)
        ctx.save_for_backward(previous.detach(), current.detach())
        ctx.steps = steps
        ctx.transition = transition
        return previous.detach(), current.detach()

    @staticmethod
    def backward(ctx, d_previous: torch.Tensor, d_current: torch.Tensor):
        previous, current = ctx.saved_tensors
        for _ in reversed(range(ctx.steps)):
            old_current = previous.detach().requires_grad_(True)
            with torch.enable_grad():
                future_delta = ctx.transition(old_current)
                torch.autograd.backward(future_delta, d_current)
            with torch.no_grad():
                old_previous = current - future_delta
                d_old_previous = d_current
                d_old_current = d_previous + old_current.grad
            previous, current = old_previous.detach(), old_current.detach()
            d_previous, d_current = d_old_previous, d_old_current
        return d_previous, d_current, None, None


class ReversibleSecondOrderRollout(nn.Module):
    """Activation-constant rollout around a deterministic transition module."""

    def __init__(self, transition: nn.Module):
        super().__init__()
        self.transition = transition

    def forward(self, current: torch.Tensor, steps: int,
                previous: torch.Tensor | None = None,
                reversible: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if previous is None:
            previous = torch.zeros_like(current)
        if previous.shape != current.shape:
            raise ValueError("previous and current state shapes differ")
        if reversible:
            return _ReversibleRolloutFunction.apply(previous, current, steps, self.transition)
        for _ in range(steps):
            previous, current = current, previous + self.transition(current)
        return previous, current

    @torch.no_grad()
    def inverse(self, previous: torch.Tensor, current: torch.Tensor,
                steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(steps):
            previous, current = current - self.transition(previous), previous
        return previous, current
