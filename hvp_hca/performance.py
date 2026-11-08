"""Explicit, tested performance adaptations for Path B models."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MixedPrecisionRMSNorm(nn.Module):
    """RMSNorm with an fp32 master weight and an input-dtype kernel operand.

    PyTorch's CUDA fused RMSNorm requires the input and affine weight to share a
    dtype.  Under bf16 autocast, ``nn.RMSNorm`` otherwise emits a warning and
    dispatches a slower fallback.  The cast remains differentiable, so the
    optimizer still updates the fp32 parameter.
    """

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float | None = None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, dtype=torch.float32))

    @classmethod
    def from_module(cls, source: nn.RMSNorm) -> "MixedPrecisionRMSNorm":
        target = cls(source.normalized_shape, source.eps)
        target.weight = source.weight
        return target

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight if self.weight.dtype == value.dtype else self.weight.to(value.dtype)
        return F.rms_norm(value, self.normalized_shape, weight, self.eps)


def enable_mixed_precision_rmsnorm(model: nn.Module) -> int:
    """Replace attention q/k RMSNorm modules and return the replacement count."""

    replaced = 0
    for module in model.modules():
        for name in ("q_norm", "k_norm"):
            child = getattr(module, name, None)
            if isinstance(child, nn.RMSNorm):
                setattr(module, name, MixedPrecisionRMSNorm.from_module(child))
                replaced += 1
    return replaced
