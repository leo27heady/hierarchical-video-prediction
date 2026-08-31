"""Fixed-width hierarchical state and coordinate-aware position contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def _check_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")


@dataclass(frozen=True)
class TokenMetadata:
    """Physical token identity, independent of its packed tensor slot."""

    xy: Tensor
    time: Tensor
    scale: Tensor
    level: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[-1] != 2:
            raise ValueError("xy must have shape [batch, tokens, 2]")
        batch, tokens, _ = self.xy.shape
        _check_shape("time", self.time, (batch, tokens))
        _check_shape("scale", self.scale, (batch, tokens, 2))
        _check_shape("level", self.level, (batch, tokens))
        _check_shape("valid", self.valid, (batch, tokens))
        if self.level.dtype not in {torch.int32, torch.int64}:
            raise ValueError("level must be an integer tensor")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        devices = {item.device for item in self.tensors()}
        if len(devices) != 1:
            raise ValueError("all metadata tensors must share one device")

    @property
    def batch_size(self) -> int:
        return self.xy.shape[0]

    @property
    def token_count(self) -> int:
        return self.xy.shape[1]

    def tensors(self) -> tuple[Tensor, ...]:
        return self.xy, self.time, self.scale, self.level, self.valid

    def gather(self, indices: Tensor) -> "TokenMetadata":
        if indices.ndim != 2 or indices.shape[0] != self.batch_size:
            raise ValueError("indices must have shape [batch, selected_tokens]")
        pair_index = indices.unsqueeze(-1).expand(-1, -1, 2)
        return TokenMetadata(
            xy=self.xy.gather(1, pair_index),
            time=self.time.gather(1, indices),
            scale=self.scale.gather(1, pair_index),
            level=self.level.gather(1, indices),
            valid=self.valid.gather(1, indices),
        )

    def concatenate(self, other: "TokenMetadata") -> "TokenMetadata":
        if self.batch_size != other.batch_size:
            raise ValueError("metadata batches must match")
        return TokenMetadata(
            xy=torch.cat((self.xy, other.xy), dim=1),
            time=torch.cat((self.time, other.time), dim=1),
            scale=torch.cat((self.scale, other.scale), dim=1),
            level=torch.cat((self.level, other.level), dim=1),
            valid=torch.cat((self.valid, other.valid), dim=1),
        )


@dataclass(frozen=True)
class HierarchicalState:
    """Coarse coverage plus a fixed quota of routed detail tokens."""

    coarse_tokens: Tensor
    detail_tokens: Tensor
    metadata: TokenMetadata
    route_indices: Tensor
    route_scores: Tensor

    def __post_init__(self) -> None:
        if self.coarse_tokens.ndim != 3 or self.detail_tokens.ndim != 3:
            raise ValueError("token tensors must have shape [batch, tokens, width]")
        batch, coarse_count, width = self.coarse_tokens.shape
        detail_batch, detail_count, detail_width = self.detail_tokens.shape
        if (detail_batch, detail_width) != (batch, width):
            raise ValueError("coarse and detail token batch/width must match")
        if self.metadata.batch_size != batch:
            raise ValueError("metadata batch does not match tokens")
        if self.metadata.token_count != coarse_count + detail_count:
            raise ValueError("metadata must describe coarse then detail tokens")
        _check_shape("route_indices", self.route_indices, (batch, detail_count))
        _check_shape("route_scores", self.route_scores, (batch, detail_count))
        if self.route_indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("route_indices must be integer")

    @property
    def detail_quota(self) -> int:
        return self.detail_tokens.shape[1]

    @property
    def packed_tokens(self) -> Tensor:
        return torch.cat((self.coarse_tokens, self.detail_tokens), dim=1)

    def payload_bytes(self, include_metadata: bool = True) -> int:
        tensors = [
            self.coarse_tokens,
            self.detail_tokens,
            self.route_indices,
            self.route_scores,
        ]
        if include_metadata:
            tensors.extend(self.metadata.tensors())
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def pack_fixed_quota(
    *,
    coarse_tokens: Tensor,
    coarse_metadata: TokenMetadata,
    fine_tokens: Tensor,
    fine_metadata: TokenMetadata,
    fine_scores: Tensor,
    detail_quota: int,
) -> HierarchicalState:
    """Select exactly ``detail_quota`` detail tokens and preserve their identity."""

    if fine_tokens.ndim != 3:
        raise ValueError("fine_tokens must have shape [batch, tokens, width]")
    batch, fine_count, width = fine_tokens.shape
    _check_shape("fine_scores", fine_scores, (batch, fine_count))
    if fine_metadata.batch_size != batch or fine_metadata.token_count != fine_count:
        raise ValueError("fine metadata does not match fine tokens")
    if coarse_tokens.shape[0] != batch or coarse_tokens.shape[2] != width:
        raise ValueError("coarse and fine token batch/width must match")
    if coarse_metadata.batch_size != batch:
        raise ValueError("coarse metadata batch does not match")
    if coarse_metadata.token_count != coarse_tokens.shape[1]:
        raise ValueError("coarse metadata does not match coarse tokens")
    if not 0 < detail_quota <= fine_count:
        raise ValueError("detail_quota must be in [1, fine_token_count]")

    selected_scores, selected_indices = torch.topk(
        fine_scores, k=detail_quota, dim=1, largest=True, sorted=True
    )
    token_index = selected_indices.unsqueeze(-1).expand(-1, -1, width)
    selected_tokens = fine_tokens.gather(1, token_index)
    selected_metadata = fine_metadata.gather(selected_indices)
    return HierarchicalState(
        coarse_tokens=coarse_tokens,
        detail_tokens=selected_tokens,
        metadata=coarse_metadata.concatenate(selected_metadata),
        route_indices=selected_indices,
        route_scores=selected_scores,
    )


class CoordinateFourierEncoding(nn.Module):
    """Position tokens by physical coordinates, age, scale, and level.

    No packed-slot index is used.  Consequently, moving a region to another tensor
    slot and moving its metadata with it only permutes the returned embeddings.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_frequencies: int = 4,
        max_levels: int = 8,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_frequencies <= 0 or max_levels <= 0:
            raise ValueError("encoding dimensions must be positive")
        feature_width = 5 * num_frequencies * 2
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.projection = nn.Linear(feature_width, d_model, bias=False)
        self.level_embedding = nn.Embedding(max_levels, d_model)

    def forward(self, metadata: TokenMetadata) -> Tensor:
        if torch.any(metadata.level < 0):
            raise ValueError("level indices must be non-negative")
        if torch.any(metadata.level >= self.level_embedding.num_embeddings):
            raise ValueError("level index exceeds max_levels")
        coordinates = torch.cat(
            (metadata.xy, metadata.time.unsqueeze(-1), metadata.scale), dim=-1
        )
        angles = coordinates.unsqueeze(-1) * self.frequencies * math.pi
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        features = features.flatten(start_dim=-2)
        encoded = self.projection(features) + self.level_embedding(metadata.level)
        return encoded * metadata.valid.unsqueeze(-1)
