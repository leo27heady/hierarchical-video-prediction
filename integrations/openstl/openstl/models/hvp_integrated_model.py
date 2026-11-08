"""Hierarchical, dynamically routed, internally reversible video predictor.

The model keeps a dense coarse description of every observed frame and gathers a
real fixed quota of fine tokens across space and time.  Packed slots have no
positional identity: every token is positioned by physical coordinates, time,
scale, and hierarchy level.  Routing is deliberately outside the reversible
transformer stack.

The reversible execution path changes only activation storage.  It evaluates the
same additive-coupling blocks as ordinary autograd and is therefore a legitimate
same-weights memory comparator.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Function


ROUTING_MODES = {"dense", "static", "density", "hybrid", "mixed"}
HIERARCHY_MODES = {"flat", "hierarchical", "coarse_only"}


@dataclass(frozen=True)
class PhysicalTokenMetadata:
    """Physical token identity independent of tensor slot."""

    xy: Tensor
    time: Tensor
    scale: Tensor
    level: Tensor

    def __post_init__(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[-1] != 2:
            raise ValueError("xy must have shape [B,N,2]")
        batch, tokens, _ = self.xy.shape
        if self.time.shape != (batch, tokens):
            raise ValueError("time must have shape [B,N]")
        if self.scale.shape != (batch, tokens, 2):
            raise ValueError("scale must have shape [B,N,2]")
        if self.level.shape != (batch, tokens) or self.level.dtype != torch.long:
            raise ValueError("level must be int64 with shape [B,N]")
        if len({item.device for item in self.tensors()}) != 1:
            raise ValueError("metadata tensors must share a device")

    @property
    def batch_size(self) -> int:
        return self.xy.shape[0]

    @property
    def token_count(self) -> int:
        return self.xy.shape[1]

    def tensors(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.xy, self.time, self.scale, self.level

    def gather(self, indices: Tensor) -> "PhysicalTokenMetadata":
        if indices.ndim != 2 or indices.shape[0] != self.batch_size:
            raise ValueError("indices must have shape [B,K]")
        pair = indices.unsqueeze(-1).expand(-1, -1, 2)
        return PhysicalTokenMetadata(
            xy=self.xy.gather(1, pair),
            time=self.time.gather(1, indices),
            scale=self.scale.gather(1, pair),
            level=self.level.gather(1, indices),
        )

    def concatenate(self, other: "PhysicalTokenMetadata") -> "PhysicalTokenMetadata":
        if self.batch_size != other.batch_size:
            raise ValueError("metadata batches differ")
        return PhysicalTokenMetadata(
            xy=torch.cat((self.xy, other.xy), dim=1),
            time=torch.cat((self.time, other.time), dim=1),
            scale=torch.cat((self.scale, other.scale), dim=1),
            level=torch.cat((self.level, other.level), dim=1),
        )

    def payload_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.tensors())


@dataclass
class PackedContext:
    tokens: Tensor
    metadata: PhysicalTokenMetadata
    route_indices: Tensor
    route_scores: Tensor
    diagnostics: Dict[str, Tensor]
    coarse_grid: Tensor

    def payload_bytes(self) -> int:
        values = self.tokens.numel() * self.tokens.element_size()
        indices = self.route_indices.numel() * self.route_indices.element_size()
        scores = self.route_scores.numel() * self.route_scores.element_size()
        return values + indices + scores + self.metadata.payload_bytes()


def _grid_metadata(
    batch: int,
    frames: int,
    height: int,
    width: int,
    *,
    level: int,
    future: bool,
    time_normalizer: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PhysicalTokenMetadata:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    spatial = torch.stack((xx, yy), dim=-1).reshape(1, 1, height * width, 2)
    if future:
        times = torch.arange(1, frames + 1, device=device, dtype=dtype)
    else:
        times = torch.arange(-(frames - 1), 1, device=device, dtype=dtype)
    times = times / max(int(time_normalizer), 1)
    spatial = spatial.expand(batch, frames, -1, -1)
    time = times[None, :, None].expand(batch, -1, height * width)
    scale = torch.tensor((1.0 / width, 1.0 / height), device=device, dtype=dtype)
    scale = scale.reshape(1, 1, 1, 2).expand(batch, frames, height * width, 2)
    levels = torch.full(
        (batch, frames, height * width), level, device=device, dtype=torch.long
    )
    return PhysicalTokenMetadata(
        xy=spatial.reshape(batch, -1, 2),
        time=time.reshape(batch, -1),
        scale=scale.reshape(batch, -1, 2),
        level=levels.reshape(batch, -1),
    )


class CoordinateFourierEncoding(nn.Module):
    """Continuous encoding of x, y, signed time, and cell scale."""

    def __init__(self, d_model: int, num_frequencies: int = 6, max_levels: int = 4):
        super().__init__()
        if d_model <= 0 or num_frequencies <= 0:
            raise ValueError("encoding dimensions must be positive")
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.projection = nn.Linear(5 * 2 * num_frequencies, d_model, bias=False)
        self.level_embedding = nn.Embedding(max_levels, d_model)

    def forward(self, metadata: PhysicalTokenMetadata) -> Tensor:
        if torch.any(metadata.level < 0) or torch.any(
            metadata.level >= self.level_embedding.num_embeddings
        ):
            raise ValueError("metadata level outside configured range")
        coordinates = torch.cat(
            (metadata.xy, metadata.time.unsqueeze(-1), metadata.scale), dim=-1
        )
        frequencies = self.frequencies.to(dtype=coordinates.dtype)
        angles = coordinates.unsqueeze(-1) * frequencies * math.pi
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return self.projection(features) + self.level_embedding(metadata.level)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class FramePyramidEncoder(nn.Module):
    """Shared stride-four fine and stride-eight coarse frame encoder."""

    def __init__(self, channels: int, d_model: int, stem_width: int):
        super().__init__()
        if stem_width <= 0:
            raise ValueError("stem_width must be positive")
        middle = max(stem_width, d_model // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, stem_width, 3, padding=1),
            nn.SiLU(),
            ResidualConvBlock(stem_width),
        )
        self.to_half = nn.Sequential(
            nn.Conv2d(stem_width, middle, 4, stride=2, padding=1), nn.SiLU()
        )
        self.to_fine = nn.Sequential(
            nn.Conv2d(middle, d_model, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(d_model),
        )
        self.to_coarse = nn.Sequential(
            nn.Conv2d(d_model, d_model, 4, stride=2, padding=1),
            nn.SiLU(),
            ResidualConvBlock(d_model),
        )

    def forward(self, frames: Tensor) -> Tuple[Tensor, Tensor]:
        if frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,C,H,W]")
        batch, time, channels, height, width = frames.shape
        if height % 8 or width % 8:
            raise ValueError("height and width must be divisible by eight")
        x = frames.reshape(batch * time, channels, height, width)
        fine = self.to_fine(self.to_half(self.stem(x)))
        coarse = self.to_coarse(fine)
        fine = fine.reshape(batch, time, *fine.shape[1:])
        coarse = coarse.reshape(batch, time, *coarse.shape[1:])
        return fine, coarse


def _standardize(values: Tensor, epsilon: float = 1e-6) -> Tensor:
    mean = values.mean(dim=1, keepdim=True)
    std = values.std(dim=1, keepdim=True, unbiased=False).clamp_min(epsilon)
    return (values - mean) / std


class SpaceTimeRouter(nn.Module):
    """Content-dependent fixed-quota selection over the fine space--time volume."""

    def __init__(
        self,
        d_model: int,
        *,
        density_weight: float = 1.0,
        change_weight: float = 1.0,
        learned_weight: float = 0.5,
        min_detail_per_frame: int = 1,
        mixed_dynamic_probability: float = 0.75,
    ) -> None:
        super().__init__()
        hidden = max(d_model // 2, 16)
        self.learned_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.score_embedding = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, d_model)
        )
        self.density_weight = float(density_weight)
        self.change_weight = float(change_weight)
        self.learned_weight = float(learned_weight)
        self.min_detail_per_frame = int(min_detail_per_frame)
        self.mixed_dynamic_probability = float(mixed_dynamic_probability)
        if self.min_detail_per_frame < 0:
            raise ValueError("min_detail_per_frame must be non-negative")
        if not 0.0 <= self.mixed_dynamic_probability <= 1.0:
            raise ValueError("mixed_dynamic_probability must be in [0,1]")

    @staticmethod
    def _static_indices(batch: int, candidates: int, quota: int, device) -> Tensor:
        # Midpoints of equal intervals provide deterministic coverage without
        # duplicated indices for every quota <= candidates.
        positions = torch.floor(
            (torch.arange(quota, device=device, dtype=torch.float64) + 0.5)
            * candidates
            / quota
        ).long()
        return positions.unsqueeze(0).expand(batch, -1)

    def _dynamic_indices(self, scores: Tensor, frames: int, cells: int, quota: int) -> Tensor:
        batch = scores.shape[0]
        if quota == frames * cells:
            return torch.arange(quota, device=scores.device).unsqueeze(0).expand(batch, -1)
        minimum = self.min_detail_per_frame
        chosen = []
        available = scores.clone()
        if minimum:
            per_frame = scores.reshape(batch, frames, cells)
            local = torch.argsort(per_frame, dim=-1, descending=True, stable=True)[..., :minimum]
            offsets = torch.arange(frames, device=scores.device).reshape(1, frames, 1) * cells
            guaranteed = (local + offsets).reshape(batch, -1)
            chosen.append(guaranteed)
            available.scatter_(1, guaranteed, -torch.inf)
        remaining = quota - minimum * frames
        if remaining:
            chosen.append(torch.argsort(available, dim=1, descending=True, stable=True)[:, :remaining])
        return torch.cat(chosen, dim=1).sort(dim=1).values

    def forward(
        self,
        fine: Tensor,
        coarse: Tensor,
        metadata: PhysicalTokenMetadata,
        quota: int,
        mode: str,
    ) -> Tuple[Tensor, PhysicalTokenMetadata, Tensor, Tensor, Dict[str, Tensor], str]:
        if mode not in ROUTING_MODES:
            raise ValueError(f"unknown routing mode: {mode}")
        batch, frames, width, fine_h, fine_w = fine.shape
        candidates = frames * fine_h * fine_w
        if not 0 < quota <= candidates:
            raise ValueError(f"detail quota {quota} outside [1,{candidates}]")
        if mode != "dense" and (self.min_detail_per_frame > fine_h * fine_w
                                or quota < frames * self.min_detail_per_frame):
            raise ValueError("infeasible minimum detail quota per frame")
        if metadata.batch_size != batch or metadata.token_count != candidates:
            raise ValueError("fine metadata does not match candidates")

        fine_flat = fine.permute(0, 1, 3, 4, 2).reshape(batch, candidates, width)
        coarse_up = F.interpolate(
            coarse.reshape(batch * frames, width, *coarse.shape[-2:]),
            size=(fine_h, fine_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, width, fine_h, fine_w)
        residual = (fine - coarse_up).square().mean(dim=2).reshape(batch, candidates)
        previous = torch.cat((fine[:, :1], fine[:, :-1]), dim=1)
        change = (fine - previous).abs().mean(dim=2).reshape(batch, candidates)
        learned = self.learned_score(fine_flat).squeeze(-1)
        density_score = self.density_weight * _standardize(residual)
        change_score = self.change_weight * _standardize(change)
        hybrid_score = density_score + change_score + self.learned_weight * _standardize(learned)

        resolved_mode = mode
        if mode == "mixed":
            # Evaluation resolves to the intended dynamic policy.  During
            # training, static batches deliberately teach route-membership
            # robustness without silently disabling routing at validation.
            use_dynamic = True if not self.training else bool(
                torch.rand((), device=fine.device) < self.mixed_dynamic_probability
            )
            resolved_mode = "hybrid" if use_dynamic else "static"
        score = density_score + change_score if resolved_mode == "density" else hybrid_score
        if resolved_mode == "dense":
            quota = candidates
            indices = torch.arange(candidates, device=fine.device).unsqueeze(0).expand(batch, -1)
        elif resolved_mode == "static":
            indices = self._static_indices(batch, candidates, quota, fine.device)
        else:
            indices = self._dynamic_indices(score, frames, fine_h * fine_w, quota)

        gather_index = indices.unsqueeze(-1).expand(-1, -1, width)
        selected = fine_flat.gather(1, gather_index)
        selected_scores = score.gather(1, indices)
        # All membership policies receive identical hybrid-score features.
        # Density changes ranking only; static no longer removes a learned path.
        feature_scores = hybrid_score.gather(1, indices)
        selected = selected + self.score_embedding(feature_scores.unsqueeze(-1))

        probability = torch.softmax(score, dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy = entropy / math.log(max(candidates, 2))
        frame_ids = torch.div(indices, fine_h * fine_w, rounding_mode="floor")
        histogram = torch.zeros(batch, frames, device=fine.device, dtype=fine.dtype)
        histogram.scatter_add_(1, frame_ids, torch.ones_like(frame_ids, dtype=fine.dtype))
        spatial_ids = indices.remainder(fine_h * fine_w)
        occupancy = torch.zeros(
            batch, frames, fine_h * fine_w, device=fine.device, dtype=torch.bool
        )
        occupancy.reshape(batch, -1).scatter_(1, indices, True)
        if frames > 1:
            intersection = (occupancy[:, 1:] & occupancy[:, :-1]).sum(dim=-1).float()
            union = (occupancy[:, 1:] | occupancy[:, :-1]).sum(dim=-1).float()
            churn = torch.where(union > 0, 1.0 - intersection / union, torch.zeros_like(union))
            route_churn = churn.mean()
        else:
            route_churn = torch.zeros((), device=fine.device, dtype=fine.dtype)
        unique_spatial = torch.zeros(
            batch, fine_h * fine_w, device=fine.device, dtype=torch.bool
        )
        unique_spatial.scatter_(1, spatial_ids, True)
        sorted_scores = score.sort(dim=1, descending=True).values
        if quota < candidates:
            margin = sorted_scores[:, quota - 1] - sorted_scores[:, quota]
        else:
            margin = torch.full((batch,), float("inf"), device=fine.device, dtype=fine.dtype)
        diagnostics = {
            "route_entropy": entropy.mean(),
            "route_temporal_coverage": (histogram > 0).float().mean(),
            "route_global_unconstrained_score_gap": margin.mean(),
            "route_selected": torch.tensor(float(quota), device=fine.device),
            "route_candidates": torch.tensor(float(candidates), device=fine.device),
            "route_frame_min": histogram.min(),
            "route_frame_max": histogram.max(),
            "route_frame_std": histogram.std(dim=1, unbiased=False).mean(),
            "route_churn": route_churn,
            "route_spatial_coverage": unique_spatial.float().mean(),
        }
        return (
            selected,
            metadata.gather(indices),
            indices,
            selected_scores,
            diagnostics,
            resolved_mode,
        )


class ReversibleSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("reversible half-width must be divisible by heads")
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.heads = heads
        self.head_width = width // heads

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = [item.transpose(1, 2) for item in (query, key, value)]
        output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.output(output.transpose(1, 2).reshape(batch, tokens, width))


class ReversibleFeedForward(nn.Module):
    def __init__(self, width: int, ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.net = nn.Sequential(
            nn.Linear(width, ratio * width),
            nn.GELU(approximate="tanh"),
            nn.Linear(ratio * width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))


def _autocast_context(device_type: str, enabled: bool, dtype: Optional[torch.dtype]):
    if enabled and device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)
    return nullcontext()


class ReversibleTransformerBlock(nn.Module):
    """Additive coupling with exactly reconstructable inputs."""

    def __init__(self, d_model: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        half = d_model // 2
        self.f = ReversibleSelfAttention(half, heads)
        self.g = ReversibleFeedForward(half, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat((y1, y2), dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        y1, y2 = y.chunk(2, dim=-1)
        x2 = y2 - self.g(y1)
        x1 = y1 - self.f(x2)
        return torch.cat((x1, x2), dim=-1)

    def backward_pass(
        self,
        y: Tensor,
        dy: Tensor,
        *,
        autocast_enabled: bool,
        autocast_dtype: Optional[torch.dtype],
    ) -> Tuple[Tensor, Tensor]:
        y1, y2 = y.chunk(2, dim=-1)
        dy1, dy2 = dy.chunk(2, dim=-1)
        with torch.enable_grad(), _autocast_context(y.device.type, autocast_enabled, autocast_dtype):
            y1_leaf = y1.detach().requires_grad_(True)
            g_y1 = self.g(y1_leaf)
            torch.autograd.backward(g_y1, dy2)
        with torch.no_grad():
            x2 = y2 - g_y1
            d_y1 = dy1 + y1_leaf.grad
        with torch.enable_grad(), _autocast_context(y.device.type, autocast_enabled, autocast_dtype):
            x2_leaf = x2.detach().requires_grad_(True)
            f_x2 = self.f(x2_leaf)
            torch.autograd.backward(f_x2, d_y1)
        with torch.no_grad():
            x1 = y1 - f_x2
            d_x2 = dy2 + x2_leaf.grad
        return torch.cat((x1.detach(), x2.detach()), dim=-1), torch.cat((d_y1, d_x2), dim=-1)


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:  # PyTorch < 2.4
        return bool(torch.is_autocast_enabled()) if device_type == "cuda" else False


def _autocast_dtype(device_type: str) -> Optional[torch.dtype]:
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):
        if device_type == "cuda":
            return torch.get_autocast_gpu_dtype()
        if device_type == "cpu":
            return torch.get_autocast_cpu_dtype()
        return None


class _ReversibleStackFunction(Function):
    @staticmethod
    def forward(ctx, x: Tensor, blocks: nn.ModuleList) -> Tensor:
        ctx.autocast_enabled = _is_autocast_enabled(x.device.type)
        ctx.autocast_dtype = _autocast_dtype(x.device.type)
        ctx.blocks = blocks
        with torch.no_grad(), _autocast_context(
            x.device.type, ctx.autocast_enabled, ctx.autocast_dtype
        ):
            for block in blocks:
                x = block(x)
        ctx.save_for_backward(x.detach())
        return x.detach()

    @staticmethod
    def backward(ctx, dy: Tensor):
        (state,) = ctx.saved_tensors
        for block in reversed(ctx.blocks):
            state, dy = block.backward_pass(
                state,
                dy,
                autocast_enabled=ctx.autocast_enabled,
                autocast_dtype=ctx.autocast_dtype,
            )
        return dy, None


class ReversibleTransformerCore(nn.Module):
    def __init__(self, d_model: int, depth: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        if depth <= 0:
            raise ValueError("core depth must be positive")
        self.blocks = nn.ModuleList(
            [ReversibleTransformerBlock(d_model, heads, mlp_ratio) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, reversible: bool) -> Tensor:
        if getattr(self, "activation_checkpointing", False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, use_reentrant=False)
        elif reversible and torch.is_grad_enabled():
            x = _ReversibleStackFunction.apply(x, self.blocks)
        else:
            for block in self.blocks:
                x = block(x)
        return self.output_norm(x)

    @torch.no_grad()
    def inverse_blocks(self, y: Tensor) -> Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key_value = nn.Linear(d_model, 2 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.heads = heads
        self.head_width = d_model // heads

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        batch, query_tokens, width = query.shape
        memory_tokens = memory.shape[1]
        q = self.query(self.query_norm(query)).reshape(
            batch, query_tokens, self.heads, self.head_width
        ).transpose(1, 2)
        kv = self.key_value(self.memory_norm(memory)).reshape(
            batch, memory_tokens, 2, self.heads, self.head_width
        )
        key, value = kv.unbind(dim=2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, key, value, dropout_p=0.0)
        return self.output(output.transpose(1, 2).reshape(batch, query_tokens, width))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.cross = CrossAttention(d_model, heads)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * d_model, d_model),
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        query = query + self.cross(query, memory)
        return query + self.mlp(self.norm(query))


class FutureFrameDecoder(nn.Module):
    def __init__(self, channels: int, d_model: int, heads: int, depth: int):
        super().__init__()
        self.query_seed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.last_projection = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, heads) for _ in range(depth)])
        widths = [d_model, max(d_model // 2, 32), max(d_model // 4, 24), max(d_model // 8, 16)]
        stages = []
        for source, target in zip(widths[:-1], widths[1:]):
            stages.extend(
                [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                 nn.Conv2d(source, target, 3, padding=1), nn.SiLU()]
            )
        self.upsampler = nn.Sequential(*stages)
        self.output = nn.Conv2d(widths[-1], channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        memory: Tensor,
        last_coarse: Tensor,
        query_position: Tensor,
        last_frame: Tensor,
        output_frames: int,
    ) -> Tensor:
        batch, width, coarse_h, coarse_w = last_coarse.shape
        spatial_tokens = coarse_h * coarse_w
        last_tokens = last_coarse.flatten(2).transpose(1, 2)
        query = self.last_projection(last_tokens)[:, None].expand(
            -1, output_frames, -1, -1
        ).reshape(batch, output_frames * spatial_tokens, width)
        query = query + query_position + self.query_seed
        for block in self.blocks:
            query = block(query, memory)
        features = query.reshape(batch, output_frames, coarse_h, coarse_w, width)
        features = features.permute(0, 1, 4, 2, 3).reshape(
            batch * output_frames, width, coarse_h, coarse_w
        )
        residual = self.output(self.upsampler(features))
        residual = residual.reshape(batch, output_frames, *residual.shape[1:])
        return last_frame[:, -1:].expand(-1, output_frames, -1, -1, -1) + residual


class HVPIntegrated_Model(nn.Module):
    """OpenSTL-native hierarchical video predictor."""

    def __init__(
        self,
        in_shape,
        aft_seq_length: Optional[int] = None,
        d_model: int = 128,
        stem_width: int = 48,
        core_depth: int = 6,
        decoder_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        detail_quota: int = 160,
        hierarchy_mode: str = "hierarchical",
        routing_mode: str = "hybrid",
        reversible_core: bool = True,
        position_frequencies: int = 6,
        min_detail_per_frame: int = 1,
        mixed_dynamic_probability: float = 0.75,
        density_weight: float = 1.0,
        change_weight: float = 1.0,
        learned_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        input_frames, channels, height, width = [int(value) for value in in_shape]
        self.input_shape = (input_frames, channels, height, width)
        self.output_frames = int(aft_seq_length or input_frames)
        self.d_model = int(d_model)
        self.detail_quota = int(detail_quota)
        self.hierarchy_mode = str(hierarchy_mode).lower()
        self.routing_mode = str(routing_mode).lower()
        self.reversible_core = bool(reversible_core)
        if self.hierarchy_mode not in HIERARCHY_MODES:
            raise ValueError(f"unknown hierarchy mode: {self.hierarchy_mode}")
        if self.routing_mode not in ROUTING_MODES:
            raise ValueError(f"unknown routing mode: {self.routing_mode}")
        if d_model % 2 or (d_model // 2) % num_heads:
            raise ValueError("d_model/2 must be divisible by num_heads")
        if height % 8 or width % 8:
            raise ValueError("input spatial dimensions must be divisible by eight")

        self.encoder = FramePyramidEncoder(channels, d_model, stem_width)
        self.router = SpaceTimeRouter(
            d_model,
            density_weight=density_weight,
            change_weight=change_weight,
            learned_weight=learned_weight,
            min_detail_per_frame=min_detail_per_frame,
            mixed_dynamic_probability=mixed_dynamic_probability,
        )
        self.position = CoordinateFourierEncoding(d_model, position_frequencies)
        self.core = ReversibleTransformerCore(d_model, core_depth, num_heads, mlp_ratio)
        self.decoder = FutureFrameDecoder(channels, d_model, num_heads, decoder_depth)
        self.last_route_mode = self.routing_mode
        self.last_diagnostics: Dict[str, Tensor] = {}

    def set_routing_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in ROUTING_MODES:
            raise ValueError(f"unknown routing mode: {mode}")
        self.routing_mode = mode

    def set_reversible_core(self, enabled: bool) -> None:
        self.reversible_core = bool(enabled)

    def _pack(
        self, fine: Tensor, coarse: Tensor, routing_mode: str
    ) -> PackedContext:
        batch, frames, width, fine_h, fine_w = fine.shape
        coarse_h, coarse_w = coarse.shape[-2:]
        time_normalizer = max(frames, self.output_frames)
        fine_metadata = _grid_metadata(
            batch, frames, fine_h, fine_w,
            level=1, future=False, time_normalizer=time_normalizer,
            device=fine.device, dtype=fine.dtype,
        )
        coarse_metadata = _grid_metadata(
            batch, frames, coarse_h, coarse_w,
            level=0, future=False, time_normalizer=time_normalizer,
            device=coarse.device, dtype=coarse.dtype,
        )
        fine_tokens = fine.permute(0, 1, 3, 4, 2).reshape(batch, -1, width)
        coarse_tokens = coarse.permute(0, 1, 3, 4, 2).reshape(batch, -1, width)
        candidates = fine_tokens.shape[1]

        if self.hierarchy_mode == "flat":
            indices = torch.arange(candidates, device=fine.device).unsqueeze(0).expand(batch, -1)
            selected = fine_tokens
            metadata = fine_metadata
            scores = torch.zeros(batch, candidates, device=fine.device, dtype=fine.dtype)
            diagnostics = {
                "route_entropy": torch.ones((), device=fine.device, dtype=fine.dtype),
                "route_temporal_coverage": torch.ones((), device=fine.device, dtype=fine.dtype),
                "route_global_unconstrained_score_gap": torch.full((), float("inf"), device=fine.device, dtype=fine.dtype),
                "route_selected": torch.tensor(float(candidates), device=fine.device),
                "route_candidates": torch.tensor(float(candidates), device=fine.device),
                "route_frame_min": torch.tensor(float(fine_h * fine_w), device=fine.device),
                "route_frame_max": torch.tensor(float(fine_h * fine_w), device=fine.device),
                "route_frame_std": torch.zeros((), device=fine.device, dtype=fine.dtype),
                "route_churn": torch.zeros((), device=fine.device, dtype=fine.dtype),
                "route_spatial_coverage": torch.ones((), device=fine.device, dtype=fine.dtype),
            }
            resolved_mode = "dense"
        elif self.hierarchy_mode == "coarse_only":
            indices = torch.empty(batch, 0, device=fine.device, dtype=torch.long)
            scores = torch.empty(batch, 0, device=fine.device, dtype=fine.dtype)
            selected = fine_tokens[:, :0]
            metadata = fine_metadata.gather(indices)
            zero = torch.zeros((), device=fine.device, dtype=fine.dtype)
            diagnostics = {
                "route_entropy": zero,
                "route_temporal_coverage": zero,
                "route_global_unconstrained_score_gap": zero,
                "route_selected": zero,
                "route_candidates": torch.tensor(float(candidates), device=fine.device),
                "route_frame_min": zero,
                "route_frame_max": zero,
                "route_frame_std": zero,
                "route_churn": zero,
                "route_spatial_coverage": zero,
            }
            resolved_mode = "coarse_only"
        else:
            quota = candidates if routing_mode == "dense" else self.detail_quota
            selected, metadata, indices, scores, diagnostics, resolved_mode = self.router(
                fine, coarse, fine_metadata, quota, routing_mode
            )

        if self.hierarchy_mode == "flat":
            tokens = selected
        else:
            tokens = torch.cat((coarse_tokens, selected), dim=1)
            metadata = coarse_metadata.concatenate(metadata)
        tokens = tokens + self.position(metadata)
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "packed_tokens": torch.tensor(float(tokens.shape[1]), device=tokens.device),
            "fine_candidates": torch.tensor(float(candidates), device=tokens.device),
            "coarse_tokens": torch.tensor(
                float(0 if self.hierarchy_mode == "flat" else coarse_tokens.shape[1]),
                device=tokens.device,
            ),
        })
        self.last_route_mode = resolved_mode
        return PackedContext(tokens, metadata, indices, scores, diagnostics, coarse)

    def encode_and_pack(self, frames: Tensor, routing_mode: Optional[str] = None) -> PackedContext:
        fine, coarse = self.encoder(frames)
        return self._pack(fine, coarse, (routing_mode or self.routing_mode).lower())

    def decode_packed(self, packed: PackedContext, frames: Tensor, reversible: bool) -> Tensor:
        memory = self.core(packed.tokens, reversible=reversible)
        batch, _, _, coarse_h, coarse_w = packed.coarse_grid.shape
        query_metadata = _grid_metadata(
            batch, self.output_frames, coarse_h, coarse_w,
            level=2, future=True,
            time_normalizer=max(frames.shape[1], self.output_frames),
            device=frames.device, dtype=memory.dtype,
        )
        query_position = self.position(query_metadata)
        return self.decoder(
            memory,
            packed.coarse_grid[:, -1],
            query_position,
            frames,
            self.output_frames,
        )

    def forward(
        self,
        x_raw: Tensor,
        *,
        routing_mode: Optional[str] = None,
        reversible_core: Optional[bool] = None,
        return_aux: bool = False,
        **kwargs,
    ):
        del kwargs
        if x_raw.ndim != 5:
            raise ValueError("input must have shape [B,T,C,H,W]")
        expected = self.input_shape
        if tuple(x_raw.shape[1:]) != expected:
            raise ValueError(f"expected input tail {expected}, got {tuple(x_raw.shape[1:])}")
        packed = self.encode_and_pack(x_raw, routing_mode)
        reversible = self.reversible_core if reversible_core is None else bool(reversible_core)
        prediction = self.decode_packed(packed, x_raw, reversible)
        self.last_diagnostics = {key: value.detach() for key, value in packed.diagnostics.items()}
        if return_aux:
            return prediction, {
                **packed.diagnostics,
                "route_indices": packed.route_indices,
                "route_scores": packed.route_scores,
                "packed_payload_bytes": torch.tensor(
                    float(packed.payload_bytes()), device=x_raw.device
                ),
                "resolved_routing_mode": self.last_route_mode,
            }
        return prediction

    def memory_contract(self, batch_size: int = 1, dtype_bytes: int = 4) -> Dict[str, float]:
        frames, _, height, width = self.input_shape
        fine = frames * (height // 4) * (width // 4)
        coarse = frames * (height // 8) * (width // 8)
        if self.hierarchy_mode == "flat":
            packed = fine
            routed = fine
        elif self.hierarchy_mode == "coarse_only":
            packed = coarse
            routed = 0
        else:
            routed = min(self.detail_quota, fine)
            packed = coarse + routed
        # Five float metadata values (x,y,time,sx,sy), one int64 level,
        # plus int64 route index and float route score for routed detail.
        token_bytes = batch_size * packed * self.d_model * dtype_bytes
        metadata_bytes = batch_size * packed * (5 * dtype_bytes + 8)
        route_bytes = batch_size * routed * (8 + dtype_bytes)
        return {
            "fine_candidates": float(fine),
            "coarse_tokens": float(coarse),
            "detail_tokens": float(routed),
            "packed_tokens": float(packed),
            "flat_to_packed_token_ratio": float(fine / packed),
            "analytic_payload_bytes": float(token_bytes + metadata_bytes + route_bytes),
        }


__all__ = [
    "CoordinateFourierEncoding",
    "FramePyramidEncoder",
    "HVPIntegrated_Model",
    "PackedContext",
    "PhysicalTokenMetadata",
    "ReversibleTransformerBlock",
    "ReversibleTransformerCore",
    "SpaceTimeRouter",
]
