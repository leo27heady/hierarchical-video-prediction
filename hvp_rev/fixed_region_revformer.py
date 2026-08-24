"""Fixed-width hierarchical predictor with a TempVerseFormer-style reversible core.

The module is deliberately separate from the frozen RHVD implementation.  It is a
development-only feasibility model with three explicit interfaces:

* each frame becomes 16 coarse tokens and 8 fixed-region fine tokens;
* information density changes the weights *inside* a fixed region, never token
  membership or sequence length;
* only the repeated temporal transformer is reversibly replayed.  The codec,
  tokenizer and regression readout remain ordinary differentiable boundaries.

At 48 frames the context therefore contains exactly 48 * (16 + 8) = 1,152
spatial tokens.  The reversible backward is independent of the number of repeated
temporal steps, apart from the final state which it reconstructs during backward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from hvp_rev.faithful_nets import LAT


@dataclass(frozen=True)
class FixedRegionLayout:
    """Geometry of the stable fine-region partition."""

    coarse_hw: int = 4
    fine_hw: int = 8
    region_rows: int = 2
    region_cols: int = 4

    @property
    def coarse_tokens(self) -> int:
        return self.coarse_hw**2

    @property
    def region_tokens(self) -> int:
        return self.region_rows * self.region_cols

    @property
    def tokens_per_frame(self) -> int:
        return self.coarse_tokens + self.region_tokens

    @property
    def cells_per_region(self) -> int:
        return (self.fine_hw // self.region_rows) * (self.fine_hw // self.region_cols)

    def validate(self) -> None:
        if self.fine_hw % self.region_rows or self.fine_hw % self.region_cols:
            raise ValueError("fine grid must divide evenly into fixed regions")


def fine_to_regions(zf: torch.Tensor, layout: FixedRegionLayout) -> torch.Tensor:
    """Return fixed fine cells as ``(B,T,R,K,C)`` without selecting or dropping cells."""

    layout.validate()
    if zf.ndim != 5 or zf.shape[2:] != (LAT, layout.fine_hw, layout.fine_hw):
        raise ValueError(f"expected (B,T,{LAT},{layout.fine_hw},{layout.fine_hw}), got {tuple(zf.shape)}")
    b, t, c, _, _ = zf.shape
    gh = layout.fine_hw // layout.region_rows
    gw = layout.fine_hw // layout.region_cols
    return (
        zf.reshape(b, t, c, layout.region_rows, gh, layout.region_cols, gw)
        .permute(0, 1, 3, 5, 4, 6, 2)
        .reshape(b, t, layout.region_tokens, layout.cells_per_region, c)
    )


def regions_to_fine(cells: torch.Tensor, layout: FixedRegionLayout) -> torch.Tensor:
    """Inverse geometric rearrangement for ``(B,T,R,K,C)`` regional cell tensors."""

    layout.validate()
    b, t, r, k, c = cells.shape
    if (r, k, c) != (layout.region_tokens, layout.cells_per_region, LAT):
        raise ValueError("regional cell tensor does not match the declared layout")
    gh = layout.fine_hw // layout.region_rows
    gw = layout.fine_hw // layout.region_cols
    return (
        cells.reshape(b, t, layout.region_rows, layout.region_cols, gh, gw, c)
        .permute(0, 1, 6, 2, 4, 3, 5)
        .reshape(b, t, c, layout.fine_hw, layout.fine_hw)
    )


def information_density_summary(zf: torch.Tensor, layout: FixedRegionLayout,
                                temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean plus variance-weighted summary for every immutable fine region."""

    cells = fine_to_regions(zf, layout)
    mean = cells.mean(dim=3)
    density = cells.var(dim=-1, unbiased=False)
    weights = torch.softmax(density / temperature, dim=3)
    weighted = (cells * weights.unsqueeze(-1)).sum(dim=3)
    return torch.cat([mean, weighted], dim=-1), weights


class FixedRegionTokenizer(nn.Module):
    """Stable coarse/fine tokenization with soft information-density pooling.

    Every fine cell contributes.  A temperature-controlled variance score only
    changes its weight inside one of eight immutable spatial regions.  Concatenating
    the regional mean and weighted value avoids the discontinuity and false-positive
    behaviour observed with global hard top-K routing.
    """

    def __init__(self, dim: int, max_frames: int = 64, temperature: float = 0.25,
                 layout: FixedRegionLayout | None = None):
        super().__init__()
        self.layout = layout or FixedRegionLayout()
        self.layout.validate()
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.coarse_projection = nn.Linear(LAT, dim)
        self.region_projection = nn.Linear(2 * LAT, dim)
        self.temporal_position = nn.Parameter(torch.zeros(max_frames, dim))
        self.coarse_position = nn.Parameter(torch.zeros(self.layout.coarse_tokens, dim))
        self.region_position = nn.Parameter(torch.zeros(self.layout.region_tokens, dim))
        self.level_embedding = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.temporal_position, std=0.02)
        nn.init.normal_(self.coarse_position, std=0.02)
        nn.init.normal_(self.region_position, std=0.02)
        nn.init.normal_(self.level_embedding, std=0.02)

    def regional_summary(self, zf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return information_density_summary(zf, self.layout, self.temperature)

    def forward(self, zc: torch.Tensor, zf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if zc.ndim != 5 or zc.shape[2:] != (LAT, self.layout.coarse_hw, self.layout.coarse_hw):
            raise ValueError(f"unexpected coarse shape {tuple(zc.shape)}")
        b, t = zc.shape[:2]
        if t > self.temporal_position.shape[0]:
            raise ValueError("sequence exceeds configured max_frames")
        coarse = zc.flatten(3).transpose(2, 3)
        coarse = self.coarse_projection(coarse)
        coarse = coarse + self.coarse_position[None, None] + self.level_embedding[0]
        regions, weights = self.regional_summary(zf)
        regions = self.region_projection(regions)
        regions = regions + self.region_position[None, None] + self.level_embedding[1]
        pos_t = self.temporal_position[:t][None, :, None]
        return torch.cat([coarse + pos_t, regions + pos_t], dim=2).reshape(
            b, t * self.layout.tokens_per_frame, -1
        ), weights


class FixedRegionReconstructor(nn.Module):
    """Compact, stable 24-token frame state and its fine-latent reconstruction.

    The persisted state is one 16-value vector for each of 16 coarse cells and
    eight fixed fine regions.  Information-density pooling happens once when a
    frame enters memory; the original 64 fine cells can then be released.
    """

    def __init__(self, temperature: float = 0.25,
                 layout: FixedRegionLayout | None = None):
        super().__init__()
        self.layout = layout or FixedRegionLayout()
        self.temperature = float(temperature)
        # Only these compact projected vectors are part of the persistent state.
        self.region_encoder = nn.Sequential(nn.Linear(2 * LAT, 2 * LAT), nn.GELU(), nn.Linear(2 * LAT, LAT))
        self.region_decoder = nn.Sequential(
            nn.Linear(LAT, 4 * LAT), nn.GELU(),
            nn.Linear(4 * LAT, self.layout.cells_per_region * LAT),
        )
        self.coarse_to_fine = nn.ConvTranspose2d(LAT, LAT, kernel_size=2, stride=2)
        # Safe residual gate: an untrained adapter behaves almost exactly like
        # the established coarse decoder instead of replacing it with noise.
        self.blend_logit = nn.Parameter(torch.tensor(-6.0))
        patch_height = 64 // self.layout.region_rows
        patch_width = 64 // self.layout.region_cols
        self.pixel_region_decoder = nn.Sequential(
            nn.Linear(LAT, 8 * LAT), nn.GELU(),
            nn.Linear(8 * LAT, patch_height * patch_width),
        )
        nn.init.zeros_(self.pixel_region_decoder[-1].weight)
        nn.init.zeros_(self.pixel_region_decoder[-1].bias)
        self.pixel_blend_logit = nn.Parameter(torch.tensor(-2.0))

    def pack(self, zc: torch.Tensor, zf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack latents into coarse cells and eight fixed regional tokens."""

        summaries, weights = information_density_summary(zf, self.layout, self.temperature)
        return zc, self.region_encoder(summaries), weights

    def unpack(self, coarse: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full fine grid from the compact state."""

        if coarse.ndim != 5 or regions.ndim != 4:
            raise ValueError("expected coarse (B,T,C,4,4) and regions (B,T,8,C)")
        b, t = coarse.shape[:2]
        cells = self.region_decoder(regions).reshape(
            b, t, self.layout.region_tokens, self.layout.cells_per_region, LAT
        )
        residual = regions_to_fine(cells, self.layout)
        base = self.coarse_to_fine(coarse.reshape(-1, LAT, 4, 4)).reshape(
            b, t, LAT, self.layout.fine_hw, self.layout.fine_hw
        )
        return base + residual

    def forward(self, zc: torch.Tensor, zf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse, regions, weights = self.pack(zc, zf)
        return self.unpack(coarse, regions), regions, weights

    @property
    def values_per_frame(self) -> int:
        return self.layout.tokens_per_frame * LAT

    @property
    def blend(self) -> torch.Tensor:
        return self.blend_logit.sigmoid()

    @property
    def pixel_blend(self) -> torch.Tensor:
        return self.pixel_blend_logit.sigmoid()

    def pixel_residual(self, regions: torch.Tensor) -> torch.Tensor:
        """Decode eight region tokens into a spatially aligned 64x64 residual."""

        if regions.ndim != 4 or regions.shape[2:] != (self.layout.region_tokens, LAT):
            raise ValueError("expected compact regions (B,T,8,LAT)")
        b, t = regions.shape[:2]
        patch_height = 64 // self.layout.region_rows
        patch_width = 64 // self.layout.region_cols
        patches = self.pixel_region_decoder(regions).reshape(
            b, t, self.layout.region_rows, self.layout.region_cols, patch_height, patch_width
        )
        return patches.permute(0, 1, 2, 4, 3, 5).reshape(b, t, 1, 64, 64)


class RevAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x: torch.Tensor, step_bias: torch.Tensor) -> torch.Tensor:
        x = self.norm(x + step_bias)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class RevMlp(nn.Module):
    def __init__(self, dim: int, ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.net = nn.Sequential(
            nn.Linear(dim, ratio * dim), nn.GELU(approximate="tanh"), nn.Linear(ratio * dim, dim)
        )

    def forward(self, x: torch.Tensor, step_bias: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x + step_bias))


class ReversibleTokenBlock(nn.Module):
    """Additive coupling block; stochastic layers are intentionally excluded."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.f = RevAttention(dim, heads)
        self.g = RevMlp(dim)

    def forward(self, x: torch.Tensor, step_bias: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y2 = x2 + self.f(x1, step_bias)
        y1 = x1 + self.g(y2, step_bias)
        return torch.cat([y1, y2], dim=-1)

    def inverse(self, y: torch.Tensor, step_bias: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-1)
        x1 = y1 - self.g(y2, step_bias)
        x2 = y2 - self.f(x1, step_bias)
        return torch.cat([x1, x2], dim=-1)

    def backward_pass(self, y: torch.Tensor, dy: torch.Tensor,
                      step_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y1, y2 = y.chunk(2, dim=-1)
        dy1, dy2 = dy.chunk(2, dim=-1)
        with torch.enable_grad():
            y2 = y2.detach().requires_grad_(True)
            g_out = self.g(y2, step_bias)
            torch.autograd.backward(g_out, dy1)
        with torch.no_grad():
            x1 = y1 - g_out
            dx2 = dy2 + y2.grad
        with torch.enable_grad():
            x1 = x1.detach().requires_grad_(True)
            f_out = self.f(x1, step_bias)
            torch.autograd.backward(f_out, dx2)
        with torch.no_grad():
            x2 = y2.detach() - f_out
            dx1 = dy1 + x1.grad
        return torch.cat([x1.detach(), x2.detach()], dim=-1), torch.cat([dx1, dx2], dim=-1)


class _ReversibleStackFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, steps: int, step_embedding: torch.Tensor,
                blocks: nn.ModuleList) -> torch.Tensor:
        with torch.no_grad():
            for step in range(steps):
                bias = step_embedding[step][None, None]
                for block in blocks:
                    x = block(x, bias)
        ctx.save_for_backward(x.detach(), step_embedding)
        ctx.steps = steps
        ctx.blocks = blocks
        return x.detach()

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        y, step_embedding = ctx.saved_tensors
        for step in reversed(range(ctx.steps)):
            bias = step_embedding[step][None, None]
            for block in reversed(ctx.blocks):
                y, dy = block.backward_pass(y, dy, bias)
        # The embedding is conditioning metadata.  It is kept fixed rather than
        # differentiated through the custom reverse path.
        return dy, None, None, None


class ReversibleTemporalCore(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4, depth: int = 2, max_steps: int = 32):
        super().__init__()
        self.dim = dim
        self.input_a = nn.Linear(dim, dim)
        self.input_b = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([ReversibleTokenBlock(dim, heads) for _ in range(depth)])
        self.step_embedding = nn.Parameter(torch.zeros(max_steps, dim), requires_grad=False)
        nn.init.normal_(self.step_embedding, std=0.02)

    def forward(self, x: torch.Tensor, steps: int, reversible: bool = True) -> torch.Tensor:
        if not 0 <= steps <= self.step_embedding.shape[0]:
            raise ValueError("steps outside configured range")
        state = torch.cat([self.input_a(x), self.input_b(x)], dim=-1)
        if steps == 0:
            return state
        if reversible:
            return _ReversibleStackFunction.apply(state, steps, self.step_embedding, self.blocks)
        for step in range(steps):
            bias = self.step_embedding[step][None, None]
            for block in self.blocks:
                state = block(state, bias)
        return state


class HierarchicalRegressionReadout(nn.Module):
    def __init__(self, dim: int, layout: FixedRegionLayout):
        super().__init__()
        self.layout = layout
        self.fuse = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())
        self.coarse_head = nn.Linear(dim, LAT)
        self.region_head = nn.Linear(dim, layout.cells_per_region * LAT)
        self.coarse_to_fine = nn.ConvTranspose2d(LAT, LAT, kernel_size=2, stride=2)
        # Start from persistence at the coarse boundary.  The transformer learns a
        # displacement-dependent residual instead of relearning codec identity.
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.region_head.weight)
        nn.init.zeros_(self.region_head.bias)

    def forward(self, state: torch.Tensor, frames: int,
                coarse_residual_base: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b = state.shape[0]
        x = self.fuse(state).reshape(b, frames, self.layout.tokens_per_frame, -1)
        coarse_tokens = x[:, :, :self.layout.coarse_tokens]
        region_tokens = x[:, :, self.layout.coarse_tokens:]
        zc = self.coarse_head(coarse_tokens).transpose(2, 3).reshape(
            b, frames, LAT, self.layout.coarse_hw, self.layout.coarse_hw
        )
        if coarse_residual_base is not None:
            if coarse_residual_base.shape != zc.shape:
                raise ValueError("coarse residual base has the wrong shape")
            zc = coarse_residual_base + zc
        regional_residual = self.region_head(region_tokens).reshape(
            b, frames, self.layout.region_tokens, self.layout.cells_per_region, LAT
        )
        residual = regions_to_fine(regional_residual, self.layout)
        base = self.coarse_to_fine(zc.reshape(-1, LAT, self.layout.coarse_hw, self.layout.coarse_hw))
        zf = base.reshape(b, frames, LAT, self.layout.fine_hw, self.layout.fine_hw) + residual
        return zc, zf


class FixedRegionRevPredictor(nn.Module):
    """Regression predictor whose ``steps`` denotes temporal displacement."""

    def __init__(self, dim: int = 64, heads: int = 4, depth: int = 2,
                 max_frames: int = 64, max_steps: int = 32, temperature: float = 0.25):
        super().__init__()
        self.tokenizer = FixedRegionTokenizer(dim, max_frames=max_frames, temperature=temperature)
        self.core = ReversibleTemporalCore(dim, heads, depth, max_steps)
        self.readout = HierarchicalRegressionReadout(dim, self.tokenizer.layout)

    @property
    def tokens_per_frame(self) -> int:
        return self.tokenizer.layout.tokens_per_frame

    def forward(self, zc: torch.Tensor, zf: torch.Tensor, steps: int,
                reversible: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, weights = self.tokenizer(zc, zf)
        state = self.core(tokens, steps=steps, reversible=reversible)
        pred_c, pred_f = self.readout(state, frames=zc.shape[1], coarse_residual_base=zc)
        return pred_c, pred_f, weights


def count_context_tokens(frames: int, layout: FixedRegionLayout | None = None) -> int:
    return frames * (layout or FixedRegionLayout()).tokens_per_frame


def openstl_mse(pred01: torch.Tensor, target01: torch.Tensor) -> torch.Tensor:
    """OpenSTL Moving-MNIST MSE: mean over batch/time, sum over C/H/W."""

    if pred01.shape != target01.shape:
        raise ValueError("prediction and target shapes differ")
    return (pred01 - target01).square().mean(dim=(0, 1)).sum()


def initialise_module(module: nn.Module) -> None:
    for item in module.modules():
        if isinstance(item, nn.Linear):
            nn.init.xavier_uniform_(item.weight)
            if item.bias is not None:
                nn.init.zeros_(item.bias)
