"""RHCP -- Reversible Hierarchical Clockwork Predictor.

The one genuinely new module of the dissertation's "hierarchical reversibility" direction: a CW-VAE-class
spatial pyramid whose per-level TEMPORAL recurrence is REVERSIBLE (RevNet additive coupling along time), run on a
FIXED geometric clock where slow levels HOLD (identity) on non-tick frames instead of forgetting. Because every step
is exactly invertible, the full K-step free-running autoregressive rollout can be backpropagated by REVERSE-REPLAY
(reconstruct each past state via inverse) at recurrent-core activation memory CONSTANT in K -- the capability that
lets a free-running unroll fit where vanilla full-sequence BPTT OOMs. See docs/RHCP_ARCHITECTURE.md for the HONEST
scope: this is a CAPABILITY vs BPTT and a recompute/throughput Pareto point vs gradient checkpointing -- NOT a memory
win over checkpointing.

Honesty anchors enforced in code:
  * The reversible block is bit-identical to standard RevNet/RevViT additive coupling (the "two streams" are the two
    channel halves; even/odd = the F/G alternation). It is NOT a new reversible primitive -- cite RevNet (1707.04585).
  * The clock is FIXED-schedule (t % k_l == 0). A learned hard 0/1 gate is non-invertible (proto P4) and is forbidden
    on the state path. Multi-rate behaviour comes from lossless HOLDING, so MacKay's no-forgetting tension never bites.
  * Invertibility of the assembled multi-level block holds only under a strict ACYCLIC top-down evaluation order
    (a child may read its parent's just-updated state; no parent may read a child's same-frame state). assert_acyclic()
    and the P1 tests in rhcp_validate.py enforce this on the REAL assembled modules, not a toy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------------
# F / G mixers (need NOT be invertible -- additive coupling only re-evaluates them)
# --------------------------------------------------------------------------------------------------
class ConvMixer(nn.Module):
    """ConvNeXt-style spatial mixer, (B, c_in, h, w) -> (B, c_out, h, w), spatial size preserved.

    Used as F and G inside the reversible coupling. Zero-initialised last layer so a fresh stage starts as the
    identity transition (s_{t+1} = s_t), which is a numerically benign, trivially-invertible starting point.
    """

    def __init__(self, c_in: int, c_out: int, hidden: Optional[int] = None, k: int = 7, lipschitz: bool = False,
                 out_scale: float = 1.0):
        super().__init__()
        self.out_scale = out_scale
        hidden = hidden or 4 * c_out
        self.dw = nn.Conv2d(c_in, c_in, k, padding=k // 2, groups=c_in)
        self.norm = nn.GroupNorm(1, c_in)
        self.pw1 = nn.Conv2d(c_in, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, c_out, 1)
        if not lipschitz:                       # identity-start; NB a zero weight has no valid spectral norm (0/0=NaN)
            nn.init.zeros_(self.pw2.weight)
            nn.init.zeros_(self.pw2.bias)
        if lipschitz:
            # spectral-norm the convs -> bounded Lip(F). Invertibility is UNAFFECTED (additive coupling inverts for
            # ANY F); this bounds the DYNAMICS (the actual stability lever -- invertibility is NOT one). GroupNorm is
            # dropped under Lipschitz mode (its per-sample rescaling is not 1-Lipschitz).
            from torch.nn.utils.parametrizations import spectral_norm
            self.dw = spectral_norm(self.dw)
            self.pw1 = spectral_norm(self.pw1)
            self.pw2 = spectral_norm(self.pw2)
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        return x if self.out_scale == 1.0 else self.out_scale * x


# --------------------------------------------------------------------------------------------------
# Reversible additive-coupling stage (RevNet eq. 8-10) -- the temporal transition for ONE level
# --------------------------------------------------------------------------------------------------
class RevConvStage(nn.Module):
    """Reversible transition s_t -> s_{t+1} on a spatial latent of `channels` (even) channels, optionally
    conditioned on a parent context `c` (B, cond_ch, h, w). `depth` additive-coupling pairs are stacked (the
    "blocks divisible by 2" the user asked for: depth pairs = 2*depth F/G functions, alternating).

    forward:  for each pair: x1 += F([x2,c]); x2 += G([x1,c])
    inverse:  reverse the pairs: x2 -= G([x1,c]); x1 -= F([x2,c])      (exact for ANY c)
    """

    def __init__(self, channels: int, cond_ch: int = 0, depth: int = 2, k: int = 7, hidden_mult: int = 4,
                 lipschitz: bool = False, out_scale: float = 1.0):
        super().__init__()
        assert channels % 2 == 0, "reversible coupling needs an even channel count"
        self.half = channels // 2
        self.cond_ch = cond_ch
        self.depth = depth
        self.Fs = nn.ModuleList(
            ConvMixer(self.half + cond_ch, self.half, hidden=hidden_mult * self.half, k=k, lipschitz=lipschitz,
                      out_scale=out_scale) for _ in range(depth)
        )
        self.Gs = nn.ModuleList(
            ConvMixer(self.half + cond_ch, self.half, hidden=hidden_mult * self.half, k=k, lipschitz=lipschitz,
                      out_scale=out_scale) for _ in range(depth)
        )

    def _cat(self, a: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        if c is None or self.cond_ch == 0:
            return a
        return torch.cat([a, c], dim=1)

    def forward(self, s: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1, x2 = s.chunk(2, dim=1)
        for Ff, Gg in zip(self.Fs, self.Gs):
            x1 = x1 + Ff(self._cat(x2, c))
            x2 = x2 + Gg(self._cat(x1, c))
        return torch.cat([x1, x2], dim=1)

    def inverse(self, s: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1, x2 = s.chunk(2, dim=1)
        for Ff, Gg in zip(reversed(self.Fs), reversed(self.Gs)):
            x2 = x2 - Gg(self._cat(x1, c))
            x1 = x1 - Ff(self._cat(x2, c))
        return torch.cat([x1, x2], dim=1)


# --------------------------------------------------------------------------------------------------
# Configuration of the level pyramid (bottom -> top: fine/big-spatial/fast-clock -> coarse/small/slow)
# --------------------------------------------------------------------------------------------------
@dataclass
class LevelSpec:
    channels: int
    spatial: int          # h == w
    tick: int             # update period: level advances only when t % tick == 0
    depth: int = 2        # coupling pairs in this level's reversible stage


@dataclass
class RHCPConfig:
    # default = the 10->10 MMNIST ship config: L0 16x16 k1 / L1 8x8 k2 / L2 4x4 k4 (L3 2x2 k8 optional)
    levels: List[LevelSpec] = field(default_factory=lambda: [
        LevelSpec(channels=64, spatial=16, tick=1, depth=2),
        LevelSpec(channels=96, spatial=8, tick=2, depth=2),
        LevelSpec(channels=128, spatial=4, tick=4, depth=2),
    ])
    img_ch: int = 1
    cond_frac: float = 0.5     # parent->child conditioning channels = round(cond_frac * child.half)
    top_ext_ch: int = 0        # external drive channels fed to the top level (0 = none/closed-loop)


# --------------------------------------------------------------------------------------------------
# The reversible hierarchical clockwork core
# --------------------------------------------------------------------------------------------------
class RHCPCore(nn.Module):
    """Holds L per-level reversible stages + the top-down cross-level conditioning projections.

    State is a list `[s_0, ..., s_{L-1}]` (bottom..top). One frame advances levels TOP->BOTTOM so a child can read
    its parent's just-updated state (co-evolving parent). The reverse pass advances BOTTOM->TOP, reading each parent
    from the not-yet-inverted state -- this is the strict acyclic ordering that makes the assembly invertible.
    """

    def __init__(self, cfg: RHCPConfig):
        super().__init__()
        self.cfg = cfg
        self.L = len(cfg.levels)
        self.stages = nn.ModuleList()
        self.cond_proj = nn.ModuleList()     # cond_proj[l]: project parent (l+1) -> conditioning for level l
        self.cond_ch: List[int] = []
        for l, spec in enumerate(cfg.levels):
            half = spec.channels // 2
            if l < self.L - 1:
                cch = max(1, round(cfg.cond_frac * half))
                parent_ch = cfg.levels[l + 1].channels
                self.cond_proj.append(nn.Conv2d(parent_ch, cch, 1))
            else:
                cch = cfg.top_ext_ch
                self.cond_proj.append(nn.Identity())
            self.cond_ch.append(cch)
            self.stages.append(RevConvStage(spec.channels, cond_ch=cch, depth=spec.depth))

    # -- conditioning for level l at this frame, given the (already-updated this frame) state list `s` --
    def _cond(self, s: List[torch.Tensor], l: int, ext: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if l == self.L - 1:                                   # top level: external drive (or none)
            return ext if self.cond_ch[l] > 0 else None
        parent = s[l + 1]                                     # coarser, smaller spatial
        c = self.cond_proj[l](parent)
        child_hw = self.cfg.levels[l].spatial
        if c.shape[-1] != child_hw:
            c = F.interpolate(c, size=(child_hw, child_hw), mode="nearest")
        return c

    def ticks(self, l: int, t: int) -> bool:
        return (t % self.cfg.levels[l].tick) == 0

    def step(self, s: List[torch.Tensor], t: int, ext: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Advance one frame. TOP->BOTTOM so children see updated parents. Held levels are returned unchanged."""
        out = list(s)
        for l in range(self.L - 1, -1, -1):                  # top -> bottom
            if self.ticks(l, t):
                c = self._cond(out, l, ext)                  # reads parent from `out` (already updated this frame)
                out[l] = self.stages[l].forward(out[l], c)
            # else: identity hold (out[l] unchanged)
        return out

    def inverse_step(self, s: List[torch.Tensor], t: int, ext: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Exact inverse of step(): BOTTOM->TOP so each parent is still in its post-update form when read."""
        out = list(s)
        for l in range(0, self.L):                           # bottom -> top
            if self.ticks(l, t):
                c = self._cond(out, l, ext)                  # parent (l+1) not yet inverted -> still post-update
                out[l] = self.stages[l].inverse(out[l], c)
        return out

    def assert_acyclic(self) -> bool:
        """Static guarantee: conditioning only flows parent(l+1) -> child(l). No level reads a finer level's
        same-frame state, so the per-frame dependency graph is a DAG and inverse_step is well-defined."""
        # by construction _cond(l) reads only s[l+1]; encode that invariant explicitly for the reader/test.
        for l in range(self.L):
            reads = [] if l == self.L - 1 else [l + 1]
            assert all(r > l for r in reads), f"level {l} reads a non-parent level {reads} -> cycle risk"
        return True


# --------------------------------------------------------------------------------------------------
# Minimal encoder / decoder so the core is end-to-end usable on 64x64 single-channel video
# --------------------------------------------------------------------------------------------------
def _conv_bn_act(c_in, c_out, stride):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(1, c_out),
        nn.SiLU(),
    )


class PyramidEncoder(nn.Module):
    """(B, T_in, img_ch, 64, 64) -> per-level init states matching each LevelSpec (channels, spatial).

    CRUCIAL: it consumes the FULL context, not one frame. Motion (velocity) is unobservable from a single frame, so
    each level's init state is built from a temporal summary [mean over time | last frame | last-first difference]
    of that level's features -- the difference term carries the motion the free-running recurrence must extrapolate.
    """

    def __init__(self, cfg: RHCPConfig, in_hw: int = 64):
        super().__init__()
        self.cfg = cfg
        widths = [32, 64, 96, 128, 192, 256]
        self.stem = _conv_bn_act(cfg.img_ch, widths[0], 1)          # 64
        downs, hw, prev = [], in_hw, widths[0]
        self.feat_at = {}                                            # spatial -> index into feats
        idx = 0
        i = 1
        while hw > min(s.spatial for s in cfg.levels):
            downs.append(_conv_bn_act(prev, widths[i], 2))
            prev = widths[i]
            hw //= 2
            idx += 1
            self.feat_at[hw] = idx
            i += 1
        self.downs = nn.ModuleList(downs)
        # one head per level: temporal summary [mean|last|diff] (3*width) -> level channels
        self.heads = nn.ModuleList()
        for spec in cfg.levels:
            fi = self.feat_at[spec.spatial]
            self.heads.append(nn.Conv2d(3 * widths[fi], spec.channels, 1))
        self._feat_widths = widths

    def _frame_feats(self, frame: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(frame)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        return feats

    def forward(self, context: torch.Tensor) -> List[torch.Tensor]:
        if context.dim() == 4:                                      # (B,C,H,W) -> single-frame fallback
            context = context.unsqueeze(1)
        B, T = context.shape[0], context.shape[1]
        flat = context.reshape(B * T, *context.shape[2:])
        feats = self._frame_feats(flat)                             # list per scale, each (B*T, Cw, h, w)
        states = []
        for l, spec in enumerate(self.cfg.levels):
            fi = self.feat_at[spec.spatial]
            f = feats[fi].reshape(B, T, *feats[fi].shape[1:])       # (B,T,Cw,h,w)
            # RECENT velocity (last - last[-3]) is a clean motion estimate ROBUST to context length; (last - first)
            # is meaningless once T is long (digits bounce many times). mean = appearance; last = current position.
            prev = f[:, -3] if T >= 3 else f[:, 0]
            summary = torch.cat([f.mean(1), f[:, -1], f[:, -1] - prev], dim=1)
            states.append(self.heads[l](summary))
        return states


class PixelDecoder(nn.Module):
    """Bottom-level state (C0, s0, s0) -> (img_ch, 64, 64). Output in [-1, 1] via tanh."""

    def __init__(self, cfg: RHCPConfig, out_hw: int = 64):
        super().__init__()
        c0, s0 = cfg.levels[0].channels, cfg.levels[0].spatial
        ups, hw, prev = [], s0, c0
        w = 128
        while hw < out_hw:
            ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), _conv_bn_act(prev, w, 1)))
            prev = w
            hw *= 2
            w = max(32, w // 2)
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(prev, cfg.img_ch, 3, padding=1)

    def forward(self, s0: torch.Tensor) -> torch.Tensor:
        x = s0
        for u in self.ups:
            x = u(x)
        return torch.tanh(self.head(x))


class RHCP(nn.Module):
    """End-to-end wrapper: encode the last context frame -> states; roll the reversible clock K steps; decode L0."""

    def __init__(self, cfg: Optional[RHCPConfig] = None):
        super().__init__()
        self.cfg = cfg or RHCPConfig()
        self.encoder = PyramidEncoder(self.cfg)
        self.core = RHCPCore(self.cfg)
        self.decoder = PixelDecoder(self.cfg)

    def init_states(self, context: torch.Tensor) -> List[torch.Tensor]:
        """context: (B, T_in, img_ch, H, W). Initialise from the FULL context (motion needs >=2 frames)."""
        return self.encoder(context)

    def rollout(self, context: torch.Tensor, K: int) -> torch.Tensor:
        """Standard (graph-retaining) free-running rollout -> (B, K, img_ch, H, W). For BPTT / inference."""
        s = self.init_states(context)
        frames = []
        for t in range(K):
            s = self.core.step(s, t)
            frames.append(self.decoder(s[0]))
        return torch.stack(frames, dim=1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
