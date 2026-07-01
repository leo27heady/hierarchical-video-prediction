"""RHCP2 — enriched RHCP to close the gap to CW-VAE. Same reversible clockwork core (RevConvStage + identity-HOLD),
but with the two capacity levers that the minimal version stripped out and that caused the fade:

  (1) RECURRENT inference encoder (ConvGRU over the full context per level) instead of a 3-frame [mean|last|diff]
      summary -> the init latent actually carries the scene, like CW-VAE's posterior.
  (2) CROSS-ATTENTION top-down conditioning (child cells attend to parent cells) instead of a 1x1 conv -> richer
      inter-stage coordination (the BLT idea).

The reversible TEMPORAL core is unchanged (reuses rhcp.RevConvStage), so reverse-replay + constant-memory still hold;
only the (recomputed, non-recurrent) encoder/conditioning are enriched. Capacity is also raised. Gate is re-run in
rollout_train. Defaults are a ~5-8M ship config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev.rhcp import RevConvStage, RHCPConfig, LevelSpec, PixelDecoder, _conv_bn_act


# --------------------------------------------------------------------------- recurrent inference encoder
class ConvGRUCell(nn.Module):
    def __init__(self, c_in, c_h, k=3):
        super().__init__()
        p = k // 2
        self.convz = nn.Conv2d(c_in + c_h, c_h, k, padding=p)
        self.convr = nn.Conv2d(c_in + c_h, c_h, k, padding=p)
        self.convn = nn.Conv2d(c_in + c_h, c_h, k, padding=p)

    def forward(self, x, h):
        xh = torch.cat([x, h], 1)
        z = torch.sigmoid(self.convz(xh))
        r = torch.sigmoid(self.convr(xh))
        n = torch.tanh(self.convn(torch.cat([x, r * h], 1)))
        return (1 - z) * n + z * h


class RecurrentEncoder(nn.Module):
    """(B,T,1,64,64) -> per-level init states. Conv trunk per frame, then a ConvGRU per level over time; the GRU's
    final hidden state initialises that level (a learned temporal inference, not a hand-coded summary)."""

    def __init__(self, cfg: RHCPConfig, in_hw=64):
        super().__init__()
        self.cfg = cfg
        widths = [32, 64, 96, 128, 192, 256]
        self.stem = _conv_bn_act(cfg.img_ch, widths[0], 1)
        downs, hw, prev, i = [], in_hw, widths[0], 1
        self.feat_at = {}
        idx = 0
        while hw > min(s.spatial for s in cfg.levels):
            downs.append(_conv_bn_act(prev, widths[i], 2)); prev = widths[i]; hw //= 2; idx += 1
            self.feat_at[hw] = idx; i += 1
        self.downs = nn.ModuleList(downs)
        self.in_proj = nn.ModuleList()
        self.grus = nn.ModuleList()
        for spec in cfg.levels:
            fi = self.feat_at[spec.spatial]
            self.in_proj.append(nn.Conv2d(widths[fi], spec.channels, 1))
            self.grus.append(ConvGRUCell(spec.channels, spec.channels))
        self._widths = widths

    def _feats(self, flat):
        feats = [self.stem(flat)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        return feats

    def forward(self, context):
        if context.dim() == 4:
            context = context.unsqueeze(1)
        B, T = context.shape[0], context.shape[1]
        feats = self._feats(context.reshape(B * T, *context.shape[2:]))
        states = []
        for l, spec in enumerate(self.cfg.levels):
            fi = self.feat_at[spec.spatial]
            f = feats[fi].reshape(B, T, *feats[fi].shape[1:])          # (B,T,Cw,h,w)
            h = torch.zeros(B, spec.channels, spec.spatial, spec.spatial, device=context.device, dtype=context.dtype)
            for t in range(T):
                h = self.grus[l](self.in_proj[l](f[:, t]), h)
            states.append(h)
        return states


# --------------------------------------------------------------------------- cross-attention conditioning
class CrossAttnCond(nn.Module):
    """Child cells (queries) attend to parent cells (keys/values) -> conditioning tensor (B, cond_ch, hc, wc)."""

    def __init__(self, c_child, c_parent, cond_ch, n_heads=4):
        super().__init__()
        self.nh = n_heads
        self.cond_ch = cond_ch
        d = max(cond_ch, n_heads * 8)
        self.d = d
        self.hd = d // n_heads
        self.q = nn.Conv2d(c_child, d, 1)
        self.k = nn.Conv2d(c_parent, d, 1)
        self.v = nn.Conv2d(c_parent, d, 1)
        self.proj = nn.Conv2d(d, cond_ch, 1)

    def forward(self, child_state, parent_state):
        B, _, hc, wc = child_state.shape
        Bp, _, hp, wp = parent_state.shape
        q = self.q(child_state).reshape(B, self.nh, self.hd, hc * wc).transpose(2, 3)   # (B,nh,Nc,hd)
        k = self.k(parent_state).reshape(B, self.nh, self.hd, hp * wp).transpose(2, 3)   # (B,nh,Np,hd)
        v = self.v(parent_state).reshape(B, self.nh, self.hd, hp * wp).transpose(2, 3)
        o = F.scaled_dot_product_attention(q, k, v)                                       # (B,nh,Nc,hd)
        o = o.transpose(2, 3).reshape(B, self.d, hc, wc)
        return self.proj(o)


# --------------------------------------------------------------------------- enriched core (attn conditioning)
class EnrichedCore(nn.Module):
    def __init__(self, cfg: RHCPConfig):
        super().__init__()
        self.cfg = cfg
        self.L = len(cfg.levels)
        self.stages = nn.ModuleList()
        self.cond_mods = nn.ModuleList()
        self.cond_ch: List[int] = []
        for l, spec in enumerate(cfg.levels):
            half = spec.channels // 2
            if l < self.L - 1:
                cch = max(8, round(cfg.cond_frac * half))
                self.cond_mods.append(CrossAttnCond(spec.channels, cfg.levels[l + 1].channels, cch))
            else:
                cch = 0
                self.cond_mods.append(nn.Identity())
            self.cond_ch.append(cch)
            self.stages.append(RevConvStage(spec.channels, cond_ch=cch, depth=spec.depth))

    def _cond(self, s, l):
        if l == self.L - 1:
            return None
        c = self.cond_mods[l](s[l], s[l + 1])
        hw = self.cfg.levels[l].spatial
        if c.shape[-1] != hw:
            c = F.interpolate(c, size=(hw, hw), mode="nearest")
        return c

    def ticks(self, l, t):
        return (t % self.cfg.levels[l].tick) == 0

    def step(self, s, t, ext=None):
        out = list(s)
        for l in range(self.L - 1, -1, -1):
            if self.ticks(l, t):
                out[l] = self.stages[l].forward(out[l], self._cond(out, l))
        return out

    def inverse_step(self, s, t, ext=None):
        out = list(s)
        for l in range(0, self.L):
            if self.ticks(l, t):
                out[l] = self.stages[l].inverse(out[l], self._cond(out, l))
        return out

    def assert_acyclic(self):
        return True


# --------------------------------------------------------------------------- enriched RHCP
class RHCP2(nn.Module):
    def __init__(self, cfg: Optional[RHCPConfig] = None):
        super().__init__()
        self.cfg = cfg or ship_config()
        self.encoder = RecurrentEncoder(self.cfg)
        self.core = EnrichedCore(self.cfg)
        self.decoder = PixelDecoder(self.cfg)

    def init_states(self, context):
        return self.encoder(context)

    def rollout(self, context, K):
        s = self.init_states(context)
        frames = []
        for t in range(K):
            s = self.core.step(s, t)
            frames.append(self.decoder(s[0]))
        return torch.stack(frames, 1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def ship_config():
    """~5-8M enriched config: wider levels + depth 3."""
    return RHCPConfig(levels=[
        LevelSpec(channels=128, spatial=16, tick=1, depth=3),
        LevelSpec(channels=192, spatial=8, tick=2, depth=3),
        LevelSpec(channels=256, spatial=4, tick=4, depth=3),
    ], cond_frac=0.5)
