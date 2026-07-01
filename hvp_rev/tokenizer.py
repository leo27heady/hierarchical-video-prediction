"""Scaled-down learned token-pyramid autoencoder (MilliVid-style) for the Chapter-C hierarchy study.

Per FRAME (time is handled downstream): build a spatial pyramid of token levels, each independently decodable. Three
variants switch the ONLY thing under study — how coarse levels are formed:
  - adaptive : LEARNED coarse tokens (a transformer sees all levels jointly; coarse tokens are free to encode layout,
               not just a blur). This is MilliVid's design.
  - cascaded : coarse tokens = mean-pool of the fine latent (a blurry downsample; MilliVid's ablation baseline).
  - flat     : single finest level only (no hierarchy).
Trained with random-level dropout (decode ONE level, others zeroed) + reconstruction (+ optional LPIPS). The question
Chapter C answers: does `adaptive` beat `cascaded`/`flat` on long-horizon prediction WHEN the task has real off-screen
memory (the looping testbed) — i.e. when the coarse level has a job, unlike on MMNIST.

  venv/Scripts/python.exe -m hvp_rev.tokenizer --selftest
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokCfg:
    img_ch: int = 1
    img_hw: int = 64
    patch: int = 8               # base grid = img_hw/patch = 8x8 = 64 tokens
    levels: int = 3              # 8x8, 4x4, 2x2
    dim: int = 256               # transformer width
    depth: int = 6
    heads: int = 8
    latent_ch: int = 16          # per-token latent (tanh)
    variant: str = "adaptive"    # adaptive | cascaded | flat


def _grid(n):
    return n * n


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


def _sincos(n_tokens_per_level, dim, device):
    """Simple learned-free positional id: per-(level, pos) index embedding via sinusoids on a flattened id."""
    ids = torch.arange(sum(n_tokens_per_level), device=device).float()
    pe = torch.zeros(len(ids), dim, device=device)
    div = torch.exp(torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(ids[:, None] * div)
    pe[:, 1::2] = torch.cos(ids[:, None] * div)
    return pe


class TokenPyramidAE(nn.Module):
    def __init__(self, cfg: TokCfg):
        super().__init__()
        self.cfg = cfg
        base = cfg.img_hw // cfg.patch                    # 8
        self.grids = [base // (2 ** l) for l in range(cfg.levels)]   # [8,4,2]
        self.ntok = [g * g for g in self.grids]           # [64,16,4]
        p = cfg.patch
        self.patch_dim = cfg.img_ch * p * p
        # encoder: patch-embed each level's mean-pooled image, joint transformer, bottleneck to latent_ch (tanh)
        self.embed = nn.Linear(self.patch_dim, cfg.dim)
        self.level_emb = nn.Embedding(cfg.levels, cfg.dim)
        self.enc = nn.ModuleList([_Block(cfg.dim, cfg.heads) for _ in range(cfg.depth)])
        self.bottleneck = nn.Linear(cfg.dim, cfg.latent_ch)
        # decoder: unbottleneck + level-conditioned transformer -> per-patch pixels for that level
        self.unbottle = nn.Linear(cfg.latent_ch, cfg.dim)
        self.dec_level_emb = nn.Embedding(cfg.levels, cfg.dim)
        self.dec = nn.ModuleList([_Block(cfg.dim, cfg.heads) for _ in range(cfg.depth)])
        self.to_pix = nn.Linear(cfg.dim, self.patch_dim)

    def _patchify_level(self, img, level):
        """img (B,C,H,W) -> tokens (B, ntok_l, patch_dim) at level's resolution (mean-pooled by 2^level)."""
        p = self.cfg.patch
        if level > 0:
            img = F.avg_pool2d(img, 2 ** level)
        B, C, H, W = img.shape
        t = img.unfold(2, p, p).unfold(3, p, p)           # (B,C,H/p,W/p,p,p)
        t = t.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return t

    def encode(self, img) -> List[torch.Tensor]:
        """-> list of per-level latents (B, ntok_l, latent_ch) in [-1,1]."""
        B = img.shape[0]
        cfg = self.cfg
        if cfg.variant == "flat":
            toks = self.embed(self._patchify_level(img, 0)) + self.level_emb(torch.zeros(1, dtype=torch.long, device=img.device))
            x = toks
            for blk in self.enc:
                x = blk(x)
            return [torch.tanh(self.bottleneck(x))]
        # adaptive/cascaded: build all levels
        parts, lids = [], []
        for l in range(cfg.levels):
            pl = self.embed(self._patchify_level(img, l))
            parts.append(pl)
            lids += [l] * self.ntok[l]
        x = torch.cat(parts, 1)                            # (B, sum ntok, dim)
        lid = torch.tensor(lids, device=img.device)
        x = x + self.level_emb(lid)[None] + _sincos(self.ntok, cfg.dim, img.device)[None]
        if cfg.variant == "adaptive":
            for blk in self.enc:                            # joint attention across levels -> LEARNED coarse tokens
                x = blk(x)
        z = torch.tanh(self.bottleneck(x))                  # (B, sum ntok, latent_ch)
        lat = list(torch.split(z, self.ntok, dim=1))
        if cfg.variant == "cascaded":
            # overwrite coarse levels with mean-pooled fine latent (a blurry downsample; MilliVid ablation)
            fine = lat[0]
            fine_img = fine.transpose(1, 2).reshape(B, cfg.latent_ch, self.grids[0], self.grids[0])
            for l in range(1, cfg.levels):
                d = F.avg_pool2d(fine_img, 2 ** l)
                lat[l] = torch.tanh(d.reshape(B, cfg.latent_ch, -1).transpose(1, 2))
        return lat

    def decode_level(self, latents: List[torch.Tensor], level: int):
        """Decode ONE level (others zeroed) -> reconstructed image (B,C,H,W). Per-level decodability."""
        cfg = self.cfg
        B = latents[0].shape[0]
        z = latents[level]
        x = self.unbottle(z) + self.dec_level_emb(torch.full((1,), level, dtype=torch.long, device=z.device))
        x = x + _sincos([self.ntok[level]], cfg.dim, z.device)[None]
        for blk in self.dec:
            x = blk(x)
        pix = self.to_pix(x)                                # (B, ntok_l, patch_dim)
        g = self.grids[level]
        p = cfg.patch
        img = pix.reshape(B, g, g, cfg.img_ch, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, cfg.img_ch, g * p, g * p)
        # this reconstructs at level resolution (g*p); upsample to full res for a common target
        if img.shape[-1] != cfg.img_hw:
            img = F.interpolate(img, size=(cfg.img_hw, cfg.img_hw), mode="nearest")
        return torch.tanh(img)

    def forward(self, img):
        lat = self.encode(img)
        return [self.decode_level(lat, l) for l in range(len(lat))], lat

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def selftest():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for variant in ["adaptive", "cascaded", "flat"]:
        cfg = TokCfg(variant=variant)
        ae = TokenPyramidAE(cfg).to(dev)
        x = torch.randn(2, 1, 64, 64, device=dev).tanh()
        recons, lat = ae(x)
        shapes = [tuple(z.shape[1:]) for z in lat]
        print(f"  {variant:>9}: levels {len(lat)} latent-shapes {shapes} recon0 {tuple(recons[0].shape)} "
              f"params {ae.n_params()/1e6:.2f}M")
        assert recons[0].shape == x.shape
    print("tokenizer selftest PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    print("TokenPyramidAE selftest")
    selftest()
