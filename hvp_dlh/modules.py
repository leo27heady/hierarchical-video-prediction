"""Reusable building blocks for DLH: image conv encoder/decoder (lifted from the CW-VAE port,
../cwvae/cwvae_torch.py — identical 4-conv/4-deconv DeepMind stack), MLP/GRU heads, Gaussian
stochastic head, and the analytic KL divergences. All shapes documented; all sizes configurable."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- distributions
def gaussian_kl(mq, sq, mp, sp):
    """Elementwise KL( N(mq, sq^2) || N(mp, sp^2) ). Inputs are means/STDDEVS (not log-vars)."""
    return torch.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2 * sp ** 2) - 0.5


def gaussian_diff_entropy(std):
    """Differential entropy of a diagonal Gaussian, summed over the last dim: 0.5*sum log(2*pi*e*std^2)."""
    return 0.5 * torch.log(2 * math.pi * math.e * std ** 2).sum(-1)


def bernoulli_kl_hard(e, p_logit):
    """KL( Ber(q) || Ber(sigmoid(p_logit)) ) for a HARD posterior q in {0,1}.
    With hard q, H[q]=0, so KL = -log p(e) = BCE-with-logits(p_logit, e)."""
    return F.binary_cross_entropy_with_logits(p_logit, e, reduction="none")


# ----------------------------------------------------------------------------- heads
def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, act=nn.ELU, out_act=None):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    if out_act is not None:
        layers += [out_act()]
    return nn.Sequential(*layers)


class GaussianHead(nn.Module):
    """feature -> (mean, std) of a diagonal Gaussian. std = softplus(raw + bias) + min_std."""

    def __init__(self, in_dim, hidden: Sequence[int], stoch_dim, min_std=0.1, std_bias=0.54):
        super().__init__()
        self.min_std, self.std_bias = min_std, std_bias
        self.net = mlp(in_dim, hidden, 2 * stoch_dim)
        self.stoch_dim = stoch_dim

    def forward(self, x):
        mean, raw = self.net(x).chunk(2, dim=-1)
        std = F.softplus(raw + self.std_bias) + self.min_std
        return mean, std


# ----------------------------------------------------------------------------- image conv enc/dec
class ConvImageEncoder(nn.Module):
    """(B,C,H,W) image -> (B, feat_dim) bottom-up base feature x^1. 4x Conv(k4,s2,LeakyReLU)."""

    def __init__(self, in_ch, channels=(32, 64, 128, 256), kernel=4, image_size=64,
                 feat_dim=200, project=True):
        super().__init__()
        layers, c = [], in_ch
        for ch in channels:
            layers += [nn.Conv2d(c, ch, kernel, stride=2), nn.LeakyReLU(0.3)]
            c = ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_ch, image_size, image_size)).flatten(1).shape[1]
        self.flat_dim = flat
        self.project = nn.Linear(flat, feat_dim) if project else nn.Identity()
        self.out_dim = feat_dim if project else flat

    def forward(self, x):                                   # (B,C,H,W) -> (B,out_dim)
        return self.project(self.conv(x).flatten(1))


class ConvImageDecoder(nn.Module):
    """(B, feat_dim) -> (B,C,H,W). Dense -> 4x ConvTranspose. Mirrors the encoder widths."""

    def __init__(self, feat_dim, out_ch, channels=(32, 64, 128, 256), image_size=64,
                 out_activation="tanh"):
        super().__init__()
        c = channels[0]
        self.dense = nn.Linear(feat_dim, 1024)
        # kernels 5,5,6,6 with stride 2 map 1->5->13->30->64 for image_size=64 (DeepMind stack)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 4 * c, 5, 2), nn.LeakyReLU(0.3),     # 1->5
            nn.ConvTranspose2d(4 * c, 2 * c, 5, 2), nn.LeakyReLU(0.3),    # 5->13
            nn.ConvTranspose2d(2 * c, c, 6, 2), nn.LeakyReLU(0.3),        # 13->30
            nn.ConvTranspose2d(c, out_ch, 6, 2),                         # 30->64
        )
        self.out_activation = out_activation

    def forward(self, x):                                   # (B,feat) -> (B,C,H,W)
        h = self.dense(x).view(x.shape[0], -1, 1, 1)
        out = self.net(h)
        if self.out_activation == "tanh":
            out = torch.tanh(out)
        elif self.out_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out
