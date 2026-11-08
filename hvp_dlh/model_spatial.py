"""Per-pixel / spatial-latent DLH (hybrid design, red-teamed). The vector DLH's update-or-persist decision
moves from per-level/per-frame on 20-dim VECTORS to per-LOCATION on 16x16 spatial MAPS: persist the static
background, update only the moving-digit locations (MMNIST: ~9.4% pixels change/frame, 60% on digits).

Every op maps 1:1 onto hvp_dlh/model.py: MLP->1x1 conv, GRUCell->ConvGRUCell, scalar e (B,1)->map (B,1,16,16).
gaussian_kl / bernoulli_kl_hard are reused VERBATIM; only the reduction changes (.sum(-1) -> channel-sum dim=1,
then spatial-sum dim=(2,3)). Three load-bearing grafts vs a naive port (each a red-team finding):
  (1) NESTED CONSTRAINT uses maxpool-3x3 parent DILATION (plain multiply collapses the upper level to 0.8% tick).
  (2) full DETERMINISTIC-CARRY of frozen locations (s,d,qmean) -> a frozen LATENT injects zero noise and advances
      no memory (defends the long-horizon edge). NB: the rendered image is only "latent-static", not bit-static --
      the 3x3 decoder conv leaks ~1px across tick boundaries (set context_carry / image-composite if that matters).
  (3) lambda_e L1 SPARSITY on the straight-through soft tick-prob (the locality premise / persist pressure).
Sizing is the MEMORY-SAFE hybrid: N=2, Cs=8, Cd=32, Cg=48, shared pf-GRU (faithful N=3/Cs=16 OOMs at 16.8 GB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_dlh.modules import gaussian_kl, bernoulli_kl_hard


# --------------------------------------------------------------------------- config
@dataclass
class SpatialDLHConfig:
    dataset: str = "mmnist"
    image_size: int = 64
    image_channels: int = 1
    # ---- spatial-latent geometry (uniform resolution, NO pyramid) ----
    latent_hw: int = 16
    levels: int = 2                          # N=2 (N=3 OOMs + re-triggers the sigma-sweep long-horizon drift)
    stoch_dim: int = 8                       # Cs per location (small = anti-drift; do NOT raise — hurts long-horizon)
    det_dim: int = 48                        # Cd context / bottom-up channels (widened 32->48 per code-review; det width OK)
    temporal_gru: int = 48                   # Cg ConvGRU hidden channels
    head_hidden: int = 96                    # hidden ch of the 1x1 Gaussian heads (widened 64->96; helps gate decision)
    separate_prior_factor_gru: bool = False  # share pf-GRU with d (halves the dominant tape; memory)
    min_std: float = 0.1
    std_bias: float = 0.54
    dec_out_activation: str = "tanh"
    # ---- decoder likelihood ----
    dec_stddev: float = 1.0                  # do NOT lower (sigma sweep: lowering hurts long-horizon)
    decoder_det_carry: bool = False
    state_kl_mode: str = "economy"           # economy | paper
    # ---- per-pixel gate ----
    persist_margin: float = 0.0              # nats added to the tick test (kl_c+margin<kl_s). 0 = faithful to vector DLH.
                                             # (0.5 on an 8-dim latent caused all-persist collapse — code-review.)
    gate_tau: float = 1.0                    # temperature of the soft tick prob
    target_tick_rate: float = 0.12           # desired upper-level tick fraction (the ~9.4% MMNIST motion rate)
    lambda_e: float = 1e-2                   # weight on the ONE-SIDED over-tick penalty relu(soft_rate - target).
                                             # Fires only ABOVE target -> caps over-ticking; never pushes below target.
    lambda_tv: float = 0.0                   # TV on the soft e-map (contiguous blobs); 0 = off
    parent_dilation: int = 3                 # maxpool kernel for nested-constraint dilation (<=1 = plain = BROKEN)
    context_carry: bool = False              # pin c on frozen locations (enable only if leakage)
    det_carry: bool = True                   # deterministic-carry of d/s/qmean on frozen loc (anti-drift)
    # ---- ELBO weighting ----
    beta_max: float = 1.0
    beta_warmup_iters: int = 6000
    free_nats_per_level: tuple = (0.0, 0.0)  # OFF (floors fight the sparsity persist pressure)
    transition_kl_weight: float = 1.0
    force_tick_iters: int = 3000
    # ---- optimisation / run (mirror DLHConfig so a trainer can reuse) ----
    lr: float = 5e-4
    adam_eps: float = 1e-4
    grad_clip: float = 100.0
    seq_len: int = 100
    batch_size: int = 32
    use_grad_checkpoint: bool = False
    seed: int = 0
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- helper modules
class ConvGaussianHead(nn.Module):
    """1x1-conv GaussianHead: feature map -> (mean, std) maps. std = softplus(raw+bias)+min_std (no spatial leak)."""
    def __init__(self, in_ch, hidden_ch, stoch_ch, min_std=0.1, std_bias=0.54):
        super().__init__()
        self.min_std, self.std_bias = min_std, std_bias
        self.net = nn.Sequential(nn.Conv2d(in_ch, hidden_ch, 1), nn.ELU(), nn.Conv2d(hidden_ch, 2 * stoch_ch, 1))

    def forward(self, x):
        mean, raw = self.net(x).chunk(2, dim=1)
        std = F.softplus(raw + self.std_bias) + self.min_std
        return mean, std


class ConvGRUCell(nn.Module):
    """Resolution-preserving ConvGRU (3x3, pad=1). Replaces nn.GRUCell."""
    def __init__(self, in_ch, hid_ch, kernel=3, pad=1):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel, padding=pad)
        self.conv_h = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel, padding=pad)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.conv_zr(torch.cat([x, h], 1))).chunk(2, dim=1)
        n = torch.tanh(self.conv_h(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class SpatialImageEncoder(nn.Module):
    """(B,1,64,64) -> (B,Cd,16,16). pad=1 stride-2 convs (verified 64->32->16), NOT the no-pad DeepMind stack."""
    def __init__(self, in_ch, Cd, image_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.3),   # 64->32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.3),      # 32->16
            nn.Conv2d(64, Cd, 3, stride=1, padding=1),                         # ->16
        )

    def forward(self, x):
        return self.net(x)


class SpatialImageDecoder(nn.Module):
    """(B,Cd,16,16) -> (B,out_ch,64,64). Learned spatial upsample 16->32->64 (preserves location<->pixel map)."""
    def __init__(self, Cd, out_ch, out_activation="tanh"):
        super().__init__()
        # Lean decoder (memory gate: the widened 256/128/64 OOMs B32 seq100). The renderer handles sharpness;
        # this decoder only needs to not bottleneck the DYNAMICS, so keep it at the original DLH-ish width.
        self.net = nn.Sequential(
            nn.Conv2d(Cd, 128, 3, padding=1), nn.LeakyReLU(0.3),                       # 16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.3),    # 16->32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.3),     # 32->64
            nn.Conv2d(32, out_ch, 3, padding=1),                                       # ->64
        )
        self.out_activation = out_activation

    def forward(self, c):
        out = self.net(c)
        if self.out_activation == "tanh":
            out = torch.tanh(out)
        elif self.out_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out


# --------------------------------------------------------------------------- model
class SpatialDLH(nn.Module):
    def __init__(self, cfg: SpatialDLHConfig):
        super().__init__()
        self.cfg = cfg
        N, Cs, Cd, Cg, H = cfg.levels, cfg.stoch_dim, cfg.det_dim, cfg.temporal_gru, cfg.head_hidden
        self.hw = cfg.latent_hw
        self.img_enc = SpatialImageEncoder(cfg.image_channels, Cd, cfg.image_size)
        self.img_dec = SpatialImageDecoder(Cd, cfg.image_channels, cfg.dec_out_activation)
        self.det_proj = nn.Conv2d(Cd + Cg, Cd, 1) if cfg.decoder_det_carry else None
        # bottom-up f_enc^n: x^n -> x^{n+1}
        self.enc = nn.ModuleList([nn.Sequential(nn.Conv2d(Cd, Cd, 1), nn.ELU(), nn.Conv2d(Cd, Cd, 1))
                                  for _ in range(N - 1)])
        # top-down f_dec^n: [s^n, c^n] -> c^{n-1}   (3x3, resolution preserving)
        self.dec = nn.ModuleList([nn.Sequential(nn.Conv2d(Cs + Cd, Cd, 3, padding=1), nn.ELU(),
                                                nn.Conv2d(Cd, Cd, 3, padding=1)) for _ in range(N)])
        self.tem = nn.ModuleList([ConvGRUCell(Cs, Cg) for _ in range(N)])
        self.pf_gru = nn.ModuleList([ConvGRUCell(Cs, Cg) for _ in range(N)]
                                    if cfg.separate_prior_factor_gru else [])
        self.pf_head = nn.ModuleList([nn.Sequential(nn.Conv2d(Cg, 32, 1), nn.ELU(), nn.Conv2d(32, 1, 1))
                                      for _ in range(N)])
        self.post = nn.ModuleList([ConvGaussianHead(Cd + Cd, H, Cs, cfg.min_std, cfg.std_bias) for _ in range(N)])
        self.prior = nn.ModuleList([ConvGaussianHead(Cg + Cd, H, Cs, cfg.min_std, cfg.std_bias) for _ in range(N)])

    # ----------------------------------------------------------------- helpers
    def _bottom_up(self, obs):
        B, T = obs.shape[:2]
        x0 = self.img_enc(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1, self.hw, self.hw)
        feats = [x0]
        for n in range(self.cfg.levels - 1):
            BT = B * T
            feats.append(self.enc[n](feats[-1].reshape(BT, -1, self.hw, self.hw)).reshape(B, T, -1, self.hw, self.hw))
        return feats

    def _init_state(self, B, device):
        N, Cs, Cg, hw = self.cfg.levels, self.cfg.stoch_dim, self.cfg.temporal_gru, self.hw
        z = lambda c: torch.zeros(B, c, hw, hw, device=device)
        return {"s": [z(Cs) for _ in range(N)], "d": [z(Cg) for _ in range(N)],
                "pf": [z(Cg) for _ in range(N)], "qmean": [z(Cs) for _ in range(N)],
                "qstd": [torch.ones(B, Cs, hw, hw, device=device) for _ in range(N)]}

    def _contexts(self, s_list):
        N = self.cfg.levels
        c = torch.zeros(s_list[0].shape[0], self.cfg.det_dim, self.hw, self.hw, device=s_list[0].device)
        cs = [None] * N
        for n in range(N - 1, -1, -1):
            cs[n] = c
            c = self.dec[n](torch.cat([s_list[n], c], 1))
        return cs, c

    # ----------------------------------------------------------------- one timestep
    def _step(self, state, x_t, mode, force_tick=False):
        cfg, N = self.cfg, self.cfg.levels
        B = state["s"][0].shape[0]; dev = state["s"][0].device; hw = self.hw
        ones = lambda: torch.ones(B, 1, hw, hw, device=dev)

        pf_in = [state["pf"][n] if cfg.separate_prior_factor_gru else state["d"][n] for n in range(N)]
        pf_logit = [self.pf_head[n](pf_in[n]) for n in range(N)]            # each (B,1,16,16)

        # ===== Sweep A: per-location e via hard-KL =====
        cs_static, _ = self._contexts(state["s"])
        e_eff = [None] * N; parent = ones()
        klcache = [(None, None)] * N
        for n in range(N):
            if n == 0:
                e_eff[0] = ones(); parent = e_eff[0]; continue
            if force_tick:
                e_raw = ones()
            elif mode == "posterior":
                mu_c, sg_c = self.prior[n](torch.cat([state["d"][n], cs_static[n]], 1))
                mu_q, sg_q = self.post[n](torch.cat([x_t[n], cs_static[n]], 1))
                kl_c = gaussian_kl(mu_q, sg_q, mu_c, sg_c).sum(dim=1, keepdim=True)
                kl_s = gaussian_kl(mu_q, sg_q, state["qmean"][n], state["qstd"][n]).sum(dim=1, keepdim=True)
                klcache[n] = (kl_c, kl_s)
                e_raw = (kl_c + cfg.persist_margin < kl_s).float()          # PERSIST BIAS
            else:
                e_raw = torch.bernoulli(torch.sigmoid(pf_logit[n]))
            if cfg.parent_dilation and cfg.parent_dilation > 1:            # DILATION (load-bearing)
                parent_d = F.max_pool2d(parent, cfg.parent_dilation, stride=1, padding=cfg.parent_dilation // 2)
            else:
                parent_d = parent
            e_eff[n] = e_raw * parent_d
            parent = e_eff[n]

        # ===== Sweep B: top-down spatial sampling + deterministic-carry =====
        c = torch.zeros(B, cfg.det_dim, hw, hw, device=dev)
        new = {k: [None] * N for k in ("s", "d", "pf", "qmean", "qstd")}
        skl_level = [torch.zeros(B, 1, hw, hw, device=dev) for _ in range(N)]
        e_soft_level = [torch.zeros(B, 1, hw, hw, device=dev) for _ in range(N)]
        carry = cfg.det_carry
        for n in range(N - 1, -1, -1):
            mu_c, sg_c = self.prior[n](torch.cat([state["d"][n], c], 1))
            e = e_eff[n]
            if mode == "posterior":
                mu_q, sg_q = self.post[n](torch.cat([x_t[n], c], 1))
                s_changed = mu_q + sg_q * torch.randn_like(sg_q)
                kl_change = gaussian_kl(mu_q, sg_q, mu_c, sg_c).sum(dim=1, keepdim=True)
                if cfg.state_kl_mode == "paper":
                    kl_static = gaussian_kl(mu_q, sg_q, state["qmean"][n], state["qstd"][n]).sum(dim=1, keepdim=True)
                    skl_level[n] = e * kl_change + (1 - e) * kl_static
                else:
                    skl_level[n] = e * kl_change
                new["qmean"][n] = e * mu_q + (1 - e) * state["qmean"][n] if carry else mu_q
                new["qstd"][n] = e * sg_q + (1 - e) * state["qstd"][n] if carry else sg_q
                if n >= 1 and klcache[n][0] is not None:                    # soft tick prob for the sparsity reg
                    kl_c, kl_s = klcache[n]
                    e_soft_level[n] = torch.sigmoid((kl_s - kl_c) / cfg.gate_tau)
            else:
                s_changed = mu_c + sg_c * torch.randn_like(sg_c)
                new["qmean"][n] = state["qmean"][n]; new["qstd"][n] = state["qstd"][n]
            s_n = e * s_changed + (1 - e) * state["s"][n]
            new["s"][n] = s_n
            d_upd = self.tem[n](s_n, state["d"][n])
            new["d"][n] = e * d_upd + (1 - e) * state["d"][n] if carry else d_upd
            if cfg.separate_prior_factor_gru:
                pf_upd = self.pf_gru[n](s_n, state["pf"][n])
                new["pf"][n] = e * pf_upd + (1 - e) * state["pf"][n] if carry else pf_upd
            else:
                new["pf"][n] = new["d"][n]
            c_new = self.dec[n](torch.cat([s_n, c], 1))
            c = e * c_new + (1 - e) * cs_static[n] if cfg.context_carry else c_new
        c_dec = self.det_proj(torch.cat([c, new["d"][0]], 1)) if self.det_proj is not None else c
        decoded = self.img_dec(c_dec)

        # ---- transition KL (per-location). Mask by the DILATED parent so it matches how e_eff[n] was gated
        # in Sweep A (else the dilation-halo ticks are predicted but never supervised — code-review item 4). ----
        trans_kl = torch.zeros(B, 1, hw, hw, device=dev); parent = ones()
        for n in range(N):
            if n >= 1:
                if cfg.parent_dilation and cfg.parent_dilation > 1:
                    pmask = F.max_pool2d(parent, cfg.parent_dilation, stride=1, padding=cfg.parent_dilation // 2)
                else:
                    pmask = parent
                trans_kl = trans_kl + bernoulli_kl_hard(e_eff[n], pf_logit[n]) * pmask
            parent = e_eff[n]
        K = torch.stack([e.mean(dim=(2, 3)) for e in e_eff], 1).sum(1)      # (B,1) tick depth
        info = {"skl_level": skl_level, "trans_kl": trans_kl.sum(dim=(2, 3)), "e": e_eff, "K": K,
                "e_soft_level": e_soft_level}
        return decoded, new, info

    # ----------------------------------------------------------------- training forward (ELBO)
    def loss(self, obs, beta=1.0, force_tick=False):
        cfg = self.cfg
        B, T = obs.shape[:2]
        feats = self._bottom_up(obs)
        state = self._init_state(B, obs.device)
        recon = torch.zeros(B, device=obs.device)
        tkl = torch.zeros(B, 1, device=obs.device)
        Ksum = torch.zeros(B, 1, device=obs.device)
        per_level_e = torch.zeros(cfg.levels, device=obs.device)
        per_level_skl = torch.zeros(cfg.levels, device=obs.device)
        soft_tick = torch.zeros((), device=obs.device)        # accumulates the SOFT tick fraction (upper levels)
        var = cfg.dec_stddev ** 2
        for t in range(T):
            x_t = [feats[n][:, t] for n in range(cfg.levels)]
            dec, state, info = self._step(state, x_t, mode="posterior", force_tick=force_tick)
            se = ((obs[:, t] - dec) ** 2).reshape(B, -1).sum(-1)
            recon = recon + 0.5 * se / var + 0.5 * math.log(2 * math.pi * var) * dec[0].numel()
            tkl = tkl + info["trans_kl"]
            Ksum = Ksum + info["K"]
            per_level_e = per_level_e + torch.stack([e.mean() for e in info["e"]])
            per_level_skl = per_level_skl + torch.stack([s.sum(dim=(2, 3)).mean() for s in info["skl_level"]])
            if cfg.levels > 1:
                soft_tick = soft_tick + torch.stack([info["e_soft_level"][n].mean() for n in range(1, cfg.levels)]).mean()
        nll = recon.mean()
        fnp = list(cfg.free_nats_per_level)[:cfg.levels] + [0.0] * max(0, cfg.levels - len(cfg.free_nats_per_level))
        floors = torch.tensor(fnp, device=obs.device) * T
        state_kl = torch.maximum(per_level_skl, floors).sum() if any(fnp) else per_level_skl.sum()
        trans_kl = tkl.mean()
        elbo = nll + beta * (state_kl + cfg.transition_kl_weight * trans_kl)
        soft_tick_rate = soft_tick / T                                     # mean soft tick fraction over the sequence
        over_tick = F.relu(soft_tick_rate - cfg.target_tick_rate)          # ONE-SIDED: penalise only ABOVE target
        loss = elbo / T + cfg.lambda_e * over_tick                         # over-tick penalty OUTSIDE beta-warmup
        metrics = {"loss": loss.detach(), "nll": (nll / T).detach(), "state_kl": (state_kl / T).detach(),
                   "trans_kl": (trans_kl / T).detach(), "K_mean": (Ksum.mean() / T).detach(),
                   "e_per_level": (per_level_e / T).detach(), "skl_per_level": (per_level_skl / T).detach(),
                   "soft_tick_rate": soft_tick_rate.detach(), "over_tick": over_tick.detach()}
        return loss, metrics

    # ----------------------------------------------------------------- open-loop rollout
    @torch.no_grad()
    def open_loop(self, obs, ctx_len, horizon):
        cfg = self.cfg
        B = obs.shape[0]
        feats = self._bottom_up(obs[:, :ctx_len])
        state = self._init_state(B, obs.device)
        for t in range(ctx_len):
            x_t = [feats[n][:, t] for n in range(cfg.levels)]
            _, state, _ = self._step(state, x_t, mode="posterior")
        preds, Ks, emaps = [], [], []
        for _ in range(horizon):
            dec, state, info = self._step(state, None, mode="prior")
            preds.append(dec); Ks.append(info["K"]); emaps.append(info["e"][-1])  # top-level e-map
        pred = torch.stack(preds, 1).clamp(0, 1)
        K = torch.stack(Ks, 1).squeeze(-1)
        emap = torch.stack(emaps, 1)
        return pred, K, emap
