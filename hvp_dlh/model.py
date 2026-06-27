"""Dynamic Latent Hierarchy (DLH) — faithful PyTorch port (arXiv:2212.14376).

Per timestep, each level n holds a two-component temporal Mixture-of-Gaussians (§2.2):
  static component  p(s^n_t | e=0) = persisted belief p(s^n_{t-1})    (carried, no network)
  change component  p(s^n_t | e=1) = p_theta(s^n_t | d^n, c^n)         (learned transition)
The binary indicator e^n_t is inferred NON-parametrically by a hard KL model-selection (Eq. 8):
  e=1 (change) iff KL[q || change] < KL[q || static].
Nested timescales (§2.3): e^n=1 requires e^{n-1}=1, so the changing levels form a contiguous block
from the bottom (level 0 = bottom always changes). The cut K_t is the adaptive abstraction depth.

Three deterministic pathways (Fig 1c): bottom-up x (image features), top-down c (context from above),
temporal d (per-level recurrent memory). ELBO (Eq. 5): recon NLL + state-KL + Bernoulli transition-KL.

Batched & masked: K_t differs per sample, so the unroll is dense with per-sample e masks
(s_t = e*s_change + (1-e)*s_prev); the nested constraint is a cumulative product down the levels.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_dlh.config import DLHConfig
from hvp_dlh.modules import (ConvImageEncoder, ConvImageDecoder, GaussianHead, mlp,
                             gaussian_kl, bernoulli_kl_hard)


class DLH(nn.Module):
    def __init__(self, cfg: DLHConfig):
        super().__init__()
        self.cfg = cfg
        N, S, D = cfg.levels, cfg.stoch_dim, cfg.det_dim

        # ---- image conv enc/dec ----
        self.img_enc = ConvImageEncoder(cfg.image_channels, cfg.conv_channels, cfg.conv_kernel,
                                        cfg.image_size, feat_dim=D, project=cfg.enc_to_feat)
        self.img_dec = ConvImageDecoder(D, cfg.image_channels, cfg.conv_channels, cfg.image_size,
                                        out_activation=cfg.dec_out_activation)

        # ---- bottom-up feature MLPs f_enc^n: x^n -> x^{n+1}  (n = 0 .. N-2) ----
        self.enc = nn.ModuleList([mlp(D, cfg.enc_mlp, D) for _ in range(N - 1)])
        # ---- top-down context MLPs f_dec^n: [s^n, c^n] -> c^{n-1}  (n = 0 .. N-1) ----
        self.dec = nn.ModuleList([mlp(S + D, cfg.dec_mlp, D) for _ in range(N)])
        # ---- temporal GRUs (d^n) and prior-factor GRUs ----
        self.tem = nn.ModuleList([nn.GRUCell(S, cfg.temporal_gru) for _ in range(N)])
        self.pf_gru = nn.ModuleList([nn.GRUCell(S, cfg.prior_factor_gru) for _ in range(N)]
                                    if cfg.separate_prior_factor_gru else [])
        self.pf_head = nn.ModuleList(
            [mlp(cfg.prior_factor_gru if cfg.separate_prior_factor_gru else D, (D,), 1)
             for _ in range(N)])
        # ---- stochastic heads: posterior q(s^n|x^n,c^n) and change-prior p(s^n|d^n,c^n) ----
        self.post = nn.ModuleList([GaussianHead(D + D, cfg.post_mlp, S, cfg.min_std, cfg.std_bias)
                                   for _ in range(N)])
        self.prior = nn.ModuleList([GaussianHead(cfg.temporal_gru + D, cfg.prior_mlp, S,
                                                 cfg.min_std, cfg.std_bias) for _ in range(N)])

    # ----------------------------------------------------------------- helpers
    def _bottom_up(self, obs):
        """obs (B,T,C,H,W) -> list over levels of (B,T,D) bottom-up features x^n."""
        B, T = obs.shape[:2]
        x0 = self.img_enc(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
        feats = [x0]
        for n in range(self.cfg.levels - 1):
            feats.append(self.enc[n](feats[-1]))
        return feats

    def _init_state(self, B, device):
        N, S = self.cfg.levels, self.cfg.stoch_dim
        z = lambda d: torch.zeros(B, d, device=device)
        return {
            "s": [z(S) for _ in range(N)],
            "d": [z(self.cfg.temporal_gru) for _ in range(N)],
            "pf": [z(self.cfg.prior_factor_gru) for _ in range(N)],
            "bmean": [z(S) for _ in range(N)],            # static-prior (persisted belief) mean
            "bstd": [torch.ones(B, S, device=device) for _ in range(N)],
        }

    def _contexts(self, s_list):
        """Top-down sweep: c^n for each level from states above. Returns [c^0..c^{N-1}] and c_bottom."""
        N = self.cfg.levels
        c = torch.zeros(s_list[0].shape[0], self.cfg.det_dim, device=s_list[0].device)
        cs = [None] * N
        for n in range(N - 1, -1, -1):
            cs[n] = c
            c = self.dec[n](torch.cat([s_list[n], c], -1))
        return cs, c

    # ----------------------------------------------------------------- one timestep
    def _step(self, state, x_t, mode, force_tick=False):
        """One timestep. mode='posterior' (obs x_t available, hard-KL e) or 'prior' (free-run, sample e).
        force_tick=True forces every level to update (dense warmup bootstrap).
        Returns decoded image (B,C,H,W), new state, and dict of per-step tensors (losses + e_eff + K)."""
        cfg, N = self.cfg, self.cfg.levels
        B = state["s"][0].shape[0]
        dev = state["s"][0].device

        # ---- prior-factor logits from history (summarise < t) ----
        pf_in = [state["pf"][n] if cfg.separate_prior_factor_gru else state["d"][n] for n in range(N)]
        pf_logit = [self.pf_head[n](pf_in[n]) for n in range(N)]    # each (B,1)

        # ===== Sweep A: e-decision =====
        cs_static, _ = self._contexts(state["s"])                  # contexts under static assumption
        e_eff, parent = [None] * N, torch.ones(B, 1, device=dev)
        for n in range(N):
            if n == 0:
                e_eff[0] = torch.ones(B, 1, device=dev)            # bottom always changes (q(e^1=1)=1)
                parent = e_eff[0]
                continue
            if force_tick:                                         # dense warmup bootstrap
                e_raw = torch.ones(B, 1, device=dev)
            elif mode == "posterior":
                mu_c, sg_c = self.prior[n](torch.cat([state["d"][n], cs_static[n]], -1))
                mu_q, sg_q = self.post[n](torch.cat([x_t[n], cs_static[n]], -1))
                kl_c = gaussian_kl(mu_q, sg_q, mu_c, sg_c).sum(-1, keepdim=True)
                kl_s = gaussian_kl(mu_q, sg_q, state["bmean"][n], state["bstd"][n]).sum(-1, keepdim=True)
                e_raw = (kl_c < kl_s).float()
            else:                                                  # free-run: sample from prior factor
                e_raw = torch.bernoulli(torch.sigmoid(pf_logit[n]))
            e_eff[n] = e_raw * parent                              # nested constraint
            parent = e_eff[n]

        # ===== Sweep B: top-down state sampling with decided e =====
        c = torch.zeros(B, cfg.det_dim, device=dev)
        new = {k: [None] * N for k in ("s", "d", "pf", "bmean", "bstd")}
        state_kl = torch.zeros(B, 1, device=dev)
        skl_level = [torch.zeros(B, 1, device=dev) for _ in range(N)]
        for n in range(N - 1, -1, -1):
            mu_c, sg_c = self.prior[n](torch.cat([state["d"][n], c], -1))
            if mode == "posterior":
                mu_q, sg_q = self.post[n](torch.cat([x_t[n], c], -1))
                s_changed = mu_q + sg_q * torch.randn_like(sg_q)
                # ELBO term (b) breakdown: KL to the SELECTED MoG component (change if tick, static if frozen)
                kl_change = gaussian_kl(mu_q, sg_q, mu_c, sg_c).sum(-1, keepdim=True)
                kl_static = gaussian_kl(mu_q, sg_q, state["bmean"][n], state["bstd"][n]).sum(-1, keepdim=True)
                skl_level[n] = e_eff[n] * kl_change + (1 - e_eff[n]) * kl_static
                state_kl = state_kl + skl_level[n]
            else:
                s_changed = mu_c + sg_c * torch.randn_like(sg_c)
            e = e_eff[n]
            s_n = e * s_changed + (1 - e) * state["s"][n]
            new["s"][n] = s_n
            # persisted belief (static prior for next step) = last change-prior where changed
            new["bmean"][n] = e * mu_c + (1 - e) * state["bmean"][n]
            new["bstd"][n] = e * sg_c + (1 - e) * state["bstd"][n]
            # temporal + prior-factor recurrences advance only on change
            d_upd = self.tem[n](s_n, state["d"][n])
            new["d"][n] = e * d_upd + (1 - e) * state["d"][n]
            if cfg.separate_prior_factor_gru:
                pf_upd = self.pf_gru[n](s_n, state["pf"][n])
                new["pf"][n] = e * pf_upd + (1 - e) * state["pf"][n]
            else:
                new["pf"][n] = new["d"][n]
            c = self.dec[n](torch.cat([s_n, c], -1))
        decoded = self.img_dec(c)

        # ---- transition KL (term c): predict the hard e (skip bottom; mask by parent active) ----
        trans_kl, parent = torch.zeros(B, 1, device=dev), torch.ones(B, 1, device=dev)
        for n in range(N):
            if n >= 1:
                trans_kl = trans_kl + bernoulli_kl_hard(e_eff[n], pf_logit[n]) * parent
            parent = e_eff[n]

        K = torch.stack([e for e in e_eff], 1).sum(1)               # (B,1) update depth this step
        info = {"state_kl": state_kl, "trans_kl": trans_kl, "e": e_eff, "K": K,
                "skl_level": skl_level}
        return decoded, new, info

    # ----------------------------------------------------------------- training forward (ELBO)
    def loss(self, obs, beta=1.0, force_tick=False):
        """obs (B,T,C,H,W) in [0,1]. Teacher-forced ELBO over the full sequence.
        force_tick=True forces all levels to update every step (dense warmup bootstrap)."""
        cfg = self.cfg
        B, T = obs.shape[:2]
        feats = self._bottom_up(obs)                                # list[N] of (B,T,D)
        state = self._init_state(B, obs.device)
        recon = torch.zeros(B, device=obs.device)
        skl = torch.zeros(B, 1, device=obs.device)
        tkl = torch.zeros(B, 1, device=obs.device)
        Ksum = torch.zeros(B, 1, device=obs.device)
        per_level_e = torch.zeros(cfg.levels, device=obs.device)
        per_level_skl = torch.zeros(cfg.levels, device=obs.device)
        var = cfg.dec_stddev ** 2
        for t in range(T):
            x_t = [feats[n][:, t] for n in range(cfg.levels)]
            dec, state, info = self._step(state, x_t, mode="posterior", force_tick=force_tick)
            se = ((obs[:, t] - dec) ** 2).reshape(B, -1).sum(-1)
            recon = recon + 0.5 * se / var + 0.5 * math.log(2 * math.pi * var) * dec[0].numel()
            skl = skl + info["state_kl"]
            tkl = tkl + info["trans_kl"]
            Ksum = Ksum + info["K"]
            per_level_e = per_level_e + torch.stack([e.mean() for e in info["e"]])
            per_level_skl = per_level_skl + torch.stack([s.mean() for s in info["skl_level"]])
        if cfg.free_nats > 0:
            skl = torch.clamp(skl, min=cfg.free_nats * T)
        nll = recon.mean()
        state_kl = skl.mean()
        trans_kl = tkl.mean()
        elbo = nll + beta * (state_kl + cfg.transition_kl_weight * trans_kl)
        loss = elbo / T
        metrics = {
            "loss": loss.detach(), "nll": (nll / T).detach(), "state_kl": (state_kl / T).detach(),
            "trans_kl": (trans_kl / T).detach(), "K_mean": (Ksum.mean() / T).detach(),
            "e_per_level": (per_level_e / T).detach(),
            "skl_per_level": (per_level_skl / T).detach(),
        }
        return loss, metrics

    # ----------------------------------------------------------------- open-loop rollout
    @torch.no_grad()
    def open_loop(self, obs, ctx_len, horizon):
        """Context frames use the posterior; then free-run the prior for `horizon` steps.
        Returns predicted frames (B, horizon, C, H, W) in [0,1] and the per-step K (B,horizon)."""
        cfg = self.cfg
        B = obs.shape[0]
        feats = self._bottom_up(obs[:, :ctx_len])
        state = self._init_state(B, obs.device)
        for t in range(ctx_len):
            x_t = [feats[n][:, t] for n in range(cfg.levels)]
            _, state, _ = self._step(state, x_t, mode="posterior")
        preds, Ks = [], []
        for _ in range(horizon):
            dec, state, info = self._step(state, None, mode="prior")
            preds.append(dec)
            Ks.append(info["K"])
        pred = torch.stack(preds, 1).clamp(0, 1)
        K = torch.stack(Ks, 1).squeeze(-1)
        return pred, K
