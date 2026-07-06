"""RHVD-mini — Stage-2 system per docs/STAGE2_DESIGN.md (+ 4-frame-window addendum).

Minimal MilliVid-style hierarchical latent video diffusion with a LONG COARSE memory, built for
the pre-registered H1/H2 rollout claims on the momentum looping testbed:

  H1: hier-long (48 coarse + 6 fine ctx) beats hier-short AND the token-budget-matched flat
      (18 fine ctx) on hard re-entry frames over 200-frame AR rollouts.
  H2: a matched regressor (reg-long) rolled out with the same scheduler fades vs diffusion.

Design (train == rollout layout, no stride mismatch):
  * predict a WINDOW of 4 consecutive frames per step, both levels (320 target tokens);
  * context: n_ctx_c consecutive coarse + n_ctx_f consecutive fine frames (fine = last frames of
    the coarse window); flat arm: n_ctx_c=0, n_ctx_f=18 (1152 ctx tokens == hier-long's budget);
  * per-TARGET-FRAME action embedding (frame j gets a_{t+j}), per-frame ids — replaces the single
    summed action code of the single-frame harness;
  * v-pred cosine diffusion (DDIM 16 at eval) or direct regression (--arm reg-long);
  * EMA 0.999 (eval on EMA), constant LR + 2k warmup (restartable), atomic ckpt/resume,
    [STD-FIX] standardized latents throughout (ActionHierData), bf16 autocast opt-in;
  * AR rollout: ring buffers (48 coarse / n_ctx_f fine) warmed from GT, append 4 predicted frames
    per step; per-horizon fg-MSE/sharp, stratified by short-window re-entry hardness; results to
    .npz so arms trained in separate processes are compared with paired CIs (--compare).

  venv/Scripts/python.exe -u -m hvp_rev.rhvd --arm hier-long --steps 30000 --bf16
  venv/Scripts/python.exe -u -m hvp_rev.rhvd --probe_steps 50 --arm hier-long --bf16
  venv/Scripts/python.exe -u -m hvp_rev.rhvd --compare hier-long,flat-budget --seed 0
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev import hier_convae, viz, wb
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import (ActionHierData, cosine_alphas, decode01, fg_mse_px_frames,
                                fg_sharp, load_ckpt, save_ckpt)
from hvp_rev.faithful_nets import LAT, DiTBlock, LatentTokenizer, TimestepEmbedder, modulate
from hvp_rev.step1 import rel_positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 4          # predicted frames per step
SPILL_SEC = 10.0

ARMS = {
    # name: (n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f)
    #   stride_c: temporal stride of the coarse context/target (1 = legacy per-frame; 4 = slow
    #             clock: one coarse SLOT per 4 frames, anchor = block-end frame, ONE coarse
    #             target slot per window).
    #   n_hist_f: extra strided FULL-RES history slots in the fine stream (hybrid control only):
    #             anchors tj-7, tj-11, ..., depth 4*n_hist_f+6; separates "coarse abstraction"
    #             from "full-res temporal subsampling" with recency (6 consecutive) held equal.
    "hier-long":  (48,      6,       "diff", 1, 0),
    "hier-short": (6,       6,       "diff", 1, 0),
    "flat-budget": (0,      18,      "diff", 1, 0),  # 18*64 = 1152 ctx tokens == hier-long's budget
    "reg-long":   (48,      6,       "reg",  1, 0),
    # fixed-WINDOW corner of the 2x2 (hierarchy-benefit exhibit): flat with the SAME 48-frame
    # window as hier-long, paying 48*64=3072 ctx tokens (2.7x hier-long's budget). seq ~3392 --
    # run with --batch 8 (+timing probe first); spill guard protects the 15GB cap.
    "flat-long":  (0,       48,      "diff", 1, 0),
    # token-matched flat control for the REGRESSION core (added after the H2 surprise: reg-long
    # beat hier-long-diffusion at rollout with equal sharpness -- the hierarchy benefit must be
    # shown for the WINNING mechanism too, or the headline is diffusion-only).
    "reg-flat":   (0,       18,      "reg",  1, 0),
    # ---- H-grid arms (HGRID_DESIGN.md v2, testbed operating point B: world 768 / loop ~192 /
    #      T=480 / WARM=192; run with --train_path data/hgrid_* --world 768 --warm 192,
    #      uniform B=8 x 40k). d0 = flat, largest affordable window (48 frames << 192 loop).
    "d0-diff-long": (0,     48,      "diff", 1, 0),
    "d0-reg-long":  (0,     48,      "reg",  1, 0),
    # d1 = slow-clock hierarchy: 48 coarse slots @ stride 4 span 192 frames (one full loop) for
    # 768 ctx tokens + 6 recent fine (384) = 1152 ctx tokens; ~1424-token seq, B=8 safe.
    "d1-diff":    (48,      6,       "diff", 4, 0),
    "d1-reg":     (48,      6,       "reg",  4, 0),
    # hybrid strided-flat control (G-H3): 47 FULL-RES slots @ stride 4 (reach 191) + 6 recent
    # fine = 3392 ctx tokens (== d0-long budget, 2.9x d1's) at d1's temporal reach — isolates
    # coarse ABSTRACTION from temporal subsampling; token asymmetry favors the CONTROL.
    "d0-strided-flat": (0,  6,       "diff", 1, 47),
    # ---- spatial factorial (docs/SPATIAL_FACTORIAL.md, testbed A/loopmom): the missing W18
    #      coarse-history cell — 18 coarse + 6 fine = 672 ctx tokens vs flat-budget's 1152 at the
    #      SAME 18-frame window. Isolates Claim 3 (spatial abstraction) from window length.
    "spat18":     (18,      6,       "diff", 1, 0),
    "spat18-reg": (18,      6,       "reg",  1, 0),
    # ---- token-budget FRONTIER completion (author request 2026-07-06): every flat point gets a
    #      hierarchical partner at the SAME budget. hier-rich = flat-long's 3072 tokens spent as
    #      48 coarse + 36 fine (window saturated at 48 by the T=100 train clips — hierarchy cannot
    #      buy MORE window here, so the pre-registered honest expectation is CONVERGENCE at the
    #      frontier top, per the factorial's abstraction-cost result, NOT hier dominance).
    #      flat-short = hier-short's ~480-token partner (8 fine = 512 tokens).
    "hier-rich":      (48,  36,      "diff", 1, 0),
    "hier-rich-reg":  (48,  36,      "reg",  1, 0),
    "flat-short":     (0,   8,       "diff", 1, 0),
    "flat-short-reg": (0,   8,       "reg",  1, 0),
    # ---- decisive rollout pair (pre-committed rule PASSED 2026-07-05): d1 with the span-224 fix
    #      (56 slots @ stride 4 > effective loop ~213; 1552-token seq) vs d0-diff-long, both at
    #      WARM=224 on hgrid3. Full grid only if this pair's reachable-HARD rollout gate is green.
    "d1-56-diff": (56,      6,       "diff", 4, 0),
}


def log(*a):
    print(*a, flush=True)


# ================================================================================================
# Model: window DiT with per-target-frame actions. Reuses LatentTokenizer pieces + DiTBlock.
# ================================================================================================
class WindowDiT(nn.Module):
    def __init__(self, n_ctx_c=48, n_ctx_f=6, win=WIN, d=384, depth=8, heads=6, lat=LAT,
                 mode="diff", stride_c=1, n_hist_f=0):
        super().__init__()
        self.n_ctx_c, self.n_ctx_f, self.win, self.lat, self.d = n_ctx_c, n_ctx_f, win, lat, d
        self.mode, self.stride_c, self.n_hist_f = mode, stride_c, n_hist_f
        # strided coarse: ONE coarse target slot per window (anchor = last window frame);
        # legacy (stride 1): per-frame coarse targets.
        self.n_tgt_c = 1 if (stride_c > 1 and n_ctx_c > 0) else win
        span_c = n_ctx_c * stride_c
        hist_reach = (n_ctx_f + 1 + 4 * (n_hist_f - 1) + 1) if n_hist_f > 0 else 0
        n_frames = max(span_c, n_ctx_f, hist_reach)       # context REACH in frames
        self.base = n_frames                              # first target frame id
        self.tok = LatentTokenizer(lat, d, n_frames + win - 1, k=win)  # frame_embedding covers base+win
        self.time_embedding = TimestepEmbedder(d)
        self.blocks = nn.ModuleList([DiTBlock(d, heads, zero_gate=(mode == "diff"))
                                     for _ in range(depth)])
        self.norm_c = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_c_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.final_f_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head_c = nn.Linear(d, lat)
        self.head_f = nn.Linear(d, lat)
        for ada in (self.final_c_ada, self.final_f_ada):
            nn.init.constant_(ada[-1].weight, 0); nn.init.constant_(ada[-1].bias, 0)
        for head in (self.head_c, self.head_f):
            nn.init.constant_(head.weight, 0); nn.init.constant_(head.bias, 0)
        self.expected_seq = (n_ctx_c * 16 + (n_hist_f + n_ctx_f) * 64
                             + self.n_tgt_c * 16 + win * 64)

    def _ctx_tokens(self, ctx_c, ctx_f):
        """Context token list + (level,role,frame) ids, all in TRUE frame units (base = tj).
        Coarse slots (stride s): anchors at frame ids s-1, 2s-1, ..., base-1 (block ends).
        Fine stream: n_hist_f strided full-res slots (hybrid control; anchors base-n_ctx_f-1-4k,
        deepest first) followed by the n_ctx_f consecutive recent frames [base-n_ctx_f..base-1].
        ctx_f is passed pre-concatenated: [:, :n_hist_f] = hist slots, [:, n_hist_f:] = recent."""
        xs, lvs, rls, fms = [], [], [], []
        for f in range(self.n_ctx_c):
            x, lv, rl, fm = self.tok.tokenize_level(ctx_c[:, f], level=0, role=0,
                                                    frame_idx=(f + 1) * self.stride_c - 1)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        for k in range(self.n_hist_f):
            fid = self.base - (self.n_ctx_f + 1 + 4 * (self.n_hist_f - 1 - k))
            x, lv, rl, fm = self.tok.tokenize_level(ctx_f[:, k], level=1, role=0, frame_idx=fid)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        f0_fine = self.base - self.n_ctx_f
        for j in range(self.n_ctx_f):
            x, lv, rl, fm = self.tok.tokenize_level(ctx_f[:, self.n_hist_f + j], level=1, role=0,
                                                    frame_idx=f0_fine + j)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        return xs, lvs, rls, fms

    def forward(self, yc, yf, ctx_c, ctx_f, t, actions):
        """yc:(B,n_tgt_c,LAT,4,4) yf:(B,win,LAT,8,8) noisy targets (or dummies for reg);
        ctx_c:(B,n_ctx_c,...) ctx_f:(B,n_hist_f+n_ctx_f,...); t:(B,); actions:(B,win) one per
        target frame. Returns per-level outputs shaped like yc/yf (v-pred for diff, x0 for reg).
        Strided coarse (n_tgt_c=1): the single slot is conditioned on the SUM of the window's
        action embeddings (multiset of actions = sufficient statistic for the displacement)."""
        B = yf.shape[0]
        xs, lvs, rls, fms = self._ctx_tokens(ctx_c, ctx_f)
        n_ctx_tok = sum(x.shape[1] for x in xs)
        groups = []                                       # (kind, j, n_tok) in token order
        if self.n_tgt_c == 1:
            xc, lvc, rlc, fmc = self.tok.tokenize_level(yc[:, 0], level=0, role=1,
                                                        frame_idx=self.base + self.win - 1)
            xs += [xc]; lvs += [lvc]; rls += [rlc]; fms += [fmc]
            groups.append(("c", 0, xc.shape[1]))
            for j in range(self.win):
                xf_, lvf, rlf, fmf = self.tok.tokenize_level(yf[:, j], level=1, role=1,
                                                             frame_idx=self.base + j)
                xs += [xf_]; lvs += [lvf]; rls += [rlf]; fms += [fmf]
                groups.append(("f", j, xf_.shape[1]))
        else:
            for j in range(self.win):
                xc, lvc, rlc, fmc = self.tok.tokenize_level(yc[:, j], level=0, role=1,
                                                            frame_idx=self.base + j)
                xf_, lvf, rlf, fmf = self.tok.tokenize_level(yf[:, j], level=1, role=1,
                                                             frame_idx=self.base + j)
                xs += [xc, xf_]; lvs += [lvc, lvf]; rls += [rlc, rlf]; fms += [fmc, fmf]
                groups.append(("c", j, xc.shape[1])); groups.append(("f", j, xf_.shape[1]))
        x = torch.cat(xs, 1)
        assert x.shape[1] == self.expected_seq, f"seq {x.shape[1]} != expected {self.expected_seq}"
        level_id, role_id, frame_id = torch.cat(lvs, 1), torch.cat(rls, 1), torch.cat(fms, 1)

        # per-token conditioning: level+frame+role + [ctx: t=0 emb] / [tgt group: t emb + action emb]
        c = (self.tok.level_embedding(level_id) + self.tok.frame_embedding(frame_id)
             + self.tok.role_embedding(role_id))
        cond_ctx = self.time_embedding(torch.zeros(B, device=x.device))
        cond_tgt = self.time_embedding(t)
        a_emb = self.tok.action_embedding(actions + 1)     # (B,win,d)
        per_tok_cond = cond_ctx[:, None, :].expand(B, n_ctx_tok, self.d)
        chunks = [per_tok_cond]
        for kind, j, ntk in groups:
            a_j = a_emb.sum(1) if (kind == "c" and self.n_tgt_c == 1) else a_emb[:, j]
            cj = (cond_tgt + a_j)[:, None, :]              # (B,1,d)
            chunks.append(cj.expand(B, ntk, self.d))
        c = c + torch.cat(chunks, 1)

        with M._ac():
            for blk in self.blocks:
                x = blk(x, c)
        x = x.float()

        # unpatchify target tokens per group per level
        out_c = torch.empty(B, self.n_tgt_c, self.lat, 4, 4, device=x.device)
        out_f = torch.empty(B, self.win, self.lat, 8, 8, device=x.device)
        pos = n_ctx_tok
        for kind, j, ntk in groups:
            seg = x[:, pos:pos + ntk]; cseg = c[:, pos:pos + ntk].float(); pos += ntk
            if kind == "c":
                sc, scl = self.final_c_ada(cseg).chunk(2, -1)
                vc = self.head_c(modulate(self.norm_c(seg), sc, scl))
                out_c[:, j] = vc.transpose(1, 2).reshape(B, self.lat, 4, 4)
            else:
                sf, sfl = self.final_f_ada(cseg).chunk(2, -1)
                vf = self.head_f(modulate(self.norm_f(seg), sf, sfl))
                out_f[:, j] = vf.transpose(1, 2).reshape(B, self.lat, 8, 8)
        return out_c, out_f


class WindowDiffusion:
    """v-pred cosine diffusion over the whole 4-frame window (shared per-sample t)."""

    def __init__(self, net: WindowDiT, n_train_steps=1000):
        self.net = net
        self.Ttr = n_train_steps
        self.abar = cosine_alphas(n_train_steps)

    def loss(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        B, dev = tgt_f.shape[0], tgt_f.device
        ti = torch.randint(1, self.Ttr + 1, (B,), device=dev)
        ab = self.abar.to(dev)[ti].view(B, 1, 1, 1, 1)
        sa, sb = ab.sqrt(), (1 - ab).sqrt()
        nc, nf = torch.randn_like(tgt_c), torch.randn_like(tgt_f)
        yc, yf = sa * tgt_c + sb * nc, sa * tgt_f + sb * nf
        vc_t, vf_t = sa * nc - sb * tgt_c, sa * nf - sb * tgt_f
        vc, vf = self.net(yc, yf, ctx_c, ctx_f, ti.float() / self.Ttr, actions)
        return F.mse_loss(vc.float(), vc_t) + F.mse_loss(vf.float(), vf_t)

    @torch.no_grad()
    def sample(self, ctx_c, ctx_f, actions, n_steps=16):
        B, dev = ctx_f.shape[0], ctx_f.device
        abar = self.abar.to(dev)
        yc = torch.randn(B, self.net.n_tgt_c, LAT, 4, 4, device=dev)
        yf = torch.randn(B, WIN, LAT, 8, 8, device=dev)
        ts = torch.linspace(self.Ttr, 1, n_steps, device=dev).round().long().clamp(1, self.Ttr)
        for i in range(n_steps):
            ti = ts[i]
            ab = abar[ti].view(1, 1, 1, 1, 1)
            sa, sb = ab.sqrt(), (1 - ab).sqrt()
            t = torch.full((B,), ti.item() / self.Ttr, device=dev)
            vc, vf = self.net(yc, yf, ctx_c, ctx_f, t, actions)
            vc, vf = vc.float(), vf.float()
            x0c, x0f = sa * yc - sb * vc, sa * yf - sb * vf
            epc, epf = sb * yc + sa * vc, sb * yf + sa * vf
            if i < n_steps - 1:
                abn = abar[ts[i + 1]].view(1, 1, 1, 1, 1)
                yc = abn.sqrt() * x0c + (1 - abn).sqrt() * epc
                yf = abn.sqrt() * x0f + (1 - abn).sqrt() * epf
            else:
                yc, yf = x0c, x0f
        return yc, yf


class WindowRegression:
    """Direct x0 window prediction (H2 control). Same net/backbone; t fixed 0."""

    def __init__(self, net: WindowDiT):
        self.net = net

    def loss(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        B, dev = tgt_f.shape[0], tgt_f.device
        zc = torch.zeros_like(tgt_c); zf = torch.zeros_like(tgt_f)
        pc, pf = self.net(zc, zf, ctx_c, ctx_f, torch.zeros(B, device=dev), actions)
        return F.smooth_l1_loss(pc.float(), tgt_c) + F.smooth_l1_loss(pf.float(), tgt_f)

    @torch.no_grad()
    def sample(self, ctx_c, ctx_f, actions, n_steps=0):
        B, dev = ctx_f.shape[0], ctx_f.device
        zc = torch.zeros(B, self.net.n_tgt_c, LAT, 4, 4, device=dev)
        zf = torch.zeros(B, WIN, LAT, 8, 8, device=dev)
        return self.net(zc, zf, ctx_c, ctx_f, torch.zeros(B, device=dev), actions)


# ================================================================================================
# Data: window batches on top of ActionHierData's encoded+standardized latents.
# ================================================================================================
def ctx_reach(n_ctx_c, n_ctx_f, stride_c=1, n_hist_f=0):
    """Deepest context reach in frames (= model.base = first target frame id)."""
    hist_reach = (n_ctx_f + 1 + 4 * (n_hist_f - 1) + 1) if n_hist_f > 0 else 0
    return max(n_ctx_c * stride_c, n_ctx_f, hist_reach)


def window_batch(data, B, n_ctx_c, n_ctx_f, train=True, seed=None, frames=False,
                 stride_c=1, n_hist_f=0):
    """Context ends at frame tj-1. Coarse: n_ctx_c slots at stride_c (anchors tj-1-s*(m) block
    ends; stride 1 = legacy contiguous). Fine: n_hist_f strided full-res slots (anchors
    tj-n_ctx_f-1-4k, deepest first) ++ n_ctx_f consecutive [tj-n_ctx_f..tj-1].
    Targets = frames [tj..tj+WIN-1]; coarse target = ONE slot (anchor tj+WIN-1) when strided;
    actions[:, j] = acts[tj+j] (moves frame tj+j-1 -> tj+j)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pool = data.train_idx if train else data.val_idx
    n_frames = ctx_reach(n_ctx_c, n_ctx_f, stride_c, n_hist_f)
    span_c = n_ctx_c * stride_c
    n_tgt_c = 1 if (stride_c > 1 and n_ctx_c > 0) else WIN
    ctx_c = torch.empty(B, n_ctx_c, LAT, 4, 4)
    ctx_f = torch.empty(B, n_hist_f + n_ctx_f, LAT, 8, 8)
    tgt_c = torch.empty(B, n_tgt_c, LAT, 4, 4)
    tgt_f = torch.empty(B, WIN, LAT, 8, 8)
    acts = torch.empty(B, WIN, dtype=torch.long)
    gt = torch.empty(B, WIN, 1, 64, 64) if frames else None
    for b in range(B):
        ci = int(pool[rng.randint(len(pool))])
        tj = int(rng.randint(n_frames, data.T - WIN + 1))
        if n_ctx_c > 0:
            ctx_c[b] = data.zc[ci, tj - span_c + stride_c - 1:tj:stride_c]
        if n_hist_f > 0:
            ctx_f[b, :n_hist_f] = data.zf[ci, tj - (n_ctx_f + 1 + 4 * (n_hist_f - 1)):
                                          tj - n_ctx_f:4]
        ctx_f[b, n_hist_f:] = data.zf[ci, tj - n_ctx_f:tj]
        if n_tgt_c == 1:
            tgt_c[b] = data.zc[ci, tj + WIN - 1:tj + WIN]
        else:
            tgt_c[b] = data.zc[ci, tj:tj + WIN]
        tgt_f[b] = data.zf[ci, tj:tj + WIN]
        acts[b] = torch.from_numpy(data.acts[ci, tj:tj + WIN].astype(np.int64))
        if frames:
            gt[b] = torch.from_numpy(data.clips_u8[ci, tj:tj + WIN].astype(np.float32)
                                     / 127.5 - 1.0).unsqueeze(1)
    out = dict(ctx_c=ctx_c.to(DEV), ctx_f=ctx_f.to(DEV), tgt_c=tgt_c.to(DEV),
               tgt_f=tgt_f.to(DEV), actions=acts.to(DEV))
    if frames:
        out["gt"] = gt.to(DEV)
    return out


def hardness_at(data, ci, tframe, n_short=6, thr=0.5):
    """Re-entry hardness of GT frame tframe vs the SHORT window [tframe-n_short..tframe-1]:
    fraction of its sprite world-columns not covered by those frames (mem_gate's novel_fg6,
    made per-frame / k-free for rollout use)."""
    rel = rel_positions(data.acts[ci], data.speed)
    frame = data.clips_u8[ci, tframe].astype(np.float32) / 255.0
    fg = np.where(frame.max(axis=0) > thr)[0]
    if len(fg) == 0:
        return 0.0
    tw = (rel[tframe] + fg) % data.W
    cov = np.zeros(data.W, bool)
    for m in range(tframe - n_short, tframe):
        cov[(rel[m] + np.arange(64)) % data.W] = True
    return float((~cov[tw]).mean())


class DtLabeler:
    """[H-grid gate variable] Per-frame dt-since-last-visible + answerability labels.

    For GT frame tf of clip ci: its NOVEL sprite columns (not covered by the 6-frame short
    window, same notion as hardness_at) each have dt = tf - (last frame the column was in the
    viewport). Labels per record:
      dt   = median dt over novel columns that WERE seen before (1e9 if none were) — the
             capability axis: dt > 48 means the content re-enters from beyond ANY flat window.
      seen = fraction of novel columns visible at least once before tf (answerable-by-
             construction; gates read on seen >= 0.5, unseen rows reported separately).
    Trajectory-only (rel_positions from actions) — identical machinery to hardness_at, so the
    two axes stratify consistently. Requires monotonically increasing tf per clip (rollout order).
    """

    def __init__(self, data, warm, n_short=6, thr=0.5):
        self.data, self.warm, self.n_short, self.thr = data, warm, n_short, thr
        self._st = {}                                    # ci -> [rel, last_seen, next_unmarked_t]

    def _init_clip(self, ci):
        rel = rel_positions(self.data.acts[ci], self.data.speed)
        last = np.full(self.data.W, -(10 ** 9), np.int64)
        for m in range(self.warm):
            last[(rel[m] + np.arange(64)) % self.data.W] = m
        return [rel, last, self.warm]

    def label(self, ci, tf):
        st = self._st.get(ci)
        if st is None:
            st = self._st[ci] = self._init_clip(ci)
        rel, last, nxt = st
        assert tf >= nxt, f"DtLabeler needs monotonic tf per clip (got {tf} < {nxt})"
        for m in range(nxt, tf):                         # mark frames up to but NOT including tf
            last[(rel[m] + np.arange(64)) % self.data.W] = m
        st[2] = tf
        frame = self.data.clips_u8[ci, tf].astype(np.float32) / 255.0
        fg = np.where(frame.max(axis=0) > self.thr)[0]
        if len(fg) == 0:
            return 0.0, 1.0
        tw = (rel[tf] + fg) % self.data.W
        dt = tf - last[tw]                               # (n_fg,) per-column dt
        novel = dt > self.n_short
        if not novel.any():
            return 0.0, 1.0
        dtn = dt[novel]
        finite = dtn < 10 ** 8
        seen = float(finite.mean())
        dt_med = float(np.median(dtn[finite])) if finite.any() else 1e9
        return dt_med, seen


# ================================================================================================
# Training (EMA + warmup-constant LR + ckpt/resume + grids)
# ================================================================================================
class EMA:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(net).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for ps, p in zip(self.shadow.parameters(), net.parameters()):
            ps.lerp_(p.detach(), 1 - self.decay)
        for bs, b in zip(self.shadow.buffers(), net.buffers()):
            bs.copy_(b)


def train(arm, data, test, ae, steps, B, seed, lr=3e-4, warmup=2000, val_every=500,
          save_every=1000, grid_every=2000):
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                    n_hist_f=n_hist_f).to(DEV)
    wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
    ema = EMA(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ckpt = f"logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt"
    start = 1
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ema.shadow.load_state_dict(st["ema"])
        start = int(st["step"]) + 1
        log(f"[ckpt] resumed {ckpt} @ {st['step']}")
    log(f"[arm {arm}] mode={mode} ctx_c={n_ctx_c}(s{stride_c}) hist_f={n_hist_f} ctx_f={n_ctx_f} "
        f"seq={net.expected_seq} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M "
        f"steps {start}..{steps} B={B} seed={seed}")
    if start > steps:
        log(f"[arm {arm}] already trained; skipping to eval")
        return wrap, ema
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train(); run, t_last = 0.0, time.time()
    for s in range(start, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, s / warmup)
        bt = window_batch(data, B, n_ctx_c, n_ctx_f, train=True, stride_c=stride_c,
                          n_hist_f=n_hist_f)
        loss = wrap.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); ema.update(net)
        run += float(loss.detach())
        dt = time.time() - t_last; t_last = time.time()
        if dt > SPILL_SEC and s > start + 3:
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(),
                      ema=ema.shadow.state_dict())
            raise RuntimeError(f"[arm {arm}] step {s} took {dt:.1f}s -> WDDM spill; ckpt saved.")
        if s % val_every == 0:
            vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
            log(f"  [{arm} {s}/{steps}] loss {run/val_every:.4f}  {dt:.2f}s/step  peak {vram:.2f}GB")
            wb.log({"train/loss": run / val_every, "train/s_per_step": dt,
                    "train/vram_gb": vram, "train/lr": opt.param_groups[0]["lr"]}, step=s)
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(),
                      ema=ema.shadow.state_dict())
        if s % grid_every == 0:
            net.eval()
            vb = window_batch(test, 8, n_ctx_c, n_ctx_f, train=False, seed=777, frames=True,
                              stride_c=stride_c, n_hist_f=n_hist_f)
            wrap_eval = WindowDiffusion(ema.shadow, wrap.Ttr) if mode == "diff" else WindowRegression(ema.shadow)
            yc, yf = wrap_eval.sample(vb["ctx_c"], vb["ctx_f"], vb["actions"], n_steps=16)
            pred01 = decode01(ae, yf[:, -1])               # last window frame (horizon 4)
            gt01 = (vb["gt"][:, -1].clamp(-1, 1) + 1) * 0.5
            gpath = f"logs_rhcp/rhvd_{arm}_s{seed}_train{s}.png"
            viz.save_samples(gpath, [("pred_h4", pred01), ("GT", gt01)], n=8)
            vfg = float(fg_mse_px_frames(pred01, gt01).mean())
            net.train()
            log(f"      >> grid {gpath}  val fg-MSE(h4) {vfg:.4f}")
            wb.log({"val/fg_mse_h4": vfg}, step=s)
            wb.image("val/sample_grid", gpath, step=s)
    return wrap, ema


# ================================================================================================
# AR rollout eval -> npz (per-frame horizon/hardness/fg-MSE/sharp records)
# ================================================================================================
@torch.no_grad()
def rollout_eval(arm, wrap, ema, data, ae, seed, n_clips=100, nfe=16, chunk=16, warm=48,
                 oracle=False, tag=""):
    """warm: UNIVERSAL GT warmup across arms of one campaign — every arm starts rollout at the
    same frame so (clip, horizon) records pair exactly. Legacy campaign: 48. H-grid (testbed B):
    192 (= one full loop seen in GT before rollout).
    oracle=True [GO gate]: bypass the model — feed GT latents through the FULL rollout/buffer/
    metric path; fg-MSE must land at the AE round-trip ceiling at every horizon, which validates
    buffer alignment, stride phasing, decode, and labels in one shot. Writes *_oracle.npz."""
    n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[arm]
    net_eval = ema.shadow
    wrap_eval = WindowDiffusion(net_eval, wrap.Ttr) if mode == "diff" else WindowRegression(net_eval)
    n_frames = ctx_reach(n_ctx_c, n_ctx_f, stride_c, n_hist_f)
    n_tgt_c = 1 if (stride_c > 1 and n_ctx_c > 0) else WIN
    WARM = warm
    assert WARM >= n_frames, f"warm {WARM} must cover the context reach {n_frames}"
    if stride_c > 1:
        assert WARM % stride_c == 0, "warm must be a multiple of the coarse stride"
    T = data.T
    n_steps_roll = (T - WARM) // WIN
    clips = data.val_idx[:n_clips]
    dtl = DtLabeler(data, WARM)
    recs = dict(h=[], hard=[], fg=[], sharp=[], pers_fg=[], clip=[], dt=[], seen=[])
    viz_h = {1, WIN, 48, 96, 192, min(288, n_steps_roll * WIN)}     # rollout strip horizons
    viz_rows = []
    for c0 in range(0, len(clips), chunk):
        cs = clips[c0:c0 + chunk]
        B = len(cs)
        if n_ctx_c > 0:
            if stride_c > 1:   # slot buffer: GT anchors at frames stride-1, 2*stride-1, ..., WARM-1
                buf_c = data.zc[cs, WARM - n_ctx_c * stride_c + stride_c - 1:WARM:stride_c] \
                    .clone().to(DEV)
            else:
                buf_c = data.zc[cs, WARM - n_frames:WARM].clone().to(DEV)
        else:
            buf_c = torch.zeros(B, 1, LAT, 4, 4, device=DEV)
        buf_f = data.zf[cs, WARM - n_frames:WARM].clone().to(DEV)
        pers01 = decode01(ae, data.zf[cs, WARM - 1].to(DEV))           # persistence reference
        for srn in range(n_steps_roll):
            tj = WARM + srn * WIN
            torch.manual_seed(100000 + 1000 * (c0 // chunk) + srn)     # CRN across arms
            acts = torch.from_numpy(data.acts[cs, tj:tj + WIN].astype(np.int64)).to(DEV)
            ctx_c = buf_c[:, -n_ctx_c:] if n_ctx_c > 0 else buf_c[:, :0]
            if n_hist_f > 0:
                lo = n_frames - (n_ctx_f + 1 + 4 * (n_hist_f - 1))     # deepest hist anchor index
                ctx_f = torch.cat([buf_f[:, lo:n_frames - n_ctx_f:4], buf_f[:, -n_ctx_f:]], 1)
            else:
                ctx_f = buf_f[:, -n_ctx_f:]
            if oracle:
                yc = data.zc[cs, tj + WIN - 1:tj + WIN].to(DEV) if n_tgt_c == 1 else \
                    data.zc[cs, tj:tj + WIN].to(DEV)
                yf = data.zf[cs, tj:tj + WIN].to(DEV)
            else:
                yc, yf = wrap_eval.sample(ctx_c, ctx_f, acts, n_steps=nfe)
            if n_ctx_c > 0:
                keep = n_ctx_c if stride_c > 1 else n_frames
                buf_c = torch.cat([buf_c, yc], 1)[:, -keep:]
            buf_f = torch.cat([buf_f, yf], 1)[:, -n_frames:]
            for j in range(WIN):
                tf = tj + j
                gt01 = torch.from_numpy(
                    data.clips_u8[cs, tf].astype(np.float32) / 255.0).unsqueeze(1).to(DEV)
                p01 = decode01(ae, yf[:, j])
                fg = fg_mse_px_frames(p01, gt01)
                pfg = fg_mse_px_frames(pers01, gt01)
                sh = fg_sharp(p01, gt01)
                for bi_, ci in enumerate(cs):
                    recs["h"].append(tf - WARM + 1)
                    # hardness axis is ARM-INDEPENDENT (always the 6-frame short window) so the
                    # paired cross-arm comparison stratifies every arm identically
                    recs["hard"].append(hardness_at(data, ci, tf, n_short=6))
                    dt_med, seen = dtl.label(ci, tf)
                    recs["dt"].append(dt_med)
                    recs["seen"].append(seen)
                    recs["fg"].append(float(fg[bi_]))
                    recs["pers_fg"].append(float(pfg[bi_]))
                    recs["sharp"].append(sh)              # batch-mean sharp per frame (coarse record)
                    recs["clip"].append(int(ci))
                if c0 == 0 and (tf - WARM + 1) in viz_h:
                    viz_rows.append((tf - WARM + 1, p01[:6].cpu(), gt01[:6].cpu()))
        log(f"  [rollout {arm}] clips {c0}..{c0+B-1} done ({n_steps_roll} steps x {WIN} frames)")
    out = {k: np.asarray(v) for k, v in recs.items()}
    path = f"logs_rhcp/rhvd_eval_{arm}_s{seed}{'_oracle' if oracle else ''}{tag}.npz"
    np.savez(path, **out)
    log(f"[rollout {arm}{' ORACLE' if oracle else ''}] saved {path}: {len(out['fg'])} "
        f"frame-records, mean fg {out['fg'].mean():.4f} (persistence {out['pers_fg'].mean():.4f})")
    if viz_rows:
        vpath = f"logs_rhcp/rhvd_rollout_{arm}_s{seed}.png"
        panels = []
        for h, p, g in sorted(viz_rows, key=lambda r: r[0]):
            panels += [(f"pred_h{h}", p), (f"GT_h{h}", g)]
        viz.save_samples(vpath, panels, n=6)
        log(f"[rollout {arm}] strip {vpath}")
        wb.image("eval/rollout_strip", vpath)
    hs, fgs = out["h"], out["fg"]
    wb.summary({"eval/fg_mse_all": float(fgs.mean()),
                "eval/pers_fg": float(out["pers_fg"].mean()),
                "eval/sharp": float(out["sharp"].mean()),
                "eval/fg_mse_h1_48": float(fgs[hs <= 48].mean()),
                "eval/fg_mse_h49_192": float(fgs[(hs > 48) & (hs <= 192)].mean()),
                "eval/fg_mse_h193plus": float(fgs[hs > 192].mean()) if (hs > 192).any() else 0.0,
                "eval/fg_mse_beyondwin_hard": float(
                    fgs[(out["dt"] > 48) & (out["hard"] > 0.1) & (out["seen"] >= 0.5)].mean())
                if ((out["dt"] > 48) & (out["hard"] > 0.1) & (out["seen"] >= 0.5)).any() else 0.0})
    return path


def paired_ci(delta, clips=None, n_boot=10000, seed=0):
    """95% CI of the mean paired delta. With clips given, resamples CLIPS (cluster bootstrap):
    frames within one AR rollout are strongly autocorrelated (measured lag-1 ~ +0.86), so
    frame-level iid resampling understates CI widths 2-7x [validity audit 2026-07-03]."""
    rng = np.random.RandomState(seed)
    if clips is None:
        idx = rng.randint(0, len(delta), (n_boot, len(delta)))
        m = delta[idx].mean(axis=1)
    else:
        uc = np.unique(clips)
        s = np.array([delta[clips == c].sum() for c in uc])
        n = np.array([(clips == c).sum() for c in uc], dtype=np.float64)
        pick = rng.randint(0, len(uc), (n_boot, len(uc)))
        m = s[pick].sum(axis=1) / n[pick].sum(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _fg_empty_mask(clip_arr, h_arr, test_path, warm, thr=0.5):
    """True where the GT frame has NO foreground above thr. [validity audit] ~21% of rollout
    records have an empty GT sprite mask; long_eval.fg_mse silently falls back to full-frame MSE
    there and hardness_at bins them 'easy' -- they are excluded from the stratified tables and
    reported as their own row. [H-grid fix] test_path/warm are REQUIRED args: the old defaults
    silently indexed the loopmom file for any campaign."""
    raw = np.load(test_path, mmap_mode="r")
    cache, out = {}, np.zeros(len(clip_arr), bool)
    for i in range(len(clip_arr)):
        key = (int(clip_arr[i]), int(h_arr[i]))
        if key not in cache:
            frame = raw[key[0], warm + key[1] - 1]
            cache[key] = bool(frame.max() <= thr * 255.0)
        out[i] = cache[key]
    return out


def compare(arms, seed, test_path, warm):
    """Paired per-frame comparison of rollout npz files (same clips/horizons across arms).
    Cluster (clip-level) bootstrap CIs; empty-GT-foreground records excluded from strat bins.
    H-grid npz (with dt/seen fields) additionally get the pre-registered GATE table: dt-since-
    last-visible bins x HARD, on answerable (seen>=0.5) records — G-H1/2/3 read on dt>48 rows."""
    d = {}
    for a in arms:
        z = np.load(f"logs_rhcp/rhvd_eval_{a}_s{seed}.npz")
        d[a] = z
    base = arms[0]
    zb = d[base]
    empty = _fg_empty_mask(zb["clip"], zb["h"], test_path, warm)
    has_dt = "dt" in zb.files
    print(f"\n{'='*100}\n  RHVD ROLLOUT COMPARISON (paired per-frame fg-MSE, base = {base}, seed {seed};"
          f" CLUSTER bootstrap by clip)\n  test={test_path} warm={warm}; empty-GT-fg records excluded"
          f" from bins: {int(empty.sum())}/{len(empty)} ({empty.mean()*100:.1f}%)\n{'='*100}")
    hbins = [(1, 8, "h 1-8"), (9, 24, "h 9-24"), (25, 60, "h 25-60"), (61, 120, "h 61-120"),
             (121, 192, "h 121-192"), (193, 288, "h 193-288")]
    hardbins = [(-0.01, 0.001, "easy"), (0.001, 0.1, "mid"), (0.1, 1.01, "HARD")]
    dtbins = [(0, 48, "dt<=48"), (48, 96, "dt 49-96"), (96, 192, "dt 97-192"), (192, 1e12, "dt>192")]
    for a in arms[1:]:
        za = d[a]
        assert np.array_equal(za["clip"], zb["clip"]) and np.array_equal(za["h"], zb["h"]), \
            "eval sets differ; re-run with same protocol"
        print(f"\n  {a} vs {base} (sharp {za['sharp'].mean():.4f} vs {zb['sharp'].mean():.4f}; "
              f"persistence {zb['pers_fg'].mean():.4f}):")
        print(f"  {'horizon':<10}{'hardness':<8}{'n':>7}{'base':>10}{'arm':>10}{'delta':>10}{'95% CI':>22}{'verdict':>9}")
        for hlo, hhi, hname in hbins:
            for blo, bhi, bname in hardbins:
                m = (zb["h"] >= hlo) & (zb["h"] <= hhi) & (zb["hard"] > blo) & (zb["hard"] <= bhi) & ~empty
                if m.sum() < 30:
                    continue
                delta = zb["fg"][m] - za["fg"][m]
                lo, hi = paired_ci(delta, clips=zb["clip"][m])
                v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "null")
                print(f"  {hname:<10}{bname:<8}{int(m.sum()):>7}{zb['fg'][m].mean():>10.4f}"
                      f"{za['fg'][m].mean():>10.4f}{delta.mean():>+10.4f}   [{lo:+.4f},{hi:+.4f}]{v:>9}")
        if has_dt and "dt" in za.files:
            print(f"\n  [GATE TABLE] dt-since-last-visible x HARD, answerable (seen>=0.5), ~empty"
                  f" — gates read on dt>48 rows:")
            print(f"  {'dt bin':<12}{'hardness':<8}{'n':>7}{'base':>10}{'arm':>10}{'delta':>10}{'95% CI':>22}{'verdict':>9}")
            ans = (zb["seen"] >= 0.5) & ~empty
            for dlo, dhi, dname in dtbins:
                for blo, bhi, bname in [(0.001, 1.01, "mid+HARD"), (0.1, 1.01, "HARD")]:
                    m = ans & (zb["dt"] > dlo) & (zb["dt"] <= dhi) & (zb["hard"] > blo) & (zb["hard"] <= bhi)
                    if m.sum() < 30:
                        continue
                    delta = zb["fg"][m] - za["fg"][m]
                    lo, hi = paired_ci(delta, clips=zb["clip"][m])
                    v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "null")
                    print(f"  {dname:<12}{bname:<8}{int(m.sum()):>7}{zb['fg'][m].mean():>10.4f}"
                          f"{za['fg'][m].mean():>10.4f}{delta.mean():>+10.4f}   [{lo:+.4f},{hi:+.4f}]{v:>9}")
            un = (zb["seen"] < 0.5) & (zb["dt"] > 48) & ~empty
            if un.sum() >= 30:
                delta = zb["fg"][un] - za["fg"][un]
                lo, hi = paired_ci(delta, clips=zb["clip"][un])
                print(f"  {'UNSEEN dt>48':<20}{int(un.sum()):>7}{zb['fg'][un].mean():>10.4f}"
                      f"{za['fg'][un].mean():>10.4f}{delta.mean():>+10.4f}   [{lo:+.4f},{hi:+.4f}]"
                      f"{'(report-only)':>14}")
        # empty-fg row (bg-only frames; full-frame-MSE fallback semantics) + ALL row, both clustered
        for label, m in (("bg-only", empty), ("ALL", np.ones(len(empty), bool))):
            delta = zb["fg"][m] - za["fg"][m]
            lo, hi = paired_ci(delta, clips=zb["clip"][m])
            v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "null")
            print(f"  {label:<10}{'':<8}{int(m.sum()):>7}{zb['fg'][m].mean():>10.4f}{za['fg'][m].mean():>10.4f}"
                  f"{delta.mean():>+10.4f}   [{lo:+.4f},{hi:+.4f}]{v:>9}")


# ================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hier-long", choices=list(ARMS))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--n_eval_clips", type=int, default=100)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--probe_steps", type=int, default=0, help=">0: timing/VRAM probe, then exit")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--go_gate", action="store_true",
                    help="GT-oracle rollout through the full eval path; PASS iff fg-MSE ~= AE ceiling")
    ap.add_argument("--eval_tag", default="", help="suffix for the eval npz (mid-training trend "
                    "probes write _probe10k etc. so the final eval is never overwritten)")
    ap.add_argument("--compare", default="", help="comma list of arms to compare from saved npz")
    # ---- H-grid campaign args (defaults = legacy loopmom campaign, unchanged) ----
    ap.add_argument("--train_path", default="data/loopmom_train_3000x100.npy")
    ap.add_argument("--test_path", default="data/loopmom_test_500x200.npy")
    ap.add_argument("--world", type=int, default=192, help="torus world width (rel_positions mod)")
    ap.add_argument("--wspeed", type=int, default=4)
    ap.add_argument("--warm", type=int, default=48, help="universal GT warmup (H-grid: 192)")
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--ae_path", default="logs_rhcp/hier_ae_mom.pt")
    args = ap.parse_args()
    M._BF16 = bool(args.bf16)

    if args.compare:
        compare([a.strip() for a in args.compare.split(",")], args.seed,
                test_path=args.test_path, warm=args.warm)
        return

    if args.arm.startswith(("d0-", "d1-", "d2-")):
        assert args.warm in (192, 224) and "hgrid" in args.train_path and args.world == 768, \
            "H-grid arms require hgrid data, world 768, warm 192 (span-192 era) or 224 (span-224 pair)"

    if not args.no_wandb and args.probe_steps == 0 and not args.go_gate:
        n_ctx_c_, n_ctx_f_, mode_, stride_c_, n_hist_f_ = ARMS[args.arm]
        run_name = f"rhvd-{args.arm}-s{args.seed}" + ("-smoke" if args.smoke else "")
        wb.init(run_name,
                config=dict(vars(args), n_ctx_c=n_ctx_c_, n_ctx_f=n_ctx_f_, mode=mode_,
                            stride_c=stride_c_, n_hist_f=n_hist_f_,
                            win=WIN, campaign="hgrid" if args.warm == 192 else "loopmom"),
                tags=["hgrid" if args.warm == 192 else "loopmom", mode_])

    ae = hier_convae.load(args.ae_path)
    data = ActionHierData(ae, args.train_path, args.train_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=4, W=args.world, speed=args.wspeed)
    test = ActionHierData(ae, args.test_path, args.test_path.replace(".npy", "_actions.npy"),
                          n_ctx=6, k=4, n_val=args.n_val, W=args.world, speed=args.wspeed,
                          stats=data.stats)

    if args.probe_steps > 0:
        n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[args.arm]
        net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                        n_hist_f=n_hist_f).to(DEV)
        wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.probe_steps):
            bt = window_batch(data, args.batch, n_ctx_c, n_ctx_f, train=True,
                              stride_c=stride_c, n_hist_f=n_hist_f)
            loss = wrap.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.probe_steps
        vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
        print(f"[TIMING {args.arm}] {dt:.3f} s/step | peak {vram:.2f} GB | "
              f"{args.steps} steps => {args.steps*dt/3600:.1f} h")
        return

    if args.go_gate:
        # [GO GATE] GT-oracle rollout through the full buffer/metric path: fg-MSE must land at
        # the AE round-trip ceiling (0.0106 on hgrid test) at every horizon.
        n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[args.arm]
        net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                        n_hist_f=n_hist_f).to(DEV)
        wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
        path = rollout_eval(args.arm, wrap, EMA(net), test, ae, args.seed, n_clips=16,
                            nfe=1, warm=args.warm, oracle=True)
        z = np.load(path)
        by_h = [(hn, float(z["fg"][(z["h"] >= lo) & (z["h"] <= hi)].mean()))
                for lo, hi, hn in [(1, 48, "h1-48"), (49, 192, "h49-192"), (193, 288, "h193+")]]
        ceiling = 0.0106
        ok = all(v <= ceiling * 1.6 for _, v in by_h)
        print(f"[GO-GATE {args.arm}] oracle fg-MSE by horizon: "
              + "  ".join(f"{hn} {v:.4f}" for hn, v in by_h)
              + f"  (AE ceiling {ceiling:.4f})  -> {'PASS' if ok else 'FAIL'}")
        return

    steps = 300 if args.smoke else args.steps
    if args.eval_only:
        n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f = ARMS[args.arm]
        net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode, stride_c=stride_c,
                        n_hist_f=n_hist_f).to(DEV)
        wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
        ema = EMA(net)
        ckpt = f"logs_rhcp/rhvd_{args.arm}_s{args.seed}_t{args.steps}.pt"
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        ema.shadow.load_state_dict(st["ema"])
        log(f"[eval_only] loaded {ckpt} @ step {st['step']}")
    else:
        wrap, ema = train(args.arm, data, test, ae, steps, args.batch, args.seed)
    rollout_eval(args.arm, wrap, ema, test, ae, args.seed,
                 n_clips=(8 if args.smoke else args.n_eval_clips), nfe=args.nfe, warm=args.warm,
                 tag=args.eval_tag)
    wb.finish()
    log("RHVD DONE")


if __name__ == "__main__":
    main()
