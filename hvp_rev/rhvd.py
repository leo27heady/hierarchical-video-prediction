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

from hvp_rev import hier_convae, viz
import hvp_rev.step1_mech as M
from hvp_rev.step1_mech import (ActionHierData, cosine_alphas, decode01, fg_mse_px_frames,
                                fg_sharp, load_ckpt, save_ckpt)
from hvp_rev.faithful_nets import LAT, DiTBlock, LatentTokenizer, TimestepEmbedder, modulate
from hvp_rev.step1 import rel_positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 4          # predicted frames per step
SPILL_SEC = 10.0

ARMS = {
    #              n_ctx_c  n_ctx_f  mode
    "hier-long":  (48,      6,       "diff"),
    "hier-short": (6,       6,       "diff"),
    "flat-budget": (0,      18,      "diff"),   # 18*64 = 1152 ctx tokens == hier-long's budget
    "reg-long":   (48,      6,       "reg"),
}


def log(*a):
    print(*a, flush=True)


# ================================================================================================
# Model: window DiT with per-target-frame actions. Reuses LatentTokenizer pieces + DiTBlock.
# ================================================================================================
class WindowDiT(nn.Module):
    def __init__(self, n_ctx_c=48, n_ctx_f=6, win=WIN, d=384, depth=8, heads=6, lat=LAT,
                 mode="diff"):
        super().__init__()
        assert n_ctx_c == 0 or n_ctx_f <= n_ctx_c
        self.n_ctx_c, self.n_ctx_f, self.win, self.lat, self.d = n_ctx_c, n_ctx_f, win, lat, d
        self.mode = mode
        n_frames = max(n_ctx_c, n_ctx_f)                  # context window length in frames
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

    def _ctx_tokens(self, ctx_c, ctx_f):
        """Context token list + (level,role,frame) ids. Coarse frames span ids [0..n_ctx_c-1];
        fine frames are the LAST n_ctx_f ids of the n_frames window (flat arm: coarse absent,
        fine spans [0..n_ctx_f-1])."""
        xs, lvs, rls, fms = [], [], [], []
        f0_fine = self.base - self.n_ctx_f
        for f in range(self.n_ctx_c):
            x, lv, rl, fm = self.tok.tokenize_level(ctx_c[:, f], level=0, role=0, frame_idx=f)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        for j in range(self.n_ctx_f):
            x, lv, rl, fm = self.tok.tokenize_level(ctx_f[:, j], level=1, role=0,
                                                    frame_idx=f0_fine + j)
            xs.append(x); lvs.append(lv); rls.append(rl); fms.append(fm)
        return xs, lvs, rls, fms

    def forward(self, yc, yf, ctx_c, ctx_f, t, actions):
        """yc:(B,win,LAT,4,4) yf:(B,win,LAT,8,8) noisy targets (or dummies for reg);
        ctx_c:(B,n_ctx_c,...) ctx_f:(B,n_ctx_f,...); t:(B,); actions:(B,win) one per target frame.
        Returns per-level outputs shaped like yc/yf (v-pred for diff, x0 for reg)."""
        B = yf.shape[0]
        xs, lvs, rls, fms = self._ctx_tokens(ctx_c, ctx_f)
        n_ctx_tok = sum(x.shape[1] for x in xs)
        tgt_meta = []                                     # (n_tok_c, n_tok_f) per frame
        for j in range(self.win):
            xc, lvc, rlc, fmc = self.tok.tokenize_level(yc[:, j], level=0, role=1,
                                                        frame_idx=self.base + j)
            xf_, lvf, rlf, fmf = self.tok.tokenize_level(yf[:, j], level=1, role=1,
                                                         frame_idx=self.base + j)
            xs += [xc, xf_]; lvs += [lvc, lvf]; rls += [rlc, rlf]; fms += [fmc, fmf]
            tgt_meta.append((xc.shape[1], xf_.shape[1]))
        x = torch.cat(xs, 1)
        level_id, role_id, frame_id = torch.cat(lvs, 1), torch.cat(rls, 1), torch.cat(fms, 1)

        # per-token conditioning: level+frame+role + [ctx: t=0 emb] / [tgt frame j: t emb + a_j emb]
        c = (self.tok.level_embedding(level_id) + self.tok.frame_embedding(frame_id)
             + self.tok.role_embedding(role_id))
        cond_ctx = self.time_embedding(torch.zeros(B, device=x.device))
        cond_tgt = self.time_embedding(t)
        a_emb = self.tok.action_embedding(actions + 1)     # (B,win,d)
        # assemble target-token conditioning frame-by-frame
        per_tok_cond = cond_ctx[:, None, :].expand(B, n_ctx_tok, self.d)
        chunks = [per_tok_cond]
        for j, (nc_, nf_) in enumerate(tgt_meta):
            cj = (cond_tgt + a_emb[:, j])[:, None, :]      # (B,1,d)
            chunks.append(cj.expand(B, nc_ + nf_, self.d))
        c = c + torch.cat(chunks, 1)

        with M._ac():
            for blk in self.blocks:
                x = blk(x, c)
        x = x.float()

        # unpatchify target tokens per frame per level
        out_c = torch.empty(B, self.win, self.lat, 4, 4, device=x.device)
        out_f = torch.empty(B, self.win, self.lat, 8, 8, device=x.device)
        pos = n_ctx_tok
        for j, (nc_, nf_) in enumerate(tgt_meta):
            xc = x[:, pos:pos + nc_]; cc = c[:, pos:pos + nc_].float(); pos += nc_
            xf_ = x[:, pos:pos + nf_]; cf = c[:, pos:pos + nf_].float(); pos += nf_
            sc, scl = self.final_c_ada(cc).chunk(2, -1)
            vc = self.head_c(modulate(self.norm_c(xc), sc, scl))
            sf, sfl = self.final_f_ada(cf).chunk(2, -1)
            vf = self.head_f(modulate(self.norm_f(xf_), sf, sfl))
            out_c[:, j] = vc.transpose(1, 2).reshape(B, self.lat, 4, 4)
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
        yc = torch.randn(B, WIN, LAT, 4, 4, device=dev)
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
        zc = torch.zeros(B, WIN, LAT, 4, 4, device=dev)
        zf = torch.zeros(B, WIN, LAT, 8, 8, device=dev)
        return self.net(zc, zf, ctx_c, ctx_f, torch.zeros(B, device=dev), actions)


# ================================================================================================
# Data: window batches on top of ActionHierData's encoded+standardized latents.
# ================================================================================================
def window_batch(data, B, n_ctx_c, n_ctx_f, train=True, seed=None, frames=False):
    """Context ends at frame tj-1 (coarse [tj-n_ctx_c..tj-1], fine [tj-n_ctx_f..tj-1]);
    targets = frames [tj..tj+WIN-1]; actions[:, j] = acts[tj+j] (moves frame tj+j-1 -> tj+j)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pool = data.train_idx if train else data.val_idx
    n_frames = max(n_ctx_c, n_ctx_f)
    ctx_c = torch.empty(B, n_ctx_c, LAT, 4, 4)
    ctx_f = torch.empty(B, n_ctx_f, LAT, 8, 8)
    tgt_c = torch.empty(B, WIN, LAT, 4, 4)
    tgt_f = torch.empty(B, WIN, LAT, 8, 8)
    acts = torch.empty(B, WIN, dtype=torch.long)
    gt = torch.empty(B, WIN, 1, 64, 64) if frames else None
    for b in range(B):
        ci = int(pool[rng.randint(len(pool))])
        tj = int(rng.randint(n_frames, data.T - WIN + 1))
        if n_ctx_c > 0:
            ctx_c[b] = data.zc[ci, tj - n_ctx_c:tj]
        ctx_f[b] = data.zf[ci, tj - n_ctx_f:tj]
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
    n_ctx_c, n_ctx_f, mode = ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
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
    ntok = n_ctx_c * 16 + n_ctx_f * 64 + WIN * 80
    log(f"[arm {arm}] mode={mode} ctx_c={n_ctx_c} ctx_f={n_ctx_f} tokens={ntok} "
        f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M steps {start}..{steps} "
        f"B={B} seed={seed}")
    if start > steps:
        log(f"[arm {arm}] already trained; skipping to eval")
        return wrap, ema
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train(); run, t_last = 0.0, time.time()
    for s in range(start, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, s / warmup)
        bt = window_batch(data, B, n_ctx_c, n_ctx_f, train=True)
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
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(),
                      ema=ema.shadow.state_dict())
        if s % grid_every == 0:
            net.eval()
            vb = window_batch(test, 8, n_ctx_c, n_ctx_f, train=False, seed=777, frames=True)
            wrap_eval = WindowDiffusion(ema.shadow, wrap.Ttr) if mode == "diff" else WindowRegression(ema.shadow)
            yc, yf = wrap_eval.sample(vb["ctx_c"], vb["ctx_f"], vb["actions"], n_steps=16)
            pred01 = decode01(ae, yf[:, -1])               # last window frame (horizon 4)
            gt01 = (vb["gt"][:, -1].clamp(-1, 1) + 1) * 0.5
            viz.save_samples(f"logs_rhcp/rhvd_{arm}_s{seed}_train{s}.png",
                             [("pred_h4", pred01), ("GT", gt01)], n=8)
            net.train()
            log(f"      >> grid logs_rhcp/rhvd_{arm}_s{seed}_train{s}.png")
    return wrap, ema


# ================================================================================================
# AR rollout eval -> npz (per-frame horizon/hardness/fg-MSE/sharp records)
# ================================================================================================
@torch.no_grad()
def rollout_eval(arm, wrap, ema, data, ae, seed, n_clips=100, nfe=16, chunk=16):
    n_ctx_c, n_ctx_f, mode = ARMS[arm]
    net_eval = ema.shadow
    wrap_eval = WindowDiffusion(net_eval, wrap.Ttr) if mode == "diff" else WindowRegression(net_eval)
    n_frames = max(n_ctx_c, n_ctx_f)
    WARM = 48   # UNIVERSAL warmup across arms: every arm starts rollout at frame 48, so the
    #             (clip, horizon) records pair exactly across arms (flat's 18-frame buffer is
    #             warmed from GT frames 30..47)
    T = data.T
    n_steps_roll = (T - WARM) // WIN
    clips = data.val_idx[:n_clips]
    recs = dict(h=[], hard=[], fg=[], sharp=[], pers_fg=[], clip=[])
    for c0 in range(0, len(clips), chunk):
        cs = clips[c0:c0 + chunk]
        B = len(cs)
        buf_c = data.zc[cs, WARM - n_frames:WARM].clone().to(DEV) if n_ctx_c > 0 else \
            torch.zeros(B, n_frames, LAT, 4, 4, device=DEV)
        buf_f = data.zf[cs, WARM - n_frames:WARM].clone().to(DEV)
        pers01 = decode01(ae, data.zf[cs, WARM - 1].to(DEV))           # persistence reference
        for srn in range(n_steps_roll):
            tj = WARM + srn * WIN
            torch.manual_seed(100000 + 1000 * (c0 // chunk) + srn)     # CRN across arms
            acts = torch.from_numpy(data.acts[cs, tj:tj + WIN].astype(np.int64)).to(DEV)
            ctx_c = buf_c[:, -n_ctx_c:] if n_ctx_c > 0 else buf_c[:, :0]
            ctx_f = buf_f[:, -n_ctx_f:]
            yc, yf = wrap_eval.sample(ctx_c, ctx_f, acts, n_steps=nfe)
            if n_ctx_c > 0:
                buf_c = torch.cat([buf_c, yc], 1)[:, -n_frames:]
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
                    recs["fg"].append(float(fg[bi_]))
                    recs["pers_fg"].append(float(pfg[bi_]))
                    recs["sharp"].append(sh)              # batch-mean sharp per frame (coarse record)
                    recs["clip"].append(int(ci))
        log(f"  [rollout {arm}] clips {c0}..{c0+B-1} done ({n_steps_roll} steps x {WIN} frames)")
    out = {k: np.asarray(v) for k, v in recs.items()}
    path = f"logs_rhcp/rhvd_eval_{arm}_s{seed}.npz"
    np.savez(path, **out)
    log(f"[rollout {arm}] saved {path}: {len(out['fg'])} frame-records, "
        f"mean fg {out['fg'].mean():.4f} (persistence {out['pers_fg'].mean():.4f})")
    return path


def paired_ci(delta, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(delta), (n_boot, len(delta)))
    m = delta[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(arms, seed):
    """Paired per-frame comparison of rollout npz files (same clips/horizons across arms)."""
    d = {}
    for a in arms:
        z = np.load(f"logs_rhcp/rhvd_eval_{a}_s{seed}.npz")
        d[a] = z
    base = arms[0]
    zb = d[base]
    key = list(zip(zb["clip"], zb["h"]))
    print(f"\n{'='*100}\n  RHVD ROLLOUT COMPARISON (paired per-frame fg-MSE, base = {base}, seed {seed})\n{'='*100}")
    hbins = [(1, 8, "h 1-8"), (9, 24, "h 9-24"), (25, 60, "h 25-60"), (61, 200, "h 61+")]
    hardbins = [(-0.01, 0.001, "easy"), (0.001, 0.1, "mid"), (0.1, 1.01, "HARD")]
    for a in arms[1:]:
        za = d[a]
        assert np.array_equal(za["clip"], zb["clip"]) and np.array_equal(za["h"], zb["h"]), \
            "eval sets differ; re-run with same protocol"
        print(f"\n  {a} vs {base} (sharp {za['sharp'].mean():.4f} vs {zb['sharp'].mean():.4f}; "
              f"persistence {zb['pers_fg'].mean():.4f}):")
        print(f"  {'horizon':<10}{'hardness':<8}{'n':>7}{'base':>10}{'arm':>10}{'delta':>10}{'95% CI':>22}{'verdict':>9}")
        for hlo, hhi, hname in hbins:
            for blo, bhi, bname in hardbins:
                m = (zb["h"] >= hlo) & (zb["h"] <= hhi) & (zb["hard"] > blo) & (zb["hard"] <= bhi)
                if m.sum() < 30:
                    continue
                delta = zb["fg"][m] - za["fg"][m]
                lo, hi = paired_ci(delta)
                v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "null")
                print(f"  {hname:<10}{bname:<8}{int(m.sum()):>7}{zb['fg'][m].mean():>10.4f}"
                      f"{za['fg'][m].mean():>10.4f}{delta.mean():>+10.4f}   [{lo:+.4f},{hi:+.4f}]{v:>9}")
        delta = zb["fg"] - za["fg"]
        lo, hi = paired_ci(delta)
        v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "null")
        print(f"  {'ALL':<10}{'':<8}{len(delta):>7}{zb['fg'].mean():>10.4f}{za['fg'].mean():>10.4f}"
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
    ap.add_argument("--compare", default="", help="comma list of arms to compare from saved npz")
    args = ap.parse_args()
    M._BF16 = bool(args.bf16)

    if args.compare:
        compare([a.strip() for a in args.compare.split(",")], args.seed)
        return

    ae = hier_convae.load("logs_rhcp/hier_ae_mom.pt")
    data = ActionHierData(ae, "data/loopmom_train_3000x100.npy",
                          "data/loopmom_train_3000x100_actions.npy", n_ctx=6, k=4)
    test = ActionHierData(ae, "data/loopmom_test_500x200.npy",
                          "data/loopmom_test_500x200_actions.npy", n_ctx=6, k=4, n_val=250,
                          stats=data.stats)

    if args.probe_steps > 0:
        n_ctx_c, n_ctx_f, mode = ARMS[args.arm]
        net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
        wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.probe_steps):
            bt = window_batch(data, args.batch, n_ctx_c, n_ctx_f, train=True)
            loss = wrap.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if DEV == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / args.probe_steps
        vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0
        print(f"[TIMING {args.arm}] {dt:.3f} s/step | peak {vram:.2f} GB | "
              f"{args.steps} steps => {args.steps*dt/3600:.1f} h")
        return

    steps = 300 if args.smoke else args.steps
    if args.eval_only:
        n_ctx_c, n_ctx_f, mode = ARMS[args.arm]
        net = WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
        wrap = WindowDiffusion(net) if mode == "diff" else WindowRegression(net)
        ema = EMA(net)
        ckpt = f"logs_rhcp/rhvd_{args.arm}_s{args.seed}_t{args.steps}.pt"
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        ema.shadow.load_state_dict(st["ema"])
        log(f"[eval_only] loaded {ckpt} @ step {st['step']}")
    else:
        wrap, ema = train(args.arm, data, test, ae, steps, args.batch, args.seed)
    rollout_eval(args.arm, wrap, ema, test, ae, args.seed,
                 n_clips=(8 if args.smoke else args.n_eval_clips), nfe=args.nfe)
    log("RHVD DONE")


if __name__ == "__main__":
    main()
