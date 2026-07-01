"""Step-1 MECHANISM comparison harness: a FAIR 3-way video-prediction shoot-out on a hierarchical
latent testbed -- REGRESSION vs DIFFUSION vs ENERGY-BASED (EBT) -- all sharing one transformer
backbone, one action-conditioning path, and one leakage-free single-step training protocol.

WHY: the newly-revealed leading strip of a panning viewport is the only genuinely generative part of
next-frame prediction (known content is a deterministic warp -> regression wins there). This harness
asks, at a SHORT context where that strip is uncertain, whether a generative mechanism (diffusion or
EBT) buys anything over regression, measured ONLY on decoded, foreground-masked PIXELS.

    venv/Scripts/python.exe -m hvp_rev.step1_mech --smoke
    venv/Scripts/python.exe -m hvp_rev.step1_mech --mode mech --ae logs_rhcp/hier_ae_mom.pt \
        --train data/loopmom_train_3000x100.npy --test data/loopmom_test_500x200.npy --steps 6000

================================================================================================
RED-TEAM FIXES (each noted inline where it lives):
  [RT-leakage]  single-step training uses GT latents for ctx+target; there is NO multi-step rollout
                here, so no prediction ever feeds itself. (A rollout eval would have to feed preds
                back -- explicitly out of scope for this mechanism probe; documented in report.)
  [RT-pxbest]   best-of-N selection is done in PIXEL space (decode, fg-mask, argmin fg-MSE), never
                in latent space -- latent L2 is not the quantity we care about.
  [RT-sharp]    fg_sharp is PER-IMAGE Laplacian energy on the per-image foreground mask, NOT a global
                batch variance (a batch variance rewards inter-sample diversity, not per-frame detail).
  [RT-nfe]      matched-NFE protocol: compare at equal forward-pass budget, not equal step count
                (diffusion step = 1 fwd; EBT step = fwd+grad ~= 2 fwd; regression sample = 1 fwd).
  [RT-shared]   all three predictors tokenize + attend + action-condition through the SAME
                LatentTokenizer/DiTBlock stack from faithful_nets -> capacity & action path matched.
  [RT-mcdrop]   regression gets N samples via MC-dropout (dropout ON at eval) so best-of-N is fair.
  [RT-fgthr]    THIS testbed's background is ~0.18-0.41 (not black); long_eval.fg_mse defaults thr=0.1
                which would mark the whole background as foreground. We pass thr=0.5 everywhere so the
                mask is the actual sprite. (Verified: bg maxes ~0.41, sprite ~0.99.)
================================================================================================
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hvp_rev import hier_convae, long_eval, viz
from hvp_rev.faithful_nets import (
    LAT,
    DiTBlock,
    DiTDenoiser,
    EnergyTransformer,
    LatentTokenizer,
    TimestepEmbedder,
    _build_sequence,
    double_backward_attention,
    modulate,
)
from hvp_rev.step1 import rel_positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FG_THR = 0.5  # [RT-fgthr] sprite threshold for THIS testbed (bg ~0.18-0.41, sprite ~0.99)

# ---- [SPEEDUP-1] opt-in bf16 autocast (default OFF -> matched-precision fp32, byte-for-byte as before).
# Set from args.bf16 in run_mech BEFORE building/training so all three models see the same setting.
# Autograd grads still resolve to fp32 (bf16 needs no GradScaler); _ac() is a no-op unless BOTH the
# flag is on AND we are on cuda, so CPU sanity checks and non-flagged runs are unchanged.
_BF16 = False


def _ac():
    return torch.autocast("cuda", dtype=torch.bfloat16) if (_BF16 and DEV == "cuda") else contextlib.nullcontext()


def log(*a):
    print(*a, flush=True)


# ================================================================================================
# 1) DATA: encode all frames once, sample single-step (ctx, target) with action + hardness labels.
# ================================================================================================
class ActionHierData:
    """Encodes every frame of every clip into (coarse,fine) latents ONCE (on CPU), then serves
    single-step batches: n_ctx context frames' latents + the target frame's latents + the k future
    actions bridging them, plus geometry-derived hardness labels (novel_fg, novel_strip)."""

    def __init__(self, ae, clips_path, actions_path, n_ctx=6, k=4, W=192, speed=4, n_val=300,
                 standardize=True, stats=None):
        """standardize: z-score latents per-channel per-level (the [STD-FIX]: tanh latents have std
        ~0.14/0.24 but diffusion noises at std 1 -- unstandardized latents silently mis-scale the
        noise schedule AND make randn inits ~4-7x off-scale for EBT/diffusion alike).
        stats: pass the TRAIN data's .stats for any held-out/test dataset so both are standardized
        with the SAME constants (never let a test set compute its own)."""
        self.n_ctx, self.k, self.W, self.speed = n_ctx, k, W, speed
        clips = np.load(clips_path)                 # (N,T,64,64) uint8
        acts = np.load(actions_path).astype(np.int64)  # (N,T) int8 -> int64
        self.N, self.T = clips.shape[0], clips.shape[1]
        self.V = clips.shape[-1]                    # viewport = 64
        self.acts = acts
        # frames normalized to [-1,1]; keep uint8 clips for GT-frame foreground geometry (cheap).
        self.clips_u8 = clips                       # (N,T,64,64) uint8, for sprite-column detection
        # ---- encode ALL frames into latents on CPU (loop over t; move per-t to cuda, then .cpu()) ----
        self.zc = torch.empty(self.N, self.T, LAT, 4, 4)   # coarse
        self.zf = torch.empty(self.N, self.T, LAT, 8, 8)   # fine
        t0 = time.time()
        with torch.no_grad():
            for t in range(self.T):
                x = torch.from_numpy(clips[:, t].astype(np.float32) / 127.5 - 1.0).unsqueeze(1)
                zc, zf = ae.encode(x.to(DEV))
                self.zc[:, t] = zc.cpu()
                self.zf[:, t] = zf.cpu()
        log(f"[data] encoded {self.N}x{self.T} frames -> zc{tuple(self.zc.shape)} zf{tuple(self.zf.shape)} "
            f"in {time.time()-t0:.1f}s")
        # ---- disjoint train/val split by CLIP index ----
        n_val = min(n_val, self.N // 2)
        self.val_idx = np.arange(self.N - n_val, self.N)
        self.train_idx = np.arange(0, self.N - n_val)
        log(f"[data] split: {len(self.train_idx)} train clips / {len(self.val_idx)} val clips; "
            f"W={W} V={self.V} speed={speed}")
        # ---- [STD-FIX] per-channel per-level z-score of the latents ----
        self.stats = None
        if standardize:
            if stats is None:
                zc_t, zf_t = self.zc[self.train_idx], self.zf[self.train_idx]
                stats = (zc_t.mean(dim=(0, 1, 3, 4)).view(LAT, 1, 1),
                         zc_t.std(dim=(0, 1, 3, 4)).clamp_min(1e-4).view(LAT, 1, 1),
                         zf_t.mean(dim=(0, 1, 3, 4)).view(LAT, 1, 1),
                         zf_t.std(dim=(0, 1, 3, 4)).clamp_min(1e-4).view(LAT, 1, 1))
                del zc_t, zf_t
            self.stats = tuple(s.clone() for s in stats)
            mu_c, sd_c, mu_f, sd_f = self.stats
            self.zc = (self.zc - mu_c) / sd_c
            self.zf = (self.zf - mu_f) / sd_f
            ae._lat_stats = self.stats          # decode01/fg_mse_px un-standardize through the AE
            log(f"[data] [STD-FIX] standardized latents per-channel: raw coarse std "
                f"{float(sd_c.mean()):.4f} / fine std {float(sd_f.mean()):.4f} -> ~unit")

    # -------- geometry hardness labels (reuses step1.rel_positions) --------
    def _hardness(self, ci, ti, n_ctx):
        """For clip ci with target frame ti and n_ctx context frames [ti-n_ctx .. ti-1]:
          novel_fg    = fraction of TARGET sprite columns whose world-col was never covered by any
                        context frame (information-theoretically unknowable from context).
          novel_strip = same, restricted to the newly-revealed leading strip (cols not covered by the
                        LAST context frame ti-1) -- the only genuinely generative region.
        Sprite columns are detected from the GT target FRAME (max over rows > FG_THR)."""
        rel = rel_positions(self.acts[ci], self.speed)  # (T,) left-edge world position
        frame = self.clips_u8[ci, ti].astype(np.float32) / 255.0
        fg_cols = np.where(frame.max(axis=0) > FG_THR)[0]
        if len(fg_cols) == 0:
            return 0.0, 0.0                              # no sprite in target -> trivially "known"
        tgt_world = (rel[ti] + fg_cols) % self.W
        # coverage by the n_ctx context frames
        covered = np.zeros(self.W, bool)
        for m in range(ti - n_ctx, ti):
            covered[(rel[m] + np.arange(self.V)) % self.W] = True
        novel = ~covered[tgt_world]
        novel_fg = float(novel.mean())
        # newly-revealed strip = target sprite cols NOT covered by the last context frame (ti-1)
        last_world = set(((rel[ti - 1] + np.arange(self.V)) % self.W).tolist())
        strip_mask = np.array([wc not in last_world for wc in tgt_world])
        novel_strip = float(novel[strip_mask].mean()) if strip_mask.any() else 0.0
        return novel_fg, novel_strip

    def batch(self, B, n_ctx=None, train=True, return_frames=False, seed=None):
        n_ctx = self.n_ctx if n_ctx is None else n_ctx
        k = self.k
        idx_pool = self.train_idx if train else self.val_idx
        rng = np.random.RandomState(seed) if seed is not None else np.random
        max_ctx = self.n_ctx                              # sample so ANY n_ctx<=self.n_ctx is valid
        assert self.T - k > max_ctx, f"T={self.T} too short for n_ctx={max_ctx}, k={k}"

        ctx_c = torch.empty(B, n_ctx, LAT, 4, 4)
        ctx_f = torch.empty(B, n_ctx, LAT, 8, 8)
        tgt_c = torch.empty(B, LAT, 4, 4)
        tgt_f = torch.empty(B, LAT, 8, 8)
        last_f = torch.empty(B, LAT, 8, 8)
        actions = torch.empty(B, k, dtype=torch.long)
        novel_fg = torch.empty(B)
        novel_strip = torch.empty(B)
        gt_frame = torch.empty(B, 1, 64, 64) if return_frames else None
        last_frame = torch.empty(B, 1, 64, 64) if return_frames else None

        for b in range(B):
            ci = int(idx_pool[rng.randint(len(idx_pool))])
            # ti indexes the TARGET frame. context = [ti-1-... ] : we take context frames ending at
            # ti-1, and the target is k steps later at ti+k-1 (matches step1/geom: predict k ahead).
            # ti0 = first context frame's index. Choose the target frame index tj in [max_ctx, T-k).
            tj = int(rng.randint(max_ctx, self.T - k))    # last-context frame index (frame tj-1 is persistence)
            tgt_i = tj + k - 1                            # actual target frame (k steps after last ctx)
            ctx_slice = slice(tj - n_ctx, tj)            # n_ctx context frames [tj-n_ctx .. tj-1]
            ctx_c[b] = self.zc[ci, ctx_slice]
            ctx_f[b] = self.zf[ci, ctx_slice]
            tgt_c[b] = self.zc[ci, tgt_i]
            tgt_f[b] = self.zf[ci, tgt_i]
            last_f[b] = self.zf[ci, tj - 1]              # persistence = fine latent of last context frame
            # the k future actions bridging last context frame (tj-1) to target (tj+k-1):
            # acts[t] moves frame t-1 -> t, so the moves are acts[tj .. tj+k-1].
            actions[b] = torch.from_numpy(self.acts[ci, tj:tj + k])
            nf, ns = self._hardness(ci, tgt_i, n_ctx)
            novel_fg[b], novel_strip[b] = nf, ns
            if return_frames:
                gt_frame[b, 0] = torch.from_numpy(self.clips_u8[ci, tgt_i].astype(np.float32) / 127.5 - 1.0)
                last_frame[b, 0] = torch.from_numpy(self.clips_u8[ci, tj - 1].astype(np.float32) / 127.5 - 1.0)

        out = dict(
            ctx_c=ctx_c.to(DEV), ctx_f=ctx_f.to(DEV),
            tgt_c=tgt_c.to(DEV), tgt_f=tgt_f.to(DEV),
            last_f=last_f.to(DEV), actions=actions.to(DEV),
            novel_fg=novel_fg, novel_strip=novel_strip,
        )
        if return_frames:
            out["gt_frame"] = gt_frame.to(DEV)
            out["last_frame"] = last_frame.to(DEV)
        return out


# ================================================================================================
# 2) METRICS -- ALL on decoded pixels, foreground-masked. Never in latent space.
# ================================================================================================
_LAP_KERNEL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def decode01(ae, zf):
    """(B,16,8,8) -> (B,1,64,64) in [0,1]. [STD-FIX] If the data layer standardized latents
    (ae._lat_stats set by ActionHierData), un-standardize the fine latent before decoding --
    every consumer (metrics, persistence, viz, EBT probes) goes through here, so the whole
    harness stays consistent in standardized space."""
    st = getattr(ae, "_lat_stats", None)
    with torch.no_grad():
        if st is not None:
            zf = zf * st[3].to(zf.device, zf.dtype) + st[2].to(zf.device, zf.dtype)
        return (ae.decode_fine(zf).clamp(-1, 1) + 1) * 0.5


def fg_mse_px(ae, pred_zf, gt_zf):
    """Decode both, fg-MSE (thr=FG_THR) on numpy (B,1,1,64,64). Returns scalar mean and per-sample (B,)."""
    p = decode01(ae, pred_zf).cpu().numpy()[:, None]       # (B,1,1,64,64)
    g = decode01(ae, gt_zf).cpu().numpy()[:, None]
    per = long_eval.fg_mse(p, g, thr=FG_THR)[:, 0]         # (B,)  [RT-fgthr]
    return float(per.mean()), per


def fg_mse_px_frames(pred01, gt01):
    """Same fg-MSE but from already-decoded [0,1] frames (B,1,64,64) -> per-sample (B,)."""
    p = pred01.cpu().numpy()[:, None]
    g = gt01.cpu().numpy()[:, None]
    return long_eval.fg_mse(p, g, thr=FG_THR)[:, 0]


def fg_sharp(frames01, gt_frames01):
    """[RT-sharp] PER-IMAGE Laplacian energy on the per-image FOREGROUND mask (gt>0.1 of sprite;
    we use FG_THR to match the sprite). Returns mean over batch of
        sum(lap^2 * fgmask) / sum(fgmask).
    NOT a global batch variance (that would reward inter-sample diversity, not per-frame detail)."""
    k = _LAP_KERNEL.to(frames01.device, frames01.dtype)
    lap = F.conv2d(frames01, k, padding=1)                 # (B,1,64,64)
    fgmask = (gt_frames01 > FG_THR).float()
    num = (lap.pow(2) * fgmask).sum(dim=(1, 2, 3))
    den = fgmask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return float((num / den).mean())


def best_of_N(sample_fn, N, ae, gt_zf, gt_frame01):
    """[RT-pxbest] Draw N samples; decode each; PER-SAMPLE pick the candidate with MIN fg-MSE vs GT
    in PIXEL space (fg-masked). Returns (best_mean_fgmse, best_fg_sharp, mean_over_N_fgmse).
      - best  = coverage/oracle (does the mechanism ever place mass right?)
      - meanN = precision (is a typical sample good?)  -- precision-vs-coverage."""
    B = gt_zf.shape[0]
    gt01 = decode01(ae, gt_zf)
    all_fgmse = np.empty((N, B), np.float32)
    all_frames = []
    for n in range(N):
        pred_zf = sample_fn()
        f01 = decode01(ae, pred_zf)
        all_frames.append(f01)
        all_fgmse[n] = fg_mse_px_frames(f01, gt01)
    best_n = all_fgmse.argmin(axis=0)                      # (B,) index of best sample per element
    best_fgmse = all_fgmse[best_n, np.arange(B)]           # (B,)
    best_frames = torch.stack([all_frames[best_n[b]][b] for b in range(B)], dim=0)  # (B,1,64,64)
    sharp = fg_sharp(best_frames, gt_frame01)
    return float(best_fgmse.mean()), sharp, float(all_fgmse.mean()), best_frames


# ================================================================================================
# 3) THREE PREDICTORS -- matched backbone (faithful_nets), one shared action path.
# ================================================================================================
class Regressor(nn.Module):
    """[RT-shared][RT-mcdrop] Direct x0-predictor: SAME tokenization + DiTBlock attention as
    DiTDenoiser, but the head outputs the target latent directly (no diffusion time; t=0 passed to
    the shared cond builder). Dropout in the backbone is kept ON at eval for MC-dropout best-of-N."""

    def __init__(self, lat=LAT, n_ctx=6, d=256, depth=6, heads=8, k=4, dropout=0.1):
        super().__init__()
        self.lat, self.n_ctx, self.d, self.k = lat, n_ctx, d, k
        self.tok = LatentTokenizer(lat, d, n_ctx, k)
        self.time_embedding = TimestepEmbedder(d)          # kept for the shared cond() signature; fed t=0
        # NOT zero_gate: a regressor is trained to output the target directly (not a residual denoiser),
        # so an identity-init block that emits ~0 would waste the head zero-init. Use active blocks.
        self.blocks = nn.ModuleList([DiTBlock(d, heads, zero_gate=False) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.norm_c = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.head_c = nn.Linear(d, lat)
        self.head_f = nn.Linear(d, lat)

    def forward(self, ctx_c, ctx_f, actions, dummy_c, dummy_f):
        """dummy_c/dummy_f: zero placeholders for the target-token SLOTS in _build_sequence (their
        values are irrelevant -- the head predicts from attended context; we pass zeros)."""
        B = ctx_c.shape[0]
        x, level_id, role_id, frame_id, n_tgt_c, n_tgt_f = _build_sequence(
            self.tok, ctx_c, ctx_f, dummy_c, dummy_f, self.n_ctx
        )
        action_code = self.tok.action_code(actions)
        cond_tgt = self.time_embedding(torch.zeros(B, device=ctx_c.device))
        cond_ctx = self.time_embedding(torch.zeros(B, device=ctx_c.device))
        c = self.tok.cond(level_id, role_id, frame_id, cond_ctx, cond_tgt, action_code)
        with _ac():  # [SPEEDUP-1] bf16 autocast (opt-in); matched-precision vs diff/ebt
            for blk in self.blocks:
                x = self.drop(blk(x, c))
        x = x.float()  # heads run in fp32 (outside autocast, fp32 Linear vs bf16 input would error)
        tgt = x[:, -(n_tgt_c + n_tgt_f):]
        xc, xf = tgt[:, :n_tgt_c], tgt[:, n_tgt_c:]
        yc = self.head_c(self.norm_c(xc)).transpose(1, 2).reshape(B, self.lat, 4, 4)
        yf = self.head_f(self.norm_f(xf)).transpose(1, 2).reshape(B, self.lat, 8, 8)
        return yc, yf


# ---- diffusion schedule (cosine, v-prediction, DDIM) ----
def cosine_alphas(T, s=0.008, device=DEV):
    steps = torch.arange(T + 1, dtype=torch.float32, device=device) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return abar.clamp(1e-4, 0.9999)                        # (T+1,)


class Diffusion:
    """v-prediction cosine-schedule diffusion wrapper around faithful_nets.DiTDenoiser."""

    def __init__(self, net: DiTDenoiser, n_train_steps=1000):
        self.net = net
        self.Ttr = n_train_steps
        self.abar = cosine_alphas(n_train_steps)           # (Ttr+1,)

    def loss(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        B = tgt_c.shape[0]
        ti = torch.randint(1, self.Ttr + 1, (B,), device=DEV)
        ab = self.abar[ti].view(B, 1, 1, 1)
        sa, sb = ab.sqrt(), (1 - ab).sqrt()
        nc, nf = torch.randn_like(tgt_c), torch.randn_like(tgt_f)
        yc = sa * tgt_c + sb * nc
        yf = sa * tgt_f + sb * nf
        # v-target = sqrt(abar)*noise - sqrt(1-abar)*x0
        vc_t = sa * nc - sb * tgt_c
        vf_t = sa * nf - sb * tgt_f
        t = (ti.float() / self.Ttr)
        with _ac():  # [SPEEDUP-1] bf16 autocast (opt-in)
            vc, vf = self.net(yc, yf, ctx_c, ctx_f, t, actions)
        # loss math in fp32 regardless of autocast (bf16 net output vs fp32 target breaks backward)
        return F.mse_loss(vc.float(), vc_t) + F.mse_loss(vf.float(), vf_t)

    @torch.no_grad()
    def sample(self, ctx_c, ctx_f, actions, n_steps=16):
        """DDIM sampling with n_steps evenly-spaced timesteps. Returns predicted (yc,yf)."""
        B = ctx_c.shape[0]
        yc = torch.randn(B, LAT, 4, 4, device=DEV)
        yf = torch.randn(B, LAT, 8, 8, device=DEV)
        ts = torch.linspace(self.Ttr, 1, n_steps, device=DEV).round().long().clamp(1, self.Ttr)
        for i in range(n_steps):
            ti = ts[i]
            ab = self.abar[ti].view(1, 1, 1, 1)
            sa, sb = ab.sqrt(), (1 - ab).sqrt()
            t = torch.full((B,), ti.item() / self.Ttr, device=DEV)
            with _ac():  # [SPEEDUP-1] bf16 autocast (opt-in); DDIM update below stays fp32
                vc, vf = self.net(yc, yf, ctx_c, ctx_f, t, actions)
            vc, vf = vc.float(), vf.float()  # keep the v->x0 arithmetic in fp32 regardless of autocast
            # v -> x0, eps
            x0c = sa * yc - sb * vc
            x0f = sa * yf - sb * vf
            epc = sb * yc + sa * vc
            epf = sb * yf + sa * vf
            if i < n_steps - 1:
                ab_next = self.abar[ts[i + 1]].view(1, 1, 1, 1)
                sa_n, sb_n = ab_next.sqrt(), (1 - ab_next).sqrt()
                yc = sa_n * x0c + sb_n * epc
                yf = sa_n * x0f + sb_n * epf
            else:
                yc, yf = x0c, x0f
        return yc, yf


class EBT:
    """S2-recipe Energy-Based-Transformer wrapper around faithful_nets.EnergyTransformer.
    Training (S2): replay buffer warm-starts half the batch; randomized #descent-steps; randomized
    per-element alpha in [alpha0/2, 2*alpha0]; Langevin noise; NO detach through descent; loss =
    final-step SmoothL1 to GT; alpha is a learnable Parameter in ITS OWN high-LR, zero-wd group."""

    def __init__(self, net: EnergyTransformer, alpha0=1.0, buf_cap=2048, max_steps=2,
                 langevin=0.01, clip=1.0):
        # max_steps=3 (not 8): the TRAINING descent keeps the graph (create_graph=True) for a
        # second-order loss, and faithful_nets' attention is the UNFUSED softmax path (required for
        # double-backward) -- memory grows ~linearly per step and blows 16GB past ~3 steps even at a
        # small batch. The replay buffer warm-start makes few descent steps sufficient. EVAL descent
        # (no create_graph, detached between steps) is cheap and uses many more steps (nfe//2).
        self.net = net
        self.max_steps = max_steps
        self.langevin = langevin
        self.clip = clip
        # learnable step size (log-parameterized so it stays positive). Its OWN optimizer group.
        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(alpha0)), device=DEV))
        # replay buffer of past FINAL (yc,yf) candidates (S2 warm-start)
        self.buf_cap = buf_cap
        self.buf_c, self.buf_f = None, None

    # ---- buffer helpers ----
    def _buf_push(self, yc, yf):
        yc, yf = yc.detach().cpu(), yf.detach().cpu()
        if self.buf_c is None:
            self.buf_c, self.buf_f = yc, yf
        else:
            self.buf_c = torch.cat([self.buf_c, yc], 0)[-self.buf_cap:]
            self.buf_f = torch.cat([self.buf_f, yf], 0)[-self.buf_cap:]

    def _init_candidate(self, B, warm):
        """Half warm-start from buffer (if available), half fresh noise (S2)."""
        yc = torch.randn(B, LAT, 4, 4, device=DEV)
        yf = torch.randn(B, LAT, 8, 8, device=DEV)
        if warm and self.buf_c is not None and self.buf_c.shape[0] >= B // 2 and B >= 2:
            h = B // 2
            sel = torch.randint(0, self.buf_c.shape[0], (h,))
            yc[:h] = self.buf_c[sel].to(DEV)
            yf[:h] = self.buf_f[sel].to(DEV)
        return yc, yf

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _descend(self, ctx_c, ctx_f, actions, yc, yf, n_steps, learning, per_elem_alpha=None):
        """Run n_steps Langevin-ish energy descent.

        [truncate_mcmc — the real EBT-repo recipe] For TRAINING (learning=True), the 2nd-order
        create_graph is kept ONLY on the FINAL descent step; all earlier steps are detached. So the
        retained double-backward graph is ~2x a single forward, CONSTANT in n_steps -- instead of the
        N-fold blowup of keeping create_graph across every step, which pegged 16GB VRAM, spilled to
        shared RAM (WDDM), and crawled at ~8 steps/min (~12h for 6000 steps). Eval (learning=False)
        never keeps the graph, so it is unchanged/cheap."""
        B = yc.shape[0]
        yc = yc.detach().requires_grad_(True)
        yf = yf.detach().requires_grad_(True)
        alpha = self.alpha if per_elem_alpha is None else per_elem_alpha
        for s in range(n_steps):
            cg = bool(learning and s == n_steps - 1)   # graph ONLY on the final training step
            step = torch.full((B,), min(s, self.net.step_embedding.num_embeddings - 1),
                              dtype=torch.long, device=DEV)
            if cg:                                       # explicit-softmax attn (twice-diff) for the
                with double_backward_attention():        # one forward that gets double-backprop'd
                    with _ac():                          # [SPEEDUP-1] bf16 autocast nested INSIDE the
                        E = self.net.energy(ctx_c, ctx_f, yc, yf, step, actions)  # double-backward ctx
            else:                                        # all other fwds use fast fused SDPA
                with _ac():                              # [SPEEDUP-1] bf16 autocast (opt-in)
                    E = self.net.energy(ctx_c, ctx_f, yc, yf, step, actions)
            gc, gf = torch.autograd.grad(E.sum(), [yc, yf], create_graph=cg)
            a_c = alpha.view(-1, 1, 1, 1) if per_elem_alpha is not None else alpha
            yc = yc - a_c * gc
            yf = yf - a_c * gf
            if self.langevin > 0:
                yc = yc + self.langevin * torch.randn_like(yc)
                yf = yf + self.langevin * torch.randn_like(yf)
            if not cg:                                  # detach between steps (and always in eval)
                yc = yc.detach().requires_grad_(True)
                yf = yf.detach().requires_grad_(True)
        return yc, yf

    def loss(self, ctx_c, ctx_f, tgt_c, tgt_f, actions):
        B = tgt_c.shape[0]
        yc0, yf0 = self._init_candidate(B, warm=True)
        n_steps = int(np.random.randint(1, self.max_steps + 1))       # randomized 1-2 steps (real-repo S2)
        # randomized per-element alpha in [alpha0/2, 2*alpha0] (S2). Built off the LEARNABLE alpha.
        scale = torch.empty(B, device=DEV).uniform_(0.5, 2.0)
        per_elem_alpha = self.alpha * scale
        yc, yf = self._descend(ctx_c, ctx_f, actions, yc0, yf0, n_steps, learning=True,
                               per_elem_alpha=per_elem_alpha)           # NO detach
        loss = F.smooth_l1_loss(yc, tgt_c) + F.smooth_l1_loss(yf, tgt_f)  # final-step SmoothL1
        self._buf_push(yc, yf)                                          # store final candidate
        return loss

    @torch.no_grad()
    def _sample_once(self, ctx_c, ctx_f, actions, n_steps):
        with torch.enable_grad():
            yc0, yf0 = self._init_candidate(ctx_c.shape[0], warm=False)  # fresh random init
            yc, yf = self._descend(ctx_c, ctx_f, actions, yc0, yf0, n_steps, learning=False)
        return yc.detach(), yf.detach()

    def sample_and_energy(self, ctx_c, ctx_f, actions, n_steps):
        yc, yf = self._sample_once(ctx_c, ctx_f, actions, n_steps)
        with torch.no_grad():
            step = torch.zeros(ctx_c.shape[0], dtype=torch.long, device=DEV)
            with _ac():  # [SPEEDUP-1] bf16 autocast (opt-in)
                E = self.net.energy(ctx_c, ctx_f, yc, yf, step, actions).squeeze(-1)  # (B,)
            E = E.float()  # energies compared/argmin'd downstream -> keep fp32 for stable ordering
        return yc, yf, E


# ================================================================================================
# TRAINING
# ================================================================================================
def save_ckpt(path, **state):
    """Atomic checkpoint (tmp + os.replace, same pattern as train_renderer.py) -- overnight runs on
    Windows must be restartable."""
    if path is None:
        return
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, opt=None, sched=None, extra=()):
    """Load an atomic checkpoint if present; returns the step to RESUME FROM (1 if none)."""
    if path is None or not os.path.exists(path):
        return 1
    st = torch.load(path, map_location=DEV, weights_only=False)
    model.load_state_dict(st["model"])
    if opt is not None and "opt" in st:
        opt.load_state_dict(st["opt"])
    if sched is not None and "sched" in st:
        sched.load_state_dict(st["sched"])
    for name, tensor in extra:
        if name in st:
            tensor.data.copy_(st[name])
    log(f"  [ckpt] resumed from {path} @ step {st['step']}")
    return int(st["step"]) + 1


def train_regressor(reg, data, steps, B, val_every=1000, ckpt=None, save_every=2000, resume=False):
    opt = torch.optim.AdamW(reg.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    start = load_ckpt(ckpt, reg, opt, sched) if resume else 1
    zc0 = torch.zeros(B, LAT, 4, 4, device=DEV)
    zf0 = torch.zeros(B, LAT, 8, 8, device=DEV)
    reg.train()
    run = 0.0
    for s in range(start, steps + 1):
        bt = data.batch(B, train=True)
        yc, yf = reg(bt["ctx_c"], bt["ctx_f"], bt["actions"], zc0[:bt["ctx_c"].shape[0]], zf0[:bt["ctx_c"].shape[0]])
        loss = F.smooth_l1_loss(yc, bt["tgt_c"]) + F.smooth_l1_loss(yf, bt["tgt_f"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
        opt.step(); sched.step()
        run += float(loss.detach())
        if s % val_every == 0:
            log(f"  [reg  {s:5d}/{steps}] loss {run/val_every:.4f}")
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=reg.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
    return reg


def train_diffusion(diff, data, steps, B, val_every=1000, test=None, ae=None, grid_every=2000,
                    ckpt=None, save_every=2000, resume=False, tag=""):
    opt = torch.optim.AdamW(diff.net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    start = load_ckpt(ckpt, diff.net, opt, sched) if resume else 1
    diff.net.train()
    run = 0.0
    for s in range(start, steps + 1):
        bt = data.batch(B, train=True)
        loss = diff.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.net.parameters(), 1.0)
        opt.step(); sched.step()
        run += float(loss.detach())
        if s % val_every == 0:
            log(f"  [diff {s:5d}/{steps}] v-mse {run/val_every:.4f}")
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=diff.net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict())
        if ae is not None and test is not None and s % grid_every == 0:
            diff.net.eval()
            vb = test.batch(8, train=False, return_frames=True, seed=777)
            _, yf = diff.sample(vb["ctx_c"], vb["ctx_f"], vb["actions"], n_steps=16)
            pred01 = decode01(ae, yf)
            gt01 = (vb["gt_frame"].clamp(-1, 1) + 1) * 0.5
            last01 = (vb["last_frame"].clamp(-1, 1) + 1) * 0.5
            viz.save_samples(f"logs_rhcp/step1_mech{tag}_diff_train{s}.png",
                             [("last", last01), ("diff", pred01), ("GT", gt01)], n=8)
            diff.net.train()
            log(f"      >> diffusion sample grid saved: logs_rhcp/step1_mech{tag}_diff_train{s}.png")
    return diff


def train_ebt(ebt, data, steps, B, val_every=1000, micro=24, test=None, ae=None, grid_every=1500,
              ckpt=None, save_every=2000, resume=False, tag=""):
    """EBT training with GRADIENT ACCUMULATION: the effective batch matches the shared B (fair vs
    reg/diff), but each micro-batch is small (<=12) so the second-order create_graph descent graph
    fits 16GB. (The unfused double-backward attention in faithful_nets caps the physical batch --
    see EBT.__init__ note. reg/diff run the full B in one shot; EBT accumulates B/micro micro-steps.)
    alpha lives in its OWN optimizer group: high LR, zero weight-decay."""
    opt = torch.optim.AdamW(
        [{"params": ebt.net.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
         {"params": [ebt.log_alpha], "lr": 5e-2, "weight_decay": 0.0}]
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[3e-4, 5e-2], total_steps=steps, pct_start=0.05)
    start = load_ckpt(ckpt, ebt.net, opt, sched, extra=[("log_alpha", ebt.log_alpha)]) if resume else 1
    ebt.net.train()
    micro = min(micro, B)
    n_accum = max(1, B // micro)
    run, n_nan = 0.0, 0
    for s in range(start, steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(n_accum):
            bt = data.batch(micro, train=True)
            loss = ebt.loss(bt["ctx_c"], bt["ctx_f"], bt["tgt_c"], bt["tgt_f"], bt["actions"]) / n_accum
            if not torch.isfinite(loss):                   # guard NaN: skip this micro-step
                n_nan += 1
                continue
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(ebt.net.parameters(), ebt.clip)  # clip grad norm on EBT
        torch.nn.utils.clip_grad_norm_([ebt.log_alpha], 1.0)
        opt.step(); sched.step()
        run += step_loss
        if s % val_every == 0:
            log(f"  [ebt  {s:5d}/{steps}] sl1 {run/val_every:.4f}  alpha {float(ebt.alpha.detach()):.4f}  "
                f"nan_skipped {n_nan}  (micro={micro}x{n_accum})")
            run = 0.0
        if s % save_every == 0 or s == steps:
            save_ckpt(ckpt, step=s, model=ebt.net.state_dict(), opt=opt.state_dict(),
                      sched=sched.state_dict(), log_alpha=ebt.log_alpha.data)
        if ae is not None and test is not None and s % grid_every == 0:   # watch mode-commitment develop
            ebt.net.eval()
            vb = test.batch(8, train=False, return_frames=True, seed=777)
            yc, yf = ebt._sample_once(vb["ctx_c"], vb["ctx_f"], vb["actions"], n_steps=8)
            pred01 = decode01(ae, yf)
            gt01 = (vb["gt_frame"].clamp(-1, 1) + 1) * 0.5
            last01 = (vb["last_frame"].clamp(-1, 1) + 1) * 0.5
            viz.save_samples(f"logs_rhcp/step1_mech{tag}_ebt_train{s}.png",
                             [("last", last01), ("EBT", pred01), ("GT", gt01)], n=8)
            ebt.net.train()
            log(f"      >> EBT sample grid saved: logs_rhcp/step1_mech{tag}_ebt_train{s}.png")
    return ebt


# ================================================================================================
# MATCHED-NFE protocol [RT-nfe]
# ================================================================================================
def sampler_for_mode(mode, reg, diff, ebt, bt, nfe):
    """Return a zero-arg sample_fn producing a predicted (yf) tensor, and the effective NFE used.
    diffusion: NFE denoise steps (1 fwd each).
    EBT:       NFE//2 descent steps (each fwd+grad ~= 2 fwd).
    regression: NFE stochastic forward passes are consumed by best-of-N (1 fwd each); a single
                sample_fn call = 1 MC-dropout forward.
    """
    ctx_c, ctx_f, actions = bt["ctx_c"], bt["ctx_f"], bt["actions"]
    B = ctx_c.shape[0]
    if mode == "reg":
        zc0 = torch.zeros(B, LAT, 4, 4, device=DEV)
        zf0 = torch.zeros(B, LAT, 8, 8, device=DEV)
        reg.train()   # [RT-mcdrop] dropout ON at eval so N passes give N samples

        def fn():
            with torch.no_grad():
                _, yf = reg(ctx_c, ctx_f, actions, zc0, zf0)
            return yf
        return fn, nfe
    if mode == "diff":
        diff.net.eval()

        def fn():
            _, yf = diff.sample(ctx_c, ctx_f, actions, n_steps=max(1, nfe))
            return yf
        return fn, nfe
    if mode == "ebt":
        ebt.net.eval()
        steps = max(1, nfe // 2)

        def fn():
            yc, yf, _ = ebt.sample_and_energy(ctx_c, ctx_f, actions, n_steps=steps)
            return yf
        return fn, steps * 2
    raise ValueError(mode)


# ================================================================================================
# EVAL
# ================================================================================================
def evaluate(mode, reg, diff, ebt, data, ae, N=8, n_eval_batches=6, B=48, nfe=16, seed=0):
    """best-of-N fg-MSE (all + hard split by novel_strip median) + fg_sharp + mean-over-N; +
    energy-argmin for EBT. Returns a dict of aggregated scalars and a sample panel batch."""
    torch.manual_seed(seed); np.random.seed(seed)
    all_best, all_meanN, all_sharp, all_strip = [], [], [], []
    ebt_argmin_best = []
    panel = None
    for bi in range(n_eval_batches):
        bt = data.batch(B, train=False, return_frames=(bi == 0), seed=1000 + bi)
        gt_frame01 = (bt.get("gt_frame").clamp(-1, 1) + 1) * 0.5 if "gt_frame" in bt else \
            decode01(ae, bt["tgt_f"])
        gt01 = decode01(ae, bt["tgt_f"])                   # for fg-MSE stratification (== gt_frame01 up
        #                                                    to AE roundtrip; kept for exact prior behavior)
        # [SPEEDUP-3] Draw the N candidate samples ONCE per batch and reuse them for (i) best-of-N
        # fg-MSE, (ii) the sharpness panel, and (iii) energy-argmin (EBT). Previously EBT sampled 3x
        # (best_of_N + allfg re-draw + energy re-draw) and reg/diff 2x (best_of_N + allfg re-draw).
        # Same estimators, single sample set -> science-neutral dedupe.
        allfg = np.empty((N, B), np.float32)
        all_frames = []
        cand_E = [] if mode == "ebt" else None
        if mode == "ebt":
            ctx_c, ctx_f, actions = bt["ctx_c"], bt["ctx_f"], bt["actions"]
            steps = max(1, nfe // 2)
            for n in range(N):
                _, yf, E = ebt.sample_and_energy(ctx_c, ctx_f, actions, n_steps=steps)
                f01 = decode01(ae, yf)
                all_frames.append(f01)
                allfg[n] = fg_mse_px_frames(f01, gt01)
                cand_E.append(E)
        else:
            sample_fn, _ = sampler_for_mode(mode, reg, diff, ebt, bt, nfe)
            for n in range(N):
                f01 = decode01(ae, sample_fn())
                all_frames.append(f01)
                allfg[n] = fg_mse_px_frames(f01, gt01)
        # oracle best-of-N: per-sample argmin fg-MSE frame (pixel space) [RT-pxbest]
        best_n = allfg.argmin(axis=0)                      # (B,) index of best sample per element
        per_best = allfg[best_n, np.arange(B)]             # (B,)
        best_frames = torch.stack([all_frames[best_n[b]][b] for b in range(B)], dim=0)  # (B,1,64,64)
        sharp = fg_sharp(best_frames, gt_frame01)          # [RT-sharp] on the argmin-fg frames
        all_best.append(per_best)
        all_strip.append(bt["novel_strip"].numpy())
        all_meanN.append(float(allfg.mean()))
        all_sharp.append(sharp)
        # EBT: energy-argmin selection from the SAME frames/energies (report vs oracle best-of-N)
        if mode == "ebt":
            Estack = torch.stack(cand_E, 0)                # (N,B)
            amin = Estack.argmin(dim=0)                    # (B,)
            picked = torch.stack([all_frames[amin[b]][b] for b in range(B)], 0)  # decoded frames reused
            ebt_argmin_best.append(fg_mse_px_frames(picked, gt01))
        if bi == 0:
            last01 = (bt["last_frame"].clamp(-1, 1) + 1) * 0.5 if "last_frame" in bt else decode01(ae, bt["last_f"])
            panel = [("last", last01), (f"pred(best{N})", best_frames), ("GT", gt_frame01)]

    per_best = np.concatenate(all_best)
    per_strip = np.concatenate(all_strip)
    # Robust hard/easy split: novel_strip is often exactly 0 (short pans reveal no genuinely novel
    # world columns) -> a plain median can be 0, collapsing everyone into "hard" and leaving "easy"
    # empty. Split HARD = strictly-positive novelty vs EASY = zero novelty; if that degenerates
    # (all-zero or all-positive), fall back to the median so both bins are non-empty.
    hard = per_strip > 0
    if hard.all() or (~hard).all():
        med = np.median(per_strip)
        hard = per_strip >= med
        split_desc = f"median={med:.3f}"
    else:
        med = 0.0
        split_desc = "novel_strip>0"
    res = dict(
        best_fgmse=float(per_best.mean()),
        best_fgmse_easy=float(per_best[~hard].mean()) if (~hard).any() else float("nan"),
        best_fgmse_hard=float(per_best[hard].mean()) if hard.any() else float("nan"),
        meanN_fgmse=float(np.mean(all_meanN)),
        fg_sharp=float(np.mean(all_sharp)),
        strip_median=float(med),
        split_desc=split_desc,
        frac_hard=float(hard.mean()),
    )
    if ebt_argmin_best:
        res["ebt_energy_argmin_fgmse"] = float(np.concatenate(ebt_argmin_best).mean())
    return res, panel


def persistence_baseline(data, ae, n_batches=6, B=48):
    """Persistence = predict last context frame's fine latent. Pixel-space fg-MSE + sharp reference."""
    fgs, sharps = [], []
    for bi in range(n_batches):
        bt = data.batch(B, train=False, return_frames=True, seed=2000 + bi)
        gt01 = (bt["gt_frame"].clamp(-1, 1) + 1) * 0.5
        pred01 = decode01(ae, bt["last_f"])
        fgs.append(fg_mse_px_frames(pred01, gt01))
        sharps.append(fg_sharp(pred01, gt01))
    return float(np.concatenate(fgs).mean()), float(np.mean(sharps))


def reference_sharpness(data, ae, n_batches=4, B=48):
    """[RT-sharp references] raw-GT-frame sharp (ceiling) and AE-roundtrip sharp (decoder ceiling)."""
    raw, rt = [], []
    for bi in range(n_batches):
        bt = data.batch(B, train=False, return_frames=True, seed=3000 + bi)
        gt01 = (bt["gt_frame"].clamp(-1, 1) + 1) * 0.5
        raw.append(fg_sharp(gt01, gt01))
        rt01 = decode01(ae, bt["tgt_f"])                   # decode_fine(encode(gt))
        rt.append(fg_sharp(rt01, gt01))
    return float(np.mean(raw)), float(np.mean(rt))


# ================================================================================================
# NFE table
# ================================================================================================
def nfe_table(reg, diff, ebt, data, ae, N=8, B=48, nfes=(4, 8, 16), modes=("reg", "diff", "ebt")):
    log("\n  matched-NFE table  [RT-nfe]  (best-of-N fg-MSE; lower=better)")
    log(f"    {'NFE':>4} | " + " ".join(f"{m:>9}" for m in modes))
    for nfe in nfes:
        row = {}
        for mode in modes:
            r, _ = evaluate(mode, reg, diff, ebt, data, ae, N=N, n_eval_batches=2, B=B, nfe=nfe)
            row[mode] = r["best_fgmse"]
        log(f"    {nfe:>4} | " + " ".join(f"{row[m]:>9.4f}" for m in modes))


# ================================================================================================
# MAIN
# ================================================================================================
def run_mech(args):
    global _BF16
    _BF16 = bool(args.bf16)  # [SPEEDUP-1] enable bf16 autocast for ALL THREE models (matched precision)
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    assert all(m in ("reg", "diff", "ebt") for m in modes), modes
    tag = f"_{args.tag}" if args.tag else ""
    log(f"=== MECH mode | device {DEV} | n_ctx {args.n_ctx} | steps {args.steps} | B {args.batch} | "
        f"N {args.N} | nfe {args.nfe} | alpha0 {args.alpha0} | bf16 {_BF16} | ebt_micro {args.ebt_micro} | "
        f"modes {modes} | std {not args.no_std} | tag '{args.tag}' ===")
    ae = hier_convae.load(args.ae)
    data = ActionHierData(ae, args.train, args.train.replace(".npy", "_actions.npy"),
                          n_ctx=args.n_ctx, k=args.k, W=args.world, speed=args.speed, n_val=args.n_val,
                          standardize=not args.no_std)
    # test data as a SEPARATE ActionHierData (held-out clips), standardized with the TRAIN stats
    test = ActionHierData(ae, args.test, args.test.replace(".npy", "_actions.npy"),
                          n_ctx=args.n_ctx, k=args.k, W=args.world, speed=args.speed,
                          n_val=min(args.n_val, 250), standardize=not args.no_std, stats=data.stats)

    reg = Regressor(n_ctx=args.n_ctx, k=args.k, d=args.d).to(DEV)
    diff = Diffusion(DiTDenoiser(n_ctx=args.n_ctx, k=args.k, d=args.d).to(DEV))
    ebt = EBT(EnergyTransformer(n_ctx=args.n_ctx, k=args.k, d=args.d).to(DEV), alpha0=args.alpha0)
    for name, m in (("regressor", reg), ("diffusion", diff.net), ("ebt", ebt.net)):
        log(f"  params[{name}] = {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    # ckpt names carry the step count: resuming a ckpt into a DIFFERENT --steps would poison
    # OneCycleLR's restored schedule (red-team #5)
    if "reg" in modes:
        log(f"\n-- training regressor ({args.steps} steps) --")
        train_regressor(reg, data, args.steps, args.batch,
                        ckpt=f"logs_rhcp/mech{tag}_reg_t{args.steps}.pt", resume=args.resume)
    if "diff" in modes:
        log(f"-- training diffusion ({args.diff_steps} steps) --")
        train_diffusion(diff, data, args.diff_steps, args.batch, test=test, ae=ae,
                        grid_every=args.grid_every, ckpt=f"logs_rhcp/mech{tag}_diff_t{args.diff_steps}.pt",
                        resume=args.resume, tag=tag)
    if "ebt" in modes:
        log(f"-- training ebt ({args.ebt_steps} steps) --")
        train_ebt(ebt, data, args.ebt_steps, args.batch, micro=args.ebt_micro,
                  test=test, ae=ae, grid_every=args.grid_every,
                  ckpt=f"logs_rhcp/mech{tag}_ebt_t{args.ebt_steps}.pt", resume=args.resume, tag=tag)

    # references + baseline
    raw_sharp, rt_sharp = reference_sharpness(test, ae)
    pers_fg, pers_sharp = persistence_baseline(test, ae)
    log(f"\n  references: raw-GT sharp {raw_sharp:.4f} | AE-roundtrip sharp {rt_sharp:.4f} (decoder ceiling)")
    log(f"  persistence baseline: fg-MSE {pers_fg:.4f} | sharp {pers_sharp:.4f}")

    # main eval table
    log(f"\n{'='*84}\n  MECHANISM COMPARISON (held-out test, best-of-{args.N}, NFE={args.nfe})\n{'='*84}")
    log(f"  {'mech':>5} | {'fgMSE(all)':>10} {'fgMSE(easy)':>11} {'fgMSE(hard)':>11} "
        f"{'meanN':>8} {'sharp':>8} {'E-argmin':>9}")
    for mode in modes:
        r, panel = evaluate(mode, reg, diff, ebt, test, ae,
                            N=args.N, n_eval_batches=6, B=args.batch, nfe=args.nfe)
        ea = r.get("ebt_energy_argmin_fgmse", float("nan"))
        log(f"  {mode:>5} | {r['best_fgmse']:>10.4f} {r['best_fgmse_easy']:>11.4f} "
            f"{r['best_fgmse_hard']:>11.4f} {r['meanN_fgmse']:>8.4f} {r['fg_sharp']:>8.4f} "
            f"{ea:>9.4f}")
        viz.save_samples(f"logs_rhcp/step1_mech{tag}_{mode}.png", panel, n=8)
    log(f"  (easy/hard split = {r['split_desc']}; frac_hard = {r['frac_hard']:.2f})")

    nfe_table(reg, diff, ebt, test, ae, N=args.N, B=args.batch, modes=modes)
    log("\nMECH DONE")


def run_smoke(args):
    log(f"=== SMOKE | device {DEV} ===")
    ae = hier_convae.load("logs_rhcp/hier_ae.pt")
    data = ActionHierData(ae, "data/loop_mom_probe_300x100.npy",
                          "data/loop_mom_probe_300x100_actions.npy",
                          n_ctx=6, k=4, W=args.world, speed=args.speed, n_val=60)
    B, steps, N = 16, 300, 4
    reg = Regressor(n_ctx=6, k=4, d=256).to(DEV)
    diff = Diffusion(DiTDenoiser(n_ctx=6, k=4, d=256).to(DEV))
    ebt = EBT(EnergyTransformer(n_ctx=6, k=4, d=256).to(DEV), alpha0=args.alpha0)
    for name, m in (("regressor", reg), ("diffusion", diff.net), ("ebt", ebt.net)):
        log(f"  params[{name}] = {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    # sanity: one batch shape check
    bt = data.batch(B, train=True, return_frames=True, seed=0)
    for kk in ("ctx_c", "ctx_f", "tgt_c", "tgt_f", "last_f", "actions", "novel_fg", "novel_strip"):
        log(f"    batch[{kk}] = {tuple(bt[kk].shape)}")
    log(f"    novel_fg mean {bt['novel_fg'].mean():.3f} | novel_strip mean {bt['novel_strip'].mean():.3f}")

    log("-- train reg --"); train_regressor(reg, data, steps, B, val_every=150)
    log("-- train diff --"); train_diffusion(diff, data, steps, B, val_every=150)
    log("-- train ebt --"); train_ebt(ebt, data, steps, B, val_every=150)

    raw_sharp, rt_sharp = reference_sharpness(data, ae, n_batches=2, B=B)
    pers_fg, pers_sharp = persistence_baseline(data, ae, n_batches=2, B=B)
    log(f"  refs: raw-GT sharp {raw_sharp:.4f} | AE-roundtrip sharp {rt_sharp:.4f}")
    log(f"  persistence: fg-MSE {pers_fg:.4f} | sharp {pers_sharp:.4f}")

    log("\n  quick eval (best-of-4, nfe=8):")
    for mode in ("reg", "diff", "ebt"):
        r, panel = evaluate(mode, reg, diff, ebt, data, ae, N=N, n_eval_batches=2, B=B, nfe=8)
        ea = r.get("ebt_energy_argmin_fgmse", float("nan"))
        log(f"    {mode:>5}: fgMSE(all) {r['best_fgmse']:.4f} hard {r['best_fgmse_hard']:.4f} "
            f"meanN {r['meanN_fgmse']:.4f} sharp {r['fg_sharp']:.4f} E-argmin {ea:.4f}")
    # save ONE sample grid (ebt panel) to prove viz works end-to-end
    viz.save_samples("logs_rhcp/step1_mech_smoke.png", panel, n=8)
    log("\nSMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mech", choices=["mech"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ae", default="logs_rhcp/hier_ae_mom.pt")
    ap.add_argument("--train", default="data/loopmom_train_3000x100.npy")
    ap.add_argument("--test", default="data/loopmom_test_500x200.npy")
    ap.add_argument("--n_ctx", type=int, default=6)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--world", type=int, default=192)
    ap.add_argument("--speed", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000, help="regressor steps (converges fast)")
    ap.add_argument("--diff_steps", type=int, default=None, help="diffusion steps (default=--steps)")
    ap.add_argument("--ebt_steps", type=int, default=None, help="EBT steps (compute-bound; default=--steps)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--grid_every", type=int, default=1500, help="save a decoded sample grid every N train steps")
    ap.add_argument("--alpha0", type=float, default=1.0)
    # [SPEEDUP-1] opt-in bf16 autocast (default OFF -> matched-precision fp32, identical to before)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for all 3 models (matched precision; cuda only)")
    # [SPEEDUP-2] EBT micro-batch for grad-accum; effective batch stays --batch. Default 24 == today's 2-pass.
    ap.add_argument("--ebt_micro", type=int, default=24, help="EBT micro-batch (grad-accum; effective batch = --batch)")
    ap.add_argument("--modes", default="reg,diff,ebt", help="comma subset of reg,diff,ebt to train+eval")
    ap.add_argument("--tag", default="", help="suffix for checkpoints/grids/log lines (provenance of a run)")
    ap.add_argument("--resume", action="store_true", help="resume each mode from its checkpoint if present")
    # [STD-FIX] latent standardization is ON by default; --no_std reproduces the old (mis-scaled) config
    ap.add_argument("--no_std", action="store_true", help="disable latent standardization (legacy config)")
    args = ap.parse_args()
    if args.diff_steps is None:
        args.diff_steps = args.steps
    if args.ebt_steps is None:
        args.ebt_steps = args.steps

    if args.smoke:
        run_smoke(args)
    else:
        run_mech(args)


if __name__ == "__main__":
    main()
