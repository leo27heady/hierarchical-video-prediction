"""OpenSTL TaxiBJ bridge -- the real-data recipe-transfer arm of the generalization chapter (G-5).

================================================================================================
PROTOCOL (OpenSTL TaxiBJ benchmark, 4 context -> 4 predicted, 32x32x2 inflow/outflow, [-1,1])
================================================================================================
* DATA = the canonical OpenSTL file (X/Y train 20461 pairs, X/Y test 500 pairs, stored in
  [-1,1]). CRITICAL CONVENTION (verified in openstl/datasets/dataloader_taxibj.py line 15):
  the OpenSTL dataloader rescales to [0,1] via (X+1)/2 and sets mean=0/std=1, so the ENTIRE
  leaderboard convention (MSE 0.31-class numbers, min-max PSNR ~39.7, SSIM with clip [0,1])
  lives on [0,1] data. This bridge applies the same rescale at load; the AE/model side maps
  [0,1] -> [-1,1] (tanh range) and back, exactly like the T7 MMNIST bridge. Test pairs are
  scored exactly once, never trained on.
* METRICS = `openstl.core.metrics.metric` imported VERBATIM from the sibling repo and called
  exactly as OpenSTL's Base_method does for TaxiBJ: mean=0, std=1, spatial_norm=False,
  metrics=[mse, mae, ssim, psnr]. That is the leaderboard convention (TAU 0.3108 MSE etc.,
  local traffic_benchmarks.md). MSE/MAE are mean-over-(N,T), SUM-over-(C,H,W) = per-pixel x2048.
  LPIPS is dropped BY DESIGN (2-channel heatmaps; pre-registered reason in the memo).
* ROWS per table G-5: model arm(s), AE round-trip ceiling (encode+decode GT targets),
  persistence (repeat last context frame).

================================================================================================
BUILD AMENDMENT vs NEXT_EXPERIMENTS_GENERALIZATION.md B2 (DISCLOSED, decided 2026-07-07)
================================================================================================
Instead of a native-32px AE + new 16+4 token geometry (the memo's B2 estimate: 0.5-1 build day,
the riskiest item), frames are x2 NEAREST-upsampled to 64x64 and run through the UNCHANGED
64-px stack: HierConvAE(img_ch=2) + the audited WindowDiT 22M with its standard coarse 4x4 /
fine 8x8 latent geometry. Nearest x2 is exactly inverted by 2x2-mean pooling, so the task is
information-identical; predictions are 2x2-mean-downsampled back to native 32x32 BEFORE scoring
in the OpenSTL convention. This keeps "the unmodified WindowDiT" literally true across every
dataset of the dissertation and removes the highest-risk build. Disclosed wherever G-5 is quoted.

AUTHOR-INSTRUCTED REGIME (2026-07-07): batch size chosen by an on-device VRAM probe to fill
~14 GB (hard cap 15 GB, WDDM spill guard active); lr kept at the campaign default 3e-4 with
2k-step linear warmup (NOT scaled with batch -- conservative, disclosed); steps 15k default.
No wandb in this bridge (file logs only, like the T7 eval invocations).

ARMS (5-tuples registered into rhvd.ARMS at import; rhvd.py untouched). Context-token budgets
at the upsampled geometry; 4->4 has NO window axis to extend, so the hier arm is the
pre-registered TOKEN-EFFICIENCY read (half the context tokens), not a memory-horizon read:
    txb-flat-reg  : n_ctx_c=0, n_ctx_f=4  (256 ctx tok)   -- PRIMARY (regression head)
    txb-hier-reg  : n_ctx_c=4, n_ctx_f=1  (128 ctx tok)   -- PRIMARY token-efficiency twin
    txb-flat-diff : same layouts, diffusion head           -- optional twins
    txb-hier-diff :
Pre-registered expectations (memo section 7): recipe transfers (trains stably, beats
persistence clearly, lands in the leaderboard's neighborhood -- NO leaderboard-win claim);
hier ~ flat (no memory horizon: boundary replication); regression >= diffusion at best-of-1.

================================================================================================
RUN COMMANDS (venv python, repo root)
================================================================================================
  venv/Scripts/python.exe -u -m hvp_rev.taxibj_bridge --stage smoke                  # ~5 min
  venv/Scripts/python.exe -u -m hvp_rev.taxibj_bridge --stage probe --target_gb 14   # BS probe
  venv/Scripts/python.exe -u -m hvp_rev.taxibj_bridge --stage ae                     # 7k steps
  venv/Scripts/python.exe -u -m hvp_rev.taxibj_bridge --stage train --arm txb-flat-reg --batch B --bf16
  venv/Scripts/python.exe -u -m hvp_rev.taxibj_bridge --stage eval  --arm txb-flat-reg
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import hvp_rev.rhvd as R
import hvp_rev.step1_mech as M
from hvp_rev.faithful_nets import LAT
from hvp_rev.hier_convae import HierConvAE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAXIBJ_NPZ = "C:/Users/leoni/Documents/projects/OpenSTL/data/taxibj/dataset.npz"
OPENSTL_REPO = "C:/Users/leoni/Documents/projects/OpenSTL"
AE_PATH = "logs_rhcp/hier_ae_taxibj.pt"

BRIDGE_ARMS = {
    # name: (n_ctx_c, n_ctx_f, mode, stride_c, n_hist_f) -- same 5-tuple as rhvd.ARMS
    "txb-flat-reg":  (0, 4, "reg",  1, 0),
    "txb-hier-reg":  (4, 1, "reg",  1, 0),
    "txb-flat-diff": (0, 4, "diff", 1, 0),
    "txb-hier-diff": (4, 1, "diff", 1, 0),
}
R.ARMS.update(BRIDGE_ARMS)


def log(*a):
    print(*a, flush=True)


def up2(x: np.ndarray) -> np.ndarray:
    """(..., H, W) nearest x2 -- exactly inverted by 2x2 mean."""
    return np.repeat(np.repeat(x, 2, axis=-1), 2, axis=-2)


def down2(x: torch.Tensor) -> torch.Tensor:
    """(..., 2H, 2W) -> (..., H, W) by exact 2x2 mean (the inverse of up2 for nearest)."""
    s = x.shape
    return x.reshape(*s[:-2], s[-2] // 2, 2, s[-1] // 2, 2).mean(dim=(-3, -1))


def load_npz():
    """Returns [0,1] arrays -- the OpenSTL dataloader convention ((X+1)/2, dataloader_taxibj.py)."""
    d = np.load(TAXIBJ_NPZ)
    xtr = np.concatenate([d["X_train"], d["Y_train"]], axis=1).astype(np.float32)  # (N,8,2,32,32)
    xte_ctx = d["X_test"].astype(np.float32)                                       # (500,4,2,32,32)
    xte_tgt = d["Y_test"].astype(np.float32)                                       # (500,4,2,32,32)
    return (xtr + 1) * 0.5, (xte_ctx + 1) * 0.5, (xte_tgt + 1) * 0.5


# ================================================================================================
# 1) AE: HierConvAE(img_ch=2) on x2-upsampled train frames (bespoke float-frame pool).
# ================================================================================================
def run_ae(steps=7000, batch=64, lr=3e-4, seed=0, out=AE_PATH):
    torch.manual_seed(seed)
    xtr, _, _ = load_npz()
    frames = torch.from_numpy(up2(xtr.reshape(-1, 2, 32, 32)) * 2 - 1)   # [0,1]->[-1,1] (tanh)
    n_val = 2048
    perm = torch.randperm(frames.shape[0], generator=torch.Generator().manual_seed(0))
    val, tr = frames[perm[:n_val]], frames[perm[n_val:]]
    ae = HierConvAE(img_ch=2).to(DEV)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    log(f"[ae] HierConvAE(img_ch=2) on {tr.shape[0]} upsampled TaxiBJ frames "
        f"params={sum(p.numel() for p in ae.parameters())/1e6:.2f}M")
    run = 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(tr.shape[0], (batch,))
        x = tr[idx].to(DEV)
        rc, rf, _, _ = ae(x)
        if torch.rand(()) < 0.5:                       # random-level dropout, coarse up-weighted
            loss = 4.0 * (F.l1_loss(rc, x) + F.mse_loss(rc, x))
        else:
            loss = F.l1_loss(rf, x) + F.mse_loss(rf, x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += float(loss.detach())
        if s % 1000 == 0:
            with torch.no_grad():
                xv = val[:512].to(DEV)
                rc, rf, _, _ = ae(xv)
                vc, vf = F.mse_loss(rc, xv).item(), F.mse_loss(rf, xv).item()
            log(f"  [ae {s:5d}] loss {run/1000:.4f}  val recon coarse {vc:.5f} fine {vf:.5f}")
            run = 0.0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": ae.state_dict()}, out)
    log(f"[ae] saved {out}")


def load_ae(path=AE_PATH):
    ae = HierConvAE(img_ch=2).to(DEV)
    ae.load_state_dict(torch.load(path, map_location=DEV, weights_only=False)["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# ================================================================================================
# 2) Data: encode all clips once (chunked), [STD-FIX] stats from TRAIN only.
# ================================================================================================
class TxbData:
    def __init__(self, ae, n_val=500, chunk=1024):
        xtr, xte_ctx, xte_tgt = load_npz()
        self.n = xtr.shape[0]
        self.zc, self.zf = self._encode(ae, xtr, chunk)              # CPU (N,8,16,4,4)/(N,8,16,8,8)
        # [STD-FIX]: per-channel stats on TRAIN latents only; held-out & test standardized with them
        mu_c = self.zc.mean(dim=(0, 1, 3, 4), keepdim=True)
        sd_c = self.zc.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)
        mu_f = self.zf.mean(dim=(0, 1, 3, 4), keepdim=True)
        sd_f = self.zf.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-4)
        self.stats = (mu_c, sd_c, mu_f, sd_f)
        self.zc = (self.zc - mu_c) / sd_c
        self.zf = (self.zf - mu_f) / sd_f
        g = np.random.RandomState(0).permutation(self.n)
        self.val_idx, self.train_idx = g[:n_val], g[n_val:]
        self.xte_ctx, self.xte_tgt = xte_ctx, xte_tgt
        log(f"[data] {self.n} train pairs encoded; raw latent std coarse "
            f"{float(sd_c.mean()):.4f} / fine {float(sd_f.mean()):.4f} -> ~unit; "
            f"{len(self.train_idx)} train / {len(self.val_idx)} val clips")

    @staticmethod
    def _encode(ae, clips, chunk):
        N, T = clips.shape[:2]
        fr = torch.from_numpy(up2(clips.reshape(N * T, 2, 32, 32)) * 2 - 1)   # [0,1]->[-1,1]
        zcs, zfs = [], []
        with torch.no_grad():
            for i in range(0, fr.shape[0], chunk):
                zc, zf = ae.encode(fr[i:i + chunk].to(DEV))
                zcs.append(zc.cpu()); zfs.append(zf.cpu())
        zc = torch.cat(zcs).view(N, T, LAT, 4, 4)
        zf = torch.cat(zfs).view(N, T, LAT, 8, 8)
        return zc, zf

    def batch(self, B, arm, train=True, rng=None):
        """4->4 window: ctx = frames 0..3, targets = 4..7 (the only valid tj). Vectorized gather."""
        n_ctx_c, n_ctx_f, _, _, _ = R.ARMS[arm]
        rng = rng or np.random
        pool = self.train_idx if train else self.val_idx
        idx = torch.from_numpy(pool[rng.randint(0, len(pool), size=B)].copy()).long()
        ctx_c = self.zc[idx, 0:4] if n_ctx_c > 0 else torch.empty(B, 0, LAT, 4, 4)
        ctx_f = self.zf[idx, 4 - n_ctx_f:4]
        tgt_c, tgt_f = self.zc[idx, 4:8], self.zf[idx, 4:8]
        acts = torch.zeros(B, R.WIN, dtype=torch.long)               # no actions: constant no-op
        return (ctx_c.to(DEV), ctx_f.to(DEV), tgt_c.to(DEV), tgt_f.to(DEV), acts.to(DEV))


# ================================================================================================
# 3) TRAIN: mirrors rhvd.train exactly (AdamW 3e-4/wd 1e-4, warmup 2k, clip 1.0, EMA 0.999,
#    spill guard, same ckpt naming/dict) minus wandb and pan-testbed machinery.
# ================================================================================================
def run_train(arm, data, steps, B, seed, lr=3e-4, warmup=2000, val_every=500, save_every=1000):
    n_ctx_c, n_ctx_f, mode, _, _ = R.ARMS[arm]
    torch.manual_seed(seed); np.random.seed(seed)
    net = R.WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
    wrap = R.WindowDiffusion(net) if mode == "diff" else R.WindowRegression(net)
    ema = R.EMA(net)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ckpt = f"logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt"
    start = 1
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=DEV, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        ema.shadow.load_state_dict(st["ema"]); start = int(st["step"]) + 1
        log(f"[ckpt] resumed {ckpt} @ {st['step']}")
    log(f"[arm {arm}] mode={mode} ctx_c={n_ctx_c} ctx_f={n_ctx_f} seq={net.expected_seq} "
        f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M steps {start}..{steps} "
        f"B={B} seed={seed} (lr {lr} NOT batch-scaled, disclosed)")
    if start > steps:
        log(f"[arm {arm}] already trained"); return
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    net.train(); run, t_last = 0.0, time.time()
    for s in range(start, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, s / warmup)
        ctx_c, ctx_f, tgt_c, tgt_f, acts = data.batch(B, arm, train=True)
        loss = wrap.loss(ctx_c, ctx_f, tgt_c, tgt_f, acts)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); ema.update(net)
        run += float(loss.detach())
        dt = time.time() - t_last; t_last = time.time()
        if dt > R.SPILL_SEC and s > start + 3:
            R.save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(),
                        ema=ema.shadow.state_dict())
            raise RuntimeError(f"[arm {arm}] step {s} took {dt:.1f}s -> WDDM spill; ckpt saved.")
        if s % val_every == 0:
            vram = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
            log(f"  [{arm} {s}/{steps}] loss {run/val_every:.4f}  {dt:.2f}s/step  peak {vram:.2f}GB")
            run = 0.0
        if s % save_every == 0 or s == steps:
            R.save_ckpt(ckpt, step=s, model=net.state_dict(), opt=opt.state_dict(),
                        ema=ema.shadow.state_dict())


# ================================================================================================
# 4) PROBE: find the batch size whose training peak lands at ~target_gb (author: ~14 GB).
# ================================================================================================
def run_probe(arm, data, target_gb=14.0, candidates=(64, 96, 128, 192, 256, 384, 512, 768, 1024)):
    n_ctx_c, n_ctx_f, mode, _, _ = R.ARMS[arm]
    best = None
    for B in candidates:
        net = R.WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
        wrap = R.WindowDiffusion(net) if mode == "diff" else R.WindowRegression(net)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            for _ in range(3):
                ctx_c, ctx_f, tgt_c, tgt_f, acts = data.batch(B, arm, train=True)
                loss = wrap.loss(ctx_c, ctx_f, tgt_c, tgt_f, acts)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            dt = (time.time() - t0) / 3
            gb = torch.cuda.max_memory_allocated() / 2**30
            log(f"  [probe {arm}] B={B:4d}  peak {gb:5.2f} GB  {dt:.3f}s/step")
            if gb <= target_gb:
                best = (B, gb, dt)
            else:
                break
        except torch.cuda.OutOfMemoryError:
            log(f"  [probe {arm}] B={B:4d}  OOM")
            break
        finally:
            del net, wrap, opt
            torch.cuda.empty_cache()
    assert best, "no candidate batch fits"
    log(f"[probe {arm}] CHOSEN B={best[0]} (peak {best[1]:.2f} GB <= {target_gb} GB, "
        f"{best[2]:.3f}s/step => 15k steps ~ {best[2]*15000/3600:.1f}h)")
    return best[0]


# ================================================================================================
# 5) EVAL: one window step on the 500 canonical test pairs; OpenSTL metric() verbatim.
# ================================================================================================
def _openstl_metric(pred, true):
    """Call openstl.core.metrics.metric exactly as Base_method does for TaxiBJ."""
    if OPENSTL_REPO not in sys.path:
        sys.path.insert(0, OPENSTL_REPO)
    from openstl.core.metrics import metric as ostl_metric
    res, _ = ostl_metric(pred, true, mean=0, std=1, metrics=["mse", "mae", "ssim", "psnr"],
                         spatial_norm=False, return_log=True)
    return {k: float(res[k]) for k in ("mse", "mae", "ssim", "psnr")}


@torch.no_grad()
def run_eval(arm, ae, data, steps, seed=0, nfe=16, tag=""):
    n_ctx_c, n_ctx_f, mode, _, _ = R.ARMS[arm]
    ckpt = f"logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt"
    st = torch.load(ckpt, map_location=DEV, weights_only=False)
    net = R.WindowDiT(n_ctx_c=n_ctx_c, n_ctx_f=n_ctx_f, mode=mode).to(DEV)
    net.load_state_dict(st["ema"]); net.eval()                       # eval on EMA weights
    wrap = R.WindowDiffusion(net) if mode == "diff" else R.WindowRegression(net)
    mu_c, sd_c, mu_f, sd_f = [t.to(DEV) for t in data.stats]

    xc, xt = data.xte_ctx, data.xte_tgt                              # (500,4,2,32,32) each
    B, T = xc.shape[0], 4
    fr = torch.from_numpy(up2(xc.reshape(B * T, 2, 32, 32)) * 2 - 1).to(DEV)   # [0,1]->[-1,1]
    zc, zf = ae.encode(fr)
    zc = ((zc.view(B, T, LAT, 4, 4).cpu() - mu_c.cpu()) / sd_c.cpu()).to(DEV)
    zf = ((zf.view(B, T, LAT, 8, 8).cpu() - mu_f.cpu()) / sd_f.cpu()).to(DEV)
    ctx_c = zc[:, 0:4] if n_ctx_c > 0 else torch.empty(B, 0, LAT, 4, 4, device=DEV)
    ctx_f = zf[:, 4 - n_ctx_f:4]
    acts = torch.zeros(B, R.WIN, dtype=torch.long, device=DEV)
    torch.manual_seed(4242)                                          # CRN across arms
    _, yf = wrap.sample(ctx_c, ctx_f, acts, n_steps=nfe)
    # decode fine level, un-standardize (stats squeezed to (LAT,1,1) for 4-D broadcast),
    # downsample to native 32x32
    sdf4, muf4 = sd_f.view(LAT, 1, 1).to(DEV), mu_f.view(LAT, 1, 1).to(DEV)
    dec = ae.decode_fine(yf.reshape(B * T, LAT, 8, 8) * sdf4 + muf4)
    pred = down2((dec.clamp(-1, 1) + 1) * 0.5).view(B, T, 2, 32, 32).cpu().numpy()   # ->[0,1]
    # AE round-trip ceiling on the GT targets
    frt = torch.from_numpy(up2(xt.reshape(B * T, 2, 32, 32)) * 2 - 1).to(DEV)
    _, zft = ae.encode(frt)
    rt = down2((ae.decode_fine(zft).clamp(-1, 1) + 1) * 0.5).view(B, T, 2, 32, 32).cpu().numpy()
    pers = np.repeat(xc[:, 3:4], T, axis=1)                          # repeat last context frame

    log(f"\n{'='*88}\n  TAXIBJ BRIDGE 4->4 (OpenSTL convention, verbatim metric(); n={B} canonical "
        f"test pairs; nfe={nfe if mode=='diff' else 0})\n{'='*88}")
    log(f"  {'row':<16}{'MSE':>9}{'MAE':>10}{'SSIM':>9}{'PSNR':>8}")
    results = {}
    for name, p in [(arm, pred), ("AE-roundtrip", rt), ("persistence", pers)]:
        m = _openstl_metric(p, xt)
        results[name] = m
        log(f"  {name:<16}{m['mse']:>9.4f}{m['mae']:>10.2f}{m['ssim']:>9.4f}{m['psnr']:>8.2f}")
    mse_h = np.mean((pred - xt) ** 2, axis=0).reshape(T, -1).sum(axis=1)
    log("  per-frame MSE (pixel-sum): " + " ".join(f"h{h+1}:{v:.4f}" for h, v in enumerate(mse_h)))
    out = f"logs_rhcp/taxibj_bridge_eval_{arm}_s{seed}_t{steps}{tag}.npz"
    np.savez(out, mse_h=mse_h, pred=pred[:8], true=xt[:8],
             **{f"{n.replace('-','_')}_{k}": v for n, m in results.items() for k, v in m.items()})
    log(f"  saved {out}")
    _save_fig(arm, pred, xt, f"logs_rhcp/taxibj_bridge_{arm}_s{seed}_t{steps}{tag}.png")
    return results


def _save_fig(arm, pred, true, path, seq=0):
    """[FIG gen-5]: inflow channel heatmaps, pred vs GT, h1..h4 for one test pair."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(10, 5.2))
        for h in range(4):
            ax[0, h].imshow(true[seq, h, 0], cmap="viridis", vmin=0, vmax=1)
            ax[0, h].set_title(f"GT h{h+1}", fontsize=9)
            ax[1, h].imshow(pred[seq, h, 0], cmap="viridis", vmin=0, vmax=1)
            ax[1, h].set_title(f"{arm} h{h+1}", fontsize=9)
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(f"TaxiBJ 4->4 inflow: GT vs {arm} (test pair {seq})", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        log(f"  saved {path}")
    except Exception as e:  # noqa: BLE001
        log(f"  (figure skipped: {type(e).__name__}: {e})")


# ================================================================================================
# 6) SMOKE: AE 200 -> both reg arms 200 @B16 -> eval. Validates the whole chain in ~5 min.
# ================================================================================================
def run_smoke(args):
    M._BF16 = bool(args.bf16)
    ae_path = "logs_rhcp/hier_ae_taxibj_smoke.pt"
    log(f"=== TAXIBJ BRIDGE SMOKE | device {DEV} | bf16 {M._BF16} ===")
    run_ae(steps=200, out=ae_path)
    ae = load_ae(ae_path)
    data = TxbData(ae)
    for arm in ("txb-flat-reg", "txb-hier-reg"):
        run_train(arm, data, steps=200, B=16, seed=args.seed, val_every=100, save_every=200)
    for arm in ("txb-flat-reg", "txb-hier-reg"):
        run_eval(arm, ae, data, steps=200, seed=args.seed, nfe=args.nfe, tag="_smoke")
    log("SMOKE OK")


def run_all(args):
    """One-shot chain: AE -> BS probe (author's ~14 GB target) -> 4 arms -> 4 evals."""
    if not os.path.exists(args.ae_path):
        run_ae(steps=args.ae_steps, out=args.ae_path)
    else:
        log(f"[all] AE exists at {args.ae_path} -- reusing")
    ae = load_ae(args.ae_path)
    data = TxbData(ae)
    # probe on the largest-sequence arm with the diffusion head; one B for all arms
    B = run_probe("txb-flat-diff", data, target_gb=args.target_gb)
    order = ("txb-flat-reg", "txb-hier-reg", "txb-flat-diff", "txb-hier-diff")
    for arm in order:
        run_train(arm, data, args.steps, B, args.seed)
    for arm in order:
        run_eval(arm, ae, data, args.steps, seed=args.seed, nfe=args.nfe)
    log("TAXIBJ_ALL_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["ae", "train", "eval", "probe", "smoke", "all"])
    ap.add_argument("--arm", default="txb-flat-reg", choices=list(BRIDGE_ARMS))
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfe", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--target_gb", type=float, default=14.0)
    ap.add_argument("--ae_path", default=AE_PATH)
    ap.add_argument("--ae_steps", type=int, default=12000)
    args = ap.parse_args()
    M._BF16 = bool(args.bf16)

    if args.stage == "smoke":
        run_smoke(args); return
    if args.stage == "ae":
        run_ae(steps=args.ae_steps, out=args.ae_path); return
    if args.stage == "all":
        run_all(args); return
    ae = load_ae(args.ae_path)
    data = TxbData(ae)
    if args.stage == "probe":
        run_probe(args.arm, data, target_gb=args.target_gb)
    elif args.stage == "train":
        run_train(args.arm, data, args.steps, args.batch, args.seed)
    elif args.stage == "eval":
        run_eval(args.arm, ae, data, args.steps, seed=args.seed, nfe=args.nfe)


if __name__ == "__main__":
    main()
