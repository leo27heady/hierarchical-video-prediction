"""E1 — on-the-fly rollout training on the CW-VAE backbone (red-teamed v2 design).

Three arms (one trained model each, via --arm):
  lambda0  : continue TEACHER-FORCED ELBO only (the 'just more fine-tuning' control)
  nograd   : ELBO_A + rollout NLL, but the free-run latents are DETACHED before decode
             (decoder-only rollout gradient; isolates 'exposure' from 'gradient through dynamics')
  withgrad : ELBO_A + rollout NLL with gradient through the free-run prior dynamics (THE method)

Corrected mechanics: Phase B calls hierarchical_unroll(use_obs=[False]*L, initial=carried) DIRECTLY
(never open_loop, which is @torch.no_grad); Phase-A->B state DETACHED (run 1); rollout loss is
Gaussian-NLL on the prior-MEAN free-run (sample=False -> no std-collapse incentive); per-frame
normalized. Warm-started from ../cwvae/logs_torch/cwvae/last.pt (pinned commit: see ../cwvae HEAD).

Live tracking + disk/GPU discipline + resumable single _last.pt (temp+os.replace). Usage:
  python -m hvp_rev.train_rollout --arm withgrad --seed 0 --steps 8000 --run-name e1_withgrad_s0
"""
from __future__ import annotations
import sys, os, math, time, json, argparse, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONST = 0.5 * math.log(2 * math.pi) * 4096                 # Gaussian-NLL constant (var=1, 4096 px)


def gauss_nll(pred, tgt, var):                             # sum over px+frames, mean over batch
    B, T = pred.shape[:2]
    return (0.5 * ((tgt - pred) ** 2) / var + 0.5 * math.log(2 * math.pi * var)).reshape(B, T, -1).sum(-1).sum(1).mean()


def kl_sum(q, p):
    return (torch.log(p["stddev"] / q["stddev"]) + (q["stddev"] ** 2 + (q["mean"] - p["mean"]) ** 2) / (2 * p["stddev"] ** 2) - 0.5).sum(-1).sum(1).mean()


def elbo_full(m, obs):                                     # teacher-forced ELBO over the whole clip (lambda0 arm)
    B, T = obs.shape[:2]; var = m.dec_stddev ** 2
    dec, pri, pos = m(obs)
    nll = gauss_nll(dec, obs, var)
    kl = obs.new_zeros(())
    for i in range(m.levels): kl = kl + kl_sum(pos[i], pri[i])
    return (nll + kl) / T, {"nll": float(nll / T), "kl": float(kl / T)}


def rollout_loss(m, obs, ctx_tf, M, lam, arm):
    """ELBO_A (teacher-forced ctx) + lam * rollout NLL (free-run prior MEAN, sample=False).
    arm='nograd' detaches the free-run latents before decode (decoder-only grad)."""
    L, B = m.levels, obs.shape[0]; var = m.dec_stddev ** 2
    enc = m.encoder(obs)
    pre, post, cl = [], [], ctx_tf
    for lvl in range(L):
        pre.append(enc[lvl][:, :cl]); post.append(enc[lvl][:, cl:]); cl = cl // m.factor
    # Phase A: posterior teacher-forced ELBO
    botA, lastA, priA, posA = m.hierarchical_unroll(pre, use_obs=True, sample=True)
    Ta = pre[0].shape[1]
    decA = m.decoder(botA.reshape(B * Ta, -1)).reshape(B, Ta, 1, 64, 64)
    klA = obs.new_zeros(())
    for i in range(L): klA = klA + kl_sum(posA[i], priA[i])
    elboA = (gauss_nll(decA, obs[:, :ctx_tf], var) + klA) / ctx_tf
    # Phase B: free-run prior MEAN (sample=False), from DETACHED carried state
    initB = [{k: v.detach() for k, v in st.items()} for st in lastA]
    botB, _, priB, _ = m.hierarchical_unroll(post, use_obs=[False] * L, initial=initB, sample=False)
    Mb = post[0].shape[1]
    botB_in = botB.detach() if arm == "nograd" else botB
    decB = m.decoder(botB_in.reshape(B * Mb, -1)).reshape(B, Mb, 1, 64, 64)
    rollout = gauss_nll(decB, obs[:, ctx_tf:ctx_tf + Mb], var) / Mb
    loss = elboA + lam * rollout
    fr_std = float(torch.stack([priB[i]["stddev"].mean() for i in range(L)]).mean())
    return loss, {"elboA": float(elboA), "rollout_SE": float(rollout) - CONST, "freerun_std": fr_std, "Mb": Mb}


# ---------------------------------------------------------------- eval
_lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
def lapv(img): return F.conv2d(img.reshape(-1, 1, 64, 64), _lap, padding=1).var(dim=(1, 2, 3)).mean().item()

@torch.no_grad()
def evaluate(m, test, lp, ctx=36, hor=50, clips=32, seed=0):
    torch.manual_seed(seed)
    x = torch.from_numpy(test[:clips, :ctx + hor].astype(np.float32) / 255.0).unsqueeze(2).to(dev)
    gt = x[:, ctx:]; gt_np = gt.cpu().numpy(); fg = gt_np > 0.1
    pred = m.open_loop(x, ctx, sample=True)[:, :hor]
    p = pred.cpu().numpy()
    def fgmse(t): return float(np.mean([((p[c, t] - gt_np[c, t]) ** 2)[fg[c, t]].mean() if fg[c, t].any() else 0.0 for c in range(clips)]))
    def fglpips(t):
        P = pred[:, t].repeat(1, 3, 1, 1) * 2 - 1; G = gt[:, t].repeat(1, 3, 1, 1) * 2 - 1
        dm = F.interpolate(lp(P, G), size=(64, 64), mode="bilinear", align_corners=False)
        mk = torch.from_numpy(fg[:, t].astype(np.float32)).to(dev)
        return float((dm * mk).sum() / mk.sum().clamp(min=1))
    # top-level KL (collapse guard) via a teacher-forced pass
    _, pri, pos = m(x[:, :ctx])
    topkl = float(kl_sum(pos[-1], pri[-1]))
    met = {"lap_var": lapv(pred)}
    for t in [4, 7, 24, 49]:                               # +5, +8(=M), +25, +50  (note +4 idx => +5 frame)
        met[f"fgmse@{t+1}"] = fgmse(t); met[f"fglpips@{t+1}"] = fglpips(t)
    met["top_kl"] = topkl
    return met, pred, gt


def save_strip(pred, gt, path, n=4):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    p = pred.cpu().numpy(); g = gt.cpu().numpy(); cols = [0, 4, 7, 24, 49]
    fig, ax = plt.subplots(2 * n, len(cols), figsize=(len(cols) * 1.1, 2 * n * 1.1))
    for c in range(n):
        for j, t in enumerate(cols):
            ax[2*c, j].imshow(np.clip(g[c, t, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax[2*c+1, j].imshow(np.clip(p[c, t, 0], 0, 1), cmap="gray", vmin=0, vmax=1)
            for r in (2*c, 2*c+1): ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if c == 0: ax[0, j].set_title(f"+{t+1}", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def disk_free_gb(p): import shutil; return shutil.disk_usage(str(p)).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["lambda0", "nograd", "withgrad"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ctx", type=int, default=36); ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--run-name", required=True); ap.add_argument("--logdir", default=str(ROOT / "logs_rollout"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.logdir); (out / "strips").mkdir(parents=True, exist_ok=True)
    if disk_free_gb(out) < 3: sys.exit(f"ABORT: free disk {disk_free_gb(out):.1f}GB < 3GB")

    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
    m.load_state_dict(torch.load(str(CW / "logs_torch" / "cwvae" / "last.pt"), map_location=dev)["model"])
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    ck = out / f"{a.run_name}_last.pt"; start = 0; hist = []
    if ck.exists():
        sd = torch.load(ck, map_location=dev); m.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        start = sd["step"]; hist = sd.get("hist", []); print(f"resumed {a.run_name} @ {start}", flush=True)

    import lpips; lp = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    train = np.load(str(CW / "data_mmnist" / "train.npy"))
    def get_batch(step):
        rng = np.random.default_rng(a.seed * 100003 + step)
        idx = rng.integers(0, train.shape[0], a.batch); t0 = rng.integers(0, train.shape[1] - (a.ctx + a.M), a.batch)
        cl = np.stack([train[i, s:s + a.ctx + a.M] for i, s in zip(idx, t0)])
        return torch.from_numpy(cl.astype(np.float32) / 255.0).unsqueeze(2).to(dev)
    test = np.load(str(CW / "data_mmnist" / "test.npy"))
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()

    print(f"[{a.run_name}] arm={a.arm} seed={a.seed} steps={start}->{a.steps} batch={a.batch} "
          f"diskfree={disk_free_gb(out):.1f}GB", flush=True)
    t0 = time.time()
    for step in range(start + 1, a.steps + 1):
        ob = get_batch(step)
        if a.arm == "lambda0":
            loss, info = elbo_full(m, ob); info = {"elboA": info["nll"] + info["kl"], "rollout_SE": 0.0, "freerun_std": -1, "Mb": a.M}
        else:
            loss, info = rollout_loss(m, ob, a.ctx, a.M, a.lam, a.arm)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0); opt.step()
        if step == start + 1 and dev.type == "cuda":
            assert torch.cuda.max_memory_reserved() < 12e9, "VRAM tripwire >12GB"
        if step % a.log_every == 0:
            vram = torch.cuda.max_memory_reserved() / 1e9 if dev.type == "cuda" else 0
            print(f"  [{step:5d}] loss {float(loss):8.1f} elboA {info['elboA']:8.1f} rollSE {info['rollout_SE']:7.2f} "
                  f"frstd {info['freerun_std']:.3f} gn {float(gn):.1f} VRAM {vram:.2f}GB {step/(time.time()-t0+1e-9):.1f}it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            met, pred, gt = evaluate(m, test, lp)
            met["step"] = step; hist.append(met)
            save_strip(pred, gt, out / "strips" / f"{a.run_name}_{step}.png")
            strips = sorted(glob.glob(str(out / "strips" / f"{a.run_name}_*.png")), key=os.path.getmtime)
            for old in strips[:-10]: os.unlink(old)               # keep last 10
            print(f"  EVAL[{step}] fgMSE@8 {met['fgmse@8']:.3f} @25 {met['fgmse@25']:.3f} @50 {met['fgmse@50']:.3f} | "
                  f"fgLPIPS@8 {met['fglpips@8']:.3f} @50 {met['fglpips@50']:.3f} | lapvar {met['lap_var']:.4f} | "
                  f"topKL {met['top_kl']:.3f} | diskfree {disk_free_gb(out):.1f}GB", flush=True)
            (out / f"{a.run_name}_hist.json").write_text(json.dumps(hist, indent=1))
        if step % a.ckpt_every == 0 or step == a.steps:
            if disk_free_gb(out) < 3: sys.exit(f"ABORT before ckpt: free {disk_free_gb(out):.1f}GB")
            tmp = str(ck) + ".tmp"
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist,
                        "cfg": vars(a)}, tmp)
            os.replace(tmp, str(ck))                               # atomic, single file
    print(f"[{a.run_name}] DONE @ {a.steps}", flush=True)


if __name__ == "__main__":
    main()
