"""H1 — perceptual/adversarial renderer on FROZEN DLH codes (the validated sharpness fix, ported to DLH).

Decouples rendering from dynamics: freeze the DLH backbone, train a sharp decoder on its 200-dim decoder-input
code `c`. Arms (--arm): l2 (regression-to-mean control) | lpips (L1+LPIPS) | gan (L1+LPIPS+adaptive-PatchGAN).
Codes (--codes): tf (teacher-forced, the 'is sharpness reachable' question — C0 said yes, lap 0.042) OR
freerun (open-loop prior codes — the RED-TEAM's open question: does the renderer sharpness SURVIVE free-run?).
Eval swaps G into DLH.img_dec and free-runs: reports OL FID / fg-LPIPS / fg-MSE / lap_var + TF recon.

Usage:
  python -m hvp_rev.train_renderer_dlh --arm lpips --codes freerun --warm-start --fg-weight 8 \
      --ckpt logs_dlh/baseline_iter8000.pt --steps 1500 --run-name dlh_lpips_freerun
"""
from __future__ import annotations
import sys, os, time, json, argparse, glob, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(ROOT))
from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_rev.codes_dlh import teacher_forced_codes_dlh, free_run_codes_dlh
from hvp_rev.modules_gan import (RendererDecoder, PatchGAN, hinge_d_loss, hinge_g_loss,
                                 adaptive_lambda, WarmStartG)
from hvp_rev import fid as fidmod
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_W = 1.0

_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
def lapv(img): return F.conv2d(img.reshape(-1,1,64,64), _lap, padding=1).var(dim=(1,2,3)).mean().item()
def disk_free_gb(p): return shutil.disk_usage(str(p)).free / 1e9


def load_dlh(ckpt):
    sd = torch.load(ckpt, map_location=dev); cfg = DLHConfig(**sd["cfg"])
    m = DLH(cfg).to(dev); m.load_state_dict(sd["model"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["l2","lpips","gan"], required=True)
    ap.add_argument("--codes", choices=["tf","freerun"], default="freerun")
    ap.add_argument("--ckpt", default=str(ROOT/"logs_dlh"/"baseline_iter8000.pt"))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--train-clips", type=int, default=120); ap.add_argument("--eval-clips", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=36); ap.add_argument("--hor", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=750); ap.add_argument("--ckpt-every", type=int, default=750)
    ap.add_argument("--log-every", type=int, default=150)
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--fg-weight", type=float, default=0.0)
    ap.add_argument("--run-name", required=True); ap.add_argument("--logdir", default=str(ROOT/"logs_render"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.logdir); (out/"strips").mkdir(parents=True, exist_ok=True)
    if disk_free_gb(out) < 3: sys.exit(f"ABORT: disk {disk_free_gb(out):.1f}GB<3")

    m, cfg = load_dlh(a.ckpt)
    import lpips; LP = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    def lpips_map(p, g): return LP(p.repeat(1,3,1,1)*2-1, g.repeat(1,3,1,1)*2-1)
    def fg_lpips(p, g):
        fg = (g > 0.1).float(); dm = F.interpolate(lpips_map(p,g), size=(64,64), mode="bilinear", align_corners=False)
        return ((dm*fg).sum()/fg.sum().clamp(min=1)).item()

    tr_raw = np.load(str(CW/"data_mmnist"/"train.npy")); te_raw = np.load(str(CW/"data_mmnist"/"test.npy"))
    def build_train(seed):
        rng = np.random.default_rng(seed); idx = rng.integers(0, tr_raw.shape[0], a.train_clips)
        if a.codes == "freerun":
            x = torch.from_numpy(tr_raw[idx, :a.ctx+a.hor].astype(np.float32)/255).unsqueeze(2).to(dev)
            return free_run_codes_dlh(m, x, a.ctx, a.hor)
        x = torch.from_numpy(tr_raw[idx, :86].astype(np.float32)/255).unsqueeze(2).to(dev)
        return teacher_forced_codes_dlh(m, x)
    tr_c, tr_f = build_train(seed=1)
    x_ev = torch.from_numpy(te_raw[:a.eval_clips, :a.ctx+a.hor].astype(np.float32)/255).unsqueeze(2).to(dev)
    ev_c, ev_f = teacher_forced_codes_dlh(m, x_ev[:, :86] if x_ev.shape[1] >= 86 else x_ev)
    gt_ol = x_ev[:, a.ctx:].reshape(-1,1,64,64)
    code_dim = tr_c.shape[-1]
    print(f"[{a.run_name}] arm={a.arm} codes={a.codes} dim={code_dim} | train {tr_c.shape[0]} | TFeval {ev_c.shape[0]} | "
          f"OLeval {gt_ol.shape[0]} | disk {disk_free_gb(out):.1f}GB", flush=True)
    mu_tf, sig_tf = fidmod._stats(fidmod.features(ev_f, dev))
    mu_ol, sig_ol = fidmod._stats(fidmod.features(gt_ol, dev))

    G = (WarmStartG(m.img_dec) if a.warm_start else RendererDecoder(code_dim)).to(dev)
    print(f"  G = {'WARM-START(img_dec)' if a.warm_start else 'fresh RendererDecoder'}", flush=True)
    optG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5,0.9))
    D = optD = None
    if a.arm == "gan":
        D = PatchGAN(1, 64, 3, spectral=True).to(dev); optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5,0.9))

    @torch.no_grad()
    def evaluate():
        G.eval()
        pr = torch.cat([G(ev_c[i:i+256]) for i in range(0,ev_c.shape[0],256)], 0).clamp(0,1)
        tf_fgmse = float((((pr-ev_f)**2)*(ev_f>0.1)).sum()/(ev_f>0.1).float().sum().clamp(min=1))
        tf = {"fid": fidmod.frechet(*fidmod._stats(fidmod.features(pr,dev)), mu_tf, sig_tf),
              "fglpips": fg_lpips(pr, ev_f), "fgmse": tf_fgmse, "lapvar": lapv(pr)}
        # open-loop: swap G into DLH's img_dec, free-run
        orig = m.img_dec; m.img_dec = G
        torch.manual_seed(0); pol, _ = m.open_loop(x_ev, a.ctx, a.hor)
        m.img_dec = orig
        polf = pol.reshape(-1,1,64,64)
        ol_fgmse = float((((polf-gt_ol)**2)*(gt_ol>0.1)).sum()/(gt_ol>0.1).float().sum().clamp(min=1))
        ol = {"fid": fidmod.frechet(*fidmod._stats(fidmod.features(polf,dev)), mu_ol, sig_ol),
              "fglpips": fg_lpips(polf, gt_ol), "fgmse": ol_fgmse, "lapvar": lapv(pol)}
        G.train(); return {"tf": tf, "ol": ol}

    def strip(step):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            G.eval(); rec = G(ev_c[:8]).clamp(0,1).cpu().numpy()
            orig = m.img_dec; m.img_dec = G; torch.manual_seed(0)
            pol, _ = m.open_loop(x_ev[:4], a.ctx, a.hor); pol = pol.cpu().numpy(); m.img_dec = orig
            G.train()
        cols = [0,4,7,24,49]; cols = [c for c in cols if c < a.hor]
        fig, ax = plt.subplots(6, len(cols), figsize=(len(cols)*1.1, 6*1.1))
        for j,t in enumerate(cols):
            for c in range(2):
                ax[c,j].imshow(np.clip(x_ev[c, a.ctx+t,0].cpu().numpy(),0,1),cmap="gray",vmin=0,vmax=1)
                ax[2+c,j].imshow(np.clip(pol[c,t,0],0,1),cmap="gray",vmin=0,vmax=1)
            ax[4,j].imshow(np.clip(ev_f[j,0].cpu().numpy(),0,1),cmap="gray",vmin=0,vmax=1)
            ax[5,j].imshow(np.clip(rec[j,0],0,1),cmap="gray",vmin=0,vmax=1)
            for r in range(6): ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
            if j==0:
                for r,l in enumerate(["GT","GT","OL","OL","TFgt","TFrec"]): ax[r,0].set_ylabel(l,fontsize=7)
        fig.suptitle(f"{a.run_name} step {step}", fontsize=8); fig.tight_layout()
        fig.savefig(out/"strips"/f"{a.run_name}_{step}.png", dpi=85); plt.close(fig)

    ck = out/f"{a.run_name}_last.pt"; start = 0; hist = []
    if ck.exists():
        sd = torch.load(ck, map_location=dev); G.load_state_dict(sd["G"]); start = sd["step"]; hist = sd.get("hist", [])
        if D is not None and "D" in sd: D.load_state_dict(sd["D"])
        print(f"  resumed @ {start}", flush=True)

    N = tr_c.shape[0]; t0 = time.time()
    if dev.type=="cuda": torch.cuda.reset_peak_memory_stats()
    for step in range(start+1, a.steps+1):
        idx = torch.randint(0, N, (a.batch,), device=dev); code = tr_c[idx]; real = tr_f[idx]
        pred = G(code)
        if a.fg_weight > 0:
            w = 1.0 + a.fg_weight*(real > 0.1).float(); l1 = (w*(pred-real).abs()).sum()/w.sum()
        else:
            l1 = F.l1_loss(pred, real)
        if a.arm == "l2":
            loss_nll = F.mse_loss(pred, real); info = {"l1": float(l1)}
        else:
            lp = lpips_map(pred, real).mean(); loss_nll = l1 + LPIPS_W*lp; info = {"l1": float(l1), "lpips": float(lp)}
        lam = 0.0; dacc = -1
        if a.arm == "gan" and step > a.warmup:
            d_real = D(real); d_fake = D(pred.detach()); ldz = hinge_d_loss(d_real, d_fake)
            optD.zero_grad(set_to_none=True); ldz.backward(); optD.step()
            dacc = float(((d_real>0).float().mean()+(d_fake<0).float().mean())/2)
            g_adv = hinge_g_loss(D(pred)); lam = adaptive_lambda(loss_nll, g_adv, G.last_layer())
            loss = loss_nll + lam*g_adv; info.update(adv=float(g_adv), lam=float(lam), dacc=dacc)
        else:
            loss = loss_nll
        optG.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 100.0); optG.step()
        if step % a.log_every == 0:
            vram = torch.cuda.max_memory_reserved()/1e9 if dev.type=="cuda" else 0
            extra = (f"lam {info.get('lam',0):.3f} dacc {info.get('dacc',-1):.2f}" if a.arm=="gan" else "")
            print(f"  [{step:5d}] nll {float(loss_nll):.4f} l1 {info['l1']:.4f} {extra} gn {float(gn):.1f} "
                  f"VRAM {vram:.2f}GB {step/(time.time()-t0):.1f}it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(); ev["step"] = step; hist.append(ev); strip(step)
            print(f"  EVAL[{step}] TF: fid {ev['tf']['fid']:.1f} fglpips {ev['tf']['fglpips']:.3f} "
                  f"fgmse {ev['tf']['fgmse']:.3f} lapvar {ev['tf']['lapvar']:.4f} | OL: fid {ev['ol']['fid']:.1f} "
                  f"fglpips {ev['ol']['fglpips']:.3f} fgmse {ev['ol']['fgmse']:.3f} lapvar {ev['ol']['lapvar']:.4f}", flush=True)
            (out/f"{a.run_name}_hist.json").write_text(json.dumps(hist, indent=1))
        if step % a.ckpt_every == 0 or step == a.steps:
            if disk_free_gb(out) < 3: sys.exit("ABORT before ckpt: disk<3GB")
            sd = {"G": G.state_dict(), "step": step, "hist": hist, "arm": a.arm, "warm": a.warm_start}
            if D is not None: sd["D"] = D.state_dict()
            tmp = str(ck)+".tmp"; torch.save(sd, tmp); os.replace(tmp, str(ck))
    print(f"[{a.run_name}] DONE @ {a.steps}", flush=True)


if __name__ == "__main__":
    main()
