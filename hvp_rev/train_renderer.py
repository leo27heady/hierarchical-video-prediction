"""Step 1 — adversarial (VQGAN) renderer on FROZEN CW-VAE codes. Red-teamed v2.

Arms (--arm): l2 (MSE, regression-to-mean control) | lpips (L1+LPIPS, the C0b baseline) |
gan (L1+LPIPS+adaptive-PatchGAN, the method). Fresh-init sigmoid decoder, all arms matched on
init/steps/data so arm 'l2' is the PRIMARY reference (isolates the objective).

Trains on TEACHER-FORCED codes (the clean 'is sharpness reachable from these codes' question).
Eval: (1) PRIMARY TF-recon FID + fg-LPIPS + fg-MSE; (2) DIAGNOSTIC open-loop (swap G into the frozen
CW-VAE decoder, free-run) FID + fg-LPIPS + fg-MSE + temporal-LPIPS. Cached real-stats; weights-only
atomic ckpt; disk<3GB & VRAM>6GB guards. Smoke: --arm gan --steps 2000 --seed 0 (one strip + D-acc/lambda).
Usage: python -m hvp_rev.train_renderer --arm gan --seed 0 --steps 8000 --run-name s1_gan_s0
"""
from __future__ import annotations
import sys, os, math, time, json, argparse, glob, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
from hvp_rev.codes import teacher_forced_codes
from hvp_rev.modules_gan import RendererDecoder, PatchGAN, hinge_d_loss, hinge_g_loss, adaptive_lambda, WarmStartG
from hvp_rev import fid as fidmod
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LPIPS_W, WARMUP_DEFAULT = 1.0, 1000


def load_cwvae():
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
    m.load_state_dict(torch.load(str(CW / "logs_torch" / "cwvae" / "last.pt"), map_location=dev)["model"]); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
def lapv(img): return F.conv2d(img.reshape(-1,1,64,64), _lap, padding=1).var(dim=(1,2,3)).mean().item()

def disk_free_gb(p): return shutil.disk_usage(str(p)).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["l2","lpips","gan"], required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    ap.add_argument("--train-clips", type=int, default=120); ap.add_argument("--eval-clips", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=36); ap.add_argument("--hor", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--warm-start", action="store_true", help="init G from the pretrained CW-VAE decoder (sharpen, not build from scratch)")
    ap.add_argument("--fg-weight", type=float, default=0.0, help="extra L1 weight on foreground pixels (GT>0.1) to pin sparse digit structure")
    ap.add_argument("--run-name", required=True); ap.add_argument("--logdir", default=str(ROOT/"logs_render"))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.logdir); (out/"strips").mkdir(parents=True, exist_ok=True)
    if disk_free_gb(out) < 3: sys.exit(f"ABORT: disk {disk_free_gb(out):.1f}GB<3")

    m = load_cwvae()
    import lpips; LP = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    def lpips_map(p, g):                                    # p,g (N,1,64,64) in [0,1] -> (N,1,h,w) distance map
        return LP(p.repeat(1,3,1,1)*2-1, g.repeat(1,3,1,1)*2-1)
    def fg_lpips(p, g):
        fg = (g > 0.1).float(); dm = F.interpolate(lpips_map(p,g), size=(64,64), mode="bilinear", align_corners=False)
        return ((dm*fg).sum()/fg.sum().clamp(min=1)).item()

    # ---- frozen code datasets (TF train, TF eval, open-loop eval frames) ----
    tr_raw = np.load(str(CW/"data_mmnist"/"train.npy")); te_raw = np.load(str(CW/"data_mmnist"/"test.npy"))
    def tf_set(raw, n, T=86, seed=0):
        rng = np.random.default_rng(seed); idx = rng.integers(0, raw.shape[0], n)
        x = torch.from_numpy(raw[idx, :T].astype(np.float32)/255).unsqueeze(2).to(dev)
        c, f = teacher_forced_codes(m, x, sample=True); return c, f
    tr_c, tr_f = tf_set(tr_raw, a.train_clips, seed=1)                  # renderer TRAIN pairs (train.npy, disjoint)
    x_ev = torch.from_numpy(te_raw[:a.eval_clips, :a.ctx+a.hor].astype(np.float32)/255).unsqueeze(2).to(dev)  # held-out test
    ev_c, ev_f = teacher_forced_codes(m, x_ev, sample=True)            # TF eval pairs (same held-out clips)
    gt_ol = x_ev[:, a.ctx:].reshape(-1,1,64,64)
    print(f"[{a.run_name}] arm={a.arm} TF-train {tr_c.shape[0]} pairs | TF-eval {ev_c.shape[0]} | OL-eval {gt_ol.shape[0]} frames | disk {disk_free_gb(out):.1f}GB", flush=True)
    # cached REAL inception stats (one each), reused across all evals
    mu_tf, sig_tf = fidmod._stats(fidmod.features(ev_f, dev))
    mu_ol, sig_ol = fidmod._stats(fidmod.features(gt_ol, dev))

    G = (WarmStartG(m.decoder) if a.warm_start else RendererDecoder(220)).to(dev)
    print(f"  G = {'WARM-START(pretrained decoder)' if a.warm_start else 'fresh RendererDecoder'}", flush=True)
    optG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5,0.9))
    D = optD = None
    if a.arm == "gan":
        D = PatchGAN(1, 64, 3, spectral=True).to(dev); optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5,0.9))
        print(f"  PatchGAN params {sum(p.numel() for p in D.parameters())/1e6:.2f}M", flush=True)

    ck = out/f"{a.run_name}_last.pt"; start = 0; hist = []
    if ck.exists():
        sd = torch.load(ck, map_location=dev); G.load_state_dict(sd["G"]); start = sd["step"]; hist = sd.get("hist", [])
        if D is not None and "D" in sd: D.load_state_dict(sd["D"])
        print(f"  resumed @ {start}", flush=True)

    @torch.no_grad()
    def evaluate():
        G.eval()
        # TF recon
        pr = torch.cat([G(ev_c[i:i+256]) for i in range(0,ev_c.shape[0],256)], 0).clamp(0,1)
        tf_fgmse = float((((pr-ev_f)**2)*(ev_f>0.1)).sum()/ (ev_f>0.1).float().sum().clamp(min=1))
        tf = {"fid": fidmod.frechet(*fidmod._stats(fidmod.features(pr,dev)), mu_tf, sig_tf),
              "fglpips": fg_lpips(pr, ev_f), "fgmse": tf_fgmse, "lapvar": lapv(pr)}
        # open-loop: swap G into the decoder, free-run
        orig = m.decoder; m.decoder = G
        torch.manual_seed(0); pol = m.open_loop(x_ev, a.ctx, sample=True)[:, :a.hor]
        m.decoder = orig
        polf = pol.reshape(-1,1,64,64)
        ol_fgmse = float((((polf-gt_ol)**2)*(gt_ol>0.1)).sum()/(gt_ol>0.1).float().sum().clamp(min=1))
        # temporal-LPIPS coherence: mean LPIPS between consecutive frames (pred vs GT)
        def tcoh(v):
            d = lpips_map(v[:,1:].reshape(-1,1,64,64), v[:,:-1].reshape(-1,1,64,64)); return float(d.mean())
        ol = {"fid": fidmod.frechet(*fidmod._stats(fidmod.features(polf,dev)), mu_ol, sig_ol),
              "fglpips": fg_lpips(polf, gt_ol), "fgmse": ol_fgmse, "lapvar": lapv(pol),
              "tcoh_pred": tcoh(pol), "tcoh_gt": tcoh(x_ev[:, a.ctx:])}
        G.train(); return {"tf": tf, "ol": ol}

    def strip(step):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        with torch.no_grad():
            G.eval()
            rec = G(ev_c[:8]).clamp(0,1).cpu().numpy()
            orig = m.decoder; m.decoder = G; torch.manual_seed(0)
            pol = m.open_loop(x_ev[:4], a.ctx, sample=True)[:, :a.hor].cpu().numpy(); m.decoder = orig
            G.train()
        cols = [0,4,7,24,49]
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
        for old in sorted(glob.glob(str(out/"strips"/f"{a.run_name}_*.png")), key=os.path.getmtime)[:-8]: os.unlink(old)

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
            # D step
            with torch.no_grad(): pdet = pred.detach()
            d_real = D(real); d_fake = D(pdet); ldz = hinge_d_loss(d_real, d_fake)
            optD.zero_grad(set_to_none=True); ldz.backward(); optD.step()
            dacc = float(((d_real>0).float().mean()+(d_fake<0).float().mean())/2)
            # G adv step (adaptive lambda)
            g_adv = hinge_g_loss(D(pred))
            lam = adaptive_lambda(loss_nll, g_adv, G.last_layer())
            loss = loss_nll + lam*g_adv; info["adv"] = float(g_adv); info["lam"] = float(lam); info["dacc"] = dacc
        else:
            loss = loss_nll
        optG.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 100.0); optG.step()
        if step==start+1 and dev.type=="cuda":
            assert torch.cuda.max_memory_reserved() < 6e9, f"VRAM {torch.cuda.max_memory_reserved()/1e9:.1f}>6GB"
        if step % a.log_every == 0:
            vram = torch.cuda.max_memory_reserved()/1e9 if dev.type=="cuda" else 0
            extra = (f"lam {info.get('lam',0):.3f} dacc {info.get('dacc',-1):.2f}" if a.arm=="gan" else "")
            print(f"  [{step:5d}] nll {float(loss_nll):.4f} l1 {info['l1']:.4f} {extra} gn {float(gn):.1f} "
                  f"VRAM {vram:.2f}GB {step/(time.time()-t0):.1f}it/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(); ev["step"] = step; hist.append(ev); strip(step)
            print(f"  EVAL[{step}] TF: fid {ev['tf']['fid']:.1f} fglpips {ev['tf']['fglpips']:.3f} fgmse {ev['tf']['fgmse']:.3f} lapvar {ev['tf']['lapvar']:.4f} | "
                  f"OL: fid {ev['ol']['fid']:.1f} fglpips {ev['ol']['fglpips']:.3f} fgmse {ev['ol']['fgmse']:.3f} lapvar {ev['ol']['lapvar']:.4f} "
                  f"tcoh {ev['ol']['tcoh_pred']:.3f}/{ev['ol']['tcoh_gt']:.3f} | disk {disk_free_gb(out):.1f}GB", flush=True)
            (out/f"{a.run_name}_hist.json").write_text(json.dumps(hist, indent=1))
        if step % a.ckpt_every == 0 or step == a.steps:
            if disk_free_gb(out) < 3: sys.exit("ABORT before ckpt: disk<3GB")
            sd = {"G": G.state_dict(), "step": step, "hist": hist, "arm": a.arm, "seed": a.seed,
                  "warm": a.warm_start, "fg_weight": a.fg_weight}
            if D is not None: sd["D"] = D.state_dict()
            tmp = str(ck)+".tmp"; torch.save(sd, tmp); os.replace(tmp, str(ck))
    print(f"[{a.run_name}] DONE @ {a.steps}", flush=True)


if __name__ == "__main__":
    main()
