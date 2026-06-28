"""Step-1 GATE — compare the renderer arms (l2 / lpips / gan) vs the frozen-baseline decoder.

Re-evaluates each arm's final G on held-out clips (TF-recon + open-loop), with the fresh-L2 arm as the
PRIMARY reference and the pretrained decoder as a secondary anchor. Pareto gate: the gan arm should
improve the distribution metrics (FID, fg-LPIPS) over lpips-only at no-worse fg-MSE/coherence. Saves a
combined GT/baseline/l2/lpips/gan open-loop strip. Usage: python -m hvp_rev.gate_renderer [--seed 0]
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
from hvp_rev.codes import teacher_forced_codes
from hvp_rev.modules_gan import RendererDecoder
from hvp_rev import fid as fidmod
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = ROOT / "logs_render"
CTX, HOR, CLIPS, RSEEDS = 36, 50, 24, 3
_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
def lapv(img): return F.conv2d(img.reshape(-1,1,64,64), _lap, padding=1).var(dim=(1,2,3)).mean().item()

def load_cwvae():
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
    m.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"]); m.eval()
    return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    import lpips; LP = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()
    def lpmap(p,g): return LP(p.repeat(1,3,1,1)*2-1, g.repeat(1,3,1,1)*2-1)
    def fglp(p,g):
        fg=(g>0.1).float(); dm=F.interpolate(lpmap(p,g),size=(64,64),mode="bilinear",align_corners=False)
        return float((dm*fg).sum()/fg.sum().clamp(min=1))
    def tcoh(v): return float(lpmap(v[:,1:].reshape(-1,1,64,64), v[:,:-1].reshape(-1,1,64,64)).mean())

    m = load_cwvae()
    te = np.load(str(CW/"data_mmnist"/"test.npy"))
    x = torch.from_numpy(te[:CLIPS, :CTX+HOR].astype(np.float32)/255).unsqueeze(2).to(dev)
    tf_c, tf_f = teacher_forced_codes(m, x, sample=True)
    gt_ol = x[:, CTX:].reshape(-1,1,64,64)
    mu_tf, sig_tf = fidmod._stats(fidmod.features(tf_f, dev))
    mu_ol, sig_ol = fidmod._stats(fidmod.features(gt_ol, dev))

    @torch.no_grad()
    def eval_decoder(dec, is_G):
        # TF recon
        if is_G:
            pr = torch.cat([dec(tf_c[i:i+256]) for i in range(0,tf_c.shape[0],256)],0).clamp(0,1)
        else:
            pr = torch.cat([dec(tf_c[i:i+256]).clamp(0,1) for i in range(0,tf_c.shape[0],256)],0)
        tf = {"fid": fidmod.frechet(*fidmod._stats(fidmod.features(pr,dev)), mu_tf, sig_tf), "fglpips": fglp(pr,tf_f),
              "fgmse": float((((pr-tf_f)**2)*(tf_f>0.1)).sum()/(tf_f>0.1).float().sum().clamp(min=1)), "lapvar": lapv(pr)}
        # open-loop with this decoder, averaged over rollout seeds
        orig = m.decoder; m.decoder = dec; fids=[]; fls=[]; lvs=[]; tcs=[]; mses=[]
        for s in range(RSEEDS):
            torch.manual_seed(s); pol = m.open_loop(x, CTX, sample=True)[:, :HOR]; polf = pol.reshape(-1,1,64,64)
            fids.append(fidmod.frechet(*fidmod._stats(fidmod.features(polf,dev)), mu_ol, sig_ol))
            fls.append(fglp(polf,gt_ol)); lvs.append(lapv(pol)); tcs.append(tcoh(pol))
            mses.append(float((((polf-gt_ol)**2)*(gt_ol>0.1)).sum()/(gt_ol>0.1).float().sum().clamp(min=1)))
        m.decoder = orig
        ol = {"fid": float(np.mean(fids)), "fid_sd": float(np.std(fids)), "fglpips": float(np.mean(fls)),
              "fgmse": float(np.mean(mses)), "lapvar": float(np.mean(lvs)), "tcoh": float(np.mean(tcs))}
        return {"tf": tf, "ol": ol}

    res = {"baseline": eval_decoder(m.decoder, is_G=False)}
    Gs = {}
    for arm in ["l2","lpips","gan"]:
        ck = LOG/f"s1_{arm}_s{a.seed}_last.pt"
        if not ck.exists(): print(f"  (missing {ck.name})"); continue
        G = RendererDecoder(220).to(dev); G.load_state_dict(torch.load(ck,map_location=dev)["G"]); G.eval()
        Gs[arm] = G; res[arm] = eval_decoder(G, is_G=True)

    print(f"\n{'decoder':>10} | {'TF_fid':>7}{'TF_fglp':>8}{'TF_fgmse':>9}{'TF_lapv':>8} | {'OL_fid':>8}{'OL_fglp':>8}{'OL_fgmse':>9}{'OL_lapv':>8}{'OL_tcoh':>8}")
    for k in res:
        t,o = res[k]["tf"], res[k]["ol"]
        print(f"{k:>10} | {t['fid']:7.1f}{t['fglpips']:8.3f}{t['fgmse']:9.3f}{t['lapvar']:8.4f} | "
              f"{o['fid']:7.1f}±{o['fid_sd']:.0f}{o['fglpips']:8.3f}{o['fgmse']:9.3f}{o['lapvar']:8.4f}{o['tcoh']:8.3f}")
    print(f"(GT lap-var ~0.056; OL_tcoh GT ~0.085; baseline OL_fid is the blur reference)")

    # ---- Pareto gate (1-seed indicative; full CI needs the seed sweep) ----
    if all(k in res for k in ["l2","lpips","gan"]):
        g,l,b = res["gan"]["ol"], res["lpips"]["ol"], res["l2"]["ol"]
        verdict = {
          "gan_beats_lpips_fid": g["fid"] < l["fid"] - max(g["fid_sd"]+ (res["lpips"]["ol"]["fid_sd"]), 5),
          "gan_fgmse_not_worse_than_lpips": g["fgmse"] <= l["fgmse"] + 0.02,
          "gan_beats_L2arm_fglpips": g["fglpips"] < b["fglpips"],
          "gan_coherent_vs_L2arm": g["tcoh"] <= b["tcoh"] + 0.02,
          "any_arm_beats_blur_baseline_fid": min(g["fid"],l["fid"]) < res["baseline"]["ol"]["fid"],
        }
        verdict["VERDICT"] = ("GAN PROMISING -> add seeds" if verdict["gan_beats_lpips_fid"] and verdict["gan_fgmse_not_worse_than_lpips"]
                              else "NO clean GAN win at 1 seed -> inspect strip; consider diffusion head (#2)")
        res["gate"] = verdict
        print("\n=== GATE (1-seed indicative) ==="); [print(f"  {k}: {v}") for k,v in verdict.items()]

    # ---- combined strip: GT / baseline / l2 / lpips / gan (open-loop) ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rows = [("GT", None)] + [("base", m.decoder)] + [(arm, Gs[arm]) for arm in ["l2","lpips","gan"] if arm in Gs]
        cols=[0,4,7,24,49]; clip=2
        preds={}
        with torch.no_grad():
            for name,dec in rows:
                if dec is None: continue
                orig=m.decoder; m.decoder=dec; torch.manual_seed(0)
                preds[name]=m.open_loop(x[:clip+1],CTX,sample=True)[:,:HOR].cpu().numpy(); m.decoder=orig
        fig,ax=plt.subplots(len(rows),len(cols),figsize=(len(cols)*1.1,len(rows)*1.1))
        gtn=x[:clip+1,CTX:].cpu().numpy()
        for r,(name,dec) in enumerate(rows):
            arr = gtn if dec is None else preds[name]
            for j,t in enumerate(cols):
                ax[r,j].imshow(np.clip(arr[clip,t,0],0,1),cmap="gray",vmin=0,vmax=1); ax[r,j].set_xticks([]);ax[r,j].set_yticks([])
                if j==0: ax[r,j].set_ylabel(name,fontsize=8)
                if r==0: ax[r,j].set_title(f"+{t+1}",fontsize=8)
        fig.suptitle("Step-1 open-loop: GT / baseline / l2 / lpips / gan",fontsize=9); fig.tight_layout()
        fig.savefig(LOG/"s1_compare_strip.png",dpi=95); plt.close(fig); print("wrote", LOG/"s1_compare_strip.png")
    except Exception as e: print("strip err", e)
    (LOG/f"s1_gate_s{a.seed}.json").write_text(json.dumps(res, indent=2)); print("wrote gate json")

if __name__ == "__main__":
    main()
