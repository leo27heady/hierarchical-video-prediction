"""E0 — measure the teacher-forced vs open-loop EXPOSURE GAP on the CW-VAE backbone (no training).

The rollout-training thesis only matters if free-running (open-loop) is much worse than the
teacher-forced upper bound at the SAME horizon. We measure fg-MSE and fg-LPIPS at +25/+50:
  - teacher-forced: posterior over the full sequence (model sees every real frame) -> reconstruction
  - open-loop: ctx posterior then free-run the prior for the horizon (model sees its own predictions)
Gap = open-loop - teacher-forced. Large gap => exposure bias dominates => rollout training justified.
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CTX, HOR, CLIPS = 36, 50, 64
x = torch.from_numpy(np.load(str(CW/"data_mmnist"/"test.npy"))[:CLIPS, :CTX+HOR].astype(np.float32)/255.0).unsqueeze(2).to(dev)
gt = x[:, CTX:CTX+HOR]; gt_np = gt.cpu().numpy(); fgm = gt_np > 0.1

m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
m.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"]); m.eval()

import lpips
_lp = lpips.LPIPS(net='alex', spatial=True).to(dev).eval()
@torch.no_grad()
def fg_lpips_at(pred, t):                                  # pred (B,HOR,1,64,64); frame index t (0-based)
    P = pred[:, t].repeat(1,3,1,1)*2-1; G = gt[:, t].repeat(1,3,1,1)*2-1
    dmap = F.interpolate(_lp(P, G), size=(64,64), mode='bilinear', align_corners=False)
    msk = torch.from_numpy(fgm[:, t].astype(np.float32)).to(dev)
    return ((dmap*msk).sum()/msk.sum().clamp(min=1)).item()
def fg_mse_at(pred, t):
    p = pred[:, t].cpu().numpy(); se = (p - gt_np[:, t])**2
    return float(np.mean([se[c][fgm[c,t]].mean() if fgm[c,t].any() else 0.0 for c in range(CLIPS)]))

with torch.no_grad():
    dec_tf, _, _ = m(x)                                    # teacher-forced reconstruction over full seq
    tf = dec_tf[:, CTX:CTX+HOR].clamp(0,1)
    torch.manual_seed(0)
    ol = m.open_loop(x, CTX, sample=True)[:, :HOR]         # open-loop free-run

res = {"config": dict(ctx=CTX, horizon=HOR, clips=CLIPS)}
print(f"{'frame':>8}{'TF fg-MSE':>12}{'OL fg-MSE':>12}{'gap':>8}{'TF fgLPIPS':>12}{'OL fgLPIPS':>12}{'gap':>8}")
for t in [24, 49]:                                         # +25, +50
    a = {"tf_fgmse": fg_mse_at(tf,t), "ol_fgmse": fg_mse_at(ol,t),
         "tf_fglpips": fg_lpips_at(tf,t), "ol_fglpips": fg_lpips_at(ol,t)}
    a["gap_fgmse"] = a["ol_fgmse"]-a["tf_fgmse"]; a["gap_fglpips"] = a["ol_fglpips"]-a["tf_fglpips"]
    res[f"+{t+1}"] = a
    print(f"{'+'+str(t+1):>8}{a['tf_fgmse']:>12.3f}{a['ol_fgmse']:>12.3f}{a['gap_fgmse']:>8.3f}"
          f"{a['tf_fglpips']:>12.3f}{a['ol_fglpips']:>12.3f}{a['gap_fglpips']:>8.3f}")

g50 = res["+50"]["gap_fglpips"]
res["verdict"] = "GAP REAL -> rollout training justified" if g50 > 0.05 else "small gap -> rollout-training thesis weak"
print("\nVERDICT:", res["verdict"], f"(fg-LPIPS gap @+50 = {g50:.3f})")
(ROOT/"logs_dlh"/"e0_exposure_gap.json").write_text(json.dumps(res, indent=2))
print("wrote", ROOT/"logs_dlh"/"e0_exposure_gap.json")
