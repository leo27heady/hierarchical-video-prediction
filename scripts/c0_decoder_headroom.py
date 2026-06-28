"""C0 — oracle-headroom probe: can a better-trained decoder on DLH's FROZEN latents render sharper?

Extracts the teacher-forced bottom code c (the exact tensor that feeds img_dec) from the frozen
8k DLH baseline, then trains TWO fresh decoders on (c -> frame): an L2 control (mimics the current
Gaussian objective) and an L1+LPIPS perceptual decoder. Compares both to the CURRENT trained decoder.

Decision (plan §C0): if the perceptual decoder gains > 0.05 LPIPS OR > 2 dB PSNR over the current
decoder at matched identity, there is headroom -> Phase C (perceptual renderer on frozen stochastic
code) is worth building. If not, the latents lack the information and M2 is unreachable cheaply.

Usage: venv/Scripts/python.exe scripts/c0_decoder_headroom.py
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

ROOT = Path(__file__).resolve().parents[1]
CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CW))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_dlh.modules import ConvImageDecoder
import lpips

NCLIP, SEQ = 40, 86          # frames per clip used for codes
EVAL_CLIPS = 8               # held-out clips
STEPS, BS = 1200, 64

# ---- load frozen DLH ----
sd = torch.load(str(ROOT / "logs_dlh" / "baseline_iter8000.pt"), map_location=dev)
cfg = DLHConfig(**sd["cfg"]); model = DLH(cfg).to(dev); model.load_state_dict(sd["model"]); model.eval()

test = np.load(str(CW / "data_mmnist" / "test.npy"))[:NCLIP, :SEQ]
x = torch.from_numpy(test.astype(np.float32) / 255.0).unsqueeze(2).to(dev)   # (NCLIP,SEQ,1,64,64)

# ---- capture the decoder-input code c via a forward hook on img_dec ----
codes = []
real_dec = model.img_dec
class Capture(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, z): codes.append(z.detach()); return self.inner(z)
model.img_dec = Capture(real_dec)

@torch.no_grad()
def extract():
    cfgm = model.cfg; B, T = x.shape[:2]
    feats = model._bottom_up(x); state = model._init_state(B, x.device)
    torch.randn_like_real = torch.randn_like
    frames = []
    for t in range(T):
        codes.clear() if False else None
        x_t = [feats[n][:, t] for n in range(cfgm.levels)]
        _, state, _ = model._step(state, x_t, mode="posterior")   # appends one code per step
        frames.append(x[:, t])
    return torch.stack(frames, 1)                                 # (B,T,1,64,64)

frames = extract()
model.img_dec = real_dec                                          # restore
C = torch.stack(codes, 1)                                         # (B,T,feat) teacher-forced codes
print(f"codes {tuple(C.shape)}  frames {tuple(frames.shape)}  feat_dim {C.shape[-1]}", flush=True)

# train/eval split by clip
tr = slice(0, NCLIP - EVAL_CLIPS); ev = slice(NCLIP - EVAL_CLIPS, NCLIP)
Ctr = C[tr].reshape(-1, C.shape[-1]); Ftr = frames[tr].reshape(-1, 1, 64, 64)
Cev = C[ev].reshape(-1, C.shape[-1]); Fev = frames[ev].reshape(-1, 1, 64, 64)
print(f"train pairs {Ctr.shape[0]}  eval pairs {Cev.shape[0]}", flush=True)

_lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
def lap_var(img): return F.conv2d(img, _lap, padding=1).var(dim=(1,2,3)).mean().item()
def fgmse(pred, tgt):
    fg = (tgt > 0.1).float(); return (((pred-tgt)**2*fg).sum()/fg.sum().clamp(min=1)).item()
def psnr(pred, tgt):
    mse = ((pred-tgt)**2).mean().item(); return 100.0 if mse == 0 else 10*np.log10(1.0/mse)
_lp_eval = lpips.LPIPS(net='alex').to(dev).eval()
@torch.no_grad()
def lpips_full(pred, tgt):
    P = pred.repeat(1,3,1,1)*2-1; G = tgt.repeat(1,3,1,1)*2-1
    return torch.cat([_lp_eval(P[i:i+256], G[i:i+256]).flatten() for i in range(0,P.shape[0],256)]).mean().item()

@torch.no_grad()
def report(name, dec):
    outs = torch.cat([dec(Cev[i:i+256]).clamp(0,1) for i in range(0,Cev.shape[0],256)], 0)
    return {"name": name, "lap_var": lap_var(outs), "fg_mse": fgmse(outs, Fev),
            "psnr": psnr(outs, Fev), "lpips": lpips_full(outs, Fev)}

print(f"\nGT eval lap_var {lap_var(Fev):.4f}")
rows = [report("current(L2,tanh)", real_dec)]   # the model's actual decoder

def train_decoder(loss_kind):
    dec = ConvImageDecoder(C.shape[-1], 1, tuple(cfg.conv_channels), 64, out_activation='sigmoid').to(dev)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-4)
    lp = lpips.LPIPS(net='alex').to(dev) if loss_kind == 'l1lpips' else None
    n = Ctr.shape[0]
    for step in range(STEPS):
        idx = torch.randint(0, n, (BS,), device=dev)
        z = Ctr[idx]; tgt = Ftr[idx]
        out = dec(z).clamp(1e-6, 1-1e-6) if False else dec(z)
        if loss_kind == 'l2':
            loss = F.mse_loss(out, tgt)
        else:
            l1 = F.l1_loss(out, tgt)
            pl = lp(out.repeat(1,3,1,1)*2-1, tgt.repeat(1,3,1,1)*2-1).mean()
            loss = l1 + 0.5 * pl
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return dec

print("training fresh L2 decoder on frozen codes ...", flush=True)
rows.append(report("fresh(L2,sigmoid)", train_decoder('l2')))
print("training fresh L1+LPIPS decoder on frozen codes ...", flush=True)
dec_pl = train_decoder('l1lpips')
rows.append(report("fresh(L1+LPIPS)", dec_pl))

# ---- visual: GT / current L2 decoder / perceptual decoder on the SAME frozen codes ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    with torch.no_grad():
        sel = torch.arange(0, Cev.shape[0], max(1, Cev.shape[0] // 8))[:8]
        gtv = Fev[sel].cpu().numpy(); cur = real_dec(Cev[sel]).clamp(0,1).cpu().numpy()
        per = dec_pl(Cev[sel]).clamp(0,1).cpu().numpy()
    fig, ax = plt.subplots(3, len(sel), figsize=(len(sel)*1.2, 3.8))
    for j in range(len(sel)):
        for r,(lab,arr) in enumerate([("GT",gtv),("current L2",cur),("L1+LPIPS",per)]):
            ax[r,j].imshow(arr[j,0], cmap="gray", vmin=0, vmax=1); ax[r,j].set_xticks([]); ax[r,j].set_yticks([])
            if j==0: ax[r,j].set_ylabel(lab, fontsize=9)
    fig.suptitle("C0 headroom: same frozen DLH codes, different decoder objective", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(str(ROOT/"logs_dlh"/"c0_headroom_strip.png"), dpi=90)
    print("wrote", ROOT/"logs_dlh"/"c0_headroom_strip.png")
except Exception as e:
    print("strip err", e)

print(f"\n{'decoder':>20}{'lap_var':>10}{'fg_mse':>10}{'psnr':>8}{'lpips':>8}")
for r in rows:
    print(f"{r['name']:>20}{r['lap_var']:>10.4f}{r['fg_mse']:>10.4f}{r['psnr']:>8.2f}{r['lpips']:>8.4f}")

cur = rows[0]; best = rows[2]
d_lpips = cur['lpips'] - best['lpips']; d_psnr = best['psnr'] - cur['psnr']; d_lap = best['lap_var'] - cur['lap_var']
print(f"\nHEADROOM (L1+LPIPS vs current): ΔLPIPS {d_lpips:+.4f}  ΔPSNR {d_psnr:+.2f} dB  Δlap_var {d_lap:+.4f}")
verdict = "HEADROOM EXISTS -> build Phase C" if (d_lpips > 0.05 or d_psnr > 2.0) else "NO headroom -> latents are the limit; M2 unreachable on frozen 8k latents"
print("VERDICT:", verdict)
import json
(ROOT / "logs_dlh" / "c0_headroom.json").write_text(json.dumps(
    {"rows": rows, "headroom": {"d_lpips": d_lpips, "d_psnr": d_psnr, "d_lap": d_lap, "verdict": verdict},
     "note": "teacher-forced frozen codes; sigmoid fresh decoders; STEPS=%d" % STEPS}, indent=2))
print("wrote", ROOT / "logs_dlh" / "c0_headroom.json")
