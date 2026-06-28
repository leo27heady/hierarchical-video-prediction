"""C0b — can a BALANCED objective capture the headroom (sharp AND coherent), or does perceptual
loss only buy sharpness by breaking structure? Sweeps loss = MSE + lambda*LPIPS on the frozen DLH
codes. If some lambda gets LPIPS well below the current decoder WITHOUT wrecking fg-MSE, Phase C is
a real win; if every sharp config tanks fg-MSE/structure, the headroom is not usefully capturable
with a plain perceptual decoder (-> need adversarial/diffusion, higher cost).
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CW))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
from hvp_dlh.config import DLHConfig
from hvp_dlh.model import DLH
from hvp_dlh.modules import ConvImageDecoder
import lpips

NCLIP, SEQ, EVAL_CLIPS, STEPS, BS = 40, 86, 8, 2500, 64
sd = torch.load(str(ROOT/"logs_dlh"/"baseline_iter8000.pt"), map_location=dev)
cfg = DLHConfig(**sd["cfg"]); model = DLH(cfg).to(dev); model.load_state_dict(sd["model"]); model.eval()
x = torch.from_numpy(np.load(str(CW/"data_mmnist"/"test.npy"))[:NCLIP,:SEQ].astype(np.float32)/255.0).unsqueeze(2).to(dev)

codes = []; real_dec = model.img_dec
class Cap(nn.Module):
    def __init__(s,i): super().__init__(); s.i=i
    def forward(s,z): codes.append(z.detach()); return s.i(z)
model.img_dec = Cap(real_dec)
with torch.no_grad():
    B,T = x.shape[:2]; feats = model._bottom_up(x); st = model._init_state(B, x.device)
    for t in range(T):
        xt=[feats[n][:,t] for n in range(cfg.levels)]; _, st, _ = model._step(st, xt, mode="posterior")
model.img_dec = real_dec
C = torch.stack(codes,1); F_ = x
tr=slice(0,NCLIP-EVAL_CLIPS); ev=slice(NCLIP-EVAL_CLIPS,NCLIP)
Ctr=C[tr].reshape(-1,C.shape[-1]); Ftr=F_[tr].reshape(-1,1,64,64)
Cev=C[ev].reshape(-1,C.shape[-1]); Fev=F_[ev].reshape(-1,1,64,64)

_lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=dev).view(1,1,3,3)
def lapv(i): return F.conv2d(i,_lap,padding=1).var(dim=(1,2,3)).mean().item()
def fgm(p,t): fg=(t>0.1).float(); return (((p-t)**2*fg).sum()/fg.sum().clamp(min=1)).item()
def ps(p,t): m=((p-t)**2).mean().item(); return 100.0 if m==0 else 10*np.log10(1/m)
_lpe=lpips.LPIPS(net='alex').to(dev).eval()
@torch.no_grad()
def lpf(p,t):
    P=p.repeat(1,3,1,1)*2-1; G=t.repeat(1,3,1,1)*2-1
    return torch.cat([_lpe(P[i:i+256],G[i:i+256]).flatten() for i in range(0,P.shape[0],256)]).mean().item()
@torch.no_grad()
def rep(dec):
    o=torch.cat([dec(Cev[i:i+256]).clamp(0,1) for i in range(0,Cev.shape[0],256)],0)
    return dict(lap_var=lapv(o), fg_mse=fgm(o,Fev), psnr=ps(o,Fev), lpips=lpf(o,Fev))

_lpl=lpips.LPIPS(net='alex').to(dev)
def train(lam):
    dec=ConvImageDecoder(C.shape[-1],1,tuple(cfg.conv_channels),64,out_activation='sigmoid').to(dev)
    opt=torch.optim.Adam(dec.parameters(),lr=3e-4); n=Ctr.shape[0]
    for s in range(STEPS):
        idx=torch.randint(0,n,(BS,),device=dev); o=dec(Ctr[idx]); t=Ftr[idx]
        loss=F.mse_loss(o,t)
        if lam>0: loss=loss+lam*_lpl(o.repeat(1,3,1,1)*2-1,t.repeat(1,3,1,1)*2-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return dec

cur=rep(real_dec); cur["name"]="current(L2,8k)"
print(f"GT lap_var {lapv(Fev):.4f}\n{'config':>18}{'lap_var':>9}{'fg_mse':>9}{'psnr':>7}{'lpips':>8}")
print(f"{cur['name']:>18}{cur['lap_var']:>9.4f}{cur['fg_mse']:>9.4f}{cur['psnr']:>7.2f}{cur['lpips']:>8.4f}")
rows=[cur]
for lam in [0.0, 0.05, 0.1, 0.2]:
    r=rep(train(lam)); r["name"]=f"MSE+{lam}*LPIPS"; rows.append(r)
    print(f"{r['name']:>18}{r['lap_var']:>9.4f}{r['fg_mse']:>9.4f}{r['psnr']:>7.2f}{r['lpips']:>8.4f}", flush=True)

# "usefully capturable" = some lambda beats current LPIPS by >0.03 with fg_mse <= 1.25x current
ok=[r for r in rows[1:] if r["lpips"] < cur["lpips"]-0.03 and r["fg_mse"] <= 1.25*cur["fg_mse"]]
print("\nVERDICT:", "CAPTURABLE (sharp+coherent reachable): "+", ".join(r["name"] for r in ok) if ok
      else "NOT cleanly capturable with plain MSE+LPIPS -> Phase C needs adversarial/diffusion (higher cost)")
import json
(ROOT/"logs_dlh"/"c0b_sweep.json").write_text(json.dumps({"gt_lap":lapv(Fev),"rows":rows,
    "capturable":[r["name"] for r in ok]}, indent=2))
print("wrote", ROOT/"logs_dlh"/"c0b_sweep.json")
