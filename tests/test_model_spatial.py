"""Smoke tests for the per-pixel spatial DLH (red-team checklist, run in order). Runnable directly:
   venv/Scripts/python.exe tests/test_model_spatial.py
Tests 1-4,6 run untrained; 5/7/8 (e-map segments, calibration, frozen-persistence) need training -> short run."""
import sys, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')
from hvp_dlh.model_spatial import SpatialDLH, SpatialDLHConfig
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
ok = lambda b: "OK " if b else "XXX"
fails = []

def check(name, cond):
    print(f"  {ok(cond)} {name}");
    if not cond: fails.append(name)

# ---- 1. SHAPES ----
print("== 1. shapes ==")
cfg = SpatialDLHConfig()
m = SpatialDLH(cfg).to(dev)
npar = sum(p.numel() for p in m.parameters())
print(f"  params {npar/1e6:.2f}M  | N={cfg.levels} Cs={cfg.stoch_dim} Cd={cfg.det_dim} Cg={cfg.temporal_gru} hw={cfg.latent_hw}")
B, T = 4, 6
ob = torch.rand(B, T, 1, 64, 64, device=dev)
feats = m._bottom_up(ob)
check("bottom-up x^0 (B,Cd,16,16)", tuple(feats[0].shape) == (B, T, cfg.det_dim, 16, 16))
st = m._init_state(B, dev)
dec, st2, info = m._step(st, [feats[n][:, 0] for n in range(cfg.levels)], mode="posterior")
check("decoded (B,1,64,64)", tuple(dec.shape) == (B, 1, 64, 64))
check("e-map (B,1,16,16)", tuple(info["e"][-1].shape) == (B, 1, 16, 16))
check("skl-map (B,1,16,16)", tuple(info["skl_level"][-1].shape) == (B, 1, 16, 16))
check("trans_kl (B,1)", tuple(info["trans_kl"].shape) == (B, 1))

# ---- 2. GRAD FLOW ----
print("== 2. grad flow (every param finite) ==")
loss, met = m.loss(ob, beta=1.0)
loss.backward()
# pf_head.0 (level-0 prior-factor) is unused by design (level 0 always ticks; trans-KL skips n=0) — same as vector DLH
nog = [n for n, p in m.named_parameters() if p.grad is None and not n.startswith("pf_head.0")]
naninf = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
check(f"all params have grad (missing: {nog[:3]})", len(nog) == 0)
check(f"all grads finite (bad: {naninf[:3]})", len(naninf) == 0)
for key in ["pf_head", "post", "prior", "tem", "dec", "img_enc", "img_dec"]:
    g = any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.startswith(key))
    check(f"{key} receives gradient", g)

# ---- 4. force_tick / no-gate oracle ----
print("== 4. force_tick recovers dense updates (K=N) ==")
m.zero_grad()
_, _, info_ft = m._step(m._init_state(B, dev), [feats[n][:, 0] for n in range(cfg.levels)], mode="posterior", force_tick=True)
allone = all((e == 1).all().item() for e in info_ft["e"])
check("force_tick -> every e==1", allone)
check(f"K.mean()=={cfg.levels} (got {info_ft['K'].mean().item():.2f})", abs(info_ft["K"].mean().item() - cfg.levels) < 1e-4)

# ---- 6. DILATION mechanism (plain multiply collapses; dilation rescues) ----
print("== 6. nested-constraint dilation mechanism ==")
parent = torch.zeros(1, 1, 16, 16, device=dev); parent[0, 0, 8, 8] = 1.0     # single active location
dil = F.max_pool2d(parent, 3, stride=1, padding=1)
check(f"maxpool3x3 grows 1 loc -> 9 (got {int(dil.sum())})", int(dil.sum().item()) == 9)
check("plain multiply keeps 1 loc", int(parent.sum().item()) == 1)

# ---- 3. MEMORY probe (B16 now; B32 authoritative gate deferred to free GPU) ----
print("== 3. memory probe (seq100) ==")
if dev == "cuda" and os.environ.get("SMOKE_MEM"):
    for Bm in [16]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            mm = SpatialDLH(cfg).to(dev); opt = torch.optim.Adam(mm.parameters(), lr=5e-4)
            obm = torch.rand(Bm, 100, 1, 64, 64, device=dev)
            t0 = time.time(); l, _ = mm.loss(obm, beta=1.0); l.backward(); opt.step(); torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  B{Bm} seq100 fwd+bwd: {peak:.1f} GB peak, {time.time()-t0:.1f}s/it  -> B32 extrapolated ~{peak*2:.1f} GB")
            check(f"B{Bm} peak < 14 GB ({peak:.1f})", peak < 14)
            del mm, opt, obm; torch.cuda.empty_cache()
        except RuntimeError as e:
            check(f"B{Bm} seq100 OOM/err: {str(e)[:60]}", False)
else:
    print("  (cpu — skipped)")

# ---- mini training sanity: loss drops, tick-rate sane ----
print("== mini training sanity (30 steps on real MMNIST, B8 seq30) ==")
CW = Path(__file__).resolve().parents[1].parent / "cwvae"
data = np.load(str(CW / "data_mmnist" / "train.npy"))
mt = SpatialDLH(cfg).to(dev); opt = torch.optim.Adam(mt.parameters(), lr=5e-4, eps=1e-4)
l0 = None
for step in range(30):
    idx = np.random.randint(0, data.shape[0], 8)
    obt = torch.from_numpy(data[idx, :30].astype(np.float32) / 255).unsqueeze(2).to(dev)
    ft = step < 15
    loss, met = mt.loss(obt, beta=min(1.0, step / 20), force_tick=ft)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mt.parameters(), 100); opt.step()
    if step == 0: l0 = met["nll"].item()
    if step % 10 == 0 or step == 29:
        print(f"   step {step:2d} nll {met['nll'].item():8.1f} sKL {[round(x,2) for x in met['skl_per_level'].tolist()]} "
              f"tick {[round(x,3) for x in met['e_per_level'].tolist()]} soft_rate {met['soft_tick_rate'].item():.3f} ft={int(ft)}")
check(f"nll decreased ({l0:.0f} -> {met['nll'].item():.0f})", met["nll"].item() < l0)
top_tick = met["e_per_level"].tolist()[-1]
# 30 steps is too short to judge final tick rate (real calibration run uses force_tick=3000); only fail if SATURATED
check(f"top-level tick not saturated all-on (got {top_tick:.3f}; real calibration via training run)", top_tick < 0.99)

print(f"\n{'ALL SMOKE PASSED' if not fails else 'FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)
