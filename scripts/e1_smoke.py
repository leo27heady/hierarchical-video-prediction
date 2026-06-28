"""E1 SMOKE (red-team's smallest decisive first step) — prove the rollout-training mechanism BEFORE any sweep.

Corrected design (all red-team blockers folded in):
  - ctx_tf=36 (multiple of factor^2=36; ctx_tf=24 crashes via _stack([]) on the empty top slice).
  - Phase B calls hierarchical_unroll(use_obs=[False]*L, initial=carried) DIRECTLY (NOT open_loop, which is
    @torch.no_grad and would silently zero all rollout gradient -> a fake null).
  - DETACH the Phase-A->B state for run 1 (train only prior dynamics; bounded memory; clean isolation).
  - Free-run is prior-only (post==prior aliased) -> NO KL there; rollout loss = Gaussian-NLL on decoded frames only.
  - Per-frame normalized losses (ELBO_A / ctx_tf, rollout / Mb).

Asserts: (1) per-level prior-param grad-norm after one rollout backward, at M in {4,8,40}, to settle empirically
WHICH levels get gradient at which window (the crux: short window barely reaches the slow levels); (2) rollout
loss drops over 200 steps; (3) ELBO_A stays finite; (4) peak VRAM (expect ~<2GB -> no checkpointing needed);
(5) free-run prior std watch (latent-collapse tripwire).
"""
import sys, math, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; CW = ROOT.parent / "cwvae"
sys.path.insert(0, str(CW))
from cwvae_torch import CWVAE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)
CTX = 36

def load_model():
    m = CWVAE(levels=3, factor=6, stoch=20, deter=200, embed=200, dense_layers=3, enc_embed=1000, cm=1, dec_stddev=1.0).to(dev)
    m.load_state_dict(torch.load(str(CW/"logs_torch"/"cwvae"/"last.pt"), map_location=dev)["model"])
    return m

def gauss_nll(pred, tgt, var):                      # summed over pixels & frames, mean over batch
    B, T = pred.shape[:2]
    return (0.5*((tgt-pred)**2)/var + 0.5*math.log(2*math.pi*var)).reshape(B, T, -1).sum(-1).sum(1).mean()

def rollout_step(m, obs, ctx_tf, M, lam, detach=True, sample_roll=True):
    L, B = m.levels, obs.shape[0]; var = m.dec_stddev**2
    enc = m.encoder(obs)
    pre, post, cl = [], [], ctx_tf
    for lvl in range(L):
        pre.append(enc[lvl][:, :cl]); post.append(enc[lvl][:, cl:]); cl = cl // m.factor
    # Phase A: posterior teacher-forced ELBO
    botA, lastA, priA, posA = m.hierarchical_unroll(pre, use_obs=True, sample=True)
    Ta = pre[0].shape[1]
    decA = m.decoder(botA.reshape(B*Ta, -1)).reshape(B, Ta, 1, 64, 64)
    nllA = gauss_nll(decA, obs[:, :ctx_tf], var)
    klA = obs.new_zeros(())
    for i in range(L):
        q, p = posA[i], priA[i]
        klA = klA + (torch.log(p["stddev"]/q["stddev"]) + (q["stddev"]**2+(q["mean"]-p["mean"])**2)/(2*p["stddev"]**2) - 0.5).sum(-1).sum(1).mean()
    elboA = (nllA + klA) / ctx_tf
    # Phase B: free-run prior WITH grad, from carried (optionally detached) state
    initB = [{k: (v.detach() if detach else v) for k, v in st.items()} for st in lastA]
    botB, lastB, priB, posB = m.hierarchical_unroll(post, use_obs=[False]*L, initial=initB, sample=sample_roll)
    Mb = post[0].shape[1]
    decB = m.decoder(botB.reshape(B*Mb, -1)).reshape(B, Mb, 1, 64, 64)
    rollout = gauss_nll(decB, obs[:, ctx_tf:ctx_tf+Mb], var) / Mb
    loss = elboA + lam * rollout
    info = {"elboA": float(elboA), "rollout": float(rollout), "Mb": Mb,
            "post_lens": [p.shape[1] for p in post],
            "freerun_std": float(torch.stack([priB[i]["stddev"].mean() for i in range(L)]).mean())}
    return loss, info, decB

def prior_grad_norms(m):
    out = []
    for l in range(m.levels):
        c = m.cells[l]; g = 0.0
        for mod in [c.prior_h1, c.prior_h2, c.prior_mean, c.prior_std]:
            for p in mod.parameters():
                if p.grad is not None: g += float(p.grad.norm())
        out.append(g)
    return out

# ---------- data ----------
test = np.load(str(CW/"data_mmnist"/"test.npy"))
def batch(n, M, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, test.shape[0], n)
    t0 = rng.integers(0, test.shape[1]-(CTX+M), n)
    clips = np.stack([test[i, s:s+CTX+M] for i, s in zip(idx, t0)])
    return torch.from_numpy(clips.astype(np.float32)/255.0).unsqueeze(2).to(dev)

# ---------- (1) per-level gradient reach vs window M ----------
print("=== gradient reach per level (one backward each) — does grad reach the SLOW levels? ===")
print(f"{'M':>4}{'postlens(b/m/t)':>18}{'gradL0':>10}{'gradL1':>10}{'gradL2':>10}")
for M in [4, 8, 40]:
    m = load_model(); m.zero_grad(set_to_none=True)
    ob = batch(8, M)
    loss, info, _ = rollout_step(m, ob, CTX, M, lam=1.0, detach=True, sample_roll=True)
    loss.backward()
    g = prior_grad_norms(m)
    print(f"{M:>4}{str(info['post_lens']):>18}{g[0]:>10.3f}{g[1]:>10.3f}{g[2]:>10.3f}")
    del m; torch.cuda.empty_cache()

# ---------- (2) longer trainability check: does the rollout SE move? + VRAM + std watch ----------
print("\n=== M=8 trainability check (800 steps, B=16, lr 3e-4, warm-start) ===")
m = load_model(); opt = torch.optim.Adam(m.parameters(), lr=3e-4)
if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
t0 = time.time(); roll_hist = []; CONST = 0.5*math.log(2*math.pi)*4096  # Gaussian const part of NLL
for step in range(1, 801):
    ob = batch(16, 8, seed=step)
    loss, info, _ = rollout_step(m, ob, CTX, 8, lam=1.0, detach=True, sample_roll=True)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0); opt.step()
    roll_hist.append(info['rollout'])
    if step % 100 == 0 or step == 1:
        vram = torch.cuda.max_memory_reserved()/1e9 if dev.type=="cuda" else 0
        print(f"  step {step:3d} | elboA {info['elboA']:8.1f} | rollout-SE {info['rollout']-CONST:7.2f} | "
              f"freerun_std {info['freerun_std']:.4f} | gnorm {float(gn):.2f} | peakVRAM {vram:.2f}GB")
first = float(np.mean(roll_hist[:100])) - CONST; last = float(np.mean(roll_hist[-100:])) - CONST
print(f"  {800/(time.time()-t0):.2f} it/s | rollout SE (NLL-const): first100 {first:.2f} -> last100 {last:.2f} "
      f"(delta {last-first:+.2f}, {'TRAINABLE' if last < first-0.5 else 'flat/inconclusive'})")
print("\nSMOKE verdict: PASS if rollout drops, elboA finite, L0 grad>>0 always, and we can SEE which M reaches L1/L2.")
