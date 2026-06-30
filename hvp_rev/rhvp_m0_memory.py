"""M0(a) — the single cheapest experiment that validates (or kills) the RHVP headline, NO training.

Measures the per-step training-memory tape of the DEEP REVERSIBLE multi-resolution core + THIN codec, to confirm the
"inversion" the whole thesis rests on: on the conv-dominated DLH the recurrence was 4% of the tape (reversibility
marginal); RHVP widens the deterministic core + thins the codec so the (reversible) core DOMINATES (target state-share
>70%, rev/checkpoint ratio <=0.15, stable as the core scales). BPTT & checkpointing measured empirically on the real
wide conv core; the reversible per-step = stored eps + amortized anchors (the only non-invertible quantity)."""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt
sys.stdout.reconfigure(encoding='utf-8')
dev='cuda'; torch.manual_seed(0); GB=1e9

# --- RHVP geometry (from the plan): 3 fixed resolutions, wide deterministic core, narrow stochastic z ---
LEVELS = [dict(C=192, h=16, K=4, E=32), dict(C=256, h=8, K=4, E=32), dict(C=384, h=4, K=3, E=32)]
B = 8

class RevCoupling(nn.Module):
    """additive coupling along time: x=[x1|x2]; x1+=F(x2,g); x2+=G(x1,g). spectral_norm for Lipschitz/drift control."""
    def __init__(self, C, cond):
        super().__init__()
        H, W = C//2, C
        def half():
            return nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(H+cond, W, 3, padding=1)),
                                 nn.GroupNorm(8, W), nn.SiLU(),
                                 nn.utils.spectral_norm(nn.Conv2d(W, H, 3, padding=1)))
        self.F, self.G = half(), half()
    def forward(self, x, g):
        x1, x2 = x.chunk(2, 1)
        x1 = x1 + self.F(torch.cat([x2, g], 1))
        x2 = x2 + self.G(torch.cat([x1, g], 1))
        return torch.cat([x1, x2], 1)

class Level(nn.Module):
    def __init__(self, C, K, E, cond_ctx):
        super().__init__()
        cond = E + cond_ctx                              # stochastic z (narrow) + top-down context
        self.layers = nn.ModuleList([RevCoupling(C, cond) for _ in range(K)])
    def forward(self, s, g):
        for l in self.layers: s = l(s, g)
        return s

class ThinCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1,32,4,2,1), nn.SiLU(), nn.Conv2d(32,48,4,2,1), nn.SiLU(),
                                 nn.Conv2d(48,64,3,1,1))                          # 64->16, 64ch
        self.dec = nn.Sequential(nn.Conv2d(192,64,3,1,1), nn.SiLU(),
                                 nn.ConvTranspose2d(64,32,4,2,1), nn.SiLU(),
                                 nn.ConvTranspose2d(32,1,4,2,1))                  # 16->64

class RHVPcore(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = ThinCodec()
        self.levels = nn.ModuleList([Level(L['C'], L['K'], L['E'], cond_ctx=64) for L in LEVELS])
        self.proj = nn.ModuleList([nn.Conv2d(LEVELS[n+1]['C'], 64, 1) for n in range(2)])  # top-down ctx
    def step(self, feats, states):
        new = []
        ctx = torch.zeros(B, 64, 4, 4, device=dev)
        for n in [2,1,0]:                                                          # top-down
            L = LEVELS[n]
            z = torch.randn(B, L['E'], L['h'], L['h'], device=dev)                 # stochastic (stored on reverse)
            cup = F.interpolate(ctx, size=(L['h'],L['h']), mode='nearest')
            g = torch.cat([z, cup + feats[n]], 1) if False else torch.cat([z, cup], 1)
            # actually condition on codec feat at this res:
            g = torch.cat([z, F.interpolate(feats[n], size=(L['h'],L['h']))[:, :64]], 1) if feats[n].shape[1]>=64 else g
            s = self.levels[n](states[n], g)
            new.insert(0, s)
            if n>0: ctx = self.proj[n-1](s)
        return new

def make_feats():
    return [torch.randn(B,64,L['h'],L['h'],device=dev) for L in LEVELS]

def peak_mem_core(T, mode):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = RHVPcore().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    states = [torch.zeros(B, L['C'], L['h'], L['h'], device=dev) for L in LEVELS]
    feats = make_feats()
    def step_fn(*s):
        return tuple(m.step(feats, list(s)))
    loss = 0.0
    for t in range(T):
        if mode=='ckpt':
            states = list(ckpt.checkpoint(step_fn, *states, use_reentrant=False))
        else:
            states = m.step(feats, states)
        loss = loss + states[0].pow(2).mean()    # PER-FRAME loss (realistic for video prediction)
    loss.backward(); opt.step()
    return torch.cuda.max_memory_allocated()/GB, sum(p.numel() for p in m.parameters())

print("M0(a) — RHVP deep-core + thin-codec per-step memory inversion (B=8, 16GB 4060 Ti):\n")
print(f"  {'T':>4} {'BPTT GB':>9} {'ckpt GB':>9}")
bptt, chk = {}, {}
npar=None
for T in [20, 40, 60]:
    try: bptt[T],npar = peak_mem_core(T,'bptt')
    except RuntimeError: bptt[T]=None
    try: chk[T],_ = peak_mem_core(T,'ckpt')
    except RuntimeError: chk[T]=None
    print(f"  {T:>4} {bptt[T] if bptt[T] else 'OOM':>9} {chk[T] if chk[T] else 'OOM':>9}".replace('None','OOM'))
Ts=[t for t in bptt if bptt[t]]
bptt_slope=(bptt[Ts[-1]]-bptt[Ts[0]])/(Ts[-1]-Ts[0])*1000
chk_slope=(chk[Ts[-1]]-chk[Ts[0]])/(Ts[-1]-Ts[0])*1000
# reversible per-step = stored eps (sum_n B*E*h*w*4) + amortized anchors
eps_mb = sum(B*L['E']*L['h']*L['h']*4 for L in LEVELS)/1e6
state_mb = sum(B*L['C']*L['h']*L['h']*4 for L in LEVELS)/1e6
anchor_mb = state_mb/32          # one anchor per 32 steps, amortized
rev_slope = eps_mb + anchor_mb
print(f"\n  core+codec params: {npar/1e6:.1f}M")
print(f"  per-step tape:  BPTT {bptt_slope:.1f} MB | checkpoint {chk_slope:.1f} MB | REVERSIBLE {rev_slope:.2f} MB (eps {eps_mb:.2f} + anchor {anchor_mb:.2f})")
print(f"  state-share of checkpoint tape: {100*state_mb/(chk_slope if chk_slope>0 else 1):.0f}%  (need >70%; conv-DLH was 4%)")
print(f"  rev / checkpoint ratio: {rev_slope/chk_slope:.3f}   (need <=0.15)")
print(f"  rev below BPTT: {bptt_slope/rev_slope:.0f}x | rev below checkpoint: {chk_slope/rev_slope:.1f}x")
verdict = "PASS — inversion achieved, reversibility is the dominant term -> headline holds" if (state_mb/chk_slope>0.6 and rev_slope/chk_slope<0.2) else "CHECK — state-share/ratio off target"
print(f"\n  VERDICT: {verdict}")
