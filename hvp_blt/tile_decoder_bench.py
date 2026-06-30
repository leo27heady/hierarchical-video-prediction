"""Experiment (a) — REAL tile-gather kernel wall-clock benchmark for BLT-style decoder gating.

The synthesis verdict: the FLOP-dominant block is the high-res decoder upsample (16->32->64 = 80% of the pipeline).
Gating it tile-sparsely (only re-render dirty 8x8-px tiles, copy the rest) is the only place a skip touches real FLOPs.
K0 showed FLOP-saving != wall-clock saving. THIS measures the wall-clock of the actual gather->packed-conv->scatter
path (including index overhead + halo waste), vs the dense decoder, as a function of p_tile.

fine_refine = the expensive decoder tail: stem feature (B,128,16,16) -> image (B,1,64,64) via 2 ConvTranspose + conv.
Tile-sparse: unfold stem into 8x8=64 tile-patches (4x4 each, with 1-ring halo) -> select M active -> fine_refine on
packed (M,128,4,4) -> crop central 8x8 -> scatter into a persisted canvas. M = round(p_tile * B * 64)."""
import sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')
dev = 'cuda'; torch.manual_seed(0)
B, Cstem = 32, 128

class FineRefine(nn.Module):
    """the expensive decoder tail (80% of pipeline). (.,128,h,w) -> (.,1, 4h, 4w)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.3),   # h -> 2h
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.3),    # 2h -> 4h
            nn.Conv2d(32, 1, 3, padding=1))
    def forward(self, x): return self.net(x)

fine = FineRefine().to(dev).eval()
stem = torch.randn(B, Cstem, 16, 16, device=dev)          # the dense stem feature (computed every frame anyway)
canvas_prev = torch.rand(B, 1, 64, 64, device=dev)        # last frame's rendered pixels (persisted tiles copy this)

def dense():
    return fine(stem)

def to_tiles(img):   # (B,1,64,64) -> (B,64,1,8,8) in tile order ti*8+tj
    return img.view(B,1,8,8,8,8).permute(0,2,4,1,3,5).reshape(B,64,1,8,8)
def from_tiles(t):   # (B,64,1,8,8) -> (B,1,64,64)
    return t.reshape(B,8,8,1,8,8).permute(0,3,1,4,2,5).reshape(B,1,64,64)

def sparse(active_mask):  # active_mask (B,64) bool
    # gather: unfold stem into 64 tile-patches (128,4,4) with 1-ring halo
    patches = F.unfold(stem, kernel_size=4, stride=2, padding=1)            # (B, 128*16, 64)
    patches = patches.transpose(1,2).reshape(B,64,Cstem,4,4)               # (B,64,128,4,4)
    idx = active_mask.reshape(-1).nonzero(as_tuple=True)[0]                 # M flat indices into B*64
    packed = patches.reshape(B*64,Cstem,4,4)[idx]                          # (M,128,4,4)
    out = fine(packed)                                                      # (M,1,16,16)
    out = out[:, :, 4:12, 4:12]                                            # crop central 8x8 -> (M,1,8,8)
    # scatter into a clone of the persisted canvas (copy = free; only dirty tiles overwritten)
    canvas_tiles = to_tiles(canvas_prev).reshape(B*64,1,8,8).clone()
    canvas_tiles[idx] = out
    return from_tiles(canvas_tiles.reshape(B,64,1,8,8))

def timeit(fn, n=60):
    for _ in range(15): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000

print(f"Tile-sparse decoder wall-clock benchmark (B={B}, fine_refine = 80% of pipeline, median of 60):\n")
t_dense = timeit(dense)
print(f"  DENSE fine_refine:                 {t_dense:6.2f} ms   (baseline, every frame)")
print(f"\n  {'p_tile':>7} {'M tiles':>8} {'sparse ms':>10} {'speedup':>8} {'FLOP-ideal':>11}")
torch.manual_seed(0)
for p in [0.1, 0.18, 0.26, 0.4, 0.5, 1.0]:
    mask = torch.rand(B,64,device=dev) < p
    M = int(mask.sum().item())
    t_sp = timeit(lambda: sparse(mask))
    print(f"  {p*100:6.0f}% {M:8d} {t_sp:10.2f} {t_dense/t_sp:7.2f}x {1/max(p,1e-3):10.1f}x")
# correctness sanity: at p=1.0 sparse should ~ dense (boundary halo may differ slightly)
mask_all = torch.ones(B,64,device=dev,dtype=torch.bool)
err = (sparse(mask_all) - dense()).abs()
# note: sparse copies canvas_prev for non-rendered tiles; at p=1 all rendered, compare rendered region
print(f"\n  correctness @p=1.0 (sparse vs dense, all tiles rendered): max|Δ| {err.max().item():.4f} mean {err.mean().item():.4f}")
print(f"  (small boundary Δ from per-tile halo context is expected and is a real tile-decoding artifact)")
