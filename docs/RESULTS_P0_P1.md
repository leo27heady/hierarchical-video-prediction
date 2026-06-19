# P0 + P1 results — eval/repro spine + autoregressive rollout & prediction horizon

> 2026-06-19. Implements P0/P1 of [PLAN_V2_ROADMAP.md](PLAN_V2_ROADMAP.md). Model: `base_64`
> (3000 steps, AdamW + warmup-cosine, wd 0.01). Rollout eval on the independently-rendered
> long-horizon set `val_long` (T=33, 120 clips): seed = first 9 frames, free-run 24 steps.

## Gate scorecard

| Gate | Pass condition | Result | Status |
|---|---|---|---|
| **G0-eval (metrics)** | changed-region/LPIPS/FVD compute correctly | persistence penalized on moving pixels; LPIPS ok; FVD(identical)≈0 | ✅ (5 tests) |
| **G0-eval (resume)** | kill + `--resume` continues at the saved step | resumed at step 60, loss continues (no reset) | ✅ (3 tests + e2e) |
| **G-AR-baseline** | rollout coherent to H ≥ 2× train horizon, drift curves logged, both modes | stable to H=24, both modes; full curve table + FVD + strip | ✅ (4 tests) |

## Prediction-horizon drift (held-out `val_long`, 24 steps, base_64)

```
  h |  AR-px  AR-lat     TF  persist | AR-px(chg) persist(chg)
  0 |  14.74  14.74  14.74   12.96   |   10.11      5.98
  1 |  13.73  12.47  14.73   11.01   |    8.94      6.11
  2 |  13.10  11.83  14.65   10.82   |    8.41      7.13
  4 |  13.16  10.61  14.67   12.12   |    8.35      8.32
  7 |  13.00   8.15  14.71   11.92   |    8.18      7.90
 11 |  13.04  11.23  14.72   12.13   |    8.38      8.07
 15 |  12.87  10.63  14.60   11.82   |    8.32      7.75
 23 |  12.89  11.12  14.66   12.08   |    8.35      7.72   (PSNR, dB)

TF→AR gap (mean over 24 horizons):     +1.61 dB     (bounded, does NOT grow)
AR(pixel) beats persistence until:     24/24 horizons
AR(pixel) − persistence (mean PSNR):   +1.14 dB
FVD: AR-pixel 186.8 · AR-latent 202.8     LPIPS AR-pixel 0.357
```

## What this establishes (the prediction-horizon question)

1. **The rollout is STABLE — it does not diverge or blob-collapse.** AR-pixel PSNR plateaus at
   ~13 dB from h=1 to h=23 (a nearly flat drift curve); the predicted shape stays coherent and
   tracks the rotation over all 24 steps (2.6× the training horizon). This directly counters the
   predecessors' documented long-horizon "baked blob" failure.
2. **The TF→AR gap is small and bounded (+1.61 dB) and does not grow with horizon** — exposure bias
   is present but mild here (the fixed 9-frame window + the decode→re-encode "manifold projection"
   each step keep it on-distribution).
3. **AR-pixel beats persistence at every horizon** (+1.14 dB mean; and on the *moving* pixels via the
   changed-region metric) — it is genuinely tracking rotation, not winning by averaging.
4. **Pixel-feedback ≫ latent-feedback.** The honest decode→re-encode mode stays coherent; the
   optimistic latent-feedback mode drifts off-manifold into checkerboard artifacts by mid-horizon
   (visible in the drift strip, and FVD 202.8 vs 186.8). **→ carry pixel-feedback forward.**

## Honest caveats

- **Blurry, not sharp.** AR-pixel tracks rotation coarsely but loses fine detail — the same L2
  mean-regression softness, now over the horizon. **P3's GMM/diffusion head is the committed fix.**
  PSNR plateaus ~1.6 dB below teacher-forced; closing that is P2 (anti-drift training) + P3.
- **Single seed.** These curves are one run; P4 multi-seeds the headline (and the cascade lift).
- **Persistence is a weak long-horizon baseline** (a frozen frame on a rotating shape); the
  changed-region metric is the harder comparison and AR still wins there at most horizons.
- `val_long` clips rotate further per clip than the T=9 training clips; absolute PSNRs are not
  directly comparable to the single-step training numbers.

## Next (per the roadmap)

- **P2 — anti-drift training**: scheduled sampling + noised-history conditioning to *close* the
  +1.61 dB TF→AR gap and push the drift curve toward teacher-forced.
- **P3 — distributional head**: GMM (then diffusion) wired through sampling + best-of-N to de-blur
  the rollout and make the per-horizon metrics honest under multimodality.
- **P4 — hierarchy-anchors-horizon**: cascade-on vs cascade-off *drift* A/B, multi-seed — the
  dissertation's money result.
