# V1 results — VQ-free hierarchical single-step cascade (gate scorecard)

> **⚠️ Provenance note (2026-06-19 review):** the numbers in this file are from the original v1 checkpoint, which was
> **overwritten** by the later `base_64` P0/P1 retrain (`_save` always writes `logs/oneshot/last.pt`). Re-running the **on-disk**
> checkpoint reproduces **recon PSNR 30.23 / SSIM 0.963** and **single-step lift +4.47 dB / +0.072 SSIM** (cascade lift bot +0.068,
> mid +0.032, top +0.000) — *not* the 32.34 dB / +5.10 dB / +0.126 SSIM below (the SSIM lift in particular is ~2× overstated).
> Trust the on-disk numbers; retain (don't overwrite) any checkpoint a headline number is quoted from. Also: the cascade "lift" is an
> **eval-time input ablation on one always-conditioned model**, single-seed — a suggestive diagnostic, not a trained-A/B causal result.

> 2026-06-19. Built per [PLAN_VQFREE_HIER_PREDICTOR.md](PLAN_VQFREE_HIER_PREDICTOR.md);
> architecture in [ARCHITECTURE.md](ARCHITECTURE.md); story in [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
> Data: rotating 3D polycubes, **64×64**, T=9, Z-axis rotation, **true cross-split** (train/val rendered
> independently: 600 / 200 clips). Topology BOT 16²@T9 / MID 4²@T5 / TOP 1²@T3 (causal temporal
> downsample 9→5→3). Training: one-shot, detached predictive target, bf16, 3000 steps. 27/27 tests green.

## Gate scorecard

| Gate | Pass condition | Result (64×64, held-out) | Status |
|---|---|---|---|
| **unit** (27 tests) | shapes, causality, 9→5→3→9 round-trip, RoPE, **mask leak-free**, conditioning-down-only | 27/27 green | ✅ |
| **G-recon** | held-out PSNR floor; no dimensional collapse | **PSNR 32.34 dB, SSIM 0.975**; effective rank bot 15.9/16, mid 7.9/8, top 6.9/8 | ✅ |
| **G0** (predictability) | MID independent-stage `ve_changed ≥ 0.05` | **MID 0.734**, bot 0.835, top 0.789 — at the downsampled stride | ✅ |
| **G-stable** | one-shot converges, no collapse/NaN, beats persistence | recon 32.3 dB; predictive VE high; eranks near full | ✅ |
| **G-cascade** | cross-attn lift > independent; no future leak | lift **bot +0.061, mid +0.022** (top +0.000, no parent); leak-free mask test green | ✅ |
| **anti-blob** (pixel) | single-step predicted frame beats persistence | **+5.10 dB** PSNR, **+0.126** SSIM; strip tracks rotation, not a static blob | ✅ |

## Headline numbers (64×64, held-out cross-split, one-shot 3000 steps)

```
reconstruction:   PSNR 32.34 dB   SSIM 0.975
single-step pred: PSNR 18.03 dB (persistence 12.93)  lift +5.10 dB
                  SSIM 0.798     (persistence 0.672)  lift +0.126

stage  ve_cascade  ve_indep   lift    erank   (predictive VE on changed cells)
bot      0.896      0.835    +0.061   15.9/16
mid      0.756      0.734    +0.022    7.9/8
top      0.789      0.789    +0.000    6.9/8
```

(An earlier 32×32 600-step smoke gave consistent results — recon PSNR 25, MID G0 ve_changed 0.837 on a
frozen encoder, cascade lift bot +0.141 / mid +0.076, pixel lift +3.77 dB — confirming the design is
resolution-robust.)

## What this establishes

- **The temporal-downsampling worry is empirically falsified** at stride 9/5/3: the MID latent (T=5) is
  highly predictable held-out (independent VE 0.73, bot 0.84) at 64×64.
- **The hierarchy pillar works**: proper RF-overlap **cross-attention** conditioning gives a positive
  lift (bot +0.061, mid +0.022) over independent stages — vs the prior degenerate upsample-add's +0.013.
- **No collapse**: VICReg variance floor + detached target keep all stages near full effective rank;
  recon is sharp (SSIM 0.975); loss does not collapse to zero.
- **No blob**: single-step predicted frames beat persistence by +5.10 dB / +0.126 SSIM and visibly track
  rotation.

## Honest caveats / next

- Predicted frames are **blurry on fine detail** — the expected L2 mean-regression softness (mild under
  near-unimodal Z-rotation). Levers: longer/larger run; the committed **GIVT-GMM distributional head**
  before SO(3); curriculum/λ tuning.
- **Cross-attn lift is modest at 64×64** (bot +0.061, mid +0.022) and should be **multi-seeded** to
  confirm it is beyond noise; BOT benefits most (real spatial alignment), TOP→MID least.
- **BOT attention is the compute bottleneck** (2304 tokens; the run was ~1.8 it/s with per-500-step val) —
  windowed BOT attention is the planned efficiency lever.
- **AR rollout drift** is the deferred v1.5 frontier (single-step only here); noised-conditioning +
  rollout-gap logging are the next build.

All metrics + a GT/Recon/Pred image table are logged to **wandb** (project `hvp-vqfree`, run `v1-64x64`).
