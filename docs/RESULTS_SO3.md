# Results — SO(3) harder data + GMM distributional head

> 2026-06-19. The review showed every architecture lever was untestable on Z-only constant-velocity
> data because a 2D rotation-warp null is near-oracle there. This is the payoff of the two unlocks:
> harder SO(3) data (defeats the null) + the GMM distributional head (best-of-N). 2000-step runs.

## 1. The harder data defeats the trivial baseline

Single-step changed-region PSNR of the training-free **rotation-warp** null (the strong null on rigid
rotation):

| data | rotation-warp changed-region | vs persistence |
|---|---|---|
| Z-only (rigid in-plane) | **25.05 dB** (near-oracle) | +19 dB |
| **SO(3)** (multi-axis) | **9.35 dB** | +2.5 dB |

3D rotation's self-occlusion + depth ambiguity is not predictable by any 2D warp, so there is finally a
real gap for a learned model to fill.

## 2. L2 baseline on SO(3): competitive, but it DRIFTS (the binding problem)

`so3_64.yaml`, L2 head, 2000 steps → recon **28.0 dB / SSIM 0.939**, no collapse (erank 15.1/8.0/7.2).

Rollout motion gate (changed-region, mean over 24 horizons):

```
mean AR(pixel) − rotation:   −0.54 dB   (was −11.0 dB on Z-only)   ← now competitive
AR beats rotation at h=1..4;  drifts below the baselines by mid-horizon
changed-region drift:        8.70 (h0) → 6.67 (h23)               ← exposure-bias drift
```

The "sophisticated copy with rotation" critique is resolved — on genuine 3D motion the trivial null no
longer dominates. **What remains is autoregressive DRIFT**, now the empirically-confirmed binding
constraint (C1) — not the representation, not capacity.

## 3. GMM distributional head on SO(3): best-of-N works

`so3_64.yaml` + `head=gmm head_components=4`, `lambda_pred=0.02` (NLL scale ≫ MSE) → recon **27.9 dB**
(preserved), mid cascade lift **+0.050** (best of any run). The GMM's point estimate (mode) is slightly
blurrier than L2's, but its **sampling** is the payoff:

```
best-of-8 changed-region (SO(3) val_long, mean over h=0..11):
  GMM:  mode 8.10  →  best-of-8 8.66   gain +0.56 dB   (consistent +0.34..+0.77 every horizon)
  L2:   mode 8.14  →  best-of-8 8.14   gain +0.00 dB   (samples identical — sanity check)
```

GMM best-of-8 (8.66) beats the rotation null (~8.1) and persistence (~6.7) on moving pixels and stays
flat across horizons where the L2 mode drifted. Under genuine multimodality the mixture places mass on
the true future that the L2 mean averages away — the MAR/GIVT result, reproduced.

## 4. Takeaways

- Harder data is the unlock: it makes the predictive task real and the drift problem measurable.
- The distributional head pays off **only where multimodality exists** (SO(3), not Z-only) — vindicating
  the "L2-now, GMM-before-SO(3)" plan and the demotion of single-sample PSNR to a sanity floor.
- **Next = P2 anti-drift training** (scheduled sampling + noised-history / self-forcing), which also
  absorbs the untrained-last-position fix (#4). Drift is now the frontier, with a dataset that exposes it.

*(best-of-N uses oracle sample selection — it measures distribution coverage of the true future, the
standard distributional-quality metric, not a deployable single-shot predictor.)*
