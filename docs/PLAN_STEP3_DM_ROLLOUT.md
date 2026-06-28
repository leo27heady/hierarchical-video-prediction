# Step 3 — distribution-matching rollout objective + reversible core (plan, pre-red-team)

> The dissertation core: fix E1's fade (MSE rollout training → blur) with a **distribution-matching rollout
> objective**, decoded through the **Step-1 sharp renderer**, then scale the rollout window with the **reversible
> temporal core** (TempVerseFormer). Three contributions, sequenced so each is independently falsifiable.
> To be red-teamed before implementation.

## Why (recap of what's established)
- **E1**: MSE rollout training is distortion-optimal → **fades to the mean** (lap-var 0.0073→0.0017). The
  mechanism (grad reaches all levels, stable) is fine; the *objective* is wrong.
- **Step 1**: a **distribution-matching (adversarial)** objective renders **sharp, coherent** frames (FID 70 vs
  blur 101). Distribution-matching > perceptual > pixel.
- **Synthesis**: train the *rollout* with a distribution-matching objective so the dynamics learn to produce
  free-run rollouts whose rendered frames look like real video — not blurry means. This is **"Self-Forcing lite"**
  (a discriminator on free-run frames; **no diffusion teacher needed**, which we don't have).

## Staged sub-experiments (cheapest decisive first)

### 3a — adversarial rollout objective (the first decisive test; NO reversibility yet)
Build on E1's rollout harness (`hvp_rev/train_rollout.py`) + the Step-1 renderer.
- After `ctx` posterior steps, free-run `M` prior steps **with grad**; decode each via the **frozen Step-1
  warm+fg GAN renderer** (isolates the dynamics — train only the prior dynamics, renderer fixed).
- **Loss = a per-frame PatchGAN discriminator on the decoded free-run frames** (real MMNIST frames vs free-run
  predictions) — distribution-matching on the rollout — **+ a light fg-MSE anchor, floored** to protect identity.
- **Gate (vs E1 + frozen baseline):** does open-loop **sharpness/FID stay flat over horizon** (no fade) at
  fg-MSE within the floor? **Decisive question: distribution-matching rollout objective > MSE rollout (E1)?**
- Controls: E1 (MSE rollout), frozen baseline (no rollout training). ≥3 training seeds; gate on FID/fg-LPIPS
  trend vs horizon + the fg-MSE floor (the DLH long-horizon margin).

### 3b — temporal-coherence discriminator (only if 3a per-frame output flickers)
Augment the per-frame D with a short-clip **temporal discriminator** (3D conv over K-frame windows) for temporal
coherence; report temporal-LPIPS + (Step-3) FVD.

### 3c — reversible temporal core + window scaling (the TempVerseFormer integration)
- Replace/augment the GRU temporal core with the **reversible temporal block** (`reversible_temporal_proto.py`,
  spectral-normed coupling), enabling **long-rollout BPTT at constant memory** (store-the-noise +
  reversal-with-forgetting for the clock — proto-demonstrated).
- **Scale M** (rollout window) using the reversible memory savings; produce the **memory-vs-M curve** (the only
  place reversibility's provable benefit is shown).
- **Gate:** does a longer-window distribution-matching rollout improve long-horizon sharpness/FID? **Pre-register
  the null** (E1's hierarchy was null on MMNIST; be ready).

## Honest framing (dissertation)
- **Claim strongly:** (1) the sharp renderer (Step 1, done); (2) a distribution-matching rollout objective that
  fixes the MSE-rollout fade (3a). Both stand without reversibility.
- **Claim modestly (measured):** reversibility = a larger BPTT window at fixed memory (3c), the *enabler*, not a
  stability claim.
- **Perception-distortion + identity:** floor fg-MSE; protect DLH's long-horizon identity edge; report the Pareto.
- **Validate dynamics claims on KTH / an event dataset** before the dissertation (MMNIST is hierarchy-null +
  background-dominated; the renderer result transfers, the rollout/clock claims need a real-events testbed).

## Risks
- **Three-way training** (dynamics + renderer + discriminator) is unstable → **3a freezes the renderer**, trains
  dynamics only, grad through the rollout dynamics.
- **Rollout-GAN is harder than per-frame GAN** (the discriminator judges moving free-run targets) → short M,
  frozen renderer, adaptive-λ + D warm-up + spectral-norm (the Step-1 stabilizers that worked).
- **Reversibility no-forgetting** (MacKay) → store-the-noise + reversal-with-forgetting + spectral coupling.
- **Compute**: rollout training + GAN + (later) reversibility — budget; start cheap (frozen renderer, short M).

## First experiment (cheapest decisive)
**3a-smoke:** E1 harness + frozen Step-1 renderer + per-frame PatchGAN rollout loss + fg-MSE floor, short M=8,
1 seed, ~2k steps. Pass = grad flows, D-acc in [0.5,0.9], rollout frames stay sharp (lap-var ~GT) over the
window, fg-MSE not exploding. Then the 3-arm (DM-rollout / E1-MSE-rollout / baseline) × 3-seed comparison.
