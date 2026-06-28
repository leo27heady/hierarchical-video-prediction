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

---
## v2 — red-team folded in (GO-WITH-CHANGES) + domain-adaptation probe result
**Probe (done, `s1fr_gan` vs `s1wfg_gan`):** training the renderer on FREE-RUN codes did NOT improve open-loop
(OL-FID 79.6 vs TF-trained 67.0; fg-LPIPS ~same). ⇒ the open-loop gap is **dynamics-limited, not a renderer
domain artifact**. Frozen TF-renderer for 3a is interpretable; the dynamics are the target. (tcoh of the
free-run-trained G = 0.085 = GT, a small coherence plus, but FID worse — not worth co-training in 3a.)

**Required changes (blockers):**
1. **Unconditional per-frame PatchGAN has no correspondence fixed point** — with the renderer frozen it teaches
   the dynamics "decode to *a* realistic digit," not "the *right* continuation," leaving fg-MSE as the only
   correspondence term (which E1 proved fades). **Promote a TEMPORAL/context-conditioned D ahead of 3a** (real =
   true K-frame continuation incl. context tail; fake = free-run). Do NOT run the unconditional per-frame D as primary.
2. **Pooled-FID gate is circular** (pools 50 frames → averages the fade away; gamed by sharpness in the incoherent
   free-run regime). **Primary gate = horizon-RESOLVED** (bin +1-10/+11-25/+26-50, gate on slope): temporal-LPIPS
   (track GT ~0.085, not →0/↑) + a correspondence/identity probe (fg-LPIPS@50, digit-identity persistence). FID
   strictly secondary relative ranking.
3. **Renderer freeze:** use the TF renderer (probe shows domain-adaptation doesn't help OL); keep a fully-frozen
   primary arm + a low-LR co-adapted-renderer arm only to bound the ceiling.
4. **adaptive-λ must reference the last DYNAMICS tensor (botB), not the frozen renderer's last layer.**
5. **fg-MSE anchor = hinge/deadband** active only above `baseline_fgMSE + k·σ` (this-backbone, k pre-registered);
   report the Pareto, gate on Pareto-dominance; add an anchor-only (no-GAN) control.
6. **Pre-register decision rule + null + arms:** DM-rollout beats E1-MSE-rollout beyond seed std AND within the
   fg-MSE Pareto band; add a **perceptual-rollout (L1+LPIPS, no D)** arm (the E1b question) + the co-adapted arm;
   high null-prior (E1, hierarchy) → **full dynamics claim conditional on KTH**.
7. **DEFER 3c (reversibility):** the CW-VAE clockwork sum-merge + multi-rate GRUs are **not a single invertible
   chain** → reversible reconstruction may be silently wrong. Scope any reversible block to one level's same-rate
   recurrence, keep merge/tiling out, **require a gradient-equivalence check vs true BPTT on the real backbone**,
   and use **gradient checkpointing** (correctness-safe, sub-linear-in-M memory) as the fallback. **Decouple the
   reversibility deliverable** (memory-vs-M curve + the P1–P4 no-forgetting figure) from the 3a quality verdict —
   those hold even with checkpointing. (Irrelevant at M=8 anyway: VRAM ~3 GB.)
8. **Budget on throughput** (~1.5–2.2 it/s with D+adaptive-λ → 3-arm×3-seed ≈ 10–11 GPU-h); spend spare VRAM on
   batch 48–64, not larger M.

**Smallest next step (after the probe):** build the **temporal/context-conditioned D** + horizon-resolved gate,
then a 3a-smoke (frozen renderer, M=8, 1 seed) checking the +26–50 coherence slope vs E1's fade. **Open
(dissertation-level, for the user):** is FVD (I3D) load-bearing vs temporal-LPIPS? Is KTH validation required
before claiming 3a/3b? Is reversibility an architectural win or a theorem-figure+memory-curve result here?
