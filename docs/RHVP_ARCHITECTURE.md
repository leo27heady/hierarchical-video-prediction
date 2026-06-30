# RHVP — Reversible Hierarchical Video Predictor (the dissertation architecture)

> The converged architecture unifying the three assets — hierarchical prediction (DLH/CW-VAE), BLT's *hierarchy*
> (cheap codec + deep multi-resolution latent; **dynamic/uncertainty clock ABANDONED**), and reversibility
> (TempVerseFormer, time-axis additive coupling) — into one buildable model + paper. Branch `explore/blt-decoder-gating`.

## One-line thesis
The first **time-reversible recurrence for a hierarchical *stochastic* (ELBO) video model** — resolving the MacKay
no-forgetting collision via *store-the-noise* — with **provably exact reverse-replay gradients**, enabling correct
long-horizon rollout training (T=200+) at constant memory.

## Architecture
- **Cheap pixel codec** (~0.8M params, gradient-checkpointed → constant in T). Sharpness is OUT of scope (decoupled
  adversarial renderer applied post-hoc to frozen codes).
- **Deep reversible multi-resolution latent core** (~20.4M params — the capacity, the 26:1 inversion of the conv-DLH
  balance). 3 FIXED resolutions, **no clock** (no `e`-gate / force_tick / transition_kl):
  | level | state s^n (B,C,h,w) | stoch z (E) | coupling depth K | role |
  |---|---|---|---|---|
  | L0 | (B,192,16,16) | 32 | 4 | fast/local |
  | L1 | (B,256,8,8) | 32 | 4 | mid |
  | L2 | (B,384,4,4) | 32 | 3 | slow/global |
- **Load-bearing split:** WIDE deterministic carry (192/256/384) holds memory + makes reversibility dominant; NARROW
  stochastic z (E=32) preserves the verified anti-drift `stoch_dim≈8`. **Drift is governed by stoch-dim, not
  det-dim** — this severs "capacity that earns the headline" from "capacity that drifts."
- **Reversible recurrence (per level, along time):** additive coupling `x1+=F(x2,g); x2+=G(x1,g)`, exact inverse;
  F/G = Conv3×3→GroupNorm→SiLU→Conv3×3 with **spectral_norm + residual gate ≤0.9** (Lipschitz control — fp32 reverse
  drift hit ~1e3 over T=64 without it) + **anchor snapshot every 32 steps** to bound drift to T=200+.
- **Stochastic, reversibly (store-the-noise):** `z=mean+std·eps`; eps enters only as *conditioning* (never added to the
  carried state) so the coupling stays bijective; eps is the only stored quantity (~0.34 MB/step, or ~0 via a per-step
  seed). This is the novel bit: no prior reversible net (RevNet/RevViT/reversible-RNN/TempVerseFormer) carries a
  sampled z inside an ELBO.
- **Loss:** the existing hierarchical ELBO **minus the clock** (recon NLL + per-level KL, σ=1.0 fixed, β-warmup). For
  long rollout: **distribution-matching** (adversarial renderer on a sparse decode subset), NOT pixel-MSE (E1: MSE
  rollout fades to the mean).

## Headline figure (3-panel)
- **A — Correctness (the non-nullable contribution):** reverse-replay gradient cosine vs full BPTT + fp32 drift, vs T,
  with/without spectral-norm+anchors. The provably-exact stochastic reversible gradient *cannot null*.
- **B — Memory:** peak GPU memory vs trained T for {BPTT (OOM wall), gradient checkpointing (the honest strong
  baseline), TBPTT (biased-grad, flagged), Reversible (constant)}.
- **C — Quality (scoped):** open-loop fg-MSE/FVD vs horizon, models trained at each method's max feasible T.

## M0(a) — first measurement (run, `hvp_rev/rhvp_m0_memory.py`)
- **Param inversion ACHIEVED:** core 20.7M : codec 0.8M (26:1) — the deep core dominates, as designed.
- **BPTT = 158 MB/step** (measured) → OOMs at long T. This is the real problem the reversible core solves.
- **Reversible = 0.42 MB/step** (eps + amortized anchors) — constant in T, ~380× below BPTT.
- **HONEST CAVEAT (the M0 go/no-go did its job):** the reversible-vs-**checkpointing** memory comparison is
  measurement-sensitive — my quick harness gives checkpoint anywhere from ~flat (final-state loss) to 91 MB/step
  (per-frame loss). Gradient checkpointing is a STRONG baseline that may match reversibility's constant memory. So the
  **memory headline is NOT a guaranteed win over checkpointing** — exactly why M0 nails it carefully first, and why the
  PRIMARY contribution is the **correctness theorem (Panel A)**, not the memory number. Reversibility's robust edge over
  checkpointing, if any, is **less recompute** (inverse < full forward recompute) — a compute/throughput edge to measure.

## Build plan (start tomorrow; each milestone a few days on the 4060 Ti)
- **M0 — gradient-equivalence + clean memory go/no-go (CHEAPEST, ~1–2 days, no training).** Build the conv
  `RevTemporalBlock`; (a) measure BPTT/checkpoint/reversible *carefully* (optimal checkpointing, per-frame loss) →
  PASS = reversible ≤ checkpointing AND stable as core scales; (b) gradient cosine vs BPTT at T=16–200, spectral-norm
  sweep for the Lipschitz bound keeping fp32 drift <1e-3, 32-step anchors. **This is Panels A+B and de-risks the thesis.**
- **M1 — overfit-a-batch smoke** (~1 day): the 11-deep core trains, no NaN.
- **M2 — teacher-forced ELBO at short T** (~3 days, plain BPTT): non-inferiority gate — wide-deep core ties the lean
  flat baseline on fg-MSE/fg-LPIPS (KTH).
- **M3 — hierarchy-necessity pre-check** (~1–2 days, GATES the campaign): a flat rev model must drift past T=50 while a
  2-rate model holds; per-level KLs must segment slow/fast. Never run the long campaign on an unverified-hierarchy
  dataset (the documented MMNIST mistake).
- **M4 — reversible long-rollout training, T=200+** (~1 week): custom `autograd.Function` (forward saves final state +
  eps + anchors; backward reconstructs via `inverse`). Distribution-matching objective. Panel C.
- **M5 — full matrix + ablations + the 3-panel figure.**

**Datasets:** KTH primary (in BOTH TempVerseFormer and DLH papers → directly comparable). Drifting-Texture-Sprites
synthetic only as a gated hierarchy testbed. MMNIST = sanity/null-honesty baseline only. Memory/correctness results are
dataset-independent → the headline survives even if the hierarchy nulls.

## Novelty (defensible)
Deterministic RevNets/RevViT (depth-axis), reversible RNNs (MacKay), and **TempVerseFormer** (your own — deterministic,
single-scale, time-axis) all carry a **deterministic** state. **None carries a sampled z inside an ELBO.** RHVP is the
first **provably-invertible stochastic-ELBO hierarchical recurrence**, resolving the no-forgetting collision via
store-the-noise + Lipschitz coupling, gradients proven exact vs BPTT. Dissertation arc: **TempVerseFormer (done,
deterministic single-scale) → RHVP (the stochastic + hierarchical extension where no-forgetting first actually bites).**
