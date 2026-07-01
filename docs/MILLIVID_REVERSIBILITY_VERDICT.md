# Verdict: "MilliVid + reversibility for eval-time AR stabilization" (option b)

3-agent review (attachment / mechanism / novelty+feasibility), 2026-07-01. All three converge: **do not build this as a
novel stabilization method.** Reversibility is set aside from the headline; the honest home is a correctness theorem.

## The four fatal findings

1. **Reversibility does NOT stabilize AR rollout — it's a category error** (mechanism agent, sourced).
   Invertibility ≠ stability. Additive coupling `s_{t+1}=s_t+F(s_t)` has det-Jacobian ≡ 1 but **unbounded singular
   values** (no Lipschitz cap in the code) → free to fade. Counterexample: `g(s)=2s` is invertible yet explodes;
   `g(s)=s/2` is invertible yet fades. **The RHCP fade (centroid 63px) is the EXPECTED outcome, not a bug.** i-ResNet
   exists precisely because residual invertibility is not stable (must enforce Lip(F)<1); "Exploding Inverses"
   (Behrmann 2021) shows coupling nets explode off-distribution — exactly where free-running rollout goes.
   Real stabilizers, ranked: **rollout/self-forcing training (distribution-matching loss) ≫ Lipschitz/spectral
   constraints ≫ contraction/noise ≫ invertibility (contributes NOTHING to stability; it's a memory tool).**

2. **MilliVid has no recurrence to make reversible** (attachment agent, code-confirmed).
   `generate()` is stateless per-step diffusion threading only a clean-token buffer; the cross-step scatter is
   `@torch.no_grad` (gradient-severed). "Add reversibility" requires FIRST building a recurrence MilliVid intentionally
   omits → honest deliverable is "a new reversible-recurrent predictor on MilliVid's frozen tokenizer", not "MilliVid +
   reversibility". Depth-reversible DiT (RevViT) is inert at eval; eval-time inversion/consistency is incoherent for a
   stochastic, in-place, masked rollout.

3. **The space is owned by 2025-26 prior art** (novelty agent).
   **Self-Forcing** (NeurIPS 2025) = frozen VAE + DiT + autoregressive rollout training + distribution-matching loss =
   almost the entire proposal minus reversibility — AND it deliberately uses **gradient truncation, NOT deep BPTT**, so
   it fixes exposure bias *without* the memory cost reversibility exists to reduce. **CrevNet** (ICLR 2020) owns
   "reversible recurrent video predictor on Moving-MNIST". A wave of 2025-26 papers (RAD, VideoSSM, RELIC, Rolling
   Forcing, BAgger, Self-Resampling) own "recurrent latent state + frozen tokenizer + drift". Reversibility-as-memory-
   saver loses to gradient checkpointing (AlignProp + this repo's own DLH measurement).

4. **MMNIST-CIFAR can't demonstrate the claim** (verified: `mnist_cifar_test_seq.npy` background corner temporal
   std = 0.0 → static per clip). It's MMNIST with a harder-to-render but still trivially-persistable background: tests
   the DECODER, not the dynamics. 20 frames, no off-screen memory, nothing to drift about. An AR-stabilization method
   and a teacher-forced baseline look identical here.

## The honest, defensible contribution (already built + verified)

Not a method — a **correctness/enabler theorem**: a provably-invertible stochastic recurrence backprop'd through a
K-step rollout at **constant memory in K**, with a **gradient-equivalence proof vs BPTT** (already measured:
grad-cos = 1.000000; memory frontier where vanilla BPTT OOMs at K=120). This extends TempVerseFormer, survives all the
prior art (it's a theorem + figure, not a benchmark number), and is the ONLY honest home for reversibility here.

## Recommended dissertation shape (from the repo's own RESEARCH_DIRECTIONS.md)

- **Direction D (theory backbone):** the invertible stochastic-ELBO recurrence + gradient-equivalence + constant-memory
  enabler. Frame reversibility as the ENABLER of long-horizon rollout training, NOT as a stabilizer.
- **Direction C (controlled science):** *when* does a learned token hierarchy beat cascaded/flat — using MilliVid's
  `adaptive` vs `cascaded` ablation (`model_autoencoder.py` Encoder) on a **proper long-looping dataset** (viewport
  panning, sprites exit + re-enter → real off-screen memory), NOT MMNIST-CIFAR. Turns the documented nulls into the
  finding.
- Any actual eval-stability comes from a **Lipschitz/spectral constraint + self-forcing rollout training**, cited to
  the prior art — not from invertibility.

## Cheapest validation gate (before any reversibility code)
Teacher-forced single-step vs rollout-trained (**ordinary checkpointing, NO reversibility**), measured by free-running
drift-vs-horizon, on a dataset with **real off-screen memory** (not MMNIST-CIFAR). If a gap appears, reversibility earns
its place scaling K. If not, the theory + hierarchy-science paper stands alone.
