# Project Summary — Hierarchical (+ Reversible) Video Prediction

> **START HERE.** Consolidated state after the DLH-reproduction → diagnosis → per-pixel → reversibility → BLT →
> RHVP → **MilliVid** arc. Supersedes `docs/HANDOFF.md` as the top-level entry. Companion: `docs/RESEARCH_DIRECTIONS.md`
> (forward plan), `docs/MILLIVID_DIRECTION.md` (the current direction in detail), `docs/RHVP_ARCHITECTURE.md`,
> `docs/PHASE0_PERPIXEL_REPORT.md`, `docs/BLT_MEMORY_EXPERIMENTS.md`. Branch: `explore/blt-decoder-gating`.

## 0. One-paragraph state
We reproduced DLH/CW-VAE (VAE+RNN hierarchical video), diagnosed why it underperforms, and tried to make the
dissertation's two pillars — a content-adaptive hierarchy and **reversibility** — pay off on it. They didn't: the
adaptive clock NULLS on Moving-MNIST, blur is inherent to the L2-VAE objective, and **both resource axes**
(compute-sparsity, memory-reversibility) **empirically failed** on a small conv-dominated model. We then found
**MilliVid** (MIT+TRI, arXiv 2606.09056) — a SOTA hierarchical *diffusion* video model with released code — and
pivoted to building on it. **Current direction:** build on MilliVid; the reversibility contribution is re-homed as the
constant-memory **enabler** of a *K-step self-conditioned rollout-unroll* objective (fixing MilliVid's clean-context
exposure bias). See §4 + Research Directions.

## 1. What was verified (the findings, with evidence)
1. **Blur is INHERENT** to the fixed-σ Gaussian-L2 VAE objective (the paper concedes it). C0 probe: on DLH's own
   frozen codes, swapping only the decoder objective moves lap-var 0.0026→0.042≈GT. → fix = a non-L2 renderer, not knobs.
2. **Instability is port-config, solved.** sigmoid-output collapses to black (sigmoid(0)=0.5 ≫ data-mean → saturates);
   tanh(0)=0 ≈ mean → stable. The dropped `(1-e)KL[q‖static]` term is NOT a bug (provably 0 under the carry rule).
3. **The adaptive clock NULLS on MMNIST.** Engaging the hierarchy (low dec_stddev, or the per-pixel/temporal clock)
   capacity-dumps with no quality benefit, and *hurts* the long-horizon edge. MMNIST is clock-null + free-to-persist
   (black bg) — it cannot show hierarchy/clock value. (e1_clock_premise: upper-level separation DEAD.)
4. **dec_stddev is a real knob** (overturned a red-team "no-op" call): lowering it 1.0→0.3 engages all 3 levels +
   sharpens 2.6×, but costs long-horizon accuracy. Most-accurate base = σ=1.0.
5. **Renderer works, with a ceiling.** distribution-matching > perceptual > pixel. GAN renderer on frozen codes →
   FID 97, GT-level sharp, but **identity-imperfect** open-loop (capped by the 2-level × 20-dim code bottleneck). It
   fixes texture-blur, not long-horizon prediction-blur (that's dynamics drift — E1).
6. **Per-pixel spatial gate: mechanism ✓, quality null.** The per-location hard-KL gate **segments the moving digits**
   (seg-corr 0.57 — premise confirmed), but the upper spatial level engages unstably and adds no quality on MMNIST.
7. **Both resource axes FAILED on the DLH-class model.** (a) Compute: tile-sparse decoder gating is 0.73-0.88× — *slower*
   than dense (decoder launch-bound at 1ms; masked-dense; gather overhead). (b) Memory: gradient checkpointing matches
   reversibility because the per-frame convs dominate the tape, not the tiny recurrence. Reversibility's win needs the
   *recurrence to dominate* memory — which fights the anti-drift small-latent design.
8. **RHVP** (a deep reversible latent core + cheap codec to make reversibility dominate) was designed; M0 showed the
   param-inversion works but reversible-vs-checkpointing memory is measurement-sensitive → the defensible contribution
   is the **correctness theorem** (first provably-invertible stochastic-ELBO recurrence), not a resource number.

## 2. Methodology lessons (do not relearn)
- **Gate on fg-MSE / FVD / fg-LPIPS, never aggregate SSIM/PSNR** on MMNIST (black bg scores SSIM 0.76).
- **Always check the visual + a coherent recipe:** per-frame FID is gamed by sharpness; LPIPS-only games fg-LPIPS.
- **Resource wins evaporate on small/conv-dominated models:** checkpointing matches reversibility; sparsity overhead
  exceeds savings at 16×16 tiles on a 4060 Ti. Measure wall-clock, not FLOPs.
- **best-of-100 eval inflates SSIM** (the paper's 0.78 vs best-of-6).
- **Process that worked:** plan → multi-agent red-team → implement → **cheapest training-free kill-shot first** → gate.
  Front-load kill-shots before building machinery. Second pair of eyes (code-review/red-team) at every decision.

## 3. Reusable assets
- **Code:** `hvp_dlh/` (faithful DLH port, + spatial per-pixel variant `model_spatial.py`, + `train_spatial.py`);
  `hvp_rev/` (renderer `train_renderer_dlh.py`, codes `codes_dlh.py`, reversible proto, `memory_axis_demo.py`,
  `rhvp_m0_memory.py`); `hvp_blt/tile_decoder_bench.py`. Renderer recipe (warm-start decoder + fg-weighted L1 +
  adaptive-λ PatchGAN) validated.
- **External base (the future):** `../millivid` (MilliVid code: hierarchical tokenizer + coarse-to-fine DiT + rollout
  strategy + a full metrics suite: DINOv2/LightGlue-keypoints/FVD/FID/LPIPS + FramePack/full-res baselines).
- **Reversibility primitive:** TempVerseFormer (user's paper) — additive coupling along time; P1-P4 proven in
  `scripts/reversible_temporal_proto.py`; K1 confirmed exact grads to T=32 (drift onset ~T=150 → needs Lipschitz +
  anchors).
- **Synthetic data:** Moving-MNIST variants, colour Moving-Ball, in `../cwvae/data_mmnist` + `data/`. (Too simple for
  long-range-consistency claims — see Directions for the purpose-built looping dataset needed.)

## 4. Current direction (decided)
**Build on MilliVid.** Reuse its tokenizer + DiT + rollout + metrics; train a **scaled-down model from scratch** on a
**purpose-built long looping dataset** (NOT MMNIST). Don't fine-tune (training latents unreleased; Minecraft doesn't
transfer). **Reversibility has no home in MilliVid as-published** (per-step decoupled denoising on clean context, no
BPTT) — its one honest home is as the **constant-memory enabler of a K-step self-conditioned rollout-unroll** objective
that fixes MilliVid's clean-context **exposure bias** (= the user's own E1/e0 exposure-gap finding). Full architecture
+ math + the integration in `docs/MILLIVID_DIRECTION.md`; forward plan in `docs/RESEARCH_DIRECTIONS.md`.

**Contribution (one sentence):** *rollout-aware training of a coarse-to-fine hierarchical latent video model — K
self-conditioned unroll steps with a distribution-matching objective, made affordable on one GPU by reversible coupling
along the rollout-step axis — cutting the long-range drift MilliVid's independent-per-step (clean-context) denoising
leaves uncorrected.*

**Hardware:** single RTX 4060 Ti 16 GB. MilliVid inference fits (slow); scaled-down from-scratch fits at batch 4-8;
the binding constraint is **task design**, not hardware.
