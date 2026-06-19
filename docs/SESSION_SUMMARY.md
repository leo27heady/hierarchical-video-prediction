# Session summary — building v1 (VQ-free hierarchical single-step predictor)

2026-06-19. This is the story of the session: the reframe that redirected the effort, the decisions, what
was built and gated, the engineering insights, a review of the metrics, and the future steps.

---

## 1. The reframe (the most important thing)

The work began from a belief that "the SEED-Voken predictor isn't working." A deep read of the three
predecessor codebases plus the project's own committed results corrected that:

- **The recursive multi-step predictor** in SEED-Voken was the thing that didn't work. A **clean
  single-step continuous predictor already worked** on the sibling branch `feat/continuous-joint-blt`:
  teacher-forced MID token accuracy 0.824 vs 0.476 persistence (+0.348 cross-split), no blob.
- The earlier "coarse/MID tokens are unpredictable → blob" conclusion was an **artifact of flow-only
  probes** (copy/shift/rot90 can't model in-place re-coding). A *learned* attention predictor predicts it.
- **Joint training** (predictive gradient into the encoder) was the one thing the project directly
  tested, and it **collapsed reconstruction ~1.7×** without raising predictability → use a **detached**
  target.
- The real open problem had migrated to **AR rollout drift**, not single-step predictability.

So the user's instinct (VQ-free, single-step, keep temporal downsampling, cross-attention) was right and
partly proven; v1's job was to (a) prove the hierarchy with proper cross-attention + temporal
downsampling, and (b) lay clean, anti-drift-ready foundations.

## 2. Decisions locked

- **Fresh rebuild** on `feat/vq-free-predictor` (clean `hvp/`, port only verified ideas).
- **Detached-target one-shot** training (honors "one run, no pretraining" without the collapse).
- **Hierarchy first, then AR** (v1 proves cross-attention + temporal downsampling help; AR drift is v1.5).
- **64×64** operating resolution; **Z-axis-only** rotation as the explicit v1 contract.

## 3. What was built

A fresh ~25-module `hvp/` package, built and gated milestone by milestone:

1. **Scaffold** — frozen-dataclass config tree, cross-split data layer, Windows/RSS test harness.
2. **VQ-free causal encoder/decoder** — causal Conv3d temporal downsampling `9→5→3`, deep-supervised
   decoder, coarse-first curriculum. Transplanted only the causal-conv primitive math from repo2.
3. **Attention stack + continuous predictor + G0 probe** — factorized 3D RoPE, SDPA+qk_norm, the
   single-step stage worker, and the predictability gate.
4. **Cross-attention cascade** — leak-free RF-overlap masks, coarse→fine conditioning, one-shot trainer.
5. **64×64 + wandb + richer metrics + docs** — true cross-split data, full logging incl. image tables.

## 4. Gate results

All gates pass (held-out cross-split). See [RESULTS_V1.md](RESULTS_V1.md) for the exact numbers and the
64×64 training run. Headlines:

- **G-recon**: held-out PSNR strong, all stages near full effective rank → no collapse.
- **G0** (decisive): MID `ve_changed ≈ 0.84` at the downsampled stride → **temporal-downsampling-is-fatal
  worry empirically falsified**.
- **G-stable**: one-shot converges, no NaN, predictive beats persistence, no collapse.
- **G-cascade**: cross-attention conditioning lifts BOT/MID predictive VE by ~+0.14 / +0.08 — an order of
  magnitude beyond the prior degenerate upsample-add (+0.013). **The hierarchy works.**
- **anti-blob** (pixel): single-step predicted frames beat persistence by several dB and visibly track
  rotation (not a static average blob).

## 5. Engineering insights (traps caught)

- **Plain GroupNorm breaks causality.** A `GroupNorm` over a 5-D `(B,C,T,H,W)` tensor pools over T, so the
  last frame leaks into every frame's statistics. Fixed with a **per-frame** `FrameNorm3d`. (This is a
  silent correctness bug — tests would still "pass" on shapes; only the causality test catches it.)
- **The RF-overlap mask leaks the future.** The inherited "receptive fields overlap" rule (`isdisjoint`)
  connects a child frame to a parent whose RF includes *later* frames. The fix is the strictly causal
  `max(RF(parent)) ≤ child_frame` rule, with a golden test asserting zero leak.
- **Detached target + VICReg is the stable anti-collapse recipe** for predict-your-own-latent; loss→0
  with falling effective rank is the *signature of collapse*, so eval gates on effective rank, not loss.
- **bf16 + qk_norm + finite masks** removed the AMP NaN risk that plagued the predecessor (which shipped a
  `debug_predictor_nan_step1.py`).
- **BLT correction:** vanilla BLT decodes fine "bytes" autoregressively one at a time; the "one latent
  shift → one parallel fine shift" intuition is the *Fast-BLT/BLT-D* variant. v1's parallel per-stage
  shift is a deliberate simplification, fine under near-deterministic Z-rotation.

## 6. Metrics — review and what else to compute

**Currently logged (train + held-out val, wandb):** reconstruction **PSNR/SSIM**; per-stage **effective
rank** (collapse canary) and latent **std**; **VICReg** var/cov; per-stage predictive **variance-explained**
(all + changed-cell); **cascade lift** (VE on − off); **pixel single-step** PSNR/SSIM vs persistence;
grad norm, throughput, λ, active depth; **GT/Recon/Pred image table**.

**State of the metrics:** they cover fidelity (PSNR/SSIM), collapse (effective rank/std), latent
predictability (VE), the hierarchy claim (cascade lift), and the anti-blob claim (pixel-vs-persistence).
That is enough to assert v1's claims rigorously. The honest gap is **distributional/perceptual** quality
under uncertainty and **long-horizon** behavior.

**Worth adding next:**
- **LPIPS** (perceptual) and **FVD** (Fréchet Video Distance) for distribution-level quality — both need
  pretrained nets (lpips / I3D); FVD especially for video realism.
- **Best-of-N + Density/Coverage (or Precision/Recall)** once the head is distributional (GMM/diffusion):
  single-sample MSE rewards the blurry mean and must not be the only number under multimodality.
- **Per-stage pixel decode** (decode each stage's prediction alone) to attribute pixel quality to stages.
- **AR rollout metrics** (v1.5): TF→AR gap, per-horizon PSNR/SSIM, drift/softening curves, and a
  rollout strip over the full horizon.
- **Changed-region pixel metrics** (PSNR/SSIM restricted to moving pixels) — persistence is trivially good
  on the static background, so a moving-region metric is a harder, more honest target.
- **Calibration** of the gaussian head (NLL, coverage) once enabled.

## 7. Future steps & considerations

1. **De-blur** the prediction: a longer/larger run, and the committed **GIVT-GMM distributional head**
   (continuous, multimodal) before SO(3). MAR-style per-token diffusion head if GMM underfits.
2. **Windowed BOT attention** — BOT (2304 tokens) is ~the entire compute budget; window it before any
   finer grid or higher resolution.
3. **AR rollout stability (v1.5, the real frontier):** the cascade already supports a one-pass forward;
   add autoregressive rollout with **noised conditioning** (Diffusion-Forcing/GameNGen) and log the TF→AR
   gap from day 0. This is where both predecessors actually failed.
4. **SO(3) rotation:** unlock the dataset's Z-axis lock (a one-line change), paired with the distributional
   head + best-of-N eval — multimodality (self-occlusion, depth/axis ambiguity) will otherwise re-blob the
   coarse stage.
5. **Reversibility** integration (dissertation goal #3) — orthogonal to the blob; bring back behind a flag.
6. **Rigor:** multi-seed the cascade-lift A/B (confirm it's beyond noise); a second independently-rendered
   cross-split is already in place but more render configs would strengthen it; hyperparameter sweeps on
   `λ`, stage weights, dim/layers.
7. **Joint training (opt-in escalation):** revisit predictive-gradient-into-encoder only behind a
   recon-non-regression gate and with VICReg active — the thesis's "organize the latent for
   predictability" claim, but unproven net of the recon cost.

## 8. Considerations / open questions

- **Resolution vs compute:** 64×64 is the operating resolution; going higher will make BOT (and the
  encoder) dominate — windowed attention and possibly a wavelet front-end (WF-VAE/Cosmos) are the levers.
- **Continuous vs light-KL:** v1 uses a plain no-KL AE (Cosmos-style). A light KL or latent-space
  regularization to a pretrained image VAE is an option if warm-starting or smoother latents are wanted.
- **Cross-stage gradient:** parent hidden is detached by default; whether end-to-end cross-stage gradient
  helps (vs collapse risk) is an explicit, gated experiment.
- **Metric honesty under multimodality:** the moment SO(3) lands, stop trusting single-sample MSE/PSNR;
  switch to best-of-N + distributional metrics.
