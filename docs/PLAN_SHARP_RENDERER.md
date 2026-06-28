# Sharp CW-VAE via distribution-matching — roadmap (web-grounded)

> After two perception-distortion walls (C0: blur = the Gaussian-L2 objective, not the latents; E1: MSE rollout
> training fades to the mean), the fix is a **distribution-matching objective**, not another pixel loss. This is
> the ranked, single-16GB-GPU-feasible plan, with the cheapest decisive experiment first.

## Why distribution-matching (the principle)
Blau–Michaeli: you cannot minimize pixel-distortion (MSE) **and** maximize perceptual quality at once. Under
future uncertainty the MSE-optimal output is the blurry mean — so any pixel-loss model blurs (and E1 showed
piling on more pixel-loss fades it further). A **divergence/distribution-matching term** (adversarial PatchGAN, or
a diffusion loss) drives `d(p_X, p_X̂) → 0`, yielding sharp **in-distribution samples**. C0 proved CW-VAE's frozen
latents already carry GT-level information — so this is a pure **objective swap at the render head**.

## Ranked approaches
1. **Adversarial renderer (VQGAN first-stage recipe) — DO FIRST.** Small L1 + LPIPS + **PatchGAN** on the frozen
   `[s,det]` code, with VQGAN's **adaptive λ** (`‖∇L_perc‖/(‖∇L_adv‖+ε)` at the decoder's last layer) and a
   **discriminator warm-up**. The PatchGAN D *is* a local-divergence estimator (the principled fix), and the
   adaptive λ + warm-up are the built-in guard against the **C0b texture-hallucination** (LPIPS-only ran away into
   incoherent texture). No teacher needed. Reuses `scripts/c0_decoder_headroom.py` almost verbatim. **High-value.**
2. **MAR-style diffusion decoder head** (~21M denoising MLP, AdaLN on the code) — models `p(frame|code)` and
   *samples* one sharp mode → structurally immune to mean-regression; no GAN instability. Rank 2 only because
   ~100 denoise steps/frame makes long rollouts slow. **Principled fallback** if the GAN destabilizes. **Promising.**
3. **Diffusion-Forcing rollout objective** (per-frame independent noise over the un-reset CW-VAE recurrence) — the
   *sequence-level* distribution-matching fix for E1's fade, and structurally identical to CW-VAE/DLH. **This is
   where reversibility (TempVerseFormer) plugs in** as the constant-memory enabler for long-rollout BPTT. Do it
   **after** the renderer is independently proven (don't couple renderer+rollout in one experiment). **Promising.**
4. **Self-Forcing / DMD2 — OUT for now.** Its score-difference distribution term needs a **pretrained diffusion
   teacher**, which doesn't exist for MMNIST; stripped of the teacher it collapses to #1 with heavier online-critic
   machinery. The headline-attractive overclaim trap. **Too-heavy / inert here.**
5. **LPIPS-only / fg-weighted-MSE — known dead end** (C0b hallucinates texture; fg-MSE over-paints). The baseline
   the others must beat, not a candidate.

## The staged thesis arc (each contribution independently falsifiable)
**Renderer (per-frame divergence) → proves sharpness is reachable** (orthogonal to dynamics; trained on frozen
codes = C0 protocol; keeps the hierarchical/reversible story separable). **→ Diffusion-Forcing rollout (sequence
divergence) on the hierarchical latent → fixes drift + fade** (the principled E1 fix). **→ reversibility as the
MEMORY enabler** (not a stability claim) making the long rollouts affordable; honor the no-forgetting caveat
(store-the-noise / reversal-with-forgetting / spectral coupling).

## Cheapest decisive experiment (next, ~few hours, no teacher)
Extend `scripts/c0_decoder_headroom.py` (already extracts frozen codes → frame, LPIPS-instrumented) with a
**PatchGAN arm**: 3 arms × 2–3 seeds — `{L2 control, L1+LPIPS (the C0b over-texture baseline),
L1+LPIPS+adaptive-PatchGAN}` — discriminator warm-up + VQGAN adaptive λ.
**Gate = FVD + fg-LPIPS at matched identity with a HARD fg-MSE floor (≤ +10–15%)** — NOT lap-var/LPIPS alone
(gameable, C0b), NOT MSE/SSIM (tied + background-dominated). **Decision:** ship the renderer iff FVD/fg-LPIPS
improve while fg-MSE stays within the floor; if the GAN destabilizes or breaks the floor, fall back to the MAR
diffusion head (#2) on the same harness.

## Honest risks
- **Perception-distortion is a frontier, not a free win:** the divergence term *will* regress fg-MSE — the one
  axis DLH leads long-horizon (~0.01–0.02). Floor fg-MSE; report the tradeoff, don't claim sharpness **and**
  accuracy.
- **FVD is mandatory** (lap-var/LPIPS gameable) and itself small-sample-noisy on MMNIST → multiple seeds + CIs.
- **MMNIST is a poor dynamics testbed** (background-dominated, hierarchy-null): the renderer result transfers, but
  any rollout/Diffusion-Forcing claim must be re-validated on **KTH / an event dataset** before the dissertation.
- **GAN instability on 16GB** (mode collapse, D>G): adaptive-λ + warm-up help but don't guarantee → diffusion-head
  fallback budgeted.
- **Training-length confound** (CW-VAE 300ep vs DLH 8k it) — compare at matched identity/budget.
- A diffusion/GAN-sampled head adds a **second stochastic source** atop the latent noise → temperature-tune or it
  competes with the dynamics' own stochasticity.
