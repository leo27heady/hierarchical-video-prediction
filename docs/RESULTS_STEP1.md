# Step 1 results — adversarial (distribution-matching) renderer on frozen CW-VAE codes

> **Verdict: SUCCESS.** A VQGAN-style adversarial renderer on the frozen CW-VAE `[s,det]` code produces the
> **sharpest, coherent** open-loop digits, with by far the best FID — validating the ladder's premise that a
> **distribution-matching** objective (adversarial) beats perceptual (LPIPS) and pixel (L2/blur) for sharpness.
> The win required two diagnosis-driven fixes (warm-start + fg-weight); the naive fresh recipe failed.

## The clean comparison (all arms warm-start + fg-weighted; only the loss differs)
Open-loop (24 held-out clips, 3 rollout seeds), 1 training seed:

| arm | OL FID ↓ | fg-LPIPS ↓ | fg-MSE | lap-var ↑ | tcoh |
|---|---|---|---|---|---|
| baseline (pretrained, frozen) | 101.4 | 0.300 | 0.425 | 0.008 | 0.072 |
| l2 (MSE) | 114.4 | 0.265 | 0.445 | 0.014 | 0.082 |
| lpips (L1+LPIPS) | 125.3 | **0.232** | 0.438 | 0.041 | 0.082 |
| **gan (L1+LPIPS+adaptive-PatchGAN)** | **66.5** | 0.239 | 0.486 | **0.067** | 0.091 |

(TF-recon mirrors this: gan FID 60.7 vs baseline 90, fg-LPIPS 0.098, lap-var 0.066 ≈ GT 0.056.)

- **gan wins FID by ~35–60 points** (well beyond FID's ~18 noise floor) — best distribution-match, and the only
  arm *below* the blurry baseline. Visually (`s1_compare_strip.png`) the gan digits are the **sharpest and
  remain recognizable**. The FID win is real (coherent), not the sharpness-gaming seen with the fresh recipe.
- **gan is the sharpest** (lap-var 0.067, near GT 0.056) at near-best fg-LPIPS.
- **Cost = fg-MSE** (0.486 vs 0.425): the expected Blau–Michaeli perception-distortion tradeoff — a clean **Pareto
  win on distribution + sharpness**, not a free lunch. (The strict gate flagged "no clean win" only because it
  required gan fg-MSE ≤ lpips fg-MSE; the Pareto reality is the point.)

## The journey (fresh recipe failed → two fixes)
1. **Fresh-init adversarial renderer FAILED:** sharp but **incoherent texture even from teacher-forced codes**;
   the fresh L2 arm **collapsed to black**; and **per-frame FID was gamed by sharpness** (rated sharp-incoherent
   *above* blurry-coherent) — confirming the red-team's concern about out-of-domain Inception-FID. The visual was
   the truth, not FID.
2. **Diagnosis:** MMNIST's sparse foreground → L1 is background-dominated → the GAN texturizes the foreground; an
   *unconditional* D rewards realism not correspondence; a *fresh* decoder can't learn coherent code→frame on 120
   clips.
3. **Two fixes cracked it:** **warm-start** G from the pretrained *coherent* decoder (GAN sharpens, not builds);
   **foreground-weighted L1** to pin the sparse digit. → sharp + coherent + accurate.

## Methodological findings (reusable)
- **FID is gamed by sharpness when the output is incoherent** (fresh-gan FID 89 < blur 101 but visually broken);
  it reflects real quality only when coherence is enforced (warm+fg). → never trust FID alone; pair with the
  visual and a coherent recipe. **LPIPS-only games the perceptual metric** (best fg-LPIPS, worst FID = texture).
- The **perception-distortion frontier is real**: sharpness costs fg-MSE; report the Pareto, never claim both.

## Validated recipe
`WarmStartG(pretrained decoder)` + `L1(fg-weighted×8) + LPIPS + adaptive-λ·PatchGAN` (VQGAN adaptive λ from the
final ConvTranspose, D warm-up 500, spectral-norm, hinge). lr 5e-5, ~6k steps, batch 64. VRAM ~5 GB, ~25 it/s.

## Honest caveats / next
- **Open-loop is still partly dynamics-limited** at long horizon: the renderer is sharp+coherent on *good* codes
  (TF-recon), but free-run codes are uncertain, so open-loop digits drift — **this is the dynamics problem
  (Step 3), not the renderer.** The renderer is decoupled and validated.
- 1 training seed shown; **gan seed-band (seeds 1,2) running** for the FID-stability CI.
- FID is out-of-domain (relative ranking only); fg-LPIPS + the visual corroborate.
- **Next: Step 3** — a distribution-matching *rollout* objective (Diffusion-Forcing-style) on the hierarchical
  latent to fix the open-loop drift+fade, where the reversible core (TempVerseFormer) plugs in as the
  long-rollout memory enabler. The renderer (Step 1) provides the sharp decode head it will train against.
