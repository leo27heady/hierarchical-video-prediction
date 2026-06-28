# E1 results — on-the-fly rollout training (MSE objective): a clean NEGATIVE

> Reduced first pass: {lambda0, nograd, withgrad} × 3 training seeds, M=8, warm-started CW-VAE,
> rollout loss = Gaussian-NLL (≈MSE) on free-run predictions (prior mean). Gate: `logs_rollout/e1_gate.json`.

## Result (open-loop, 64 clips, 3 rollout seeds, mean±std over 3 training seeds)

| arm | fg-MSE@8 | fg-LPIPS@8 | lap-var | fg-MSE@50 | fg-LPIPS@50 |
|---|---|---|---|---|---|
| baseline (frozen) | 0.304 | 0.246 | ~0.0073 | 0.519 | ~0.37 |
| lambda0 (TF fine-tune control) | 0.303 | 0.248 | ~0.0073 | 0.521 | — |
| nograd (decoder-only rollout) | 0.508 | 0.525 | 0.0054 | 0.524 | — |
| **withgrad (the method)** | **0.299** | **0.418** | **0.0017** | **0.511** | **0.740** |

**Gate verdict: NULL.** `withgrad` shaved fg-MSE marginally (0.304→0.299 @8; 0.519→0.511 @50) but **destroyed
sharpness**: lap-var 0.0073→**0.0017** (~4×, now ~3% of GT) and fg-LPIPS@8 0.246→**0.418**. Visually
(`logs_rollout/e1_baseline_vs_withgrad.png`) the rollout-trained model **fades to faint blobs by +25/+50**
while the baseline keeps digit structure. `lambda0` ≈ baseline (more TF fine-tuning changes nothing — a clean
control). `nograd` degraded everywhere (training the decoder alone on rollout latents is an impossible mapping).

## Why (the mechanism)
The rollout loss is **MSE on free-run predictions**, and the future is **uncertain** (a digit can be anywhere).
The MSE-optimal prediction for an uncertain future is the **blurry mean**, and as the horizon grows and
uncertainty grows, that mean → **fainter/emptier**. So MSE rollout training *explicitly optimizes toward blur
and fade*. Teacher-forced training avoids this (the posterior sees the real frame each step, low per-step
uncertainty); free-run rollout training exposes the model to its own uncertain predictions and the MSE loss
**rewards hedging**. The latent did NOT collapse (top-KL stayed ~3.7–4.5; the `sample=False` mitigation worked),
so the blur is purely the **objective**, not posterior collapse.

## The lesson (dissertation-relevant)
A clean, defensible negative: **naive exposure-bias correction (MSE rollout training) on a stochastic video
model trades long-horizon sharpness for negligible pixel-accuracy — predictions fade to the mean.** This
empirically demonstrates that the rollout-training objective **must be distribution-matching / perceptual, not
pixel-MSE** — which is exactly why Self-Forcing / Diffusion-Forcing use distribution-matching losses, and it
reconnects to the C0/C0b finding that the Gaussian-L2 objective is the sharpness bottleneck.

**Sequencing consequence:** fix the OBJECTIVE before scaling the window with reversibility. Scaling a
blur-inducing objective to longer rollouts (the reversibility payoff) would make the fade *worse*. So E1b
(perceptual rollout objective) gates the reversibility stage.

## Next: E1b — rollout training with a PERCEPTUAL objective
Replace/augment the rollout NLL with **LPIPS (+ optional light adversarial)** so the loss rewards sharp,
realistic frames instead of blurry means; keep the in-window gate, the controls, and the lap-var/top-KL guards.
Decisive question: does a *perceptual* rollout loss close the exposure gap **without** the blur? If yes →
reversibility (longer M) is worth building. If it still blurs → the rollout-training direction is the wrong lever
and the renderer (C1) is the move. Plan → red-team → implement (per the established loop).
