# Step 1 — adversarial (VQGAN) renderer on frozen CW-VAE codes (plan, pre-implementation)

> Ladder step #1. Tests the diagnosis: a distribution-matching (PatchGAN) term on the FROZEN CW-VAE
> `[s,det]` code yields SHARP open-loop frames without breaking fg-MSE beyond a floor. No teacher needed.
> To be red-teamed before implementation.

## Objective & success criterion
On a held-out set, the **adversarial arm** (L1+LPIPS+PatchGAN) must beat the baseline decoder AND the L2/
LPIPS-only controls on **FID + fg-LPIPS** (the distribution-matching gate) while keeping **fg-MSE within a hard
floor (≤ +10–15% vs the frozen baseline decoder)**, over 2–3 seeds (mean±std, win beyond combined std). Measured
on **open-loop** predictions (the actual goal), with teacher-forced reconstruction reported too. lap-var = a
gameable diagnostic only; SSIM/PSNR continuity-only.

## Design (to critique)
**Frozen backbone & codes.** Freeze CW-VAE (`../cwvae`, pretrained `last.pt`). Extract the bottom output
`[s,det]` (220-dim) per frame. **Train on teacher-forced codes** (posterior, real frames → the C0 protocol that
decouples the renderer from the dynamics). Also cache **free-run codes** for eval-time transfer.

**Generator (new decoder G).** Mirror `ConvDecoder` (220 → 64×64), `out_activation` to match data ([0,1]; use
sigmoid or clamp). Initialize fresh (not the pretrained decoder) so all arms start equal.

**3 arms (each 2–3 seeds):**
- **A — L2** (Gaussian-NLL/MSE): the regression-to-mean control (should blur).
- **B — L1+LPIPS**: the C0b baseline (should sharpen but risk incoherent texture).
- **C — L1+LPIPS+PatchGAN** (the method): pix2pix 70×70-receptive-field discriminator (~21M), **unconditional**
  (frame realism = the divergence estimator). With **(i) discriminator warm-up** (first `W` steps pure L1+LPIPS,
  GAN off) and **(ii) VQGAN adaptive λ**: `λ_adv = ‖∇_lastG L_nll‖ / (‖∇_lastG L_adv‖ + ε)`, clamped to `[0,1e4]`,
  detached — the built-in guard against the C0b texture-runaway. Non-saturating GAN (or hinge) loss.

**Optimization.** Adam G (lr 1e-4) + Adam D (lr 1e-4, β1=0.5), 1:1 updates after warm-up; ~8–12k steps; batch
~32; spectral-norm or R1 on D for stability. Frozen-code dataset from N≈40 clips × ~50 frames (train/eval split
by clip).

## Eval / gate (the validation step)
1. **Teacher-forced recon** (decoder sees real-frame codes): FID, fg-LPIPS, lap-var, fg-MSE — does G render sharp
   from real codes? (upper bound).
2. **Open-loop** (the goal): swap G into CW-VAE's decoder, run `open_loop` (free-run codes), eval **FID + fg-LPIPS
   + fg-MSE + lap-var at +8/+25/+50** vs the frozen-baseline decoder. **Transfer check:** does sharpness trained on
   TF codes survive on free-run codes? If not → Step-1b fine-tune G on free-run codes.
3. **Gate:** arm C beats B and the baseline on FID+fg-LPIPS (open-loop) beyond the 2–3-seed band, fg-MSE within the
   floor. If the GAN destabilizes or breaks the floor → fall back to the MAR diffusion head (ladder #2) on the same
   harness.

## Metric & sample tracking (live)
Every 500 steps: G loss (L1/LPIPS/adv), D loss, adaptive-λ, D real/fake accuracy (collapse watch), grad-norms,
peak VRAM. Every ~2k steps: eval (FID/fg-LPIPS/lap-var/fg-MSE on TF-recon + open-loop) + **GT vs A/B/C sample
strip** to `logs_render/strips/` (the sharpness-tendency images). Plot metric-vs-step.

## FID infra (de-risk first)
FID via torchvision InceptionV3 (pool3, 2048-d); resize 64→299, gray→3ch (standard). MMNIST is out-of-domain for
Inception but FID is a **relative** distribution distance across arms — valid for ranking. **Verify InceptionV3
weights load (download) BEFORE the run**; if unavailable, fall back to a frame-FID on the LPIPS-AlexNet features
(already cached) as a documented proxy. (FVD/I3D = temporal, deferred to the rollout step #3.)

## Resource discipline (disk; GPU 16 GB)
- Frozen-code cache (~40×86×220 floats ≈ small) + decoder/D checkpoints (single `_last.pt` per arm/seed, atomic
  temp+os.replace, ckpt_every 2k). InceptionV3 weights ~100 MB (one-time). Strips capped at 10/run.
- `df`/`nvidia-smi` at start/mid/end; **abort if free disk < 3 GB**. Estimated footprint < 500 MB.
- GPU: G + PatchGAN D + InceptionV3 fit 16 GB with room (CW-VAE frozen, codes precomputed). VRAM tripwire 12 GB.

## Open questions (for red-team)
- Train on TF codes only vs TF+free-run mix from the start (transfer risk vs clean decoupling)?
- Unconditional vs code-conditional PatchGAN?
- GAN stabilizer: spectral-norm vs R1 vs both; warm-up length W; adaptive-λ clamp range?
- Is per-frame FID (InceptionV3, out-of-domain MNIST) a sound enough gate, or is a temporal FVD needed even at
  Step 1?
- How many seeds is enough for the FID/fg-LPIPS band given FID's small-sample noise?

---
## v2 — red-team folded in (GO-WITH-CHANGES)
- **Backbone = CW-VAE (user's explicit choice).** The red-team pushed DLH because the *old* c0 harness loads DLH;
  instead I build a FRESH CW-VAE renderer (`hvp_rev/codes.py` 220-dim `[s,det]`, `modules_gan.py`). The **smoke is
  the fresh CW-VAE headroom check** — I do NOT inherit DLH's C0 headroom claim.
- **Gate = PARETO, no absolute fg-MSE floor (it auto-failed: C0b perceptual = +45% fg-MSE).** Primary = **TF-recon
  FID + fg-LPIPS**; arm C must improve FID+fg-LPIPS over the **fresh-L2 arm A (primary ref, matched init/steps/
  activation)** and beat arm B (LPIPS-only) on FID **at no-worse fg-MSE than B**. Pretrained decoder = secondary
  anchor only. Open-loop = **transfer diagnostic** (report FID/fg-LPIPS/fg-MSE/temporal-LPIPS); open-loop rollout
  fg-MSE must not inflate > ~0.006–0.009 in DLH-leading windows (principled, from a0); a regression triggers
  **Step-1b** (free-run-code fine-tune), not a NO-GO.
- **FID validated** (monotonic in blur; ~18 noise floor). Report **InceptionV3-FID AND LPIPS-AlexNet-feature-FID**
  (directional agreement) + fg-LPIPS primary; clip-bootstrap CI (win = CI excludes 0); ≥5 sampling + ≥3 training seeds.
  Add **temporal-LPIPS** coherence on open-loop. Full FVD deferred to Step 3.
- **adaptive-λ spec:** `last_layer = G.net[-1].weight` (final ConvTranspose, not the dense); `g_nll = grad(L1+LPIPS,
  last_layer)` (full non-adv loss incl. LPIPS); `λ=(‖g_nll‖/(‖g_adv‖+1e-4)).clamp(0,1e4).detach()`. Pre-register
  warm-up W, clamp, spectral-norm (VQGAN defaults); freeze across seeds; never tune on the eval clips.
- **Compute:** PatchGAN ~2.8M (not 21M); VRAM tripwire **6 GiB**; weights-only atomic `_last.pt` (~26 MB/arm);
  FID over the full eval-frame draw with one cached real-stats. **Smoke first** (1 seed, arm C, TF, ~2k steps:
  D-acc∈[0.5,0.9], λ finite, recon not destroyed, strip sharper) → then 3-arm × 3-seed sweep.
