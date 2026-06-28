# E1 — on-the-fly rollout training on the CW-VAE backbone (plan, pre-implementation)

> Gated by E0 (exposure gap CONFIRMED: open-loop fg-MSE 0.49 vs teacher-forced 0.14 @+25; fg-LPIPS gap
> 0.12–0.15). Goal: train the model on its OWN free-running rollouts to close that gap. **No reversibility
> yet** — isolate the method (reversibility enters at E2/E4 only to scale the window). To be red-teamed
> before implementation.

## Objective & success criterion
Reduce the **open-loop** fg-MSE/fg-LPIPS@+25/+50 vs the frozen CW-VAE baseline, **beyond a 3-seed CI**,
**without** (a) regressing short-horizon (+1..+10) accuracy, (b) losing sharpness (lap-var must not drop),
or (c) collapsing the latent (KL stays > 0). SSIM/PSNR reported only (background-dominated).

## Design (to critique)
**Backbone & location.** CW-VAE (`../cwvae/cwvae_torch.py`), **warm-started** from the pretrained
`logs_torch/cwvae/last.pt`. New trainer `../cwvae/train_rollout.py` (reuses `CWVAE`, data loader); keep the
original `train_torch.py` untouched as the baseline.

**The training step.** For a clip of length `L = ctx_tf + M`:
1. Encode all `L` frames.
2. *Phase A (teacher-forced, ctx_tf steps):* posterior `hierarchical_unroll(use_obs=True)` → carried state;
   standard ELBO (recon + per-level KL) on these frames.
3. *Phase B (free-run, M steps, WITH grad):* `hierarchical_unroll(use_obs=False, initial=carried)` → decode
   the M predicted frames; **rollout recon loss** = Gaussian-NLL(pred, real_future). (No posterior in
   free-run → no KL there.)
4. `loss = ELBO_A + λ_roll · rollout_recon_B` (start `λ_roll = 1`, `ctx_tf = 24`, sweep `M ∈ {4, 6, 8}`).

**Scheduled sampling (variant, behind a flag).** With prob `p_tf` (anneal 1→0) inject the *real* frame
(posterior) instead of the prior sample at each Phase-B step, to ease optimization. Default OFF for the
first run (pure free-run) so the result is unambiguous; enable only if optimization is unstable.

**Memory.** BPTT through M steps only (`M ≤ 8`). Gradient-checkpoint the conv enc/dec. Benchmark s/step and
peak VRAM at `M=8` BEFORE any multi-hour run (the no-assume rule).

**Optimization.** Adam, lr `1e-4` (lower than 3e-4 pretrain — fine-tuning), grad-clip, `total ≈ 10–15k`
fine-tune steps from the warm start. Batch sized to fit VRAM (start 25).

## Metric & sample tracking (live)
Every 500 steps: log `loss`, `elbo`, `rollout_recon`, per-level KL, grad-norm, **peak VRAM**.
Every 2000 steps (eval): open-loop on ≥48 held-out clips → **fg-MSE@25/50, fg-LPIPS@25/50, lap-var**
(sharpness tendency), short-horizon fg-MSE@5; **save a GT-vs-pred strip** to `logs_rollout/strips/`;
plot the metric-vs-step curves. Always print Δ vs the E0 baseline. Sharpness watch = lap-var trend.

## Post-training validation / gate
Re-run the E0/A0 harness on the final model vs the frozen baseline: open-loop fg-MSE/fg-LPIPS@25/50,
**3 rollout seeds, 95% CI**, best-of-N by fg-MSE. **Adopt only if** the open-loop gap shrinks beyond the
CI AND short-horizon + lap-var don't regress. Report the teacher-forced-vs-open-loop gap before/after
(did rollout training close it?). Save a before/after comparison strip.

## Resource discipline (disk = 13 GB free; GPU = 16 GB)
- **Checkpoints: `_last.pt` only, overwrite, `ckpt_every=2000`; never keep history.** Delete stale ckpts.
- Strips are small PNGs; cap kept strips (last ~10). Eval JSON only.
- `nvidia-smi` + `df -h` logged at start, mid, and end of training; abort if free disk < 3 GB.
- Estimated footprint: one 49 MB checkpoint + < 50 MB strips/plots = < 150 MB. Safe.

## Risks / open questions (for red-team)
- Hierarchical free-run with grad: upper levels tick every `k=6`, so for `M ≤ 8` the top level ticks ~0–1×.
  Is the rollout loss well-defined / does grad flow into the slow levels at all within a short window?
- Pure free-run training may be unstable (no teacher signal in Phase B) → may need scheduled sampling from
  the start, not as a fallback.
- Warm-start vs from-scratch: warm-start risks the model staying near the teacher-forced optimum; from
  scratch is far more expensive. Is warm-start the right call?
- λ_roll, ctx_tf, M are coupled — is a 1-var-at-a-time sweep affordable, or do we need a tiny grid?
- Does closing the open-loop gap actually require *gradient through the rollout*, or would simpler
  data-augmentation (add noise to inputs) get most of it? (cheaper baseline to include?)
- Is `../cwvae` the right home, or should this live in this repo as a clean module for the dissertation?
