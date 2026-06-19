# PLAN P2 — anti-drift training (the binding frontier)

> 2026-06-19. From a 7-agent design/critique/synthesis pass grounded in the actual code + SOTA, after
> SO(3) confirmed **autoregressive drift** as the binding constraint (changed-region 8.70→6.67 over 24
> steps; the model is competitive single-step but drifts to baseline-level). This is the plan to close it.

## Headline

P2 = a **grad-enabled short self-rollout (Self-Forcing) term** added to `compute_oneshot_loss`: the
cascade consumes its OWN detached, fed-back latent predictions for K steps and is supervised by
`head_nll` against the encoder's **detached** future latents, with stochastic one-step gradient
truncation (~1 extra cascade forward+back). It beats the predecessors' failed *light-noise + 1-step
scheduled sampling* on three code-grounded counts: (1) **K>1 lets errors compound inside the loss**, so
the model trains on the real drift distribution (drift is a many-step phenomenon a 1-step perturbation
never reaches); (2) it feeds back the model's **actual on-manifold error** via `head_mode`/`head_sample`,
not isotropic Gaussian noise; (3) it is the only term that consumes the previously-unsupervised position
T-1 — **absorbing Fix #4**. The detached-target discipline is strictly preserved (targets stay
`stopgrad(z)`, every fed-back latent is detached, encoder sees only recon+VICReg), so the recon-collapse
failure is structurally impossible (locked by a zero-encoder-grad test).

## Ground-truth corrections this pass surfaced (propagate everywhere)

- **The latent rollout does WHOLE-STACK replacement** (`rollout.py` `z = z_pred`, each stage stays at its
  native length 9/5/3) — there is **no window-slide / per-stage native-rate slide** anywhere in the repo.
  Earlier design notes assumed that nonexistent slide; the grad-capable unroll must reuse the exact
  whole-stack branch.
- `tests/predictor/_golden` does **not** exist on disk — re-pin all "bit-identical when flag off" claims
  to the real flat guards (`test_rollout.py`, `test_cascade.py`, `test_train_guard.py`).
- **Data prereq:** the train caches are T=9 = the model window, so honest T-1 supervision needs clips of
  length ≥ seed_len+K. K=1 (P2.1) is the only phase that needs no new data.

## Prerequisites

1. **Refactor (the one real prereq):** extract a grad-capable `unroll_latent(cascade, z0, n_steps, sample)`
   from the `@torch.no_grad` `free_run` by factoring out ONLY the latent branch (decoder-free); `free_run`
   keeps its no_grad wrapper and calls the shared core. Test: under `no_grad` the refactor is bit-identical;
   `test_step0_matches_teacher_forced` + `test_gmm_cascade_rollout_point_vs_sampled` still pass.
2. **Config:** add a frozen `SchedConfig` nested under `PredictorConfig` (`predictor.sched.*`), all defaults
   = today's behavior (`rollout_enabled=False`, `rollout_steps=1`, `lambda_rollout=0.0`,
   `rollout_warmup_steps=2000`, `sample_rollout=False`, `feedback='latent'`, `detach_feedback_target=True`).
3. **Data (for K≥2):** render a T≥11 SO(3) train cache via `gen_data_so3.py`.
4. **Tests:** flag-default-off byte-identity of `compute_oneshot_loss`; zero-encoder-grad (backward of the
   rollout term alone leaves `tok.encoder` grads None/zero).

## Phased plan (cheapest-falsification-first)

| Phase | Goal | Gate (pre-registered) | Data |
|---|---|---|---|
| **P2.0** Plumbing + day-0 drift instrumentation | grad-capable unroll core + SchedConfig (no-op default) + in-training TF→AR & changed-region logging (latent probe in `validate`) | all tests green; byte-identity when off; `val/tf_ar_gap_mean` matches offline `eval_rollout.py` on a held ckpt | T=9 |
| **P2.1** K=1 grad self-feedback (cheapest falsification) | prove feeding back the model's own detached `head_mode` prediction moves the TF→AR gap *at all* | **EARLY KILL:** `train/sf_tf_ar_proxy` must trend down within ~500 steps of λ activating, else **reject the grad-unroll family → pivot to Fix #5**. Continue if `val/tf_ar_gap_mean` drops vs control AND recon within 0.3 dB of 28 | T=9 (none new) |
| **P2.2** K=2 compounding + Fix #4 (the ship phase) | train on compounding multi-step error + honestly supervise T-1 | **SHIP:** on `eval_rollout` PIXEL mode, SO(3) long: `tf_ar_gap_mean` drops ≥0.3 dB vs control (>2× predecessor's +0.151) AND `changed_psnr/pixel@h23` ≥0.3 dB vs control AND `beats_rotation_changed_until` +≥2 horizons AND recon within 0.3 dB AND logvar stable. **DECISIVE FALSIFICATION:** if latent gap closes but pixel@h23 does NOT move → the binding constraint is the multi-rate OFF-DISTRIBUTION DECODE (Fix #5), not latent exposure bias → P2 rejected | **T≥11** |
| **P2.3** sampled-GMM feedback (escalation only) | align training feedback with the deployed best-of-8 sampled rollout | sampled beats mode by ≥0.2 dB `changed_psnr/pixel@h23` without raising `grad_skip_rate`, else ship P2.2 (mode) | T≥11 |
| **P2.4** parent scheduled-sampling on the cross-attn KV (optional, composable) | attack the distinct pixel-rollout channel (child cross-attends a fully-predicted parent) — faithful second-pass variant only | with parent-SS on, `changed_psnr/pixel@h23` improves further over P2.2 at matched recon+single-step | T≥11 |

## The first experiment (do this first)

**P2.1 only** — one training run on the existing T=9 SO(3) cache, `rollout_enabled=True rollout_steps=1`
`feedback=latent head_mode`, λ warmed in after `pred_warmup`, vs an identical `rollout_enabled=False`
control. Cheapest possible falsification: K=1 needs **no new data** and **no native-rate bookkeeping**
(the single unrolled step is supervised against the same within-window `z_det` `pred_loss` already uses),
reuses the existing whole-stack latent feedback verbatim, ~1 extra cascade forward+back. It tests the
load-bearing precondition before spending anything: does a true grad-enabled exposure-bias term (not light
noise) move `sf_tf_ar_proxy` down within ~500 steps at matched recon? If flat → the whole grad-unroll
family is suspect for this architecture and we pivot to Fix #5 **without** building the longer cache, the
K>1 indexing, or the GMM-variance machinery.

## Risks (sharpest first)

1. **Binding-constraint mismatch (highest):** every P2 term lives in detached-LATENT space and never
   decodes during training, but the confirmed drift is measured in PIXEL feedback (decode→re-encode).
   Latent robustness may not transfer. → the ship gate is PIXEL `changed_psnr@h23`; if latent closes but
   pixel doesn't, the constraint is the multi-rate decode (Fix #5) and P2 is correctly *falsified*, not tuned.
2. **Data prereq unmet:** T=9 = window, so K≥2 would silently supervise T-1 against a within-window target
   and the "absorbs Fix #4" claim would be false. K=1 screens the family before paying for the T≥11 cache.
3. **Non-differentiable GMM sample + no GradScaler:** `head_sample`'s Categorical pick has no `rsample`;
   sampled-feedback grads are high-variance under SO(3) multimodality and the bf16 grad guard *silently
   skips* noisy steps. → lead with `head_mode`; escalate to sampled only in P2.3 behind a `grad_skip_rate` monitor.
4. **Variance-gaming optimum:** a latent-NLL rollout term can be satisfied by inflating logvar or shrinking
   output variance, degrading the sharp single-step TF the model relies on (detached encoder blocks ENCODER
   collapse, not this cascade-side optimum). → logvar-floor / entropy canary.
5. **Non-stationary detached target:** the encoder co-trains live, so `z_det` moves; a multi-step rollout vs
   a moving target is subtle. → warm λ_rollout in only after `pred_warmup`; watch recon every `val_every`.
6. **Multi-rate native-rate indexing bug (P2.2):** per-stage 9/5/3 target indexing on the longer clip is the
   most likely silent wrong-target bug. → a shape-assert test that K-step per-stage targets keep
   `topo.stage_temporal()` lengths. FVD must NOT gate (rank-deficient); gates rest on `changed_psnr@h23` +
   `tf_ar_gap_mean`.
