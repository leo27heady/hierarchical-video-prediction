# E7a alternate-codec robustness protocol

**Status:** frozen before alternate-codec training, predictor training, or holdout generation.
**Scientific role:** representation-seed robustness; not model selection, a repair of E7c, or a second E2 confirmation.

## Question and intervention

E7a asks whether four central LoopMom diffusion contrasts retain their direction when the learned
two-level autoencoder is independently initialized and trained. The alternate codec uses seed 1;
architecture, 300,000-frame training array, 7,000 updates, batch 64, Adam learning rate 0.0003,
OneCycle schedule, random-level objective, and coarse loss multiplier remain exactly fixed. Each
of seven predictors is then retrained from predictor seed 0 in that new latent coordinate system.

This is a whole-codec-seed intervention. It includes the resulting encoder/decoder weights,
latent geometry, codec-specific train-set normalization constants, and the predictors trained in
that geometry. It does not identify which one of those representation properties causes any
change. It is also conditional on predictor seed 0; prior experiments, rather than E7a, carry the
project's training-seed evidence.

## Fixed matrix

The seven alternate-codec predictor arms are `hier-min`, `flat-min`, `hier-long`, `flat-budget`,
`strided-flat`, `hier-c2`, and `flat-c2`, all seed 0, B16x30k, BF16, AdamW 0.0003, warmup 2,000,
EMA 0.999. Training uses the unchanged H3b-parity kernel and the existing development grid only;
it performs no terminal development rollout. Exactly four contrasts are registered:

1. 896-token lower frontier: `hier-min` versus `flat-min`;
2. 1,152-token interior: `hier-long` versus `flat-budget`;
3. 1,152-token package shape: `strided-flat` versus `hier-long`;
4. 2,432-token upper inversion: `flat-c2` versus `hier-c2`.

After all training is complete, the seven original-codec seed-0 checkpoints and seven
alternate-codec checkpoints are evaluated at NFE 16, sampler seed 0, on the same independently
generated purpose-locked `codec_robustness_holdout`. Thus the original-codec rows are a
same-population control; E7a never compares effects measured on different datasets.

## Codec floor before predictors

Before any predictor training or holdout access, both codecs are evaluated on the final fixed
4,000 frames of the training array, which `FramePool` excludes from codec training. Alternate
fine/coarse full-frame and foreground reconstruction MSE must remain within the frozen relative
plus absolute limits in the JSON design. All outputs must be finite and every coarse and fine
latent channel must have standard deviation at least 0.005. Failure stops the study without a
retry, third codec seed, predictor training, or locked-data access.

The floor is a validity screen, not codec selection. Passing says the alternate representation is
usable under the same reconstruction objective; it does not require identical reconstructions or
latent axes.

## Endpoints and decision

Primary effects are baseline minus candidate legacy foreground MSE over all aligned records;
positive favors the registered candidate. The eight codec-by-contrast effects receive one
studentized max-statistic clip-cluster simultaneous 95% family. The original-codec effects are
necessary holdout controls because E7c showed that the 2,432-token cell can be family-wise weak.

- `HOLDOUT_CONTROL_FAILURE`: any original-codec point estimate is non-positive. The new
  population does not reproduce the whole control family, so codec attribution is ambiguous.
- `CODEC_ROBUST_CI_CLEAN`: controls are positive and all alternate-codec estimates and
  simultaneous lower bounds are positive.
- `CODEC_DIRECTIONALLY_ROBUST`: controls and alternate-codec point estimates are positive, but
  at least one alternate-codec lower bound includes zero.
- `CODEC_SENSITIVE`: controls are positive and any alternate-codec point estimate is
  non-positive.

Complete false-positive-sensitive metrics are corroborating. Differences between alternate and
original effect sizes are descriptive diagnostics, not an extra significance family and not a
decision override. No post-result cell, codec, seed, threshold, endpoint, or retry may be added.

## Data isolation and operational budget

The new 500x200 set uses seed 20260716 and the legacy LoopMom pixel path. It is collision-checked
against development, E2, and E7c hashes and cannot be opened until all alternate-codec training
and the codec floor are complete. Every access is logged before hashing/opening the array.

Expected cost is about 31.2 project GPU-hours. The hard maximum is 36.0 hours: 0.6 for codec
training/floor, 31.4 allocated across seven predictor trainings, and 3.3 for fourteen evaluations.
Accounting includes failed attempts, resume overhead, gates and upload time while a project
process owns the GPU. The controller stops before a cell whose allocation cannot fit the
remaining authorization. W&B online logging and content-addressed local artifacts are mandatory.
