# E7c sampler/NFE robustness protocol

**Status:** frozen before dataset generation or evaluation.  
**Scientific role:** checkpoint-only robustness; not a new model-selection or confirmation study.

## Question and estimand

E7c asks whether four central LoopMom diffusion contrasts retain their registered direction when
the DDIM computation per four-frame rollout step changes and when rollout noise is resampled.
It does not reuse the one-time E2 confirmation set. A new purpose-locked `robustness_holdout` is
generated from the legacy pixel path with seed 20260715 and is never used for training, monitoring,
checkpoint selection, threshold selection, or extension of the factorial.

The four fixed contrasts are the 896-token reach pair, the 1,152-token interior pair, the
1,152-token spatial-versus-temporally-sparse package pair, and the 2,432-token flat inversion.
All use frozen seed-0 B16x30k checkpoints and the frozen seed-0 autoencoder.

## Factorial

For each of seven unique arms evaluate exactly five variants:

1. NFE 8, sampler seed 0;
2. NFE 16, sampler seed 0;
3. NFE 32, sampler seed 0;
4. NFE 16, sampler seed 1;
5. NFE 16, sampler seed 2.

Sampler seed 0 preserves the exact legacy common-random-number formula. Additional seeds use
disjoint million-seed blocks. Within a variant, compared arms receive identical random-number
coordinates. With 38 four-frame rollout steps, the per-clip inference allocations are 304, 608,
and 1,216 network evaluations at NFE 8, 16, and 32 respectively.

## Endpoints and fixed decision

The primary outcome is baseline minus candidate legacy foreground MSE on all aligned records;
positive favors the registered candidate. The 20 contrast-by-variant effects form one correlated
family and receive studentized max-statistic clip-cluster simultaneous 95% intervals.

- `ROBUST_CI_CLEAN`: every point estimate and simultaneous lower bound is positive.
- `DIRECTIONALLY_ROBUST`: every point estimate is positive, but at least one lower bound includes
  zero.
- `SAMPLER_OR_NFE_SENSITIVE`: any point estimate is zero or negative.

False-positive-sensitive complete metrics are corroborating and cannot override the primary
decision. No post-result NFE, sampler seed, checkpoint, metric threshold or contrast may be added.

## Operational and budget gate

The 35 immutable evaluations are expected to consume about 6.03 GPU-hours from measured E2
runtime. The hard E7c authorization is 8.0 project GPU-hours, including failed attempts and upload
time. The controller stops before another cell if remaining authorization is insufficient and
stops after any cell taking more than twice its NFE-scaled reference time. Every run must have a
content-addressed RunSpec, complete metrics, checkpoint/data/AE hashes, W&B cloud record, artifact
manifest, access-log entry and GPU-ledger row. No training occurs in E7c.

E7a remains separately gated. E7c results cannot alter E7a's cells or authorize an extension.
