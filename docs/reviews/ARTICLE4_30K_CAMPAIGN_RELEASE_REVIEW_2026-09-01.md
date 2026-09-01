# Critical release review: Article 4 30k campaign

**Review date:** 2026-09-01  
**Scope:** Path B Article 4 code, data flow, experiment matrix, recovery, logging, and scientific
interpretation  
**Verdict:** **GO for controlled execution after the final clean-commit/W&B preflight; no result
claim is pre-approved**

The clean online W&B/GPU/data preflight passed and created project
`leo27heady/hierarchical-video-prediction-article4-v2`; the scientific queue has not been started.

## Executive assessment

The redesigned campaign is materially stronger than the previous Chapter 4 experiment collection.
It has one causal question per family, exact token budgets, shared target horizons, physical position
identities, two fixed seeds, a final 30k endpoint, explicit cross-family checkpoint reuse, and a
machine-readable dependency rule for the two experiment families that cannot be specified before
A4-ALC finishes. The code can now execute the whole training and post-training workflow from one
restartable command.

The implementation is ready to produce valid evidence for a **TRB-specific hierarchy allocation
study**. It cannot guarantee in advance that hierarchy wins, that a 3--4x memory ratio will appear,
or that any result transfers to OpenSTL. Those are empirical outcomes, not release criteria. A null
or negative result must remain a valid campaign outcome.

## What is verified

### Experimental identity

- 43 unique configurations and 86 two-seed training runs are deduplicated correctly.
- Wave 1 contains 80 concrete, tracked, hash-authorized specs.
- A4-LVL and A4-OVR contribute exactly six new Wave 2 specs after development-only dependency
  resolution.
- All primary runs use 30k updates, seeds 0/1, width 256, depth 6, eight heads, batch 16, the same
  frozen codec, the same split, and the same four target positions.
- A4-FRT has only Flat/Linear/Geometric; A4-STS has only budget 1,792, as requested.

### Data and target correctness

- The development manifest is content-hashed and explicitly excludes every test artifact.
- Train and validation partitions are clip-disjoint and exhaustive.
- Standardization constants use only development-train clips.
- Allocation-invariant sampling produces identical clips, starts, target frames, and actions for
  identical seeds.
- Every context time is earlier than the first target; targets are four consecutive frames.
- Fine gathers preserve exact physical cell identities and avoid a dense temporal fine tensor.
- The prior action/pixel audit checked 4,096 transitions and 15,844,864 overlapping pixels with zero
  mismatch.

### Architecture and position correctness

- All allocations share a 96-frame context coordinate system and target positions 96--99.
- Coarse, fine, and optional top levels have distinct level identities; fine cells keep physical
  spatial IDs.
- The level-3 path consumes real top latents and is context-only at the top level.
- Exclusive support uses disjoint physical times; residual support subtracts the same-time
  coarse-parent projection and is therefore behaviorally distinct from nested support.
- CUDA forward/backward checks passed for level 3, exclusive, and residual paths at exact 1,792
  context tokens and 2,112 total attention tokens.

### Runtime and recovery

- The hot path uses CUDA bf16 autocast, fp32 masters, fused scaled-dot-product attention, a
  dtype-compatible RMSNorm path, CPU-resident latent storage, and direct packed transfer.
- Representative GPU worker smokes passed at budgets 320 and 3,072 and for A4-ALC, A4-CEX, and
  A4-STS. The largest tested representative used about 5.97 GiB peak allocated VRAM; dependent
  1,792-token paths used about 1.08 GiB in their bounded checker.
- TRB encoding costs approximately 13--14 seconds in current smokes. It is not a material fraction
  of a multi-hour run and does not justify a persistent latent cache.
- Atomic resume includes model, EMA, optimizer, CPU/CUDA RNG, and sampler RNG. A verified smoke
  restored step 2, continued to step 4, and retained exactly one training and validation record for
  each completed point.
- Completed hashes are skipped; active duplicate PIDs are rejected; stale locks remain as evidence.

### Evidence and observability

- Local manifests, JSONL histories, checkpoints, panels, raw panel arrays, logs, status transitions,
  and hashes are canonical.
- W&B receives train, validation, quality, timing, memory, throughput, system, and image evidence
  with deterministic resumable IDs.
- W&B failure is nonfatal and explicitly marked; local replay is available.
- Post-training selection is locked before fresh holdout generation. Confirmation auditing checks
  hashes, dynamics, uniqueness, and collision absence.

## Important limitations and interpretation boundaries

1. **Two seeds support a focused resource-constrained comparison, not strong distributional
   certainty.** Show both seed points and paired means. Do not manufacture narrow 95% intervals
   from two seeds or imply a population estimate that the design cannot support.
2. **Token-state savings and GPU peak memory are different quantities.** Hierarchy can reduce stored
   contextual state while joint-attention activations and fixed target tokens dominate peak training
   memory. Report context payload, metadata, total sequence length, peak allocated/reserved CUDA
   memory, and quality together. A claimed “3x memory win” is allowed only if the measured quantity
   is named and the corresponding ratio is observed.
3. **The level-3 comparison is not exact parameter parity.** It adds 1,280 positional/level
   parameters. This is negligible relative to 7.56M but must appear in the methods/limitations.
4. **Residual overlap is an embedding-space residual, not a decoded-pixel Laplacian pyramid.** Its
   interpretation should remain bounded to the implemented projection difference.
5. **A4-MEC attention values are descriptive.** Entropy and source-level mass can help explain a
   result but cannot establish causal mediation.
6. **The sealed evaluation is a fixed four-frame prediction at the end of 200-frame clips.** It
   measures re-entry-conditioned performance under long observed context, not a 100-step
   autoregressive rollout. Do not describe it as long-horizon rollout stability.
7. **TRB is intentionally synthetic and mechanistic.** Article 4 can establish hierarchy-allocation
   behavior on a controlled unequal-information environment. Competitiveness on natural or OpenSTL
   data belongs to Article 5.
8. **30k no longer fits the original one-GPU one-week promise.** The direct estimate is 187--205
   GPU-hours; operational reserve is 205--234. A strict seven-day deadline requires another
   comparable GPU or reduced scope, neither of which is assumed here.
9. **No superiority outcome is guaranteed.** The early A4-COR ordering reversal is direct warning
   that architecture rankings can change with optimization time. The campaign is designed to make
   a null or adverse finding interpretable rather than to force a positive headline.

## Release recommendation

Proceed with the final clean-commit preflight and then launch only through
`scripts/v2/run_article4_campaign.py run --execute`. Do not start individual run specs manually,
change a JSON file after preflight, inspect the fresh holdout before the development selection
ledger exists, or merge legacy Chapter 4 values into the new matrix.

For dissertation construction, retain Chapters 1 and 2 semantically for now, rebuild Chapter 3
around this controlled hierarchy study, and make the integrated Article 5 model a separate Chapter
4 only if its two-dataset gates stand out. That conditional migration decision remains appropriate;
Article 4 implementation readiness is not evidence that the integrated Article 5 hypothesis will
pass.

The operational companion is
`docs/plans/ARTICLE4_30K_CAMPAIGN_IMPLEMENTATION_AND_OPERATIONS_2026-09-01.md`.
