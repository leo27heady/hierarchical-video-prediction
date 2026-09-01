# A4-COR execution status

**Status:** all preflight gates and both 30k schedule calibrations completed; 30,000 updates are
recommended, but the schedule is not frozen until Leonid confirms it.

## Locked paired calibration

| Arm | Config hash | Role |
|---|---|---|
| `A4_COR_B1792_L1_FLAT_RECENT_EXCL_S00_T30K` | `15088e9c0c9da5e4d6616e7136c4fc9368c8d949559072454b33ca8473c09c7a` | flat recent control |
| `A4_COR_B1792_L2_BAL_RECENT_NEST_S00_T30K` | `c5ba8de728ce16340c25626ebab49dc4d0377fd8c20d5220086b44e8333024ad` | balanced hierarchy |

Both use seed 0, budget 1,792, the same initialized 100-position temporal table, first-target
anchor 96, target frames 96--99, width 256, depth 6, eight heads, batch 16, diffusion
v-prediction, bf16 autocast/fp32 masters, and the same fixed validation clips and sampler seeds.
The only intended difference is context allocation.

## Passed gates

1. The manifest and autoencoder hashes resolve to the registered bytes.
2. The 2,700/300 development split is clip-disjoint; no test split is present.
3. Across 4,096 sampled transitions and 15,844,864 overlapping pixels, action-derived spatial
   shifts had zero mismatches. The observed vocabulary is exactly `{-1,0,1}`.
4. Unit tests prove exact selected-cell gather, packed-token counts, serialized-state accounting,
   target/context disjointness, allocation-invariant targets, local/absolute temporal-ID separation,
   gapped coarse histories, and equal flat/hierarchical parameter counts.
5. Real-data two-update smoke runs passed for both packed paths at approximately 3.79 GiB peak
   allocated CUDA memory.
6. Repeated-batch 200-update overfit gates passed with equal 201,824-parameter compact models:
   flat fixed-noise loss ratio 0.613 and balanced-hierarchy ratio 0.616. Seeded loss replay was
   exact within each arm.
7. The complete repository suite passed: 180 tests.

The machine-readable preflight is
`artifacts/v2/correctness/article4_preflight_2026-09-01.json`.

## Calibration result

Both runs completed with all 15 validation milestones and checkpoint hashes verified. The flat arm
ran for 2.25 hours and the balanced-hierarchy arm for 2.39 hours; each used about 3.79 GiB peak
allocated CUDA memory.

The original 8k and 12k choices fail. On horizon-4 foreground MSE, the hierarchy-to-flat ordering
changes between 12k and 16k, and at 24k the balanced arm remains 6.80% above its best later value.
Only 30k satisfies the paired 2% and ranking-stability rule. Because the original protocol did not
unambiguously name one primary metric, the analysis also applies the rule independently to all six
recorded validation metrics; every metric selects 30k. The worst final-to-best gap in the 24k--30k
late window is 1.07%.

The reproducible analysis is `docs/experiments/v2/A4_SCHEDULE_CALIBRATION_2026-09-01.md`; its
machine-readable result is `artifacts/v2/analysis/article4_schedule_calibration_2026-09-01.json`.

## Pending confirmation

1. Leonid must confirm or reject the recommended 30,000-update final schedule.
2. Leonid must choose between the full 30k matrix (about 265--295 GPU-hours on this host), a staged
   one-week screen with only selected configurations extended to 30k, or added comparable GPUs.
3. Only after confirmation may the full config set, campaign launcher, and new W&B project be
   created or any further experiment be launched.

These calibrations are schedule evidence only. They remain seed-0 development runs, do not establish
a hierarchy advantage, and did not access a test split.
