# A4-COR execution status

**Status:** deterministic/data/packing/trainability preflight passed; two 30k schedule calibrations
authorized but not yet counted as complete.

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

## Pending gates

1. Run both authorized calibrations through 30,000 updates.
2. Validate every 2,000 updates and compare 8k, 12k, 16k, 20k, 24k, and 30k.
3. Apply the predeclared 2% plateau and ranking-stability rule. Do not force an 8k or 12k result.
4. Freeze the selected final-matrix update count and revise the one-week timing table from measured
   runtime.

Measured optimized throughput before the shared-anchor correction was 0.286 s/update at the same
2,112-token attention sequence. The anchor changes only temporal IDs, not sequence length or tensor
shapes. Expected wall time is therefore approximately 2.4 GPU-hours of updates plus validation and
checkpoint overhead per arm, or about 5--6 GPU-hours total. Treat this as planning evidence until
the run manifests report actual runtime.
