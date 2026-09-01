# Article 4 schedule-calibration analysis

**Decision status:** recommendation awaiting Leonid's confirmation.

Operational headline endpoint: `horizon4_foreground_mse` (lower is better). Registered tolerance: 2%.
The pre-run documents named the required metrics but did not explicitly designate one as
primary. To avoid a post-hoc metric choice, the recommendation is also required to agree
across every recorded validation metric; all six independently select 30k.

| Updates | Flat H4 FG-MSE | Hierarchy H4 FG-MSE | Flat gap to best later | Hierarchy gap to best later | Ranking stable | Pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 8,000 | 0.046516 | 0.042223 | 110.01% | 66.41% | no | no |
| 12,000 | 0.033052 | 0.029133 | 49.22% | 14.82% | no | no |
| 16,000 | 0.026283 | 0.028735 | 18.66% | 13.25% | yes | no |
| 20,000 | 0.023915 | 0.027803 | 7.97% | 9.57% | yes | no |
| 24,000 | 0.022439 | 0.027099 | 1.30% | 6.80% | yes | no |
| 30,000 | 0.022150 | 0.025646 | 0.00% | 0.00% | yes | yes |

## Recommendation: 30,000 updates

The original 8k and 12k options fail decisively. The decoded foreground endpoint continues
to improve, and the flat-versus-hierarchy ordering reverses between 12k and 16k. At 24k,
the flat arm is within 2% of its best later value, but the balanced hierarchy remains more
than 2% away. The 30k endpoint is within 2% of each arm's best late milestone and preserves
the post-16k ordering. Therefore 30k is the earliest registered candidate satisfying the
paired rule. This is a schedule recommendation, not confirmation of a hierarchy advantage.
At 30k, each endpoint is also within 2% of its arm's best value in the 24k--30k late
window (worst final gap: 1.07%).

The result also invalidates the original one-GPU/one-week full-matrix estimate. Running every
agreed two-seed cell for 30k requires a revised staged scope, more wall time, or additional
comparable GPUs. That execution choice must be confirmed before the campaign is generated.

Figure: `figures/v2/article4_schedule_calibration_2026-09-01.png`

## Provenance

- `A4_COR_B1792_L1_FLAT_RECENT_EXCL_S00_T30K`: config `15088e9c0c9da5e4d6616e7136c4fc9368c8d949559072454b33ca8473c09c7a`, commit `2e3b46d007db1d70e377c481efd1c7251b81c964`, runtime 2.25 h, peak allocated 3.79 GiB, metrics SHA-256 `d4f897831a74e85387b7aee3101463e9acb57a355766f526a2fb61414d75ba37`.
- `A4_COR_B1792_L2_BAL_RECENT_NEST_S00_T30K`: config `c5ba8de728ce16340c25626ebab49dc4d0377fd8c20d5220086b44e8333024ad`, commit `2e3b46d007db1d70e377c481efd1c7251b81c964`, runtime 2.39 h, peak allocated 3.79 GiB, metrics SHA-256 `116de2a709650ea33da0adf4e5750ba665efe0531a0b5093a1092f0c92911a43`.

No test split was accessed, and these seed-0 calibration runs are not final two-seed Article 4 results.
