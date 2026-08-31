# Article 4: one-week hierarchy experiment plan

## 1. Objective

Establish a clean, reproducible account of how spatial and temporal hierarchy allocate a fixed memory budget, compared with flat context, and how the allocation changes quality, temporal reach, and measured inference state.

The controlled benchmark should be named **Toroidal Re-entry Benchmark (TRB)** in prose. Existing `LoopMom` paths remain as legacy artifact identifiers. TRB is a synthetic, action-conditioned controlled benchmark; it must not be presented as evidence of general natural-video superiority.

## 2. Primary questions

1. At equal token budget, which allocation across hierarchy levels gives the best quality?
2. At equal temporal reach and target protocol, how much retained state does hierarchy save versus a flat fine-token context?
3. With a fixed fine-detail quota, how valuable is expansion of the coarse temporal history?
4. Does nested temporal overlap help, or are redundant coarse tokens wasteful when fine tokens cover the same time point?
5. How do spatial detail quota and temporal retention trade against one another under one global budget?
6. Are conclusions stable over training seeds rather than only clip resampling?

## 3. Non-negotiable protocol

- One immutable train/validation/test split manifest.
- Validation only for checkpoint selection, early stopping, and configuration selection.
- Test evaluated once for locked confirmation configurations.
- Four training seeds for primary claims: `0, 1, 2, 3`.
- Seed `0` may be used for screening, but screening results are labeled exploratory.
- Fixed prediction target, target stride, loss, optimizer, schedule, batch protocol, embedding width, and parameter-matching rule within each comparison family.
- Every run has an experiment ID and full config dump.
- Report mean, standard deviation, and preferably a seed-level interval. Clip bootstrap intervals are allowed only as conditional within-model diagnostics.
- Memory metrics include analytic token count, serialized retained-state bytes, CUDA allocated peak, CUDA reserved peak, and end-to-end wall time. State clearly whether parameters, optimizer state, gradients, decoder, and dataloader are included.

## 4. Naming convention

```text
A4_<family>_<budget>_<levels>_<allocation>_<overlap>_<seed>
```

Examples:

- `A4_FRT_B1152_L1_FLAT_EXCL_S00`
- `A4_FRT_B1152_L2_GEOM_NEST_S03`
- `A4_STS_B1792_L2_K16R64_EXCL_S01`

`FRT` means frontier, `ALC` allocation, `CEX` coarse expansion, `LVL` level count, `OVR` overlap, and `STS` spatial--temporal surface.

## 5. Experiment series

### A4-COR: correctness and protocol oracles

Purpose: ensure data generation, causal masks, token accounting, target alignment, checkpoint selection, and memory meters are valid.

Runs/tests:

- Verify TRB toroidal re-entry, action labels, deterministic seed replay, temporal indexing, and no split overlap.
- Verify that changing context stride cannot change target stride or target token count.
- Compare analytic token count with serialized state shape and measured state bytes.
- Overfit a tiny sample with flat and hierarchical configurations.
- Confirm causal invariance by perturbing future inputs.
- Run identical config twice with the same seed and quantify deterministic tolerance.

Estimate: 0.5 day engineering; 2--4 GPU-hours. Passing A4-COR is required for all other series.

### A4-FRT: ten-point budget frontier

Purpose: answer the user's requested low-to-high context study.

Budgets: ten predeclared points spanning the feasible range, provisionally `512, 640, 768, 896, 1152, 1408, 1792, 2176, 2432, 3072` tokens. The exact endpoints may be adjusted once by the profiling pilot, then frozen.

Policies at every budget:

- flat fine-token context;
- two-level balanced allocation;
- two-level linear-by-age allocation;
- two-level geometric/exponential-by-age allocation.

Stage 1: all 40 cells at seed 0, short common training budget.
Stage 2: select five informative budgets and the flat plus two best hierarchy policies. Confirm 15 cells with seeds 0--3; seed 0 may reuse the qualifying screening run only if its protocol is identical.

Primary outputs:

- quality versus retained-state bytes, one panel per metric;
- reachable history versus retained-state bytes;
- policy allocation diagram, conceptual only;
- one compact result table containing the seed-confirmed primary frontier points.

Estimate: 40 screening runs x 0.5--1.0 hour plus 45 additional confirmation runs x 0.75--1.5 hours = 54--108 GPU-hours. With one GPU, some confirmation cells may continue beyond the calendar week; with two GPUs this series fits comfortably.

### A4-ALC: same budget, different level distribution

Purpose: isolate allocation from budget.

At budgets `1152` and `1792`, compare at least:

- flat;
- fine-heavy (`75/25` fine/coarse payload);
- balanced (`50/50`);
- coarse-heavy (`25/75`);
- linear-by-age;
- geometric-by-age;
- reach-optimized allocation.

Use seed 0 screening for all cells. Confirm the two strongest hierarchy allocations and flat with four seeds at each budget. If these configurations are already present in A4-FRT, reference the same immutable runs rather than retraining.

Estimate: 14 screening cells plus at most 18 confirmation cells, 14--42 GPU-hours before deduplication.

### A4-CEX: fixed detail, expanding coarse context

Purpose: test whether coarse history adds useful reach when recent detail capacity is held constant.

Keep the fine window and fine tokens fixed; add coarse slots in monotonic steps, for example `0, 4, 8, 16, 32, 64` coarse time positions. Match all other settings. Report both quality and marginal gain per added kilobyte.

Estimate: 6 seed-0 screens; confirm baseline plus two knees with four seeds = 9 additional runs; 10--24 GPU-hours.

### A4-LVL: number of hierarchy levels

Purpose: distinguish the effect of hierarchy depth from token budget.

Compare one, two, and three levels at budgets `1152` and `1792`, with equal total tokens and matched effective reach where possible. A three-level system is accepted only if its actual allocation and decoding semantics are fully specified.

Estimate: 6 screens plus 12 confirmation runs for selected contrasts; 10--27 GPU-hours.

### A4-OVR: overlapping versus exclusive time support

Purpose: test whether retaining a coarse token for a time point already represented at the fine level is useful.

- `nested`: recent times appear at coarse and fine levels;
- `exclusive`: fine representation replaces the corresponding coarse token;
- optional `residual`: coarse base plus fine residual, both retained deliberately.

Compare at equal total token budget, equal target, and as close as possible to equal reach. Confirm at one medium and one high budget if the seed-0 effect is material.

Estimate: 6 screens plus up to 12 confirmation runs; 8--24 GPU-hours.

### A4-STS: spatial-detail versus temporal-reach surface

Purpose: explicitly test the dissertation's central allocation interpretation.

Use spatial detail quotas such as `K = 64, 32, 16, 8` fine cells per represented time point. Spend saved tokens on longer coarse or mixed-resolution temporal history while keeping the global token budget fixed. At minimum, test two global budgets and four spatial quotas, with three temporal policies:

- recent-detail priority;
- uniform temporal coverage;
- geometric age allocation.

This is an allocation experiment, not learned routing. The selected spatial cells may be determined by a fixed oracle/variance rule only if that rule is applied consistently and its oracle status is explicit.

Stage 1: 24 seed-0 cells.
Stage 2: confirm flat/full-detail, the best memory-efficient point, and one balanced knee with four seeds at one primary budget.

Outputs:

- a heat map or small-multiple frontier with exactly one role: quality across spatial quota and temporal reach;
- a table of the three locked confirmation points;
- a decomposition of saved spatial tokens into purchased temporal reach.

Estimate: 24 screens plus 9 additional confirmation runs; 20--45 GPU-hours.

### A4-MEC: mechanism diagnostics

Purpose: retain useful diagnostics without overloading primary result figures.

- head entropy and temporal coverage;
- identity/coverage under re-entry;
- per-level utilization;
- attention or selection maps for a small fixed qualitative set.

These are explanatory diagnostics, not separate superiority claims. Run only on the locked flat, best hierarchy, and one failure case.

Estimate: 0.5 day analysis; little or no retraining.

## 6. One-week calendar

| Day | Work | Exit criterion |
|---|---|---|
| 1 | A4-COR; freeze splits, budgets, config schema, and meters | all correctness oracles pass |
| 2 | A4-FRT screening; begin A4-ALC deduplicated screen | 10-point curves available for seed 0 |
| 3 | A4-CEX, A4-LVL, A4-OVR screening | primary contrasts and failure cases identified |
| 4 | A4-STS screening | spatial--temporal allocation surface available |
| 5 | launch four-seed primary confirmation | no protocol change after launch |
| 6 | continue confirmation; run A4-MEC on locked models | seed table and mechanism diagnostics complete |
| 7 | test-set evaluation for locked configs; generate atomic figures/tables | gate decision and reproducibility bundle complete |

This schedule assumes roughly 1-hour screening and 1--1.5-hour confirmation runs and at least two comparable GPUs, or aggressive successive-halving on one GPU. On one GPU, prioritize A4-COR, A4-FRT, A4-STS, and four-seed confirmation of 6--9 primary configurations; defer secondary confirmations.

### 6.1 Executable queue on the local one-GPU host

The inspected machine has one RTX 4060 Ti with 16 GB VRAM. The week has 168 theoretical GPU-hours, but 105--125 usable training/profile hours is a safer planning capacity after debugging, evaluation, analysis, and failed jobs.

The mandatory one-week queue is therefore:

| Queue | Unique work | Time assumption | GPU-hours |
|---|---:|---:|---:|
| A4-COR | correctness/oracle checks and tiny overfits | bounded | 2--4 |
| A4-FRT screen | 10 budgets x 4 policies x seed 0 | 30--60 min | 20--40 |
| targeted mechanism screen | 6 coarse-expansion, 3 level-count, 3 overlap, 8 spatial--temporal cells | 30--60 min | 10--20 |
| primary confirmation | 6 locked configurations x 4 seeds | 1--2 h | 24--48 |
| repeated memory profiles | locked configurations, 5 repeats | short | 3--6 |
| **Total** | **60 screening + 24 confirmation fits** | | **59--118** |

The six locked configurations should cover three comparisons with maximum reuse: low/middle/high flat versus the best predeclared hierarchy policy. One hierarchy point is also chosen to serve the overlap/level or spatial--temporal contrast where possible. The other mechanism curves remain exploratory during the week.

If the first five representative runs exceed two hours, immediately reduce the screen to six budget points while retaining the low, previous 896/1,152 region, 1,792 region, and high-budget endpoint. Do not reduce the four confirmation seeds. If median confirmation runtime exceeds two hours, the one-week deliverable becomes a gate report rather than a completed Article 4 result package.

### 6.2 Completion queue after the one-week sprint

To conclude every planned Article 4 claim rather than only the primary frontier:

- confirm the full selected 8--12-arm design with four seeds;
- complete two-budget A4-ALC, A4-LVL, and A4-OVR confirmations;
- complete the two-budget 4 x 3 A4-STS surface and confirm its three representative points;
- run the bounded alternate codec and predictor-head robustness;
- regenerate final test-set tables and figures only after configuration lock.

Expected additional work after a successful sprint: 90--250 GPU-hours, 3--7 calendar days of unattended local training plus 3--5 researcher days for analysis and writing. The broader 200--450 GPU-hour estimate remains appropriate if screening runs must use the full training schedule or if more than 12 arms survive.

## 7. Minimum publishable package

The minimum Article 4 result set is:

- four-seed flat versus hierarchical comparisons at five budgets or at least three budgets plus two equal-reach controls;
- an equal-reach retained-state comparison;
- a fixed-detail coarse-expansion curve;
- nested versus exclusive overlap at one or two budgets;
- a spatial-detail versus temporal-reach surface;
- a seed-level table and fully specified configurations;
- a single unambiguous primary metric plus clearly labeled secondary diagnostics.

If a full 10-point four-seed frontier is required, expect approximately 160 model fits for four policies and 120--240 GPU-hours at the assumed durations. It is not the recommended use of a one-week window; screen all points and confirm the knees/endpoints.

At the end of the one-week local sprint, the work can support a strong go/no-go decision and a seed-confirmed primary frontier if runtimes match the pilot. It should not yet be called the final Article 4 package unless the completion queue is also finished.

## 8. Article and dissertation figures

Recommended main figures:

1. Architecture and allocation schema: levels, token budgets, and temporal support.
2. Ten-point quality--memory frontier.
3. Equal-reach memory comparison.
4. Spatial-detail/temporal-reach allocation surface.
5. Overlap comparison or coarse-expansion curve, whichever answers the stronger mechanistic question.

Recommended main tables:

1. Dataset, split, training, and hardware protocol.
2. Exact named configurations and token accounting.
3. Four-seed locked results.

All other diagnostics belong in supplementary material or the dissertation appendix.

## 9. Stop rules

- Stop and repair if validation/test separation fails.
- Stop a policy after the shared screening budget if it is dominated at every point and exhibits no mechanistic value.
- Do not promote a one-seed result to a primary claim.
- If hierarchy has no quality--memory advantage, publish the negative allocation result honestly and do not build Article 5 around an unsupported hierarchy configuration.
