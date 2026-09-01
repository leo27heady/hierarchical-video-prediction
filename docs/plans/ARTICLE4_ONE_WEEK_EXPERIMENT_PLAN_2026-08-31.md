# Article 4: revised one-week hierarchy experiment plan

**Revision:** 2026-09-01

**Status:** paired convergence gate complete; 30k recommended, awaiting user confirmation and
revised timing/scope decision

**Dataset:** Toroidal Re-entry Benchmark (TRB) only
**Training seeds:** `0, 1`

## 1. Objective and claim boundary

Establish a controlled account of how a fixed context-token budget can be allocated across
spatial detail, temporal reach, and hierarchy levels. The comparison is against a flat recent
fine-token context under an invariant target and training protocol.

TRB is a synthetic, action-conditioned mechanism benchmark. It supports causal statements about
this controlled hierarchy design; it does not establish natural-video or general OpenSTL
superiority. Long Moving MNIST is removed from Article 4.

The principal outputs are:

1. quality versus retained-state memory across six budgets;
2. equal-budget allocation effects at 1,792 tokens;
3. the marginal value of expanding coarse history at fixed recent detail;
4. level-count and overlap effects at 1,792 tokens;
5. a two-budget spatial-detail/temporal-coverage surface.

No memory ratio is promised in advance. A 3--4x retained-state reduction is a stretch hypothesis
for equal-reach coarse-versus-flat packages, not an acceptance assumption.

## 2. Primary questions

1. At equal token budget, which hierarchy allocation gives the best predictive quality?
2. At equal temporal reach, how much retained state does hierarchy save relative to flat fine
   context?
3. With recent fine detail fixed, when does adding coarse history stop helping?
4. Does a third level add value after total tokens and the allocation rule are held fixed?
5. Is redundant nested overlap useful, or should fine tokens replace coarse tokens at the same
   time point?
6. When the fine spatial quota decreases, can the saved payload provide useful temporal detail
   coverage?
7. Are the direction and practical size of the effects reproduced by both training seeds?

## 3. Frozen protocol

### 3.1 Data and split policy

- Use only `loopmom_train_3000x100.npy` and its aligned actions as the TRB development corpus.
- Use the hash-pinned seeded clip split in
  `experiments/v2/manifests/trb_v2_development.manifest.json`: 2,700 train and 300 validation clips.
- The legacy file named `loopmom_test_500x200.npy` is excluded from Path B selection and claims
  because it has already been used during model development.
- Generate a fresh 200-frame TRB confirmation set, including actions, positions, object metadata,
  generator revision, and seed, only after all configurations and selection rules are locked.
- Validation chooses checkpoints and configurations. The new test set is read once for locked
  configurations.

The development artifact's original generator seed was not recorded. This limitation is
preserved in its manifest; exact clip/action SHA-256 hashes now identify the data actually used.

### 3.2 Replication and inference

- Every final matrix cell is trained with seeds `0` and `1`; there is no one-seed result promoted
  as a primary result.
- With only two training seeds, report both points, their mean, and the paired seed-wise effect.
  Do not report a seed-level 95% confidence interval or imply a precise population variance.
- Clip bootstrap intervals may be reported only as within-model/within-seed conditional
  uncertainty. They do not replace training-seed replication.
- A third seed is a post-week robustness option for 6--8 locked primary configurations, not part
  of the approved one-week matrix.

### 3.3 Proposed controlled instrument and fixed prediction head

The current 384-wide, eight-block historical model is too slow for the full matrix on one local
GPU. Use the following smaller common instrument if A4-COR establishes convergence and ranking
stability:

| Field | Proposed value |
|---|---:|
| predictor width | 256 |
| blocks | 6 |
| attention heads | 8 |
| parameters | approximately 7.56 M; exactly matched across one- and two-level allocation arms |
| temporal embedding | one shared learned table of 100 TRB frame positions for every arm |
| precision | bf16 autocast with fp32 optimizer/master weights |
| batch size | 16 at all six budgets |
| prediction target | four future fine frames; invariant across context arms |
| target-time sampling | common first-target anchor 96; target frames 96--99 in every arm |
| schedule evidence | 8k--24k rejected; 30k recommended by A4-COR, awaiting confirmation |
| checkpoint selection | validation only, shared evaluation cadence |

The primary predictor head is latent diffusion with v-prediction and a deterministic DDIM sampler,
matching the current Article 4 architecture. Direct regression is a bounded robustness/control
head only; it is not selected separately per allocation.

This is an experimental instrument, not a silent replacement claim for the historical 22 M
parameter predictor. After the matrix, repeat 4--6 decisive configurations with the historical
384x8 instrument as a bounded capacity-robustness check if time permits.

### 3.4 Memory endpoints

For every locked configuration record separately:

- analytic context-token payload;
- serialized retained-state bytes, including token metadata;
- inference peak allocated and reserved CUDA memory;
- training peak allocated and reserved CUDA memory;
- parameters, gradients, optimizer state, activations, decoder, and dataloader inclusion flags;
- update time, end-to-end runtime, and evaluation runtime.

Ratios with different denominators are never multiplied.

## 4. Allocation definitions

### 4.1 Common representation

- A coarse TRB latent time position costs 16 tokens.
- A complete fine latent time position costs 64 tokens.
- A partial fine package contains selected physical cells from the 8x8 fine grid. Cell identity
  and absolute temporal position are carried separately from packed tensor slot.
- Fine-cell selection in Article 4 is deterministic and pixel-independent. It uses registered,
  nested, phase-cycled spatial lattices for quotas 8, 16, 32, and 64. Learned/content-dependent
  routing belongs to Article 5.

### 4.2 Four frontier policies

At budget `B`, the flat arm retains `B/64` complete recent fine positions. Each hierarchical
frontier arm assigns exactly half the payload to a contiguous coarse base and half to fine cells:
`N_c = B/32` coarse positions and `K_f = B/2` fine cells. The three policies differ only in where
the fine cells occur within the coarse temporal support:

- **Balanced recent:** fill the newest times first, using all 64 cells before moving older.
- **Linear age:** fine-cell quota decreases linearly with age; deterministic integer rounding
  preserves exactly `K_f` cells.
- **Geometric age:** fine-cell quota follows a predeclared geometric decay with age; deterministic
  largest-remainder rounding preserves exactly `K_f` cells.

The decay parameter is frozen by A4-COR, not fitted separately at each budget.

### 4.3 Seven 1,792-token allocation policies

| Paper label | Exact payload definition |
|---|---|
| Flat recent | 28 complete recent fine positions |
| Fine-heavy | 28 coarse positions + 21 complete recent fine positions |
| Balanced recent | 56 coarse positions + 896 newest-first fine cells |
| Coarse-heavy | 84 coarse positions + 7 complete recent fine positions |
| Linear age | 56 coarse positions + 896 linearly age-weighted fine cells |
| Geometric age | 56 coarse positions + 896 geometrically age-weighted fine cells |
| Reach-oriented | 96 coarse positions + 4 complete recent fine positions |

All seven total exactly 1,792 context tokens. ALC uses nested support; A4-OVR later changes only
the overlap rule.

## 5. Experiment series

### A4-COR -- correctness, convergence, and throughput oracles

Required checks:

- **DATA:** shapes/dtypes, action vocabulary, train/validation disjointness, source hashes, exact
  action-to-pixel shift on sampled transitions, and deterministic generator replay on fixtures;
- **INDEX:** local fine-frame gather indices are distinct from absolute temporal conditioning
  indices inside a longer coarse history;
- **TARGET:** context stride/allocation cannot change target horizon, target stride, or target
  token count;
- **TOK:** analytic budget equals packed tensor count and serialized state bytes;
- **CAUS:** causal-boundary invariance -- every context index is earlier than the first target,
  context and target sources are disjoint, and no target latent is supplied as context. The selected
  diffusion head jointly denoises the four target frames, so attention among noisy target tokens is
  intentional and is not tested as an autoregressive mask;
- **OVERFIT:** flat and hierarchical tiny-sample overfits;
- **DET:** same configuration and seed reproduce within the predeclared deterministic tolerance;
- **PERF:** profile all six budgets, verify fused SDPA use, bf16 path, memory headroom, and batch
  assembly fraction;
- **CVG:** train the 1,792-token Flat recent and Balanced recent arms at seed 0 through 30k, with
  immutable fixed-validation snapshots every 2k. Compare 8k, 12k, 16k, 20k, 24k, and 30k. Freeze
  the earliest candidate within 2% relative of its best later primary metric, with an unchanged
  flat--hierarchy direction at all later candidates and no sustained later improvement or
  overfitting. If neither 8k nor 12k passes, select the evidence-supported later schedule and
  revise the one-week matrix rather than forcing the original estimate.

Estimate: 4--6 researcher-hours and approximately 5--7 GPU-hours for the two 30k calibrations plus
validation. Exit criterion: every oracle passes and one
training schedule is frozen.

### A4-FRT -- six-point budget frontier

Budgets: `320, 768, 1152, 1792, 2432, 3072` context tokens.

Policies at every budget: Flat recent, Balanced recent, Linear age, and Geometric age.

- Unique configurations: 24.
- Final run instances: 48 (24 x two seeds).
- Main output: one quality-versus-retained-state frontier and one temporal-reach-versus-state
  curve. Seed points remain visible.

Estimated compact-instrument training time: 28--31 GPU-hours at 8k, or 42--46 GPU-hours at 12k.

### A4-ALC -- allocation at 1,792 tokens

Run the seven exact policies in Section 4.3. Flat recent, Balanced recent, Linear age, and
Geometric age reuse immutable A4-FRT runs. Only Fine-heavy, Coarse-heavy, and Reach-oriented add
training.

- New unique configurations: 3.
- New run instances: 6.
- Main output: one direct policy plot at fixed budget; a compact table gives token allocation,
  reach, retained bytes, and both seed results.

Estimated additional time: 3.8 GPU-hours at 8k or 5.7 GPU-hours at 12k.

### A4-CEX -- fixed detail with expanding coarse history

Fix six complete recent fine positions (384 tokens). Add exclusive older coarse positions in
`0, 4, 8, 16, 32, 64` steps. Total context budgets are therefore
`384, 448, 512, 640, 896, 1408`; all other settings remain fixed.

- Unique configurations: 6.
- Run instances: 12.
- Main output: validation quality and marginal gain per added retained-state kilobyte versus
  coarse positions.

Estimated time: 3.0 GPU-hours at 8k or 4.5 GPU-hours at 12k.

### A4-LVL -- one, two, and three levels at 1,792 tokens

Compare levels `1, 2, 3` at exactly 1,792 tokens. The two-level allocation is the predeclared
winner selected from A4-ALC by mean validation performance across both seeds. The three-level arm
extends the same allocation principle to the top/coarse/fine pyramid with explicit 4/16/64 token
costs and exact integer accounting. The one-level arm is the existing flat comparator.

Where the one- and two-level cells exactly match existing A4-FRT/A4-ALC runs, reuse them.

- New unique configurations after expected reuse: 1.
- New run instances: 2.
- Main output: one three-point level-count comparison, with the selection dependency disclosed.

Estimated additional time: 1.3 GPU-hours at 8k or 1.9 GPU-hours at 12k.

### A4-OVR -- temporal support overlap at 1,792 tokens

Apply the A4-ALC winning two-level payload to:

- **Nested:** coarse and fine may represent the same time point;
- **Exclusive:** a fine package replaces coarse tokens for that time point;
- **Residual:** the coarse base and its fine residual are both deliberately retained.

Total tokens and target remain exactly equal. Match nominal reach wherever the integer schedule
allows it and disclose any one-slot difference.

- Unique configurations: 3, with one likely reused from A4-ALC.
- New run instances: 4--6.
- Main output: one direct three-category comparison, not a compound panel.

Estimated additional time: 2.5--3.8 GPU-hours at 8k or 3.8--5.7 GPU-hours at 12k.

### A4-STS -- spatial-detail/temporal-coverage surface

Use budgets `1152` and `1792`, spatial quotas `K = 64, 32, 16, 8`, and three temporal placement
policies:

- recent-detail priority;
- uniform temporal coverage;
- geometric-age coverage.

Reserve 256 tokens for fine detail at both budgets. The remaining tokens buy a complete coarse
base: 56 coarse positions at 1,152 tokens and 96 at 1,792 tokens. Thus `K` controls whether the
256 fine cells cover 4, 8, 16, or 32 time positions, while the placement policy controls their
ages. This cleanly separates spatial resolution per selected time from temporal detail coverage.

- Unique configurations: 24.
- Run instances: 48.
- Main output: one heat map or small-multiple surface whose only role is quality across quota and
  temporal placement; a companion accounting plot maps the fixed 256 fine tokens to 4/8/16/32
  detailed time positions.

Estimated time: 24--26 GPU-hours at 8k or 36--39 GPU-hours at 12k.

### A4-MEC -- bounded mechanism diagnostics

On the locked Flat recent, best hierarchy, and one failure configuration only, measure temporal
coverage, per-level utilization, attention entropy, and re-entry population performance. These
are explanations, not separate superiority claims. No new training is permitted for A4-MEC.

Estimate: 3--5 researcher-hours and 1--2 GPU-hours for evaluation.

## 6. Deduplicated matrix and one-week estimate

**Calibration correction:** the table below is retained as the original planning estimate. A4-COR
rejected 8k and 12k. Scaling the agreed matrix to the recommended 30k requires approximately
265--295 GPU-hours before evaluation, so it does not fit one week on the local GPU. Do not use the
old 8k/12k totals for launch planning.

| Work | Unique configs | Run instances | 8k GPU-h | 12k GPU-h |
|---|---:|---:|---:|---:|
| A4-COR pilots/oracles | 2 paired 30k calibrations + executable tests | 2 + tests | 5--7 | 5--7 |
| A4-FRT | 24 | 48 | 28--31 | 42--46 |
| A4-ALC additions | 3 | 6 | 3.8 | 5.7 |
| A4-CEX | 6 | 12 | 3.0 | 4.5 |
| A4-LVL addition | 1 | 2 | 1.3 | 1.9 |
| A4-OVR additions | 2--3 | 4--6 | 2.5--3.8 | 3.8--5.7 |
| A4-STS | 24 | 48 | 24--26 | 36--39 |
| evaluation/profiling reserve | -- | -- | 6--8 | 8--10 |
| **Total** | **60--61 final configs** | **120--122 final runs** | **72--87** | **106--118** |

The local RTX 4060 Ti provides 168 theoretical hours per week, but 105--125 usable GPU-hours is
the realistic ceiling after debugging, evaluation, and restarts. Therefore:

- an 8k schedule fits with useful slack if A4-COR proves it converged;
- a 12k schedule is feasible but tight and requires an uninterrupted overnight queue;
- the historical 384x8, 30k protocol does not fit. Its full matrix is roughly 250--350 GPU-hours
  before evaluation.

These are measured planning estimates, not generic assumptions. On this host, the compact 256x6
flat instrument required approximately 0.22, 0.43, 0.67, 0.95, 1.62, and 2.16 hours for 12k
updates at the six budgets before applying the measured RMSNorm speed adjustment to all points.
The 1,792-token optimized probe ran at 0.286 s/update with 3.76 GiB peak allocated memory.

## 7. Calendar and dependency order

| Day | Queue | Exit criterion |
|---|---|---|
| 1 | finish v2 packer/runner; A4-COR DATA/INDEX/TARGET/TOK/CAUS/PERF; launch CVG traces | schedule and implementation frozen |
| 2 | A4-FRT low and middle budgets, both seeds | first four budget points complete |
| 3 | finish A4-FRT high budgets; run A4-ALC additions | 1,792-token allocation winner locked |
| 4 | A4-CEX then A4-LVL and A4-OVR | all sequential mechanism questions complete |
| 5 | A4-STS at 1,152 tokens | first surface complete |
| 6 | A4-STS at 1,792 tokens; memory repeats | full final matrix complete |
| 7 | validation selection, locked test generation/evaluation, A4-MEC, atomic figures/tables | reproducibility bundle and gate verdict complete |

If A4-COR selects 12k, Days 6--7 may extend by 12--18 hours. Test generation happens only after
the configuration hashes and selection rules are frozen; it is not available for queue decisions.

## 8. Runtime and storage engineering findings

### 8.1 Measured bottlenecks

- Encoding the full 3,000-clip development corpus takes about 14--16 seconds.
- Assembling and transferring a 1,792-token batch takes about 1.06 ms, compared with 286 ms for a
  complete optimized training update. The Python batch loop is under 0.5% of update time and is
  not the priority bottleneck.
- Full self-attention is the dominant, expected cost. PyTorch fused scaled-dot-product attention
  is already enabled.
- The legacy q/k RMSNorm path used bf16 activations with fp32 affine weights, forcing a CUDA
  fallback. The Path B mixed-precision RMSNorm keeps fp32 master weights but presents bf16 kernel
  operands. At 1,792 tokens this reduced measured update time from 0.336 to 0.286 s (about 15%)
  and peak allocation from 4.55 to 3.76 GiB (about 17%). Output/gradient parity is tested in fp32.
- A prepared global selected-fine-token path conflated local gather indices with absolute temporal
  positions. The Path B fix separates them and has a regression test for a short fine window
  inside longer coarse history.

### 8.2 Safe operating decisions

- Keep batch 16 across all six compact-model budgets; the 3,072-token probe fits at approximately
  7.2 GiB before the RMSNorm optimization.
- Do not adopt `torch.compile` or a new sparse-attention kernel after training begins. Either is a
  separate pre-lock benchmark because compilation and kernel changes can alter determinism and
  failure behavior.
- Optional latent caching saves only about 30--35 minutes over 120 separate processes. It may be
  added for operational convenience, but it is not required for the one-week budget.
- Store one rolling full training checkpoint per active run and a compact selected EMA/model state
  per completed run. Keeping optimizer-bearing checkpoints for all 120 runs would unnecessarily
  consume well over 10 GB.

## 9. Naming in artifacts and dissertation

Full machine IDs remain mandatory in configs, logs, appendices, and the reproducibility bundle:

```text
A4_<family>_B<budget>_L<levels>_<policy>_<overlap>_S<seed>
```

They do not need to burden dissertation prose. Use human labels in figures and text:

| Machine policy | Dissertation/figure label |
|---|---|
| `FLAT_RECENT` | Flat recent (`F`) |
| `BAL_RECENT` | Balanced hierarchy (`H-B`) |
| `LINEAR_AGE` | Linear-age hierarchy (`H-L`) |
| `GEOM_AGE` | Geometric-age hierarchy (`H-G`) |
| `FINE_HEAVY` | Fine-heavy (`H-F`) |
| `COARSE_HEAVY` | Coarse-heavy (`H-C`) |
| `REACH_OPT` | Reach-oriented (`H-R`) |

The caption or method table maps each short label to its exact immutable run ID. Raw run IDs
should not appear in narrative sentences.

## 10. Figures and tables

Main figures, each with one role:

1. architecture and token-allocation schema;
2. six-point quality--retained-state frontier;
3. allocation policy at 1,792 tokens;
4. fixed-detail coarse-expansion curve;
5. level count or overlap, whichever gives the stronger mechanistic result;
6. spatial-detail/temporal-coverage surface.

Main tables:

1. exact data, model, schedule, hardware, and memory protocol;
2. exact allocation definitions and token accounting;
3. both-seed locked primary results.

Configuration registries, all secondary metrics, checkpoint traces, and per-run memory records go
to supplementary material or the dissertation appendix. Do not use 95% training-seed intervals
with `n=2`.

## 11. Stop and fallback rules

- Stop and repair if any split, target, causal, token-accounting, or index oracle fails.
- Stop the full matrix if representative rankings are still changing materially at 12k updates.
- If a policy is numerically unstable, retain the failed run and diagnose it; do not silently
  replace its definition.
- If hierarchy has no quality--memory advantage, report the negative allocation result and do not
  choose a hierarchy configuration for Article 5 merely because it was expected to win.
- If the one-week queue overruns, preserve A4-FRT, A4-ALC, A4-CEX, and one complete A4-STS budget
  first. The second STS budget and the third-level comparison are the first deferrals; seed count
  is not reduced.
