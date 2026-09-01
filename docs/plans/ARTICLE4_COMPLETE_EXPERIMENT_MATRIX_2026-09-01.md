# Article 4 complete experiment matrix

**Revision:** 2026-09-01  
**Status:** experiment content agreed; final update count awaiting Leonid's decision  
**Dataset:** Toroidal Re-entry Benchmark (TRB) development split only until locked confirmation  
**Training seeds:** `0` and `1` for every primary cell

This is the authoritative, deduplicated matrix behind the Article 4 campaign. It expands the
compact list in the decision register, states what every family tests, identifies reused cells,
and separates implemented definitions from definitions that must still be locked in code. It does
not freeze 16k or 30k updates.

## 1. Common protocol for every training cell

- Data: hash-pinned 2,700/300 TRB train/validation clip split. The historical test file is never
  used for selection.
- Target: common first-target anchor 96 and target frames 96--99 for every allocation.
- Model: latent diffusion transformer, v-prediction, width 256, six blocks, eight heads,
  approximately 7.56 M parameters, shared 100-position temporal embedding, and a four-frame target.
- Optimization: batch 16, bf16 autocast with fp32 optimizer/master weights, identical optimizer,
  schedule, decoder, EMA, validation records, and deterministic DDIM evaluation.
- Allocation is the manipulated variable. Model capacity and prediction target stay fixed except
  for the explicitly disclosed level-count construction.
- Replication: two complete training seeds per cell. Report both seed points, their mean, and paired
  seed-wise effects; do not report a seed-level 95% confidence interval for `n=2`.
- Memory reporting: context token payload, serialized retained-state bytes including metadata,
  training and inference CUDA allocated/reserved peaks, parameters, optimizer, activations,
  workspace, update time, and evaluation time are separate endpoints.

The pending schedule choice has two different meanings:

- **16k:** a common fixed-compute endpoint. Claims must say “after 16,000 equal updates.” It is not
  a convergence claim under the registered 2% rule.
- **30k:** the first registered checkpoint that passed the paired calibration rule. It supports a
  late-training/convergence interpretation, but the complete matrix no longer fits one calendar
  week on the local GPU.

## 2. A4-COR: correctness and schedule calibration

These are gates, not primary hierarchy-result cells.

| Item | Arms / checks | Purpose | Status |
|---|---|---|---|
| DATA | source hashes, shapes, dtypes, split isolation, action/pixel alignment | establish that clips and controls are loaded and aligned correctly | passed |
| INDEX | local packed gather index versus absolute physical time | prevent a short fine window from receiving wrong time identities | passed |
| TARGET | invariant frames 96--99 and four target frames | prevent allocation from silently changing the prediction task | passed |
| TOK | analytic count, packed count, and serialized bytes | verify the nominal budget represents actual retained state | passed |
| CAUS | all context before target; disjoint source indices | exclude target leakage while retaining intentional joint target denoising | passed |
| OVERFIT | flat and hierarchy tiny-sample fits | establish trainability of both paths | passed |
| DET | deterministic resume/replay and RNG restoration | establish interruption-safe scientific equivalence | passed |
| PERF | loader, packed transfer, SDPA, bf16, memory headroom | identify runtime bottlenecks and safe batch size | passed |
| CVG | Flat recent and Balanced recent, budget 1,792, seed 0, 30k | select an update count from fixed milestones | complete; 30k recommended |

Calibration run stems:

- `A4_COR_B1792_L1_FLAT_RECENT_EXCL_S00_T30K`
- `A4_COR_B1792_L2_BAL_RECENT_NEST_S00_T30K`

They are not counted in the 120 final training runs and are not automatically promoted to
two-seed results.

## 3. A4-FRT: six-point quality--memory frontier

**Question:** how does quality change with retained-state budget, and do hierarchical allocation
policies dominate or extend the flat recent-context frontier?

At budget `B`, Flat retains `B/64` complete fine time positions. Every hierarchical arm spends half
the budget on a contiguous coarse base (`B/32` positions at 16 tokens each) and half on fine cells
(`B/2` selected cells). Balanced, linear, and geometric policies differ only in the temporal
placement of those fine cells. Geometric decay is fixed at 0.94.

| Budget | Flat fine times | Hierarchy coarse times | Hierarchy fine cells | Policies (each at seeds 0,1) |
|---:|---:|---:|---:|---|
| 320 | 5 | 10 | 160 | Flat recent; Balanced recent; Linear age; Geometric age |
| 768 | 12 | 24 | 384 | Flat recent; Balanced recent; Linear age; Geometric age |
| 1,152 | 18 | 36 | 576 | Flat recent; Balanced recent; Linear age; Geometric age |
| 1,792 | 28 | 56 | 896 | Flat recent; Balanced recent; Linear age; Geometric age |
| 2,432 | 38 | 76 | 1,216 | Flat recent; Balanced recent; Linear age; Geometric age |
| 3,072 | 48 | 96 | 1,536 | Flat recent; Balanced recent; Linear age; Geometric age |

Matrix expansion: `6 budgets x 4 policies = 24 configurations x 2 seeds = 48 runs`.

Primary output: quality versus measured retained state, with all seed points shown. Secondary
output: temporal reach versus state. This family alone does not prove why a policy works.

## 4. A4-ALC: allocation policies at exactly 1,792 tokens

**Question:** at one equal token budget, which balance of recent spatial detail and temporal reach
is most effective?

| Label | Levels | Exact payload | Reach | Training disposition |
|---|---:|---|---:|---|
| `F` Flat recent | 1 | 28 complete fine times | 28 | reuse A4-FRT |
| `H-F` Fine-heavy | 2 | 28 coarse times + 21 complete recent fine times | 28 | new |
| `H-B` Balanced | 2 | 56 coarse times + 896 newest-first fine cells | 56 | reuse A4-FRT |
| `H-C` Coarse-heavy | 2 | 84 coarse times + 7 complete recent fine times | 84 | new |
| `H-L` Linear age | 2 | 56 coarse times + 896 linearly age-weighted fine cells | 56 | reuse A4-FRT |
| `H-G` Geometric age | 2 | 56 coarse times + 896 geometrically age-weighted fine cells | 56 | reuse A4-FRT |
| `H-R` Reach-oriented | 2 | 96 coarse times + 4 complete recent fine times | 96 | new |

All rows contain exactly 1,792 context tokens. Matrix expansion: seven displayed configurations,
of which four are reused and three are new; `3 x 2 seeds = 6 new runs`.

Primary output: equal-budget quality comparison, accompanied by exact reach and measured retained
bytes. The winner is selected by the predeclared validation rule across both seeds and becomes an
input to A4-LVL and A4-OVR. **Proposed dependency rule, still to lock before launch:** if Flat wins,
it remains the scientific winner, while the downstream hierarchy-only mechanism tests use the best
**non-flat hierarchical** allocation so that “levels” and “overlap” remain defined.

## 5. A4-CEX: fixed detail with expanding older coarse context

**Question:** with recent fine detail held fixed, what is the marginal value of increasingly old
coarse history, and when does it saturate?

Every cell has six complete recent fine positions (384 fine tokens). Coarse support is exclusive
and begins at age 7, so it never duplicates those six fine time positions.

| Coarse positions | Coarse ages | Total budget | Total reach | Seeds |
|---:|---|---:|---:|---|
| 0 | none | 384 | 6 | 0, 1 |
| 4 | 7--10 | 448 | 10 | 0, 1 |
| 8 | 7--14 | 512 | 14 | 0, 1 |
| 16 | 7--22 | 640 | 22 | 0, 1 |
| 32 | 7--38 | 896 | 38 | 0, 1 |
| 64 | 7--70 | 1,408 | 70 | 0, 1 |

Matrix expansion: `6 configurations x 2 seeds = 12 runs`.

Primary output: quality and marginal quality gain per added retained-state byte versus number of
coarse positions. Because budget varies, this is not an equal-memory comparison.

## 6. A4-LVL: one, two, and three hierarchy levels

**Question:** after holding total payload at 1,792 tokens, does adding hierarchy levels improve the
allocation, or does it merely add abstraction loss?

| Levels | Construction | Reuse / new | Seeds |
|---:|---|---|---|
| 1 | Flat recent, 28 complete fine times | reuse A4-FRT | 0, 1 |
| 2 | best non-flat A4-ALC hierarchy under the registered selection rule | reuse A4-ALC | 0, 1 |
| 3 | deterministic top/coarse/fine extension of the selected two-level rule, exactly 1,792 tokens | new | 0, 1 |

Matrix expansion after reuse: one new configuration and two new runs.

**Pre-launch lock still required:** the code supports level metadata but does not yet implement the
exact adaptive three-level allocation. Before launching A4-LVL, freeze and test a deterministic
mapping from the selected two-level payload to top/coarse/fine token counts (costs 4/16/64 per full
time position), including exact budget, reach, time identities, and parameter-count parity. This
must be a predeclared transformation, not a schedule chosen after seeing the three-level result.

Primary output: one three-point comparison. Any unavoidable capacity or representation change is
reported explicitly rather than called a pure allocation effect.

## 7. A4-OVR: nested, exclusive, and residual temporal support

**Question:** when coarse and fine representations refer to the same physical time, is redundancy
helpful, wasteful, or best expressed as a residual?

The base payload is the best non-flat hierarchical A4-ALC allocation at 1,792 tokens.

| Mode | Intended scientific meaning | Reuse / new | Seeds |
|---|---|---|---|
| Nested | retain coarse tokens and ordinary fine tokens for the same selected times | reuse the matching A4-ALC cell | 0, 1 |
| Exclusive | fine tokens replace coarse tokens at selected times; freed tokens extend/redistribute coarse support while total budget stays fixed | new | 0, 1 |
| Residual | retain coarse tokens plus fine residual information relative to the coarse reconstruction at those times | new | 0, 1 |

Target matrix: three displayed configurations, one reused and two new, hence four new runs.

**Pre-launch lock still required:** `overlap` is presently metadata on `AllocationPlan`; the packed
builder does not yet give all three modes distinct scientific semantics. In particular, “nested”
and “residual” would currently select the same token identities. A4-OVR is not launch-ready until
exclusive reallocation and the fine-residual transform are implemented and unit-tested. This is a
code/design gap, not an experimental result.

Primary output: a direct three-category equal-budget plot. If equal nominal reach is impossible
because of integer token costs, disclose the one-slot difference.

## 8. A4-STS: spatial-detail versus temporal-detail coverage

**Question:** if a fixed fine-token payload is spread over more time positions, how does reduced
spatial detail per time trade against denser temporal coverage?

At both budgets exactly 256 tokens are reserved for fine cells. The rest buys a complete coarse
base: 56 times at budget 1,152 and 96 times at budget 1,792. Fine quota `K` is the number of spatial
cells retained at each selected time, so the same 256 fine tokens cover 4, 8, 16, or 32 times.

| Budget | Coarse base | Quota K | Detailed times | Temporal placements (each at seeds 0,1) |
|---:|---:|---:|---:|---|
| 1,152 | 56 times / 896 tokens | 64 | 4 | Recent; Uniform; Geometric |
| 1,152 | 56 times / 896 tokens | 32 | 8 | Recent; Uniform; Geometric |
| 1,152 | 56 times / 896 tokens | 16 | 16 | Recent; Uniform; Geometric |
| 1,152 | 56 times / 896 tokens | 8 | 32 | Recent; Uniform; Geometric |
| 1,792 | 96 times / 1,536 tokens | 64 | 4 | Recent; Uniform; Geometric |
| 1,792 | 96 times / 1,536 tokens | 32 | 8 | Recent; Uniform; Geometric |
| 1,792 | 96 times / 1,536 tokens | 16 | 16 | Recent; Uniform; Geometric |
| 1,792 | 96 times / 1,536 tokens | 8 | 32 | Recent; Uniform; Geometric |

Placement definitions:

- Recent selects the newest `256/K` time positions.
- Uniform spreads them deterministically across the complete coarse reach.
- Geometric concentrates more selections near the present while retaining older coverage.
- At each selected time, cells come from a deterministic nested, phase-cycled spatial lattice;
  there is no content-dependent selection in Article 4.

Matrix expansion: `2 budgets x 4 quotas x 3 placements = 24 configurations x 2 seeds = 48 runs`.

The authoritative allocation generator currently uses **nested** coarse/fine support. An older
seed-0 scaffold filename/config says `EXCL`; it must be regenerated and must not be launched.

Primary output: a quality surface over spatial quota and temporal placement, plus a separate token
accounting graphic. This family studies deterministic spatial sparsity and temporal placement; it
does not test Article 5's learned dynamic routing.

## 9. A4-MEC: bounded mechanism diagnostics

**Question:** what observable mechanism distinguishes the selected hierarchy from Flat and from a
representative failed allocation?

No model is trained for A4-MEC. Evaluate the locked checkpoints for:

1. best Flat configuration;
2. best hierarchical configuration;
3. one predeclared failure/underperforming configuration.

For both seeds, measure temporal coverage, per-level utilization, attention entropy, and
performance on re-entry-defined clip populations. This is six evaluation instances, not six new
training runs. The diagnostics explain results; they do not independently prove superiority.

## 10. Deduplicated total

| Family | Displayed configs | New unique configs after reuse | New training runs (2 seeds) |
|---|---:|---:|---:|
| A4-FRT | 24 | 24 | 48 |
| A4-ALC | 7 | 3 | 6 |
| A4-CEX | 6 | 6 | 12 |
| A4-LVL | 3 | 1 | 2 |
| A4-OVR | 3 | 2 | 4 |
| A4-STS | 24 | 24 | 48 |
| **Primary matrix** | **67 displayed family cells** | **60 unique trained configs** | **120 runs** |
| A4-MEC | 3 selected checkpoints | 0 | 0 (six evaluations) |
| A4-COR calibration | 2 | separate completed calibration | separate; not in final 120 |

The target total of 60/120 assumes the dependency-based reuse described above. If a downstream
definition is changed so it is no longer byte-identical to its source cell, it becomes a new
configuration and the campaign manifest must update the count rather than silently reuse it.

## 11. Optional work outside the core 120 runs

- Historical-capacity robustness: repeat 4--6 decisive configurations with the 384x8 instrument
  (`8--12` additional training runs). This is optional and never mixed into the primary matrix.
- Third-seed robustness: seed 2 for 6--8 locked cells (`6--8` additional runs), only after reviewing
  the two-seed effects.
- Regression-head control: bounded robustness analysis only; not a per-cell alternative head.
- Fresh TRB confirmation: generate only after configs, code, selection rules, and hashes are locked;
  evaluate selected checkpoints only, with no test-driven reselection.

## 12. Time implication of a 16k choice

The paired 30k calibration measured 2.25 h for Flat and 2.39 h for Balanced at budget 1,792. Runtime
also changes with token budget, so a single per-run multiplication is insufficient. Scaling the
budget-aware campaign estimate gives approximately **141--157 GPU-hours of training at 16k**, plus
roughly **6--10 GPU-hours** for validation, profiling, mechanism diagnostics, and confirmation.

That is about **6.1--7.0 days of perfectly uninterrupted GPU occupancy** and realistically about
**8--11 calendar days** on the local machine once process transitions, W&B/image logging,
evaluation, recovery margin, and active use are included. Therefore 16k is much closer to the
one-week goal than 30k, but the full matrix still has essentially no one-week slack.

At 30k, the current estimate remains approximately **265--295 GPU-hours before final evaluation**,
or 11--12.3 uninterrupted GPU-days and roughly 2--3 calendar weeks locally.

If 16k is selected, the scientifically honest staged rule is:

1. train every primary cell to exactly 16k and interpret the matrix as equal-compute screening;
2. lock the decisive/Pareto configurations using validation only;
3. resume only those selected checkpoints to 30k to test whether their ranking persists;
4. reserve “converged” or “best achievable quality” language for the resumed comparison.

This preserves the broad matrix without pretending the A4-COR convergence criterion passed at
16k.

## 13. What “ranking reversal” means

It is an **observed effect in the validation curves**, not a code feature and not an operation that
reverses tokens, levels, gradients, labels, or ranks.

For the seed-0 calibration at budget 1,792, lower horizon-4 foreground MSE is better:

| Updates | Flat | Balanced hierarchy | Better arm |
|---:|---:|---:|---|
| 8k | 0.046516 | 0.042223 | hierarchy |
| 12k | 0.033052 | 0.029133 | hierarchy |
| 16k | 0.026283 | 0.028735 | Flat |
| 20k | 0.023915 | 0.027803 | Flat |
| 24k | 0.022439 | 0.027099 | Flat |
| 30k | 0.022150 | 0.025646 | Flat |

The ordering crosses between 12k and 16k: hierarchy looks better early, then Flat overtakes it.
That is the rank reversal. It is compatible with a training-dynamics explanation: the balanced
hierarchy may provide a useful early inductive bias, while the flat representation continues to
improve to a better late value. The calibration alone cannot establish that mechanism.

Known implementation checks make a simple bookkeeping explanation unlikely: both arms have the
same parameter count and initialization, the same 1,792 context-token count and 2,112-token total
attention sequence, the same targets and validation records, exact action/pixel alignment, exact
packed gathers and physical time IDs, no target leakage, successful overfit/smoke tests, and a full
passing test suite. Nevertheless this remains one seed and two allocation policies, so the crossover
is a schedule warning, not a publishable general conclusion about hierarchy.

At 16k the direction remains Flat-better at every later registered milestone, which makes 16k a
reasonable fixed-compute comparison point. But neither arm met the registered convergence rule at
16k: Flat was 18.66% above its best later primary value and Balanced hierarchy was 13.25% above its
best later value.
