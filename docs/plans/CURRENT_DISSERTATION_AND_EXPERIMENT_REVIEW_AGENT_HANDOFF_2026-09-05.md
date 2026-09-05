# Current dissertation and experiment review: independent-agent handoff

**Snapshot date:** 5 September 2026, Europe/Kyiv

**Purpose:** give a new independent reviewer enough verified context to audit the current
dissertation, the running A4 hierarchy campaign, and the prepared integrated OpenSTL model without
depending on conversation history.

This is a review handoff, not authorization to change or launch experiments. The reviewer may use
read-only status commands and bounded CPU tests. It must not interrupt the running GPU process,
authorize the integrated campaign, regenerate experimental manifests, edit result artifacts, or
insert partial results into the dissertation.

## 1. Binding author decisions

1. The dissertation remains in Ukrainian. New manuscript prose must be formal, final Ukrainian;
   the institutionally required English abstract is the only continuous English-prose exception.
2. The current four-chapter structure is retained. Front matter, Introduction, Chapters 1 and 2
   remain mostly unchanged. Chapters 3 and 4 retain their legacy structure and evidence while being
   locally expanded around the new experiments and integrated architecture.
3. Existing equations, figures, and tables remain in the main text. Only genuinely large result
   registries, supplementary result tables, or full-size supplementary montages belong in appendices.
4. Existing Article 1–3 material and figures remain part of the dissertation foundation.
5. A4 uses only the Toroidal Re-entry Benchmark (TRB; internal/legacy artifact name `LoopMom`).
   Standard and long Moving MNIST do not belong to the A4 matrix.
6. Existing Long Moving MNIST E6 and Figure 4.11 are retained as concise legacy boundary evidence.
   No new Long Moving MNIST training is planned.
7. The new A4 matrix uses 30,000 training steps and seeds 0 and 1 for every primary configuration.
8. There will be one future paper, not two: it will combine the extended A4 hierarchy study and a
   separately controlled integrated-model study as two evidence blocks.
9. The final integrated model is evaluated on Moving MNIST 10→10 and TaxiBJ 4→4. The four final
   full-model runs are dataset × seed 0/1, but component ablations and CUDA gates are separate work.
10. No expected result may be written as a fact. A result section is added only with complete
    artifacts, both seeds where required, final analysis, a figure/table, and an explicit claim limit.

## 2. Repository and worktree map

| Role | Path | Branch / commit at snapshot | Mutation rule |
|---|---|---|---|
| final dissertation refactor | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-dissertation` | `codex/dissertation-refactor-v2`, `fb5a7e1487ff935300f3641117e811d93e1b13e9` before this handoff update | review; only the requested review report may be added |
| running A4 hierarchy campaign | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-path-b` | `codex/hierarchy-allocation-v2`, `7a5979615f5354564a793083b673cf1ed9ec77be` | read-only while controller runs |
| integrated campaign controller and plans | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration` | `codex/integrated-openstl-v2`, `344f7a79a4905bbd0a7cd7809ebd3cb30de50e98` | review only; campaign is not authorized |
| integrated OpenSTL model | `C:\Users\leoni\Documents\projects\OpenSTL-integrated` | `codex/hvp-integrated-v2`, `ca335ba206ef6f51515a4f8eefcad1eed8c36c10` | bounded CPU/read-only review; no training |
| historical TempVerseFormer OpenSTL checkout | `C:\Users\leoni\Documents\projects\OpenSTL` | `OpenSTL-Lightning`, `b658dab3da427c8750c8595316e7ae9d70b818df` | dirty with author work; never clean, reset, commit, or reuse as the integrated implementation |
| original repository checkout | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction` | shared repository source/venv | do not overwrite the dissertation worktree |

The dissertation branch is pushed to `origin/codex/dissertation-refactor-v2`. The integrated
OpenSTL branch has no upstream shown by `git status -sb`; do not infer that it has been published.

## 3. Authoritative reading order

Read these documents completely before drawing conclusions:

1. `docs/plans/DISSERTATION_CONSERVATIVE_UKRAINIAN_REFACTORING_PLAN_2026-09-04.md` in the
   dissertation worktree — binding manuscript plan and integrated blind-review dispositions.
2. `docs/dissertation/condensed_translated_humanized_ukrainian/00_FRONT_MATTER.md` through
   `08_APPENDICES.md` — current manuscript in order.
3. `docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md` —
   formula, table, figure, claim, code, and artifact map.
4. `docs/plans/DISSERTATION_RESULT_INSERTION_LEDGER_2026-09-04.md` — the only authoritative map
   of result sections that are still gated.
5. `docs/reviews/DISSERTATION_CHAPTER_BY_CHAPTER_SELF_REVIEW_2026-09-04.md` — chapter-level
   implementation record.
6. `docs/reviews/BLIND_REVIEW_FINAL_CONSERVATIVE_UKRAINIAN_REFACTOR_2026-09-04.md` and
   `docs/reviews/FINAL_BLIND_REVIEW_FIX_INTEGRATION_REPORT_2026-09-04.md` — independent findings
   and their verified closure.
7. In Path B: `docs/plans/ARTICLE4_COMPLETE_EXPERIMENT_MATRIX_2026-09-01.md`,
   `docs/plans/ARTICLE4_30K_CAMPAIGN_IMPLEMENTATION_AND_OPERATIONS_2026-09-01.md`, and
   `docs/plans/PATH_B_DECISION_REGISTER_2026-09-01.md`.
8. In the integration worktree: `docs/plans/INTEGRATED_OPENSTL_MODEL_AND_SINGLE_ARTICLE_PLAN_2026-09-02.md`,
   `docs/plans/INTEGRATED_OPENSTL_IMPLEMENTATION_AND_OPERATIONS_2026-09-02.md`, and
   `docs/plans/INTEGRATED_OPENSTL_VERIFICATION_REPORT_2026-09-02.md`.
9. Historical context only: `docs/reviews/DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`
   in the original repository. Later author decisions and the 4 September conservative plan override
   its earlier English-rewrite/two-paper suggestions.

## 4. Current dissertation state

The manuscript passed its final conservative refactor at commit `fb5a7e1`:

- 119 numbered equations;
- 82 figure captions and 85 image insertions;
- 174 defined and 174 cited references;
- 11 explicit author/institution fields, all administrative;
- no projected-result shells or scientific placeholders.

Preservation against baseline commit `cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8` was independently
verified:

- all 114 legacy main-text equations remain in order;
- all 77 legacy numbered main-text figures and 79 legacy image insertions remain;
- all 22 legacy main-text tables remain;
- appendix inventory remains 18 tables and 4 figures;
- no legacy image asset was changed; the only new images are Figure 3.20 and the lower panel of
  composite Figure 4.3.

Chapter 3 contains the completed method contracts for A4 and the final integrated architecture.
Chapter 4 contains all legacy E2/E4/E5a/E6/E7/E8 evidence and only one newly completed numerical
result: A4-COR schedule calibration. Unfinished A4 family results and final integrated-model quality
or CUDA-memory claims are intentionally absent.

## 5. Claim boundaries the review must enforce

### 5.1 Hierarchical context

- Defensible legacy statement: hierarchy can buy temporal reach under a fixed retained-token budget;
  dense flat detail can be preferable after reach is sufficient.
- The verified legacy frontier is discrete, not a fitted continuous crossover: hierarchy at 896 and
  1,152 context tokens; unresolved at 1,792; flat at 2,432 and 3,072.
- `48 coarse + 6 fine = 1,152` and `48 coarse + 2 fine = 896` versus `48 fine = 3,072` establish
  2.67× and 3.43× token reductions for nominal 48-frame reach. They do not establish equal quality,
  equal information, or the same reduction in total GPU memory.
- A4 studies one jointly predicted four-frame target (physical times 96–99) from a common 96-frame
  coordinate system. Context reach is not long autoregressive stability.

### 5.2 Reversibility

- TempVerseFormer supports activation-memory reduction inside its verified reversible temporal
  state. Encoder, projections, fusion, decoder, and loss are outside the reversible boundary.
- The previous integrated prototype’s 2.58× peak ratio compares different external dynamics after
  the first transition. It is engineering evidence, not a same-forward causal estimate.
- The final integrated architecture corrects this by making ordinary and reversible execution share
  the identical Transformer-core forward function. Its CUDA benefit is still unmeasured.

### 5.3 Dynamic spatial routing

- V-BLT’s entropy maps are a diagnostic signal; V-BLT did not implement dynamic patch allocation.
- The final integrated router uses exactly three numeric terms: coarse/fine residual energy,
  temporal change, and a learned correction.
- Routing occurs before the reversible core. Selected indices and metadata remain outside the
  reversible state. A fixed token quota keeps tensor shapes stable for training and inference.
- At equal quota, static versus density/hybrid routing tests allocation quality, not memory savings.
  A routing memory advantage requires a smaller quota that still meets a predeclared quality target.

### 5.4 Memory accounting

Never multiply or merge these quantities:

1. logical token or payload ratio;
2. complete live state, including coarse grids, selected indices, metadata, queries, and buffers;
3. core activation peak;
4. end-to-end CUDA peak including parameters, gradients, optimizer, allocator, and temporary memory;
5. inference-only peak.

Every memory graph must state device, precision, batch, horizon, measurement endpoint, model state,
and whether the comparison shares weights and a forward function.

## 6. A4 experiment contract

A4-COR used two seed-0 calibration trajectories at budget 1,792. It is outside the 86 primary
training runs. Fifteen validation points through 30k showed a rank change between 12k and 16k; 24k
still missed the registered 2% convergence rule; 30k was the first accepted common slice across all
six metrics. This justifies duration, not superiority of either allocation.

The primary campaign contains 49 displayed cells, 43 deduplicated configurations, and 86 runs:

| Family | Contract | New training runs after reuse | Main interpretation |
|---|---|---:|---|
| A4-FRT | budgets 320, 768, 1,152, 1,792, 2,432, 3,072 × Flat recent, Linear age, Geometric age | 36 | quality/state and reach/state frontier; Balanced is excluded |
| A4-ALC | seven allocations at exactly 1,792 tokens: Flat, Fine-heavy, Balanced, Coarse-heavy, Linear, Geometric, Reach-oriented | 8 | equal-budget allocation comparison; three configurations reused from FRT |
| A4-CEX | fixed six recent fine positions plus 0, 4, 8, 16, 32, or 64 exclusive coarse positions | 12 | marginal value of older coarse context; budget varies |
| A4-LVL | 1, 2, 3 levels at 1,792, using the selected best non-flat ALC hierarchy downstream | 2 | abstraction-depth test; three-level tokenizer adds 1,280 parameters |
| A4-OVR | nested, exclusive, residual at 1,792 using an overlap-feasible selected ALC base | 4 | temporal ownership/representation; one nested cell reused |
| A4-STS | budget 1,792; K = 64, 32, 16, 8; Recent, Uniform, Geometric temporal placements | 24 | spatial-detail versus temporal-detail placement; deterministic, not learned routing |
| A4-MEC | best flat, best hierarchy, one predeclared underperformer × two seeds | 0 training / 6 evaluations | bounded mechanism diagnostic, not independent proof |

Selection-dependent LVL and OVR configurations are generated only after the complete ALC family is
finalized. Confirmation uses 500 fresh clips and a frozen shortlist; it must not reselect models on
confirmation data.

### 6.1 Live A4 status at this snapshot

At 05:19 Kyiv time on 5 September:

- Wave 1: 37 complete, 1 running, 0 failed, 42 pending;
- active run: `A4_ALC_B1792_L2_FINE_HEAVY_NEST_S01_T30K`, step 29,000/30,000;
- Wave 2: not yet generated because its selection dependency is unresolved;
- post-training finalization: pending.

The original direct estimate is 187–205 GPU-hours, with a 205–234 GPU-hour recovery reserve. Do
not infer a new ETA from one run without using actual completed durations and budget weighting.

Read-only status command from Path B:

```powershell
& 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction\venv\Scripts\python.exe' `
  scripts/v2/run_article4_campaign.py status
```

Do not invoke `run`, `preflight`, W&B synchronization, finalization, or analysis while reviewing.
Do not use partial within-family values in the dissertation.

## 7. Final integrated architecture and experiment status

The implemented OpenSTL path is:

`multiscale encoder → complete coarse grid + physically gathered fine quota → physical coordinate
encoding and level embedding → same-forward reversible Transformer core → future queries and
cross-attention → residual frame decoder`.

The interface has separate Moving MNIST 10→10 and TaxiBJ 4→4 configurations. CPU verification at
the reviewed state established 20 OpenSTL model/data/CLI tests and five campaign controller/worker
tests. These tests establish shapes, quotas, coordinate contracts, permutation consistency,
forward/gradient parity, data split contracts, and restart behavior. They do not establish predictive
competitiveness or CUDA memory.

All four confirmation runs remain pending and deliberately unauthorized:

- `INT_MM_S00`, `INT_MM_S01`;
- `INT_TB_S00`, `INT_TB_S01`.

Before authorization the project requires, in order:

1. bounded overfit tests on 32–64 clips for both datasets;
2. 200-update timing/capacity/precision screens;
3. same-checkpoint vanilla/reversible and flat/hierarchical CUDA profiles;
4. static/density/hybrid routing comparison on Moving MNIST validation;
5. a 5–10 epoch screen on both datasets;
6. freezing configs, commits, validation metric, update count, duration estimate, baseline contract,
   and acceptable quality tolerance;
7. explicit author confirmation and only then a matching authorization ledger.

Read-only status command from the integration worktree:

```powershell
& 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction\venv\Scripts\python.exe' `
  scripts/v2/run_integrated_openstl_campaign.py status
```

The eventual resumable command exists in the operations plan, but the reviewer must not execute it.

## 8. Resource snapshot and review safety

At the snapshot, the RTX 4060 Ti has 16,380 MiB total memory, reports 100% utilization, and is owned
by the A4 campaign. System RAM has only approximately 6.4 GiB free of 31.8 GiB. Disk free space is
approximately 16.6 GiB on C: and 74.9 GiB on D:.

Therefore:

- do not run GPU tests or start another trainer;
- avoid the complete CPU test suite while A4 is active;
- if a CPU test is necessary, hide CUDA and restrict math threads to one;
- do not prepare/copy large datasets or checkpoints onto C:;
- do not mutate W&B, authorization ledgers, manifests, result directories, checkpoints, or locks;
- do not clean any dirty worktree, especially the historical `OpenSTL` checkout.

Safe manuscript gates from the dissertation worktree are:

```powershell
$python = 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction\venv\Scripts\python.exe'
& $python scripts/check_manuscript_finalization.py
& $python scripts/check_text_consistency.py --gate
& $python scripts/check_rft_consistency.py
& $python scripts/audit_ukrainian_dissertation.py
git diff --check cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8..HEAD
```

Do not use `audit_ukrainian_translation_parity.py` as a current gate. It compares against an obsolete
pre-expansion English tree and is expected to fail after the approved Ukrainian additions.

## 9. Required independent review

The assigned reviewer should produce an evidence-backed report that:

1. independently reads the current manuscript and checks it against the binding author decisions;
2. identifies scientific, structural, language, evidence, figure/table/formula, and provenance
   defects with exact file and line locations;
3. checks that A4-COR is the only newly inserted completed result and that current partial A4 results
   are not leaked into claims;
4. checks current A4 health, progress, recent logs, timing behavior, failed/restarted attempts,
   checkpoint continuity, disk growth, and whether validation curves look numerically plausible;
5. reviews the A4 matrix for duplicates, missing cells, dependency correctness, two-seed coverage,
   target invariance, token accounting, and fair interpretation;
6. reviews the integrated model code and plans for positional encoding, router/reversibility boundary,
   stable shapes, inference routing modes, data contracts, resumability, and fair CUDA profiling;
7. states separately what is already verified, what depends on ongoing experiments, what requires an
   author decision, and what would require new code or relaunch;
8. explicitly answers whether any current finding requires interrupting/restarting A4 (the default
   presumption is no unless direct evidence shows corruption);
9. does not invent likely positive outcomes or promise a specific 3–4×/2–3× component gain;
10. classifies findings P0–P3 and ends with a concrete continue/fix/relaunch recommendation.

The reviewer may create one report in the dissertation worktree under `docs/reviews/`. It should not
edit the dissertation, plans, experiment code, manifests, or artifacts. Any proposed manuscript
change must remain a recommendation until reviewed by the primary agent and author.

## 10. Outstanding author decisions

The review should preserve, not silently resolve, these choices:

1. Whether to evaluate long autoregressive rollouts for a frozen shortlist of A4 checkpoints; if yes,
   horizon, buffer update, population, metrics, seeds, and checkpoint-selection rule must be frozen
   first. This can use trained checkpoints and need not relaunch all 86 models.
2. The short list for one-time A4 confirmation.
3. Which local OpenSTL models are primary matched baselines for each dataset.
4. The predeclared quality tolerance defining “competitive” integrated performance.
5. The permitted component-ablation budget and whether activation checkpointing is included after
   mandatory flat/reversible/inference profiles.
6. Final A4 title, author list, venue, status, and contribution statement.
7. The 11 institutional/administrative manuscript fields: UDC/title details, two funded projects,
   implementation acts, six conference records, authorship/contribution details, and final pagination
   statistics.

## 11. Current expected conclusion of a careful review

The conservative manuscript refactor itself is release-clean at the documented commit and passed a
previous independent blind review after all P1–P3 findings were fixed. The new reviewer is expected
to challenge that conclusion independently, not repeat it. The ongoing A4 campaign currently shows
normal forward progress with zero recorded failures, but family-level scientific interpretation is
premature. The integrated architecture is implemented and CPU-verified, but its central quality and
CUDA-memory hypotheses remain open and must pass the bounded gates before the four long runs are
authorized.
