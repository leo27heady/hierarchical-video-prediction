# Master implementation plan: Reach, Fidelity, and Trust

**Date:** 2026-07-11
**Status:** historical governing plan; live authority moved to
`docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md`
**Scope:** experiments, evaluation, code, evidence, article, dissertation, repository, and long-term maintenance
**Supersedes for prioritisation:** the execution ordering in older completion/experiment plans where they conflict with this document
**Does not supersede:** measured artifacts, preregistrations, or the historical record

> **Terminal update, 2026-07-15:** E5a did not replicate the seed-0 late trust inversion. The
> canonical fourth-article title is now **Reach and Fidelity in Budgeted Video Memory: A Controlled
> Context-Packaging Study**. The reach–fidelity–trust formulation below is retained as the original
> planning hypothesis, not as the terminal claim state.

## 1. Decision summary

The project should continue, but it should change from an expanding experiment programme into a controlled claim-consolidation programme.

The scientific identity is:

> **Context packaging is the manipulated design space; reach, fidelity, and trust are the explanatory axes.**

- **Reach:** whether the supplied context includes the memory-relevant past.
- **Fidelity:** how much relevant information survives the representation and token budget.
- **Trust:** whether stored information remains usable after repeated self-fed generation.

The working article title is:

> **Reach, Fidelity, and Trust in Budgeted Video Memory: A Controlled Context-Packaging Study**

Immediate operational decision:

1. Allow the already-running seed-2 trio to finish.
2. Stop the old master queue before it advances.
3. Run only the already preregistered c2 B8x60k regime control.
4. Leave the GPU idle after c2.
5. Do **not** launch the current Phase C design.
6. Do not launch another GPU experiment until the evaluation redesign, untouched-confirmation protocol, run identity, and experiment authorization gates below are satisfied.

The current Phase C training comparison is not inherently meaningless, but its identity, trajectory, and equivalence interpretation is invalid under the existing evaluator. It is cheaper and scientifically cleaner to redesign the endpoints before paying for the run.

## 2. Governing rules

These rules apply to every workstream.

1. **One experiment controller and one execution owner.** No two agents or humans may launch or mutate the same queue.
2. **Claims are frozen before holdout access.** Analysis decisions are recorded before the untouched confirmation set is generated or opened.
3. **No silent overwrite.** Every run and artifact has an immutable identity; an existing run directory is never reused.
4. **CPU gates precede GPU work.** Dataset, evaluator, assignment, metrics, resume, and artifact tests must pass on small CPU fixtures before a full run is authorized.
5. **Measured fact, interpretation, and future hypothesis stay separate.** Use `MEASURED`, `INTERPRETATION`, `EXPLORATORY`, `PENDING`, and `FUTURE` labels.
6. **Negative results remain visible.** B3, reversibility limitations, trust failures, and seed-sensitive signs are not rewritten into success narratives.
7. **No causal wording stronger than the intervention.** A family/configuration comparison is not a pure context-stride effect; a two-point sign change is not a unique crossing; cancellation is not equivalence.
8. **No new flagship claim without a suitable baseline and construct-valid endpoint.**
9. **Live artifacts are preserved.** In particular, the current generated CW table and Figure 12 remain untouched until their producing queue has completed and they are integrated deliberately.

## 3. What is currently defensible

### 3.1 Core claims to retain

- The matched-budget context-packaging comparisons reveal large, repeatable differences in long-range re-entry error.
- Temporal reach can dominate spatial detail when relevant evidence lies outside a short context.
- More detailed or flatter context becomes preferable in tested higher-budget regimes.
- The stride-2 B1 **configuration** is strongly better than its compared short flat configuration in the measured diffusion and regression runs.
- Pair-specific reach-differential bands provide strong mechanistic evidence, but the current analysis is post-hoc and must be preregistered for confirmation.
- Self-fed rollout can destroy information that is addressable under teacher forcing; this is presently a measured boundary/failure case, not a universal law.
- Regression changes the package contrast consistently relative to diffusion across the two measured seeds, while the regression package sign itself is seed-sensitive.

### 3.2 Required wording corrections

Replace broad or over-strong statements with the following contracts:

| Avoid | Use |
|---|---|
| exact arithmetic edge / unique crossing | largest tested budget favouring A and smallest tested budget favouring B |
| dense-family edge is 896 | switch-on is bracketed by the tested 512 and 896 configurations |
| stride-2 lower edge is 512 | stride-2 wins at 512; its lower edge is unlocated |
| exactly matched bottom comparison | near-budget comparison of 480 versus 512 context tokens |
| B1 isolates context stride | B1 compares complete stride-2 and short-flat configurations |
| B8x60k uses twice the data | matched-example, changed-batch/update/EMA-regime control |
| regression tie / equivalence | seed-sensitive sign reversal; pooled cancellation is not equivalence |
| trustworthy-memory law | measured addressable-versus-self-fed boundary at the tested horizon and seed |
| information-theoretically required | exactly labelled geometric visibility/observability in this generator |
| prior work lacks memory evaluation | this study adds a controlled matched-budget observability instrument and inversion analysis |
| matched compute, when only model size is matched | parameter/capacity-matched, with inference allocation reported separately |
| AE upper bound | AE reconstruction reference or codec-imposed reconstruction floor, as appropriate |

### 3.3 Claims to quarantine pending redesign or replication

- Existing Phase C identity, trajectory, and replicated-equivalence conclusions.
- General trust laws from one decisive long-loop seed.
- Equivalence claims from Moving-MNIST 10-to-10 or two-seed regression averaging.
- A pure causal context-stride interpretation of B1.
- A unique continuous frontier crossing inferred from two tested budgets.
- Broad first-ever or no-prior-work novelty statements.

## 4. Execution phases and release gates

The phases are ordered. Work inside a phase may run in parallel only when it does not create competing experiment controllers or expose the holdout early.

### Phase 0 — Stop boundary, preservation, and ownership

**Objective:** finish only the safe pre-C work and establish a clean handoff.

Tasks:

- [ ] Let the active seed-2 trio reach its normal completion marker.
- [ ] Verify the watcher terminates the old master controller before it advances.
- [ ] Run the preregistered c2 control alone.
- [ ] Verify c2 has a completion marker and valid artifacts.
- [ ] Verify there is no Phase C process, marker, child process, or stale launch shell.
- [ ] Verify the old master PID is gone and the stale queue lock is removed.
- [ ] Verify the GPU is idle.
- [ ] Copy/hash the live queue logs, CW table, Figure 12, NPZs, checkpoints, and preregistration text before integration.
- [ ] Record one named experiment execution owner in `docs/EXPERIMENT_CONTROL.md`.
- [ ] Put every proposed run into the authorization matrix in Section 9.

**Gate 0:** no old controller is alive; c2 is complete or explicitly failed; Phase C is absent; artifacts are hashed; one owner is recorded.

### Phase 1 — Claim and analysis freeze

**Objective:** define exactly what the current evidence says before changing evaluation or touching a new holdout.

Tasks:

- [ ] Update `docs/CLAIM_LEDGER.md` with the wording contracts in Section 3.
- [ ] Add evidence status, population, seeds, endpoint, intervention, comparator, and artifact hash to each material claim.
- [ ] Mark pair-specific reach-differential results `EXPLORATORY_POST_HOC` on the current development pool.
- [ ] Mark Phase C factor claims `BLOCKED_BY_MEASUREMENT_VALIDITY`.
- [ ] Mark B1 as a configuration/family comparison.
- [ ] Mark c2 as a matched-example regime control.
- [ ] Mark regression as a seed-sensitive sign interaction, not equivalence.
- [ ] Create a holdout analysis specification with primary, secondary, and diagnostic endpoints.
- [ ] Freeze table shells and figure shells before the holdout is evaluated.
- [ ] Write outcome-independent success/failure language for the untouched confirmation.

**Gate 1:** every headline sentence maps to an artifact and an appropriately scoped estimand; the holdout protocol is frozen and timestamped.

### Phase 2 — Evaluation and data-validity remediation

**Objective:** make the metric suite sensitive to hallucinations, missing objects, identity, coverage, and reach.

This phase is detailed in Sections 5 and 6.

**Gate 2:** all evaluator invariance tests pass on CPU; hand-constructed failure cases receive the expected penalties; no metric depends on left-to-right object order.

### Phase 3 — Run identity, provenance, and reproducibility

**Objective:** make every new result immutable, reconstructable, and safe from smoke-run or resume collisions.

This phase is detailed in Section 7.

**Gate 3:** a CPU smoke run produces a self-contained run directory, refuses overwrite, resumes trajectory-reproducibly, and validates artifact metadata before comparison.

### Phase 4 — Untouched confirmation

**Objective:** confirm only the frozen central claims on a truly untouched set.

Tasks:

- [ ] Prefer a newly generated confirmation set with a recorded generator commit, configuration, seed, source-glyph split, and SHA-256 manifest.
- [ ] Never use the confirmation set for qualitative monitoring, arm design, stopping, threshold selection, or debugging.
- [ ] Run the frozen evaluator once after all CPU and development-set checks pass.
- [ ] Retain the complete result, including failures, coverage, empty-frame strata, and false positives.
- [ ] Report current-pool results as development evidence and new-set results as confirmation evidence.
- [ ] Do not tune margins, bins, or exclusions after seeing confirmation outcomes.

**Gate 4:** the confirmation report is generated from the frozen specification and immutable artifacts without ad-hoc reruns.

### Phase 5 — Minimal targeted GPU programme

**Objective:** run only experiments that resolve a live causal ambiguity or support a central claim.

Priority ordering:

1. Untouched confirmation evaluation, usually requiring no retraining if dataset compatibility is preserved.
2. Fixed-target-interface B1 control if the stride/family mechanism remains central.
3. Long-loop trust replication and/or trust-isolation diagnostics if trust remains in the title.
4. Complete regression four-cell seed 2, followed by seeds 3–5 only if the interaction remains central and variable.
5. Horizon dose-response at approximately 24/48/96 rather than an undirected E-320 extension.
6. Redesigned Phase C only after all Phase C gates pass.
7. Counterfactual-world and dynamic-content factorials as the strongest next-paper direction.

**Gate 5:** each run has a preregistration, authorization status, CPU dry run, cost estimate, stop condition, run-spec hash, and named owner.

### Phase 6 — Article and dissertation reconstruction

**Objective:** make the written contribution match the validated evidence rather than the historical order of experimentation.

This phase is detailed in Sections 10 and 11.

**Gate 6:** all claims, tables, figures, references, and supplementary artifacts pass an automated marker and artifact audit plus a manual adversarial review.

### Phase 7 — External release and maintenance

**Objective:** turn strong internal research code into a reproducible public research package.

**Gate 7:** clean-environment CPU reproduction works from the README; manifests resolve; tests collect normally; no local absolute path is required; archive and citation metadata are complete.

## 5. Evaluation-suite implementation plan

### 5.1 Separate estimands instead of one overloaded `ALL`

Add record-level metrics with explicit populations:

| Metric | Population | Purpose |
|---|---|---|
| `mse_full` | every frame | total reconstruction/generation error |
| `mse_gt_fg_nonempty` | frames with non-empty GT foreground | legacy foreground fidelity without mixing empty records |
| `mse_gt_bg` | GT background pixels | false-positive and background corruption sensitivity |
| `mse_union` | union of GT and predicted foreground | penalizes missed and hallucinated content |
| `fg_precision`, `fg_recall`, `fg_f1` | all evaluable frames | foreground occupancy quality |
| component precision/recall | all evaluable frames | missed and extra objects |
| component IoU | matched components | shape/localization fidelity |
| identity accuracy | persistent labelled objects | identity retention/re-entry |
| trajectory error | persistent labelled objects | localization dynamics |
| count error | all labelled frames | missing/extra object sensitivity |
| scoring coverage | all requested records | prevents survivorship selection |

Empty-foreground records remain a separate, named stratum. They must never silently fall back into a differently defined foreground estimand.

### 5.2 Correct record-level implementation bugs

- Make sharpness a true per-image or per-record statistic; do not duplicate a batch mean across records.
- Store raw numerators/denominators or sufficient statistics where possible, not only reduced means.
- Store why a record is unscorable rather than only `NaN`.
- Do not intersect arms to create the primary population. Report each arm's coverage and use an intention-to-score policy with explicit penalties where scientifically justified.
- Preserve paired clip/horizon keys and cluster-bootstrap at clip level.
- Report the minimum number of unique clips/clusters supporting every interval.

### 5.3 Persistent-object Phase C redesign

Modify the generator and dataset schema to save, for every clip and object:

- stable object ID;
- source glyph index and class;
- official source split (`train` or `test`);
- initial appearance instance identifier;
- per-frame centre, bounding box, visibility, occlusion, and trajectory;
- z-order or overlap metadata if compositing makes it relevant;
- entry, exit, and re-entry events;
- generator configuration and seed.

Evaluator requirements:

- Use one-to-one Hungarian assignment with a documented cost combining localization, mask overlap, and optional class evidence.
- Never infer identity from left-to-right ordering.
- Penalize missed GT objects and extra predicted objects explicitly.
- Handle overlaps without merging identities in ground truth.
- Report assignment ambiguity and coverage.
- Validate any digit classifier on clean crops, translated crops, overlaps, decoder reconstructions, and generated samples.
- Seed and version the classifier; save its validation report and confusion matrices.
- Separate identity evidence from appearance-class evidence.
- Justify equivalence margins substantively in native units using empirical noise/reliability, not convenience.

Required CPU tests:

- two objects cross while identities remain stable;
- left/right order swaps;
- objects overlap and separate;
- one object disappears;
- an extra hallucinated object appears;
- all objects disappear;
- detections are permuted;
- a class is correct but instance identity is wrong;
- assignment ties occur;
- both arms have different coverage.

Phase C is authorized only when every test passes and the complete protocol has been preregistered.

### 5.4 Reach-differential instrumentation

Implement a generic pair-aware analysis from arm metadata rather than hard-coded names:

1. Derive the maximum visible temporal reach for each compared arm.
2. Define the causal band as records whose relevant past lies inside the longer arm's reach and outside the shorter arm's reach.
3. Require non-empty and geometrically answerable records for the designated primary reach estimand.
4. Report all-record and coverage diagnostics alongside it.
5. Use paired clip-level bootstrap intervals.
6. Save the derived band definition in result metadata.

The current `14 < dt <= 48` and `8 < dt <= 48` findings remain exploratory. These exact pair-specific definitions become frozen primary analyses for the untouched confirmation where the same arm geometries apply.

### 5.5 Observability terminology and counterfactual validity

- Rename the present label `geometric observability` where appropriate.
- Reserve `information-theoretic necessity` for paired counterfactual worlds that are indistinguishable over the short context but demand different re-entry outputs.
- Add a future benchmark mode with random instance textures, barcodes, or paired hidden state so class priors cannot reconstruct the answer reliably.

## 6. Untouched confirmation protocol

### 6.1 Preferred data design

Generate a new confirmation dataset after the analysis specification is frozen.

Required metadata:

- generator code commit and dirty-state hash;
- full generator configuration;
- RNG algorithm and seed;
- official MNIST split and source glyph indices;
- number of clips and frames;
- image dimensions and renderer version;
- SHA-256 for each data shard and a manifest hash;
- creation timestamp and creator;
- explicit `confirmation_holdout=true` marker.

The project may audit an untouched subset of the existing 500 clips as a fallback, but a new set is preferable because the existing pool has informed qualitative monitoring, theory, arm selection, and stopping decisions.

### 6.2 Frozen endpoints

Primary endpoints should be limited to the central causal claims:

- paired package difference on non-empty geometrically answerable records;
- pair-specific reach-differential effect;
- full-frame and union-mask corroboration;
- coverage and component miss/extra rates.

Secondary endpoints:

- GT-foreground MSE;
- background error;
- standard hardness strata;
- decoded sharpness;
- seed-specific effects.

Diagnostic endpoints:

- empty-frame stratum;
- classifier/assignment uncertainty;
- qualitative strips selected by a frozen sampling rule;
- per-horizon curves.

### 6.3 Holdout access policy

- The holdout loader should require an explicit `--confirmation-eval` switch.
- The switch records the run-spec hash, commit, and timestamp in an append-only access log.
- Training and monitoring commands must reject confirmation manifests.
- No confirmation images appear in training-time grids.
- Only the frozen analysis command may emit the official confirmation report.

## 7. Code, provenance, and repository work

### 7.1 Immutable run specification

Create a typed `RunSpec` that includes at minimum:

- dataset manifest hash;
- AE checkpoint and hash;
- arm registry ID and full resolved arm configuration;
- model configuration and parameter count;
- seed;
- train examples, optimizer updates, batch size, accumulation, precision;
- optimizer, scheduler, EMA configuration;
- context and target token counts;
- warmup, rollout horizon, evaluation clips, NFE/sampler configuration;
- source commit and dirty diff hash;
- software/environment fingerprint;
- free-form human tag that is not used as the sole identity.

Canonical identity is a content hash of the resolved run spec. Artifacts live under a run-specific directory, for example:

```text
runs/<study>/<arm>/<seed>/<run_spec_hash>/
  run_spec.json
  checkpoints/
  metrics/
  figures/
  logs/
  manifest.json
```

Requirements:

- refuse to write into an existing completed directory;
- write NPZ/JSON/checkpoints atomically via a temporary file and rename;
- give qualitative PNGs the same run identity as metrics;
- route smoke runs to a temporary or explicit smoke namespace;
- never allow `--smoke` to resolve to a canonical result filename;
- include the run-spec hash in W&B identity and local logs.

### 7.2 Checkpoint and resume correctness

Checkpoint all state required for trajectory-reproducible continuation:

- model, optimizer, scaler, scheduler, EMA;
- current update/example counters;
- Python RNG state;
- NumPy RNG state;
- Torch CPU and CUDA RNG states;
- data sampler/order state;
- resolved run spec and hashes;
- latent normalization/statistics;
- environment fingerprint.

Resume rules:

- resume is explicit (`--resume`), not unconditional filename discovery;
- a mismatch in any scientific field aborts unless an explicit noncanonical override is recorded;
- completed runs cannot silently resume or append;
- provide a CPU test comparing uninterrupted and interrupted/resumed trajectories.

Existing resumed runs should be described as distributionally reproducible, not bitwise trajectory-reproducible.

### 7.3 Single arm registry

Replace duplicated arm definitions across model code, documentation tables, shell scripts, and tests with one typed registry that can emit:

- resolved model/context configuration;
- context reach and token counts;
- target token counts and total sequence length;
- parameter count;
- table rows;
- CLI validation;
- comparison compatibility metadata.

The registry must distinguish:

- context temporal stride;
- target temporal stride;
- context spatial resolution;
- target interface;
- auxiliary losses;
- sampler/inference allocation.

Name current flat comparators accurately as single-stream context baselines when they still use hierarchical target objectives.

### 7.4 Comparison validation

Before a paired comparison, assert equality or explicitly report differences in:

- dataset and AE hashes;
- source split;
- clip/horizon keys;
- warmup;
- hardness/observability metadata;
- sampler and NFE;
- precision and decoding configuration;
- metric implementation version;
- arm target interface;
- train-example and optimizer-update budgets.

The report must state whether a comparison is matched by context tokens, total tokens, parameters, examples, updates, total NFE, or none of these. Never collapse these into the generic phrase “matched compute.”

### 7.5 Test and CI repair

- Convert script-style tests that execute on import or call `sys.exit` into normal pytest functions.
- Separate unit, integration, slow, external-data, and GPU tests with markers.
- Use `tmp_path` for all test outputs.
- Add CPU fixtures for generator, evaluator, arm registry, RunSpec, resume, comparison validation, and artifact manifests.
- Make normal `pytest` collection succeed without local data or a GPU.
- Add a small CPU CI job for lint/static checks, unit tests, smoke training/evaluation, and documentation-marker checks.
- Add a GPU workflow only as optional/manual infrastructure if a compatible runner exists.

### 7.6 Packaging and paths

- Add a root `README.md` with project identity, evidence status, quick start, reproduction tiers, and artifact retrieval.
- Add `pyproject.toml` and a locked environment strategy.
- Remove hard-coded local dataset and checkpoint paths; use CLI/config/environment resolution.
- Provide example CPU configurations and a documented GPU profile.
- Record supported Python, PyTorch, CUDA, and OS versions.
- Add a data/artifact manifest utility with SHA-256 validation.
- Add licenses/citation metadata and an archival release process.

### 7.7 Refactoring order

Do not perform a broad rewrite before claims and evaluation semantics are frozen. Refactor in this order:

1. extract pure metric and metadata functions with tests;
2. introduce RunSpec and immutable output routing;
3. introduce the arm registry;
4. split training, evaluation, reporting, and CLI orchestration from monolithic modules;
5. classify remaining modules as `current`, `legacy`, or `archived-negative`;
6. delete nothing until artifact reproduction and historical references have been checked.

### 7.8 Artifact policy

The current repository contains large untracked data/log evidence. Create:

- `artifacts/manifest.json` or an equivalent generated manifest;
- per-artifact SHA-256, size, type, producing command, run-spec hash, and claim IDs;
- an immutable external store or archival release for large evidence;
- a lightweight checked-in index that does not pretend unavailable local files are reproducible;
- retention rules for checkpoints, raw NPZs, derived tables, figures, and transient logs.

Before deleting or relocating anything, take a verified backup and reconcile it with the claim ledger.

### 7.9 Initial file-by-file implementation map

This map identifies the first expected edit sites. It is a guide, not permission to preserve poor module boundaries indefinitely.

| Existing location | Required change | Likely extracted/new component |
|---|---|---|
| `hvp_rev/long_eval.py` | split empty/non-empty populations; add full/background/union/component metrics; correct per-record sharpness | `hvp_rev/evaluation/frame_metrics.py` |
| `hvp_rev/mmnist_metrics.py` | remove order-based identity and many-to-one matching; consume persistent metadata; emit coverage and miss/extra penalties | `hvp_rev/evaluation/object_metrics.py`, `hvp_rev/evaluation/assignment.py` |
| `hvp_rev/phasec_eval.py` | use versioned dataset schema and corrected evaluator; save raw record-level outputs and reasons | `hvp_rev/evaluation/phasec.py` |
| `hvp_rev/phasec_summary.py` | remove finite-arm intersection as the primary policy; report coverage and justified equivalence analyses | `hvp_rev/reporting/phasec_report.py` |
| `scripts/gen_moving.py` | configurable roots; official source split; persistent object/glyph/trajectory metadata | `hvp_rev/data/moving_mnist.py` plus schema tests |
| `scripts/gen_looping.py` | manifest/config output; confirmation-holdout marker; no hard-coded paths | `hvp_rev/data/looping.py` plus manifest utility |
| `hvp_rev/rhvd.py` | remove collision-prone names and unconditional autoresume; separate train/eval/report CLI; consume registry/RunSpec | `hvp_rev/config/run_spec.py`, `hvp_rev/cli/`, `hvp_rev/training/` |
| `hvp_rev/compare.py` and analysis scripts | validate fingerprints and estimands; expose cluster counts and reach bands | `hvp_rev/evaluation/compare.py`, `hvp_rev/evaluation/reach.py` |
| `scripts/run_newvision_queue.sh` and `.ps1` | retire auto-advance into blocked studies; require authorization token/run spec | one guarded queue launcher or scheduler adapter |
| `scripts/run_phaseC_mmnist.sh` | hard block legacy schema; accept only validated preregistration/run spec | redesigned Phase C launcher after Gate 2–3 |
| `scripts/make_cwstyle_table.py`, `make_fig9_frontier.py`, `make_fig13_family_edges.py`, `make_fig14_level_blinding.py` | consume manifests; show scope/seed/regime/uncertainty; remove overclaiming labels | shared reporting/table helpers |
| script-style tests including model/layout probes | normal pytest functions, markers, `tmp_path`, no import-time exit | `tests/unit/`, `tests/integration/`, `tests/gpu/` |
| `docs/CLAIM_LEDGER.md` | add frozen wording, status, estimand, seeds, evidence hash, confirmation status | remains scientific source of truth |
| `docs/SPATIAL_FACTORIAL.md` | relabel B1/configuration, c2/regime, edges/brackets, regression interaction | historical detail retained with corrected summaries |
| `docs/draft/ARTICLE_SKELETON.md` and `CH1`–`CH8` | restructure around reach–fidelity–trust; move peripheral strands; update literature and claims | final article source or generated assembly |
| `docs/dissertation/rewrite/00_FRONT_MATTER.md`–`07_REFERENCES.md` | synthesis, evidence boundaries, administrative completion, terminology/reference pass | final dissertation source |
| repository root | README, environment lock, pyproject, CI, licenses/citation, artifact index | public reproduction entry point |

Before moving code, add golden tests around current intended behavior and explicit failing tests around the defects being corrected. Preserve legacy readers for old NPZ/checkpoint schemas long enough to regenerate every published table.

## 8. Causal controls and redesigned experiments

### 8.1 Fixed-target-interface B1 control

Purpose: isolate context packaging/stride from target-interface and optimization changes.

Design requirements:

- identical target token count and target temporal interface across compared arms;
- identical auxiliary target losses or a fine-output-only objective for both arms;
- matched context-token budget where claimed;
- report total sequence length separately;
- same model-size, examples, updates, EMA, sampler, and evaluation settings;
- preregister the contrast and relevant reach-differential band.

Interpretation:

- Current B1 remains the replicated whole-configuration effect.
- The control estimates the incremental effect under a fixed target interface.

### 8.2 Regression head x package factorial

The four cells are diffusion/regression crossed with temporal/spatial package. Analyse the difference-in-differences per seed.

- Finish seed 2 only if the interaction is retained as a central article result.
- Add seeds 3–5 only under a written rule based on interval width/heterogeneity, not the desired sign.
- Report each seed and a hierarchical/random-effects summary if sample size supports it.
- Never describe opposite significant seed effects as equivalence.

### 8.3 Trust isolation and horizon dose-response

Replace the weakly targeted E-320 extension with horizons such as 24, 48, and 96, chosen before results.

Add diagnostics that distinguish:

- addressable memory under teacher forcing;
- degradation from self-fed coarse state;
- decoder/representation degradation;
- sampler accumulation.

Candidate interventions:

- periodic ground-truth coarse-buffer refresh;
- controlled latent corruption at known amplitudes;
- self-forcing or rollout-aware training;
- fixed sampler seed and multiple sampler-seed evaluation;
- matched total NFE versus matched per-step NFE.

If trust remains a title-level axis, replicate the decisive long-loop contrast at another seed or task/horizon.

### 8.4 Representation sensitivity

Test whether the observed frontier is tied to a particular autoencoder or codec:

- at least one alternate AE seed;
- simple spatial pooling/downsampling baseline;
- optionally another codec or latent dimensionality;
- decoded reconstruction references for each representation.

This can be secondary unless the article makes architecture-independent claims.

### 8.5 Counterfactual and dynamic-content benchmark

This is the strongest directional extension, likely a follow-up study rather than a prerequisite for the current article.

Factorial design:

- static versus dynamically evolving off-screen content;
- low versus high spatial-detail identity;
- short and long re-entry delays;
- temporal-rate versus spatial-fidelity packages.

Add paired counterfactual worlds that are identical over the visible short context but differ in the hidden state that determines re-entry. This supports necessity claims that ordinary MNIST class priors cannot.

### 8.6 Baselines for defensibility

Add or clearly discuss:

- persistence/copy baseline;
- classical pose-indexed retrieval, panorama/mosaic, or action-warp memory;
- learned retrieval/compression baseline;
- self-forcing/rollout-training baseline for trust;
- parameter counts and total inference allocations for every neural baseline.

The goal is not a baseline zoo. Each baseline should answer whether the effect requires hierarchical generation, learned compression, or merely access to the relevant past.

## 9. Experiment authorization matrix

This table is the default policy. Any change requires a dated ledger entry and named owner.

| Experiment | Status | Why | Preconditions |
|---|---|---|---|
| active seed-2 trio | **AUTHORIZED / FINISH** | sunk cost, healthy, directly closes registered cells | preserve logs and artifacts |
| c2 B8x60k | **AUTHORIZED / RUN ALONE** | preregistered regime robustness control | old queue stopped; no Phase C child |
| existing Phase C queue item | **PROHIBITED** | invalid identity/trajectory/coverage semantics | full redesign and Gate 2–3 |
| untouched confirmation evaluation | **REQUIRED** | repairs development-contaminated confirmatory status | Gates 1–3; frozen holdout protocol |
| fixed-target B1 control | **CONDITIONAL HIGH VALUE** | isolates the main B1 confound | design/preregistration/CPU smoke |
| regression seed 2 factorial | **CONDITIONAL** | tests promising head x package interaction | centrality decision; complete four cells |
| regression seeds 3–5 | **CONDITIONAL** | estimates heterogeneity | outcome-independent expansion rule |
| long-loop replication | **CONDITIONAL HIGH VALUE** | required for title-level trust claim | frozen trust estimands and horizon |
| 24/48/96 horizon dose-response | **PREFERRED NEW STUDY** | tests the independent variable directly | trust design and budget approval |
| E-320 as currently conceived | **DEPRIORITIZED** | less diagnostic than horizon dose-response | specific unresolved hypothesis required |
| current B3 expansion | **DEPRIORITIZED / APPENDIX** | weak relevance to central paper | only if a concrete gap emerges |
| redesigned Phase C | **CONDITIONAL** | valuable only with valid labels/assignment | persistent IDs, classifier validation, margins, Gates 2–3 |
| alternate AE/codec | **CONDITIONAL** | representation robustness | central generality claim retained |
| counterfactual dynamic benchmark | **FUTURE HIGH VALUE** | strongest necessity/generalization extension | separate protocol and resources |
| natural/geometry-grounded flagship | **FUTURE** | external relevance | synthetic mechanisms stabilized first |

## 10. Article reconstruction plan

### 10.1 Proposed structure

1. **Problem and thesis:** bounded video memory forces choices in reach, fidelity, and self-fed reliability.
2. **Controlled instrument:** looping generator, geometric observability, context-packaging families, budgets, and estimands.
3. **Reach–fidelity frontier:** matched-budget tested configurations, seed-resolved results, and appropriately bracketed inversions.
4. **Mechanism:** pair-specific reach-differential analysis and the B1 configuration result/control.
5. **Representation and trust:** head x package interaction, seed sensitivity, and addressable-versus-self-fed boundary.
6. **Confirmation, limits, and implications:** untouched holdout, false-positive-sensitive metrics, prior-art comparison, and scoped conclusions.

Move EBT, reversibility, B3, historical queue narrative, and detailed preregistration chronology to appendices, the dissertation, or separate papers.

### 10.2 Required tables

**Table A — Configuration and cost**

- stream structure;
- temporal span/stride;
- spatial resolution;
- context tokens;
- target tokens;
- total sequence length;
- parameters;
- training examples and optimizer updates;
- EMA setting;
- NFE and sampler allocation;
- runtime and peak VRAM where measured.

**Table B — Claim and evidence**

- claim ID;
- contrast and estimand;
- per-seed effect and interval;
- reach-differential effect where applicable;
- development versus confirmation status;
- intervention scope and limitations.

**Table C — Prior-art positioning**

- memory mechanism;
- retrieval/compression;
- matched budget;
- analytic observability;
- dynamic re-entry;
- self-fed evaluation;
- realism/task domain.

### 10.3 Required figures

- Frontier figure with individual seeds and separate B16x30k versus B8x60k regimes; do not interpolate a unique crossing.
- Reach-differential figure showing the exact pair-specific causal bands and coverage.
- Metric-decomposition figure separating foreground, background, union, component misses/extras, and full-frame behavior.
- Trust figure separating teacher-forced addressability from rollout reliability by horizon and seed.
- Head x package interaction figure with all four cells and seed-resolved contrasts.
- Qualitative strips generated by a frozen selection rule and uniquely tagged with run identity.

All tables and plots must be generated from artifact manifests/NPZs, never from hard-coded manuscript numbers.

### 10.4 Statistical reporting

- State the unit of analysis and bootstrap cluster.
- Report per-seed effects rather than only pooled means.
- Report clip counts, record counts, coverage, and number of unique clusters.
- Separate confirmatory and exploratory analyses visibly.
- Use equivalence testing only with justified margins and valid coverage policy.
- Do not interpret a confidence interval excluding zero as practical importance by itself.
- For multiple related secondary endpoints, state whether inference is descriptive or adjusted.

### 10.5 Literature and novelty

Conduct a fresh primary-source sweep immediately before submission. At minimum position against:

- CW-VAE;
- flexible diffusion/video context work such as FDM;
- recent long-context state-space video models;
- spatial-memory world models;
- WorldPack;
- Self Forcing;
- HyDRA/HM-World;
- MBench;
- CaR/SceneFly;
- MemLearner;
- GIM-World.

The defensible novelty is the combination of:

- inexpensive analytic geometric observability labels;
- matched-budget rate-versus-resolution comparisons;
- tested lower and upper inversions;
- pair-specific reach-differential instrumentation;
- separation of addressable memory from self-fed reliability;
- unusually explicit negative-result and boundary reporting.

Do not claim that prior work lacks memory-focused evaluation, compression, retrieval, or autoregressive mismatch research.

### 10.6 Final manuscript hygiene

- Remove lab-notebook prose, queue state, local paths, and commit hashes from the main article.
- Put reproducibility commands, preregistrations, and artifact hashes in the supplement/repository.
- Resolve all `EXPECTED`, `AUTHOR`, `REF`, `VERIFY`, figure, and citation markers.
- Correct terminology for persistence baselines and AE reconstruction references.
- Revisit blinding controls: out-of-distribution corruption can inflate loss and is not a lower bound on stream contribution. Prefer in-distribution shuffle/freeze/cross-clip controls or a trained missing-stream condition.
- Ensure every abstract/conclusion claim appears in the claim ledger with matching scope.

## 11. Dissertation reconstruction plan

### 11.1 Integrating thesis

Present the dissertation as a layered toolbox for reliable long-horizon video prediction, not as a single already-integrated production system.

Create a synthesis table with columns:

- dissertation problem;
- proposed instrument or method;
- strongest evidence;
- known boundary;
- relation to reach, fidelity, or trust;
- degree of integration in the final executable pipeline.

### 11.2 Chapter-level corrections

- Keep V-BLT/RoPE as its own representation/sequence strand; do not imply the final predictor uses it if learned embeddings are used.
- Do not present reversibility as the cause of Chapter 4 hierarchy gains without a direct experiment.
- Narrow reversibility novelty in light of Reversible RNNs: focus on stochastic conditioning, noise replay, or demonstrated applicability.
- Ensure the task labelled “integration” either corresponds to an executable demonstrator or is softened to design-level integration.
- Treat EBT, reversibility, and B3 as supporting/negative strands rather than forcing them into the central article claim.
- Synchronize chapter conclusions, overall conclusions, novelty statements, and the evidence ledger.

### 11.3 Administrative and language completion

- Complete the Ukrainian authoring/editing pass with technical terminology kept consistent.
- Replace donor/example publication lists and every author/admin placeholder.
- Rebuild front matter, contents, abbreviations, references, and publication mapping.
- Ensure figures and tables are final, numbered consistently, and licensed/cited.
- Preserve honest implementation acts and evidence archive references.
- Conduct a final external-reader pass for overclaiming and internal consistency.

Administrative placeholders are release blockers, not cosmetic cleanup.

## 12. Repository maintenance model

### 12.1 Source-of-truth hierarchy

1. `docs/CLAIM_LEDGER.md` — scientific claim status and evidence links.
2. `docs/EXPERIMENT_CONTROL.md` — queue owner, authorization, current/next run.
3. run-spec and artifact manifests — immutable technical provenance.
4. this master plan — ordering and acceptance gates.
5. older plans/dossiers — historical context, not current authority where conflicts exist.

### 12.2 Change control

For each material change:

- open a dated design note or ledger entry;
- state whether it affects an estimand, implementation, presentation, or only maintenance;
- update tests before or with semantic code changes;
- regenerate derived tables/figures through scripts;
- record affected claim IDs;
- obtain a second adversarial review before holdout evaluation or manuscript freeze.

### 12.3 Commit sequence

Use small, reviewable commits in approximately this order:

1. integrate and hash current seed-2/c2 outputs;
2. claim-language and authorization-ledger corrections;
3. evaluator tests exposing current failures;
4. persistent metadata and corrected assignment;
5. expanded metric suite and reach-differential analysis;
6. RunSpec/output immutability/resume fixes;
7. arm registry and comparison validation;
8. pytest/CPU CI/environment/README;
9. untouched confirmation protocol and frozen table shells;
10. confirmation results;
11. only then, authorized targeted experiments;
12. article/dissertation regeneration and release archive.

Do not mix live result integration with broad refactoring in one commit.

## 13. Workstream acceptance criteria

### Scientific claims

- Every headline claim names the tested population, contrast, seeds, metric, and limitation.
- No unique crossing, equivalence, law, or pure intervention is claimed without the required design.
- Development and confirmation evidence are visibly distinct.

### Evaluation

- Hallucinated foreground is penalized.
- Empty frames do not change the meaning of a foreground metric.
- Object crossings and overlaps do not change GT identity.
- Misses, extras, ambiguity, and coverage are reported.
- Pair-specific reach bands derive from arm geometry.

### Code and provenance

- Smoke runs cannot overwrite canonical artifacts.
- Qualitative images and metrics share a unique run identity.
- Resume restores RNG and sampler state and rejects incompatible specs.
- `pytest` collects and passes the CPU suite without external data/GPU.
- No scientific command depends on a developer-specific absolute path.

### Experiments

- One owner controls the queue.
- Every GPU run is explicitly authorized and preregistered.
- Phase C cannot launch through an old queue path.
- Stop conditions and resource estimates are recorded before launch.

### Article and dissertation

- Tables and figures are artifact-generated.
- Literature claims reflect current primary work.
- No unresolved author/reference/expected-result marker remains.
- Article and dissertation use the same claim status and terminology.

### Release

- A clean environment can run CPU smoke reproduction from the README.
- Every published number resolves to an immutable artifact hash.
- Large artifacts have an accessible archival location and checksum manifest.

## 14. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| old queue launches Phase C | costly invalid run | watcher/kill boundary, verify children, remove stale lock, no auto-queue reuse |
| two experiment owners | duplicate/conflicting runs | explicit single-owner handoff and experiment-control ledger |
| holdout is viewed during debugging | confirmation becomes development data | access switch/log, manifest rejection in training, frozen protocol |
| evaluator redesign changes conclusions | apparent inconsistency | preserve legacy metric, report side-by-side, explain construct correction |
| false positives reverse some effects | narrower claims | accept outcome; treat complete metric suite as authoritative |
| persistent-ID generator breaks compatibility | delayed Phase C | version schema, migration reader, CPU fixtures, do not mutate old data |
| broad refactor introduces scientific drift | irreproducibility | freeze semantics first, extraction with golden tests, small commits |
| c2/resumed paths are not trajectory-reproducible | overstated robustness | describe as regime/distributional robustness; fix RNG for future runs |
| literature moves quickly | novelty overclaim | final primary-source sweep immediately before submission |
| trust remains one-seed | title overstates generality | replicate or demote trust to measured boundary |
| artifact volume becomes unmanageable | lost evidence | manifest, retention policy, immutable external archive |

## 15. Recommended ownership and handoff protocol

The project should have a single experiment execution owner after the current watcher completes. Transferring ownership is useful if the same owner will implement the evaluator/provenance gates and control the next queue; it should not happen mid-run by starting a second controller.

Required handoff packet:

- current branch, commit, and dirty-file list;
- active/completed process list and GPU state;
- final watcher and c2 logs;
- queue completion markers and lock state;
- list of checkpoints/NPZs/figures produced since the dossier commit;
- hashes and run commands;
- preregistration texts;
- known failures/retries/resumes;
- experiment authorization matrix signed off by the user;
- explicit statement: `NEW OWNER IS THE ONLY PARTY AUTHORIZED TO LAUNCH GPU RUNS`.

The previous owner may continue reviewing text or results, but should not launch experiments after handoff unless ownership is explicitly transferred back.

## 16. First concrete implementation sprint

The first sprint after c2 should be entirely CPU-first and should not consume GPU time.

1. Integrate and hash seed-2/c2 outputs without altering their content.
2. Create `docs/EXPERIMENT_CONTROL.md` and update the claim ledger.
3. Add failing unit tests for crossings, overlap, misses, extras, coverage, empty foreground, and sharpness records.
4. Version the dataset schema and add persistent object metadata.
5. Implement Hungarian assignment and the expanded metric suite.
6. Implement pair-specific reach-band derivation.
7. Introduce RunSpec, immutable output directories, and explicit resume.
8. Repair pytest collection and add the CPU smoke path.
9. Draft and freeze the untouched-confirmation protocol and table shells.
10. Run an adversarial CPU audit; only then request authorization for confirmation and any targeted GPU work.

This sprint resolves the largest scientific and operational risks while preserving all existing evidence.

## 17. Definition of project completion

The current project is complete when:

- central reach/fidelity claims survive or are honestly revised after untouched confirmation;
- the trust claim is replicated or explicitly scoped as a boundary case;
- Phase C is either validly redesigned and run or removed from the core claim set;
- the B1 confound is controlled or its family-level interpretation is maintained consistently;
- complete false-positive-sensitive metrics are reported;
- every result has immutable provenance and a claim-ledger entry;
- CPU tests and clean-environment reproduction work;
- the article tells one coherent reach–fidelity–trust story;
- the dissertation accurately synthesizes all strands without false integration;
- all placeholders, references, figures, tables, and administrative fields are final;
- an external reviewer can trace every published number from prose to artifact to producing configuration.

Until these conditions are met, additional broad experiment expansion is lower priority than measurement validity, confirmation, and provenance.
