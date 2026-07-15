# Implementation report: reach–fidelity–trust remediation

> **Historical implementation record.** Superseded for live status and launch order by
> `docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md` and
> `experiments/authorization.json`. Preserve it as provenance, not authority.

**Date:** 2026-07-11
**Scope:** code, experiment control, evaluation, data, provenance, tests, fourth article,
dissertation synthesis, literature, and remaining studies
**Live-run constraint:** all implementation commands were CPU-only; the active pre-C GPU process
was not modified or interrupted

## 1. Outcome

The project now has a fail-closed scientific preparation path. The fourth article has a new
canonical reach–fidelity–trust narrative; the claim ledger no longer treats tested brackets,
configuration effects, seed cancellation, or invalid Phase C endpoints as stronger evidence than
they are. The dissertation topic and [A1]–[A3] publication strands are preserved and connected by a
layered synthesis rather than retroactive content changes.

The legacy GPU trainer remains temporarily active only for the already-running seed-2/c2 sequence.
New evaluation, provenance and control modules are implemented alongside it and CPU-tested. They
will replace the legacy output/resume path only after c2 completes, because changing `rhvd.py`
before the later pre-C arms start would make the seed-2/c2 cells execute different code from the
earlier cells.

## 2. Implemented code changes

### 2.1 Complete frame evaluation

`hvp_rev/evaluation/frame_metrics.py` now emits record-level:

- full-frame MSE;
- non-empty GT-foreground MSE with `NaN` rather than silent fallback on empty GT;
- GT-background and union-mask MSE;
- foreground precision, recall and F1;
- per-record foreground Laplacian energy;
- GT/predicted components, misses, extras, IoU and centroid error.

`hvp_rev/rollout_artifacts.py` writes those arrays atomically with clip/horizon and geometric labels.
This fixes the construct semantics and the batch-mean sharpness problem for all new artifacts.

### 2.2 Persistent-object scoring

`hvp_rev/evaluation/object_metrics.py` uses one-to-one Hungarian assignment over normalized
centroid distance and bounding-box overlap. It is invariant to detection ordering and retains:

- misses and extras;
- intention-to-score trajectory penalties;
- identity failures caused by missed labelled objects;
- identity and object coverage;
- matched-only diagnostics and assignment records.

The tests explicitly cover crossings/order swaps, permutations, missing objects, extra objects and
empty predictions.

### 2.3 Persistent Moving-MNIST schema

`hvp_rev/data/moving_mnist.py` and `scripts/gen_moving_v2.py` generate:

- stable object IDs;
- source glyph indices and class labels;
- official source-split provenance;
- per-object trajectories and boxes;
- overlap state;
- versioned metadata and SHA-256 manifest;
- separate development versus confirmation purpose.

Dataset output is immutable and refuses overwrite. The Phase C v2 contract requires MNIST train
glyphs for training clips and official test glyphs for evaluation clips.

### 2.4 Reach and factorial analysis

- `hvp_rev/evaluation/reach.py` derives pair-specific causal bands from arm geometry.
- `hvp_rev/analysis.py` provides clip-clustered paired effects and a head × package
  difference-in-differences with explicit sign convention.
- `hvp_rev/comparison_v2.py` rejects dataset, AE, split, metric, horizon, sampler, NFE and precision
  mismatches and reports which cost definitions are actually matched.
- `hvp_rev/trust_diagnostics.py` provides seeded latent corruption, refresh schedules and the
  addressable-to-rollout trust penalty.

### 2.5 B1 fixed-target causal control

`hvp_rev/arm_registry.py` separates context stride from target stride. The prepared controls are:

- `hier-s2-fixed-target`: 512 context tokens, 48-frame reach;
- `flat-short-fixed-target`: 512 context tokens, 8-frame reach;
- both: four coarse plus four fine target slots, 320 target tokens and identical loss weights.

`hvp_rev/fixed_target_control.py` implements the decoupled training batch, per-frame coarse target
interface and correct strided rollout-buffer anchoring. It is CPU-tested with loss/backpropagation.

### 2.6 Immutable run and checkpoint provenance

`hvp_rev/provenance.py` implements:

- content-addressed `RunSpec` identity;
- dataset/AE/config/seed/examples/updates/batch/precision/NFE/horizon/commit/metric fields;
- atomic JSON, NPZ and checkpoint writes;
- completed-run overwrite refusal;
- manifest hashes;
- Python, NumPy, Torch CPU/CUDA RNG capture and restoration;
- scientific compatibility checks on resume;
- confirmation-purpose validation and append-only holdout access logs.

Existing legacy checkpoints remain distributionally, not trajectory-exactly, reproducible.

### 2.7 Experiment authorization

`experiments/authorization.json` is now the launch authority. It records:

- seed-2 trio: finish only;
- c2: run alone;
- legacy Phase C: prohibited;
- confirmation, fixed-target B1, regression seed 2, trust horizons and Phase C v2: blocked or
  conditional.

Every new launcher calls `scripts/check_experiment_gate.py`. The legacy Phase C bash script also
fails before data generation or GPU work. Direct legacy Phase C evaluation and summary commands
exit unless a forensic-only invalid-legacy switch is supplied.

Prepared gated scripts:

- `scripts/generate_confirmation_set.ps1`;
- `scripts/run_b1_fixed_target.ps1`;
- `scripts/run_regression_factorial_seed2.ps1`;
- `scripts/run_trust_horizon_dose_response.ps1`;
- `scripts/run_phasec_v2.ps1`.

Training launch bodies remain deliberately disabled while their statuses are not authorized.

### 2.8 Classifier gate

`hvp_rev/digit_classifier.py`, `hvp_rev/classifier_validation.py` and
`scripts/validate_phasec_classifier.py` provide seeded training and domain-stratified validation.
The gate requires clean, translated, rendered-component, AE-reconstruction and generated/degraded
performance plus overlap diagnostics, confusion matrices and calibration. Clean-MNIST accuracy
alone cannot authorize Phase C.

### 2.9 Package and maintenance structure

Added:

- root README with research identity, evidence status and reproduction tiers;
- `pyproject.toml` and a CPU environment snapshot;
- CPU GitHub Actions workflow;
- explicit pytest markers;
- SHA-256 artifact-manifest builder;
- generated-gate documentation;
- configurable v2 dataset paths rather than developer-specific constants;
- CPU-safe pytest replacement for the prior import-time/external-data script.

The `.gitignore` now ignores only the root data/log directories, so `hvp_rev/data/` is correctly
versionable.

## 3. Experiment and analysis protocols

Implemented documents:

- `docs/EXPERIMENT_CONTROL.md` — one-owner handoff and launch policy;
- `docs/UNTOUCHED_CONFIRMATION_PROTOCOL_2026-07-11.md` — frozen populations, endpoints and
  outcome-independent interpretation branches;
- `docs/PHASE_C_V2_PROTOCOL_2026-07-11.md` — persistent-object, assignment, classifier and
  equivalence gates;
- `experiments/configs/*.json` — B1 fixed target, regression factorial, trust horizon and Phase C
  configurations.

`experiments/evidence_snapshot_2026-07-11.json` is the structured source for
`docs/draft/RFT_GENERATED_TABLES.md`, which includes:

1. configuration/interface/cost distinctions;
2. seed-resolved claim/evidence status;
3. prior-art comparison.

Pending values remain explicit instead of being inserted as expected results.

## 4. Article changes

`docs/draft/ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md` is the canonical fourth article. It is
organized as:

1. empirical problem and four research questions;
2. controlled context-packaging instrument;
3. complete endpoints and inference;
4. development evidence;
5. boundaries and prior-art positioning;
6. outcome-independent conclusion and required confirmation inserts.

The central wording is:

> Context packaging is the manipulated design space; reach, fidelity and trust are explanatory
> axes to be tested.

The text now treats:

- reach-differential effects as post-hoc development evidence pending confirmation;
- 1,792/2,432 as tested configurations, not a unique crossing;
- B1 as a configuration effect pending a fixed-target control;
- c2 as changed optimizer regime at matched examples;
- regression as seed-sensitive signs plus a promising interaction;
- trust as a measured boundary pending replication;
- geometric observability as distinct from information-theoretic necessity;
- Moving-MNIST/TaxiBJ as descriptive anchors;
- EBT, reversibility and B3 as supplementary/dissertation strands.

The old chapter files are retained as historical source maps and marked noncanonical so evidence is
not lost while obsolete wording cannot silently return at assembly.

## 5. Dissertation changes

The approved topic, purpose/tasks and [A1]–[A3] strands were not changed.

`docs/dissertation/REACH_FIDELITY_TRUST_SYNTHESIS_2026-07-11.md` maps:

- [A1] adaptive decomposition to fidelity/resource allocation;
- [A2] reversibility to long-horizon computational feasibility;
- [A3] consistency criteria to reliability definition;
- [A4] context packaging to the direct reach–fidelity–trust study.

The introduction, experimental chapter and conclusions now state that integration is layered. The
Chapter 4 Phase C expected-equivalence block was replaced by the conditional v2 protocol. The
conclusion no longer uses universal-law language for the fourth article. Reversibility remains an
enabler and is not claimed as the cause of the memory results.

Administrative author fields, publication metadata and the Ukrainian final pass remain authoring
work because they require facts or decisions not present in the repository.

## 6. Literature update

The related-work boundary now explicitly acknowledges:

- WorldPack/LoopNav compressed memory and retrieval;
- MBench memory evaluation;
- CaR/SceneFly context compression and implicit retrieval;
- MemLearner adaptive context queries;
- GIM-World geometry-aware implicit memory;
- MemoBench dynamic disappear/reappear evaluation;
- Self Forcing rollout-aware training.

Verified arXiv titles/authors were inserted into the dissertation reference inventory. This
prevents a broad “prior work lacks memory evaluation” novelty claim. The novelty is the controlled
analytic instrument, matched-budget package inversions, pair-specific reach bands, and separation
of addressability from rollout reliability.

## 7. CPU validation results

At this report revision:

- `pytest`: **30 passed**;
- Python compile check: passed;
- scientific configuration validator: passed;
- reach–fidelity–trust text consistency gate: passed;
- legacy Phase C summary block: passed (the command exits before reading artifacts);
- `git diff --check`: passed;
- CUDA was hidden with `CUDA_VISIBLE_DEVICES=-1`.

The final generated `gates/cpu_validation.txt` binds the pass to a commit/diff hash. It must be
regenerated after any semantic change.

## 8. Intentionally deferred post-c2 integration

The following changes are designed and tested but not switched into the live `rhvd.py` path while
the pre-C queue is running:

1. replace collision-prone result/checkpoint names with `RunSpec` directories;
2. make resume explicit and RNG-complete;
3. route rollout frames through the complete immutable artifact writer;
4. replace duplicated arm tuples with the typed registry;
5. validate metadata before every comparison;
6. use per-record sharpness rather than the legacy batch mean;
7. prevent smoke/qualitative overwrite through run-specific output routing;
8. retire the old master queue's Phase C line after its final historical run.

Reason: remaining seed-2 and c2 processes import `rhvd.py` when each arm starts. Switching now
would create a mixed-code experiment. Integration is the first CPU task after the c2 handoff and
should take roughly **4–8 focused hours**, followed by a CPU smoke/regression pass.

## 9. Remaining experiments and meaning

Estimates assume one RTX 4060 Ti-class 16 GB GPU, sequential execution, and no WDDM spill. They are
planning ranges, not promises.

| Priority | Study | Scientific meaning | Estimated remaining time |
|---:|---|---|---:|
| 0 | Finish seed-2 trio + c2 | closes registered seed/regime cells; c2 tests batch/update/EMA sensitivity at matched examples | other agent estimate: about **23–26 h from the 15:36 snapshot**, conservative ~29 h |
| 1 | Post-c2 artifact integration and freeze | makes later results immutable and construct-valid; no scientific outcome | **4–8 CPU h** |
| 2 | Untouched confirmation | determines whether central reach effects survive a fresh set and false-positive-sensitive metrics | dataset/freeze **2–4 CPU h**; model evaluation approximately **4–8 GPU h** if no retraining |
| 3 | Fixed-target B1 control | isolates context packaging from target slots/loss/sequence changes | 2 arms × 3 seeds: approximately **20–25 GPU h** plus 1–2 h analysis; a two-seed first stage ~14–17 h |
| 4 | Trust replication + 24/48/96 dose response | tests reliability decay directly and decides whether trust remains title-level | evaluation/refresh/corruption **4–8 GPU h**; rollout-aware retraining, if added, **12–30 GPU h** |
| 5 | Regression seed-2 factorial | tests the same-direction head × package shift and seed heterogeneity | likely two missing regression cells: **7–10 GPU h** including evaluation |
| 6 | Phase C v2 | tests long Moving-MNIST identity/trajectory only with valid assignment and coverage | classifier/data **3–6 CPU h**; six training arms/seeds + long evaluation approximately **30–38 GPU h** |
| 7 | Representation sensitivity | tests whether the frontier is tied to one AE/codec | approximately **15–30 GPU h** depending on reused arms |
| 8 | Counterfactual static/dynamic benchmark | strongest test of necessity and data-dependent package crossover | **2–5 development days** plus roughly **30–60 GPU h** for a first controlled factorial |
| 9 | Natural re-entry flagship | external relevance beyond synthetic geometry | separate study; likely **weeks**, not a current-paper prerequisite |

Recommended stopping logic:

- confirmation is required;
- fixed-target B1 is required only if a context-stride causal interpretation remains central;
- trust replication is required if “Trust” remains in the title;
- regression seed 2 is required only if the head interaction remains central;
- Phase C v2 is optional for the present article and should not delay submission unless synthetic
  cross-dataset identity retention is a headline claim;
- representation, counterfactual and natural-video work are follow-up directions.

## 10. Non-experimental remaining work

- integrate completed seed-2/c2 artifacts and update generated tables without touching raw output;
- generate a checked evidence manifest and external archive location;
- replace remaining dissertation `[AUTHOR]` fields using verified author information;
- perform the Ukrainian final-language pass;
- verify all remaining bibliography placeholders against primary PDFs/publisher records;
- assemble final article figures after confirmation;
- build or soften the claimed end-to-end demonstrator/integration task;
- run the final no-`EXPECTED`/no-`PENDING` assembly gate.

## 11. Recommended next decision

Allow the current watcher to finish seed 2 and c2. Transfer sole experiment ownership after the
handoff. Then perform the 4–8-hour post-c2 integration and artifact freeze before authorizing the
untouched confirmation. Do not authorize Phase C v2 merely because its code skeleton exists; its
official test glyph files, classifier-domain fixtures, margins/policy and protocol marker are not
yet complete.
