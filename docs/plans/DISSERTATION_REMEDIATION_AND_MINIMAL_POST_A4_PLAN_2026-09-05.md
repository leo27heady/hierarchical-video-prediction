# Dissertation corrections and minimal post-A4 experiment plan

Date: 5 September 2026. Status: proposed implementation plan; storage migration completed as recorded below. This document does not authorize new training or change the running campaign.

Basis: [independent review](../reviews/INDEPENDENT_DISSERTATION_AND_LIVE_EXPERIMENT_REVIEW_2026-09-05.md), findings R1–R18, and [reviewer handoff](CURRENT_DISSERTATION_AND_EXPERIMENT_REVIEW_AGENT_HANDOFF_2026-09-05.md). The review contains the source locations and supporting evidence. Estimates below are planning allowances, not measured performance of the integrated model.

## 1. Outcome and scope

Finish a coherent Ukrainian dissertation about retaining useful video context under resource constraints. Separate three levels of evidence: historical published components; controlled A4 allocation experiments; and the new integrated architecture. Explain how the components inform the final design without implying that every historical module appears unchanged in it.

Keep the four chapters, existing scientific content, equation numbering, legacy figures and tables, and negative results. Make local corrections and add only evidence needed to support the final claims. Preserve original historical assets and distinguish any corrected presentation from its source. Produce one future article combining expanded A4 with the integrated evidence.

The smallest defensible package is:

1. Complete the registered A4 campaign and its existing confirmation workflow.
2. Correct analysis and provenance; reuse existing checkpoints for a small robustness check and, if retaining a long-horizon claim, one rollout diagnostic.
3. Repair integrated tests and profiling before spending on full training.
4. Run bounded development checks, then the four already planned integrated confirmations: two seeds each on Moving MNIST and TaxiBJ.
5. Integrate results, qualify unsupported claims, and typeset and inspect the actual Ukrainian manuscript.

No default new dataset, architecture search, LongMMNIST training, third seed, broad quota sweep, or A4 continuation beyond 30k. An unproven claim can be narrowed instead of triggering more experiments. Passing the document gates is necessary but is not scientific or visual acceptance.

## 2. Execution order and dependencies

| Phase | Work | Exit condition | Estimated effort |
|---|---|---|---|
| P0 — protect current work | Completed-output migration; preserve A4 revision and manifest | Verified files, unchanged campaign, active run available | Migration completed; final active-directory cleanup later |
| P1 — correct narrative and contracts | R1, R2, R6–R11, R15–R18; freeze final analysis definitions | Claim/evidence ledger and analysis contract reviewed before examining new confirmatory results | 1–2 focused working days |
| P2 — repair integrated readiness | R3/R4/R9/R10/R11 implementation and meaningful CPU tests | Nondegenerate tests; fair profiling implementation; data/split checks | 1–2 days, can overlap ongoing A4 without CUDA |
| P3 — close A4 | Existing finalization plus E1 below; E2 only for a retained long-horizon claim | Registered outputs accounted for; duplicate designs identified; final analysis reproducible | 0.5–1 day analysis, plus existing campaign time |
| P4 — bounded integrated gates | E3/E4 after GPU release | Numerical/memory gates pass; throughput and full-run cost measured; final manifest locked | 0.5–1 day plus bounded GPU use |
| P5 — four confirmations | E5, only after readiness and the existing launch-authorization process | Four completed runs, frozen checkpoint selection, complete provenance | GPU time must be recalculated from E4 |
| P6 — manuscript release | Tables, figures, examples, claims, references, actual Ukrainian PDF | Four gates pass; current PDF inspected; only explicitly unresolved administrative items remain | 2–3 days after results exist |

Allow roughly 5–8 focused working days of implementation, analysis, and editorial effort, partly overlapping training. This is not a calendar completion promise. A4 remaining training and integrated throughput determine elapsed time.

Do not edit tracked files, campaign definitions, or source revision in the active `hierarchical-video-prediction-path-b` worktree. Develop analysis fixes in an isolated revision and record which analyzer produced each result. Preserve the historical analyzer outputs. Integrated code changes belong in the dedicated integration/OpenSTL-integrated worktrees; leave unrelated dirty worktrees alone.

## 3. Manuscript change list

All chapter names below refer to `docs/dissertation/condensed_translated_humanized_ukrainian/`.

| Review | Location / object | Concrete change | Acceptance evidence |
|---|---|---|---|
| R1 | Chapter 3, equations 3.49–3.53 and adjacent text; Chapter 4 old memory result | Retain prototype equations. Explain that the second recurrence differs from ordinary repeated prediction. Replace claims of an identical-forward causal comparison; label 2.58 as a historical engineering comparison of different dynamics. Audit abstract/conclusions for propagation. | Chapter 3 and Chapter 4 make the same statement; identical-forward claims refer only to the new internal core. |
| R2 | Chapter 4 schedule calibration; conclusions | Describe 30k as a common fixed optimization budget. Explain the terminal acceptance property and the possibility of further improvement. Retain calibration curves and registered schedule. | No assertion that final-endpoint acceptance proves convergence; no retrospective stopping change. |
| R6 | A4 methods, counts, result tables | Separate scheduled executions, distinct configurations, semantic designs, and seeds. Document the ALC/STS alias after comparing actual ordered plans. Preserve all 86 scheduled executions. | Counts generated from semantic fingerprints and campaign records; aliases are not counted as independent scientific evidence. |
| R7/R8 | Evaluation methods and statistical tables | State validation sampling with replacement: 64 draws per check, 59 distinct clips under the inspected seeds. Define foreground weighting, empty-mask handling, horizon, selection rule, interval unit, and seed aggregation. Keep selection scores separate from final per-clip metrics. | A reader can reproduce every primary number; no claim that selection used all 300 validation clips. |
| R9/R10 | Chapter 3 routing equations and algorithm | Specify which information enters every token. Correct the minimum-quota feasibility statement; align constrained selection, tie-breaking, and margin definition with code. | Worked feasible and infeasible examples match implementation; routing comparisons name all changed factors. |
| R11 | Resource model, memory formulas and tables | Separate retained logical token bytes, unique live tensor storage, saved activations, and total device peaks. Include coarse bypass, dtype, gradients, parameters, and optimizer state where applicable. | No multiplication of unrelated savings; analytical and measured values are labeled separately. |
| R12 | Whole manuscript | Build a fresh Ukrainian PDF from the current sources; preserve the English abstract only where intended. | PDF source revision and build command recorded; every page checked after final build. |
| R13 | Figures 3.20, 2.18, 4.12 and mixed-language figures | Redraw the new integrated diagram with readable query labels, coarse-to-router input, and last-frame residual bypass. Add Ukrainian explanations for preserved English figures. Explain truncated axes or supply a clearly labeled corrected companion plot when source data exist. Keep the old prototype limitation beside its figure. | Legible at printed size; diagram agrees with code; caption explains units, arrows, conditions and limitations. |
| R14 | Chapter 4 qualitative evidence | Add one compact panel with context/time age, ground truth, hierarchical and flat predictions, and same-scale errors. Freeze a selection rule for hierarchy-better, flat-better, and both-poor cases. | Cases trace to checkpoints and clip IDs; no suggestion of representative frequency from three examples. |
| R15 | Tables 2.6 and 2.10, provenance appendix | Attach profiling receipts where available: hardware, precision, batch, mode, repeats and WDDM behavior. Explain published MSE/PSNR aggregation or scaling only when documented. If unavailable, mark the convention unresolved and keep the published values attributed. | No invented reconstruction of missing raw evidence; no direct comparison across incompatible conventions. |
| R16 | Introduction, scope, results and conclusions | Separate context reach, four-frame A4 prediction, recursive rollout and integrated short-horizon benchmarks. Use E2 only if needed for an actual long-horizon claim. | Conclusions stay inside the tested datasets, budgets, horizons and seed count. |
| R17 | Related work / method transition | Add a small lineage table: inherited principle, historical component, final implementation, corresponding evidence. Clarify that residual/change/learned routing is not the published entropy router. | The claimed integration matches the actual architecture. |
| R18 | All chapters | Remove editorial explanations about what this edition “retains”; streamline repeated caveats while keeping them near the affected results. Standardize Ukrainian terminology, acronym definitions and table precision. | Four-chapter flow reads as a dissertation rather than a revision report. |

Formula audit: check tensor dimensions, index ranges, units, quota feasibility, recurrence initialization, and whether each equality describes the historical or current model. Start with equations 3.49–3.55 and adjacent memory expressions, then check all changed formulas against their definitions. Preserve numbering and make a change ledger; the prior review did not rederive all 119 equations.

Table policy: one primary A4 comparison table, one compact integrated quality table, and one resource table should carry the new conclusions. Put exhaustive run lists, per-seed/per-horizon detail and provenance in appendices. Report both seed values when there are only two; clip bootstrap intervals do not estimate variability across training seeds.

Figure policy: reuse existing results; avoid decorative additions. The required new assets are the corrected integrated diagram and one qualitative panel. Add a rollout curve only if E2 is performed. Keep full data scales consistent, disclose broken/truncated axes, and use readable Ukrainian captions and legends. Retain historical originals.

Final visual check: title/front matter, contents and page numbering; Ukrainian fonts; equation clipping and numbering; image resolution and small labels; caption placement; table overflow and repeated headers; cross-references, citations, URLs, widows/orphans, and appendix ordering. Resolve the 11 known administrative fields from authoritative user/institutional information; do not invent them.

## 4. Code and analysis repairs

### C1 — Nondegenerate correctness tests (R4; required before GPU gates)

Target: `OpenSTL-integrated/tests/test_models/test_hvp_integrated.py` and `openstl/models/hvp_integrated_model.py` if tests expose a defect.

- Keep persistence initialization as a separate test. For parity and dependency tests initialize a small nonzero prediction head deterministically.
- Compare eager versus reversible outputs, input gradients and named parameter gradients under the same scalar loss. Assert finite values and meaningful nonzero upstream gradients, not merely `.grad is not None`.
- Test packed-token permutation with coordinates, and include a negative control: corrupt meaningful memory/coordinates and require predictions to change. Avoid expecting gradients through discrete top-k membership itself.
- Use CPU FP32 first. Later exercise supported CUDA precisions with explicit dtype-specific tolerances and failure messages. The review's successful tiny probe is encouraging evidence, not a substitute for committed tests.

### C2 — Fair memory and timing profiler (R3/R11)

Target: `hierarchical-video-prediction-integration/scripts/v2/profile_integrated_openstl_memory.py`.

Implement identical-forward eager, activation-checkpointed and reversible-core modes from the same weights, inputs, batch, loss, precision and training state. Recreate/reset model and optimizer state for each case. Warmup must not silently turn a shared checkpoint into successively different models. Measure inference separately from forward/backward/optimizer training.

Report absolute and incremental allocated/reserved peaks; parameter, gradient and optimizer bytes; core-only versus whole-model scope; actual batch; actual tensor dtypes; synchronization/warmup/repeat protocol; median timing and spread. Include dense encoder and coarse-grid bypass in the whole-model accounting. Distinguish logical bytes from unique storage and allocator peaks. Use supported autocast consistently; use GradScaler for FP16 training or explicitly reject unsupported FP16 paths. Run output/input-gradient/parameter-gradient parity under the precision being profiled, with declared tolerances.

### C3 — Router contract (R9/R10)

Target: `OpenSTL-integrated/openstl/models/hvp_integrated_model.py` plus tests/config descriptions.

Choose one common token-feature policy for the minimal static-versus-mixed comparison. Prefer computing the same available score features in both and changing only membership; otherwise explicitly call the result a routing-package comparison and drop a pure-membership causal claim. Do not add a factorial study.

Fail early on infeasible requested minimum quotas, or expose and document `m_eff`; select one behavior consistently. Test quotas, boundary ties and determinism. Either implement a constraint-aware selection margin or rename the existing global Kth/K+1 statistic so it is not described as the actual constrained boundary. Keep default feasible configurations unchanged unless another defect requires a change.

### C4 — A4 analysis and provenance (R2/R6/R7/R8)

Targets: calibration, campaign-summary and confirmation-analysis scripts under `scripts/v2/` in a separate development revision.

Do not reinterpret final-endpoint acceptance as convergence. Add semantic-plan aliases while retaining the live controller's expected execution count. Freeze one versioned final analysis contract: primary horizon 4, explicit foreground metric and empty-mask policy, paired clip IDs, direction of differences, paired bootstrap resampling unit, and handling of the two training seeds. Predeclare the small family of primary comparisons and its multiplicity treatment. Claim noninferiority only if a justified margin was fixed before seeing relevant outcomes; otherwise present estimates and uncertainty without that verdict.

Add small numerical fixtures that distinguish pixel-weighted versus per-clip aggregation and detect unpaired IDs, horizon mixing and empty masks. Preserve historical metrics and generate new outputs under a distinct analyzer version. If confirmation results are already available when this is implemented, disclose the timing and label new analyses appropriately; do not retroactively call them preregistered.

## 5. Minimal experiment matrix after A4

GPU allowances refer to the local RTX 4060 Ti, exclusive use, and are provisional. E1–E4 require no broad training campaign. A4's already committed 86 executions and existing confirmation work are not additional experiments in this matrix.

| ID / priority | Question and design | Models / data / budget | Deliverable and decision | Additional GPU allowance |
|---|---|---|---|---|
| E1 — required analysis robustness | Does the frozen A4 shortlist behave similarly on the complete validation set? | At most 3 configurations × 2 seeds; existing checkpoints; all 300 validation clips, once; shortlist frozen from registered selection | Small robustness table. Do not reselect the winner or present validation as independent confirmation. Use existing registered confirmation for primary held-out claims. | 0.25–1 h |
| E2 — conditional on long-horizon claims | Does a useful short-window result persist under recursive prediction? | One frozen hierarchical/flat pair × 2 seeds; 100 newly fixed diagnostic clips; one rollout to 64 frames, summaries at 4/16/32/64; no retraining | Error/foreground error by horizon, trajectory failures and one panel. Freeze inputs, feedback protocol and seeds first. Label diagnostic; if unsuccessful, narrow the claim. | 0.5–2 h |
| E3 — required integrated gate | Is reversibility numerically correct and useful for the same computation? | Three execution modes, both dataset shapes; matched weights/batches; FP32 plus the intended training precision; inference and training; bounded repeats | Parity assertions and complete resource table; reject unsupported memory claims. Core savings alone do not establish whole-model savings. | 0.5–1.5 h |
| E4 — required bounded development | Is the selected integrated recipe trainable, and are minimal controls informative? | Full model on each dataset; on Moving MNIST only, one flat and one static-hierarchy control at identical updates/seed/feature policy; maximum four short trajectories, at most 1 GPU h each including validation | Loss/gradient health, throughput, parameter counts, selection behavior; no quality superiority claim from these short runs. Freeze architecture/schedule after the screen. | 1–4 h total |
| E5 — required for final integrated results | Does the locked integrated model work on both planned tasks? | Exactly INT_MM_S00/S01, INT_TB_S00/S01; Moving MNIST 10→10 and TaxiBJ 4→4; fixed splits, precision, schedule and checkpoint rule | Two-seed benchmark table, per-horizon detail, qualitative panel and checkpoint provenance | Measure from E4; see estimate method below |
| E6 — conditional, not automatic | Is a missing comparator essential to a proposed quality claim? | First reuse a compatible, independently selected baseline checkpoint. If none exists, narrow to feasibility; only if stronger comparison is indispensable propose a separate bounded baseline budget | No “better than baseline” conclusion from incompatible published numbers or short E4 controls | Zero in default scope; separately cost before adding |

E4 is a smoke/development screen, not a statistically supported component ablation. Checkpoint mode and reversible mode do not require separately trained quality models because E3 establishes their identical-forward contract. If the dissertation insists on full-training causal quality claims about hierarchy or adaptive membership, matched full-training controls would be necessary; the recommended minimal plan instead avoids those claims.

**Budget stop rule:** the upper allowances for E1–E4 sum to 8.5 GPU hours with E2, or 6.5 without it. Reserve no more than 9 additional GPU hours before the four confirmations. Stop a failing gate, repair the diagnosed defect, and reassess the remaining budget; do not silently expand into a sweep. Avoid concurrent GPU work while A4 is active.

### E5 estimate and schedule lock

Current configuration defaults are 200 epochs/batch 8 for Moving MNIST and 100 epochs/batch 16 for TaxiBJ. They are prelock settings, not a measured runtime commitment. Measure representative end-to-end training throughput and validation overhead in E4 using the actual split and loader; include startup, checkpointing and data conversion separately.

For each dataset d, estimate `hours_d = 2 × (epochs_d × steps_per_epoch_d × seconds_per_step_d + total_validation_seconds_d + checkpoint_seconds_d) / 3600`. Use actual loader length (including drop-last/distributed behavior), then add 20% operational contingency. Sum the two datasets. Also project output storage before locking.

Illustrative sensitivity only: if the two datasets together require 600,000 updates across the four runs, weighted step times of 0.10/0.20/0.40 s imply 16.7/33.3/66.7 GPU hours before validation and contingency. Neither the 600,000 count nor those speeds is a measurement. Replace this illustration with measured values at E4 exit.

Recommended economy target: at most 48 GPU hours for all four confirmations, including contingency. If measured projection exceeds that, decide on a scientifically defensible common schedule during development, or explicitly revise the estimate before locking. Do not truncate runs after seeing favorable test results, claim convergence from the budget alone, or automatically launch extra controls. The existing confirmation manifest remains `launch_authorized: false` until its established readiness/authorization requirements are met.

### Selection, failure and stopping rules

- Keep all current A4 runs and their registered checkpoint selection. Do not restart training to repair a description or metric-reporting convention.
- Freeze E1/E2 model identities and qualitative selection rules before looking at the new outputs. For a qualitative panel, rank a defined error difference for the “better” examples and a defined joint-error measure for “both poor”; identify these as deliberately selected diagnostic cases.
- E3 fails on numerical mismatch, missing gradient paths, unfair state/batch/precision, or missing required accounting. A small whole-model saving is a result, not an invitation to optimize until a large ratio appears.
- E4 fails on nonfinite training, broken data separation, ineffective gradients, or inability to complete within the resource envelope. Fix the cause; do not use the final test set to choose the repair.
- Two E5 seeds support limited replication, not population-level training-seed confidence. Report disagreement transparently. No default third seed or “best seed” headline.
- A4 training beyond 30k is unnecessary if the text says fixed-budget performance. No new natural-video benchmark is required for explicitly scoped synthetic/traffic findings.

## 6. Storage: completed action and remaining handling

Migration receipt at 09:56 UTC on 5 September 2026:

- Moved **40 completed A4 run directories**, **8,014,884,073 bytes (7.46 GiB)**, to `D:/hierarchical-video-prediction-storage/article4_primary_30k/runs`.
- Verified every copied file by relative path, length and SHA-256 before removing its original regular directory. Left a Windows directory junction at each original run path.
- Created D-backed destinations/junctions for **39 not-yet-started wave-one runs**, so their future output grows on D.
- Left the currently writing `A4_ALC_B1792_L2_COARSE_HEAVY_NEST_S00_T30K` directory on C. Its controller source remains pinned to `7a5979615f5354564a793083b673cf1ed9ec77be`.
- Free space at migration completion: **C: 26.319 GiB; D: 67.465 GiB**. C is above the existing 25 GiB restart threshold, but with limited headroom.
- The campaign file hash remained `F2BE4544227F1B3128C70121F44CD85C5A27BCC6426743303D2F41CDA0039E01`. The live worktree remained clean.

Audit material: `D:/hierarchical-video-prediction-storage/storage-migrations/2026-09-05-a4/summary.json`, `migration.jsonl`, per-run hash manifests and `migrate-completed-a4.ps1`.

The approximately 70 GB was available capacity on D, not the amount of completed A4 results found to move. Active checkpoints, campaign metadata and logs were not relocated by this operation. Junctions require D to remain mounted. This frees space but is not a backup: each migrated output has one data copy on D.

Remaining operational work: after the skipped run completes, migrate it using the same inactive/hash-verified procedure. Handle the six wave-two output directories once their exact paths are known, without modifying the live registered manifest. Future integrated output already targets `D:/hierarchical-video-prediction-storage/integrated_openstl_v2`; honor its 15 GiB post-projection reserve. Do not move datasets or delete historical checkpoints merely to increase the free-space number. At campaign closure, consider moving remaining bulk outputs only after checking every consumer that resolves paths.

## 7. Release checklist and scope guard

Before accepting the dissertation:

- R1–R18 each have a correction, evidence-backed resolution, or explicit limitation in a finding ledger.
- The final result tables link to immutable configs, checkpoints, splits, analyzer revisions, and profile receipts; no provisional live values are presented as final.
- Integrated tests are nondegenerate; profiler comparisons hold computation constant; router and resource equations match code.
- A4 aliases, validation coverage, foreground aggregation, horizon and uncertainty units are explicit.
- Actual integrated quality evidence is present, or all unfinished integrated results remain clearly prospective. Negative results are retained.
- The four existing manuscript checks pass: `check_manuscript_finalization.py`, `check_text_consistency.py --gate`, `check_rft_consistency.py`, and `audit_ukrainian_dissertation.py`; `git diff --check` passes for the changes.
- Current Ukrainian PDF, source revision, build recipe and page-inspection record exist. Legacy image/table/equation inventories are reconciled with intentional additions and local corrections.
- Administrative placeholders are resolved before submission; until then describe the manuscript as scientifically/technically checked with administrative completion pending.

Do not add experiments merely because the previous result is negative. The stop condition is a consistent, reproducible dissertation whose claims fit its evidence, not universal superiority of the integrated model.
