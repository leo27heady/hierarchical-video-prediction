# Dissertation refactoring and final review

Date: 5 September 2026. Implements the [approved correction plan](../plans/DISSERTATION_REMEDIATION_AND_MINIMAL_POST_A4_PLAN_2026-09-05.md), based on the [independent review](INDEPENDENT_DISSERTATION_AND_LIVE_EXPERIMENT_REVIEW_2026-09-05.md).

**Assessment:** the revised manuscript has a coherent four-chapter argument, corrected method contracts, useful completed-run evidence, and explicit boundaries around unfinished work. It is a review manuscript, not a completed experimental submission. Unfinished measurements have placeholders; favorable expectations are hypotheses, never substituted numbers.

## What changed

The central narrative now distinguishes historical published components, the controlled context-allocation experiments, the old integrated prototype, and the final OpenSTL architecture. Chapter 3's old reversible recurrence no longer claims an identical-forward comparison. The 30k calibration is described as a common fixed budget, with its automatic terminal acceptance disclosed. Chapter 4 separates the old confirmed frontier from the new validation snapshot, which does not show a monotonic budget frontier.

The method description now specifies quota feasibility, deterministic tie handling, identical score-feature treatment across routing policies, the actual unconstrained score-gap diagnostic, the coarse decoder bypass, and the distinction between token bytes, live storage and training peaks. Table 3.5 traces the final architecture's principles without presenting V-BLT entropy, RoPE-ND or the TempVerseFormer temporal module as literal final-model components.

Added tables 4.8–4.9 for pending integrated quality and resource measurements, table 4.10 for completed FRT results, and figure 4.15 for paired saved predictions. Figure 3.20 was redrawn to include coarse input to the router, future-query initialization and the last-frame residual bypass. Existing equation numbering, historical figures/tables and published-figure files were preserved. Reviewer registries and figure provenance were synchronized with the additions.

## Completed results used

The manuscript uses a **frozen 11:52 UTC snapshot of 40 completed runs**, including **all 36 FRT runs**. Training-log, validation-log and final-checkpoint hashes were checked, along with the pinned clean launch revision and manifest/status configuration identity. The snapshot generator reuses this frozen run set on subsequent execution; it does not silently pull later runs into the manuscript.

All 18 FRT policy/budget combinations are shown with both seeds at 30k. These are fixed-step validation scores, not new held-out confirmation results. For the geometric-versus-flat comparison, geometric allocation has lower validation foreground MSE for both seeds at 768, 1792 and 2432 tokens, while flat allocation has lower error at 1152. Directions differ across seeds at 320 and 3072. Different policies, training budgets and populations prevent direct substitution for the historical frontier.

The qualitative panel uses saved final-step arrays from a frozen B1152 pair, seed 0, horizon 4. Clip IDs, target arrays and last-context arrays are identical across the pair. Three distinct diagnostic examples were selected by a declared error rule; they are not a representative frequency estimate. No new inference was needed.

Receipts:

- [Completed-run snapshot and panel provenance](a4_completed_snapshot_2026-09-05/snapshot.json).
- [Physical-plan alias audit](a4_completed_snapshot_2026-09-05/semantic_plan_aliases.json): 80 wave-one executions map to 78 distinct ordered physical plans including seed. The ALC ReachOptimized/STS K64 Recent alias is confirmed for both seeds.

Later ALC completions remain outside this frozen manuscript snapshot. At the late operational check, the controller was on wave-one entry 44/80, `A4_ALC_B1792_L2_REACH_OPT_NEST_S01_T30K`, and continued training. The live source remained clean and pinned to `7a5979615f5354564a793083b673cf1ed9ec77be`.

## Code changes and verification

In `OpenSTL-integrated`, static, density and hybrid membership now receive the same hybrid-score embedding. Infeasible minimum quotas fail instead of being silently clamped. Stable sorting resolves tied scores by physical index. The diagnostic is named `route_global_unconstrained_score_gap`, including its logging consumers. A checkpointed execution path was added for the same core blocks.

Correctness tests separate persistence initialization from meaningful dependency tests. With a nonzero prediction head they compare outputs, input gradients and named parameter gradients, require upstream gradient magnitude, check full-quota feature equivalence, test checkpoint parity, and require corrupted memory to change predictions.

The integrated profiler is version 2, mirrored into the dissertation branch. It compares eager, checkpointed and reversible execution; separates inference/training and whole-model/core-operation scope; uses the actual batch and precision; records absolute/incremental allocated/reserved peaks, parameter/gradient/optimizer bytes, actual packed-state dtypes and storage, forward saved-tensor accounting, and timing repeats. AdamW uses zero learning rate and fresh optimizer state for each measured repeat, so checkpoint weights remain fixed and optimizer-state allocation is included. FP16 uses GradScaler and rejects overflow-skipped steps. Parity includes named parameter gradients and rejects degenerate persistence probes. Synthetic profiling is explicitly not a quality experiment.

The separate A4 analyzer preserves historical calibration decisions but labels terminal acceptance correctly. New paired analysis aligns clip IDs, requires a horizon-4 record for every clip, distinguishes foreground aggregations, bootstraps clips within one training seed, exposes family size and produces descriptive adjusted intervals. It does not reselect models or issue a noninferiority verdict. Original confirmation arrays do not contain foreground-pixel counts, so the new analyzer explicitly cannot reconstruct the number of empty masks from those arrays.

Verified:

- **24 integrated model/data CPU tests passed.** Two expected warnings concern logging without a Lightning trainer and pinned memory with CUDA hidden.
- **2 profiler contract CPU tests passed**, including rejection of a zero-head probe, acceptance of a nonzero-head probe, actual-batch parity and unique-storage accounting.
- **3 paired-analysis tests passed**, including aggregation differences, empty masks, ID alignment and missing/duplicate horizon rejection.
- All four manuscript gates passed. The Ukrainian inventory contains **119 equations, 83 captions, 86 image insertions and 174 cited references**. The gate explicitly reports **11 administrative inputs and 14 experimental table cells containing “очікується”**; those are not measured results.
- Modified/new Python files passed compilation checks; whitespace checks passed across the edited worktrees.

No CUDA memory/precision claim was accepted from CPU tests. No new training, held-out evaluation, campaign finalization or integrated launch was performed. The user allowed unfinished work to remain described with placeholders, and A4 still occupies the GPU.

## Final content and flow review

The topic is now consistently framed as resource-constrained video-context retention and conditional consistency, with distinct limits for reach, fidelity, identity and activation memory. The chapter sequence remains problem/criteria → foundations → method/contracts → evidence/decision rules. Historical negative results remain visible. The new FRT snapshot follows the existing system decision rule and explicitly tests its limits; it is not spliced into the old confirmatory tables. The integrated result slots precede the system synthesis and have clear evidence labels.

The formula pass checked the changed recurrence, quota constraints, standardization floor, coordinate definitions, reversible block equations and decoder broadcasting against implementation. Existing numbering and all mathematical cross-references were checked mechanically. This was not a fresh derivation of every historical equation or a re-execution of every cited experiment.

The current Ukrainian sources were rendered using the repository's document machinery and Microsoft Word, with native equations, repeated table headers, intact table rows, corrected cell indentation, Times New Roman body text, plain page numbers, a generated contents field and heading bookmarks. The packaged DOCX renderer was attempted but could not run because LibreOffice is absent; Word PDF export plus PyMuPDF provided the rendering and inspection path.

All pages were rendered and checked for off-page text blocks, missing replacement glyphs and near-empty pages. Contact-sheet review covered the full document, with enlarged inspection of the revised diagram, formulas, result/placeholder tables and qualitative panel. Initial defects in table-cell indentation, header placement and argmax conversion were fixed and re-rendered. The PDF is a typeset review edition: administrative front matter and institutional submission certification remain unresolved.

The delivered [review PDF](../../output/pdf/DISSERTATION_UKRAINIAN_REVIEW_2026-09-05_LAYOUT.pdf) has **283 pages, 174 heading bookmarks and 192 links**. Its [render audit and PDF hash](a4_completed_snapshot_2026-09-05/render_audit.json) and [manuscript source hashes](a4_completed_snapshot_2026-09-05/manuscript_source_hashes.json) preserve the exact reviewed edition. Reproduction uses `scripts/build_ukrainian_review.py`, then `scripts/export_ukrainian_review.ps1`, then `scripts/inspect_ukrainian_pdf.py`; the export requires installed Microsoft Word, and the Python rendering dependencies are documented by the task-local runtime setup. Superseded PDF drafts are retained under `tmp/ukrainian_pdf/superseded/` rather than beside the delivered edition.

## Finding-by-finding disposition

| Finding | Disposition |
|---|---|
| R1 | Corrected old recurrence/causal-memory statements; historical 2.58 ratio retained with its limitation. |
| R2 | Corrected fixed-budget interpretation in method, calibration, conclusions and analyzer output. No continuation required for that narrower claim. |
| R3 | Profiler repaired and CPU contract-tested; actual CUDA execution remains pending. |
| R4 | Nondegenerate tests implemented and passed. |
| R5 | Original skipped completed directory also migrated after it stopped writing; active training untouched. |
| R6 | Semantic alias audit implemented; manuscript counts corrected; all registered executions preserved. |
| R7 | Actual validation coverage stated; full-300 shortlist evaluation remains pending. |
| R8 | Versioned paired analyzer and fixtures added; aggregation distinction explained; actual final confirmation analysis remains pending. |
| R9 | Common score-feature treatment implemented and tested; no full-training causal routing-quality claim added. |
| R10 | Feasibility enforced; deterministic ties; diagnostic renamed; equations/prose aligned. |
| R11 | Coarse bypass and actual storage/dtype accounting included; saved-tensor diagnostic added; empirical peaks pending. |
| R12 | Current Ukrainian PDF built and visually checked; final institutional/admin completion remains open. |
| R13 | New architecture diagram corrected; published axes/language explained without altering frozen artwork. |
| R14 | Paired examples added from verified saved arrays; case-selection limitations explicit. |
| R15 | Historical profiling and MSE/PSNR convention gaps disclosed. Missing primary receipts remain unresolved; no values invented. |
| R16 | Short-window versus long-rollout claims separated; conditional rollout protocol described; no scope expansion. |
| R17 | Lineage table added; final router/position implementation distinguished from published modules. |
| R18 | Targeted editorial residue removed and cross-chapter claims aligned; historical scientific material retained. |

## Remaining work and risks

1. Finish A4 and its registered confirmation workflow, then replace the frozen development snapshot only through a new dated, verified result update. Complete full-validation robustness on a frozen shortlist; do not select winners from final test results.
2. Run integrated CUDA parity/resource gates, bounded training screens and measured runtime projection after GPU release. Only then lock and run the four integrated confirmations. Keep the minimal budget and stopping rules in the approved plan.
3. Replace the explicit integrated placeholders with actual measurements, including normalization/aggregation, checkpoint/split provenance and per-seed results. Narrow claims if outcomes are unfavorable. Short development controls do not establish full-training component superiority.
4. Recover historical profiling receipts and V-BLT metric conventions if they can be found; otherwise retain the current reproducibility limitation.
5. Complete the 11 author/institution fields and validate the final institutional layout. Long historical tables and original English figure labels remain acceptable for this review edition but need a final print-size institutional pass; a contact sheet is not a line-by-line typographic certification.

The semantic change to router features makes a new implementation revision necessary for future integrated runs. Old checkpoints that used the former policy must retain their original implementation provenance; loading their weights into the revised route is not evidence of identical predictions.

Storage follow-up moved one additional completed directory (200,380,596 bytes), making **41 completed directories migrated across the two operations**. All 80 wave-one run paths now use D-backed directories; the current run was not relocated while active. At the follow-up receipt C had 26.195 GiB free and D had 66.727 GiB. Full receipts remain under `D:/hierarchical-video-prediction-storage/storage-migrations/2026-09-05-a4/`; the JSONL preserves both migration operations. Wave-two paths and final bulk-output retention still require their planned handling.

Changes are left reviewable in the dissertation, integration and OpenSTL-integrated worktrees. They have not been committed or pushed. The active A4 worktree and launch authorization remain unchanged.
