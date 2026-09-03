# Dissertation refactor: implementation, verification and remaining decisions

Date: 3 September 2026.

## 1. Outcome and exact completion boundary

The English refactoring has been applied to all nine manuscript files, on top of the existing Ukrainian dissertation. The four-chapter order and the scientific foundation from Articles 1–3 remain. Chapters 3 and 4 now describe the exact new hierarchy programme and the OpenSTL-native integrated architecture, with explicit result placeholders wherever the required evidence is not yet finalized.

This deliverable is a working, reproducible **evidence-gated dissertation draft**, not a completed submission manuscript. The editable implementation, migration, existing-evidence presentation, result-input infrastructure, and draft-level typesetting/QA are implemented. Experimental result lock, final result-dependent conclusions, language harmonization, final numbering/pagination, institutional Word/PDF formatting and administrative/permission approvals remain future gates. It would be incorrect to mark every acceptance condition of the original plan as finished while those inputs are absent.

Working location:

`C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation`

Branch: `codex/dissertation-refactor-v2`.

Manuscript: `docs/dissertation/condensed_translated_humanized_ukrainian/`.

Read [the complete typeset draft](../dissertation/refactor_v2/preview/dissertation_reading_draft.html), [editing/reproduction instructions](../dissertation/refactor_v2/README.md), and [the independent blind review](BLIND_REVIEW_DISSERTATION_IMPLEMENTATION_2026-09-03.md).

The source manuscript remains recoverable unchanged from commit `cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8`. Before the new worktree was created, its nine source files were byte-identical to the Path B manuscript. Editing in a separate worktree allowed useful work while training continued. No training code, live campaign manifest, checkpoint, dataset, CUDA process or W&B run was changed by this editorial implementation.

The original plan in the integration worktree also receives a documentation-only continuation pointer to this deliverable. Its scientific code and prelock launch authorization are unchanged. The point-by-point checklist is [PLAN_ACCEPTANCE_MATRIX.md](../dissertation/refactor_v2/PLAN_ACCEPTANCE_MATRIX.md).

## 2. What changed, component by component

### Front matter and introduction

Both abstract scientific blocks now share one English editing source. Result-dependent summary keys are reused in the abstracts, novelty, Chapter 4 and general conclusions, so later insertion cannot silently create different claims in different parts of the dissertation. Existing institutional identities and author/contribution records remain preserved.

The introduction distinguishes retained state, internal activations, parameters/optimizer state and workspace. It explains why spatial compression can finance temporal reach without guaranteeing quality, preserves six dissertation-level tasks, and nests the detailed hierarchy/integration questions under them. The new work is one manuscript in preparation, not two promised publications. Old prototype results are not presented as completed results of the new integration.

The final six-item conclusion is deliberately a task-aligned draft skeleton. The source-era conclusion is archived. The intended longer final synthesis must be written after the Chapter 4 claim ledger is locked, not filled now with projected successes.

### Chapter 1: preserved system argument, clearer intervention boundaries

The chapter retains its system/partial-observability, consistency, resource-bottleneck, literature and requirements structure. The retained consistency synthesis covers appearance/style, identity, geometry/shape, concept, trajectory and cross-level consistency; Article 3 is not reduced to a gallery.

New English passages separate spatial hierarchy, temporal resolution/age allocation, nested/exclusive ownership, reversible activation storage and dynamic space–time routing. The resource map has four accounting terms. Analytic token savings are explicitly not total-VRAM savings, and independently measured component ratios are not multiplied.

The static-control/diffusion tutorial gallery is moved to the supporting appendix. Important scientific ideas, attribution and source assets remain available. The hypotheses are falsifiable and distinguish long retained history from a long self-fed forecast. A stale Ukrainian statement that the new integration had already been tested was replaced as a complete English paragraph during blind-review remediation.

### Chapter 2: published methods remain substantive

TempVerseFormer retains its reversible temporal model, forward/inverse boundary, executed objective, controlled quality results, OpenSTL evidence, memory and runtime results. Reversible forward/inverse relations and replay are grouped; standard algebra and stochastic extensions move to the dependency archive or compact inline form. A reduced formula count does not remove the variables required by the surrounding prose.

V-BLT retains its local–global–local architecture, causal byte prediction, RoPE-ND, entropy diagnosis and published 2-D/3-D results. The boundary to the new router is explicit: the published system has fixed patches and entropy diagnostics; it did not demonstrate the integrated residual/change/learned router or its memory gain.

The five-panel V-BLT chart mixing model cost, quality metrics and dataset color meanings is no longer in the main narrative. Original source artwork remains in Appendix Е. Published multi-dataset tables are split by population and row-wise winner boldface is removed; their numerical values are not changed. This produces more small tables than the indicative table-count target, but each is simpler and single-population. Original combined tables and machine-readable transcriptions remain auditable.

The chapter synthesis explains what each mechanism changes and what it leaves unchanged, including why a smaller training tape cannot recover an observation that has left memory.

### Chapter 3: exact current method and experiment contract

The 3.1–3.8 progression remains. The frozen two-level codec, packed context, physical token identities, trainable/frozen boundaries, shared WindowDiT and jointly denoised four-frame target are described directly. Standard Transformer/DDIM derivations no longer dominate the chapter.

The chapter specifies the actual 30k programme: FRT, ALC, CEX, LVL, OVR and STS; exact budgets and quotas; two seeds; reused cells; dependent selected configurations; development selection; sealed confirmation and bounded attention diagnostics. CEX varies budget/compute/reach together. LVL changes the registered stream construction, rounding and parameter count. OVR changes ownership/representation and possibly ages. None is described as a pure one-variable causal intervention.

Three important implementation details were corrected through the blind audit:

1. The executed diffusion loss sums the coarse and fine **per-stream MSE means**, rather than an unweighted norm of concatenated unequal-size streams. The optional top stream is context-only.
2. Development selection uses horizon-four foreground MSE with target intensity above 0.5, foreground-pixel weighting within each fixed batch, then batch-size weighting across four batches of 16. It is not a mean of individually normalized clip errors.
3. Historical E2's pair-specific support is distinct from v2 confirmation's **common 96-frame/28-frame sprite-visibility frame flag**. The finalizer reports separately normalized clip/frame errors averaged over all four horizons, re-entry frames or non-re-entry frames. Those fields are not the development horizon-four metric.

MEC is described accurately as one fixed clip, one direct denoiser forward at diffusion time 0.5, with head-averaged target-query attention by source span/layer. It is not a sampled trajectory, a causal old-context ablation or a second independent quality experiment.

The integrated architecture replaces the old fixed-region/second-order prototype in the main method. It describes genuine fine-token gather, dense coarse coverage, physical x/y/time/scale/level metadata, additive reversible coupling and future-query decoding. Static/density/hybrid inference policies are distinguished from mixed route-aware training. Packed-slot position is not physical identity. Coordinate normalization is tied to the grid/protocol, not to selected membership; changing horizons is a distribution/protocol shift, not guaranteed extrapolation.

Encoder, router, dense candidates, retained coarse grid, decoder and optimizer remain outside the reversible core's saving. CPU inverse/parity/quota/permutation/shape/smoke evidence is bounded to the recorded tests. Dynamic full-model parameter parity and deployed CUDA mixed-precision checks remain gates. Top-k non-differentiability, tie behavior, selected-score feature changes, margin limitations and adjacent-frame versus cross-call route churn are stated openly.

### Chapter 4: new evidence takes the headline, historical boundaries remain

The chapter retains its experiment-map, mechanism-control, budget/reach, allocation, interaction, robustness, negative-boundary, integration, selection-rule, practical and validity roles.

The new hierarchy families have primary positions and traceable pending slots. The full frontier is labelled registered development evidence; only actual selected roles may receive sealed-confirmation wording. No incomplete progress snapshot is inserted as a final number.

The 30k calibration is already plotted from archived, hash-verified records. The figure shows all saved 2k validations for two seed-0 configurations, not two training-seed replicates. Its logarithmic error axis makes the late ordering change visible without discarding the early trajectory. The convergence statement is restricted to the observed training range.

Legacy E2 remains substantive self-fed evidence. E4 is simplified to individual seed effects by head. E5a retains the non-replicated late reversal; E6 retains the coverage/trajectory-versus-identity boundary. EBT, raw-memory E8a, detailed sampler/codec robustness and superseded integration are bounded or archived rather than competing with the new headline results. Long Moving-MNIST remains historical evidence only, not a new benchmark run.

The E4/E5 plots now state signed foreground-MSE units and contrast direction. Captions identify the actual population, warmup, rollout band and uncertainty unit. E5's unequal context budgets are explicit. Source tables/paragraphs, not newly read protected arrays, supply the plotted historical values.

Section 4.8 has separate cells for correctness, component memory, checkpointing, route switches, training-regime effects, quota/stability, Moving MNIST and TaxiBJ. Missing components contract the claim. Equal-quota routing is not another memory multiplier. The default 3.2× count ratio is labelled token arithmetic, not measured CUDA memory. Published unmatched OpenSTL numbers cannot establish a matched Pareto advantage.

## 3. Preservation and traceability

The generated ledgers preserve:

- all **114 original numbered formula objects**, with source context, dependencies and destinations;
- all **35 current numbered equations**, including 19 new/grouped definitions with source-file/hash or explicit conceptual-status bindings;
- **443** Chapter 1/2 source paragraph/object records: 236 remained verbatim in the main text and 200 verbatim in the expanded appendix at the first complete pass; later complete-paragraph corrections remain baseline-recoverable and line-traceable;
- **121** inventoried source figure/table objects, original referenced-asset file hashes and the baseline figure Git-tree inventory;
- all **80 original nonempty appendix records**, verified directly against the assembled file after line-ending normalization;
- the existing Ukrainian appendix sequence А, Б, В, Г, Д, Е, Ж, И;
- publication/source-asset attribution, permissions and administrative text, without inventing approvals.

The smaller current equation count is a draft outcome, not a quota-based deletion. Source derivations remain available, needed compact relations remain inline, and final result/decision equations can be added only when useful. Current tags and old source IDs are deliberately stable before final numbering.

`PUBLICATION_TRANSFER_MAP.md` separates genuine article/source anchors from mechanical paragraph preservation. The paragraph ledger does not infer that every sentence in a chapter belongs to one article. The independent review found no material loss of the core published contributions, but original publication permissions and full bibliographic certification remain their own responsibilities.

## 4. Result infrastructure and repeatable build

The implementation includes a single editorial build command, deterministic assembly, original/current formula ledgers, a 43-configuration/86-run registry, placeholder contracts, acceptance adapters, table/figure generators, shared claim insertion and an offline typeset HTML build.

Logical run counts include reuse: FRT 36; ALC 14; CEX 12; LVL 6; OVR 6; STS 24. They are not summed as new training work. The unique programme remains 86 runs. The frozen editorial input has 80 concrete specifications and six dependent rows; no final dependent ID/hash is invented before resolution.

The A4 adapter checks the exact finalized schema, 86-run membership, final 30k endpoint, canonical nested-spec hashes, campaign/selection/dependency bindings, seed membership and logical-family completeness. It distinguishes development, actual confirmation fields and MEC probe fields. It reads no tensor/checkpoint or protected pixel arrays. Explicit human-reviewed acceptance remains required.

The plotting/table interface requires finalized accepted bundles, metric/population/units, source/config/checkpoint/data identities and both seeds. It supports single-purpose response curves, dots, STS mean/seed surfaces, Pareto scatter and real Markdown tables. Pairing uses a stable scientific cell ID so different seed memory measurements remain valid. Styles are stable across input ordering. Table labels escape reserved Markdown characters.

Shared claims and generated artifacts are inserted only when their hashes match. The acceptance ledger reapplies them consistently after rebuild; incomplete or changed inputs fail rather than silently replacing results. Interpretation is reviewed as positive, neutral or negative, not selected by an optimistic layout assumption.

The integrated prelock snapshot includes four run IDs, both config file hashes, model/method/test/controller hashes, Path B source commit `7a5979615f5354564a793083b673cf1ed9ec77be` and OpenSTL-integrated commit `ca335ba206ef6f51515a4f8eefcad1eed8c36c10`. It explicitly retains `launch_authorized=false`; this record does not authorize training or masquerade as an evidence lock.

## 5. Verification performed

The one-command build and CPU-only editorial tests run successfully. The regression tests cover section boundaries, original ledgers, pending input rejection, two-seed requirements, logical reuse counts, incomplete families, wrong final endpoints, wrong metrics, stale acceptance/config hashes, tampered nested specifications, table construction and stable method styling. Tests use temporary fixtures, never fabricated dissertation results.

Final verification passed **eight regression tests**, including shared-claim insertion and rejection after its underlying evidence changes. A second complete assembly produced identical hashes for **53 manuscript/registry/figure files**. A direct post-assembly comparison reconfirmed all **80/80** original appendix blocks. `git diff --check` passed. Read-only checks found the Path B and OpenSTL-integrated worktrees clean at their recorded scientific commits.

Automated draft QA passes. The release check intentionally fails with 25 unique pending result/summary keys plus author/institution, language, final numbering/pagination and institutional rendering/permission gates. The numeric citation round trip has no unresolved or uncited numbered bibliography entries in the retained main-plus-appendix corpus; this is a reference-resolution check, not independent certification of every publication.

The complete nine-section HTML renders 1,506 mathematical expressions using vendored KaTeX 0.16.22 MathML, with zero parse errors. Headless browser checks report zero broken images, zero unresolved local anchors, no document-wide overflow and no horizontally overflowing mathematical expressions. One wide archival table remains horizontally scrollable in the reading draft; institutional pagination must address it later.

All 12 new SVG assets were rasterized and visually inspected. Six representative typeset views were also inspected: Chapter 1 memory framing, Chapter 2 split results, Chapter 3 executed loss and physical position, Chapter 4 pending integration cells and appendix correctness gates. These checks verify a usable draft; they do not certify the pre-refactor Word/PDF files or final submission pagination.

No claim of newly verified GPU memory, trained quality, deployment stability or experiment completion follows from these editorial tests.

## 6. Blind-review findings and remediation

The separate blind agent read the full plan, draft, source interfaces and registries independently. Its original report remains intact. It began closure checks, then hit a usage limit before delivering a final signed-off report. The following is the implementing agent's remediation record, not an edited independent verdict.

| Finding | Implemented correction | Verification boundary |
|---|---|---|
| R1: wrong confirmation population/metric and broad MEC language | Separate historical E2 from common 96/28 v2 frame mask; exact four-horizon reductions and fixed direct attention probe | Reviewer explicitly confirmed the revised geometry during closure; assembled text/registry checked locally |
| R2: loss and development metric weighting ambiguous | Sum of per-stream means; threshold and batch reduction stated exactly | Reviewer confirmed revised Chapter 3 consistency; typeset equation checked locally |
| R3: missing reused cells and result-generation paths | Logical 14/6/6 families; finalized A4 adapter; actual Markdown tables; immutable shared-key insertion; endpoint-aware branches | Reviewer confirmed reuse; local regression tests verify missing/stale/wrong inputs |
| R3 follow-up: Pareto seeds paired by memory x | Stable configuration cell identity, actual per-seed memory retained | Regression test passes with unequal x values |
| R3 follow-up: unstable style, table pipe and nested-spec trust | Deterministic method palette, escaped table labels, recomputed canonical nested-spec hash/ID | Local tests; final independent rerun unavailable after usage limit |
| R4: known integrated provenance omitted | Dated prelock source/config/test/controller snapshot and exact four run IDs in Appendix Ж/registry | Files/hashes recorded read-only; not launch authorization |
| R5: orphaned prose and generic formula inventory | Necessary compact equations retained inline; 114 source-specific roles; separate 35-object current implementation/dependency ledger | Complete source/destination and math rendering checks; no claim that all historical proposals are implemented |
| R6: mixed-purpose inherited visuals and vague axes | Split published tables; remove irrelevant winner bolding; archive composite; precise E4/E5 axes/captions; calibration log scale; duplicate caption/name correction | New figures and representative typeset result views inspected locally |
| R7: preservation confused with article attribution | Separate paragraph-preservation fields and explicit publication transfer map; no keyword-based article assignment | Attribution scope is explicitly limited to actual source citations/anchors |
| R8: no current-tree typeset build | Full nine-file offline MathML HTML, browser checks and screenshot QA | Draft rendering complete; final institutional Word/PDF deliberately pending |
| Closure: Chapter 1 still implied composition already tested | Replace the complete paragraph after equation 1.7 with English pending/historical boundary | Reassembled and locally checked; no inherited performance result remains there |

A final independent post-result-lock scientific/typeset sign-off is still recommended. This report does not rename the interrupted closure check into a pass.

## 7. What still depends on experiments

### Existing hierarchy campaign — no restart or redesign required

Keep the current 86-run training programme, 30k endpoint and two seeds. Nothing in this editorial review requires relaunching it. After it completes, accept the immutable aggregate and resolved dependency/selection ledgers, then insert the development response curves and exact selected confirmation results. Preserve negative or crossing outcomes. The full development frontier does not become a confirmed frontier merely because selected roles pass.

### Optional v2 long autoregressive evaluation

The manuscript is already honest without it because legacy E2/E5/E6 carry the actual self-fed evidence. A selected-checkpoint v2 rollout would strengthen continuity, but it requires a separately authorized evaluator, sealed long TRB manifest, fixed warmup/horizons/NFE/populations and timing preflight. It needs inference GPU time, **not retraining of the 86 cells**. It must not reselect models from rollout outcomes.

### Integrated architecture and external validation

Before claiming performance or memory benefit, complete the declared compatibility/deployed-dtype gates, then matched CUDA measurements and trained-quality checks. The desired factors do not automatically create independent multiplicative wins:

- hierarchy must show its measured state/peak-memory and quality trade-off against a properly trained flat comparator;
- reversibility needs same-forward/checkpoint parity and matched training-memory/throughput measurements;
- checkpointing is the practical comparator, or the claim must be restricted to ordinary autograd;
- dynamic routing at fixed quota tests allocation, while memory gain requires an actually smaller quota that preserves the declared quality tolerance;
- route-aware versus static training is a separate trained ablation, not an inference toggle;
- four full runs on Moving MNIST/TaxiBJ supply two-seed full-model results, not all component/baseline cells;
- matched local baseline checkpoints or additional baseline training are required for a matched Pareto claim.

Runtime estimates must follow the GPU timing preflight and exact baseline/ablation choices. No new estimate or guaranteed 2–4× total-memory advantage is inferred from the editorial build. If a component or dataset fails, the prepared negative/neutral branches narrow the conclusion rather than hiding that outcome.

## 8. Decisions or information needed from the author later

| Input / decision | When needed | Why it cannot be inferred now |
|---|---|---|
| Author acceptance of this preserved four-chapter draft | Before final result insertion/editing lock | Confirms editorial emphasis without reopening the whole structure |
| Whether to authorize selected-checkpoint v2 long rollout | Before generating/opening its sealed evaluation set | Extra inference scope; current 86-run programme remains unchanged either way |
| Integrated primary local comparator(s), quality tolerances and allowed extra baseline/ablation budget | Before final integrated selection/test access | Competitive/Pareto claims need a predeclared matched contract; four full runs alone are insufficient |
| Approval for additional component training if required | After correctness and timing/quality preflight | Some effects need independently trained models; toggles cannot substitute |
| Final accepted aggregate/selection/confirmation and integrated evidence bundles | After experiments finish | Required to replace 25 pending keys with measured content and shared claims |
| Final manuscript language | Before submission editing | English changes are intentional now; institutional Ukrainian/English requirements still govern final harmonization |
| Publication status, final article title/authors and approved personal-contribution wording | Before abstracts/publication list lock | The combined work currently remains in preparation; institutional/coauthor facts cannot be invented |
| Missing institutional records, dates, signatures and implementation-act information | Before submission | Original wording is preserved, not guessed or newly approved |
| Figure reuse permissions/licences and institutional formatting rules | Before release rendering | Retention/provenance is not permission approval; final page-layout rules must be supplied |

No immediate answer is needed to retain or review this draft. These are explicit gates for later evidence acceptance and submission, not reasons to stop the current safe editorial work.

## 9. Continuation order

1. Review this draft and its independent report; keep experiment worktrees unchanged.
2. Let the existing hierarchy campaign finalize and accept its exact artifacts.
3. Resolve the optional rollout and integrated comparator/tolerance/budget decisions before their protected evaluation.
4. Insert accepted figures/tables and choose measured positive/neutral/negative interpretations through shared keys.
5. Write the full task-aligned Chapter 4/general conclusions and synchronized abstracts last.
6. Freeze final equation/figure/table numbering; regenerate cross-reference and provenance counts.
7. Harmonize the final language and institutional records, build current-source Word/PDF, inspect pagination/figures/tables, and obtain a final independent scientific/editorial check.

After context compaction, read the updated detailed plan, this report and `refactor_v2/README.md`. Never overwrite the source archive, relaunch the live campaign for editorial reasons, or fill missing results with probable good outcomes.
