# Independent review of the dissertation refactor implementation

Date: 3 September 2026. Review began against the uncommitted implementation on `codex/dissertation-refactor-v2`, based on `cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8`.

## Verdict

**Substantial, scientifically cautious editorial implementation; not yet evidence of full completion of the plan.** The revised narrative preserves the four-chapter argument, keeps the published methods and historical long-rollout boundaries, and does not fabricate unfinished experimental outcomes. Its most important initial defects are exact-method mismatches and incomplete evidence-generation/provenance contracts, not a wholesale loss of Ukrainian or published material.

Unfinished experiments and institutional approvals are legitimate pending dependencies. Missing reusable adapters, available source/config identities, a scientific dependency audit, and draft-level visual verification are implementation work and should not be classified as waiting for experimental results.

This review reports the initial findings and separately records any independently checked corrections. The implementing agent received concrete findings during the audit and is making changes concurrently. A change described by that agent is not treated as verified until the changed artifact is inspected. Section references below are stable scientific locations; initial line numbers can move after assembly.

## Scope and independence

I read the full plan at `C:/Users/leoni/Documents/projects/hierarchical-video-prediction-integration/docs/plans/DISSERTATION_DETAILED_REFACTORING_PLAN_2026-09-02.md`, inspected the nine current manuscript files, English fragments, migration/evidence registries, and the actual assembly/plotting/audit code. I independently compared the relevant implementation contracts in:

- Path B: `hvp_hca/allocation.py`, `hvp_hca/postcampaign.py`, `hvp_rev/rhvd.py`, `scripts/v2/train_article4_campaign.py`, and `scripts/v2/finalize_article4_campaign.py`;
- OpenSTL-integrated: `openstl/models/hvp_integrated_model.py` and its model/test interfaces;
- the integrated controller's four-run prelock manifest.

I did not launch training or evaluation, allocate CUDA, open protected arrays, run campaign controllers, or modify manuscript, scientific source, or controller files. This report is the only file written by this reviewer. Existing QA summaries were not accepted as evidence that their checks were sufficient.

The reference worktrees observed during review were Path B `7a5979615f5354564a793083b673cf1ed9ec77be` and OpenSTL-integrated `ca335ba206ef6f51515a4f8eefcad1eed8c36c10`. These identify the inspected source, not an authorization or final experimental evidence lock.

## Findings requiring correction

### R1 — P1: The new confirmation estimand was described using the historical package-specific population

Locations: `04_CHAPTER_3.md`, Sections 3.5.2–3.5.4; `05_CHAPTER_4.md`, Section 4.3.3; `refactor_v2/registries/placeholders.json`, `A4_CONFIRMATION_TABLE` and `A4_MEC_DIAGNOSTIC_FIG`.

The initial Chapter 3 presents a world-point, package-specific set `R_(H,F)`, then discusses selected confirmation without distinguishing that historical construction from the actual v2 finalizer. The confirmation placeholder names `horizon4_foreground_mse`.

However, Path B's `finalize_article4_campaign.py:334` (`visibility_labels`) hard-codes a 96-frame observed window and a 28-frame recent reference. It flags a clip/horizon when any currently visible sprite was last observed before the recent reference; visibility is overlap of sprite and viewport world-column supports. The mask is common to the selected models, not recomputed from their individual packed tokens or spatial coverage. `evaluate_selected_run` then averages separately normalized clip/frame foreground errors over all four horizons, or over flagged re-entry frames. Its summary fields are `mean_foreground_mse` and `reentry_foreground_mse`, not the development horizon-four selection statistic.

The MEC evidence is also specifically a one-clip, one-direct-denoiser-forward attention probe at diffusion time 0.5. It is not an old-context ablation or a replicated quality test. A schematic package population and a generic placeholder would permit insertion of the wrong endpoint under an apparently precise method claim.

Required correction: label the package-specific population as historical E2; define the actual v2 fixed-reference frame mask and reductions; bind confirmation and MEC to their actual fields, role selection, arrays, and limitations. No campaign change is needed or authorized.

### R2 — P1: The displayed diffusion loss did not specify the executed weighting

Location: `04_CHAPTER_3.md`, Section 3.3.1, equation `3.P5`.

The initial expression uses one squared norm over an unspecified combined target. `hvp_rev/rhvd.py:493–520` instead sums the coarse and fine per-stream MSE means. Because those streams have different numbers of spatial cells, a concatenated unweighted norm or mean is not the same objective. The optional top stream in this campaign is context-only.

The development metric likewise initially says only “implementation-defined” foreground mask/reduction. `train_article4_campaign.py:215–258` uses target intensity greater than 0.5, foreground-pixel weighting within each fixed validation batch, then batch-size weighting of those batch metrics. This differs from the confirmation finalizer's per-frame normalization.

Required correction: state the sum of per-stream means and the exact development reduction, keeping development and confirmation weighting distinct.

### R3 — P1: Logical comparison families and result generators were not fully implemented

Locations: `scripts/register_dissertation_v2_evidence.py`; `scripts/plot_dissertation_v2_results.py`; `refactor_v2/registries/placeholders.json`.

At first inspection, the ALC figure registry lists only eight newly trained runs, omitting the six reused frontier runs needed for its seven-policy comparison. LVL lists only its two new dependency runs, omitting the reused one- and two-level comparator pairs. OVR lists four new runs, omitting its reused nested pair. The correct logical result counts are 14, 6, and 6, although unique training counts must remain 8, 2, and 4.

The plotting utility initially supports only `curve`, `dots`, `heatmap`, and `pareto`, saves a Matplotlib figure, and has no table or claim insertion mode. Nevertheless every pending table and summary claim points to it as its generator. Passing the declared `.md` table output to `savefig` is not a table-generation implementation. Nor is there an adapter from the known finalizer schema to the accepted plotting schema. The generic validator checks paired seeds in whatever cells are supplied, not that every registered family cell is present.

There is also a concrete Pareto-schema bug: the validator identifies a paired cell by `(method, x, y_category)` and requires both seeds at that exact `x`. When `x` is measured memory, the two trained seeds normally have different measurements; a valid two-seed Pareto bundle is therefore rejected unless their memory values happen to coincide. A stable configuration/quota/comparison-cell identifier must define pairing, while each seed retains its measured x-coordinate. Method colors should also be keyed by method identity rather than first-appearance order.

Most positive/neutral/negative variants are one repeated generic template; calibration, MEC, routing stability, and cross-dataset synthesis have different estimands and cannot all be filled as a candidate-versus-comparator quality effect. Dataset-specific integrated rows initially still say “Moving MNIST ... or TaxiBJ ...”.

Required correction: explicit logical reuse mappings, dependency-resolution imports, family completeness checks, field-specific adapters, table generation, shared claim insertion, and endpoint-specific outcome branches. Actual absent results should still produce no values. Verification should include incomplete-family, wrong-endpoint, and stale-hash rejection tests, not only a successful sample plot.

### R4 — P1: Known integrated source and configuration provenance was deferred unnecessarily

Locations: `refactor_v2/english/REPRODUCTION_APPENDIX.md`, Appendix Ж.v2; integrated placeholder rows; reference 178.

The initial reproduction appendix names the model/controller files and says the manifest declares four runs, but records no integrated source commit, source/test hash, concrete run IDs, or configuration hash. Reference 178 says the local revision is recorded in Appendix Ж, but that revision was absent there. All integrated result rows initially have empty exact-configuration lists.

The existing manifest `experiments/v2/campaigns/integrated_openstl_confirmation_2026-09-02.json` already names `INT_MM_S00`, `INT_MM_S01`, `INT_TB_S00`, and `INT_TB_S01`, and their two dataset config paths. It is explicitly `prelock_confirmation` with `launch_authorized: false`. Those facts can be recorded now without claiming a final evidence lock or authorizing a run.

Required correction: an immutable, dated editorial source snapshot linking the inspected model/tests/controller, four prelock run IDs/configs, dataset contracts, source commits, and SHA-256 values. Keep future accepted checkpoint/results separate. This is a reproducibility task, not a missing-measurement dependency.

### R5 — P2: Mechanical formula removal left incomplete reading passages and weak scientific dependencies

Locations: `03_CHAPTER_2.md`, initial Sections 2.2.1 and 2.3.2–2.3.5; `refactor_v2/registries/formula_dependencies.json`; `scripts/refactor_dissertation_v2.py`.

Examples include a Ukrainian sentence introducing a computation “as follows:” followed by the English sentence “The supporting relation is preserved as source equation ...”, then more variable definitions for the absent expression. The initial V-BLT architecture passage repeats this for equations 2.29 and 2.30. These are mechanically demoted displays, not a coherent complete English replacement paragraph. They leave the main narrative dependent on repeated appendix lookup without actually reducing the explanatory clutter.

The initial dependency ledger inventories the original 114 equations and preserves their surrounding text, which is useful. But `implementation_location` and `result_estimand` are generally repeated generic text, and figure/table dependencies are incomplete. In particular, `hvp_rev/evaluation` is not the actual path implementing the new primary metric or the historical E2 population. A count of 114 entries is not a formula-to-code/results audit. The new scientific equations also need explicit provenance/dependency records, not just the old-number migration inventory.

Required correction: replace affected passages as whole paragraphs/subsections, retain a compact mapping/object where needed, and fill actual code/function, estimand, destination, replacement, and downstream-object fields. Mark conceptual relations honestly rather than attaching a generic executable path.

### R6 — P2: The inherited result objects still violate the new figure/table rules

Locations at initial inspection: `03_CHAPTER_2.md`, Tables 2.3/2.4/2.10 and Figure 2.18; `05_CHAPTER_4.md`, Figures D4-E4/D4-E5; `02_CHAPTER_1.md`, Figure D1-resources; Chapter 3 caption 3.14.

I visually inspected the retained V-BLT point-estimate figure. It is a five-panel bar composite joining model size/GFLOPs with MSE/SSIM/PSNR, while color means model on the left and dataset on the right. It also duplicates the cost and quality tables. Table 2.4 mixes four dataset populations and a “best result” interpretation column. The published tables retain boldface on many independent metric winners, contrary to the declared selection-criterion rule. Preservation of publication evidence does not require retaining this main-text presentation; source copies can remain in the appendix.

The new E4/E5 plots initially label the axis only “Effect in declared metric units”; Section 4.5 and its caption do not name the measured endpoint. The reader cannot infer the meaning of a difference such as +0.02 without consulting the archive. They need the actual signed foreground-MSE contrast, units, population/horizon, and uncertainty unit. The new calibration is legible, although the early high errors visually compress the late ordering change; a late-window inset would improve that question without inventing data.

There are also two D1-resources captions after the image was deduplicated, and the retained Chapter 3 public caption still calls the benchmark LoopMom. These are small but concrete incomplete migrations.

Required correction: simplify the main objects, keep source assets in provenance, apply actual endpoint labels, and regenerate callouts. Do not manufacture uncertainty for the published point estimates.

### R7 — P2: The content-transfer ledger confuses preservation with publication attribution

Location: `scripts/refactor_dissertation_v2.py`, content-transfer construction; `refactor_v2/registries/content_transfer.json`.

The initial script assigns A3 when a Chapter 1 paragraph contains strings such as `консистент`, including dissertation synthesis, and assigns A2 to every Chapter 2 paragraph before Section 2.3 and A1 to every later one. The ledger is therefore a mechanical source-paragraph inventory, not verified claim-level attribution to Articles 1–3. A heading and a dissertation-level generalization can be labelled A3 merely because they contain the word consistency. Generic destinations such as “English restatement/source dependency record; baseline preserved” do not locate the actual replacement claim.

Required correction: separate `source_manuscript_preservation` from `publication_claim_origin`; record genuine article section/page/figure/table and exact replacement/appendix anchor for claim-bearing transfers. Unattributed dissertation synthesis should remain labelled as such. No loss of the core published evidence was found, but the ledger should not certify more than it checks.

### R8 — P1 for full-plan completion: Final build and release checks are not complete

Locations: the refactor build/QA scripts and `docs/dissertation/final`.

At initial inspection, the existing DOCX builder is hard-coded to `condensed_english` and its English front-matter headings. It does not build this mixed-language source tree. The files in `docs/dissertation/final` are pre-refactor artifacts. A source inventory and existence check for image paths cannot verify equations, table overflow, figure legibility, callout resolution, pagination, or submission formatting.

During review a navigable HTML reading draft and rasterized SVG gallery appeared under `refactor_v2/preview`. I verified the gallery assets exist and visually inspected the new calibration, WindowDiT, and E4 images. This is useful draft-level progress. It is not yet a typeset full-document or institutional Word/PDF verification; mathematics in that HTML is deliberately source-form.

Final result lock, final numbering, language harmonization, approved administrative information, and permissions remain legitimate release gates. But the completed-plan claim must remain false until the actual selected source tree builds and its final artifact is audited. A placeholder-bearing typeset draft can be verified before the experiments finish.

## What is implemented well

- All four chapter roles are retained. The new programme is nested under the six dissertation tasks rather than replacing the contributions of Articles 1–3.
- Both abstracts, novelty, Chapter 4, and conclusions share result-pending summary keys. The old integrated prototype is not passed off as the new architecture.
- Chapter 1 distinctly discusses retained state, internal activations, parameters/optimizer, and workspace; spatial/temporal allocation, reversibility, and routing are not multiplied into a claimed total saving.
- The V-BLT-to-router boundary is explicit. The integrated position encoding is correctly identified as physical Fourier features, not RoPE-ND.
- The integrated description matches the inspected code in its genuine gather, coordinate origin/cell-centre convention, signed time and protocol denominator, level identity, ordinary encoder/router/decoder boundaries, and identical-forward internal reversible coupling. It candidly separates static/density/hybrid inference from mixed training and notes non-differentiable membership, selected-score gradients, tie/margin limitations, and uncompleted deployed-dtype tests.
- CEX/LVL/OVR are described as compound registered constructions, not isolated causal effects of reach, level count, or overlap. The main frontier budgets and unique-run accounting are correct.
- Historical E2 remains substantive; E5a's non-replicated reversal and E6's coverage/identity boundary are retained. The new four-frame endpoint is not claimed to establish long self-fed stability.
- I independently split the baseline appendix into 80 nonempty paragraph blocks after line-ending normalization: every block is present verbatim in the current appendix. The Ukrainian appendix-letter sequence and institutional/provenance material remain. Nearly all baseline Chapter 1/2 paragraphs are still present in the main text or archive; the exceptions inspected are deliberate replacement/callout changes. No material loss of Articles 1–3 was identified.
- Calibration values are linked to the archived analysis and copied records; the 30k discussion correctly limits the convergence claim to the observed range and identifies the two arms as seed-0 configurations, not two seed replicates.

## Legitimately pending versus missing implementation

| Work | Correct status |
|---|---|
| New hierarchy result values, selected confirmation, completed dependent hashes | Pending finalized campaign artifacts; no invented values |
| Optional selected-checkpoint rollout | Inactive unless separately authorized and preregistered; no retraining implied |
| Integrated CUDA profiles, checkpointing comparator, trained ablations, smaller quota, matched baselines, four full runs | Conditional evidence cells; absent cells must narrow claims |
| Final author list, conference records, signed institutional wording/scans, figure permissions | Author/institution input; do not infer approval |
| Shared summary claim filling and expanded final conclusions | After evidence lock; current six-item conclusion is a skeleton, not the intended final 1,500–2,000-word synthesis |
| Family reuse bindings, exact endpoint adapters, available source/config hashes, formula/content dependencies | Implementable now; these are not experimental delays |
| Coherent demotion prose and single-purpose inherited visuals | Implementable now |
| Draft visual/typesetting checks | Implementable with placeholders; final pagination/numbering remains after result lock |
| Final bilingual/institutional release audit | Not complete; requires both technical and external dependencies |

## Bibliography assessment

The core publication records are preserved. The new DynamicViT entry agrees with the [primary paper record](https://arxiv.org/abs/2106.02034); the [official OpenSTL repository](https://github.com/chengtan9907/OpenSTL) is an appropriate contextual implementation reference. The [PyTorch checkpointing documentation](https://docs.pytorch.org/docs/stable/checkpoint.html) is a mutable version redirect, so the experimental runtime version must still be pinned locally. Attempts to open the three publication DOI links through the browsing service failed; that is not evidence that the DOIs are invalid, and I do not claim an independent full bibliographic-metadata verification.

The initial round-trip registry leaves references 5, 6, 7, 8, 10, 11, 41, and 136 uncited and treats a numeric array as citation 0. Some of these are being repaired during review. A robust final citation pass must distinguish prose citations from arrays, confidence intervals, code, and bibliography entries, and must not add a blanket citation merely to suppress an orphan warning. The newly added reference 178 also depends on correcting the missing local-revision record noted in R4.

## Remediation checks during this review

At 08:56–08:59 local time, the English Chapter 3 fragment contains a revised per-stream loss, the exact batch-weighted development metric, an explicit historical E2 package-population label, and the separate fixed 96/28 v2 confirmation rule. These are scientifically appropriate corrections to R1/R2. The assembled manuscript and generated registry still need to be regenerated consistently and checked before those findings can be closed.

The HTML preview and PNG gallery are verified present. Several other announced corrections are still in progress at this report's first write; they are not silently counted as completed. An updated verification addendum should name the assembled-artifact hashes and the checks that actually passed.

## Acceptance recommendation

Accept this as a **revised evidence-gated dissertation draft**, after closing the currently implementable defects and verifying the regenerated artifacts. Do not label every point of the plan complete or call the work a release candidate. The remaining experimental and institutional dependencies should be enumerated explicitly, and full completion should be decided against the plan's acceptance checklist after evidence lock, final build, and independent typeset/provenance review.
