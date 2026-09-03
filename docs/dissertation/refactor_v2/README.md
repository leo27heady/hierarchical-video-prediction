# English refactor on the preserved Ukrainian manuscript

The editable dissertation is in `../condensed_translated_humanized_ukrainian/`, on the dedicated `codex/dissertation-refactor-v2` branch in `hierarchical-video-prediction-dissertation`. This is an evidence-gated draft, not a submission release. No active Path B code/config, GPU process or experiment has been changed by this editorial implementation.

Start with [the full typeset reading draft](preview/dissertation_reading_draft.html), then [the implementation and acceptance report](../../reviews/DISSERTATION_REFACTOR_IMPLEMENTATION_REPORT_2026-09-03.md). The [independent blind review](../../reviews/BLIND_REVIEW_DISSERTATION_IMPLEMENTATION_2026-09-03.md) preserves initial findings; the implementing agent's closure/status record is separate because the independent final re-check was interrupted.

## Editing and rebuilding

New English paragraphs live in `english/SURGICAL_BLOCKS.md`, `english/CHAPTER_3.md`, `english/CHAPTER_4.md`, and `english/REPRODUCTION_APPENDIX.md`. The surrounding Ukrainian source is recovered deterministically from frozen Git commit `cf3f661aa6e809dd30ca92f91e2d17ec257ac4b8`. Change these source fragments or the documented assembly mapping, not only the generated nine manuscript files: regeneration would otherwise replace an unported manual edit. Preserve the baseline and review diffs.

From the dissertation worktree, one command assembles all nine files, regenerates ledgers/assets, runs the draft audit and tests, builds the complete MathML preview, and checks it in an isolated CPU-only headless browser:

```powershell
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/build_dissertation_v2.py --preview --browser-check
```

Python assembly uses the standard library and pure configuration modules only; it never imports torch or opens data/checkpoint tensors. The SVG generators use archived calibration logs and source-era table transcriptions, not the incomplete live campaign. The preview uses bundled Node, `marked`, `sharp`, and Playwright; KaTeX 0.16.22 is vendored with its MIT licence. Set `DISSERTATION_NODE`, `CODEX_DEPENDENCY_NODE_MODULES` or `DISSERTATION_BROWSER` if moving to another host. No visible browser session or W&B run is created.

The release gate is deliberately separate and must currently fail:

```powershell
python scripts/audit_dissertation_v2.py --release
```

It is not legitimate to suppress this failure while result, author/institutional, final-language, numbering/pagination, permission or Word/PDF gates remain unresolved. The HTML is a complete typeset reading draft, not institutional page-layout certification. The existing `docs/dissertation/final` files are pre-refactor artifacts and are not this deliverable.

## Results: fail closed, then insert once

`registries/placeholders.json` records 27 evidence keys: two bounded filled slots (calibration and CPU gate summary), with 25 unique result/summary slots still pending. The new campaign remains 86 unique training runs. Logical families include reuse: FRT 36, ALC 14, CEX 12, LVL 6, OVR 6, STS 24. Six dependent training IDs and the reused selected hierarchical/nested roles resolve from the controller's locked dependency ledger; they are not invented in the editorial snapshot.

1. After finalization, review and create an immutable editorial acceptance JSON with `status: accepted_for_dissertation`. Its `artifacts` mapping names `aggregate`, `resolution`, `selection`, `final_report`, and `data_manifest`, each with an absolute `path` and SHA-256. This is a human-reviewed acceptance of already completed artifacts, not a command to open a holdout or accept partial progress.
2. Run `scripts/adapt_dissertation_a4_results.py` with that acceptance file and its hash. It rejects incomplete 86-run aggregates, stale campaign/config/dependency bindings, missing logical comparators, wrong training endpoint, incomplete seeds and wrong metrics. `--family FRT|ALC|CEX|LVL|OVR|STS` uses only `horizon4_foreground_mse`. `--family CONFIRMATION` uses one actual finalizer summary endpoint at a time: `mean_foreground_mse`, `reentry_foreground_mse`, or `non_reentry_foreground_mse`. Keep those populations separate; the all-frame table can occupy the main slot while the re-entry/non-re-entry tables are supplementary evidence. `--family MEC` reads one declared `normalized_entropy` or `source_mass:<span>` probe endpoint per output.
3. Render the accepted bundle with `scripts/plot_dissertation_v2_results.py --kind curve|dots|heatmap|pareto|table`. Every output carries its exact input rows/hash. A table is Markdown, not a plotting call with a `.md` extension. For a Pareto view, `cell_id` identifies the paired configuration; each seed keeps its own measured memory x-coordinate.
4. Insert a generated table/SVG through `scripts/insert_dissertation_v2_results.py --key ... --bundle ... --sha256 ... --artifact ...`. Its sidecar must match the accepted bundle hash. Shared summary claims use the same tool without `--artifact`, with explicitly reviewed `interpretation: positive|neutral|negative` and complete `interpretation_text` inside a bundle bound to the corresponding `placeholder_key`.
5. `accepted_insertions.json` records immutable insertions. Rebuild reapplies them only while all source/bundle/artifact hashes remain unchanged. This keeps abstract, novelty, Chapter 4 and conclusion occurrences of one key synchronized. An intentionally changed claim requires an explicit version-controlled acceptance replacement, not silent overwriting.

For a table sharing a family with a primary figure, pass the registered table key to the adapter, for example `--family FRT --key A4_FRT_RESULTS_TABLE`. For a synthesis spanning distinct datasets/endpoints, use a claim bundle with `kind: claim`, `status: accepted`, `placeholder_key`, `interpretation`, `interpretation_text`, and `evidence_bundles: [{path, sha256}, ...]`. Each referenced endpoint bundle is validated separately: a cross-dataset sentence must not create a pooled numerical metric simply to satisfy a plotting schema.

Example command shapes (the paths/hashes below are illustrative arguments, not existing accepted results):

```powershell
python scripts/adapt_dissertation_a4_results.py accepted/a4_acceptance.json --sha256 VERIFIED_ACCEPTANCE_SHA256 --family FRT --output accepted/a4_frt.json
python scripts/plot_dissertation_v2_results.py accepted/a4_frt.json --sha256 VERIFIED_BUNDLE_SHA256 --kind curve --xlabel 'Retained tokens' --output figures/dissertation_v2/a4_frt_budget_response.svg
python scripts/insert_dissertation_v2_results.py --key A4_FRT_PRIMARY_FIG --bundle accepted/a4_frt.json --sha256 VERIFIED_BUNDLE_SHA256 --artifact figures/dissertation_v2/a4_frt_budget_response.svg
```

Integrated measurements use the same accepted-row interface, with dataset-specific metrics, matched host/batch/dtype/model/checkpoint contracts and separate memory/throughput endpoints. Their current source snapshot is explicitly prelock: `INT_MM_S00/S01` and `INT_TB_S00/S01` have known config-file hashes but no finalized checkpoint or quality result. Future profiler-specific adapters must match the actual accepted output schema; the four final runs do not manufacture trained component ablations or local baselines.

## Navigation and provenance

- `EXPERIMENT_MATRIX.md` and `registries/experiment_matrix.json`: exact frozen run/config identities, budgets, ages and cell assignments.
- `TASK_CLAIM_MAP.md`: six dissertation tasks, chapters, conclusions and publications.
- `PUBLICATION_TRANSFER_MAP.md` and `registries/content_transfer.json`: article-specific scientific anchors and separate source-paragraph preservation.
- `registries/formula_dependencies.json`: every one of the 114 original formula objects and its destination.
- `registries/current_formula_dependencies.json`: all current numbered equations, including the 19 new/grouped definitions and exact or explicitly conceptual implementation contracts.
- `registries/asset_disposition.json`, `generated_assets.json`, `source_figure_tree.json`: original and generated asset hashes and provenance.
- `registries/appendix_migration.json`: all 80 original nonempty appendix records preserved verbatim; Ukrainian letter order unchanged.
- `registries/integrated_source_snapshot.json`: inspected source commits, model/test/controller/config hashes, four prelock IDs; no experiment authorization.
- `QA_REPORT.md`, `math_render.json`, `browser_qa.json`, `preview/figure_gallery.html`: draft checks and visual evidence.

After any context compaction, reread the detailed plan, its implementation-status addendum, this README, the implementation report and the independent review before editing. Do not restart the live hierarchy campaign or substitute unfinished results for the pending slots.
