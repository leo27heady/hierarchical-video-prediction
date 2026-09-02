# Blind Review of the Dissertation Refactoring Plan

**Date:** 2 September 2026

**Reviewed plan:** `docs/plans/DISSERTATION_DETAILED_REFACTORING_PLAN_2026-09-02.md`

**Review mode:** independent audit against the current Ukrainian manuscript, the 31 August critical
review, the Path B decision record and experiment plans, the integrated-model plan and verification
report, and read-only campaign artifacts/code. The reviewed plan and decision register were not
edited.

## Executive verdict

The plan is directionally strong and follows the author's latest requirements better than the
earlier proposed wholesale chapter reorder. It correctly preserves the current four-chapter
semantic order, keeps Chapters 1--2 as the foundation, treats the hierarchy campaign and integrated
OpenSTL model as one new article, separates retained state from activation memory and total VRAM,
rejects the legacy second-order integration as the final architecture, and adopts sensible rules for
seed display, figures, tables, and unfinished results.

It is **not ready to execute unchanged**. Two P0 claim-to-design mismatches would permit the revised
dissertation to overstate what the new evidence establishes:

1. the new TRB campaign is a distant-context allocation study with a four-frame prediction target,
   not an evaluated long autoregressive rollout; and
2. the sealed confirmation set is evaluated only for a selected shortlist, not for the complete
   budget frontier, allocation, level, overlap, and space--time matrices.

Several P1 omissions would also weaken the thesis if the plan were followed literally: front matter
is not planned asset-by-asset, published contribution tasks could be displaced by the replacement
task list, the proposed appendix replacement drops mandatory administrative/contribution material,
and the integrated result section assumes component and baseline evidence that the current run plan
does not yet guarantee.

**Recommendation:** approve the structural direction only after the P0 corrections below are made
and the P1 items are converted into explicit editing/evidence gates. No current partial result needs
to be discarded; the correction is chiefly to narrow labels, preserve complementary legacy
evidence, and make evidence status visible.

## Sources and checks

The audit covered:

- all nine files in the current Ukrainian dissertation tree, including the actual chapter headings,
  numbered equations, figures, tables, front matter, conclusions, publication map, and appendices;
- `DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`;
- all plans in both requested plan directories, with particular attention to the detailed
  refactoring plan, decision register, complete Article 4 matrix, 30k operations plan, integrated
  architecture plan, and verification report;
- the read-only Article 4 status and the post-campaign evaluation path. At audit time, 21 run
  directories were complete and one was running; this matches the detailed plan's 21 complete, one
  running, and 58 pending statement;
- the confirmation evaluator and packed-window batcher, without running CUDA or modifying the live
  Path B worktree.

The plan's basic manuscript inventory is consistent with the earlier audited totals: Chapters 1--4
contain 114 numbered equations and 77 figures. The reported completed FRT means are appropriately
labelled incomplete development observations. The numerical status snapshot was not treated as a
final dissertation result.

## P0 findings

### P0-1 — The central “long-horizon” wording exceeds the new campaign's temporal prediction design

**Evidence.** The detailed plan proposes H1 as an improvement in “long-horizon primary error”
(lines 393--394), calls the retained-context study part of long-horizon prediction throughout, and
plans to make it Chapter 4's primary evidence. The actual v2 batcher supplies observed latent states
at selected ages and jointly predicts only four target frames. Development uses the common first
target at frame 96 and targets 96--99; sealed confirmation uses target start 196 and again predicts
four frames. The confirmation evaluator calls the denoiser once for this four-frame window. It does
not self-feed predictions through a long multi-window rollout.

This design can test whether distant observed evidence and its spatial resolution affect a short
future window. It does **not** by itself test error accumulation, retained-state updates with
predicted values, or consistency over a long autoregressive forecast. Those mechanisms are central
to the current manuscript: Chapter 1 frames self-feedback as a source of failure; Chapter 3 defines
four-frame ring-buffer updates; and legacy E5/E6 explicitly evaluate long self-fed behavior.

**Risk.** Replacing the old headline with the new matrix while continuing to call its endpoint a
long-horizon prediction error conflates long *context/reach* with long *forecast horizon*. A reader
could reasonably infer evidence about autoregressive stability that was never measured.

**Required correction.** Make the distinction a global terminology and claim gate:

- call A4-FRT/ALC/CEX/LVL/OVR/STS a **long-context/distant-evidence allocation study with a
  four-frame prediction endpoint**;
- rewrite H1 to refer to horizon-4 foreground MSE conditioned on distant observed context, not
  “long-horizon primary error”;
- reserve “long-horizon rollout,” “autoregressive consistency,” and “self-fed stability” for evidence
  that actually iterates predicted states;
- retain E5 and E6, and any sound E2 rollout material, in the main text in a clearly bounded but
  substantive role because they test a dimension the new matrix does not replace; and
- if a selected-model autoregressive rollout can be performed under a predeclared, leakage-safe
  protocol, add it as a separate experiment and placeholder. Do not make it a silent extension of
  the current confirmation family.

The revised narrative should explicitly say that long retained reach is a necessary design property
for some long-horizon systems, but a four-frame endpoint does not demonstrate long-rollout stability.

### P0-2 — “Confirmed frontier” language is broader than the sealed confirmation scope

**Evidence.** The plan correctly says in Table 7.1 that confirmation covers “selected arms only,”
but later instructs Chapter 4 conclusions to begin with a “confirmed hierarchy frontier” (line 804)
and makes the new campaign and locked confirmation jointly the primary hierarchy evidence. The
campaign's post-training contract selects a small set of roles from final development means: best
Flat at 1,792, best non-flat hierarchy at 1,792, a failure case for mechanism diagnostics, and the
smallest hierarchy within the registered development tolerance. Only the selected confirmation
roles are evaluated on the fresh 500-clip holdout. The full six-budget frontier, seven-policy ALC,
three-level comparison, three overlap modes, and 4-by-3 STS surface are not independently confirmed
as complete families on that holdout.

Several families are also adaptively dependent: ALC selects inputs for LVL and OVR, and the same
development results define which arms reach confirmation. Two displayed training seeds improve
transparency but do not support a training-seed population confidence interval.

**Risk.** The plan could turn a well-controlled development matrix plus shortlist confirmation into
a claim that every shape and interaction of the frontier was confirmatory. That would undermine the
otherwise strong separation between development, selection, and holdout evidence.

**Required correction.** Add an evidence-status vocabulary that is mandatory in headings, captions,
tables, abstract, and conclusions:

- **registered development estimate:** all 30k matrix results;
- **direction reproduced across two training seeds:** only when both seed directions agree;
- **sealed-holdout confirmation:** only the exact selected role/configuration and registered
  population evaluated on the fresh holdout;
- **mechanism diagnostic:** A4-MEC, never a separate quality confirmation; and
- **historical/transfer boundary:** legacy E2--E8 evidence.

Replace “confirmed hierarchy frontier” with wording such as “registered development frontier with
sealed confirmation of the selected contrasts,” unless the whole family is later evaluated under an
authorized confirmation contract. Add a claim-to-artifact matrix listing each final sentence-level
claim, its population, config(s), seed unit, selection status, and whether it was exposed to the
sealed holdout.

## P1 findings

### P1-1 — The plan omits the front matter as a first-class refactoring target

The detailed section-by-section plan begins with the Introduction. The actual `00_FRONT_MATTER.md`
contains both Ukrainian and English abstracts, the contribution list, novelty statements, software
claims, the Article 1--4 publication list, and repeated descriptions and numbers for the obsolete
fixed-region/second-order prototype. These are among the highest-risk locations for stale claims,
yet there is no asset disposition, placeholder scheme, or acceptance check for them.

Add a dedicated front-matter phase covering both abstracts, keywords, novelty/contribution bullets,
publication list, software outputs, old prototype numbers, and final counts. Every Chapter 4 result
placeholder that can affect an abstract or novelty claim needs a linked front-matter placeholder.
Abstract and conclusion claims must be generated from the same claim ledger as the result sections,
not updated manually.

### P1-2 — Replacing the research-task list may erase the logical role of Articles 1 and 2

The plan tells the Introduction to replace the old task list with questions about the new hierarchy
families and integrated composition. The current six tasks, however, provide the formal dissertation
home for TempVerseFormer, V-BLT/RoPE-ND, the evaluation methodology, and the retained-context model.
If those tasks are replaced by seven campaign questions, Chapters 1--2 may remain physically present
but become detached from the stated aim, novelty, conclusions, and publication contribution map.

Retain top-level dissertation tasks for:

1. system analysis and task-specific consistency;
2. reversible temporal computation;
3. V-BLT, RoPE-ND, and spatial-complexity diagnosis;
4. hierarchical retained-context design;
5. evaluation/observability/statistics; and
6. empirical operating region and integrated composition.

Place FRT/ALC/CEX/LVL/OVR/STS and routing/reversibility questions underneath tasks 4--6 as registered
research questions or hypotheses. Then require a task-to-chapter-to-conclusion-to-publication map.

### P1-3 — Article 3 preservation cannot be reduced to a gallery of moved figures

The plan's Appendix H is titled “Preserved publication figures,” and Chapter 1 moves much of the
static consistency/control sequence there. Article 3's dissertation value is not merely its pixels:
it supplies the taxonomy of appearance/style, identity, geometry/shape, concept consistency, and
the comparison of input-, weight-, structure-, inversion-, and attention-based control interfaces.
Removing the explanatory synthesis but retaining a gallery would preserve provenance without
preserving scientific contribution.

Use a content-transfer ledger at paragraph/claim/figure/table level for all three articles. Keep a
compact Article 3 synthesis in Chapter 1, including its limits as a qualitative static-image review,
and move only tutorial redundancy and secondary illustrations. For Articles 1 and 2, retain the
executed method boundaries, published numerical evidence, and author-contribution statements in the
main narrative; appendices should carry derivations and exhaustive assets, not the entire scientific
substance.

### P1-4 — The proposed appendix replacement drops mandatory and already useful material

Section 11 replaces the present appendix scheme with A--H but does not give a destination to the
current administrative records and acts of implementation. It also risks losing the publication/
personal-contribution table, interpretation licences, protected-set evidence, and figure permission
records. These are not decorative: some are institutional requirements and others constrain what
may be claimed.

There is also a direct internal cross-reference error: line 583 says Appendix G contains the complete
configuration/hash matrix, while Section 11 assigns the hierarchy registry to Appendix D and
software/provenance to Appendix G.

Do not replace the appendices by title alone. Create an old-to-new migration matrix for every current
appendix section and asset. Preserve dedicated destinations for:

- publications, result mapping, and personal contribution;
- administrative records and signed implementation acts;
- comparison licences and evidence-status rules;
- statistical contracts and protected-set results;
- software reproduction;
- figure sources, licences, and permissions; and
- historical integration evidence.

Use the institution's required Ukrainian appendix-letter convention or document the approved
conversion before changing labels. Fix Appendix D/G references before any skeleton is generated.

### P1-5 — The integrated Chapter 4 evidence objects are not fully backed by a locked experiment schedule

The plan proposes a six-configuration component matrix, matched local baselines, routing variants,
quota frontiers, and cross-dataset Pareto claims. The current integrated execution contract guarantees
four long full-model runs (two datasets by two seeds) after bounded gates. Flat/hierarchical and
vanilla/reversible profiles can be paired only where mathematically possible; static versus dynamic
quality requires route-aware training; and Flat versus hierarchy generally requires separately
trained models. Existing baseline checkpoints are usable only if their data, preprocessing,
selection, schedule, precision, and metric contracts truly match.

The plan must therefore attach a status and run source to every intended evidence cell:

- CPU correctness gate;
- same-checkpoint execution/profile comparison;
- separately trained development ablation;
- long full-model confirmation;
- matched local baseline; or
- published contextual reference only.

If the six-cell component matrix or matched baseline runs are not completed, 4.8.2 and 4.8.6 must
contract to feasibility/development evidence. A full model cannot be called Pareto-improving merely
because its two seeds are competitive against unmatched published numbers. Pareto claims require
the declared matched comparator set and the same memory-measurement protocol.

### P1-6 — Checkpointing is missing from the integrated reversibility comparison

The current Chapter 2 appropriately treats checkpointing as the practical alternative to
reversibility, and the 31 August review explicitly requested ordinary BPTT **and checkpointing** under
the same model and loss. The detailed plan's integrated matrix compares vanilla and reversible modes
but never requires a checkpointed core/end-to-end comparator.

Add checkpointing as a practical baseline at matched model, batch, precision, and horizon, reporting
peak allocated/reserved memory, throughput, and recomputation. If it is not run, restrict the claim
to “lower than ordinary autograd storage” and do not imply best available memory--compute trade-off.

### P1-7 — Routing modes and training regimes are conflated

The detailed plan asks for equal-quota comparison of “static, density, hybrid, and mixed-route
behavior.” In the implemented contract, `mixed` is a training exposure policy; evaluation uses a
specified route (currently hybrid). It is not a fourth inference routing estimator comparable to
static, density, and hybrid. A same-checkpoint switch test is a distribution-shift diagnostic, while
separately trained static and mixed models answer a training-regime question.

Split the evidence design into:

- inference policy comparison on a route-aware checkpoint: static/density/hybrid, labelled as a
  switch diagnostic;
- training-regime comparison: static-trained versus mixed/route-aware-trained, requiring separately
  trained models; and
- quota comparison: smaller real `K` versus larger `K`, with actual gathered tensor and metadata
  bytes.

Do not attribute gains from mixed training solely to the router score. Record the top-k
non-differentiability, selected-score path, route collapse/margin, and membership stability as method
limitations.

### P1-8 — LVL, OVR, and CEX need narrower estimands than the detailed prose currently gives them

The plan says LVL holds “the selected policy” fixed, but the registered three-level construction
adds a top stream, reallocates the remaining coarse:fine payload, rounds coarse slots, and adds 1,280
parameters. It is a whole-representation comparison at fixed total payload, not a pure effect of the
number of levels. OVR changes representation semantics (ordinary fine versus residual fine), and
exclusive support relocates coarse ages; it is not overlap alone. CEX keeps recent fine detail fixed
but increases total budget, token length, attention work, and reach together.

Use the following estimands:

- LVL: effect of the **registered one-/two-/three-level construction at 1,792 tokens**;
- OVR: effect of the **registered ownership/representation package**, not temporal duplication in
  isolation; and
- CEX: marginal response to adding exclusive older coarse slots, with increasing budget and compute,
  not a fixed-memory causal reach effect.

These are still useful scientific questions, but their captions, hypotheses, and conclusions must
name the compound intervention.

### P1-9 — Placeholder coverage does not yet match the proposed evidence coverage

The placeholder protocol is a good foundation, but the key list omits several objects the plan later
requires: A4-MEC, integrated gate summary, matched baseline results, checkpointing comparison,
component-ablation table, routing stability, and separate Moving-MNIST/TaxiBJ conditional claim
blocks. `{{INT_PARETO_FIG}}` is also ambiguous because the plan calls for dataset-specific reporting.

Expand the registry so every result-bearing sentence, not only every graphic, has a stable key. Each
row should include evidence status, exact config IDs and hashes, data population/split, selection
role, training seeds, uncertainty unit, metric implementation, generator commit, expected artifact,
failure/absence behavior, and allowed positive/neutral/negative wording. Use separate Pareto keys per
dataset and a cross-dataset synthesis key only if its estimand is explicitly defined.

## P2 findings

### P2-1 — Article numbering is factually wrong in the Chapter 1 asset instructions

The plan calls the static-control material “Articles 1 material” and labels Figure 1.6 for an
“Article 1 support appendix.” In the current publication map, V-BLT is [A1], TempVerseFormer is [A2],
and the static diffusion-consistency review is [A3]. Correct these to Article 3/[A3]. Use titles or
stable publication keys in the transfer ledger so numeric labels cannot drift.

### P2-2 — “Every tensor shape in one architecture figure” conflicts with the one-question rule

One Chapter 3 instruction requires a single architecture figure to display every tensor shape and
trainable/frozen boundary. That is likely to recreate the composite-density problem the plan is
trying to solve. Keep the architecture figure conceptual and legible; place exhaustive tensor shapes
in a compact adjacent contract box or appendix table. A small main-text method table may be clearer
than forcing configuration data into a schematic, even if result tables remain the default numbered
tables.

### P2-3 — The zero-method-table rule is too rigid for reproducibility

“Main-text numbered tables are for empirical results” is a useful default, not a scientifically
universal rule. Readers need to understand the manipulated packages without repeatedly consulting an
appendix. Preserve one compact, single-purpose method summary when it materially clarifies exact
budget, reach, level, and overlap differences. If institutional style demands numbered tables, label
it explicitly as a method/configuration table rather than converting it into an overloaded figure.

### P2-4 — Positional encoding needs a fully reproducible normalization contract

The plan correctly requires physical x/y, signed time, scale, level, and packed-permutation
invariance. Add the exact origin, denominator/range, cell-centre convention, input/future time sign,
dataset-resolution handling, and treatment of changed input/output horizons. State that
normalization is fixed by the physical grid/protocol, not recomputed from the selected token set.
Include future decoder queries in the positional contract and report numerical tolerance and tested
dtypes for permutation invariance.

### P2-5 — Partial-results prose needs direct provenance despite being explicitly provisional

Section 7.2 gives rounded development means and status counts but does not cite the aggregate file,
checkpoint step, per-run config hashes, or status snapshot identifier. Add a generated snapshot path
and hash/date, or remove the numeric table from the enduring plan once its layout purpose is served.
This prevents provisional numbers from becoming uncited quasi-results after the campaign advances.

### P2-6 — Formula reduction needs a “scientific dependency” audit, not only range-level actions

The formula policy is sound, but broad moves such as sending metric, assignment, and inference
details to appendices can break the trace from estimand to result. Before moving an equation, record
every downstream text, code-contract, figure, and table reference. Retain in the main text the exact
definitions of each primary endpoint and any nonstandard re-entry/foreground population. Standard
MSE, DDIM, softmax, and elementary inverse derivations can be safely compressed; bespoke selection
populations and compound object metrics should not disappear merely to meet a count range.

### P2-7 — Final feasibility estimates omit institutional and bilingual cleanup work

The 19--31 focused-day estimate is plausible for technical refactoring after results lock, but it
does not visibly allocate time for updating both abstracts, institutional publication/implementation
records, appendix relabelling, permission checks, or final mixed-language harmonization. Treat these
as separate schedule items and dependencies on author/institution input. Otherwise the technical
manuscript can be “complete” while still not submission-ready.

## What should be retained from the plan

The following decisions are scientifically and editorially sound and should remain:

- preserve the current Ukrainian manuscript and edit from a clean post-campaign worktree;
- keep the current four-chapter semantic order and most Chapters 1--2 prose;
- draft new/refactored blocks in English without fragmentary code-switching;
- treat the TRB hierarchy programme and integrated OpenSTL model as two evidence blocks of one new
  article;
- distinguish retained state, activations, parameters/optimizer state, and workspace;
- keep routing outside the reversible boundary and require identical-forward output/input-gradient/
  parameter-gradient parity;
- position routed tokens by physical metadata rather than packed slot;
- require real gather-before-transformer packing and include routing metadata in state accounting;
- never claim equal-quota routing as a memory reduction;
- display both training-seed outcomes and distinguish clip-level uncertainty from seed variation;
- retain E5a non-replication and E6 coverage--identity tension as visible boundary evidence;
- use outcome-conditional prose and forbid invented multipliers or Pareto claims; and
- generate figures/tables and their plotted rows from hash-addressed machine-readable artifacts.

## Minimum correction set before manuscript editing

1. Insert the long-context versus long-forecast distinction and revise H1 and all “long-horizon
   error” labels.
2. Replace “confirmed frontier” with an exact development/confirmation status map.
3. Add front matter, conclusions, references, and every existing appendix to the migration ledger.
4. Preserve the six dissertation-level tasks and nest the new experiment questions beneath them.
5. Correct Article 3 attribution and preserve its conceptual synthesis, not only its figures.
6. Add an old-to-new appendix map, including administrative acts, personal contribution, licences,
   and protected evidence; fix the D/G cross-reference.
7. Add an integrated evidence-cell ledger and gate 4.8's size/claims on actually completed matched
   baselines, component ablations, and profiles.
8. Add checkpointing or explicitly narrow the reversibility claim.
9. Separate routing inference policies from mixed-training regimes.
10. Expand placeholders to cover claim text, MEC, gates, baselines, routing stability, checkpointing,
    and dataset-specific Pareto outcomes.
11. Narrow LVL/OVR/CEX language to their registered compound interventions.
12. Run a final claim audit from abstract through general conclusions against the same evidence
    ledger before removing any placeholder.

With these corrections, the plan would provide a defensible, feasible route to a cleaner thesis
without sacrificing the valuable published and legacy evidence already present in the manuscript.
