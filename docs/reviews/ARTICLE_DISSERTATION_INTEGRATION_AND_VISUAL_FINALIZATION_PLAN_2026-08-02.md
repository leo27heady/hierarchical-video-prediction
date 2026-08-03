# Article and Dissertation Integration, Comparison, and Visual Finalization Plan

**Date:** 2 August 2026  
**Original status (2 August):** read-only review and execution plan; no manuscript, figure,
experiment, or code changes were made as part of the planning audit.  

**Execution status (3 August 2026):** implemented. All P0 and P1 content, mathematics,
cross-strand integration, comparison-table, figure-integration, and non-published visual actions
defined by this plan have been completed and validated. The remaining actions are external
author/institution/venue actions rather than manuscript-content defects. The post-implementation
decision and evidence are recorded in
`docs/reviews/ARTICLE_DISSERTATION_FINAL_AUDIT_2026-08-03.md`.

**Author override, 3 August 2026:** figures reproduced from already published articles are frozen
and must not be redrawn, re-exported, regenerated, or overwritten. In particular this applies to
`fig53`--`fig55` and `fig57`--`fig60` together with their PDF companions. Any legibility concern is
to be handled through placement, scaling, a full-page or landscape slot, captions, and surrounding
explanation. This override supersedes the redraw/re-export recommendations for those assets below.

**Primary sources reviewed:**

- `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`;
- `docs/dissertation/final/DISSERTATION_MANUSCRIPT.md`;
- the authoritative dissertation sources under `docs/dissertation/rewrite/`;
- `docs/CONTENT_FINALIZATION_EXECUTION_PLAN_2026-07-19.md`;
- `docs/CONTENT_FINALIZATION_REPORT_2026-07-19.md`;
- `docs/CONTENT_FINALIZATION_CLAIM_MATRIX_2026-07-19.md`;
- `docs/MASTER_IMPLEMENTATION_PLAN_REACH_FIDELITY_TRUST_2026-07-11.md`;
- `docs/RESOURCE_EFFICIENT_LONG_HORIZON_VISION_2026-07-18.md`;
- `docs/PLAN_CWVAE_IMPROVEMENTS.md`;
- `docs/CWSTYLE_TABLE.md`;
- `docs/dissertation/rewrite/appendices/CHAPTER4_CAMPAIGN_ARCHIVE.md`;
- `docs/reviews/INDEPENDENT_MANUSCRIPT_CONTENT_REVIEW_2026-07-25.md`;
- every image linked from the final article and assembled dissertation.

The older dissertation supplied by the author,
`C:/Users/leoni/Downloads/LKulyk_dissertation_v2.docx`, remains the structural reference for the
conventional Introduction sequence and institutional genre. Its superseded scientific scope must
not be restored.

Priority labels used below are: **P0**, mandatory before any final/submission claim; **P1**,
high-priority pre-submission revision; **P2**, strengthening, compression, or polish; and **P3**,
optional enhancement or future scope.

---

## 1. Executive decision

### 1.1 Is the scientific programme complete?

**Yes, for the current conditional claims.** No additional GPU experiment is required. The central
reach--fidelity result, robustness programme, negative replication, explicit-access reference, E6
boundary, and reversible-computation measurements are sufficient for the current article and
dissertation.

Untested cells become scientifically necessary only if the manuscripts try to claim a pure
temporal-stride effect, universal depth benefit, head-independent frontier, natural-video
generalization, or multiplicative end-to-end context-memory and activation-memory savings. None of
those claims is needed.

### 1.2 Are the manuscripts final?

**No.** Their readiness differs:

| Artifact | Scientific core | Narrative | Comparison completeness | Visual state | Honest status |
|---|---|---|---|---|---|
| Fourth article | Strong and calibrated | Strong, focused, internally coherent | Mostly sufficient; full cost/study map belongs in supplement | Mostly strong; one main-figure P0 defect and several P1 layout issues | Near-final content; not submission-final |
| Dissertation | Defensible contribution set | Shared thesis is strong at the beginning and end but uneven through Chapters 1--3 | Claim tables are good, but the comprehensive arm/component map is absent from the assembled manuscript | Uneven; conceptual and inherited figures are not publication-unified | Advanced review draft; not final |

The earlier report's phrase **"scientific content final"** is too strong in light of the later
independent review. The more accurate status is:

> **The experimental evidence is closed; manuscript correctness, cross-strand synthesis,
> comparison presentation, visual finalization, and author/institution fields are not closed.**

### 1.3 Central integration judgment

The dissertation has a real unifying thesis, not merely an invented after-the-fact slogan:

> **Represent and retain long-lived information economically; reconstruct temporary computation
> when storing it is wasteful; evaluate both with criteria that expose long-horizon failure.**

The two-memory decomposition is the strongest spine:

$$
M_{\mathrm{system}}=
M_{\mathrm{parameters}}+
M_{\mathrm{optimizer}}+
M_{\mathrm{context/state}}+
M_{\mathrm{activations}}+
M_{\mathrm{transient}}.
$$

However, that spine is concentrated in the abstract, Introduction, late Chapter 4 synthesis, and
General Conclusions. Chapter 2 often reads as two prior articles placed together, and Chapter 3
contains too much history of superseded designs. The present integration is approximately
**3.5/5**: defensible, but not yet monograph-coherent.

---

## 2. What is already strong and should be preserved

1. **The article's central argument is strong.** Its research questions, contributions, untouched
   frontier, negative replication, E6 boundary, and conclusion all support the same conditional
   claim: hierarchy buys reach in the tested reach-limited region; flat context recovers when
   fine-detail fidelity dominates.
2. **The dissertation correctly distinguishes context/state memory from activation memory.** It
   does not claim their savings automatically multiply.
3. **The dissertation explicitly rejects a false end-to-end pipeline claim.** This is scientifically
   honest and must remain.
4. **The A3 criteria strand is genuinely used downstream.** Trajectory consistency is
   operationalized by the Chapter 3 instrument and Chapter 4 endpoints; this is a real
   methodological connection, not thematic adjacency.
5. **Chapter 4's question-driven order is successful.** It now reads as an argument rather than an
   experiment diary.
6. **The evidence calibration is excellent in the article and Chapter 4.** The manuscripts retain
   1,792 as unresolved, reject interpolation/equivalence, disclose B1 confounding, preserve the
   failed trust replication, and report difficult absolute E6 performance.
7. **The experiment programme is purpose-designed, not an accidental collection of scores.** It
   contains matched pairs, a true five-seed factorial, protected confirmation, robustness
   interventions, privileged references, and external-boundary evidence.
8. **All current linked figures exist.** All 42 unique linked dissertation/article assets have
   opaque white backgrounds; 39 have PDF companions.

These strengths should be protected during revision. The objective is not to rewrite the science
or inflate the novelty. It is to make the final structure accurately express the science already
present.

---

## 3. Prior-plan compliance audit

### 3.1 Content-finalization plan

| Planned work | Current status | Gap | Criticality |
|---|---|---|---|
| Headline reach--fidelity synchronization | Mostly complete | Chapters 1--3 still contain isolated contradictory or overbroad claims | P0/P1 |
| Article reconstruction | Mostly complete | Full configuration/cost table, public study map, visual cleanup, comparator verification, and availability/ethics end matter remain | P1 |
| Conventional dissertation Introduction | Structurally complete | Cross-strand contribution map is absent; novelty superlatives and author fields remain | P1; author fields P0 for deposit |
| Chapters 1--3 scientific red-team | **Not complete in outcome** | False rollout-distribution statement, invalid Lipschitz reasoning, overbroad constant-memory wording, CLIP causal claim, hierarchy guarantees, and rewrite-history prose survived | **P0/P1** |
| Question-driven Chapter 4 | Complete | Keep present order | Complete |
| Chapter 4 all-configuration/component presentation | Partial; absent from assembled manuscript | Broad grid exists only in `CWSTYLE_TABLE.md` and campaign archive | P1 |
| Two-resource synthesis | Mostly complete in prose/table | Missing visible full-strand crosswalk and two-panel synthesis figure; accounting wording still inconsistent in Chapter 2 | P0/P1 |
| Reference expansion | Numerically complete | Claim-level primary-source verification remains, especially novelty/comparator cells | P1 |
| Automated verification | Complete for implemented gates | Gates do not detect mathematical or conceptual overclaims | Human P0 review needed |
| Final red-team review | Performed once but insufficient | A new independent pass is required after correction | P1 |

### 3.2 Required synthesis artifacts from the systems vision

| Required artifact | Current state | Decision |
|---|---|---|
| Reach--fidelity frontier table with allocation, reach, target interface, parameters, per-seed effects and simultaneous intervals | Fragmented across Tables 3.1, 4.2, 4.3, prose, and the figure | Partial; rebuild one canonical table |
| Memory-efficiency arithmetic with tokens and actual bytes | Token arithmetic present; actual latent scalar/byte/storage accounting absent | Partial; add byte accounting with dtype and scope |
| Reach-differential mechanism table/figure | Pair-specific effects and whole-population effects present; mechanism corroboration mostly prose | Partial; add compact quantitative mechanism panel/table |
| Two-panel context-token and activation-VRAM frontier | Separate ingredients exist; combined synthesis not assembled | Missing; high-value integration artifact |
| Applicability/boundary table | Chapter 4.8 includes dense, spatial, temporal, checkpointing, and reversibility | Present but expand to all strands and final-predictor status |
| Post-hoc Pareto analysis | Optional and not needed | Do not add unless clearly labelled exploratory and artifact-derived |

### 3.3 CW-VAE improvement plan

`docs/PLAN_CWVAE_IMPROVEMENTS.md` belongs to an earlier system decision and is not a binding list of
unfinished experiments for the final joint-attention predictor. Its renderer and CW-VAE-backbone
programme should be marked **historical/superseded** or moved under a legacy-plans namespace. Its
unexecuted items are not current dissertation gaps.

---

## 4. Why the article strands still feel disconnected

### 4.1 The connection currently succeeds at only two levels

| Connection type | Current state | Required final form |
|---|---|---|
| Shared problem | Strong | Preserve: resource-constrained consistent long-horizon modelling |
| Shared systems principle | Strong | Make recurrent, not only introductory/concluding |
| Methodological handoff | Partial | State what each strand consumes or informs downstream |
| Evidentiary integration | Correctly separate | Preserve separate evidence and limitations |
| Executable integration | Absent and correctly denied | Do not fabricate one pipeline |
| Narrative continuity | Uneven | Add one crosswalk, one systems map, and chapter handoffs |

The dissertation should explicitly distinguish these four labels:

- **problem-level integration** -- all strands address the same resource-constrained process;
- **methodological integration** -- each strand acts on a named resource or measurement problem;
- **evidentiary separation** -- each result retains its own experiment, population, and boundary;
- **executable integration** -- only components actually used in Chapter 3 belong to the final
  predictor.

This vocabulary resolves the false choice between "four unrelated papers" and "one integrated
model." The correct claim is an integrated systems methodology with separately measured modules.

### 4.2 Misleading bridge passages that must be corrected

The following references are to the assembled dissertation and must be edited in their
authoritative `docs/dissertation/rewrite/` sources:

| Assembled location | Current problem | Required relationship |
|---|---|---|
| Lines 1188--1192 | "supported by" entropy adaptation and reversible optimization can imply component dependence | Shared resource methodology; not an implemented dependency |
| Lines 1292--1295 | Says RoPE-3D and temporal chaining are mechanisms used by the developed final technology | Separate A1/A2 mechanisms; Chapter 3 uses learned embeddings and window training |
| Lines 1410--1412 | Implies the validated long trajectory depends on time-agnostic backpropagation | Reversibility is relevant only when differentiating through a long temporal process |
| Lines 1594--1599 | Calls Chapter 3 the mature form of V-BLT | Shared allocation principle, not direct architectural lineage |
| Lines 2245--2247 | Says stochastic reversibility is precisely required by Chapter 3 | Direct contradiction: final predictor does not use reversible rollout training |

The document also repeats defensive non-integration disclaimers many times. Replace them with one
authoritative crosswalk plus short chapter-specific scope statements. Repetition currently makes
non-integration more salient than the actual systems connection.

### 4.3 Historical-design narration weakens the final scientific voice

Chapter 3 repeatedly discusses "old design," "earlier design," "superseded," "salvaged," and
"original draft." This explains document repair, not the final model. Retain the final rationale in
the chapter; move any institutionally useful deviation record to a compact appendix:

| Earlier proposal | Final implemented decision | Evidence/reason | Claim consequence |
|---|---|---|---|

---

## 5. Three viable integration strategies

### Strategy A -- minimal bridge repair

**Effort:** approximately 1--2 focused days.  
**Use only if time is extremely limited.**

1. Correct the five misleading bridge passages.
2. Add a one-page contribution crosswalk after the research tasks.
3. Add a short handoff paragraph at the end of Chapters 1--3.
4. Expand Chapter 4.8 to include A1, A3, and final-predictor status.
5. Replace repeated non-integration disclaimers with references to the crosswalk.

This fixes correctness and makes the map visible, but Chapter 2 still feels like two papers and
Chapter 3 retains too much revision history.

### Strategy B -- moderate systems-spine refactor (**recommended**)

**Effort:** approximately 4--6 focused content days, excluding figure production.

Apply Strategy A, then:

1. Add an early systems synthesis figure with parallel lanes.
2. Introduce one careful organising formulation:

   $$
   \min_{\mathcal R,\mathcal P,\mathcal A}
   \mathbb E[\mathbf L_{\mathrm{cons}}]
   \quad\text{subject to}\quad
   B_{\mathrm{ctx}}(\mathcal P)\le B,
   \;M_{\mathrm{act}}(\mathcal A)\le M,
   \;C_{\mathrm{train/infer}}\le C.
   $$

   Here A1 informs representation choice/allocation `R`, A4 manipulates the retained context
   package `P`, A2 supplies activation-management method `A`, and A3 defines the consistency
   criteria in the loss/evaluation vector. Label this an organising systems formulation, not an
   experimentally joint optimizer.
3. Reframe Chapter 2 around two resource questions: what information deserves representational
   capacity, and what temporal computation must be stored versus reconstructed?
4. Add a "published basis and dissertation-level extension" box distinguishing A1/A2 results from
   new dissertation synthesis and stochastic extensions.
5. Remove rewrite-history narration from Chapter 3.
6. Broaden Chapter 4.8 into a genuine all-strand operating-region synthesis.
7. Add a paragraph explaining why Article 4 is the empirical culmination of the state-memory and
   measurement arc, not a pipeline combining every article.
8. Lead the General Conclusions with three systems propositions before the seven-task receipt.

This gives the strongest coherence-to-effort ratio and preserves the accepted chapter structure.

### Strategy C -- monograph consolidation

**Effort:** approximately 8--12 content days before visuals and QA.

Apply Strategy B, then move much generic Chapter 1 material and detailed reproductions of A1/A2
methods to appendices; rewrite Chapter 2 entirely around systems questions; reserve the main text
for dissertation-level synthesis.

This would be the most monograph-like version, but it creates unnecessary numbering, pagination,
and institutional risk. It is not recommended unless the supervisor explicitly rejects a
publication-strand dissertation structure.

---

## 6. Recommended systems-spine implementation

### 6.1 Contribution crosswalk after the research tasks

Add one page after "Aim and Research Tasks" and before "Object and Subject of Research":

| Strand | Research problem | Intervention | System/resource term | Evidence | Principal result | Boundary | Used in final predictor? |
|---|---|---|---|---|---|---|---|
| A1 V-BLT | Non-uniform information density | Entropy-guided adaptive decomposition | Representation allocation | A1 experiments | Capacity follows estimated information density | Entropy is a proxy, not automatically calibrated | No |
| A1 RoPE-ND | Multi-axis positional structure | Independent rotary coordinates | Representation/coordinate interface | A1 experiments | Supports variable/cross-resolution positioning | Does not guarantee correct routing | No |
| A2 TempVerseFormer | Long differentiation tape | Reversible temporal replay | Activation-tape memory | A2 measurements | Constant-in-horizon local activation graph in tested recurrence | Replay metadata/noise and stability must be accounted for | No |
| A3 criteria | Undefined "consistency" | Conceptual, hierarchical, trajectory criteria | Measurement/decision layer | Analytical development | Turns consistency into explicit failure modes | Proxies do not guarantee causality or coherence | Methodologically consumed |
| A4 predictor | Retaining distant evidence under a context budget | Flat/spatial/temporal packages | Context/state memory | LoopMom, E2, E4--E8 | Tested reach--fidelity operating regions | Synthetic scope; no universal winner | Yes |
| Dissertation synthesis | Auditable operating-region decisions | LoopMom, cluster inference, provenance | Evidence/control layer | Full programme | Conditional resource design rule | Separate experiments, not one combined model | Toolkit level |

### 6.2 Early systems map

Place after the Introduction's resource equation and slogan. It should use parallel lanes, not a
serial architecture diagram:

1. **Representation allocation:** V-BLT and RoPE-ND.
2. **Retained state:** flat, spatially hierarchical, and temporally sparse context packages.
3. **Training tape:** BPTT, checkpointing, and reversible replay.
4. **Evaluation plane spanning all lanes:** consistency criteria, exact observability, endpoints,
   inference, and provenance.

Use solid arrows only for actual implemented dependencies. Use dashed arrows labelled
"conceptual transfer" or "compatible extension". Include a legend stating "separate
implementations and measurements; not one integrated executable predictor."

### 6.3 Chapter handoff pattern

End each substantive chapter with the same five-part receipt:

1. intervention developed;
2. system/resource term affected;
3. evidence used;
4. limitation/boundary;
5. handoff to the next chapter.

This is more effective than repeating which modules are not integrated.

### 6.4 Chapter 2 refactor

- Add an opening map of the representation-allocation and activation-tape questions.
- Make the transition between RoPE-ND and reversibility a resource-decision transition, not simply
  "a separate constraint."
- Separate reproduced A1/A2 material from dissertation-level stochastic recurrence and
  applicability taxonomy.
- Correct all memory accounting and stability mathematics before using the strand in synthesis.

### 6.5 Chapter 3 refactor

- Present only the final joint-attention architecture in the main narrative.
- Move superseded designs to an appendix deviation table.
- State that A3's trajectory criterion is implemented through the testbed/endpoints.
- State that Article 4 closes the context-state and measurement loop.
- Do not attribute the package effects to RoPE-ND or reversible training.

### 6.6 Chapter 4.8 and General Conclusions

Rename Chapter 4.8 to something like:

> **Integrated operating-region synthesis across representation, retained state, and activation
> memory**

Expand the table to include entropy allocation, RoPE-ND, consistency/LoopMom, dense flat context,
spatial hierarchy, temporal subsampling, checkpointing, and reversible replay. Add columns for the
question answered, resource/uncertainty term, measured payoff, boundary, evidence source, and
final-predictor status.

Lead the General Conclusions with:

1. information must be allocated across representation fidelity and temporal reach;
2. retained-state memory and activation-tape memory are distinct resource axes;
3. the correct intervention depends on the operating region.

Then retain a shorter seven-task crosswalk.

---

## 7. Chapter 4 and CW-VAE-style comparison completeness

### 7.1 Direct answer

The assembled dissertation does **not** currently contain the comprehensive CW-VAE-style
all-configuration table.

- Table 3.1 contains only the initial/core configuration subset.
- Chapter 4 contains strong claim-specific tables.
- The broad arm grid exists in `docs/CWSTYLE_TABLE.md` and the campaign archive.
- The assembled dissertation promises supplementary appendices but ends after the references; the
  promised configuration/evidence appendices are not actually bound.

This is a substantial transparency and assembly gap, not a missing-experiment gap.

Table 3.1 must also stop calling itself the complete design. It omits the later frontier pairs,
B1, temporal package, alternate regimes, and depth probe.

### 7.2 Why one giant causal factorial would be wrong

The programme contains different comparison licenses:

- **MATCHED** -- exact matched-budget/package pairs;
- **NEAR_MATCHED** -- the 1,088-versus-1,152 temporal/spatial package comparison;
- **WHOLE_CONFIGURATION** -- B1, where target interface and other factors co-change;
- **DEVELOPMENT_FACTORIAL** -- the five-seed head-by-package study;
- **DEVELOPMENT_DIAGNOSTIC** -- depth and stream-blinding probes;
- **ROBUSTNESS_ONLY** -- sampler/NFE and codec interventions;
- **PRIVILEGED_REFERENCE** -- pose-indexed raw mosaic;
- **EXTERNAL_BOUNDARY** -- E6;
- **NOT_EXECUTED / OUT_OF_SCOPE** -- structurally absent cells.

Rows with different licenses must not be compared as if they were one full Cartesian experiment.

### 7.3 Required Chapter 4 table architecture

#### Table 4.1 -- experimental study map

| Study | Scientific question | Testbed/population | Intervention | Held fixed | Co-changed/confounded | Heads | Seeds | Evidence level | Inferential family | Verdict |
|---|---|---|---|---|---|---|---:|---|---|---|

This gives the reader the design before the results.

#### Table 4.2 -- complete arm and interface registry

Include at minimum `flat-short`, `hier-s2-min`, `flat-min`, `hier-min`, `strided-flat`,
`flat-budget`, `hier-long`, `flat-c1`, `hier-c1`, `flat-c2`, `hier-c2`, `flat-long`, and
`hier-rich`.

| Arm | Package family | Lag/stride geometry | Fine tail | Reach | Context tokens | Target tokens | Total tokens | Head(s) | Training regime | Seeds | Comparison role |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|

Put `hier-short`, `spat18`, `hier3`, and stream-blinding variants in a clearly secondary diagnostic
block.

#### Table 4.3 -- mechanism-selection contract

| Mechanism | Objective/prediction object | Additional input | Inference | Single-step result | Best-of-1 rollout result | Final role |
|---|---|---|---|---|---|---|
| Regression | Direct latent regression | Context | One pass | Weaker best-of-8 coverage | Stronger deterministic rollout | Deterministic control |
| Diffusion | Velocity target | Noisy candidate and noise level | DDIM | Stronger stochastic coverage | Weaker deterministic rollout | Primary generative frontier head |
| EBM | Scalar energy | Context and candidate | Candidate descent | Ordering/descent failure modes | No rollout grid after gate failure | Characterized negative |

This resolves the apparent tension between choosing diffusion for coverage-oriented generation and
regression winning deterministic rollout.

#### Table 4.4 -- strengthened tested frontier

| Budget | Flat arm/reach | Hierarchical arm/reach | Seed/regime | Effect | Simultaneous interval | Decision |
|---:|---|---|---|---:|---|---|

The current Table 4.2 is honest but forces the reader to reconstruct arm identities and intervals
from other sections.

#### Table 4.5 -- complete package-form factorial

Restore the article's full five-seed table to the dissertation:

| Seed | Diffusion temporal effect | Regression temporal effect | Head x package interaction |
|---:|---:|---:|---:|

The current dissertation prints diffusion effects while relegating regression and interaction to
prose.

#### Table 4.6 -- context-form evidence map

| Question | Tested comparison | What changes | What is isolated | What remains confounded | Evidence level | Conclusion |
|---|---|---|---|---|---|---|

Required rows: reach, fixed-span spatial abstraction, temporal-versus-spatial package, B1 stride-2
family, and the qualified depth probe.

### 7.4 Curated appendix, not a campaign diary

Bind a real supplementary appendix containing:

1. the full observed arm grid;
2. per-seed raw cells for the head-by-package factorial;
3. all sampler/NFE effects;
4. codec intervention details;
5. depth and stream-blinding diagnostics with confound warnings;
6. parameters, examples, updates, EMA, NFE, runtime, and VRAM where measured;
7. a study-to-artifact provenance map.

Generate it from frozen artifacts and the canonical arm registry. Do not paste
`docs/CWSTYLE_TABLE.md` unchanged: it mixes evidence levels/regimes and ends with "Pending" cells.
Replace those with **not executed / outside the frozen programme**.

### 7.5 Historical assets that must not be resurrected unchanged

- `fig4_claim3_factorial.png` contains a wrong "coarse WINS" label in a panel whose values show a
  coarse-resolution loss under diffusion.
- `fig13_family_edges.png` contains stale seed-queue language and an overstrong "edge law" title.
- `fig12_cwstyle_curves.png` is crowded and descriptive, lacks uncertainty, and belongs at most in
  a regenerated appendix.
- The old `fig1`--`fig14` family includes development-era or retired interpretations. Use current
  frozen claim artifacts, not those visuals, as the regeneration source.

---

## 8. Mandatory scientific-content corrections

### 8.1 P0 -- must be fixed before any "final" claim

| ID | Location in assembled dissertation | Problem | Required correction |
|---|---|---|---|
| D-P0-1 | 3001--3007 | Rollout buffer values are called distributionally identical to clean training contexts; clean-context training is called sufficient | Say only layout and temporal indexing are identical; values are self-generated and exposure bias remains |
| D-P0-2 | Eq. 2.17, 2268--2285 | Uses an invalid approximate equality/lower-bound argument and concludes additive residual maps are inherently non-contractive | Use the valid upper bound `Lip(I+F) <= 1 + Lip(F)`; separate mathematics from empirical perturbation growth; do not infer inherent non-contractivity |
| D-P0-3 | Lemma/table/conclusions around 2088--2247, 2320--2324, 2420--2429 | "Constant memory" conflicts with stored noise growing as `O(T D_epsilon)` | Define constant activation-tape/local-graph memory; include replay metadata/noise or document bounded RNG regeneration; distinguish algebraic inverse from measured floating-point error |
| D-P0-4 | 2245--2247 and bridge passages | Claims stochastic reversibility is required by the Chapter 3 predictor | Replace with a compatible/future applicability relationship; the final predictor does not use it |
| D-P0-5 | 1328--1365 | CLIP similarity is treated as causal understanding; hierarchical consistency is treated as guaranteed | CLIP is a semantic-alignment proxy; joint/coarse conditioning biases coherence but cannot guarantee it |
| D-P0-6 | Front matter and Introduction | Nine `[AUTHOR: ...]` markers and promised appendices remain | Resolve for submission or keep a clearly labelled review edition; bind the actual appendices and live ToC |
| A-P0-1 | Article Figure 5 / `fig18_trust_negative_replication` | Verdict annotation obscures a data interval | Move verdict outside data; retain marker/line-style redundancy |
| V-P0-1 | Dissertation Figure 2.10 / `fig49_reverse_local_replay` | Figure says memory does not scale with trajectory length | Say activation-tape/local graph does not scale in the tested replay; account for stored replay signal |
| V-P0-2 | Dissertation Figure 2.8 / `fig55_tempverseformer_architecture_published` | Too tall/dense/handwritten to be legible at page width | Split into a clean overview and reversible-core detail; preserve original in appendix if desired |
| V-P0-3 | Caption after Figure 1.10 | Editorial residue `Source: NEW (draw)` remains | Replace with a final source/caption statement |
| V-P0-4 | Reused published Figures 2.3--2.8 | Reproduction rights not yet recorded | Verify publisher licence/permission and cite the reused version accurately |

### 8.2 P1 -- high-priority pre-submission work

- Reframe hierarchy as an inductive bias/resource-allocation strategy, not a logical necessity.
- Qualify entropy as an information/uncertainty proxy requiring calibration.
- Correct absolute topology/equivariance and relative-position memory-complexity language.
- State checkpointing schedule assumptions; remove universal constants and "zero memory cost."
- Replace universal "reversibility cannot pay" with tested-regime and crossover-condition wording.
- Remove Chapter 3 rewrite-history narration.
- Explain mechanism-specific objective/input/inference differences and diffusion selection.
- Change "aggregate metrics are excluded" to "not accepted as sole/headline evidence."
- Remove raw repository paths, script names, and function names from main captions; keep them in a
  provenance appendix.
- Verify every "first," "unique," "guarantees," "proves," "must," and "cannot" claim.
- Fix table numbering, including the out-of-order Table 3.4-A / Table 3.2 sequence.
- Number and caption material Chapter 4 tables that are currently unnumbered.

### 8.3 P2 -- compression and polish

- Shorten generic textbook material in Chapter 1 that does not feed a later definition or decision.
- Reduce repeated frontier receipts outside the canonical Chapter 4 table and final conclusions.
- Standardize `exact` into algebraically invertible, numerically reconstructed within tolerance,
  and statistically confirmed.
- Standardize memory into parameters, optimizer state, context/state, activation tape, replay
  metadata/noise, and transient derivative graph.
- Add a private claim-to-primary-source audit sheet.

---

## 9. Fourth-article finalization plan

The article should remain narrow. Do not add V-BLT or TempVerseFormer material merely to make the
dissertation appear connected.

1. Keep the current title, RQs, contribution framing, untouched-first results order, and
   conditional conclusion.
2. Preserve the exact phrases **nominal addressable reach**, **supports reach**, **unresolved**, and
   **not equivalence** during shortening.
3. Keep a compact main-text interface table and add a supplementary full cost table with
   parameters, training examples/updates, EMA, sampler/NFE, runtime, VRAM, seed coverage, and
   evidence role.
4. Add a compact publication-facing study map: study, population, intervention, protected status,
   seeds, endpoints, inferential family, and verdict.
5. Keep the frontier, failed replication, and E6 boundary in the main paper.
6. Consider moving detailed sampler, codec, and privileged-access receipts to the supplement when
   venue length is known. The raw-access result may remain main-text if addressing versus storage
   is central to the selected venue story.
7. Add data/code/model availability, licences, expected archive sizes, and a concise
   human-subjects/ethics statement.
8. Verify every yes/no cell in the comparator table against the original papers.
9. Add a selection-rule-controlled qualitative montage only if the venue expects qualitative
   evidence. Use fixed clip IDs, median effects, or preregistered quantiles; do not choose an
   aesthetically dramatic anecdote.
10. Correct Figures 3--5, test every figure at single-column size and greyscale, then perform a
    venue-specific render review.

---

## 10. Visual finalization plan

### 10.1 Audit summary

| Class | Count | Assets |
|---|---:|---|
| Keep | 8 | `fig15`, `fig16`, `fig19`, `fig20`, `fig22_phasec_v2_boundary`, `fig46`, `fig56`, `fig22_token_sequence_layout` |
| Minor polish | 9 | `article_confirmed_frontier`, `fig17`, `fig21`, `fig34`, `fig35`, `fig41`, `fig57`, `fig58`, `fig51` |
| Redraw or structurally consolidate | 20 | `fig18`, `fig30`, `fig31`, `fig32`, `fig33`, `fig36`, `fig37`, `fig38`, `fig39`, `fig40`, `fig42`, `fig43`, `fig44`, `fig45`, `fig47`, `fig48`, `fig49`, `fig50`, `fig52`, `fig23` |
| Re-export/redraw; retain original only where useful | 5 | `fig59`, `fig60`, `fig53`, `fig54`, `fig55` |

All current canvases are already opaque white. The main remaining issues are routing, semantic
accuracy, legibility, annotation collisions, style consistency, and figure selection.

### 10.2 Highest-priority figure actions

1. Fix `fig18` data occlusion.
2. Fix `fig49` memory scope and reverse-time arrow direction.
3. Split/redraw `fig55`.
4. Move `fig17` across-seed annotations away from seed-4 estimates.
5. Move the frontier's blue memory annotation off the red bar and simplify its internal legend.
6. Redraw `fig30` as a clean closed feedback loop.
7. Redraw `fig32` to remove the unexplained representation baseline and show refinement as a loop.
8. Replace `fig31`'s empirical-looking 3-D surface with a clean conceptual 2-D view.
9. Clarify the disconnected macro level and arrow directions in `fig33`.
10. Combine `fig36`--`fig38` into one standard forward/reverse latent-diffusion figure; use standard
    `x_0 -> x_T` and `x_T -> x_0` notation.
11. Redraw `fig39` without arbitrary empirical-looking curves.
12. Redraw `fig40` with exact token/reach quantities.
13. Correct `fig42` to show the ControlNet branch injecting residuals into the frozen backbone.
14. Correct `fig43` to show reference/target branches entering shared attention in parallel.
15. Remove numeric empirical appearance from the illustrative curves in `fig44`.
16. Redraw `fig47` as actual independent rotary subspaces, not only a coordinate scatter.
17. Redraw `fig48`--`fig50` with filled arrowheads and clearly separated forward/reverse lanes.
18. Scope `fig52` to tested operating regions rather than universal "pays" claims.
19. Fix `fig23` so the return path visibly updates both coarse and fine buffers.
20. Re-export published A1/A2 empirical plots with readable labels, scales, and reuse notices.

### 10.3 Consolidate the figure programme

The Introduction currently contains many one-concept pedagogical figures. Consolidate where this
strengthens the argument:

- merge Figures 1.1 and 1.3;
- merge Figures 1.2 and 1.16;
- merge Figures 1.6 and 2.1 as fixed versus adaptive allocation;
- merge Figures 1.7--1.9 into one diffusion figure;
- combine Figures 1.13--1.15 into one external-conditioning comparison.

This creates room for the missing dissertation-level systems map and two-resource synthesis.

### 10.4 New required visuals

1. **Dissertation systems map** with actual versus conceptual links.
2. **Two-panel resource synthesis:** context-token/reach frontier beside activation VRAM/horizon.
   Do not normalize or merge the axes; the interventions were measured separately.
3. **Executed-configuration matrix** showing package, head, budget, codec, sampler, dataset, seed
   coverage, evidence tier, and unexecuted cells.
4. **Optional mechanism-selection panel** after repairing `fig24`--`fig28` sources.

Do not insert `fig25_ebt_generation_loop` unchanged: its feedback arrow returns the update to the
context rather than the candidate state. Do not insert `fig28_memory_gate_optimization` until its
annotation is moved away from the interval.

### 10.5 Unified design system

- Canvas: opaque `#FFFFFF`.
- Text/outlines: `#1F2933`.
- Hierarchy/long-reach/reversible: `#0072B2`.
- Flat/dense/checkpoint comparator: `#E69F00`.
- Privileged/reference: `#009E73`.
- Failure/negative: `#D55E00`.
- Unresolved/neutral: `#6B7280`.
- Additional mechanism: `#CC79A7`.
- Every color distinction must also use a marker, fill, label, or line-style distinction.
- Use one sans-serif family and a compatible math face.
- Minimum final-size text: approximately 8.5--9 pt.
- Put sentence-length claims in captions, not artwork titles.
- Use bold A/B/C panel labels consistently.
- Use orthogonal or gently curved routes, filled arrowheads, and explicit return arrows.
- Use dashed edges only for conceptual, untested, or optional connections and label them.
- Never place annotations over points or intervals.
- Distinguish measured and extrapolated regions by line style.
- Export SVG/PDF with embedded fonts; retain 300--600 dpi PNG as a compatibility fallback.
- Test at actual page width, greyscale, and common color-vision simulations.

---

## 11. Literature and citation plan

The current counts--46 article references and 123 dissertation references--are not a problem by
themselves. A mechanical target of 200 references is not recommended. The bibliography may grow
substantially if a claim-level audit shows real gaps, but every addition must support a specific
sentence.

Create a private checklist with:

| Manuscript location | Claim type | Primary source | Exact support | Final calibrated wording | Verified date |
|---|---|---|---|---|---|

Prioritize:

1. uncertainty decomposition and calibration;
2. medical-image hallucination risk, only if those applied claims remain;
3. remote-sensing/physics-constrained claims, only if retained;
4. video-evaluation validity and FVD limitations;
5. exposure bias at the corrected Chapter 3 rollout passage;
6. checkpoint schedules and gradient equivalence;
7. positional encoding/equivariance and materialized memory;
8. visual/video hallucination rather than NLP-only analogies;
9. object identity, tracking, and occlusion evaluation;
10. stochastic reversible replay and exact stored-noise/RNG accounting;
11. every novelty superlative and every comparator-table yes/no cell, especially 2025--2026 work.

The final source must cite material factual, causal, comparative, and methodological claims. It
does not need a citation after self-contained definitions or every sentence.

---

## 12. Ordered execution plan

### Phase 0 -- establish the correction ledger

**Outputs:** frozen baseline, issue table, source map, explicit no-new-experiment boundary.  
**Acceptance:** every issue in this document has an owner, target source, priority, and validation
method.

### Phase 1 -- correct scientific blockers

1. Correct rollout-distribution wording.
2. Correct Eq. 2.17 and the associated stability argument.
3. Reconcile activation-tape memory with replay-noise/metadata accounting.
4. Correct CLIP, hierarchical consistency, entropy, topology, and hierarchy-necessity claims.
5. Correct bridge passages that imply unused components are required by Chapter 3.
6. Recheck every theorem, lemma, table, figure, abstract, chapter conclusion, and General
   Conclusion that inherits those claims.

**Acceptance:** no contradiction between body, abstract, figures, and conclusions; an independent
mathematical reviewer accepts the revised statements.

### Phase 2 -- install the systems spine

1. Add the four-connection vocabulary.
2. Add the contribution crosswalk.
3. Add the early systems map.
4. Add chapter handoffs.
5. Reframe Chapter 2.
6. remove Chapter 3 rewrite-history narration.
7. Expand Chapter 4.8 and restructure the opening of General Conclusions.

**Acceptance:** a reader can explain in one paragraph how A1--A4 relate, which links are measured,
and which components are not in the final predictor.

### Phase 3 -- rebuild comparison presentation

1. Rename Table 3.1 as the core initial configurations.
2. Add the study map and complete arm registry.
3. Add the mechanism-selection contract.
4. Strengthen the frontier table.
5. Restore the full five-seed factorial.
6. Add the context-form/comparison-license map.
7. Generate and bind a curated appendix.
8. Remove all "pending" language for terminally unexecuted cells.

**Acceptance:** every claim-bearing arm is visible once in a canonical registry; every comparison
has an explicit causal license; no absent cell is implied to exist.

### Phase 4 -- refocus and complete the article package

1. Add the supplementary cost and study maps.
2. Decide main versus supplement placement after venue selection.
3. Add availability, licence, ethics, and archive information.
4. Verify comparator cells.
5. Preserve the central wording guardrails.

**Acceptance:** the main text remains one conditional argument; supplementary detail makes the
study independently auditable.

### Phase 5 -- visual finalization

1. Repair P0 figures.
2. Create the systems map and two-resource synthesis.
3. Create the executed-configuration matrix.
4. Redraw scientifically ambiguous schematics.
5. Split/re-export inherited figures and verify reuse rights.
6. Apply the unified visual system.

**Acceptance:** every figure passes page-width, greyscale, colorblind, arrow-routing, data-occlusion,
caption, source, and scientific-scope review.

### Phase 6 -- references, numbering, and editorial closure

1. Run the claim-to-primary-source audit.
2. Normalize the bibliography.
3. Renumber all equations, figures, tables, and cross-references.
4. Remove raw paths/editorial residue from reader-facing prose.
5. Bind actual appendices.
6. Resolve author/institution fields or retain an explicitly labelled review edition.

**Acceptance:** no unresolved scientific/editorial markers; only externally owned facts remain in
the separate author checklist.

### Phase 7 -- fresh red-team and final freeze

1. Reassemble from the authoritative Markdown sources.
2. Run existing automated gates.
3. Perform a fresh independent mathematical, causal, literature, and visual review.
4. Read the article as a skeptical venue reviewer.
5. Read the dissertation as a committee member checking topic--aim--tasks--novelty--results--
   conclusions alignment.
6. Freeze content only after all P0/P1 items close.

**Acceptance:** no reviewer can reasonably infer universal hierarchy, equivalence, a unique
crossing, natural-video validation, a stable trust law, or one integrated executable pipeline.

### Phase 8 -- later institutional/venue typesetting

This phase is outside the current content-only request: select the venue, transfer Markdown to the
required templates, generate live ToC/cross-references, insert signed acts, apply the required
bibliographic standard, and perform page-by-page render QA.

---

## 13. Estimated effort

| Workstream | Focused estimate |
|---|---:|
| Scientific P0 correction and propagation | 1--2 days |
| Systems-spine refactor | 4--6 days |
| Chapter 4 tables and curated appendix | 2--4 days |
| Article focus/supplement/end matter | 1--2 days |
| Figure redesign, consolidation, and export | 8--12 days |
| Claim-level reference and novelty audit | 2--4 days |
| Numbering, assembly, and fresh red-team QA | 2--3 days |

A realistic content-and-visual finalization is approximately **three to five focused weeks**, not
including delays for publisher permissions, author/institution facts, venue formatting, or
supervisor feedback. No GPU time is required.

---

## 14. Final readiness gates

### Article may be called submission-ready only when

- the Figure 5 occlusion is fixed;
- cost/study supplement is complete;
- comparator cells are verified;
- availability/licence/ethics text is complete;
- main-versus-supplement balance fits the venue;
- final render passes single-column and greyscale review.

### Dissertation may be called content-final only when

- all scientific P0 items are corrected everywhere;
- the cross-strand map and systems spine are visible through the middle chapters;
- misleading component-dependency language is removed;
- the canonical configuration registry and comparison-license map are bound;
- actual appendices are included;
- visual P0/P1 items and table numbering are fixed;
- novelty/comparator claims pass primary-source review;
- a fresh independent red-team finds no blocker.

### Dissertation may be called submission-final only when, additionally

- all author and institutional fields are supplied;
- permissions and signed implementation acts are included;
- institutional styles, pagination, live ToC, and bibliography standard are applied;
- the final PDF/DOCX render passes page-by-page visual QA.

---

## 15. Final recommendation

Adopt **Strategy B**, the moderate systems-spine refactor.

Do not run more experiments and do not attempt to turn the dissertation into one executable model.
Correct the content blockers, add one authoritative crosswalk and systems map, restore a curated
all-configuration comparison layer, expand the all-strand synthesis, remove revision-history
narration, and rebuild the visual programme under one design system.

After that work, the defensible dissertation identity is:

> **An integrated systems methodology comprising separately measured representation,
> retained-state, activation-memory, and evaluation contributions--not one executable pipeline.**

That is coherent, scientifically honest, and strong enough to support the dissertation without
inflating the evidence.
