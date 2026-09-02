# Detailed Dissertation Refactoring Plan

**Date:** 2 September 2026

**Status:** revised after independent blind audit; manuscript text has not yet been changed

**Working manuscript:**
`C:\Users\leoni\Documents\projects\hierarchical-video-prediction-path-b\docs\dissertation\condensed_translated_humanized_ukrainian`

**Safe planning branch:** `codex/integrated-openstl-v2`

**Related review:**
`C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\reviews\DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`

## 1. Decision and scope

The dissertation will be **refactored from the current Ukrainian manuscript rather than rewritten
from scratch**. The current four-chapter order remains the organizing structure:

1. system problem, consistency, hierarchy, and memory;
2. published methods for activation memory and spatial representation;
3. the hierarchical retained-context architecture and experimental method;
4. results, operating region, integration, and practical conclusions.

Chapters 1 and 2 retain most of their prose and the useful material imported from Articles 1--3.
Chapters 3 and 4 retain their current semantic roles but receive the largest evidence-level
refactoring. New or materially rewritten passages are drafted in English. Language normalization
of the final full manuscript is a later editorial operation; during this phase an English insertion
must be a complete paragraph or subsection, never a sentence fragment embedded inside Ukrainian
syntax.

The redesigned TRB hierarchy study and the OpenSTL integrated model are two evidence blocks of
**one new article**. Historical experiment prefixes `A4_*` and `A5_*` remain stable provenance
identifiers only. The dissertation must not describe them as two promised publications.

This plan does not authorize invented measurements. Where an experiment is unfinished, the
manuscript receives a traceable placeholder, a generation contract, and conditional interpretation
language. Optimistic expected behavior may guide layout, but no provisional number is printed as
an empirical result.

The new TRB matrix is a **long-context/distant-evidence allocation study with a four-frame
prediction endpoint**. It is not, by itself, a test of long autoregressive forecast stability.
“Long retained reach” and “long forecast horizon” are different axes throughout the manuscript.
Legacy E5/E6 and other valid self-fed rollout evidence remain substantive because they test the
second axis. A new rollout of selected v2 checkpoints, if authorized, is a separate post-campaign
evaluation rather than a reinterpretation or relaunch of the 86 training runs.

Evidence status uses five exact labels:

- **registered development estimate** for the complete 30k matrix;
- **direction reproduced across two training seeds** only when both seed contrasts agree;
- **sealed-holdout confirmation** only for the exact selected roles and registered populations
  opened on the fresh holdout;
- **mechanism diagnostic** for A4-MEC and similar probes; and
- **historical/transfer boundary** for legacy E2--E8 evidence.

The complete development frontier must never be called a confirmed frontier unless the complete
family is evaluated under a separately authorized confirmation contract.

## 2. What the author's concerns mean in editorial and scientific terms

The concerns are coherent and point to one central problem: the dissertation contains the right
scientific ingredients, but the visual and mathematical hierarchy does not yet match the hierarchy
of the claims.

### 2.1 The intended scientific thesis

Hierarchy has several implementations and operates on different resources:

- **spatial hierarchy** replaces uniform high-resolution coverage with coarse global coverage plus
  selected detail;
- **temporal hierarchy** spends more resolution on recent observations and cheaper tokens on older
  observations, thereby extending retained reach;
- **computational hierarchy/reversibility** reconstructs internal training activations instead of
  storing the complete differentiation tape; and
- **dynamic space--time routing** decides where a fixed fine-detail quota is most useful without
  treating packed tensor position as physical identity.

Memory is the common scarce resource, but it is not a single scalar. The dissertation must keep at
least four terms separate:

1. retained state/context;
2. attention and other intermediate activations;
3. parameters and optimizer state; and
4. temporary workspace/allocator overhead.

The central hierarchy claim is consequently conditional: hierarchy may buy access to useful older
evidence under a constrained retained-state budget, while a dense flat representation may regain
an advantage when it already reaches the relevant event and fine identity becomes limiting.
Reversibility does not improve this reach, and routing at an unchanged quota is not an additional
memory multiplier.

### 2.2 The current presentation problem

The present main text has approximately 114 numbered formulas and 77 figures. This gives standard
definitions, original mechanisms, diagnostic formulas, tutorial illustrations, and primary results
nearly equal visual importance. The reader must reconstruct the argument from many individually
reasonable objects.

The most visible issues are:

- Chapter 1 contains a ten-figure static-control/diffusion gallery inside one subsection even though
  static image control is contextual evidence rather than the dissertation's main intervention;
- Chapter 2 numbers long standard derivations and optional replay extensions at the same level as
  the implemented reversible and V-BLT interfaces;
- Chapter 3 assigns 53 numbers to architecture, diffusion, DDIM, metrics, inference, and the now
  superseded integration prototype;
- several Chapter 4 tables mix raw results, confidence intervals, configuration arithmetic, and
  interpretation in the same object;
- several figures answer more than one question, use panels with different estimands, or repeat a
  table without providing a clearer relationship;
- the old E2/E4/E5/E6/E7/E8 narrative competes with, instead of supporting, the new 86-run hierarchy
  study; and
- the fixed-region/second-order integrated prototype occupies the architectural position of the
  new physically positioned, genuinely packed, internally reversible model.

### 2.3 The reporting principle

Every main-text evidence object must pass a one-sentence test:

> What single question can the reader answer from this figure or table?

If the answer contains “and,” split the object unless the panels are inseparable parts of one causal
comparison. Every result subsection follows the same order:

1. question and estimand;
2. controlled contrast and invariant conditions;
3. one primary figure;
4. exact values in a result table only when they are needed;
5. direct interpretation;
6. boundary or alternative explanation.

This follows established figure guidance to identify the message and design for the intended
audience, and the recommendation to show individual observations for small samples rather than
conceal them behind bars: [Rougier et al., *Ten Simple Rules for Better Figures*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4161295/),
[Nature Biomedical Engineering, *Show the dots in plots*](https://www.nature.com/articles/s41551-017-0079),
and [Weissgerber et al., *Beyond Bar and Line Graphs*](https://journals.plos.org/plosbiology/article/file?id=10.1371%2Fjournal.pbio.1002128&type=printable).
The experiment text must state what every error bar represents and what sources of variation were
measured, consistent with the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)
and the warning that error-bar meaning is often misread
([Krzywinski and Altman, *Error bars*](https://www.nature.com/articles/nmeth.2659)).

## 3. Current state and target state

| Part | Current role | Current load | Target treatment | Target main-text load |
|---|---|---:|---|---:|
| Introduction | motivation, aim, novelty, contribution | about 3,136 words | update aim/tasks/publication map; keep structure | similar length |
| Chapter 1 | system analysis, consistency, bottlenecks, literature | 14 equations, 25 figures, 2 tables | preserve prose; foreground four memory resources and spatial/temporal hierarchy; remove tutorial gallery | 6--8 equations, 9--11 figures, 0 tables |
| Chapter 2 | TempVerseFormer and V-BLT plus controlled datasets | 42 equations, 19 figures, 10 tables | preserve published evidence; compress derivations; clarify which ideas are diagnostics versus implemented mechanisms | 16--20 equations, 13--15 figures, 5--6 result tables |
| Chapter 3 | retained-context architecture, TRB, hypotheses, old integration | 53 equations, 19 figures, 3 tables | keep role; add exact new campaign design; replace obsolete integration architecture | 20--24 equations, 12--14 figures, 0--1 result tables |
| Chapter 4 | hierarchy results, robustness, integration, decision rule | 5 equations, 14 figures, 7 tables | make new hierarchy matrix primary; retain selected historical boundary evidence; add integrated OpenSTL results | 4--8 equations, 14--18 figures, 6--8 result tables |
| Conclusions | consolidated findings | about 1,395 words | rewrite only after results lock | 1,500--2,000 words |
| Appendices | provenance/configuration/derivations | 18 tables, 4 figures | expand as the authoritative technical registry | no artificial limit |

The target is a range rather than a quota. A necessary original equation or single-purpose figure
must not be deleted merely to hit a number. Conversely, numbering must not be retained only to
preserve the old count; final numbering occurs after the evidence structure is frozen.

### 3.1 Current structure retained as the comparison baseline

The audited manuscript currently has the following semantic structure. This is the baseline against
which Section 6 should be compared; the plan does not silently replace it with the alternative
chapter order proposed in the 31 August review.

- **Introduction:** relevance; research-program context; aim and tasks; object and subject;
  methods; novelty; practical value; validity; author contribution; publications; structure.
- **Chapter 1 — system analysis and requirements:**
  - 1.1 video prediction as a partially observable controlled process: system boundaries/state,
    uncertainty under autoregression, and resource-constrained multiobjective prediction;
  - 1.2 consistency: inadequacy of one metric, appearance/object/concept consistency,
    temporal/trajectory consistency, hierarchical/cross-level consistency, and the static-control
    bridge to video;
  - 1.3 bottlenecks: activation memory, uniform spatial tokenization, retained-context reach, and
    the system map;
  - 1.4 literature: temporal/long-horizon methods, adaptive/hierarchical representations, and
    long-context/multirate memory;
  - 1.5 problem decomposition, requirements, hypotheses, and conclusions.
- **Chapter 2 — methods for scaling constraints:**
  - 2.1 controlled procedural basis: generation, 2-D rotations, 3-D polycubes, and dataset roles;
  - 2.2 reversible temporal computation: activation cost, block, chaining, reverse pass, controlled
    and OpenSTL results, memory--compute boundary, stochastic replay, and applicability;
  - 2.3 V-BLT: fixed-patch problem, architecture, causal byte objective, RoPE-ND, entropy,
    experiments, and measured/proposed boundary;
  - 2.4 implications and conclusions.
- **Chapter 3 — hierarchical retained-context model:**
  - 3.1 retained context, package geometry, and reach--fidelity hypothesis;
  - 3.2 hierarchical latent state, streams, embeddings, attention, and gradient boundaries;
  - 3.3 diffusion/Transformer/DDIM prediction with a regression control;
  - 3.4 rollout, ring buffers, anchors, causality, and token cost;
  - 3.5 TRB/LoopMom, observability, frame/object metrics, and statistics;
  - 3.6 hypotheses;
  - 3.7 fixed-region, packed-SimVP, second-order reversible integration prototype;
  - 3.8 conclusions.
- **Chapter 4 — experimental validation and operating region:**
  - 4.1 experiment map and comparison rules;
  - 4.2 regression/diffusion/EBT mechanism choice;
  - 4.3 fixed-budget reach evidence;
  - 4.4 reach--fidelity boundary and token arithmetic;
  - 4.5 package-shape/model interaction;
  - 4.6 sampler and codec robustness;
  - 4.7 negative and boundary evidence;
  - 4.8 fixed-region integration feasibility;
  - 4.9 system-level selection rule;
  - 4.10 software/practical application;
  - 4.11 validity and conclusions.
- **General conclusions, references, and appendices:** synthesis, bibliography, detailed evidence
  registries, formulas/configuration support, and software provenance.

The proposed structure preserves every top-level role above. It changes emphasis inside the roles:
new controlled allocation evidence replaces the old frontier as the headline; the new integrated
architecture replaces the old feasibility prototype; and secondary/tutorial material moves to the
appendices.

## 4. Global rules for formulas, figures, tables, and names

### 4.1 Formula rule

A formula is numbered in the main text only if at least one condition holds:

1. it defines an original model object or intervention used later;
2. it is implemented and needed to reproduce the method;
3. it defines the primary estimand, budget, or decision rule used in results; or
4. later reasoning refers to it more than once.

Standard Transformer, diffusion, DDIM, elementary RoPE, MSE, PSNR, and textbook bootstrap
derivations are cited, written inline, grouped into one aligned system, or moved to an appendix.
Conceptual decompositions that are not executable models are written as prose unless the symbols
are used downstream. “One displayed expression” and “one numbered scientific object” are not the
same thing: closely linked forward/inverse relations should share one number with sublabels.

Formula captions do not exist, so every retained formula receives an immediate sentence stating:

- what the left-hand side measures;
- which quantities are fixed or varied;
- whether the relation is exact, estimated, or conceptual; and
- where it is implemented or evaluated.

### 4.2 Figure rule

Main figures use a consistent visual grammar:

- Flat is a neutral dark gray, temporal/coarse hierarchy is blue, dynamic routed hierarchy is
  orange, and the full integrated model is a distinct dark purple;
- no rainbow scale and no red--green-only contrast; use perceptually ordered and color-vision-safe
  maps as recommended by [Crameri et al.](https://www.nature.com/articles/s41467-020-19160-7);
- the same method keeps the same color, marker, and line style across the dissertation;
- axes state metric direction (“lower is better”) and physical units;
- token budgets use the exact values `320, 768, 1152, 1792, 2432, 3072`, not evenly spaced category
  labels when their numeric distance matters;
- both seed values are visible for every two-seed result; the mean may be connected or emphasized,
  but no Gaussian-looking violin, SEM bar, or 95% training-seed interval is drawn from `n=2`;
- clip-level intervals, where valid, are visually and verbally distinguished from training-seed
  variation;
- qualitative predictions are fixed-index or rule-selected and appear after aggregate evidence;
- panel letters are used only when panels answer one shared question;
- a caption states question, population, metric, direction, uncertainty unit, and limitation, but
  does not repeat an entire discussion paragraph.

The primary quantitative view is an estimation plot or response curve, not a bar chart. Effect
sizes and their uncertainty are preferable to significance-only decoration
([Ho et al., estimation graphics](https://pubmed.ncbi.nlm.nih.gov/31217592/)).

### 4.3 Table rule

The author's preference is adopted as the default: **main-text numbered tables are for empirical
results**. Method configuration, symbol dictionaries, dataset roles, experiment registries, tensor
shapes, hashes, and arithmetic checks normally move to appendices or concise prose/figures. One
compact, single-purpose method table is allowed when exact budget/reach/level/overlap differences
would otherwise require an overloaded figure or repeated appendix lookup. It must be labelled as a
method contract and must not mix configuration with outcomes.

Each result table has:

- one population and one comparison family;
- metric names with units and direction;
- seed values or `mean (seed 0, seed 1)` when only two seeds exist;
- a separate column stating the uncertainty unit if intervals are included;
- boldface only for the declared selection criterion, not every row-wise maximum;
- no mixture of empirical values and hypothetical projected ratios;
- no ambiguous “±” notation; use explicit SD across training seeds or interval limits and name the
  unit;
- no merged cell that combines two unrelated categories; and
- a machine-readable source path in its provenance record.

### 4.4 Experiment names in dissertation prose

Long internal identifiers are necessary in manifests but not in continuous prose. At first use,
write a human label followed by the stable family code, for example:

> the six-point context-budget frontier (`A4-FRT`)

Thereafter use “budget frontier,” “allocation study,” “coarse-expansion study,” “level-depth study,”
“overlap study,” or “space--time allocation study.” Full identifiers such as
`A4_FRT_B1792_L2_LINEAR_AGE_NEST_S01_T30K` belong in the appendix registry, artifact metadata, and
figure data files. Legends use short labels: Flat, Linear-age, Geometric-age, Fine-heavy,
Coarse-heavy, and so on.

The dataset name in public prose becomes **Toroidal Re-entry Benchmark (TRB)**. `LoopMom` remains a
legacy code/data identifier only.

## 5. Revised narrative spine

The dissertation should be readable as five linked statements:

1. Long-horizon video prediction under self-feedback is a partially observed memory problem, and
   consistency is task-specific rather than one scalar perceptual score.
2. Memory pressure appears at different interfaces: retained context, training activations, and
   spatial allocation. Hierarchy has a different implementation at each interface.
3. TempVerseFormer and V-BLT establish two supporting mechanisms: reversible activation replay and
   structured spatial representation/complexity diagnosis.
4. The retained-context study determines how budget, temporal reach, resolution, number of levels,
   overlap, and space--time allocation change prediction behavior under controlled conditions.
5. The integrated OpenSTL model tests composition: genuine packed hierarchy, physical positional
   metadata, dynamic routing, and an identical-forward reversible core must jointly occupy a useful
   quality--memory Pareto point on Moving MNIST and TaxiBJ.

The word “superiority” is used only for a specified comparator, endpoint, dataset, and tolerance.
The conclusion is an operating region and a design rule, not a claim that hierarchy universally
beats flat prediction.

## 6. Detailed section-by-section refactoring

### 6.0 Front matter

Treat `00_FRONT_MATTER.md` as a first-class result-bearing manuscript component. Audit and update
both Ukrainian and English abstracts, keywords, novelty statements, contribution bullets,
publication list, software claims, manuscript counts, and every occurrence of the old
fixed-region/second-order prototype numbers.

Each abstract or novelty sentence that depends on unfinished evidence receives a claim placeholder
linked to the same claim-to-artifact ledger used by Chapter 4. Abstract and conclusion claims are
filled from that ledger after result lock rather than edited independently. Preserve institutional
wording and author-contribution statements unless the author or institution supplies an approved
replacement.

**Visuals/equations/tables:** none.

**Gate:** front matter may be finalized only after Chapter 4, general conclusions, publication
status, and final figure/formula/table counts are frozen.

### 6.1 Introduction

Preserve the current headings and most text. Make the following surgical edits:

- In the relevance section, introduce the four memory terms before naming individual models.
- State explicitly that spatial compression can free state budget for temporal reach, but this is
  an allocation opportunity rather than a guaranteed quality improvement.
- Preserve dissertation-level tasks for: system analysis and task-specific consistency;
  reversible temporal computation; V-BLT/RoPE-ND and spatial-complexity diagnosis; hierarchical
  retained-context design; observability/evaluation/statistics; and the empirical operating region
  plus integrated composition. Nest the FRT/ALC/CEX/LVL/OVR/STS and integration questions under
  the last three tasks instead of replacing the contribution logic of Articles 1--3.
- Update the novelty list so that old feasibility results are not presented as the final integrated
  contribution.
- Map Articles 1--3 to their actual supporting chapters and call the new combined work “manuscript
  in preparation” until its publication status changes.
- Add one paragraph distinguishing the controlled TRB claim from OpenSTL external validation.
- Keep institutional contribution, implementation, and publication subsections; update counts and
  names only after final numbering.

**Visuals:** none added. If the current introductory overview duplicates Chapter 1's system map,
retain only one instance.

**Equations:** none added.

**Tables:** none.

### 6.2 Chapter 1 — keep the structure, sharpen the hierarchy/memory framing

#### 1.1 Video prediction as a partially observable controlled process

Keep the current POMDP, uncertainty, self-feedback, and resource-constrained objective. Merge the
state/observation and conditional-future definitions into one compact notation block. Make clear
that action-conditioned TRB is an instrument, not the definition of video prediction in general.

**Keep:** system boundary, self-fed prediction, multiobjective constraint.

**Add:** one sentence connecting occlusion/re-entry to retained evidence.

**Remove/move:** tutorial expansion of standard conditional prediction notation.

**Target:** 2--3 numbered equations; 2 figures; no tables.

#### 1.2 Consistency as a task-dependent set of constraints

Keep appearance, object, trajectory, temporal, and cross-level consistency, but stop illustrating
every static-control mechanism separately. Present one taxonomy, one failure montage, and one
object/trajectory example. Preserve Article 3/[A3]'s conceptual synthesis of appearance/style,
identity, geometry/shape, and concept consistency, together with the comparison of input-, weight-,
structure-, inversion-, and attention-based control interfaces. Move tutorial redundancy and
secondary illustrations, not the scientific synthesis, to an appendix.

**Add:** a paragraph explaining that hierarchy must be evaluated at the level whose information it
is expected to preserve: coarse coverage/trajectory and fine identity/texture need different
endpoints.

**Target:** 1--2 numbered equations; 3 figures; no tables.

#### 1.3 Resource bottlenecks and manifestations of hierarchy

Retain the current activation-memory, fixed spatial tokenization, retained-context reach, and system
map subsections. Insert a new English subsection titled:

> **1.3.4 Spatial, temporal, and computational hierarchy as distinct interventions**

It should define:

- spatial hierarchy as coarse full coverage plus selected detail;
- temporal hierarchy as resolution/slot allocation across age;
- temporal overlap as nested versus exclusive ownership of a time point;
- computational hierarchy/reversibility as an activation-storage intervention; and
- dynamic routing as assignment of a fixed detail quota over space--time.

The existing systems map is then revised to show which resource each intervention changes. Add a
warning that hierarchy's analytic token ratio is not a complete GPU-memory ratio.

**Target:** memory decomposition, fixed-grid token count, and reach definition as 3 numbered
equations; 4 figures; no tables.

#### 1.4 Literature and unresolved gap

Keep the existing long-horizon, multiscale, adaptive representation, and long-context review. Add
a short explicit comparison of spatial token pruning/routing, temporal subsampling, multiscale
memory, checkpointing, and reversible blocks by **intervention point**. Do not add another survey
table; use one literature-positioning diagram or prose.

Delete or move the extended diffusion/ControlNet/StyleAligned/inversion tutorial sequence. It can
remain in a provenance-preserving appendix if required to demonstrate Article 3/[A3] continuity, but it
must not dominate the dissertation problem statement.

**Target:** no numbered equations; 1 figure; no tables.

#### 1.5 Requirements, hypotheses, and chapter conclusions

Keep the requirements structure and replace the hypotheses with aligned, falsifiable statements:

- H1: at low retained-token budgets, a hierarchy can reduce horizon-4 foreground error when useful
  distant observed context would otherwise be excluded;
- H2: the optimal cross-level allocation depends on total budget and cannot be inferred from one
  hand-selected package;
- H3: more levels and nested overlap help only if their extra representation justifies displaced
  detail;
- H4: space--time placement determines whether fine tokens preserve recent motion, distributed
  anchors, or high-information regions;
- H5: internal reversibility reduces training activation memory for the same forward function;
- H6: dynamic routing at fixed quota improves allocation quality, and creates a memory advantage
  only if a smaller quota preserves quality;
- H7: the full integrated model is useful only if it is competitive on matched OpenSTL protocols
  and improves the measured quality--memory frontier.

Add a separate rollout hypothesis only where supported: selected models that use retained context
may or may not remain stable when their own predictions update that context across many windows.
This is evaluated by legacy E5/E6 or an explicitly registered selected-checkpoint rollout; it is not
inferred from H1.

End with a claim-to-evidence map in prose or one simple diagram. No result is stated here.

### 6.3 Chapter 2 — preserve Articles 1--3, reduce derivational weight

#### 2.1 Controlled experimental basis

Keep procedural generation and the 2-D/3-D dataset ladder because they connect the published
methods. Merge redundant sample galleries where possible. The main text needs one representative
sequence for 2-D and one for 3-D plus one dataset-ladder diagram. Generator algorithms and complete
parameter tables move to appendices.

#### 2.2 Reversible temporal computation

Keep the current conceptual order: activation memory, reversible block, temporal chaining,
backward replay, controlled results, OpenSTL validation, memory--compute boundary, stochastic
extensions, and applicability.

Refactor formulas as follows:

- group the reversible forward and inverse relations into one numbered system;
- retain one temporal-composition relation, one executed loss, one reverse-replay/adjoint relation,
  and one memory--compute relation;
- move the Jacobian expansion and most stochastic replay/Lipschitz/anchor derivations to an
  appendix;
- explicitly say which stochastic machinery is implemented in the published system and which is a
  general extension;
- emphasize that reversibility changes stored activations, not the predictive target or retained
  inference context.

Keep the published quality, memory, and runtime results as three separate evidence objects. Do not
combine runtime and memory merely because they share the horizon axis.

#### 2.3 V-BLT and spatial information density

Keep the local--global--local architecture, causal byte objective, RoPE-ND, predictive entropy, and
the published 2-D/3-D results. Use one compact multi-axis RoPE equation instead of the five-step
elementary derivation. Preserve the entropy definition because it supports the later routing
motivation.

Rewrite the final “measured versus proposed” subsection. Its boundary must become:

- V-BLT implements fixed hierarchical patches and measures predictive entropy;
- predictive entropy is evidence that spatial complexity is nonuniform;
- V-BLT did not itself establish a dynamic token router;
- the later integrated model uses an independently specified fine--coarse residual/temporal-change
  route score, physical metadata, and a real fixed quota;
- therefore the integrated routing result cannot be retrospectively attributed to V-BLT.

Delete the hypothetical numbered dynamic-partition formula `(2.42)` from the main text. If useful,
retain it in the appendix as a historical design proposal labelled as such.

#### 2.4 Chapter synthesis

Expand the conclusion by one English page. It must say that Chapters 2's methods solve adjacent
resource constraints but do not by themselves answer the central retained-context allocation
question. End with a three-column conceptual map in prose: mechanism, changed resource, unchanged
resource. Do not use a numbered table in the main text.

Maintain a content-transfer ledger for Articles 1--3 at the paragraph/claim/figure/table level. For
Articles 1 and 2, the executed method boundary, published numerical evidence, and author
contribution remain in the main narrative. For Article 3, the compact conceptual taxonomy remains
in Chapter 1. Appendices hold derivations, exhaustive assets, and provenance rather than the full
scientific substance.

### 6.4 Chapter 3 — retain the architecture/method role, replace the experiment design layer

The Chapter 3 title and the 3.1--3.8 progression remain. The chapter becomes the complete method
contract for both blocks of the combined article.

#### 3.1 Retained context as a finite-model problem

Keep re-entry, context-package fields, token budget, and nominal reach. Convert current Table 3.1
to a concise definition list or appendix symbol table. Keep three main equations: package/anchors,
retained-context cost, and reach. Make the reach--fidelity decomposition prose because it is an
explanatory partition, not a fitted probabilistic model.

Replace the current conceptual “frontier” illustration with a design-space diagram whose axes are
measured quantities: retained tokens, oldest available lag, and spatial detail. Do not draw a smooth
empirical frontier before data establish it.

#### 3.2 Hierarchical latent state

Keep the frozen two-level codec, stream definitions, metadata, joint attention, gradient boundary,
and asymmetric outputs. Consolidate codec/tensor-shape Tables 3.1--3.2 into appendix configuration
tables. The main architecture figure shows the conceptual data flow and trainable/frozen boundary;
an adjacent contract box or compact method table carries exhaustive tensor shapes. Do not force
every dimension into the schematic.

Add exact definitions for the allocation strategies actually implemented by the new campaign:
Flat, Linear-age, Geometric-age, and the seven fixed-budget allocation policies. Define nested,
exclusive, and residual overlap without implying they are equivalent-cost unless the manifest
confirms exact reassignment.

#### 3.3 Generative prediction

Keep the shared WindowDiT predictor and the deliberate joint denoising of the four-frame target
window. State the causal contract directly: all context precedes the first target; target latents do
not enter context; target-to-target attention is intentional for a jointly denoised future window.

Retain the executed diffusion objective and one conditional-predictor equation. Compress standard
Transformer, AdaLN, diffusion-forward, DDIM, and regression-control derivations into citations or an
appendix. The architecture, not a textbook diffusion derivation, should occupy the reader's visual
attention.

#### 3.4 Rollout and memory update

Keep the four-frame update, ring buffers, anchors, causal checks, and separate context/target/token
cost. Merge the buffer and anchor diagrams into one timeline. State explicitly that the current
central study stops gradients between calls; its long horizon comes from retained state and
self-feeding, not end-to-end BPTT through all forecast windows.

#### 3.5 TRB and evaluation

Rename public prose to Toroidal Re-entry Benchmark. Keep the world/viewport mechanism, exact
observability, pair-specific reach populations, primary frame/object metrics, and clip-clustered
analysis. Move metric edge cases, Hungarian assignment details, bootstrap algorithm, and
max-statistic derivation to appendices.

The main statistical contract is:

- primary development-selection metric: horizon-4 foreground MSE;
- final confirmation population and all task-specific re-entry metrics remain locked by the
  campaign ledger;
- training seeds 0 and 1 are independent displayed outcomes;
- clip-level uncertainty is conditional on a trained model and never relabelled as training-seed
  uncertainty;
- with two training seeds, report both dots, their arithmetic mean, and the direction agreement;
  do not manufacture a meaningful t interval from two points.

Add a terminology box distinguishing:

- **context age/reach:** how far before the target an observed state is retained;
- **prediction endpoint:** the four jointly predicted frames used by the v2 allocation matrix; and
- **autoregressive rollout horizon:** the number of predicted frames subsequently fed back into
  future model calls.

Only the third supports claims about self-fed accumulation or long-rollout stability.

#### 3.6 Registered hierarchy experiment programme

Retain the current hypotheses heading but replace its five broad hypotheses with the exact design
below. Suggested subsubsections are:

##### 3.6.1 Common protocol and 30k calibration

Explain the two 30k calibration arms, validation every 2k updates, the registered 2% convergence
rule, the observed ordering change between 12k and 16k, and why 30k is a common endpoint rather
than per-arm early stopping. State that “ranking reversal” is an observed crossing, not a code
feature.

##### 3.6.2 Context-budget frontier (`A4-FRT`)

Six budgets (`320, 768, 1152, 1792, 2432, 3072`) by three strategies (Flat, Linear-age,
Geometric-age), two seeds: 36 runs. Question: how does the quality cost of compression change over a
tenfold retained-token range?

##### 3.6.3 Allocation at 1,792 tokens (`A4-ALC`)

Seven distribution policies, one budget, two seeds. Question: which distribution across levels is
best when the total retained-token budget is invariant? Reused FRT cells are identified rather than
retrained.

##### 3.6.4 Coarse-context expansion (`A4-CEX`)

Hold six recent fine positions fixed and add `0, 4, 8, 16, 32, 64` coarse time positions, two
seeds: 12 runs. Question: what is the response to adding exclusive older coarse slots while total
budget, sequence length, attention work, and reach increase together? This is not a fixed-memory
causal estimate of reach alone.

##### 3.6.5 Hierarchy depth (`A4-LVL`)

One, two, and three levels at budget 1,792 using the best ALC strategy, two seeds. Reuse the one- and
two-level cells; train only the genuinely new three-level pair. Question: does additional depth help
for the registered one-/two-/three-level construction at fixed total payload? The three-level arm
adds a top stream, reallocates remaining coarse:fine payload, rounds slot counts, and changes
parameter count; it is not a pure scalar “number of levels” intervention.

##### 3.6.6 Temporal overlap (`A4-OVR`)

Nested, exclusive, and residual ownership at budget 1,792, two seeds. Question: is it useful to
use the registered ownership/representation packages? Exclusive support relocates coarse ages and
residual fine tokens change representation semantics, so the result is not attributed to overlap
alone.

##### 3.6.7 Space--time detail allocation (`A4-STS`)

At budget 1,792, cross fine quotas `64, 32, 16, 8` with recent, uniform, and geometric temporal
placement, two seeds: 24 runs. Question: how should limited fine spatial detail be distributed over
time? This is the explicit bridge between temporal hierarchy and later dynamic spatial routing.

##### 3.6.8 Selection, holdout, and mechanism diagnostics

Describe the hash-locked two-seed development selection, fresh 500-clip TRB confirmation set, and
bounded `A4-MEC` attention/re-entry diagnostics. The holdout is opened only after the selection
ledger is written. `A4-MEC` explains model use of context but does not create another independent
quality claim.

Across families there are 43 unique configurations and 86 two-seed training runs. The main text
uses human names; the configuration and interpretation registry (current Додаток В) contains the
complete identifier/config/hash matrix.

##### 3.6.9 Selected-checkpoint autoregressive rollout (optional additional evidence)

This is not part of the current 86-run training campaign. If authorized before its evaluation data
are opened, register a separate inference-only study using the selected Flat and hierarchical
checkpoints for both training seeds. Warm every model from the same ground-truth history, then
iterate four-frame predictions through the identical causal buffer-update path on a newly generated,
hash-locked long TRB set. Predeclare rollout lengths, NFE, common random numbers, clip count,
primary horizon bands, re-entry population, failure handling, and whether the selected smaller
Pareto hierarchy is included.

This add-on answers whether the selected v2 context policies remain useful and stable when predicted
states re-enter memory. It requires no retraining and must not reselect configurations. If it is not
run, Chapter 4 relies on legacy E5/E6 for long-rollout behavior and describes the new matrix only as
long-context allocation evidence.

#### 3.7 Integrated OpenSTL architecture — replace the old prototype

Delete the current fixed-region soft weighting and second-order rollout as the final method. Retain
at most one short historical paragraph explaining why that feasibility design was superseded:
fixed slots could not identify moving regions, full fine maps remained materialized in the early
dynamic version, repeated pixel decode/re-encode was lossy, and the second-order recurrence changed
the predictive dynamics.

Replace 3.7 with:

##### 3.7.1 Composition hypotheses and external contract

Define the OpenSTL `(B,T_in,C,H,W) -> (B,T_out,C,H,W)` interface and the separate claims for
hierarchy, reversibility, routing, and full-model Pareto position.

##### 3.7.2 Shared fine/coarse pyramid and genuine packed state

Describe stride-4 fine candidates, stride-8 coarse coverage, global fixed-quota gather before the
Transformer, exact retained tensor shapes, and why padding dense candidates would not count as a
memory saving.

##### 3.7.3 Routing policies and route-aware training

Define content, fine--coarse residual, temporal-change, and learned score components; minimum
per-frame coverage; deterministic ordering; and route stability endpoints. Make clear that a
static-only model cannot safely switch to dynamic inference without route-aware training.

Separate two factors. `static`, `density`, and `hybrid` are inference membership policies that can
be applied to a route-aware checkpoint as a switch/distribution-shift diagnostic. `mixed` is a
training exposure regime, not a fourth inference router; comparing static-trained with mixed-trained
models is a separately trained development ablation. Record top-k non-differentiability, the
selected-score gradient path, route collapse/margin, and membership stability as limitations.

##### 3.7.4 Physical position and hierarchy metadata

Define normalized spatial centre, signed time, scale, and level encoding. State the packed-slot
permutation invariance requirement. This subsection directly resolves the concern that positional
encoding designed for static slots cannot support changing membership.

The reproducibility contract specifies coordinate origin, cell-centre convention, denominator and
range for x/y, past/zero/future time signs, scale normalization, hierarchy-level indexing,
dataset-resolution handling, and behavior when input/output horizons change. Normalization is fixed
by the physical grid and protocol, never recomputed from selected tokens. Include future decoder
queries in the same contract and record permutation-test tolerances for fp32 and the deployed mixed
precision dtype.

##### 3.7.5 Identical-forward reversible core and future decoder

Use one grouped additive-coupling forward/inverse equation. Routing, encoder, and decoder remain
ordinary boundaries. Reversible and vanilla modes must match predictions, loss, input gradients,
and parameter gradients. Describe future coarse queries, cross-attention to packed memory, and the
residual full-frame decoder.

##### 3.7.6 Verification gates and result boundary

List CPU correctness, quota, route response, permutation, switch, inverse reconstruction,
gradient-parity, shape, checkpoint, and OpenSTL smoke gates. State that these prove compatibility,
not quality or CUDA memory savings.

#### 3.8 Chapter conclusions

Summarize the two method blocks without presenting results. End with the exact questions that
Chapter 4 answers.

### 6.5 Chapter 4 — preserve the results/operating-region structure, replace the headline evidence

#### 4.1 Experimental map and comparison rules

Keep the current purpose and budget vocabulary. Replace the legacy-code-first opening with a map of
questions:

1. training endpoint and predictor mechanism;
2. budget frontier and purchased reach;
3. allocation, level depth, overlap, and space--time placement;
4. robustness and known failure conditions;
5. integrated component composition and OpenSTL external behavior.

Move the full program matrix to an appendix. Keep one compact sign convention only if it prevents
real ambiguity. Every comparison says exactly what is matched: retained tokens, forward function,
quota, training schedule, or dataset protocol.

Every heading, caption, result table, abstract sentence, and conclusion uses the evidence-status
vocabulary defined in Section 1. Add a claim-to-artifact matrix containing the final sentence-level
claim, population, exact configurations, training-seed unit, selection role, source artifact, and
whether the sealed holdout was exposed.

#### 4.2 Training endpoint and bounded mechanism controls

Rename the section to **Training endpoint and predictor-mechanism controls**.

- 4.2.1 presents the 30k calibration curves for the two reference arms and explains the common
  endpoint.
- 4.2.2 compresses the current regression-versus-diffusion result to the evidence needed to justify
  WindowDiT's selected head.
- 4.2.3 reduces EBT to a concise negative/diagnostic result. Its composite plot and extended energy
  discussion move to the appendix unless the new article makes EBT a named secondary contribution.

The NFE arithmetic is inline; the generic energy-descent equation moves to the appendix. Current
Table 4.1 is split: prediction results remain a compact result table, while computation counts and
interpretation become prose or appendix fields.

#### 4.3 Context budget and purchased temporal reach

Make the complete `A4-FRT` and `A4-CEX` development evidence the primary budget-allocation study.
Retain sound E2 rollout evidence in a bounded but substantive main-text role because it tests
self-fed behavior that the four-frame v2 endpoint does not. It is not silently discarded or treated
as interchangeable with the new matrix.

Suggested subsubsections:

- **4.3.1 Tenfold context-budget response:** six budgets, three strategies, both seeds.
- **4.3.2 Purchased coarse reach with fixed recent detail:** six CEX points.
- **4.3.3 Re-entry confirmation and context-use diagnostic:** locked holdout and bounded MEC.

Do not fit a smooth crossing or announce a universal token threshold. The observed budgets are
discrete interventions.

#### 4.4 Allocation, hierarchy depth, overlap, and space--time placement

Replace the present token-arithmetic/frontier mixture with four direct registered-package
questions:

- **4.4.1 Allocation policy at 1,792 tokens (`A4-ALC`)**;
- **4.4.2 One, two, or three levels (`A4-LVL`)**;
- **4.4.3 Nested, exclusive, or residual overlap (`A4-OVR`)**;
- **4.4.4 Fine quota and temporal placement (`A4-STS`)**.

Use exact compound estimands: CEX varies older coarse slots together with budget/compute/reach; LVL
compares the registered multi-level constructions; OVR compares ownership/representation packages.
Captions must not label these as isolated effects of reach, level count, or overlap.

The analytic token arithmetic remains as a short methods check or appendix. Any 2.67x or 3.43x
ratio is labelled retained-token/state arithmetic, not measured end-to-end VRAM or equal-quality
gain.

#### 4.5 Package/model interactions

Keep the useful E4 result as secondary historical evidence that package ranking can depend on the
prediction head and training seed. Reduce the current table-plus-figure duplication to one
seed-level estimation plot. The long table of five seed-specific clip intervals moves to the
appendix.

Connect E4 to the new study carefully: it motivates fixed heads, common schedules, and two-seed
visibility, but it is not pooled statistically with the new campaign because protocols differ.

#### 4.6 Robustness

Lead with the new campaign's uniform two-seed replication, immutable configs, and final 30k
endpoint. Retain sampler/NFE and alternate-codec evidence only as bounded robustness checks of the
legacy mechanism. Place their full simultaneous intervals in the appendix. One main robustness
figure is enough if it answers whether direction persists; otherwise describe them briefly and
remove both figures from the main text.

The full matrix is a registered development estimate. “Direction reproduced across two seeds” is
used only for contrasts with aligned directions. Sealed-holdout language is restricted to selected
roles actually evaluated by the finalizer; it does not promote ALC/LVL/OVR/STS families wholesale
to confirmation status.

#### 4.7 Negative and boundary evidence

Keep negative results, but rank them by relevance:

1. E5a non-replicated late reversal: keep one concise figure and the corrected conclusion;
2. E6 coverage/trajectory versus identity: keep because it directly supports the reach--fidelity
   boundary; split its current two-panel table into one result figure plus appendix exact values;
3. E8a privileged raw memory: move to appendix or a short diagnostic paragraph because it is not a
   comparator;
4. old short Moving-MNIST/TaxiBJ feasibility numbers: remove from this section after the new
   matched OpenSTL results exist;
5. Long Moving-MNIST is not part of the new hierarchy or integration campaign; retain E6 only as
   historical boundary evidence, not as a new benchmark claim.

#### 4.8 Integrated model composition and OpenSTL results

Replace current 4.8 entirely after the integrated gates and measurements complete. Suggested
subsubsections:

##### 4.8.1 Correctness and parity gates

Report quota correctness, true packed sizes, positional permutation invariance, static/dynamic
switch behavior, reversible inverse drift, and forward/gradient parity. This is a compact gate
table in the appendix; the main text gives a pass/fail summary.

##### 4.8.2 Component effects on measured memory

Compare Flat-Dense, Hier-Static, Flat-Reversible, Hier-Static-Reversible, Hier-Dynamic, and Full
under paired host/batch/precision/protocol conditions. Report retained bytes, inference peak,
training peak, activation increment, throughput, and recomputation overhead. Do not use a waterfall
unless components are truly additive under one baseline; a paired dot/slope plot is safer.

Add activation checkpointing as the practical same-model comparator to ordinary autograd and
reversible execution. Match model, checkpoint, batch, precision, and horizon; report allocated and
reserved peaks, throughput, and recomputation. If this comparator is not executed, restrict the
claim to “lower than ordinary autograd storage” and do not claim the best available
memory--compute trade-off.

##### 4.8.3 Routing quality, quota frontier, and stability

First compare static, density, and hybrid inference membership on a route-aware checkpoint as a
switch diagnostic. Separately compare static-only training with mixed/route-aware training when
both checkpoints exist. Report quality, coverage, entropy, churn/Jaccard, route margin, and
flicker/seam diagnostics. Then vary quota to test whether dynamic routing preserves quality at a
smaller real retained tensor. Equal-quota routing is an allocation result, not a memory saving, and
gains from mixed training are not attributed solely to the routing score.

##### 4.8.4 Moving MNIST

Report seeds 0 and 1 for the final full configuration, matched local baselines, validation-only
checkpoint selection, and one locked test evaluation. Use the 10-to-10 OpenSTL protocol.

##### 4.8.5 TaxiBJ

Repeat the same structure for the two-channel 4-to-4 real-data protocol. The same architecture must
handle both shapes without dataset-specific positional slots.

##### 4.8.6 Cross-dataset quality--memory frontier and claim decision

Show full-model and matched-baseline positions on measured peak memory versus quality. The full
model may be described as competitive if it satisfies the declared quality tolerance and as
Pareto-improving only if no comparator is both more accurate and lower-memory under the matched
protocol. Report dataset-specific results even if only one dataset passes.

Every intended cell is labelled as one of: CPU correctness gate, same-checkpoint execution/profile,
separately trained development ablation, long full-model confirmation, matched local baseline, or
published contextual reference. The size of Sections 4.8.2--4.8.6 contracts automatically when the
corresponding scheduled evidence is absent. Unmatched published numbers never establish a Pareto
claim.

#### 4.9 System-level selection rule

Keep and redraw the useful two-resource/decision-flow synthesis, expanded to four questions:

1. Is relevant past evidence outside the flat retained window?
2. Is the error dominated by coverage/trajectory or fine identity/detail?
3. Is deployment state, training activation memory, or both limiting?
4. Is information density sufficiently uneven and stable for routing to justify its overhead?

The new figure must not imply that component ratios multiply. Replace “fixed regions are the
starting point” with the measured conclusion from the new packed router.

#### 4.10 Software and practical application

Update package/test counts, branches, configs, and commands only from generated manifests. Separate
the released public package from later research branches. Add the single-command Article 4 and
integrated campaign controllers, resume semantics, local-authoritative logging, W&B mirroring, and
artifact hashes to the software and reproduction appendix (current Додаток Ж) rather than filling
the narrative with paths.

#### 4.11 Validity and chapter conclusions

Retain internal, construct, statistical, and external validity headings. Rewrite the limitations
after final results so they match the new design:

- two seeds show reproducibility of direction but do not characterize a broad training-seed
  distribution;
- TRB has a static toroidal world and action-conditioned camera motion;
- six budget points define observed regimes, not a continuous law;
- Moving MNIST and TaxiBJ establish bounded synthetic/real external behavior, not natural-video
  generality;
- routing selection and quota choice must be validation-only;
- same-host profiles are necessary for memory comparisons;
- measured total memory, retained state, and core activation memory are separate endpoints.

End with claims ordered by evidence strength: registered development frontier with sealed
confirmation of selected contrasts; allocation mechanisms at their exact evidence status; legacy
long-rollout and negative boundaries; component memory effects; and full-model OpenSTL position.

### 6.6 General conclusions and references

Rewrite `06_CONCLUSIONS.md` only after the Chapter 4 claim ledger is locked. Preserve one conclusion
for each dissertation-level task, in the same order as the Introduction, and attach every numerical
or comparative sentence to its evidence-status/artifact row. Distinguish clearly among long-context
allocation, long autoregressive rollout, activation-memory reduction, spatial routing, and the
full-model OpenSTL result. A negative or absent integrated result changes the conclusion rather than
being omitted.

Audit `07_REFERENCES.md` after content migration. Remove a citation only when no retained main-text
or appendix claim depends on it; add sources for new routing, checkpointing, visualization, and
OpenSTL statements; preserve the publication records required by the contribution map; and verify
that every bibliography entry is cited and every citation resolves. Record moved Article 1--3
claims in the content-transfer ledger so appendix relocation does not orphan their sources.

**Visuals/equations/tables:** none added to general conclusions or references.

**Gates:** task-to-conclusion alignment, claim-to-artifact completeness, citation round trip, and
front-matter/conclusion agreement.

## 7. Experiment-to-evidence design

### 7.1 New hierarchy campaign

| Family | Runs represented | Primary question | Main figure | Main table | Secondary/appendix |
|---|---:|---|---|---|---|
| 30k calibration | 2 calibration arms | Why is 30k the common endpoint? | learning curve of primary metric at every 2k, both arms | none | convergence-rule ledger and all metrics |
| A4-FRT | 36 | How do three allocation strategies respond over a tenfold budget range? | horizon-4 foreground MSE versus numeric token budget, both seed dots and mean line | concise six-budget means with seed values | other horizons, pixel MSE, SSIM, PSNR, train curves |
| A4-ALC | 14 logical results, 8 new runs | Which of seven policies is best at 1,792? | horizontal dot plot ordered by two-seed mean | selected contrasts only | complete seven-policy table |
| A4-CEX | 12 | What is the response to adding exclusive older coarse slots while budget, compute, and reach increase? | response curve versus coarse time positions | optional if exact selected contrasts are central | all component metrics and token arithmetic |
| A4-LVL | 6 logical results, 2 new runs | How do the registered one-/two-/three-level constructions compare at fixed payload? | paired one/two/three-level dots | none unless selected as primary claim | exact configs/tensor shapes/parameter change |
| A4-OVR | 6 logical results, 4 new runs | How do the registered ownership/representation packages compare? | nested/exclusive/residual paired dots | none | overlap sets, residual semantics, token audit |
| A4-STS | 24 | How do fine quota and time placement interact? | 4x3 heat map of two-seed mean primary metric | best/worst/registered contrasts | both seed heat maps, all secondary metrics |
| A4-MEC | no new training | Does the selected model actually use older/re-entry context? | one attention/re-entry diagnostic | none | per-head/layer diagnostics |
| Confirmation | selected arms only | Does locked development selection reproduce on fresh TRB? | effect-size plot on registered population | final confirmation results | selection ledger and clip-level arrays |
| Optional selected-checkpoint rollout | no retraining; selected checkpoints only | Do selected policies remain useful/stable when predictions update memory over many windows? | error/effect versus rollout horizon | selected horizon-band results | full per-frame arrays and fixed qualitative strips |

With two seeds, uncertainty graphics show the two independent points. Clip-level confidence
intervals may be displayed only for the locked confirmation estimand and must be labelled
“conditional on trained seed/model.” The main selection statistic is horizon-4 foreground MSE at
30k on development validation; final task-specific re-entry results come from the locked fresh
confirmation set.

### 7.2 Current partial campaign evidence and expected shape

At the read-only status check on 2 September 2026, Wave 1 had 21 complete runs, one running, 58
pending, and no failures. The transient aggregation read each completed run's final-step
`validation_metrics.jsonl` under
`artifacts/v2/article4_primary_30k/runs/` and used the manifest-declared
`horizon4_foreground_mse`. Because that live directory changes as the controller advances and no
immutable snapshot artifact was created, the provisional numeric table has been removed from this
enduring plan rather than allowed to become an uncited quasi-result.

The observed snapshot suggested that strategy order can change with budget, but it is a
**development, incomplete-campaign observation**, not a dissertation-final result. It supports
three layout decisions only:

1. strategy order can cross as budget changes, so one winner bar chart would be misleading;
2. the primary metric must remain visually separate from SSIM/PSNR, whose order can differ; and
3. no monotone fitted frontier or universal crossover should be asserted.

The probable final narrative, if the remaining points preserve the pattern, is that hierarchy is
most valuable under severe budget pressure and Flat recovers as detailed context becomes
affordable. If remaining results disagree, retain the same figures and change the interpretation;
the structure is outcome-independent.

Any later numeric progress snapshot must be generated as a dated JSON/CSV artifact with source
checkpoint step, run/config hashes, aggregation code commit, and file SHA-256. Final dissertation
numbers come only from the post-training aggregate and selection/confirmation ledgers.

### 7.3 Integrated-model evidence

| Evidence block | Required comparisons | Main figure | Main table | Decision criterion |
|---|---|---|---|---|
| correctness | vanilla/reversible; token permutations; route switches | architecture with tested boundaries | appendix gate matrix | all release gates pass |
| hierarchy memory | Flat-Dense vs Hier-Static at matched core/decoder | retained and measured memory versus quota/horizon | paired profile results | actual retained reduction and measured peak change |
| reversibility | same checkpoint/config, vanilla vs reversible | peak training memory versus horizon or batch | parity/memory results | same forward; gradients within tolerance; lower core/end-to-end peak |
| inference routing switch | static/density/hybrid at fixed quota on a route-aware checkpoint | quality and route-stability plots | route metrics | equal-quota switch behavior and stability |
| routing training regime | static-trained versus mixed/route-aware-trained checkpoints | seed-level quality/stability comparison | training-regime results | benefit, if any, of route-aware exposure rather than a router-score-only claim |
| quota frontier | large versus smaller dynamic `K` | measured quality--memory Pareto curve | selected quotas | smaller real tensor within declared quality tolerance |
| final MM | full seeds 0/1 plus matched baselines | seed-level quality--memory plot and one fixed prediction strip | MSE/MAE/SSIM/PSNR, memory, throughput | competitive quality and memory position |
| final TaxiBJ | same | same visual grammar | same | same, reported independently |

The primary component matrix is Flat-Dense, Hier-Static, Flat-Reversible,
Hier-Static-Reversible, Hier-Dynamic, and Full. Component ratios are never multiplied. A routing
gain at fixed quota and a hierarchy state reduction are reported separately from a reversible core
activation reduction.

Before drafting Section 4.8 as a result section, maintain this evidence-cell ledger:

| Intended evidence | Current commitment | Training need | Claim if absent |
|---|---|---|---|
| CPU architecture/parity gates | implemented and verified | none | method correctness only |
| vanilla vs reversible same-checkpoint profile | planned | none beyond usable checkpoint | no reversible memory result until measured |
| activation checkpointing comparator | newly required | none beyond usable checkpoint | restrict comparison to ordinary autograd if absent |
| Flat-Dense vs Hier-Static quality | bounded development ablation, not one shared checkpoint | separate short/adequate training | retained-size arithmetic and feasibility only |
| static/density/hybrid switch | planned diagnostic on route-aware checkpoint | no extra training | distribution-shift diagnostic only |
| static-trained vs mixed-trained | not guaranteed by the four final runs | separate development training | no claim about training-regime benefit |
| dynamic quota frontier | planned but conditional after routing gate | additional bounded quota evaluations/training as required | equal-quota allocation result only |
| final full model on MM/TaxiBJ | four long runs planned, conditional on gates and explicit authorization | two datasets x seeds 0/1 | no full-model result until authorized and completed |
| matched local SimVP/TAU/other baseline | conditional on exact compatible checkpoints or reruns | possibly substantial baseline training | published numbers are context only; no matched Pareto claim |

This ledger prevents the four full-model runs from being mistaken for six component-training cells
or a complete baseline campaign. Runtime/storage estimates are updated after the 200-step GPU
measurement and after the exact matched-baseline set is resolved.

### 7.4 Experiment impact of the independent review

The review does **not** invalidate or change any of the 86 running Article 4 training cells. Do not
stop, restart, or relaunch them. Their target, data split, 30k endpoint, seed pair, and post-training
selection remain unchanged.

Three types of follow-up are distinguished:

1. **Text/evidence-status correction, required and training-free.** Rename the new study as
   long-context allocation with a four-frame endpoint; retain legacy E5/E6 for self-fed stability;
   restrict sealed-confirmation language to selected roles.
2. **Selected-checkpoint long-rollout evaluation, recommended but separately authorized.** Adapt
   and test a causal packed-buffer evaluator, generate a new sealed long TRB set, and evaluate the
   selected Flat/hierarchical checkpoints for seeds 0/1. This adds inference/evaluation GPU time and
   no training. Run a small timing preflight before estimating the total; it should be measured in
   hours rather than assumed to require another multi-day 86-run campaign.
3. **Integrated evidence additions, conditional on desired claims.** Activation checkpointing is an
   extra same-checkpoint profile with no training. Flat-versus-hierarchy quality, static-trained
   versus mixed-trained routing, and unmatched local baselines may require additional bounded or
   full training. If they are not run, narrow the component/Pareto claims and keep the four final
   full-model runs unchanged.

No new experiment is launched or inserted into the live campaign manifest merely because it appears
in this editorial plan. Each additional evaluation/training cell receives its own authorization,
immutable manifest, preflight, runtime estimate, and evidence status.

## 8. Asset-by-asset disposition

### 8.1 Chapter 1 figures

- **1.1 system boundary:** keep and simplify.
- **1.2 hidden viewport:** merge with 1.1 or retain only if it explains partial observation better.
- **1.3 teacher-forced/self-fed:** keep.
- **1.4 resource locations:** keep and expand to the four-resource decomposition.
- **1.5 consistency matrix:** keep; merge taxonomy content from 1.7.
- **1.6 combined SDXL/video example:** move to Article 3/[A3] support appendix.
- **1.7 taxonomy:** merge into 1.5.
- **1.8 published visual-perception example:** move to appendix.
- **1.9 failure montage:** keep.
- **1.10 trajectory:** keep; merge identity/support message from 1.21 if legible.
- **1.11 cross-level consistency:** retain as a small single-purpose schematic or merge into 1.5.
- **1.12 static-to-video bridge:** move to literature appendix.
- **1.13--1.19 diffusion, image-to-image, ControlNet, inversion, and StyleAligned sequence:** remove
  from main text; preserve selected published assets in appendix/provenance.
- **1.20 interfaces:** merge with the resource-locations/system map.
- **1.21 support/identity:** merge with 1.10 or keep only if E6 becomes a headline boundary.
- **1.22 activation tape:** keep.
- **1.23 fixed grid:** keep.
- **1.24 flat context:** keep.
- **1.25 system map:** keep and redraw after integrated results.

### 8.2 Chapter 2 figures

- **2.1--2.2 dataset sequences:** retain one main sequence and move the second to appendix.
- **2.3 dataset ladder:** keep.
- **2.4 activation-storage schedules:** keep.
- **2.5 published reversible block / 2.6 TempVerse architecture:** make 2.6 the main architecture;
  preserve 2.5 with the publication material in appendix if redundant.
- **2.7 temporal chaining / 2.8 backward replay:** merge into one replay timeline if readable.
- **2.9 quality, 2.10 memory, 2.11 runtime:** keep as three distinct result figures.
- **2.12 V-BLT flow / 2.13 published architecture:** use 2.13 as the main architecture and move the
  redundant flow to appendix.
- **2.14 causal mask:** use as an inset or appendix figure.
- **2.15 RoPE-ND:** keep.
- **2.16 entropy pipeline and 2.17 entropy result:** keep; they answer different questions.
- **2.18 point estimates:** keep if it is the clearest published result plot.
- **2.19 measured/proposed boundary:** replace with the explicit V-BLT-to-router evidence boundary.

### 8.3 Chapter 3 figures

- **3.1 re-entry and 3.15 storyboard:** keep one mechanism illustration; move the second to the TRB
  appendix.
- **3.2 package comparison:** keep and redraw with exact new policies.
- **3.3 conceptual frontier:** replace with a non-empirical design-space diagram.
- **3.4 codec dimensions:** merge into the primary architecture figure.
- **3.5 token sequence, 3.6 joint attention, 3.9 full architecture:** consolidate into one
  architecture/data-flow figure with one purpose.
- **3.7 gradient paths:** move to appendix or a small architecture inset.
- **3.8 objective interfaces:** move to Chapter 4 mechanism control or appendix.
- **3.10 DDIM timeline:** appendix.
- **3.11 ring buffer and 3.12 anchors:** merge.
- **3.13 token costs:** replace with exact allocation-policy design figure.
- **3.14 TRB:** keep and rename.
- **3.16 reach population:** keep.
- **3.17 object assignment:** appendix unless E6 remains a main boundary result.
- **3.18--3.19 old integrated architecture/fixed regions:** remove from main method and preserve as
  historical feasibility assets in appendix; replace with the new integrated architecture and
  route/position schematic.

### 8.4 Chapter 4 figures

- **4.1 generation objectives:** keep only if required to understand the bounded head control.
- **4.2 EBT composite:** appendix.
- **4.3 old 6k/20k optimization:** replace with the two-arm 30k calibration curve.
- **4.4 old E2 effects and 4.5 old frontier:** supersede them as the primary allocation evidence
  with new FRT/CEX plots, while retaining sound old self-fed rollout evidence in a bounded main-text
  figure or subsection and preserving the complete results in provenance.
- **4.6 head/package interaction:** retain one simplified seed-level plot.
- **4.7 sampler and 4.8 codec robustness:** move one or both to appendix unless needed for a distinct
  main claim.
- **4.9 non-replicated late reversal:** keep, simplified.
- **4.10 privileged raw memory:** appendix or short diagnostic.
- **4.11 coverage/identity boundary:** keep.
- **4.12 old integrated memory:** replace with same-forward reversible and full component profiles.
- **4.13 two resource axes:** redraw as four intervention/resource axes.
- **4.14 decision flow:** keep and update after results.

Add new single-purpose figures for A4-FRT, A4-ALC, A4-CEX, A4-LVL, A4-OVR, A4-STS,
confirmation/MEC, integrated memory, routing stability/quota, Moving MNIST Pareto, and TaxiBJ
Pareto. Some may move to the article supplement if Chapter 4 becomes too dense.

### 8.5 Current main tables

- Chapter 1 resource and consistency tables become prose/figures or appendix definitions.
- Chapter 2 keeps only empirical TempVerseFormer and V-BLT quality/memory/cost tables. The
  reversibility boundary, strategy synthesis, component/evidence-status, and dataset-role tables
  move to appendices.
- Chapter 3's package fields, latent interface, and dataset/evidence roles move to appendices.
- Chapter 4 keeps empirical result tables. Configuration arithmetic and experiment maps move to
  appendices. The current E4 and E6 multi-purpose tables are split or simplified.
- The old integrated quality and memory tables are preserved in a historical appendix and replaced
  in the main text by the new matched results.

## 9. Formula disposition by current range

| Current range | Action |
|---|---|
| 1.1--1.4 | group state/observation/prediction; keep objective only if reused |
| 1.5--1.7 | keep one memory decomposition and one constrained objective; make consistency vector prose if not estimated |
| 1.8--1.11 | move detailed object/trajectory/mask/cross-level metrics to evaluation appendix; retain at most one task-level relation |
| 1.12--1.14 | keep activation scaling, fixed-grid count, and reach because they define the three resource bottlenecks |
| 2.1--2.3 | group or move procedural-generation details to appendix |
| 2.4--2.6 | retain activation-memory scaling in compact form |
| 2.7--2.11 | one grouped reversible forward/inverse system |
| 2.12--2.18 | retain composition, executed loss, replay, and cost; move algebraic expansion |
| 2.19--2.27 | move stochastic replay, anchors, and stability derivations to appendix except one used contract |
| 2.28--2.34 | retain compact V-BLT mapping, causal objective, and byte loss |
| 2.35--2.39 | replace with one multi-axis RoPE-ND equation |
| 2.40--2.41 | retain predictive entropy as one grouped definition |
| 2.42 | remove from main text; label historical proposal if retained in appendix |
| 3.1--3.5 | reduce to package/anchors, budget, and reach |
| 3.6 | prose only: conceptual risk partition |
| 3.7--3.16 | reduce codec, token/metadata, attention, and gradient boundary to 5--6 equations |
| 3.17--3.28 | retain executed diffusion/predictor objective; cite standard DiT/DDIM/regression details |
| 3.29--3.32 | retain stream update and state-cost relation; attention complexity may be inline |
| 3.33--3.40 | retain TRB rendering/observability population and primary estimand; move metric details |
| 3.41--3.45 | retain across-seed effect and one cluster-aware inference relation; move algorithms |
| 3.46--3.53 | delete as final architecture; replace with new packing, routing, physical position, reversible-coupling, and decoder/state-cost equations |
| 4.1 | keep only if a unified effect sign is still needed |
| 4.2--4.3 | make NFE arithmetic inline; move EBT descent to appendix |
| 4.4 | move detailed paired-estimator derivation to methods/appendix |
| 4.5 | keep retained-token gain or replace with the exact new measured state-cost relation |

The resulting target is approximately 46--60 numbered equations in the main text. Final numbers are
assigned only after text, appendices, and figure/table callouts stabilize.

Before moving or unnumbering any formula, add it to a scientific-dependency ledger with columns:
current number, scientific role, implementation location, downstream prose references, dependent
figures/tables, result estimand, destination, and replacement callout. Bespoke re-entry populations,
primary endpoints, package construction, routing, and compound object metrics remain traceable in
the main text even when their derivations move. Standard MSE, softmax, DDIM, elementary inverse, and
bootstrap algorithm steps can be compressed safely only after the dependency check passes.

## 10. Placeholder and result-insertion protocol

Every unfinished result is represented by a visible token, for example:

`[RESULT PENDING: A4-FRT primary figure; source=post_training/aggregate.json; generator=... ]`

Use stable keys:

- `{{A4_CAL_CONVERGENCE_FIG}}`
- `{{A4_FRT_PRIMARY_FIG}}`
- `{{A4_ALC_PRIMARY_FIG}}`
- `{{A4_CEX_PRIMARY_FIG}}`
- `{{A4_LVL_PRIMARY_FIG}}`
- `{{A4_OVR_PRIMARY_FIG}}`
- `{{A4_STS_PRIMARY_FIG}}`
- `{{A4_CONFIRMATION_TABLE}}`
- `{{A4_MEC_DIAGNOSTIC_FIG}}`
- `{{A4_SELECTED_ROLLOUT_FIG}}`
- `{{INT_GATE_SUMMARY}}`
- `{{INT_COMPONENT_ABLATION_TABLE}}`
- `{{INT_CHECKPOINTING_PROFILE}}`
- `{{INT_MATCHED_BASELINE_TABLE}}`
- `{{INT_COMPONENT_MEMORY_FIG}}`
- `{{INT_ROUTING_SWITCH_FIG}}`
- `{{INT_ROUTING_TRAINING_REGIME_FIG}}`
- `{{INT_ROUTING_STABILITY_FIG}}`
- `{{INT_QUOTA_FRONTIER_FIG}}`
- `{{INT_MM_RESULTS_TABLE}}`
- `{{INT_TAXIBJ_RESULTS_TABLE}}`
- `{{INT_MM_PARETO_FIG}}`
- `{{INT_TAXIBJ_PARETO_FIG}}`
- `{{INT_CROSS_DATASET_SYNTHESIS}}`

Each placeholder receives a registry row containing source artifact, config hashes, generator,
metric, population, expected file path, completion check, and allowed claim language.

The registry applies to result-bearing sentences as well as figures and tables. Each row also
records evidence status, exact configuration IDs, data split, selection role, training seeds,
uncertainty unit, metric implementation/version, generator commit, failure/absence behavior, and
the positive/neutral/negative text block that is allowed. Dataset-specific Pareto placeholders stay
separate; a cross-dataset synthesis is filled only when its estimand is explicitly defined.

Planned generated asset names are stable descriptive names before final dissertation numbering:

- `figures/dissertation_v2/a4_calibration_30k.svg`;
- `figures/dissertation_v2/a4_frt_budget_response.svg`;
- `figures/dissertation_v2/a4_alc_policy_effect.svg`;
- `figures/dissertation_v2/a4_cex_coarse_reach.svg`;
- `figures/dissertation_v2/a4_lvl_depth_effect.svg`;
- `figures/dissertation_v2/a4_ovr_overlap_effect.svg`;
- `figures/dissertation_v2/a4_sts_space_time_surface.svg`;
- `figures/dissertation_v2/a4_confirmation_effect.svg`;
- `figures/dissertation_v2/integrated_architecture.svg`;
- `figures/dissertation_v2/integrated_component_memory.svg`;
- `figures/dissertation_v2/integrated_routing_switch.svg`;
- `figures/dissertation_v2/integrated_routing_stability.svg`;
- `figures/dissertation_v2/integrated_quota_frontier.svg`;
- `figures/dissertation_v2/integrated_moving_mnist_pareto.svg`;
- `figures/dissertation_v2/integrated_taxibj_pareto.svg`.

Final chapter numbers are assigned by the manuscript build, not encoded in source filenames. Every
generator also writes the exact plotted rows to a companion CSV/JSON file so the figure is auditable
without reverse-engineering pixels.

Prepare three outcome-independent paragraph variants for each primary hypothesis:

- **positive:** the candidate satisfies the declared direction/tolerance;
- **neutral/parity:** quality is within the declared margin, with the measured resource difference;
- **negative:** the experiment does not establish the expected advantage and the claim remains
  component-specific or unsupported.

The positive template may say “lay on the empirical Pareto frontier” only after measured data prove
it. It may not contain invented ratios such as 3--4x hierarchy or 2--3x routing. Analytic retained
token ratios, same-forward reversible activation ratios, and total GPU peak ratios are separate
fields.

## 11. Appendix migration and expansion

Do not replace the present appendices by title alone. Preserve the institution's existing Ukrainian
letter sequence and create an old-to-new migration ledger for every subsection, table, figure,
administrative record, and permission entry. The default mapping is:

- **Додаток А — Publications, dissertation results, and personal contribution:** retain the current
  publication/result/contribution map; add the one-article mapping only when its status is accurate.
- **Додаток Б — Administrative records and implementation acts:** preserve signed wording, dates,
  scans, and institutional records unchanged unless an authorized replacement is supplied.
- **Додаток В — Configuration and interpretation registry:** expand with symbols/tensor interfaces,
  method-contract tables, comparison licences, evidence-status vocabulary, and the complete 43-cell/
  86-run hierarchy identifier/config/hash matrix.
- **Додаток Г — Evidence and provenance index:** add claim-to-artifact rows, datasets/split manifests,
  source/config/checkpoint hashes, publication content-transfer records, and protected-set access
  status.
- **Додаток Д — Statistical reporting contract:** retain primary estimands, nonstandard re-entry
  populations, seed units, clip-clustered inference, assignment/object metrics, selection rules, and
  formula-dependency records. Standard derivations may be placed here without removing bespoke
  endpoint definitions from the main text.
- **Додаток Е — Additional methods and negative results:** retain EBT, sampler/codec robustness,
  raw-memory E8a, old E2--E8 detail, old fixed-region integration evidence, stochastic replay/RoPE/
  DDIM derivations, and selected publication-support figures whose main-text role was removed.
- **Додаток Ж — Software and reproduction map:** add Article 4 and integrated campaign controllers,
  resume/W&B procedures, integrated configs and evidence-cell status, parity/checkpointing/routing
  profiles, final four runs, generator commits, artifact hashes, and QA commands.
- **Додаток И — Figure provenance and permissions:** preserve every current source/licence/permission
  record and add new figure/table generator inputs and output hashes.

The migration ledger has columns `old location`, `asset/claim`, `new location`, `reason`, `source`,
`permission/administrative constraint`, and `verification status`. No old appendix item is deleted
until it has a destination and its institutional/provenance role has been checked. Any change to the
appendix-letter convention requires explicit institutional approval rather than an editorial
assumption.

## 12. Implementation sequence

### Phase 0 — freeze and inventory (0.5--1 day)

1. Do not edit the active Path B worktree while the Article 4 controller owns it.
2. After the campaign finishes, create a clean dissertation-editing worktree/branch from its pinned
   scientific commit.
3. Hash the current Ukrainian manuscript and figure tree as the preserved fallback.
4. Regenerate formula/figure/table/heading inventories programmatically.
5. Create the placeholder registry before moving content.

### Phase 1 — structural skeleton without claims (1--2 days)

1. Copy the current manuscript tree in the new editing worktree.
2. Add the revised English subsection headings and placeholder blocks.
3. Move obsolete/tutorial material to appendices without deleting provenance.
4. Do not renumber formulas/figures/tables yet.

### Phase 1b — front matter and migration ledgers (1--2 days)

1. Inventory both abstracts, novelty/contribution claims, publication records, and obsolete
   prototype numbers.
2. Build task-to-chapter-to-conclusion-to-publication, content-transfer, appendix-migration,
   formula-dependency, and claim-to-artifact ledgers.
3. Preserve administrative acts, personal-contribution records, licences, and permissions without
   paraphrasing institutional text.

### Phase 2 — Chapters 1 and 2 surgical refactor (4--6 days)

1. Insert the spatial/temporal/computational hierarchy framing.
2. Reduce Chapter 1's static-control gallery.
3. Group/demote Chapter 2 derivations while preserving Articles 1--3 evidence.
4. Write the V-BLT-to-dynamic-routing boundary.
5. Update citations and provenance entries for every moved asset.

### Phase 3 — Chapter 3 method rebuild (4--6 days)

1. Consolidate the WindowDiT architecture figures and formulas.
2. Insert the exact 30k/86-run hierarchy design.
3. Replace the old integration prototype with the verified OpenSTL-native architecture.
4. Add correctness and claim-boundary language.

### Phase 4 — Chapter 4 result skeleton (3--4 days)

1. Replace legacy headline positions with new experiment-family placeholders.
2. Retain selected E4/E5/E6/E7 evidence in bounded roles.
3. Move full legacy tables/figures to appendices.
4. Prepare single-role generators and caption templates.

### Phase 5 — result lock and insertion (4--7 days after experiments)

1. Accept only finalized, hash-locked aggregate/confirmation artifacts.
2. Generate all figures/tables from machine-readable sources.
3. Fill positive/neutral/negative interpretation branches from measured results.
4. Insert integrated results only after validation-only selection and locked test evaluation.
5. Rewrite Chapter 4 conclusion and general conclusions last.

### Phase 6 — final editorial and reproducibility QA (3--5 days)

1. Freeze numbering once.
2. Regenerate provenance counts and cross-reference maps.
3. Validate every image link, equation/table/figure callout, citation, and appendix reference.
4. Run dissertation audit, text consistency, scientific config validation, and link checks.
5. Render the complete document and inspect page breaks, figure legibility, table overflow, and
   mixed-language transitions.
6. Record hashes for the final manuscript, generators, data tables, and evidence bundles.

### Phase 7 — bilingual and institutional submission cleanup (3--6 days plus external response time)

1. Harmonize English insertions with the final required manuscript language without changing
   scientific meaning.
2. Update both abstracts, keywords, publication/contribution records, implementation acts, and
   appendix lettering.
3. Confirm figure permissions and institutional formatting requirements.
4. Obtain author/institution input for wording that cannot be inferred from the repository.
5. Re-run the full claim, provenance, cross-reference, and rendered-document audit.

Expected technical and editorial effort is about **23--39 focused working days**, excluding GPU
time, article submission, and waiting for institutional responses. A narrower dissertation-only
pass can be completed sooner, but the bilingual, administrative, figure-permission, and provenance
layers are submission dependencies rather than optional polish.

## 13. Acceptance checklist

The refactoring is complete only when:

- the current Ukrainian manuscript remains recoverable unchanged from a frozen commit;
- both abstracts, novelty/contribution claims, publication records, and general conclusions resolve
  through the same claim-to-artifact ledger as Chapter 4;
- every new English insertion is a coherent complete paragraph/subsection;
- Chapter 1 distinguishes spatial, temporal, activation, and routing interventions;
- long retained context/reach is never used as a synonym for long autoregressive forecast horizon;
- Chapters 1 and 2 preserve the scientifically useful Articles 1--3 material;
- dissertation-level tasks remain mapped to chapters, conclusions, and publications;
- Chapter 3 exactly matches implemented allocation policies and the released integrated
  architecture;
- Chapter 4 labels the 30k matrix as registered development evidence and restricts sealed-holdout
  confirmation to the selected contrasts actually evaluated;
- legacy E5/E6 or a separately registered selected-checkpoint evaluation carries any long-rollout
  stability claim;
- the old fixed-region/second-order prototype is not described as the final integrated model;
- every main figure has one question and every main table contains empirical results, except a
  deliberately retained single-purpose method-contract table;
- both training-seed points are visible and uncertainty units are explicit;
- no placeholder remains in a release candidate;
- no incomplete development result is described as confirmatory;
- retained-token, activation-memory, and total-VRAM ratios remain separate;
- no component ratios are multiplied;
- integrated component/baseline/checkpointing claims contract when their scheduled evidence cells
  are absent;
- routing inference policies are not conflated with mixed route-aware training;
- CEX, LVL, and OVR are described as their registered compound interventions;
- every current appendix and administrative/permission item has a verified migration destination;
- every figure/table is generated from an identified machine-readable artifact;
- formula/figure/table counts and callouts are regenerated, not edited manually in multiple places;
- the final build, link audit, provenance audit, and consistency gates pass.

## 14. Recommended default decisions requiring no immediate clarification

Unless the author changes them, proceed with these defaults when manuscript editing begins:

1. Keep the current four chapter titles and top-level semantic order.
2. Draft all new material in English now; perform final language harmonization later.
3. Keep E6 and the non-replicated E5a result in Chapter 4 because they sharpen the claim boundary.
4. Move EBT, raw-memory E8a, full sampler/codec robustness, and the old integration prototype to
   appendices or brief secondary paragraphs.
5. Use the new FRT/ALC/CEX/LVL/OVR/STS campaign as Chapter 4's primary registered development
   hierarchy evidence, with sealed confirmation restricted to selected contrasts.
6. Give the integrated model a full 4.8 section only after it passes correctness, matched-memory,
   and at least bounded quality gates; otherwise retain it as an explicitly negative composition
   study.
7. Keep publication assets from Articles 1--3 through the publication-content transfer ledger,
   additional-methods appendix, and figure-provenance appendix rather than deleting them.
8. Never prewrite successful numeric outcomes. Prewrite structure and conditional interpretation.
9. Preserve legacy E5/E6 as the current evidence for autoregressive stability. Add a v2
   selected-checkpoint rollout only as a separately registered inference evaluation; do not restart
   the 86-run training campaign.
10. Include activation checkpointing in the integrated profile if feasible; otherwise narrow the
    reversible comparison explicitly.

## 15. Independent-review incorporation matrix

| Blind-review finding | Incorporated correction | Plan location |
|---|---|---|
| P0-1 long context was conflated with long rollout | global terminology gate, revised H1, retained E5/E6, optional inference-only selected-checkpoint rollout | Sections 1, 6.2, 6.4, 7.1, 13--14 |
| P0-2 full frontier was called confirmed | five evidence-status labels and claim-to-artifact matrix; sealed confirmation limited to selected roles | Sections 1, 6.5, 13--14 |
| P1-1 front matter omitted | dedicated front-matter audit and linked claim placeholders | Sections 6.0, 12--13 |
| P1-2 dissertation tasks risked displacement | preserve six top-level tasks and nest experiment questions | Sections 6.1 and 13 |
| P1-3 Article 3 reduced to figures | retain its conceptual taxonomy and add claim-level content-transfer ledger | Sections 6.2--6.3 and 8.1 |
| P1-4 appendix/administrative material at risk | retain current Ukrainian appendix sequence and add old-to-new migration ledger | Section 11 |
| P1-5 integrated evidence not fully scheduled | evidence-cell ledger and automatic contraction of claims/section size | Sections 6.5 and 7.3 |
| P1-6 checkpointing missing | add same-model activation-checkpointing comparator or narrow claim | Sections 6.5, 7.3, 13--14 |
| P1-7 routing policy/training regime conflated | separate static/density/hybrid inference switches from mixed route-aware training | Sections 6.4--6.5 and 13 |
| P1-8 CEX/LVL/OVR estimands too broad | describe exact registered compound interventions | Sections 6.4--6.5, 7.1, and 13 |
| P1-9 placeholder registry incomplete | add MEC, rollout, gates, baselines, checkpointing, ablations, routing, and dataset-specific keys | Section 10 |
| P2-1 article attribution wrong | correct static-control material to Article 3/[A3] | Sections 6.2 and 8.1 |
| P2-2 architecture figure overloaded | conceptual figure plus adjacent contract box/optional method table | Section 6.4 |
| P2-3 zero-method-table rule too rigid | allow one single-purpose method-contract table | Sections 4.3 and 13 |
| P2-4 position normalization incomplete | add exact origin/range/time/scale/resolution/horizon/dtype contract | Section 6.4 |
| P2-5 provisional numbers lacked immutable provenance | remove transient numeric table; require dated hash-addressed snapshot for future values | Section 7.2 |
| P2-6 formula moves lacked dependency audit | add formula-to-code/result/reference dependency ledger | Section 9 |
| P2-7 schedule omitted bilingual/institutional work | add front-matter/migration and bilingual/institutional phases; revise estimate | Section 12 |
