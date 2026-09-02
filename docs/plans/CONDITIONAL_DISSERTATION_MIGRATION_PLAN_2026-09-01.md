# Conditional dissertation migration plan

> Update (2 September 2026): the two-new-article assumption is superseded. Preserve the existing
> Ukrainian manuscript tree and Chapters 1--2, draft new insertions in English, and treat the TRB
> hierarchy campaign plus integrated OpenSTL model as two evidence blocks of one article. Chapter
> 3/4 changes remain evidence-triggered; do not rebuild the dissertation from scratch.

**Decision:** preserve the current Chapter 1 and Chapter 2 order while Articles 4 and 5 are being
tested. Reorganize only as far as the new evidence justifies.

## 1. Why this replaces an immediate full restructure

The current first two chapters already provide a coherent systems analysis and the published
TempVerseFormer/V-BLT mechanisms. Moving them before the new experiments are known would create
substantial rewrite risk without improving the evidence. The weak part is not their existence; it
is the present architecture/experiment organization in Chapters 3 and 4.

The migration therefore has two evidence stages:

1. Article 4 replaces the hierarchical architecture and controlled-experiment core.
2. Article 5 is promoted to a distinct final technical chapter only if its integrated architecture
   is correct, memory-efficient, and competitive. Otherwise it remains a bounded feasibility and
   limitations section.

The new manuscript is written directly in English. The current Ukrainian manuscript remains a
frozen fallback and provenance source.

## 2. Stable content before the experiment gates

### Chapter 1 -- Systems analysis and requirements for consistent long-horizon video prediction

Keep the current chapter order and most of its content. Make only surgical changes now:

- define retained context, training activations, model/optimizer state, and temporary workspace as
  distinct memory pools;
- introduce spatial hierarchy, temporal hierarchy, and computational/reversible hierarchy as
  distinct manifestations;
- add the spatial--temporal budget exchange: reducing fine spatial payload can purchase a longer
  coarse history or detail coverage over more times;
- state Article 4 ratios as hypotheses, not findings;
- update the publication map so Articles 1 and 4 jointly support the hierarchy framing.

Keep or transfer from Articles 1--3: the consistency taxonomy, system diagrams, hierarchy examples,
and literature maps that already have clear provenance. Replace decorative galleries with one-role
figures.

Formula policy: retain only the state/observation definition, autoregressive factorization, one
memory decomposition, one global spatiotemporal budget constraint, and the few scaling relations
used later. Target: 5--8 numbered formulas.

### Chapter 2 -- Methods for removing scaling limits in hierarchical prediction

Keep the present semantic role and the published TempVerseFormer and V-BLT material:

- controlled experimental foundation from Articles 2 and 3;
- reversible temporal computation and its activation-memory boundary;
- V-BLT, RoPE-ND, and predictive entropy as a spatial-complexity diagnostic;
- explicit implemented-versus-proposed boundary for adaptive partitioning;
- a revised closing section explaining how these mechanisms become candidates for Article 5.

Transfer published architecture figures and result tables directly through the asset ledger.
Do not redraw them unless readability or permission requires an author-derived schematic.

Formula policy: one grouped reversible forward/inverse system, one memory--compute relation, the
executed V-BLT objective, RoPE-ND, and entropy. Derivations that are not referenced by an
experiment move to an appendix. Target: 10--14 numbered formulas.

## 3. Chapter 3 after Article 4

# Chapter 3 -- Hierarchical context allocation under fixed memory budgets

This chapter replaces the current hierarchical-model description and absorbs the controlled
Article 4 results. Architecture and evidence remain in one chapter so readers do not cross a
chapter boundary to understand a configuration.

## 3.1 Context allocation problem

### 3.1.1 Context packages, token cost, temporal reach, and memory endpoints

Define coarse/fine/top payloads, physical coordinates, time/age, level, overlap, analytic token
cost, retained-state bytes, and measured CUDA endpoints.

- Figures: one level/cost/reach schematic.
- Formulas: package definition; exact token/byte cost; reach.

### 3.1.2 Spatial and temporal allocation policies

Define Flat recent, Balanced, Fine-heavy, Coarse-heavy, Linear-age, Geometric-age, and
Reach-oriented policies.

- Figures: one equal-area policy diagram.
- Formulas: fixed-budget constraint; linear/geometric normalized schedules.

### 3.1.3 Nested, exclusive, and residual support

- Figures: one direct three-row timeline.
- Formulas: support-set relations only.

### 3.1.4 Spatial quota and temporal detail coverage

Explain how 256 fine cells cover 4, 8, 16, or 32 time positions while a coarse base provides
full-horizon support.

- Figures: one spatial-quota/temporal-coverage grid.
- Formulas: quota and purchased-detail-coverage relation.

## 3.2 Predictor architecture

### 3.2.1 Frozen multilevel codec and invariant prediction target

- Figures: codec with tensor shapes and frozen/trainable boundaries.
- Formulas: executed encoder/decoder interface and target definition.

### 3.2.2 Context packer and physical token identity

Explain local gather indices, absolute time, cell coordinates, level, role, overlap, and exact
ordering. This section includes the corrected selected-token contract.

- Figures: token sequence and metadata schematic.
- Formulas: token/metadata tuple and deterministic packing map.

### 3.2.3 Shared transformer prediction core

- Figures: one Article 4 architecture diagram.
- Formulas: conditional predictor and executed training loss; standard attention need not be
  re-derived unless modified.

### 3.2.4 Autoregressive update and retained-state accounting

- Figures: ring-buffer/timeline.
- Formulas: state update and exact retained-state cost.

## 3.3 Toroidal Re-entry Benchmark and protocol

### 3.3.1 World, viewport, actions, and re-entry mechanism

- Figures: toroidal world/viewport; one exit/re-entry sequence.
- Formulas: viewport position and rendering relation.

### 3.3.2 Development, validation, and sealed confirmation sets

Explain legacy provenance limits, the new 2,700/300 split, exclusion of the old test file, and
one-time use of the new confirmation set.

- Tables: one dataset/split manifest table.
- Formulas: none.

### 3.3.3 Metrics, memory measurement, and two-seed reporting

- Figures: none.
- Formulas: primary paired effect; do not present a two-seed 95% interval.

## 3.4 Registered experiment programme

### 3.4.1 Correctness and convergence gates (A4-COR)
### 3.4.2 Six-point context-budget frontier (A4-FRT)
### 3.4.3 Allocation at 1,792 tokens (A4-ALC)
### 3.4.4 Fixed detail with coarse expansion (A4-CEX)
### 3.4.5 One, two, and three levels (A4-LVL)
### 3.4.6 Temporal overlap (A4-OVR)
### 3.4.7 Spatial-detail/temporal-coverage surface (A4-STS)

- Figures: one experiment dependency map, not a result collage.
- Tables: exact config registry belongs in the appendix/reproducibility bundle.

## 3.5 Results

### 3.5.1 Quality--memory frontier
### 3.5.2 Allocation policy at fixed budget
### 3.5.3 Value of coarse expansion
### 3.5.4 Level-count and overlap effects
### 3.5.5 Spatial-detail/temporal-coverage surface
### 3.5.6 Mechanism diagnostics, failures, and robustness

Each subsection has one principal figure and at most one result table. Figures expose both seeds.
Tables contain results only; configuration metadata is not mixed into metric cells.

## 3.6 Operating region and threats to validity

Separate TRB mechanism validity, codec dependence, two-seed limitations, and lack of natural-video
external validity.

## 3.7 Chapter conclusions

Target for Chapter 3: 12--16 numbered formulas, 9--12 figures, and 3--4 main result tables.

## 4. Chapter 4 decision after Article 5

### Promotion gate

Article 5 becomes the main subject of Chapter 4 only if all conditions hold:

1. coordinate/route identity, causal, pack/unpack, and reversible output/gradient parity tests pass;
2. dynamic routing reduces the actual packed state rather than only reweighting fixed tokens;
3. reversible internal blocks materially reduce measured full training peak, with component and
   end-to-end denominators both reported;
4. every retained component has a non-negative role-specific ablation result, or a documented
   trade-off that the full model needs;
5. the full model is on or near the internal ablation Pareto frontier;
6. on at least two OpenSTL datasets, the full model is within the predeclared quality tolerance of
   matched internal and rerun external baselines while producing a material measured memory gain;
7. both training seeds reproduce the direction of the primary effect.

The Article 4 decision to use TRB only does not replace Article 5's separate OpenSTL external
validation requirement.

### Branch S -- Article 5 succeeds

# Chapter 4 -- Integrated memory-efficient hierarchical video prediction

## 4.1 Composition problem and resource-specific hypotheses

### 4.1.1 What hierarchy, reversibility, and routing each change
### 4.1.2 Canonical hierarchical state and fixed-width contract
### 4.1.3 Acceptance gates and fair-comparison rules

- Figures: component-to-memory-endpoint map; canonical state schema.
- Formulas: state tuple; component-wise memory decomposition.

## 4.2 Coordinate-aware dynamic regioning

### 4.2.1 Physical spatial, temporal, scale, and level coordinates
### 4.2.2 Router, fixed-quota packer, and non-reversible boundary
### 4.2.3 Route stability, hysteresis, and inference-time operation
### 4.2.4 Coarse base and scattered fine residual decoder

- Figures: coordinate encoding; routing/packing flow; decoder scatter/composition.
- Formulas: coordinate encoding; top-k/quota contract; stability loss if executed.

## 4.3 Reversible predictor core

### 4.3.1 Same external transition with ordinary and reversible internal blocks
### 4.3.2 Reverse replay, randomness, and numerical error
### 4.3.3 Activation-memory accounting and compute overhead

- Figures: reversible boundary and backward replay.
- Formulas: grouped coupling/inverse; parity relation; activation-memory endpoint.

## 4.4 Integrated architecture

### 4.4.1 End-to-end encoder--router--packer--predictor--decoder flow
### 4.4.2 Positional and causal semantics under route changes
### 4.4.3 Training, autoregressive inference, and inference-time routing switch

- Figures: primary integrated architecture with tensor shapes; rollout state transition.
- Formulas: executed end-to-end objective and state update only.

## 4.5 OpenSTL experimental protocol

### 4.5.1 Datasets, splits, and matched budgets
### 4.5.2 Internal six-arm component ablation
### 4.5.3 External ConvLSTM/SimVP/TAU comparisons
### 4.5.4 Memory, quality, runtime, and route-stability metrics

- Tables: protocol; exact model sizes/training time; no copied external numbers presented as direct
  comparisons.

## 4.6 Results

### 4.6.1 Hierarchy retained-state effect
### 4.6.2 Reversibility activation-memory effect
### 4.6.3 Static versus dynamic regioning
### 4.6.4 Full composition and Pareto frontier
### 4.6.5 Dataset transfer and failure cases

- Figures: role-specific memory bars; component ablation; one quality--memory Pareto plot; route
  stability visualization.
- Tables: locked two-seed internal ablation and rerun external baselines.

## 4.7 Applicability, limitations, and chapter conclusions

Target for successful Chapter 4: 10--14 numbered formulas, 8--11 figures, and 3--5 result tables.

### Branch F -- Article 5 does not pass

# Chapter 4 -- Experimental synthesis, operating limits, and integration feasibility

Keep a shorter final chapter with:

1. cross-resource synthesis of Articles 2--4;
2. selected robustness and negative evidence from the current Chapter 4;
3. the tested integrated state/positional/reversible contract;
4. failed or mixed Article 5 ablations, clearly labeled as feasibility evidence;
5. a system-level method-selection rule and threats to validity.

The integrated prototype does not receive a superiority chapter, and no component ratios are
multiplied. Useful negative results remain publishable and may support an Article 5 focused on
composition limits rather than a claimed winning architecture.

## 5. Asset and text transfer policy

- Articles 1--3 remain the primary provenance sources for their published figures, tables, and
  established results.
- Existing dissertation images are transferred only through
  `docs/dissertation/v2_english/CONTENT_TRANSFER_LEDGER.md` with source, destination, permission,
  and evidential role.
- New Article 4/5 figures are generated from immutable result manifests and have one analytical
  role each.
- Text is rewritten in one English dissertation voice. Published claims and numeric results keep
  explicit attribution; reused published pixels are not silently redrawn.
- Formula numbering is rebuilt only after the chapter branch is chosen. A formula remains numbered
  only if it defines an implemented operation, a measured quantity, or a relation used by a later
  argument.

## 6. Decision dates

- After Article 4: freeze the Chapter 3 evidence structure and begin English drafting.
- After Article 5 correctness/memory gates: decide whether external comparisons are warranted.
- After two-dataset Article 5 confirmation: choose Branch S or Branch F and only then finalize
  Chapter 4 numbering, cross-references, figures, and conclusions.
