# Path B dissertation outline and asset plan (archived reorder alternative)

**Status:** superseded as the active drafting structure on 2026-09-01. The conservative,
evidence-triggered structure is defined in
`docs/plans/CONDITIONAL_DISSERTATION_MIGRATION_PLAN_2026-09-01.md`. Keep this document only as the
previous full-reorder alternative; do not renumber or migrate chapters from it before the Article
5 promotion gate.

## Narrative

The dissertation is organized by the resource problem, not by publication chronology. Chapter 1 defines hierarchy and separates retained state, activations, and spatial allocation. Chapter 2 performs the main controlled hierarchy investigation. Chapter 3 shows why context hierarchy is insufficient and presents the two published mechanisms for activation memory and spatial information density. Chapter 4 integrates the mechanisms only if the Article 5 gates establish a useful OpenSTL quality--memory Pareto point.

## Chapter 1 -- Hierarchy and memory in long-horizon video prediction

**Role:** define the system, resource vocabulary, hierarchy manifestations, literature gap, and falsifiable dissertation hypotheses.

### 1.1 Long-horizon prediction as partial observation with feedback

#### 1.1.1 State, observation, action, and self-fed prediction

Define the prediction boundary and explain why generated frames become future inputs.

- Figures: system-boundary diagram; compact teacher-forced/self-fed timeline.
- Numbered equations: state/observation relation; autoregressive future distribution.

#### 1.1.2 Consistency across appearance, identity, trajectory, time, and levels

Replace the current generic illustration gallery with one failure taxonomy tied to later metrics.

- Figures: one consistency taxonomy.
- Numbered equations: none.

#### 1.1.3 Memory as several resources

Separate retained context, training activations, parameters/optimizer, and temporary workspace. Explain why a gain in one is not automatically a gain in another.

- Figures: one memory-location/dependency diagram.
- Numbered equations: grouped memory decomposition.

### 1.2 Manifestations of hierarchy

#### 1.2.1 Spatial representational hierarchy

Explain coarse full-field representation, fine detail, local/global processing, and nonuniform information density.

- Figures: coarse-base/fine-detail representation.
- Numbered equations: spatial token count only if reused in Chapter 2.

#### 1.2.2 Temporal retention hierarchy

Explain multiple temporal rates, anchors, reach, and the distinction between old coarse evidence and recent fine evidence.

- Figures: multi-rate temporal timeline.
- Numbered equations: nominal reach definition.

#### 1.2.3 Computational hierarchy and reversibility

Position reversible reconstruction as a training-memory mechanism, distinct from retained context compression.

- Figures: forward storage versus reverse reconstruction concept.
- Numbered equations: none; details belong in Chapter 3.

#### 1.2.4 Spatial--temporal budget exchange

State the new central insight: reducing spatial detail per retained time point can buy more temporal coverage under a global token budget, while temporal abstraction decides which ages require which resolution.

- Figures: a token-budget balance showing spatial savings converted into temporal reach.
- Numbered equations: one global spatiotemporal budget constraint.

#### 1.2.5 Why transformer architectures make hierarchy critical

Relate token count to retained-state and attention/workspace costs without implying that all costs scale identically.

- Figures: dense versus hierarchical scaling sketch.
- Numbered equations: grouped token/storage/attention scaling.

### 1.3 Existing approaches and unresolved gaps

#### 1.3.1 Long context, recurrence, retrieval, sparse attention, and state-space models

#### 1.3.2 Multiscale and hierarchical video representations

#### 1.3.3 Checkpointing and reversible training

#### 1.3.4 Adaptive tokenization and information-density routing

Use Article 1 review material selectively, then update the hierarchy/memory literature for the dissertation. The section compares intervention interfaces rather than cataloguing models.

- Figures: one literature-positioning map.
- Numbered equations: none.

### 1.4 Research gap, objective, hypotheses, and contributions

#### 1.4.1 Missing controlled evidence

Identify unaligned budgets, targets, temporal reach, seed coverage, and memory definitions.

#### 1.4.2 Central objective and resource-specific hypotheses

State hierarchy, reversibility, routing, and integration as separate testable hypotheses.

- Figures: claim-to-evidence map.
- Numbered equations: constrained optimization statement only if used by Chapters 2 and 4.

#### 1.4.3 Contribution and publication map

Map Articles 1 and 4 to Chapters 1--2, Articles 2 and 3 to Chapter 3, and conditional Article 5 to Chapter 4.

### 1.5 Chapter conclusions

**Target:** 5--7 numbered equations, 6--8 figures, no main-text table unless institutionally required.

## Chapter 2 -- Hierarchical context allocation under fixed memory budgets

**Role:** central architectural and empirical chapter; basis of the redesigned Article 4.

### 2.1 Problem definition and allocation design space

#### 2.1.1 Context levels, anchors, costs, and reach

Define every level by spatial token cost, temporal stride, slot count, anchor set, and byte cost.

- Figures: level/anchor diagram.
- Numbered equations: package, token/byte budget, and reach.

#### 2.1.2 Allocation policies at fixed budget

Define flat, balanced, linear-age, geometric-age, and reach-oriented allocations with a common target.

- Figures: equal-area allocation policy diagram.
- Numbered equations: budget constraint and policy normalization.

#### 2.1.3 Nested versus exclusive temporal support

Define whether fine tokens supplement or replace coarse tokens for the same time point.

- Figures: direct nested/exclusive timeline.
- Numbered equations: level-support set relation.

#### 2.1.4 Spatial quota versus temporal reach

Vary the fine spatial quota and spend the saved tokens on temporal history. Separate fixed/oracle allocation from learned routing.

- Figures: spatial-quota/temporal-reach design grid.
- Numbered equations: global quota and purchased-reach relation.

#### 2.1.5 Research questions and registered hypotheses

State budget frontier, policy, level count, overlap, and spatial--temporal exchange as distinct questions.

### 2.2 Hierarchical predictor architecture

#### 2.2.1 Multi-level codec and invariant target interface

Show tensor dimensions, frozen/trainable boundaries, and a target that does not change between context arms.

- Figures: codec with tensor shapes.
- Numbered equations: codec interface and executed reconstruction constraint.

#### 2.2.2 Context assembly and coordinate/level/time metadata

Describe token ordering, anchors, actions, positional semantics, and overlap.

- Figures: token-sequence schematic.
- Numbered equations: token and metadata construction.

#### 2.2.3 Shared prediction core

Describe one predictor for all context levels and target tokens; cite standard transformer machinery.

- Figures: primary Article 4 architecture diagram.
- Numbered equations: conditional predictor and executed loss.

#### 2.2.4 Autoregressive update and state accounting

Define one prediction step, per-level update, state bytes, and inference horizon.

- Figures: ring-buffer/timeline.
- Numbered equations: grouped stream update and exact retained-state cost.

### 2.3 Toroidal Re-entry Benchmark and evaluation

#### 2.3.1 Mechanism, data splits, and correctness oracles

Present TRB as a static toroidal world observed through a controlled moving viewport; retain `LoopMom` only in legacy artifact identifiers.

- Figures: world/viewport diagram; one sequence strip.
- Numbered equations: viewport position/rendering relation.

#### 2.3.2 Observability and pair-specific populations

Define which old evidence is observable to each package.

- Figures: pair-specific reach bands.
- Numbered equations: coverage set and comparison population.

#### 2.3.3 Metrics and seed-aware inference

Use seed as the replication unit and clip resampling only as conditional diagnostics.

- Figures: none.
- Numbered equations: primary paired effect and across-seed estimator; detailed bootstrap in appendix.

### 2.4 Experiment programme

#### 2.4.1 Invariant protocol and semantic experiment names

#### 2.4.2 Ten-point context-budget frontier

#### 2.4.3 Same budget, different level allocation

#### 2.4.4 Fixed fine detail with expanding coarse history

#### 2.4.5 One, two, and three levels at fixed budget

#### 2.4.6 Nested versus exclusive overlap

#### 2.4.7 Spatial-detail/temporal-reach surface

#### 2.4.8 Bounded head, sampler, and codec robustness

- Figures: one experiment-design matrix only; no results in this section.
- Tables: full configuration registry in appendix/JSON, not main text.

### 2.5 Results

#### 2.5.1 Quality--reach--memory frontier

- Figures: seed-visible frontier; separate measured-memory Pareto plot.
- Tables: locked four-seed results.

#### 2.5.2 Allocation-policy effect

- Figures: performance versus budget by policy.
- Tables: primary contrasts only.

#### 2.5.3 Coarse expansion and number of levels

- Figures: one direct curve/comparison per question.

#### 2.5.4 Overlap effect

- Figures: paired nested/exclusive result.

#### 2.5.5 Spatial--temporal allocation surface

- Figures: heat map/small multiples with one role; saved spatial tokens mapped to purchased temporal reach.
- Tables: three locked representative points.

#### 2.5.6 Robustness, negative evidence, and failure cases

- Figures: seed heterogeneity and, only if distinct, one robustness result.

### 2.6 Operating region and validity limits

### 2.7 Chapter conclusions

**Target:** 14--18 numbered equations, 10--12 figures, 3--5 main-text result tables.

## Chapter 3 -- Activation memory and spatial information density

**Role:** transfer Articles 2 and 3 after the central hierarchy study and show which costs context allocation does not remove.

### 3.1 Limits not removed by context hierarchy

#### 3.1.1 Activation storage in long unrolled training

#### 3.1.2 Uneven spatial information density

- Figures: one two-bottleneck transition from Chapter 2.
- Numbered equations: refined memory decomposition.

### 3.2 Reversible temporal computation

#### 3.2.1 TempVerseFormer architecture and reversible boundary

- Figures: published architecture and reversible-block figures, preserving internal pixels.
- Numbered equations: one grouped additive-coupling forward/inverse system.

#### 3.2.2 Reverse replay, stochastic state, and numerical stability

- Figures: published backward/replay or chaining diagram.
- Numbered equations: replay relation and memory--compute trade-off.

#### 3.2.3 Controlled and OpenSTL results

- Figures: selected published quality summary; memory versus horizon; runtime versus horizon.
- Tables: published empirical quality and memory results.

#### 3.2.4 Applicability and limitations

State explicitly that reversible computation saves activation storage and does not itself improve prediction quality.

### 3.3 Hierarchical byte-latent spatial modeling

#### 3.3.1 V-BLT local--global--local architecture

- Figures: published architecture; optional author tensor-flow redraw.
- Numbered equations: model interface and byte loss.

#### 3.3.2 RoPE-ND

- Figures: published causal mask or compact axis/subspace diagram.
- Numbered equations: one multi-axis rotation equation.

#### 3.3.3 Predictive entropy as spatial-complexity diagnostic

- Figures: entropy pipeline plus published 2-D/3-D maps.
- Numbered equations: entropy.

#### 3.3.4 Results and implemented/proposed boundary

- Figures: published point-estimate visualization if useful.
- Tables: 2-D/3-D published result tables.
- Boundary: fixed patches and entropy diagnostics are established; learned adaptive partitioning is not an Article 3 result.

### 3.4 Synthesis: resource and interface map

Relate retained history, reconstructed activations, and allocated spatial detail without multiplying incompatible memory ratios.

- Figures: one component-to-resource interface map.
- Numbered equations: none.

### 3.5 Chapter conclusions

**Target:** 12--16 numbered equations, 8--10 figures, 4--5 result tables.

## Chapter 4 -- Integrated memory-efficient hierarchical video predictor

**Role:** conditional Article 5 chapter. It is promoted from feasibility to a main contribution only after the correctness and OpenSTL Pareto gates pass.

### 4.1 Requirements and backbone selection

#### 4.1.1 Pareto selection from Chapter 2

Choose the hierarchy configuration by a frozen rule.

- Figures: Chapter 2 frontier with the selected point highlighted.
- Numbered equations: selection criterion if formalized.

#### 4.1.2 Integration hypotheses and non-inferiority margins

Define independent expectations for hierarchy, reversibility, routing, and their composition.

### 4.2 Architecture and state contract

#### 4.2.1 Fixed-width multi-level state

Define coarse tokens, fixed-quota detail tokens, coordinates, time/age, scale/level, masks, and route metadata.

- Figures: stored-state layout with bytes.
- Numbered equations: pack/unpack interface and exact state cost.

#### 4.2.2 Coordinate-aware positional encoding

Replace static slot positions for moving detail tokens with spatial coordinates, time/age, scale, and level using coordinate RoPE or Fourier features.

- Figures: coordinate/scale/time positional axes.
- Numbered equations: compact coordinate positional transform.

#### 4.2.3 Density-aware detail routing

Place top-k/quantization outside the reversible boundary. Use a fixed tensor quota and explicitly store routing metadata. Include route stability.

- Figures: score, quota, pack, and metadata path.
- Numbered equations: quota selection and optional stability regularizer.

#### 4.2.4 Predictor with reversible internal blocks

Keep the external prediction transition identical in ordinary and reversible modes. Reconstruct only internal block activations.

- Figures: end-to-end architecture; reversible-block inset.
- Numbered equations: external transition and grouped reversible coupling.

#### 4.2.5 Coarse-base/fine-residual decoder

Scatter fine residuals by coordinates over a full coarse reconstruction and define overlap/seam behavior.

- Figures: decoder compositing schematic.
- Numbered equations: compositing rule.

#### 4.2.6 Training and inference paths

Show which components are active and which memory pool is affected. An inference-only router is a control unless the model was trained for changing routes.

- Figures: two-lane training/inference diagram.
- Numbered equations: complete training objective.

### 4.3 OpenSTL-compatible evaluation

#### 4.3.1 Moving MNIST, TaxiBJ, and KTH boundaries

#### 4.3.2 Matched internal and external baselines

#### 4.3.3 Quality, retained state, inference peak, training peak, runtime, and Pareto endpoints

#### 4.3.4 Correctness, route-stability, and reversible-parity tests

- Tables: protocols/configs in appendix; only empirical results in main text.
- Numbered equations: memory endpoints and non-inferiority/Pareto criterion.

### 4.4 Component ablations

#### 4.4.1 Hierarchy versus flat retained state

#### 4.4.2 Reversible versus ordinary internal blocks

#### 4.4.3 Static, deterministic, and learned routing

#### 4.4.4 Full B0--B5 composition

- Figures: separate state-memory, training-memory, routing, and full Pareto plots.
- Tables: exact quality, memory, runtime, parameters, and seeds.

### 4.5 Dataset-level comparisons

#### 4.5.1 Moving MNIST

#### 4.5.2 TaxiBJ

#### 4.5.3 KTH

#### 4.5.4 Cross-dataset Pareto analysis

### 4.6 Robustness, failure analysis, and validity limits

### 4.7 Chapter conclusions

**Target:** 10--14 numbered equations, 8--10 figures, 4--6 result tables.

## Main-text budget

| Part | Equations | Figures | Tables |
|---|---:|---:|---:|
| Chapter 1 | 5--7 | 6--8 | 0--1 |
| Chapter 2 | 14--18 | 10--12 | 3--5 |
| Chapter 3 | 12--16 | 8--10 | 4--5 |
| Chapter 4 | 10--14 | 8--10 | 4--6 |
| Total target | 41--55 | 32--40 | 11--17 |

The top of the equation range should be used only when an equation is cited later or defines an executed model/estimand. The current manuscript's standard derivations and administrative registries move to appendices.
