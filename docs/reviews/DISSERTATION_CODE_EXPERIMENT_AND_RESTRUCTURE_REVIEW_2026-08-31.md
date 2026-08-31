# Critical review of the dissertation, codebase, and next experimental programme

**Date:** 31 August 2026
**Review target:** current Ukrainian dissertation sources, the hierarchy/LoopMom evidence programme, the integrated prototype, and the repository state
**Recommended writing language for the replacement:** English
**Overall recommendation:** preserve the current dissertation and evidence chain as a frozen fallback, but develop the new hierarchy and integration studies in a clean branch/worktree and a new English manuscript tree. Do not rewrite the current evidence-producing code in place.

> **Path B execution addendum, 31 August 2026.** The author accepted Path B and the immediate stabilization actions. The old compatibility figure filenames were restored alongside the normalized Ukrainian filenames, after which the current Ukrainian audit, compact-English audit, manuscript-finalization check, and text-consistency gate all passed. The Ukrainian/old-English translation-parity check remains intentionally non-authoritative because a new English v2 manuscript tree will replace the old compact English tree. The executable continuation plan and separate Article 4/5 schedules are in `docs/plans/PATH_B_MASTER_EXECUTION_PLAN_2026-08-31.md`, `docs/plans/ARTICLE4_ONE_WEEK_EXPERIMENT_PLAN_2026-08-31.md`, and `docs/plans/ARTICLE5_ONE_WEEK_INTEGRATION_EXPERIMENT_PLAN_2026-08-31.md`.

## 1. Executive verdict

The dissertation has a defensible scientific core, but its current organization does not yet express the strongest version of the thesis.

The strongest claim is the controlled, budget-aware hierarchy result: **hierarchy can buy temporal reach, while dense flat context can recover fine fidelity once reach is sufficient**. The supporting evidence is unusually well documented. The repository contains immutable result summaries, configuration hashes, protected-set controls, completion validators, and explicit restrictions on what may be claimed. The 99-test suite passes, the scientific configuration validator passes, and the public package can be built and imported from a clean wheel.

The main problems are different:

1. The thesis is organized around the historical order of several projects more than around one architectural investigation.
2. The central hierarchy method and its experiments are split between Chapters 3 and 4, while auxiliary methods occupy Chapter 2 before the main contribution appears.
3. The manuscript contains too many numbered equations and explanatory figures. Many standard, derived, or hypothetical equations are given the same visual status as the original model definitions and measured estimands.
4. The current hierarchy frontier has unequal seed coverage across budgets. Several confidence intervals quantify clip-level uncertainty conditional on one trained model, not variation across training initializations.
5. The current integrated prototype is a valid feasibility demonstration, but it is not ready to support a fifth article or a confirmatory dissertation chapter. Its quality result is single-seed, its adapter selection touches the canonical Moving-MNIST test set, and its reversible long-rollout wrapper changes the forward dynamics rather than merely changing how the same dynamics are differentiated.
6. The current working tree is in the middle of a figure/formula renumbering operation. Ukrainian and English source trees are no longer structurally aligned, and old English image links currently point to files deleted by the Ukrainian renumbering.

I therefore agree with the proposed high-level chapter order:

- Chapter 1: the problem, hierarchy, memory, consistency, and literature;
- Chapter 2: the main hierarchical-context architecture and a redesigned hierarchy investigation;
- Chapter 3: activation-memory and spatial-information mechanisms, based on TempVerseFormer and V-BLT;
- Chapter 4: a genuinely integrated architecture and OpenSTL evaluation, but only after the integration passes explicit gates.

I do **not** recommend replacing the current dissertation immediately. The safer strategy is to keep it buildable as a fallback while the new Chapter 2 and Article 4 are developed. Chapter 4 and Article 5 should be conditional: if the integrated model fails to establish a useful quality–memory Pareto point, the dissertation should retain a bounded integration/feasibility section rather than claim a new superior architecture.

## 2. What was inspected and verified

The review used the current Ukrainian source tree and the provenance guide as the primary manuscript sources:

- `docs/dissertation/condensed_translated_humanized_ukrainian/`;
- `docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md`;
- the central article at `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`;
- experiment configurations under `experiments/configs/`;
- frozen results under `artifacts/`;
- the main WindowDiT implementation in `hvp_rev/rhvd.py`;
- the integrated prototype in `hvp_rev/fixed_region_revformer.py`, `hvp_rev/packed_simvp.py`, and `hvp_rev/reversible_rollout.py`;
- the integration training and profiling scripts;
- the LoopMom generator, data card, manifest, and tests;
- the repository history, branch state, ignored storage, and release checks.

Verification results on the current worktree:

- `venv\Scripts\python.exe -m pytest`: **99 passed**;
- `scripts/validate_scientific_configs.py`: **passed**, validating 10 configs and the fail-closed ledger;
- `scripts/check_text_consistency.py --gate`: **passed**;
- `scripts/check_rft_consistency.py`: **passed**;
- `scripts/check_manuscript_finalization.py`: **passed** for the checks it performs;
- `scripts/release_check.py`: **passed**, including wheel build and clean import;
- `scripts/audit_ukrainian_translation_parity.py`: **failed** because the current Ukrainian tree has intentionally diverged from the older English compact tree;
- `scripts/audit_compact_dissertation.py`: **failed** on five English image links whose old figure files are currently deleted/renamed in the working tree.

The passing release check should therefore not be read as “the current Ukrainian manuscript is fully synchronized.” It assembles and checks a different English source path and does not close the Ukrainian/English parity failures.

## 3. Current state dashboard

### 3.1 Manuscript inventory

| Part | Words | Numbered formulas | Numbered figures | Numbered tables | Assessment |
|---|---:|---:|---:|---:|---|
| Chapter 1 | 8,555 | 14 | 25 | 2 | Conceptually strong, visually overloaded |
| Chapter 2 | 9,418 | 42 | 19 | 10 | Two published methods plus extensions are crowded together |
| Chapter 3 | 10,041 | 53 | 19 | 3 | Main architecture, evaluation theory, statistics, and integration are over-compressed |
| Chapter 4 | 8,877 | 5 | 14 | 7 | Results are evidence-rich but combine too many roles |
| Appendices | 6,840 | 0 | 4 | 18 | Valuable provenance material; suitable place for configs and derivations |

The current main text has 114 numbered equations and 77 numbered figures; the appendices add four numbered figures and 18 tables. The discrepancy with earlier descriptions of 112 equations, 42 tables, or 80 figures is a sign that the inventory changed during recent normalization and should be regenerated automatically after the structure is frozen.

### 3.2 Code and repository inventory

- The current branch is `research/ebt-hierarchical-video`.
- It is one commit ahead of its remote tracking branch and 302 commits ahead of `main`, with no commits unique to `main` after the merge base.
- The repository contains about 854 tracked files and approximately 14,243 lines across top-level `hvp_rev` Python modules.
- `hvp_rev/rhvd.py` is approximately 1,391 lines and combines the arm registry, architecture, batching, training, rollout, evaluation, and CLI behavior.
- Large data, checkpoints, run directories, and the local feasibility store are correctly ignored, but many generated document binaries and historical figures are tracked. Git object storage is already substantial.
- The worktree currently contains modified manuscript sources, modified figure generators/provenance files, deleted old figure names, and untracked new names. It is not a clean release boundary.

## 4. What is already strong and should be preserved

### 4.1 The central scientific distinction is good

The dissertation now distinguishes three resources that are often incorrectly collapsed into “memory”:

- retained context/state during prediction;
- activations required by backward propagation;
- spatial representational capacity and its allocation.

This is the correct conceptual foundation. The revised introduction already explains that reducing one resource does not imply reduction of the others. The new dissertation should make this distinction earlier and more visually, but it should not discard it.

### 4.2 The main hierarchy claim has a defensible boundary

The current evidence does not claim universal superiority. It records confirmed hierarchy-favouring points at 896 and 1,152 context tokens, an unresolved point at 1,792, and flat-context-favouring points at 2,432 and 3,072. It explicitly avoids fitting a continuous crossing. This restraint is scientifically valuable.

The strongest arithmetic result is also correctly bounded: 48 coarse plus 6 or 2 fine frames require 1,152 or 896 tokens, versus 3,072 for 48 fine frames. This is a context-token ratio, not a whole-GPU-memory ratio and not an equal-quality statement.

### 4.3 Provenance and experiment closure are unusually good

The codebase contains:

- purpose-locked datasets and access logs;
- immutable artifacts with hashes;
- explicit comparison licences and claim boundaries;
- completion validators;
- outcome-independent launch authorization;
- cluster-aware statistics;
- records of negative and unresolved findings.

These controls should be reused in the new code rather than replaced. They are a major asset of the project.

### 4.4 The LoopMom generator is coherent for its intended mechanism

The generator produces a `192×64` toroidal world viewed through a moving `64×64` window, records actions and absolute positions, and preserves the legacy pixel/RNG path when adding metadata. The confirmation manifest is hash-pinned, the action/position consistency gate passes, and the generator test verifies pixel and RNG identity between legacy and metadata paths.

It is valid as a controlled **action-conditioned re-entry instrument**. It is not evidence of natural-video generality, object dynamics, or unconditional future prediction. The world itself is static; the apparent motion comes primarily from the viewport. This limitation is acceptable if it remains explicit.

## 5. Critical findings

### 5.1 P0: the current integration quality result is not confirmatory

`scripts/train_fixed_region_adapter.py` loads the canonical Moving-MNIST test set, evaluates the adapter on it every evaluation interval, and chooses the best checkpoint from those values. It then evaluates the selected model on the same test set again. The relevant logic is at lines 107–108 and 146–158.

That means the adapter used by the later packed-SimVP fine-tuning has already been selected with test feedback. The reported MSE 28.72 on 10,000 test clips can remain a development/feasibility number, but it must not be used as a clean benchmark result for Article 5.

Required correction for the new study:

1. create immutable train/validation/test splits before any run;
2. train and select the codec, adapter, router, and predictor using train/validation only;
3. open the test set once for the frozen selected configuration;
4. use the same rule for every baseline;
5. store the split manifest and all checkpoint-selection decisions.

### 5.2 P0: the reversible wrapper changes the predictive dynamics

The current long-rollout wrapper implements

\[
p' = q, \qquad q' = p + F(q),
\]

not ordinary repeated prediction `q' = F(q)`. The implementation is explicit in `hvp_rev/reversible_rollout.py`. Starting from `p=0` makes only the first step equal to `F(q)`; later steps follow second-order additive dynamics.

This is mathematically reversible and the gradient-parity tests are useful. However, it is not a memory-only execution change around the same trained autoregressive model. It introduces a different recurrence. The current 20- and 40-frame profiles use this recurrence and compute loss against zero tensors to compare memory/gradient execution, not predictive quality against future frames.

Therefore the present result supports only:

- technical feasibility of reverse replay around a deterministic transition;
- same-recurrence reversible-versus-vanilla gradient and memory comparisons;
- bounded memory measurements for that prototype.

It does not establish that the integrated model makes useful 20- or 40-frame forecasts.

For Article 5, reversibility should be moved inside the repeated temporal core so that reversible and ordinary backpropagation execute the **same forward function**. The existing `ReversibleTemporalCore` pattern is closer to the required contract than the second-order wrapper. A forward-parity test must compare predictions, loss, and parameter gradients for the same model and batch.

### 5.3 P0: the current worktree should not be committed as a final state yet

The Ukrainian renumbering has created new figure files and deleted old figure names. The older English compact and restructured sources still reference the deleted names. Translation parity also reports mismatched headings, figures, equations, and tables.

Before the next commit intended as a stable checkpoint:

- either retain compatibility copies/aliases for old figure names or update all English references and provenance records atomically;
- decide whether the Ukrainian tree or a new English v2 tree is authoritative;
- rerun both Ukrainian-specific audits and the compact English audit;
- regenerate the provenance guide counts;
- verify `git diff --check`, local image links, and clean status;
- commit generated figures together with the generator and provenance changes.

A WIP commit is acceptable if explicitly labelled, but it should not be tagged as the baseline for new experiments.

### 5.4 P1: training-seed coverage is not aligned across the main frontier

The current frontier contains different numbers of independent training initializations:

| Budget/contrast | Training-seed coverage | Consequence |
|---|---:|---|
| Near-bottom | 2 | Limited seed-level evidence |
| 896 | 2 | Direction replicated, but still small seed sample |
| 1,152 | 3 | Stronger than neighboring points |
| 1,792 | 1 | Cannot characterize training variability |
| 2,432 | primarily seed 0, with two training lengths | Training length is not an independent seed |
| 3,072 | 2 | Limited seed-level evidence |
| Head × package | 5 | Best seed design in the current programme |
| E5a boundary | 3 | Adequate for a bounded replication statement |
| E7a/E7c robustness | predictor seed 0 | Conditional robustness only |
| Integrated prototype | 1 | Feasibility only |

Clip-cluster bootstrap intervals are useful for evaluation uncertainty within a trained seed. They cannot be presented as uncertainty over training randomness. The new figures should use two distinct visual objects:

- faint within-seed clip intervals;
- prominent seed-level means and intervals, with the seed as the independent replication unit.

For the new primary hierarchy study, use four seeds for every confirmatory arm. Five or six would be statistically safer, but four is a reasonable minimum if the design and interpretation remain conservative.

### 5.5 P1: some context comparisons change more than context

The legacy stride-based arms can change context stride, coarse target stride, the number of target tokens, auxiliary loss, sequence length, and optimization load together. The repository already acknowledges this and contains prepared fixed-target controls.

The new study must make the target interface invariant across all context-allocation comparisons:

- identical future window;
- identical target resolutions and target token count;
- identical loss terms and weights;
- identical predictor depth, width, feature dimension, parameter count, optimizer, update count, batch size, precision, and sampler;
- only the retained context package changes.

If model capacity must change for a level-count experiment, that experiment must be labelled a whole-architecture comparison rather than a pure allocation effect.

### 5.6 P1: “information density” is not yet adaptive allocation

The current fixed-region packer computes variance-based softmax weights inside eight immutable regions. Every fine cell contributes, region membership is fixed, and the output remains eight region tokens. This is a stable aggregation mechanism, but it does not allocate a variable number of tokens to dense regions and does not use V-BLT entropy.

For Article 5, choose one of two honest designs:

1. **Fixed-budget soft aggregation:** keep a constant number of tokens and call the component “density-weighted regional aggregation.” This is simple and stable but should not be sold as adaptive token allocation.
2. **True adaptive allocation:** assign a variable quota of fine tokens/patches to regions under a fixed global budget, store routing indices as part of the state cost, and compare it with fixed geometry and random/quota controls.

The second design is scientifically stronger but adds discontinuity, routing overhead, and possible membership churn. It should be introduced only after the fixed hierarchy is stable.

### 5.7 P1: external baseline comparisons are not yet matched

The current integration table combines:

- one locally fine-tuned packed SimVP;
- one local raw SimVP;
- published OpenSTL rows;
- memory profiles with different sequence or prediction horizons.

This is useful context, but it is not a causal architecture ranking. For Article 5, all primary baselines must be rerun under one protocol. Published values may remain a secondary reference table.

The comparison contract should fix:

- dataset split and preprocessing;
- input/output horizon;
- resolution;
- precision;
- batch size or a clearly declared maximum-memory batch protocol;
- optimizer, schedule, epochs/updates, and selection rule;
- parameter-count tolerance, preferably within ±10%;
- four seeds;
- the same metric implementation;
- training peak memory, inference state bytes, inference peak, throughput, and wall time as separate endpoints.

### 5.8 P1: the central code is scientifically frozen but poor scaffolding for new work

`hvp_rev/rhvd.py` is a useful frozen record, but it is not the right place for the new factorial programme. Its `ARMS` dictionary uses historical names such as `hier-c1`, `d1-56-diff`, and `strided-flat`, and the same module contains architecture, data construction, optimization, rollout, evaluation, and command-line behavior.

The public `ArmSpec` already improves on this by separating context and target interfaces. The new code should use data-driven configuration objects from the beginning, not add more tuples and special cases to `ARMS`.

### 5.9 P2: formulas are over-numbered

The 114 equations include several categories that should not have equal status:

- original model/interface definitions;
- standard literature equations;
- elementary algebraic inverses;
- implementation details;
- diagnostic definitions;
- hypothetical future extensions.

Examples of material that should be collapsed, moved to appendices, or cited rather than numbered include basic softmax and cross-entropy, separate forward and inverse lines that can be one aligned system, standard DDIM steps, standard AdaLN residual updates, and the unexecuted dynamic-region optimization.

Target: approximately **40–50 numbered equations in the whole main text**, with the rest inline or in appendices.

### 5.10 P2: figures are too numerous and several have multiple roles

The strongest current result figures are the direct forest plots, especially the pair-specific reach plot. The weakest pattern is a composite that joins an architecture schematic, a training trajectory, and an endpoint result in one figure. The EBT composite is the clearest example: its three panels answer three different questions. The frontier figure also combines an inferential frontier with token arithmetic.

Recommended rule:

- one figure, one research question;
- no figure should duplicate a small table exactly;
- architecture figures show components and tensor flow, not results;
- result figures show one endpoint family or one factor interaction;
- configuration grids and exhaustive registries belong in appendices;
- no more than about 8–12 figures per technical chapter.

### 5.11 P2: the name “LoopMom” is not informative

The name does not reveal the controlled mechanism and is difficult to defend in a formal article. Recommended replacement:

**Toroidal Re-entry Benchmark (TRB)**

Other acceptable options:

- Controlled Viewport Re-entry Benchmark;
- Long-Horizon Re-entry Testbed;
- ReEntryGrid.

To preserve provenance, do not rename old artifacts or hashes. Introduce the name as “Toroidal Re-entry Benchmark (TRB; called LoopMom in legacy artifact identifiers)” and use `trb-v2` only for newly generated datasets.

## 6. Current dissertation structure for comparison

### Chapter 1 — System analysis and requirements for consistent long-horizon video prediction

1.1 Video prediction as a partially observed dynamic system
1.1.1 System boundary, state, observation, and control
1.1.2 Uncertainty growth in autoregressive generation
1.1.3 Multi-objective prediction under resource constraints

1.2 Consistency as a multi-criteria property
1.2.1 Why one quality metric is insufficient
1.2.2 Appearance, object, and conceptual consistency
1.2.3 Temporal and trajectory consistency
1.2.4 Hierarchical and cross-level consistency
1.2.5 From static control criteria to video requirements

1.3 Bottlenecks of long-horizon inconsistency
1.3.1 Training-memory growth
1.3.2 Spatial token allocation
1.3.3 Limited retained-context reach
1.3.4 System map of bottlenecks and interventions

1.4 Existing methods
1.4.1 Long-horizon temporal modeling and resource-efficient training
1.4.2 Adaptive and hierarchical visual representation
1.4.3 Long context and multi-rate memory

1.5 Problem decomposition and requirements
1.5.1 Design requirements
1.5.2 Research hypotheses
1.5.3 Chapter conclusions

**Current assessment:** the logic is sound, but the chapter has 25 figures. The gallery of general diffusion/control illustrations overwhelms the hierarchy-and-memory argument.

### Chapter 2 — Methods for removing scaling constraints in hierarchical prediction

2.1 Controlled experimental foundation
2.1.1 Procedural video generation
2.1.2 2-D rotations
2.1.3 3-D polycubes
2.1.4 Dissertation datasets

2.2 Reversible temporal processes
2.2.1 Activation memory
2.2.2 Reversible block
2.2.3 Temporal chaining
2.2.4 Horizon-independent backward memory
2.2.5 Controlled temporal-pattern results
2.2.6 OpenSTL validation
2.2.7 Memory–compute boundary
2.2.8 Stochastic replay extensions
2.2.9 Applicability to BPTT and checkpointing

2.3 Hierarchical spatial representation and entropy
2.3.1 Fixed-patch allocation problem
2.3.2 V-BLT architecture
2.3.3 Causal attention and byte target
2.3.4 RoPE-ND
2.3.5 Predictive entropy
2.3.6 2-D/3-D results
2.3.7 Measured fixed method versus proposed dynamic regions

2.4 Implications for the hierarchical-context model
2.4.1 Conceptually relevant findings
2.4.2 Chapter conclusions

**Current assessment:** scientifically rich but structurally premature. It delays the central hierarchy investigation and spends 42 equations on two auxiliary strands and several extensions.

### Chapter 3 — Hierarchical contextual model for long-horizon video prediction

3.1 Retained context as a finite-model problem
3.1.1 Consistency and memory reach
3.1.2 Context-package design space
3.1.3 Reach–fidelity hypothesis

3.2 Hierarchical latent state
3.2.1 Frozen two-level autoencoder
3.2.2 Token/embedding construction
3.2.3 Joint attention
3.2.4 Gradient boundary and information asymmetry

3.3 Generative prediction core
3.3.1 Latent diffusion objective
3.3.2 Transformer backbone
3.3.3 DDIM sampling
3.3.4 Regression control

3.4 Autoregressive rollout and memory update
3.4.1 Ring-buffer recurrence
3.4.2 Anchor geometry
3.4.3 Context, target, and total cost

3.5 Evaluation instruments and estimands
3.5.1 Why short tests miss reach
3.5.2 LoopMom generator
3.5.3 Analytic observability
3.5.4 Frame/object metrics
3.5.5 Persistent objects and standard datasets
3.5.6 Statistical protocol

3.6 Registered hierarchy hypotheses
3.7 Controlled integration extension
3.7.1 Goal and boundary
3.7.2 Fixed-region state
3.7.3 Regression transition and reversible rollout
3.7.4 Protocol and allowed claims

3.8 Conclusions

**Current assessment:** this is the conceptual centre of the thesis, but methods, estimands, statistical derivations, and a later integration prototype compete for space. It should be merged with the central results and shortened.

### Chapter 4 — Experimental validation and operating region

4.1 Experiment map and comparison rules
4.2 Generation-mechanism selection, including EBT
4.3 Fixed-budget hierarchical reach
4.4 Reach–fidelity frontier and memory efficiency
4.5 Package shape and model interaction
4.6 Robustness to sampler and codec
4.7 Negative evidence and external boundaries
4.8 Integrated feasibility of spatial packing and reversible training
4.9 System-level method-selection rule
4.10 Software and practical application
4.11 Validity threats and conclusions

**Current assessment:** the chapter contains almost every remaining story. The core hierarchy result, EBT diagnostics, robustness, failed replication, privileged reference, persistent-object boundary, integration, software architecture, and implementation claims do not form one clean experimental arc.

## 7. Proposed dissertation structure

## Chapter 1 — Hierarchy and memory in long-horizon video prediction

**Purpose:** establish the central problem and vocabulary before presenting any author model. Hierarchy is introduced as a family of resource-allocation mechanisms, not as one fixed architecture.

### 1.1 Long-horizon prediction as partial observation with feedback

#### 1.1.1 State, observation, action, and self-fed prediction

Short description: formalize video prediction as a partially observed controlled process and explain why self-fed errors alter subsequent inputs.

- Figures: one system-boundary diagram; one compact teacher-forced versus self-fed timeline.
- Numbered equations: state/observation model; autoregressive future distribution.

#### 1.1.2 Task-specific consistency

Short description: distinguish appearance, identity, spatial support, trajectory, temporal continuity, and cross-level consistency.

- Figures: one consistency-failure taxonomy with examples.
- Numbered equations: none; keep metric definitions for Chapters 2 and 4.

#### 1.1.3 Why memory is the unifying scarce resource

Short description: separate retained state, training activations, parameters/optimizer, and temporary workspace; explain why the first two are controlled by different mechanisms.

- Figures: one memory-location diagram.
- Numbered equations: one memory decomposition.

### 1.2 Manifestations of hierarchy

#### 1.2.1 Representational, temporal, and computational hierarchy

Short description: define hierarchy at three interfaces: spatial resolution, temporal rate/reach, and computation/reconstruction.

- Figures: one diagram showing the three manifestations and their interfaces.
- Numbered equations: none.

#### 1.2.2 Why hierarchy matters especially in transformers

Short description: explain token-count and attention-cost growth, and why flat dense context becomes expensive.

- Figures: one scaling sketch comparing dense and hierarchical context.
- Numbered equations: token count and attention/storage scaling, preferably as one grouped equation.

#### 1.2.3 Reach, fidelity, and information density

Short description: introduce the central trade-off and the distinction between storing old coarse evidence and preserving recent fine identity.

- Figures: one conceptual reach–fidelity plane.
- Numbered equations: none; exact package equations belong in Chapter 2.

### 1.3 Existing approaches and unresolved gaps

#### 1.3.1 Long context, recurrence, retrieval, sparse attention, and state-space models

#### 1.3.2 Multiscale and hierarchical video representations

#### 1.3.3 Checkpointing and reversible training

#### 1.3.4 Adaptive tokenization and information-density routing

Short description: compare intervention interfaces and identify what each method changes. Avoid a long generic diffusion gallery.

- Figures: one literature-positioning map, not a table.
- Numbered equations: none; cite standard methods.

### 1.4 Research gap, objective, hypotheses, and contributions

#### 1.4.1 Missing controlled evidence

Short description: identify the absence of aligned budget, target interface, seed coverage, and reach-aware evaluation.

#### 1.4.2 Research objective and hypotheses

Short description: state the central question and three resource-specific hypotheses.

- Figures: one claim-to-evidence map.
- Numbered equations: one constrained optimization statement if it is used later; otherwise prose.

#### 1.4.3 Contribution and publication map

Short description: map the review article and hierarchy article to Chapter 1/2, TempVerseFormer and V-BLT to Chapter 3, and the new integration article to Chapter 4.

### 1.5 Chapter conclusions

**Target for Chapter 1:** 5–7 numbered equations, 6–8 figures, no main-text tables unless an institutional convention requires a literature comparison table.

## Chapter 2 — Hierarchical context allocation under fixed memory budgets

**Purpose:** become the central dissertation chapter and the basis of the redesigned fourth article. It combines the essential method from the current Chapter 3 with the central experiments from the current Chapter 4.

### 2.1 Problem definition and hierarchy design space

#### 2.1.1 Context levels, anchors, and budgets

Short description: define each level by spatial token cost, temporal stride, slot count, and newest/oldest anchor.

- Figures: one level/anchor diagram.
- Numbered equations: context-package definition; context token budget; nominal reach.

#### 2.1.2 Allocation policies across hierarchy levels

Short description: define flat, balanced, linear-age, geometric-age, and reach-optimized allocations under the same total token budget.

- Figures: one allocation-policy diagram using identical token area.
- Numbered equations: budget constraint and normalized allocation policy.

#### 2.1.3 Nested versus exclusive temporal overlap

Short description: define whether a time point is represented at multiple levels or only at its finest retained level. In the exclusive design, saved overlapping tokens must be reassigned so total budget remains fixed.

- Figures: one direct nested/exclusive timeline.
- Numbered equations: overlap count or set relation.

#### 2.1.4 Research questions and falsifiable hypotheses

Short description: state the budget frontier, allocation-policy, number-of-levels, and overlap hypotheses separately.

### 2.2 Hierarchical predictor architecture

#### 2.2.1 Multi-level codec and invariant target interface

Short description: describe tensor shapes, frozen/trainable boundaries, reconstruction floors, and the target interface held constant across context comparisons.

- Figures: one codec diagram with tensor shapes.
- Numbered equations: encoder/decoder interface and reconstruction loss only.

#### 2.2.2 Context-token assembly and positional metadata

Short description: show token order, level/role/time embeddings, actions, and exactly where overlap is represented.

- Figures: one token-sequence schematic.
- Numbered equations: token projection and metadata sum as one equation.

#### 2.2.3 Shared prediction core

Short description: emphasize that one predictor jointly processes all context levels and future target tokens. Explain capacity, parameter count, and target attention semantics.

- Figures: one full architecture diagram; this should be the primary architecture figure of Article 4.
- Numbered equations: conditional predictor and the executed loss. Standard Transformer/AdaLN equations should be cited, not rederived.

#### 2.2.4 Autoregressive update

Short description: define one four-frame prediction step and the update of each retained stream.

- Figures: one ring-buffer/timeline diagram.
- Numbered equations: one stream-update equation.

### 2.3 Toroidal Re-entry Benchmark and evaluation

#### 2.3.1 Dataset mechanism and correctness checks

Short description: present TRB as a controlled, action-conditioned re-entry benchmark; disclose static-world, camera-motion, and toroidal assumptions.

- Figures: one world/viewport diagram and one short sequence strip.
- Numbered equations: viewport position and rendering relation.

#### 2.3.2 Exact observability and pair-specific populations

Short description: define what evidence is available to one package but not another.

- Figures: one pair-specific reach-band diagram.
- Numbered equations: coverage set and pair-specific population.

#### 2.3.3 Metrics and seed-aware inference

Short description: keep only primary frame/object metrics in the main text. Move full metric and bootstrap definitions to the appendix.

- Figures: none.
- Numbered equations: primary paired effect and across-seed estimator; detailed bootstrap/max-statistic derivation in appendix.

### 2.4 Experimental programme

#### 2.4.1 Invariant training protocol and experiment naming

Short description: define the fixed target, model capacity, optimizer, update count, data split, precision, sampler, and four-seed rule.

- Figures: one experiment-design matrix.
- Tables: none in the main text; place the full configuration registry in an appendix and JSON artifacts.

#### 2.4.2 Fixed-budget allocation sweep

Short description: same total token budget and embedding width; vary only allocation among levels.

#### 2.4.3 Context-budget frontier

Short description: approximately ten budget points from low to high context capacity, evaluated with a small predeclared set of allocation policies.

#### 2.4.4 Fixed fine context with expanding coarse/top context

Short description: hold recent fine slots fixed and increase older lower-resolution slots to isolate purchased reach.

#### 2.4.5 Number of levels at fixed budget

Short description: compare one, two, and three levels with the same context budget and target interface.

#### 2.4.6 Nested versus exclusive overlap

Short description: compare duplicate multilevel representation with exclusive temporal ownership under equal cost and reach wherever feasible.

#### 2.4.7 Predictor-head control and robustness

Short description: keep regression/diffusion as a bounded secondary factor, not the organizing axis of the whole chapter.

### 2.5 Results

#### 2.5.1 Quality–reach–memory frontier

- Figures: a single primary frontier plot with all four seed points visible; a separate Pareto plot if measured GPU state/attention cost is included.
- Tables: one primary result table with seed means, seed variability, and effect direction.

#### 2.5.2 Allocation-policy effects

- Figures: one curve of performance versus budget, faceted only by allocation policy.
- Tables: one confirmatory contrast table.

#### 2.5.3 Number-of-levels effect

- Figures: one direct one/two/three-level comparison.
- Tables: optional result table if exact values are not readable from the plot.

#### 2.5.4 Overlap effect

- Figures: one paired nested-versus-exclusive plot.
- Tables: one result table only if this is a primary claim.

#### 2.5.5 Robustness and failure cases

- Figures: one seed-heterogeneity plot; one robustness plot if it answers a distinct question.
- Tables: one concise result table, with negative/unresolved results retained.

### 2.6 Discussion, operating region, and validity limits

### 2.7 Chapter conclusions

**Target for Chapter 2:** 14–18 numbered equations, 10–12 figures, 3–5 main-text tables, all of them result tables.

## Chapter 3 — Activation memory and spatial information density

**Purpose:** place TempVerseFormer and V-BLT after the main hierarchy result, as additional mechanisms needed because context hierarchy does not solve every memory cost.

### 3.1 Limits not removed by context hierarchy

#### 3.1.1 Activation storage during long-horizon training

#### 3.1.2 Uneven spatial information density

- Figures: one two-bottleneck transition from Chapter 2.
- Numbered equations: one refined memory decomposition.

### 3.2 Reversible temporal computation

#### 3.2.1 TempVerseFormer architecture and reversible boundary

- Figures: one architecture diagram with reversible and non-reversible components.
- Numbered equations: one grouped additive-coupling forward/inverse system.

#### 3.2.2 Reverse replay, stochastic state, and numerical stability

- Figures: one backward-replay diagram.
- Numbered equations: adjoint/replay relation; memory–compute trade-off. Move optional stochastic/RNG contracts to the appendix unless used by Chapter 4.

#### 3.2.3 Experimental results and OpenSTL evidence

- Figures: memory versus horizon; runtime versus horizon; one quality summary.
- Tables: published quality results and memory results only.

#### 3.2.4 Applicability and limitations

Short description: clarify that reversibility changes training storage, not prediction quality by itself.

### 3.3 Hierarchical byte-latent spatial modeling

#### 3.3.1 V-BLT local–global–local architecture

- Figures: one architecture diagram.
- Numbered equations: one model mapping and byte-classification loss.

#### 3.3.2 RoPE-ND

- Figures: one axis/subspace diagram.
- Numbered equations: one compact multi-axis rotation equation; omit separate elementary 1-D derivations.

#### 3.3.3 Predictive entropy as a spatial-complexity diagnostic

- Figures: one entropy pipeline and one result map.
- Numbered equations: entropy only.

#### 3.3.4 Experimental results and boundary of the method

- Figures: one quality/cost result plot.
- Tables: 2-D and 3-D result tables.
- Explicit boundary: fixed patches are implemented; dynamic allocation is not yet established.

### 3.4 Synthesis: what each mechanism changes

Short description: context hierarchy compresses retained history; reversibility reconstructs training activations; density-aware representation allocates spatial detail. Keep the resources separate.

- Figures: one interface map.
- Numbered equations: none.

### 3.5 Chapter conclusions

**Target for Chapter 3:** 12–16 numbered equations, 8–10 figures, 4–5 result tables.

## Chapter 4 — Integrated memory-efficient hierarchical video predictor

**Purpose:** become the basis of Article 5 only if the new architecture passes quality and memory gates on OpenSTL-managed datasets.

### 4.1 Requirements and selection of the Chapter 2 backbone

#### 4.1.1 Pareto selection from the hierarchy study

Short description: choose a high-performing, memory-efficient hierarchy configuration using a predeclared rule, not post-hoc preference.

- Figures: the selected point highlighted on the Chapter 2 Pareto frontier.
- Numbered equations: selection rule only if formalized.

#### 4.1.2 Integration hypotheses and non-inferiority margins

Short description: state that reversibility is expected to reduce training memory, not directly improve quality; hierarchy and routing must justify any quality/state trade-off.

### 4.2 Architecture

#### 4.2.1 Fixed-width multi-level state

- Figures: one state-layout diagram with actual stored bytes and metadata.
- Numbered equations: pack/unpack interface and exact state cost.

#### 4.2.2 Density-aware detail allocation

- Figures: one routing diagram.
- Numbered equations: fixed-budget quota or weighting equation. Use “adaptive allocation” only if token counts truly move between regions.

#### 4.2.3 Predictor and reversible temporal core

- Figures: one end-to-end component diagram and one reversible-block detail.
- Numbered equations: predictor transition and a grouped reversible forward/inverse system.
- Required property: reversible and ordinary modes have the same forward predictions.

#### 4.2.4 Training and inference paths

- Figures: one two-lane diagram separating training activation storage from inference retained state.
- Numbered equations: full training objective.

### 4.3 OpenSTL evaluation protocol

#### 4.3.1 Datasets and task boundaries

Use Moving-MNIST, TaxiBJ, and KTH through one OpenSTL-compatible data interface. Note that Moving-MNIST is still synthetic; “OpenSTL-only” should mean no custom LoopMom/TRB evidence in Article 5, not “all datasets are natural.” TaxiBJ is real spatiotemporal grid data, while KTH is the clearest real-video component.

- Figures: one small dataset sample strip if needed.
- Tables: configuration details in appendix only.

#### 4.3.2 Matched baselines and training budget

Short description: rerun selected OpenSTL baselines with the same splits, updates, selection, precision, and seed count. Match parameter count or report it explicitly.

#### 4.3.3 Quality, memory, compute, and Pareto endpoints

- Numbered equations: training peak, inference state bytes, throughput/compute, and Pareto/non-inferiority criterion. Do not combine resource ratios.

#### 4.3.4 Component-ablation design

Minimum set:

- flat baseline;
- hierarchy only;
- hierarchy plus density-aware aggregation/allocation;
- hierarchy plus reversible training;
- full hierarchy plus density mechanism plus reversible training;
- full model with hierarchy disabled but the rest retained;
- fixed routing versus adaptive routing if adaptive routing is implemented.

### 4.4 Component effects on memory and quality

#### 4.4.1 Retained-state effect of hierarchy

- Figures: state-memory bars or waterfall.
- Tables: exact state bytes and quality.

#### 4.4.2 Activation-memory effect of reversibility

- Figures: peak training memory versus horizon, with checkpointing as a matched comparator.
- Tables: exact peaks, runtime, and gradient/forward parity.

#### 4.4.3 Spatial-allocation effect

- Figures: region/token allocation map and one direct ablation plot.
- Tables: fixed versus adaptive/random routing results.

#### 4.4.4 Full integration

- Figures: component-ablation plot in which the full model is consistently marked, but only allowed to “stand out” if the measurements support it.
- Tables: full factorial/fractional-factorial result summary.

### 4.5 Dataset-level comparison

#### 4.5.1 Moving-MNIST

#### 4.5.2 TaxiBJ

#### 4.5.3 KTH

#### 4.5.4 Cross-dataset quality–memory Pareto frontier

- Figures: one result figure per dataset role; one final normalized Pareto figure.
- Tables: one primary table per dataset or one carefully structured cross-dataset result table.

### 4.6 Robustness, failure analysis, and validity limits

### 4.7 Chapter conclusions

**Target for Chapter 4:** 10–14 numbered equations, 8–10 figures, 4–6 result tables.

## 8. New Article 4 experiment design

Recommended title:

**Hierarchical Context Allocation Under Fixed Memory Budgets for Long-Horizon Video Prediction**

### 8.1 Do not run the full Cartesian product

The proposed axes are scientifically appropriate, but their full product would be wasteful and difficult to interpret. Use a staged design.

### Stage A — correctness and deterministic accounting

- validate dataset generation, action/position consistency, train/validation/test separation, foreground thresholds, and observability;
- validate every package’s exact anchors, overlap, reach, token count, state bytes, target interface, and expected sequence length;
- run persistence, last-frame, flat oracle, pose-indexed oracle, and shuffled-history controls;
- ensure that each comparison changes only its declared factor.

### Stage B — exploratory screen

- about ten token budgets;
- four allocation policies: flat recent, balanced, linear-age, geometric-age/reach-oriented;
- two seeds or one seed plus an aggressive early-training screen;
- fixed target interface and predictor;
- no protected confirmatory test access.

This stage identifies useful regions and obvious failures. Its results are exploratory and should not become headline confidence statements.

### Stage C — confirmatory hierarchy study

- select approximately 8–12 arms using the predeclared Stage B rule;
- run four fresh seeds for every arm;
- include low, middle, unresolved/crossover, and high budgets;
- include one/two/three-level contrast;
- include nested/exclusive overlap contrast;
- evaluate once on the protected confirmation set;
- perform seed-level inference, with clip-level paired effects nested within seed.

### Stage D — bounded robustness

- one alternate codec seed;
- one alternate sampler or regression head;
- no adaptive extension triggered by favorable outcomes.

### 8.2 Example budget grid

Use exact budgets compatible with the token geometry, for example 320, 384, 512, 640, 768, 896, 1,152, 1,792, 2,432, and 3,072 tokens. The exact grid should be generated by an integer allocator, not hand-edited. If some policies cannot realize a budget exactly, either use an explicit near-match tolerance or select a common feasible grid.

### 8.3 Experiment names

Human-readable and machine-readable names should expose the manipulated factors. Example:

`HCA-B1152-L2-G48-F6-NEST-DIFF-S00`

where:

- `HCA` = hierarchical context allocation study;
- `B1152` = context-token budget;
- `L2` = two active levels;
- `G48` = 48 coarse slots/declared coarse reach setting;
- `F6` = six recent fine slots;
- `NEST` = nested overlap (`EXCL` for exclusive);
- `DIFF` = prediction head;
- `S00` = training seed.

The actual run identity should also include a content hash of the full resolved config. Paper labels can be shorter, but artifact names should never use ambiguous labels such as `c1`, `d1`, `min`, or `long` without resolved metadata.

## 9. New Article 5 design and decision gate

Recommended provisional title:

**A Memory-Efficient Hierarchical Video Predictor with Density-Aware Detail and Reversible Training**

### 9.1 Architecture recommendation

Do not make the current packed-SimVP plus second-order wrapper the final architecture. Instead:

1. select the Chapter 2 hierarchy package by a frozen Pareto rule;
2. implement a fixed-width hierarchical state with exact byte accounting;
3. initially use fixed regional aggregation to establish a stable baseline;
4. add true density-aware allocation only as a separately testable component;
5. use reversible coupling inside the repeated temporal core so forward predictions are identical in reversible and ordinary training modes;
6. decode through one end-to-end trainable, OpenSTL-compatible predictor;
7. compare reversible replay with ordinary BPTT and checkpointing under the same model and loss.

### 9.2 Success criteria

The fifth article does not need to win every quality metric. A defensible success is a reproducible Pareto improvement.

Suggested gates:

- **G1 — correctness:** forward/loss/gradient parity between reversible and ordinary modes within declared tolerance;
- **G2 — no leakage:** no test-based component or checkpoint selection;
- **G3 — memory:** at least 25–30% lower peak training memory than matched BPTT at the target horizon, with checkpointing reported;
- **G4 — quality:** non-inferior primary quality within a predeclared margin, or superior quality at the same memory limit;
- **G5 — replication:** four seeds for primary ablations;
- **G6 — external scope:** the full model reaches a useful Pareto point on at least two OpenSTL-managed datasets, one of which should be KTH or another real-video dataset;
- **G7 — component logic:** the full integration is better than or meaningfully trades off against the component variants; otherwise the paper should report the negative interaction rather than force a synergy claim.

If G3–G6 fail after the pilot, stop the fifth-article claim. Keep the result as a dissertation feasibility/negative-result section and do not build the whole thesis around it.

### 9.3 Efficient ablation schedule

Avoid an eight-cell, three-dataset, four-seed full factorial from the start.

1. Moving-MNIST development: five core configurations, two seeds.
2. TaxiBJ development: surviving configurations, two seeds.
3. Freeze architecture and margins.
4. Confirm on Moving-MNIST and TaxiBJ with four fresh/complete seeds.
5. Run KTH only for the flat baseline, hierarchy-only, reversible-only where meaningful, and full model.
6. Add adaptive-routing ablations only if the fixed full model passes the quality–memory gate.

## 10. Codebase and Git strategy

### 10.1 Recommended option: same repository, clean branch and worktree

Do not create a separate repository yet. The existing repository contains the frozen evidence, public evaluation/statistics/provenance interfaces, and manuscript history. A new repository would duplicate those assets and weaken traceability.

Recommended sequence:

1. finish or explicitly checkpoint the current Ukrainian renumbering work;
2. make the current branch clean and passing for both its intended Ukrainian and English checks;
3. tag or branch it as the frozen dissertation-v1 fallback;
4. create a new branch such as `research/hierarchy-allocation-v2`;
5. create a separate Git worktree for that branch so current manuscript maintenance and new experiments cannot overwrite each other;
6. create a new English manuscript tree, for example `docs/dissertation/v2_english/`;
7. preserve `hvp_rev` unchanged;
8. build new code in a clean namespace such as `hvp_next/` or `hvp_hca/`;
9. write new resolved configs under `experiments/v2/configs/` and small immutable summaries under `artifacts/v2/`;
10. keep raw runs/checkpoints in ignored external storage, referenced by manifests and hashes.

Suggested new package structure:

```text
hvp_hca/
  config.py
  allocation.py
  codec.py
  predictor.py
  reversible.py
  data/
    trb.py
    openstl.py
  training/
    engine.py
    checkpoint.py
  evaluation/
    metrics.py
    seed_inference.py

experiments/v2/
  schemas/
  configs/
  manifests/
  protocols/

artifacts/v2/
  hierarchy_screen/
  hierarchy_confirmation/
  integration_openstl/

docs/dissertation/v2_english/
```

### 10.2 What not to do

- Do not rewrite `hvp_rev` and then attempt to prove all old artifacts still mean the same thing.
- Do not create an orphan branch that discards provenance history.
- Do not use `git filter-repo` or history rewriting during the dissertation unless repository publication size becomes an unavoidable blocker.
- Do not copy the whole repository into a new untracked directory and continue without hashes or shared tests.
- Do not mix new exploratory results into the frozen `artifacts/confirmation_*` directories.
- Do not overwrite old “LoopMom” artifact names merely to apply the new TRB name.

### 10.3 When a separate repository becomes reasonable

A separate public repository is useful only after the Article 5 architecture stabilizes and has a clean dependency/licensing boundary from OpenSTL. At that point it can be extracted from `hvp_hca` with a documented commit and artifact manifest. It should be a release operation, not the environment in which the new science is first developed.

## 11. Paths forward and estimates

All estimates assume one main developer/researcher and one RTX 4060 Ti-class GPU. They are ranges, not promises; failed pilots can increase them substantially.

### Path A — stabilize and defend the current dissertation

Scope:

- finish current renumbering and provenance synchronization;
- reduce obvious figure/formula overload without changing the empirical programme;
- keep the current four-chapter structure;
- retain integration as feasibility only;
- submit the current fourth article without the proposed new hierarchy sweep.

Estimate:

- editorial/code cleanup: 1–3 weeks;
- new GPU work: minimal;
- scientific risk: low;
- contribution ceiling: moderate;
- fifth article: not supported.

This is the best fallback if dissertation timing is dominant.

### Path B — staged hierarchy redesign, with conditional integration (recommended)

Scope:

- freeze current dissertation as fallback;
- write the new English Chapter 1 and Chapter 2 structure;
- implement the clean hierarchy allocation study;
- run exploratory then four-seed confirmatory experiments;
- rewrite Article 4 around the aligned study;
- begin Article 5 only after an integration pilot passes.

Estimate for Article 4 / new Chapter 2:

- repository and protocol setup: 3–5 days;
- clean model/config implementation and parity tests: 1–2 weeks;
- dataset/oracle validation and pilots: about 1 week;
- GPU: approximately 200–450 GPU-hours for screen plus confirmation, depending on how many arms are retained;
- analysis, figures, and Article 4 rewrite: 2–3 weeks;
- Chapter 1/2 manuscript integration: 1–2 weeks;
- total elapsed: roughly 7–12 weeks if the first design works.

Conditional Article 5 / Chapter 4:

- architecture correction and no-leakage pipeline: 2–4 weeks;
- pilots and gates: 1–2 weeks;
- GPU: approximately 300–900 GPU-hours for selected baselines, ablations, datasets, and four seeds;
- analysis, paper, and dissertation chapter: 3–5 weeks;
- additional elapsed: roughly 7–12 weeks after Article 4, longer if KTH training or architecture debugging is expensive.

Combined realistic range: **14–24 weeks**.

### Path C — full reset and complete re-experimentation before writing

Scope:

- replace both central and integrated architectures;
- run all proposed factors at four seeds;
- rerun all OpenSTL baselines;
- rewrite the entire dissertation only after all experiments complete.

Estimate:

- elapsed: 4–6 months or more;
- GPU: 800–1,500+ hours;
- scientific risk: high;
- schedule risk: very high;
- benefit: cleanest final narrative if everything succeeds.

This path is not recommended unless there is no near-term submission deadline and compute is reliably available.

### Path D — continue current writing and redo experiments in parallel without a freeze

This is the riskiest operational path. Both manuscript numbering and scientific claims would move while new code and new evidence are changing. It invites stale figures, ambiguous sources, accidental mixing of exploratory and confirmatory results, and unrecoverable comparison problems.

If parallel work is necessary, use two worktrees and treat the current manuscript branch as maintenance-only. Do not edit the same manuscript tree from both branches.

## 12. Immediate recommended sequence

1. **Stabilize the present worktree.** Resolve the renamed/deleted figure references and Ukrainian/English audit scope. Commit a clean, explicitly named baseline.
2. **Create the v2 branch and worktree.** Preserve `hvp_rev`; start `hvp_hca` and `docs/dissertation/v2_english`.
3. **Freeze the new Article 4 questions.** Budget frontier, allocation policy, level count, and overlap are four separate questions.
4. **Implement configuration-first experiments.** Every run gets a resolved config, semantic name, content hash, seed, dataset manifest, and selection status.
5. **Validate TRB-v2.** Keep the old generator provenance, add oracle and leakage checks, and use the new formal name only for new artifacts.
6. **Run a small screen.** Do not commit to the full 200–450 GPU-hour programme until basic learning curves and metric directions are credible.
7. **Confirm Article 4 with aligned seeds.** Four seeds for every selected arm; protected-set access only after selection.
8. **Rewrite Chapters 1 and 2 in English.** Use the proposed structure and reduced figure/formula budget.
9. **Rebuild integration semantics.** Same forward function for reversible and ordinary training; clean validation/test split; matched OpenSTL baselines.
10. **Apply Article 5 gates.** Promote the work to a full chapter/article only if it establishes a reproducible Pareto result.

## 13. Bottom line

Your diagnosis is substantially correct:

- hierarchy should be the organizing idea, not one method among several;
- memory is the best unifying resource, but it must remain decomposed into retained state, activations, and spatial capacity;
- the main hierarchy architecture and evidence should appear before TempVerseFormer and V-BLT;
- the current Chapter 4 mixes too many experimental roles;
- seed coverage and experiment names need normalization;
- the integrated model is promising as a feasibility prototype but not yet strong enough for a separate article;
- a clean branch/worktree and new code namespace are preferable to extending the frozen monolith;
- a completely separate repository is premature.

The most defensible strategic choice is Path B: **freeze the current work, rebuild the hierarchy study cleanly, rewrite Chapters 1–2 in English, and make the fifth article conditional on an honest OpenSTL Pareto result.**
