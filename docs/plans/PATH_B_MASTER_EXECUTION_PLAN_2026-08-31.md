# Path B master execution plan

**Decision date:** 2026-08-31
**Status:** approved direction; execution baseline
**Language of the new manuscript:** English
**Existing Ukrainian manuscript:** frozen fallback and provenance source, not overwritten

## 1. Decision

Proceed with Path B: preserve the current dissertation as a defensible fallback, create a clean branch and worktree for the new experiments, and rebuild the dissertation around one central question:

> How can spatial and temporal hierarchy allocate a limited memory budget in transformer-based video prediction, and how do allocation, reversible computation, and information-density-aware routing affect predictive quality and practical memory use?

The dissertation should not promise a predetermined 3--4x or 2--3x gain. Those values are experimental hypotheses and decision gates. The final claim must report the measured quantity precisely: token payload, retained-state bytes, saved activations, incremental CUDA allocation, or full peak training/inference memory.

## 2. What is preserved and what is rebuilt

### Preserved

- The current Ukrainian manuscript and its provenance records.
- Published content, diagrams, results, and source citations from Articles 1--3.
- Existing EBT/WindowDiT experiments as pilot evidence and as a source of hypotheses.
- Existing tested code where unit tests and parity checks establish a reliable component.
- Legacy artifact identifiers needed for reproduction.

### Rebuilt

- Article 4 experiments, with named configurations, fixed targets, explicit budgets, two-seed replication, and one-role figures/tables.
- The integrated Article 5 architecture around an explicit token-state contract.
- Dynamic spatial allocation so that it changes retained state, rather than merely reweighting fixed regions.
- The reversible predictor so that reversibility is internal to a predictor block and does not silently change the intended temporal recurrence.
- The English dissertation structure and chapter flow.
- Experiment manifests, seed policy, run registry, hardware metadata, and memory measurement protocol.

## 3. Scientific decomposition

The work contains three distinct memory mechanisms. They must be measured separately before they are combined.

1. **Spatial hierarchy and allocation** reduce the number or precision of retained spatial tokens. A coarse base plus a quota of fine detail tokens can reduce stored context per frame.
2. **Temporal hierarchy and retention** allocate different horizons to different levels. Old observations can remain available at coarse resolution while recent or important observations retain detail.
3. **Reversible computation** reduces training activation storage inside a fixed-width differentiable core. It does not reduce the inference context state by itself.

Dynamic regioning connects the first two mechanisms: reducing fine spatial coverage at each time point can purchase a longer temporal horizon under the same global token budget. This coupling belongs in Chapter 2 as a controlled allocation study. Learned dynamic routing remains an Article 5 integration component.

## 4. Integrated architecture contract

### 4.1 Canonical state

At time step `t`, the predictor receives and returns a packed hierarchical state:

```text
H_t = {
  coarse_tokens: C_t[N_c, d],
  detail_tokens: D_t[K, d],
  coordinates:   X_t[N_c + K, 2],
  time_or_age:   T_t[N_c + K],
  scale_level:   L_t[N_c + K],
  validity_mask: M_t[N_c + K],
  route_metadata: R_t
}
```

`K` is fixed for a compiled experiment configuration. The selected content may change, but the tensor width must remain fixed unless a genuinely packed sparse kernel is used. Padding a variable number of tokens to a fixed maximum does not provide a real GPU-memory saving.

### 4.2 Positional encoding

Static learned embeddings tied to token slot are prohibited for dynamically moving detail tokens. Each token must be positioned by its physical meaning:

- normalized spatial centre `(x, y)`;
- time index or age `tau`;
- patch scale and hierarchy level;
- optional region/track identity, used only as metadata rather than as the primary position.

The first implementation should use factorized continuous Fourier features or coordinate-conditioned RoPE for `x`, `y`, and `tau`, plus an additive level embedding. The same coordinate system must be used by encoder, attention, state update, and decoder scatter/gather. A learned slot embedding may be added only as a small auxiliary term after coordinate ablation proves it is harmless.

### 4.3 Routing boundary

Top-k selection, quantization, and token dropping are not invertible. Therefore:

```text
encoder -> router -> fixed-budget packer -> reversible predictor core -> decoder
                         ^ non-reversible boundary ^
```

The router must save or regenerate exactly the coordinates, masks, and ordering used by the packed state. The reversible core reconstructs only its internal activations; it does not attempt to invert routing. Dropout/stochastic depth inside a reversible block must either be disabled or replay the exact RNG state.

### 4.4 Temporal transition

Reversibility must not alter the requested forecasting dynamics. The current second-order coupling (`p' = q`, `q' = p + F(q)`) is a valid reversible dynamical system, but it is not equivalent to repeatedly applying the original predictor `q' = F(q)`. The new implementation must expose two explicit modes:

- `predictor_reference`: ordinary predictor rollout, used for quality equivalence;
- `predictor_reversible_blocks`: the same external state transition with reversible internal block stacks.

A stepwise output/gradient parity test is required before memory claims are accepted.

### 4.5 Decoder

The decoder produces a coarse full-frame base and coordinate-scattered fine residuals. Missing fine regions remain represented by the coarse path. Overlaps have a declared compositing rule, and routing boundaries are tested for seams and temporal flicker.

## 5. Hypotheses and acceptance gates

| Gate | Hypothesis | Passing evidence | Failure consequence |
|---|---|---|---|
| G0 | The new evaluation is uncontaminated | validation selects checkpoints; test is read once after lock | no publishable quality claim |
| G1 | Hierarchy improves the quality--memory frontier | at least one hierarchical point dominates or matches flat within declared tolerance at equal reach/budget | restrict claim to reach/representation analysis |
| G2 | Equal-reach hierarchy reduces retained context | measured serialized/CUDA state plus analytic count, target >=2x; 3--4x is stretch | report actual ratio; do not inflate claim |
| G3 | Reversible blocks preserve outputs and gradients | numerical parity on deterministic small models and equivalent convergence | keep reversibility as diagnostic only |
| G4 | Reversibility reduces training peak memory | target >=1.7x on block-dominated setup; report full-peak and incremental values | narrow claim to activation component |
| G5 | Dynamic routing actually changes stored state | fixed quota, coordinates/masks, measured state reduction; not only weighted aggregation | rename as attention weighting, not compression |
| G6 | Dynamic routing is stable | churn, coverage, route entropy, seam/flicker diagnostics acceptable | add hysteresis/EMA or keep static allocation |
| G7 | Integrated model is competitive | within predeclared quality tolerance of matched flat/OpenSTL baselines while materially lower in measured memory | publish component analysis, not superiority |
| G8 | Components compose without hidden regression | full model is on or near the ablation Pareto envelope | remove the harmful component from final model |

Recommended initial competitiveness tolerances: no more than 5% relative degradation in the primary validation metric for a memory-focused comparison, and no more than 2% for a claim of quality parity. These thresholds must be declared before confirmation runs.

## 6. Two-week execution boundary

The two one-week windows are feasibility and evidence-consolidation sprints. They cannot guarantee a finished two-paper experimental package on unknown hardware. They can produce a go/no-go answer, a defensible primary frontier, and a validated integrated prototype.

- **Week 1 -- Article 4:** clean controlled hierarchy experiments, seed-confirmed primary contrasts, and the spatial--temporal allocation surface.
- **Week 2 -- Article 5:** integration contract implementation, parity/memory gates, dynamic routing feasibility, and limited OpenSTL comparisons.

If both weeks pass their gates, budget an additional 1--3 GPU-weeks for four-seed confirmation on all selected OpenSTL datasets, final long-horizon runs, and camera-ready ablations. The exact duration scales approximately inversely with the number of comparable GPUs.

## 7. Repository strategy

Use one repository with a clean worktree rather than a disconnected new repository.

- Freeze the current validated state in a named commit on `research/ebt-hierarchical-video`.
- Create branch `codex/hierarchy-allocation-v2` in a sibling worktree.
- Put all new experiment definitions under a versioned namespace; do not reuse ambiguous legacy configuration names.
- Reference legacy artifacts by immutable path and SHA-256 rather than copying them without provenance.
- Do not rewrite history or delete current generated artifacts.

Proposed new namespaces:

```text
experiments/v2/configs/article4/
experiments/v2/configs/article5/
hvp_hca/
scripts/v2/
artifacts/v2/manifests/
docs/dissertation/v2_english/
docs/experiments/v2/
```

Each run manifest records experiment ID, git commit, config hash, seed, dataset/split hash, checkpoint-selection split, GPU model, software versions, start/end time, status, output paths, and memory-profiling mode.

The inspected local host has one NVIDIA GeForce RTX 4060 Ti with 16 GB VRAM. The one-week schedules therefore distinguish a one-GPU decision sprint from the later publication-completion queue. Parallel two-GPU estimates in the article plans are optional acceleration, not the assumed local capacity.

## 8. Dissertation migration

The new English manuscript is drafted in a new tree. Published material from Articles 1--3 is transferred through a ledger that records source, allowed reuse, adaptation, destination, and figure provenance. Text is rewritten into one dissertation voice; results and images retain their original attribution.

Migration is now evidence-triggered rather than an immediate chapter reorder. Preserve the current semantic order of Chapters 1 and 2, rebuild Chapter 3 around the redesigned Article 4 architecture and experiments, and choose Chapter 4 only after Article 5 gates. A successful two-dataset integrated result produces a distinct integrated-model Chapter 4; a mixed or negative result produces a synthesis/limitations chapter with bounded feasibility evidence. The complete branch definitions are in `docs/plans/CONDITIONAL_DISSERTATION_MIGRATION_PLAN_2026-09-01.md`.

The current manuscript remains readable throughout migration. A chapter is promoted to the new manuscript only when its claim ledger, figure provenance, equation necessity review, and experiment references pass.

## 9. Immediate execution checklist

- [x] Critical code/manuscript audit completed.
- [x] Current Ukrainian and compact manuscript structural audits pass.
- [x] Compatibility figure names restored without deleting the normalized names.
- [x] Freeze current branch in a baseline commit.
- [x] Create and verify the clean Path B worktree.
- [x] Add versioned config schema and run registry scaffold.
- [x] Purpose-lock a new TRB development split and exclude the previously used legacy test file.
- [ ] Finish the config-driven Article 4 packer/runner and convergence gate before training.
- [ ] Add the integrated state-contract and positional-encoding tests.
- [ ] Execute Article 4 Week 1 gates.
- [ ] Execute Article 5 Week 2 gates.
- [ ] Start the English manuscript with the approved structure and transfer ledger.

## 10. Continuation record

This file is the durable continuation point if agent context is compacted. Before resuming execution, read:

1. this master plan;
2. `docs/reviews/DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`;
3. the two article experiment plans in `docs/plans/`;
4. `docs/dissertation/DISSERTATION_REVIEWER_PROVENANCE_AND_REPRODUCTION_GUIDE_2026-08-29.md`.

Do not assume that a completed pilot is a publishable confirmed experiment. Check the run registry and gate status first.
