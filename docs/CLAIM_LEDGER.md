# Claim ledger — scientific source of truth

**Revision:** 2026-07-12 verified post-c2 evidence integration
**Rule:** a headline claim must name its testbed, intervention, comparator, metric population,
representation, head, seeds, training regime, and evidence artifact.  Opposite signs are not
equivalence; tested brackets are not continuous crossings; configuration changes are not pure
factor interventions.

Context packaging is the manipulated design space.  **Reach, fidelity, and trust are explanatory
axes to be tested**, not assumed laws:

- reach: the relevant past is included in the package;
- fidelity: relevant information survives the stored representation;
- trust: stored information remains usable after self-fed generation.

Status vocabulary: `MEASURED`, `CONFIRMATION_REQUIRED`, `EXPLORATORY_POST_HOC`,
`BLOCKED_VALIDITY`, `CONDITIONAL`, and `FUTURE`.

## Stable L1–L10 compatibility map

The public article/dissertation identifiers L1–L10 remain stable. The R/F/T/M/D/X/V rows below
are finer-grained evidence views, not replacements that invalidate existing citations.

| Stable ID | Current evidence rows | Stable meaning / status |
|---|---|---|
| L1 | R1, R2 | Reach threshold and comparator-specific observability mechanism; measured on the current pool, untouched confirmation required. |
| L2 | F1 | Upper-budget saturation/inversion regularity; tested configurations and optimizer regimes, never an exact crossing law. |
| L3 | F2 | Stride-family 512-token configuration effect; causal stride isolation remains conditional on the fixed-target control. |
| L4 | F3 | Spatial-versus-temporal package shape; diffusion replicated, regression head/seed-sensitive. |
| L5 | D1 | Depth-plus-objective/allocation probe on a single-scale testbed; appendix evidence, not a general depth law. |
| L6 | T1 | Addressable-versus-trustworthy memory boundary; replication required before generalization. |
| L7 | M1, M2 | Prediction-mechanism regime dependence and the EBT characterized negative. |
| L8 | X1, P1 | External boundary programme; short controls measured, legacy Phase C factor claims invalid, hybrid Phase C conditional. |
| L9 | V1 | Reversible temporal optimization as a dissertation enabler, not the cause of package gains. |
| L10 | future counterfactual/multiscale study | A genuine multiscale/depth or counterfactual hidden-state benefit remains a future claim, not required for present closure. |

Old prose may cite L1–L10 through this map. New evidence discussions may append sublabels such as
L1-R, L2-F or L6-T, but no source should cite a missing ledger row.

## Central article claims

| ID | Defensible claim | Evidence now | Status | Required limitation / next gate |
|---|---|---|---|---|
| R1 | On loopmom under the diffusion head, packages whose reach includes the ~48-frame re-entry past can outperform shorter token-matched packages on the tested development set. | 896-token `hier-min` effects +0.0399/+0.0358 across seeds 0/1; at 1,152 tokens `hier-long` beats `flat-budget` by +0.0189/+0.0354/+0.0299 across seeds 0/1/2. | `MEASURED; CONFIRMATION_REQUIRED` | Current test pool informed development. Confirm once on a new hash-pinned holdout with complete metrics. |
| R2 | The effect is concentrated in records visible to the longer package but outside the comparator's reach. | `hier-min` vs `flat-min`, `14 < dt <= 48`: +0.0880 and +0.0809; B1 vs `flat-short`, `8 < dt <= 48`: +0.0675 and +0.0709, clip-bootstrap intervals excluding zero. | `EXPLORATORY_POST_HOC` | Preregister these geometry-derived bands as primary on the untouched confirmation set. |
| F1 | At B16x30k, 1,792 is the largest tested budget favouring the spatial package and 2,432 the smallest tested budget favouring flat full-resolution context. | R-F/R-G tested points; at 2,432 flat wins at seed 0 under B16x30k by 0.0116 and under B8x60k by 0.0077; flat also wins at 3,072 on two seeds. | `MEASURED; REGIME_QUALIFIED` | No monotonicity, uniqueness, continuous crossing or asymptotic edge is established. c2 holds examples approximately fixed while changing batch/update/noise/grouping/EMA regime. |
| F2 | The stride-2 B1 configuration strongly outperforms its compared short-flat configuration. | Diffusion +0.0330/+0.0338; regression seed 0 +0.0639 [0.0516, 0.0763]. | `MEASURED_CONFIGURATION_EFFECT` | Context stride, target slots, auxiliary loss, sequence length and optimization difficulty change together. Fixed-target control is prepared. |
| F3 | Temporal subsampling and spatial coarsening are not empirically interchangeable at the tested static-world operating point. | Diffusion temporal advantage +0.0197/+0.0046/+0.0111 at seeds 0/1/2; regression package signs +0.0107/−0.0104 at seeds 0/1. | `MEASURED; DIFFUSION_REPLICATED; HEAD_AND_SEED_SENSITIVE` | Diffusion has the same sign in three seeds. Regression cancellation is not equivalence. Report every seed-specific contrast; complete regression seed 2 only after H3b parity because it extends a legacy factorial. |
| T1 | Memory addressable under teacher forcing can become unreliable under self-fed rollout at the tested long-loop operating point. | Retrieval gate wins when span covers the loop; rollout inversion appears after roughly 24 frames in the decisive seed-0 programme. | `MEASURED_BOUNDARY; REPLICATION_REQUIRED` | Do not call a general trustworthy-memory law. Replicate or demote; add refresh/corruption and 24/48/96 horizon diagnostics. |

## Supporting and boundary claims

| ID | Defensible claim | Evidence now | Status | Scope |
|---|---|---|---|---|
| M1 | Generation-head preference depends on evaluation regime. | Regression wins deterministic best-of-1 rollout; diffusion has different coverage behaviour; seed-resolved package interaction. | `MEASURED` | Fixed backbone/representation and reachable scale; protocol allocation must be reported. |
| M2 | EBT is a characterized negative at the tested scale: ordering and descent reachability dissociate. | Official-code controls and two recipes. | `MEASURED_NONCORE` | Does not establish impossibility at paper scale. Move from main fourth article. |
| D1 | Adding a third level at fixed budget costs performance on the single-scale loopmom probe. | B3 seed 0 −0.0196 [−0.0248, −0.0146] versus two-level arm; disclosed objective/allocation confounds. | `MEASURED_APPENDIX` | A depth-plus-objective-allocation probe, never a general statement that depth hurts. |
| X1 | Short-horizon Moving-MNIST and TaxiBJ are competence/boundary anchors, not strong equivalence or transfer evidence. | T7 descriptive near-null; TaxiBJ in-band competence/economy. | `MEASURED_DESCRIPTIVE` | Moving-MNIST uses one seed/unequal budgets/no paired equivalence test; TaxiBJ is not a memory-re-entry test. |
| V1 | The stochastic reversible recurrence supports exact reverse replay and constant activation memory relative to BPTT. | Reconstruction/gradient/memory panels. | `MEASURED_DISSERTATION_STRAND` | Enabler, not cause of Chapter 4 gains; novelty scoped against reversible RNNs. Move out of core fourth article. |

## Invalid, blocked, or retired claims

| ID | Retired wording or analysis | Status | Remedy |
|---|---|---|---|
| P1 | Legacy Phase C identity, trajectory, coverage and replicated equivalence. | `BLOCKED_VALIDITY` | Persistent IDs/classes/trajectories; official train/test glyph split; Hungarian assignment; miss/extra penalties; coverage; classifier validation; justified margins. |
| P2 | “Exact arithmetic edge,” “switch occurs at 896,” or a unique crossing in (1792,2432). | `RETIRED_OVERCLAIM` | Use largest/smallest tested winning configurations and tested brackets only. |
| P3 | B1 as a pure context-stride intervention. | `RETIRED_CAUSAL_OVERCLAIM` | Call it a whole-configuration result until the fixed-target-interface control lands. |
| P4 | B8x60k as twice the data or closer to asymptotic training. | `RETIRED_REGIME_OVERCLAIM` | Same examples; changed batch, updates, noise, grouping and effective EMA horizon. |
| P5 | Two-seed regression “tie” or equivalence. | `RETIRED_EQUIVALENCE` | Report +0.0107 and −0.0104 separately plus the same-direction difference-in-differences. |
| P6 | Geometric labels as information-theoretic necessity. | `RETIRED_TERMINOLOGY` | Use exact geometric observability. Reserve necessity for paired counterfactual worlds. |
| P7 | OOD blinding loss as a lower bound on stream contribution. | `RETIRED_INTERPRETATION` | OOD corruption can inflate loss. Prefer in-distribution shuffle/freeze/cross-clip or trained missing-stream controls. |

## Cost and comparison vocabulary

Every table must report separately:

- context tokens;
- target tokens;
- total sequence length;
- parameters;
- training examples;
- optimizer updates;
- batch size and EMA;
- per-step and total NFE;
- runtime and peak VRAM when measured.

Use “matched context-token budget,” “parameter-matched,” “matched examples,” or “matched total NFE”
only when literally true.  Do not use “matched compute” as a substitute for these distinctions.

## Confirmation and expansion order

1. Finish the already-running seed-2 trio and c2; stop before Phase C.
2. Freeze current claims and analysis/table shells.
3. Validate the complete metric suite and immutable provenance on CPU.
4. Evaluate the untouched confirmation set once.
5. Run the fixed-target B1 control if F2 remains central.
6. Replicate/isolate trust if T1 remains title-level.
7. Complete the regression seed-2 factorial only if F3 remains central.
8. Run redesigned Phase C only after its validity and classifier gates.
9. Treat counterfactual/dynamic-content and natural re-entry studies as future work.

Pre-C seed-2 and c2 results must be integrated as new evidence rows without silently upgrading the
scope of these claims.
