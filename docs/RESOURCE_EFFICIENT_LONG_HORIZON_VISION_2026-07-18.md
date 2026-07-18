# Resource-efficient long-horizon vision

**Recorded:** 2026-07-18
**Status:** binding narrative and authoring decision for the dissertation and fourth article
**Scope constraint:** the approved dissertation topic and the content/identity of the first three
article strands remain unchanged. This document governs their synthesis in the dissertation and
the freely revisable fourth article.

## 1. Authoring-language decision

All new and revised dissertation and article prose must be written in **English**:

- English is the authoritative working and review language for chapters, captions, tables,
  appendices, references, claim wording and figure annotations.
- Do not automatically translate the dissertation into Ukrainian.
- Existing Ukrainian source headings or archival originals may remain as provenance, but they do
  not determine the language of new prose.
- If a Ukrainian submission is later required, treat it as a separate localization pass after the
  English scientific text is frozen, and perform it only on the author's explicit request.
- Scientific edits must land in the actual English article/dissertation sources, not only in
  planning or commentary documents.

## 2. Dissertation-level systems thesis

Long-horizon video prediction has two distinct memory costs:

1. **Context/state memory:** the cost of retaining past information for future prediction.
2. **Activation memory:** the cost of retaining intermediate computation for training through a
   long temporal process.

The dissertation addresses these costs through complementary mechanisms:

- **Hierarchical/multirate context packaging** reduces context-state cost by allocating cheap
  coarse tokens to distant history and expensive fine tokens to recent detail.
- **TempVerseFormer and reversible temporal optimization** reduce temporal activation-memory
  growth by reconstructing intermediate states during backward propagation rather than retaining
  the entire temporal tape.

The preferred compact synthesis is:

> **Compress what must be remembered; reconstruct what need not be stored.**

The more formal defended proposition is:

> Hierarchical packaging improves the task-relevant information retained per context token,
> while reversible temporal optimization reduces the activation memory required per modeled
> temporal step. These are orthogonal resource axes whose operating regions and limitations must
> be measured separately.

The project must never imply that reversibility causes the hierarchical prediction advantage, or
that spatial hierarchy is the only form of efficient long-range context.

## 3. Fourth-article scientific identity

**Context packaging** is the manipulated design space. **Reach and fidelity** are the confirmed
explanatory axes. **Trust** is a measured self-fed robustness question, not a general law.

The central article/dissertation result is:

> At matched context-token budgets, hierarchical coarse-to-fine memory outperforms dense flat
> full-resolution memory when the flat window cannot reach the task-relevant past. Once the
> budget is sufficient for flat context to retain that past at full resolution, the hierarchy
> advantage disappears and can reverse.

The preferred memorable form is:

> **Hierarchy buys reach; flat context buys fidelity.**

This is a conditional matched-budget advantage, not universal hierarchical dominance.

## 4. Memory-efficiency interpretation

For the measured 64×64 latent representation, one fine frame costs 64 context tokens and one
coarse frame costs 16. A 48-frame history therefore costs:

| Package | Context tokens | Temporal reach |
|---|---:|---:|
| 48 fine frames | 3,072 | 48 frames |
| 48 coarse frames | 768 | 48 frames |
| 48 coarse + 2 recent fine | 896 | 48 frames |
| 48 coarse + 6 recent fine | 1,152 | 48 frames |

Thus the measured hierarchy obtains 48-frame reach with approximately 2.7–3.4× fewer context
tokens than dense fine-level storage. The causal evidence is stronger than arithmetic alone:
on the untouched LoopMom confirmation set, 896- and 1,152-token hierarchical packages beat their
matched-budget dense-flat comparators, and the largest effects occur on records visible to the
long package but outside the flat comparator's reach.

The inference-safe wording is:

> For a fixed context-token budget, coarse-to-fine allocation extends addressable temporal reach
> and improves long-horizon prediction when dense full-resolution storage cannot cover the
> task-relevant past.

Three adjacent statements must remain distinct:

- **Confirmed:** better performance at the same tested budget in the hierarchy-favouring region.
- **Arithmetic and measured reach:** the same 48-frame reach with fewer context tokens.
- **Secondary until formally analysed:** the same prediction performance with fewer tokens.

An iso-performance claim requires an explicit non-interpolated Pareto/threshold analysis with
uncertainty. Do not infer a continuous memory-efficiency curve between tested configurations.

## 5. Current evidence that must appear

The complete untouched E2 frontier is material evidence, including both directions:

| Tested context budget | Result |
|---:|---|
| Near-bottom 480 versus 512 | seed 0 unresolved; seed 1 flat-favouring |
| 896 | hierarchy wins on seeds 0–1 |
| 1,152 | hierarchy wins on seeds 0–2 |
| 1,792 | directionally hierarchical but simultaneous interval unresolved |
| 2,432 | flat wins under both tested optimizer regimes |
| 3,072 | flat wins on seeds 0–1 |

The scientific interpretation is a tested **reach–fidelity phase diagram**:

- dense flat context is token-inefficient in reach-limited regimes;
- hierarchy is advantageous when cheap coarse storage includes otherwise inaccessible history;
- the abstraction cost becomes visible after dense flat context can cover the relevant past;
- the upper flat-fidelity inversion is part of the result, not an inconvenient exception.

The following boundaries must also remain visible:

- temporally subsampled full-resolution history is a strong alternative and, under diffusion, its
  tested package-shape effect is positive across five training seeds;
- regression package effects and the head × package interaction are seed-heterogeneous;
- adding a third level in the measured single-scale B3 probe hurts, with disclosed
  objective/allocation confounds;
- the seed-0 late trust inversion does not replicate at seeds 1–2;
- privileged raw pose-indexed access outperforms all tested neural packages but is not
  storage-, compute- or information-matched;
- E6 returns a valid `MIXED_BOUNDARY` on persistent-object long Moving-MNIST: union error favors
  hierarchy on all three seeds; trajectory and coverage favor hierarchy cleanly on seeds 0 and 2;
  flat context preserves digit identity better on all three seeds; foreground-only error is
  unresolved. This is the external expression of the same reach–fidelity tension: coarse memory
  better preserves where an object is and whether it persists, while fine flat memory better
  preserves what digit it is.

These findings do not weaken the contribution. They define when hierarchy pays and prevent an
architecture-promotion claim from replacing the controlled scientific result.

## 6. TempVerseFormer and reversibility bridge

TempVerseFormer is the dissertation's measured response to the activation-memory axis. The fourth
article's context-packaging result is the response to the stored-context axis. Their relationship
must be presented as a systems synthesis:

| Resource problem | Measured response | Defensible meaning |
|---|---|---|
| Distant full-resolution history is token-expensive | Hierarchical/multirate packaging | More task-relevant temporal reach per context token |
| Long temporal backpropagation stores activations | Reversible temporal transformations | Remove linear activation-tape growth where the temporal recurrence dominates |
| Reversibility may be applied where other operations dominate memory | Checkpointing-baselined applicability study | Reversibility is conditional, not a blanket memory-speed win |

The current DLH memory-axis study must be disclosed: checkpointing provides the dominant measured
gain (approximately 4× larger trainable window), while reversible gating adds only approximately
4% beyond checkpointing because per-frame convolutions dominate the activation tape. The
inner-optimization-loop experiment likewise shows only a 14 MB / 0.26% peak saving versus
checkpointing at roughly twice the wall-clock time.

Therefore:

- retain TempVerseFormer's time-axis reversible contribution where recurrence dominates;
- present the stochastic exact-replay extension and three-axis taxonomy as a scoped systems
  contribution;
- do not claim that reversibility materially improves the current hierarchical predictor unless a
  direct end-to-end comparison demonstrates it;
- do not claim that the two memory savings multiply automatically.

## 7. Reporting and qualitative-selection policy

The narrative may foreground the hierarchy-favouring long-memory regime, but it may not hide
material flat-superior regimes or failure modes.

- The main article must include the complete frontier or a direct pointer to it.
- At least one main-text table/figure must show the hierarchy-favouring lower/interior region and
  the flat-favouring upper inversion together.
- Qualitative examples must follow a disclosed selection rule such as a registered causal-band
  median, fixed quantiles or fixed clip identifiers.
- A hierarchy-winning main example is acceptable only alongside aggregate inference and an honest
  high-budget/flat-favouring boundary panel or appendix reference.
- Do not select examples solely because they look favourable.
- Negative/null outcomes remain in the claim ledger, article limitations and dissertation
  discussion.

## 8. Required synthesis artifacts

The final English dissertation/article package should include:

1. **Reach–fidelity frontier table:** budget, coarse/fine allocation, reach, target interface,
   parameters, per-seed effects and simultaneous intervals.
2. **Memory-efficiency arithmetic table:** tokens and actual bytes per stored frame/stream, with
   48-frame reach ratios and no unmeasured interpolation.
3. **Reach-differential mechanism table/figure:** whole-population and pair-specific causal-band
   results with false-positive-sensitive corroboration.
4. **Two-panel resource frontier:**
   - prediction quality versus context tokens/reach;
   - activation VRAM versus temporal horizon for BPTT, checkpointing and reversible replay.
5. **Applicability/boundary table:** where hierarchy, temporal subsampling, dense flat context,
   checkpointing and reversibility pay or do not pay.
6. **Optional post-hoc Pareto analysis using frozen artifacts:** minimum measured tokens at selected
   error thresholds and non-dominated measured configurations. It must be labelled exploratory
   unless its contrast family was frozen before protected access.

## 9. Final narrative guardrail

The desired dissertation message is positive but conditional:

> Resource-efficient long-horizon prediction requires managing both the representational cost of
> remembered history and the computational cost of temporal optimization. Multirate hierarchical
> packaging allocates inexpensive coarse tokens to distant history and expensive fine tokens to
> recent detail, while reversible temporal transformations remove linear activation-memory
> growth where recurrence dominates. The dissertation identifies experimentally where each
> mechanism pays and where dense flat storage, temporal subsampling or checkpointing remains
> preferable.

This wording is the binding narrative target unless later evidence requires a narrower claim.
