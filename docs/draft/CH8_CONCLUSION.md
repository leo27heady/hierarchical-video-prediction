# Chapter 8 — Conclusion

> **Detailed fourth-article source chapter; not the binding submission narrative.** Revise before
> assembly under `docs/CLAIM_LEDGER.md` and the binding compact
> reach–fidelity specification. “Laws,” exact switch points, invalid Phase C factor claims and pooled
> regression-equivalence wording remain superseded.

> **Status (2026-07-16):** detailed source synchronized through E7c. Expected transfer shells are
> future work and do not enter conclusions. The claim ledger and binding compact govern wording.

## 8.1 One claim

**Under bounded context, long-horizon prediction depends on how a context-token budget is packaged
across temporal reach and representational fidelity.** Context packaging is the manipulated design
space; reach and fidelity are the explanatory axes supported by the controlled evidence. The
project does not assert that a hierarchy always wins, that token arithmetic alone fixes an exact
edge, or that the sampled frontier is monotone.

On LoopMom, packages that include the memory-relevant approximately 48-frame past outperform
shorter token-matched packages at the tested lower/interior budgets. On the untouched holdout, the
registered `hier-min` versus `flat-min` reach-band effects are +0.0980 and +0.0914 across seeds
0/1. The simultaneous frontier confirms hierarchy at 896 and 1,152 tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. Therefore 1,152 is the
largest tested budget with a family-wise-confirmed hierarchy win and 2,432 the smallest with a
confirmed flat win; no continuous or unique crossing is inferred.

Package shape is also conditional. Temporally sparse full-resolution history beats the spatially
coarse package under diffusion on all five training seeds, whereas regression and the cross-head
interaction change sign. The stride-2 512-token arm is retained as a whole-configuration effect
because its target interface also changes. E7c shows that four central diffusion directions retain
their point-estimate signs across the frozen NFE/sampler family, while qualifying the precision of
the upper inversion at two alternate sampler seeds. The spatially coarse two-level model is one
measured implementation of the design space, not the conclusion itself.

## 8.2 Empirical regularities under fixed conditions

Every regularity below is scoped by construction — testbed, head, representation, seed level — with the
single source of truth in the claim ledger (`docs/CLAIM_LEDGER.md`); a regularity appears in this
section only when its evidence row is complete, and its wording carries its scope.

- **L1 — Reach regularity.** Long-reach packages win at tested lower/interior budgets when they
  include the relevant re-entry past and their comparators do not. The registered pair-specific
  reach bands confirm this on the untouched holdout across seeds 0/1. Scope: LoopMom and the
  frozen diffusion representation; geometric observability is exact, statistical necessity is not.
- **L2 — Fidelity inversion regularity.** Once the flat full-resolution package approaches the
  relevant span, abstraction cost can dominate. The simultaneous family confirms hierarchy
  through the tested 1,152-token point and flat at the tested 2,432- and 3,072-token points;
  1,792 is unresolved. This is a tested bracket, not a crossover interval or monotone law. The
  B8×60k result is a matched-example changed-optimizer-regime control, not more-data evidence.
- **L3 — Stride-family configuration effect.** The stride-2 512-token configuration wins strongly
  across diffusion seeds, consistent with its larger span per token. It also changes target
  slots, target-token count, auxiliary supervision and total sequence length, so it does not
  isolate context stride. A fixed-target control is required only if causal stride wording returns.
- **L4 — Package-shape boundary.** Spatial coarsening and temporal sparsification are not
  interchangeable under the diffusion head at this operating point: strided full-resolution
  history wins on all five seeds (mean +0.0098, t95 [+0.0023, +0.0173]). Regression has two
  temporal and three spatial material winners, and the cross-head interaction t95 crosses zero.
  Scope: one adaptively used static-world pool (maximally kind to temporal subsampling) and this
  autoencoder's 4×4 code; diffusion is stable, but cross-head generalization is unsupported.
- **L5 — Depth hypothesis (allocation probe, not a law).** Depth should pay only where the data
  contain another scale to own. Measured (seed 0): the three-stage probe *costs* at fixed budget
  on single-scale loopmom (−0.0196 [−0.0248, −0.0146] vs the two-stage arm; null vs flat; the
  class-lowest sharpness matching the disclosed loss-dilution signature) — reported as a
  depth-plus-objective-allocation result with three inseparable mechanisms (loss dilution,
  optimization load, top-stream information floor), never as "depth hurts."
  A depth *law* requires the multi-timescale testbed of §8.5.
- **L6 — Self-fed reliability: negative replication, not a law.** Single-step retrieval is real
  and span-threshold-gated (2/2 above-loop versus 0/4 at-or-below-loop), and the early self-fed
  hierarchy benefit is positive at all three seeds. The late inversion appears only at seed 0;
  seeds 1–2 instead show positive, CI-clean late effects. No stable trust boundary is established.
- **F4 — Inference robustness.** On a separate purpose-locked set, all 20 registered E7c effects
  retain direction across NFE 8/16/32 and sampler seeds 0/1/2; 18 simultaneous lower bounds are
  positive. The 2,432-token inversion is family-wise unresolved at alternate sampler seeds 1/2.
  Scope: frozen seed-0 checkpoints and one codec; this is not training-seed or codec robustness.

**Supporting boundary conditions (measured):** retrieval *emerges* only under a sufficient
optimization budget (null at 6k steps, two-seed win at 20k; matched-budget delta doubles at 2×
budget — short-budget ablations spuriously kill memory mechanisms); the matched-budget win is
mechanism-agnostic at the interior point (both heads, two seeds); the generation head is
regime-dependent (diffusion wins coverage, regression wins best-of-1 rollout; no "one best
head" survives); and the negative control holds — on standard Moving MNIST, with no memory
horizon, flat ≈ hierarchical (180.91 vs 181.29), as the scoped account predicts. Legacy Phase C
identity/trajectory conclusions are excluded; a redesigned Phase C remains conditional on its
persistent-object, assignment, classifier and article-utility gates.

## 8.3 The two honest negatives that scope the claim

(i) **Energy-based descent** is a two-sided characterized negative at 16 GB-reachable scale:
one published recipe yields a stably mis-ordered landscape (21/21 probes); the other repairs the
ordering (15/15 from step 6k) while gradient descent — including the paper's own inference
recipe — still fails to reach structured outputs. Ordering and reachability dissociate; the
configuration gap to the published setting is tabulated. (ii) **Reversibility** delivers exact
gradients at constant memory versus BPTT (OOM at K=120 vs ≈1.1–1.2 GB through K=160) but is
~Pareto with checkpointing on the temporal axis and provably cannot pay on the
inner-optimization axis (0.26% of peak, measured; ≈0.6–1.0% analytic band) — and invertibility
does not imply stability (perturbation growth 2.7e4× at Lip(F) ≤ 1).

## 8.4 Practical reading

For a practitioner the frontier reads as a sizing rule: **spend tokens on span until the memory
horizon is covered — using the cheapest abstraction family that covers it — and only then on
resolution and recency; add depth only when the data has scales left to own.** At the tightest budgets that cover the
horizon, the hierarchical packaging delivers the largest gains (a quality that flat context
reaches only at a larger token budget in the measured frontier, with quadratic attention making
the compute gap larger still); the exact upper multiple remains optimization-budget-sensitive.
Within the spatially coarse family, regression plus long history is a strong measured arm, but the
five-seed factorial shows that package ordering under regression is heterogeneous. Across
packaging families, sparse full-resolution history is stable under diffusion across five seeds.
E7c makes this practical reading less dependent on a single inference allocation, while E7a must
still test whether it depends on one learned latent geometry.

## 8.5 Future work

Four directions follow from the measured boundaries rather than from speculation: (1) paper-scale
EBT — the ordering-repaired branch leaves scale as the open variable; (2) a newly preregistered,
multi-seed and multi-task self-fed reliability study, if this question remains important, crossing
teacher-forced, self-fed and controlled-refresh conditions rather than extending the unreplicated
seed-0 boundary; (3) a natural-video instantiation of the re-entry-stratified protocol (hardness
labels estimated rather than exact), building on the OpenSTL bridge; (4) a sharp renderer over the
frozen latents (the adversarial-renderer precedent) to lift the decoder-owned blur ceiling that
bounds all arms equally. A fifth
follows from the shape study's measured depth-plus-objective-allocation loss: the arena where depth *can* win is a
**multi-scale testbed with a visible slow factor** — a slowly evolving, always-visible global
variable (period ≫ the pan loop) layered under the fast re-entry dynamics. Slow-but-visible
state is continuously refreshed, so a deep hierarchy's top level can own it without depending on
long occluded recall; this is the CW-VAE slowness regime rather than the retrieval regime, and the
natural home for the re-based factor-4 pyramid
(16×16 / 4×4 / 1×1) that a 16 GB budget cannot pair with long flat windows.
