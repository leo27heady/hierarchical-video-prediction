# Chapter 8 — Conclusion

> **Historical conclusion; do not assemble unchanged.** The canonical conclusion is in
> `ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md`. “Laws,” exact switch points, legacy Phase C and
> pooled regression-equivalence wording below are superseded by `docs/CLAIM_LEDGER.md`.

> **Status (2026-07-08):** journal-register prose, content-complete; all numbers measured except
> the Ch. 5 [EXPECTED] transfers (generalization campaign) and the R-C/R-D robustness reads.
> R-A + R-B landed 2026-07-08: BOTH band edges (896 hier-win, 3,072 flat-win) are TWO-SEED.

## 8.1 One claim

**Under bounded context and computation, long-horizon consistency depends on how a fixed
context-token budget is allocated across temporal rate and spatial resolution. Hierarchical
packaging — any multi-rate and/or multi-resolution allocation — earns its budget exactly when it
covers task-relevant memory more cheaply than a flat full-resolution window can; the budgets at
which this happens are predictable from token arithmetic; and spatial compression, temporal
subsampling, and depth are distinct mechanisms with distinct, measured boundary conditions.**
On a testbed whose memory demand is an exact label, the frontier of long-span packaging against
flat context, measured at six matched budgets, has the bounded shape this account predicts:
at ≈500 tokens the contiguous flat window and the stride-1 coarse family do not cover the
~48-frame loop and flat wins that within-family comparison (−0.0036 [−0.0063, −0.0010]);
the stride-2 coarse family does cover it at 512 tokens and wins its separate comparison. Dense
long-span packaging wins from the geometric threshold —
48×16 coarse tokens plus a minimal fine tail ≈ 900 — where the effect is largest (+0.0399
[+0.0299, +0.0501] seed 0 and +0.0358 [+0.0267, +0.0454] seed 1 at 896 tokens — two seeds, every
horizon × hardness cell on both) through the two-seed interior point at 1,152 (+0.0189 /
+0.0354); and flat wins again at window saturation (3,072 tokens, −0.0131 / −0.0107 — two
seeds), where full-resolution memory information-dominates any coarse record of the same span.
Within the band, the *shape* of the packaging matters and is not interchangeable: the band's
lower edge follows each family's token arithmetic (896 → 512 when the coarse tick halves;
[EXPECTED-B1: s1 queued]), and a temporally-subsampled full-resolution packaging — itself a
two-rate hierarchy in the clockwork sense — matches or exceeds the spatially coarse form under
the diffusion head on this static-world testbed [EXPECTED-B2-s1/REG: seed and head resolution
queued]. The spatially coarse two-level model remains the per-level-decodable form of the claim
and the first one measured; it is not the claim itself.

## 8.2 Empirical regularities under fixed conditions

Every regularity below is scoped by construction — testbed, head, representation, seed level — with the
single source of truth in the claim ledger (`docs/CLAIM_LEDGER.md`); a regularity appears in this
section only when its evidence row is complete, and its wording carries its scope.

- **L1 — Span-threshold law.** Long-span packaging wins only once its span covers the task's
  memory horizon while flat context cannot: flat wins at 512 tokens (two-seed), hierarchy from
  896 (two-seed, every cell, +0.0399/+0.0358), the switch-on at the token-arithmetic prediction.
  Scope: loopmom; edges measured under the diffusion head (regression confirms the 1,152
  interior pair, not the edges).
- **L2 — Saturation regularity.** Once flat full-resolution context covers the same span, the
  abstraction cost can dominate: flat wins at 3,072 (two-seed, −0.0131/−0.0107), and the
  flat/hier crossover at B16×30k is measured to lie in (1,792, 2,432). R-C shows that hierarchical
  arms can mature later than flat partners, so a pre-registered B8×60k c2 control is required
  before treating that bracket as optimization-budget robust. Same scope as L1.
- **L3 — Family-edge regularity.** The band's lower edge is a property of the *family*, not the
  testbed: halving the coarse tick moved it from 896 to 512 tokens (+0.0330 [+0.0246, +0.0419],
  every cell, span dt-signature), consistent with the arithmetic prediction at both tested
  budgets; each family carries
  its own plateau level (stride-2 ≈ +0.017 above stride-1). Status: seed 0; seed-1 replication
  queued before headline use.
- **L4 — Shape law (resolving).** Spatial coarsening and temporal sparsification are not
  interchangeable: at this operating point, strided full-resolution history beats the coarse
  hierarchy under the diffusion head (+0.0197 [+0.0143, +0.0252], seed 0; seed-1 required and
  queued; regression twins queued — the head interaction is the open half). Scope: one
  static-world setting (maximally kind to temporal subsampling), this autoencoder's 4×4 code;
  data-dependence untested (a rate-stress suite is future work).
- **L5 — Depth hypothesis (allocation probe, not a law).** Depth should pay only where the data
  contain another scale to own. Measured (seed 0): the three-stage probe *costs* at fixed budget
  on single-scale loopmom (−0.0196 [−0.0248, −0.0146] vs the two-stage arm; null vs flat; the
  class-lowest sharpness matching the disclosed loss-dilution signature) — reported as a
  depth-plus-objective-allocation result with three inseparable mechanisms (loss dilution,
  optimization load, top-stream information floor), never as "depth hurts."
  A depth *law* requires the multi-timescale testbed of §8.5.
- **L6 — Trustworthy-memory regularity.** A span sufficient for single-step retrieval can fail under
  self-fed rollout: at the ~213-frame-loop operating point, retrieval is real and
  span-threshold-gated (2/2 vs 0/4) yet inverts after ~24 self-fed frames — hierarchical memory
  pays where its contents remain trustworthy.

**Supporting boundary conditions (measured):** retrieval *emerges* only under a sufficient
optimization budget (null at 6k steps, two-seed win at 20k; matched-budget delta doubles at 2×
budget — short-budget ablations spuriously kill memory mechanisms); the matched-budget win is
mechanism-agnostic at the interior point (both heads, two seeds); the generation head is
regime-dependent (diffusion wins coverage, regression wins best-of-1 rollout; no "one best
head" survives); and the negative control holds — on standard Moving MNIST, with no memory
horizon, flat ≈ hierarchical (180.91 vs 181.29), exactly as the thesis predicts. [EXPECTED-
PhaseC: the null extends to ~300-frame rollouts with identity/trajectory metrics per the
pre-registered mandate.]

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
Within the spatially coarse family, regression head + long history is the strongest replicated
recipe (reg-long: 0.1682 / 0.1503 across two seeds, mean 0.1593, at 1,152 tokens). Across packaging
families, strided full-resolution history is the best seed-0 diffusion result; its seed and head
resolution are therefore reported before any global superlative.

## 8.5 Future work

Four directions follow from the measured boundaries rather than from speculation: (1) paper-scale
EBT — the ordering-repaired branch leaves scale as the open variable; (2) distribution-matching
rollout objectives (Self-Forcing-class) whose long-horizon training memory the store-the-noise
recurrence makes affordable — the direct route to pushing the trustworthy-memory boundary
outward; (3) a natural-video instantiation of the re-entry-stratified protocol (hardness labels
estimated rather than exact), building on the OpenSTL bridge; (4) a sharp renderer over the
frozen latents (the adversarial-renderer precedent) to lift the decoder-owned blur ceiling that
bounds all arms equally. A fifth, cheaper item is measurement, not construction: mapping where
between loop 48 (works) and loop 213 (fails) the trustworthy-memory boundary sits. A sixth
follows from the shape study's measured depth-plus-objective-allocation loss: the arena where depth *can* win is a
**multi-scale testbed with a visible slow factor** — a slowly evolving, always-visible global
variable (period ≫ the pan loop) layered under the fast re-entry dynamics. Slow-but-visible
state is continuously refreshed, so a deep hierarchy's top level can own it without fighting the
trustworthy-memory boundary that self-fed occluded recall runs into; this is the CW-VAE slowness
regime rather than the retrieval regime, and the natural home for the re-based factor-4 pyramid
(16×16 / 4×4 / 1×1) that a 16 GB budget cannot pair with long flat windows.
