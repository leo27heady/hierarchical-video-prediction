# Chapter 1 — Introduction

> **Detailed fourth-article source chapter; not the binding submission narrative.** Preserve this
> chapter's measured evidence and citations, but resolve title, causal scope and claims through
> `ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md` and `docs/CLAIM_LEDGER.md`. A final assembled article
> will become the sole submission source. Older law, exact-edge and invalid Phase-C wording remains
> retired.

> **Status (2026-07-18):** detailed source synchronized through terminal E7a, E8a and E6. Written from
> `ARTICLE_SKELETON.md` §0.1/§1 and revised to the binding reach–fidelity account. Canonical chapter numbering fixed at integration:
> 1 introduction · 2 testbed · 3 mechanisms · 4 memory-horizon results · 5 generalization ·
> 6 reversibility · 7 related work · 8 conclusion. Title, packaging-reframe candidate
> (2026-07-10, external-review convergence): **"When Does Hierarchical Context Pay? Rate,
> Resolution, and Memory Span in Long-Horizon Video Prediction."** [AUTHOR: decision at venue;
> prior candidate "When Does Hierarchy Pay? A Memory-Horizon Study of Hierarchical Latent Video
> Prediction at Matched Token Budgets" remains valid — both defended by the data.]
> **Reframe status:** this chapter is written to the packaging vision (claim ledger
> `docs/CLAIM_LEDGER.md`). Completed seed/head/sampler results are stated directly; unexecuted
> transfers remain future work and do not enter the abstract or contributions.

## Abstract (packaging-reframe draft)

Hierarchical latent video models are usually motivated by efficiency, yet controlled evidence
that the hierarchy itself — rather than merely added context — improves prediction is scarce, and
our own attempts on standard benchmarks returned nulls. We argue the nulls are a property of the
*testbeds*, and build a memory-horizon testbed in which content verifiably exits a panning
viewport and re-enters ~48 frames later at an analytically known position, so that geometric
observability from a specified context is exactly computable. This does not make unseen content
information-theoretically unknowable because shared digit priors permit guesses. On this instrument we treat long-horizon
prediction as a **constrained context-packaging problem**: at a matched context-token budget,
a fixed backbone consumes context streams parameterized by spatial resolution and temporal
rate, and "hierarchy" is the family of multi-rate and/or multi-resolution packagings. The measured
answer is a set of empirical regularities under stated conditions. On an untouched holdout,
reach-differential effects confirm that long-span packaging helps most where the relevant past is
inside its reach but outside the flat comparator's. A simultaneous frontier confirms hierarchy at
the tested 896- and 1,152-token points, leaves 1,792 unresolved, and confirms flat full-resolution
context at 2,432 and 3,072; this is a tested bracket, not an interpolated law. The stride-2
512-token arm also wins, but its target interface changes and it is therefore a configuration
result. The packaging axes are not interchangeable: temporally subsampled full-resolution history
beats spatial coarsening under diffusion on all five training seeds, whereas regression and the
head × package interaction are seed-heterogeneous. A frozen NFE/sampler family preserves all 20
central effect directions, with 18 simultaneous lower bounds positive and the upper inversion
unresolved at two alternate sampler seeds. An independently trained codec preserves all four
directions at the point-estimate level, with the upper inversion again precision-limited. A
privileged pose-indexed raw-memory reference shows how much error explicit lossless access can
remove. On protected persistent-object long Moving-MNIST, hierarchy better preserves spatial
support and continuity, while flat context better preserves digit identity. The head ranking itself is
regime-dependent (regression, not diffusion, wins deterministic best-of-1 rollout). Energy-based
descent is a two-sided characterized negative at reachable scale, and an exactly invertible
stochastic recurrence (gradients equal to backpropagation through time at constant activation
memory) is proven, measured, and honestly delimited.

## 1.1 The question

Hierarchical structure is one of the oldest proposals in video prediction: let an upper level
carry slow, abstract state while a lower level renders detail. It is also one of the least
verified. Published hierarchical video models are typically evaluated on aggregate pixel metrics
over standard benchmarks, where two confounds hide the mechanism. First, the benchmarks
themselves rarely *require* memory: in Moving MNIST nothing ever leaves the frame, so there is
nothing a memory hierarchy could retrieve that a short context does not already see — and indeed
we measure flat ≈ hierarchical there (§5.3, the predicted negative control). Second, hierarchy
is almost never compared at a *matched context-token budget*: an upper level that extends the
effective context is credited to "hierarchy" when the credit belongs to "more context." Our own
preliminary line of work ran into both confounds: replications of clockwork-style hierarchical
models on standard benchmarks produced clean nulls (aggregate metrics dominated by static
background; temporal abstraction acting as a capacity dump), which is what forced the question
this article answers — posed, from the start, at the packaging level rather than the
architecture level: **under bounded context and computation, how should a fixed token budget be
allocated across temporal rate and spatial resolution — and when does any hierarchical packaging
of it actually pay?** "Hierarchy" throughout means multi-rate and/or multi-resolution context
(§4.4); the spatially coarse two-level model is its first measured form, not its definition.

## 1.2 The research program this article completes

This article is the third step of a program. Its first step was a **two-stage hierarchical
adaptation of video prediction** — the V-BLT architecture [REF: A1], in which a local byte-level
encoder/decoder pair surrounds a global latent transformer and an entropy criterion adaptively
allocates capacity: the first realization, in our line of work, of the two-level decomposition
whose mature form is studied here. Its second step, TempVerseFormer [REF: A2], built the
*machinery* for long temporal horizons: reversible transformer blocks chained along time, giving
time-agnostic backpropagation whose activation memory does not grow with the horizon — validated
on deterministic rotation dynamics. The present article joins the two lines and measures the
conditions under which the joined system pays: (i) it instruments the question with a testbed
whose memory demand is exact and whose evaluation is stratified by an analytic hardness label;
(ii) it selects the per-frame generation mechanism by controlled comparison rather than
convention; (iii) it maps the hierarchy-versus-flat frontier across token budgets — and then the
**shape of the hierarchy itself**: spatial versus temporal abstraction, depth, and allocation,
compared as budget-matched families in one grid (§4.4); and (iv) it
extends the reversibility machinery to stochastic recurrences as a correctness-and-enabler
theorem, with an honest account of where it cannot help. The framing was fixed before the
results: *earlier work proposed the machinery; this article measures the conditions under which
it pays* — so a null on any axis is a boundary-condition result, not a self-contradiction. The
measured outcome delivered both: a reach–fidelity frontier with tested boundaries (Ch. 4) and two characterized
negatives (Ch. 3, Ch. 6) that scope it.

## 1.3 The four moves

1. **A purpose-built memory-horizon testbed** (Ch. 2). A 64-px viewport pans a 192-px toroidal
   world under a recorded momentum action walk, $x_t = (x_{t-1} + 4a_t) \bmod 192$: loop period
   ≈ 48 frames, 67% of the world off-screen at any instant. Because the world is procedural and
   the actions are recorded, "this content left the view $k$ frames ago and is now re-entering"
   is an exactly computable label, and every evaluation in the article is stratified by it.
2. **Mechanism science at matched capacity** (Ch. 3). Regression, diffusion, and energy-based
   descent on one frozen two-level latent interface, one backbone, one harness. Regression
   averages over possible futures; diffusion commits to sharp, correctly located content; and
   the energy-based route yields a two-sided characterized negative (landscape ordering ×
   descent reachability dissociate), run on the official implementation at the scale a 16 GB
   device can reach, with the configuration gap tabulated rather than hidden.
3. **The frontier result** (Ch. 4). Hierarchy versus a token-budget-matched flat control over
   rollouts of 152 autoregressive frames after a 48-frame ground-truth warm-up, stratified by
   horizon × analytic hardness, across SIX matched budgets (512–3,072 tokens).
   The simultaneous family confirms hierarchy at the tested 896- and 1,152-token lower/interior
   points, leaves 1,792 unresolved, and confirms flat fidelity at 2,432 and 3,072. The frontier
   question is then posed across hierarchy *shapes* (§4.4): spatial and
   temporal abstraction, depth, and allocation, compared as budget-matched families. A second
   operating point with a ~213-frame loop tests self-fed robustness: single-step retrieval over
   slot memory is real and span-threshold-gated, but the seed-0 late inversion does not replicate
   at seeds 1–2. This is a negative replication and scope condition, not a trustworthy-memory law.
4. **Supporting theory** (Ch. 6). An exactly invertible stochastic recurrence (store-the-noise):
   closed-form inverse, reconstruction exact to numerical precision, reverse-replay gradients
   equal to BPTT (cosine similarity 1.000000), activation memory constant in rollout length —
   plus a three-axis taxonomy that closes, by measurement, the axis where reversibility cannot
   pay (inner optimization loops: 0.26% of peak vs checkpointing).

## 1.4 Contributions

- **C1.** A memory-horizon testbed with exact re-entry-stratified evaluation: analytic hardness
  labels from recorded control tracks, per-operating-point floors and ceilings, clip-clustered
  paired-bootstrap statistics — reproducible on a single 16 GB GPU (Ch. 2).
- **C2.** A matched-capacity, matched-NFE comparison of prediction mechanisms with the
  latent-standardization corrective; a controlled replication of "regression averages, diffusion
  commits," and its regime-dependent reversal at rollout (Ch. 3–4).
- **C3.** A two-sided, official-code, scale-scoped EBT capability study: ordering repaired under
  one recipe, reachability failing under both — with the configuration-gap table making
  extrapolation risk explicit (Ch. 3).
- **C4.** The frontier result: comparator-specific reach bands and a simultaneous six-budget
  family locate tested lower/interior hierarchy wins and upper flat-fidelity inversions without
  interpolation; a five-seed package factorial and a 20-effect NFE/sampler family delimit its
  head, training-seed, inference and codec scope. This is paired with an outcome-independent
  negative replication of the proposed late trust inversion, a privileged explicit-access
  reference, a standard-benchmark negative control and a valid persistent-object external
  boundary separating support/continuity from identity fidelity.
- **C5.** The exactly invertible stochastic recurrence and the three-axis reversibility taxonomy
  with the measured inner-loop no-go (Ch. 6).

## 1.5 Reader's map

Chapter 2 defines the instrument; Chapter 3 selects the mechanism; Chapter 4 carries the
headline frontier and its boundary conditions; Chapter 5 tests the selected recipe on external
datasets (with the memory-horizon null as a *prediction*); Chapter 6 gives the reversibility
theory and its honest limits; Chapter 7 positions the work and lists explicit non-claims;
Chapter 8 concludes. Journal format note: Chapters 3 (Part B) and 6 compress to sections in a
~12-page version; the dissertation carries them in full.
