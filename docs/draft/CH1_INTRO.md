# Chapter 1 — Introduction

> **Canonical fourth-article source.** This chapter belongs to the authoritative CH1–CH8 article
> record. Apply `docs/CLAIM_LEDGER.md` to every claim. The separate reach–fidelity–trust draft is a
> discussion/restructuring resource, not a replacement manuscript; older law, exact-edge and
> invalid Phase-C wording remains retired.

> **Status (2026-07-07):** journal-register prose, content-complete. Written from
> `ARTICLE_SKELETON.md` §0.1/§1 and updated to the measured 2026-07-07 state (frontier band with
> lower edge; pair-RED closure; T7 bridge). Canonical chapter numbering fixed at integration:
> 1 introduction · 2 testbed · 3 mechanisms · 4 memory-horizon results · 5 generalization ·
> 6 reversibility · 7 related work · 8 conclusion. Title, packaging-reframe candidate
> (2026-07-10, external-review convergence): **"When Does Hierarchical Context Pay? Rate,
> Resolution, and Memory Span in Long-Horizon Video Prediction."** [AUTHOR: decision at venue;
> prior candidate "When Does Hierarchy Pay? A Memory-Horizon Study of Hierarchical Latent Video
> Prediction at Matched Token Budgets" remains valid — both defended by the data.]
> **Reframe status:** this chapter is written to the packaging vision (claim ledger
> `docs/CLAIM_LEDGER.md`); clauses whose final strength depends on the queued seed/head runs
> carry [EXPECTED-…] markers per the standing convention.

## Abstract (packaging-reframe draft, ~250 words; [EXPECTED] clauses finalize with S1/reg runs)

Hierarchical latent video models are usually motivated by efficiency, yet controlled evidence
that the hierarchy itself — rather than merely added context — improves prediction is scarce, and
our own attempts on standard benchmarks returned nulls. We argue the nulls are a property of the
*testbeds*, and build a memory-horizon testbed in which content verifiably exits a panning
viewport and re-enters ~48 frames later at an analytically known position, so that "hard because
it must be remembered" is an exact, computable label. On this instrument we treat long-horizon
prediction as a **constrained context-packaging problem**: at a matched context-token budget,
a fixed backbone consumes context streams parameterized by spatial resolution and temporal
rate, and "hierarchy" is the family of multi-rate and/or multi-resolution packagings. The
measured answer is a set of empirical regularities under stated conditions. Long-span packaging wins
the band of budgets in which its span covers the memory horizon while a flat window cannot —
from the geometric span threshold (≈900 tokens, predicted before measurement; the largest
effect, +0.0399/+0.0358 across two seeds) to a B16×30k upper-edge bracket in
(1,792, 2,432) tokens — and inverts outside that measured band. R-C shows the upper edge is
potentially optimization-budget-sensitive, so a B8×60k c2 control is pre-registered. The band's lower edge moves with the
packaging family's token arithmetic (896 → 512 tokens when the coarse tick halves;
[EXPECTED-B1: two-seed on replication]). The packaging axes are not interchangeable: temporally
subsampled full-resolution history matches or exceeds the spatially coarse hierarchy under the
diffusion head on this static-world testbed [EXPECTED-B2-s1/REG: strength and head-dependence
finalize with the queued seed and regression runs], while the head ranking itself is
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
measured outcome delivered both: a win band with predictable edges (Ch. 4) and two characterized
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
   The hierarchy wins exactly the band where its coarse span covers the ~48-frame loop and the
   flat window does not — the band's lower edge lands at the geometric prediction (48×16 coarse
   tokens), its upper edge is bracketed by measurement to (1,792, 2,432) — and inverts outside
   the band. The frontier question is then posed across hierarchy *shapes* (§4.4): spatial and
   temporal abstraction, depth, and allocation, compared as budget-matched families. A second
   operating point with a ~213-frame loop bounds the mechanism from above: single-step retrieval
   over slot memory is real and span-threshold-gated, but does not survive pure self-feeding at
   this scale (the trustworthy-memory boundary).
4. **Supporting theory** (Ch. 6). An exactly invertible stochastic recurrence (store-the-noise):
   closed-form inverse, reconstruction exact to numerical precision, reverse-replay gradients
   equal to BPTT (cosine similarity 1.000000), activation memory constant in rollout length —
   plus a three-axis taxonomy that closes, by measurement, the axis where reversibility cannot
   pay (inner optimization loops: 0.26% of peak vs checkpointing).

## 1.4 Contributions

- **C1.** A memory-horizon testbed with exact re-entry-stratified evaluation: analytic hardness
  labels from recorded control tracks, per-operating-point floors and ceilings, clip-clustered
  paired-bootstrap statistics — reproducible on a single 16 GB GPU (Ch. 2).
- **C2.** A matched-capacity, matched-compute comparison of prediction mechanisms with the
  latent-standardization corrective; a controlled replication of "regression averages, diffusion
  commits," and its regime-dependent reversal at rollout (Ch. 3–4).
- **C3.** A two-sided, official-code, scale-scoped EBT capability study: ordering repaired under
  one recipe, reachability failing under both — with the configuration-gap table making
  extrapolation risk explicit (Ch. 3).
- **C4.** The frontier result: hierarchy wins the token-budget band whose edges are predictable
  from geometry — measured at six budgets (two seeds everywhere except the upper-edge bracket),
  both heads (mechanism-agnostic), with the trustworthy-memory boundary from the second operating
  point (Ch. 4) and the negative control on the field's own benchmark (Ch. 5). [EXPECTED-B1,
  add on two-seed confirmation only: "…and the band's lower edge is family-parametric, moving
  from 896 to 512 tokens when the coarse tick halves, at the token-arithmetic prediction both
  times."]
- **C5.** The exactly invertible stochastic recurrence and the three-axis reversibility taxonomy
  with the measured inner-loop no-go (Ch. 6).

## 1.5 Reader's map

Chapter 2 defines the instrument; Chapter 3 selects the mechanism; Chapter 4 carries the
headline frontier and its boundary conditions; Chapter 5 tests the selected recipe on external
datasets (with the memory-horizon null as a *prediction*); Chapter 6 gives the reversibility
theory and its honest limits; Chapter 7 positions the work and lists explicit non-claims;
Chapter 8 concludes. Journal format note: Chapters 3 (Part B) and 6 compress to sections in a
~12-page version; the dissertation carries them in full.
