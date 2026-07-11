# Reach, Fidelity, and Trust in Budgeted Video Memory:
# A Controlled Context-Packaging Study

> **Canonical fourth-article direction, 2026-07-11.** This document changes neither the
> dissertation topic nor the subject matter of the three existing articles. It reorganizes the
> final article around the questions the experiments can actually answer. Current-pool analyses
> are development evidence. Confirmation clauses remain unavailable until the untouched protocol
> is frozen and evaluated once.

## Abstract

A bounded-context video predictor must decide not only how much history to retain, but how to
represent it. More temporal reach can be purchased by reducing spatial fidelity or temporal rate;
the stored evidence must then remain usable during self-fed generation. We study this as a
controlled **context-packaging** problem. The manipulated factors are temporal sampling rate,
spatial resolution and package depth; the explanatory questions are whether a package reaches the
memory-relevant past, preserves sufficient fidelity, and remains trustworthy during rollout.

We construct a procedural panning-world instrument in which foreground content exits and re-enters
the viewport and its geometric observability is known from recorded camera motion. A shared
backbone consumes flat or multistream context packages at tested token budgets. Development-set
experiments locate a reach–fidelity frontier rather than assuming that hierarchy should win. A
long-reach spatial package strongly outperforms shorter packages when the relevant evidence lies
inside its 48-frame reach but outside theirs: the post-hoc pair-specific effects are +0.0880 and
+0.0809 for the 14-to-48-frame band across two seeds. At larger tested budgets, full-resolution
context recovers and can become preferable. A stride-2 package wins at 512 tokens across diffusion
seeds, but the legacy implementation also changes its target interface, so this is reported as a
configuration effect pending a fixed-target control. Temporal subsampling versus spatial
coarsening is head- and seed-sensitive under regression, and self-fed rollout exposes a boundary
between addressable and reliable memory.

These results support a scoped account: context reach explains where memory can help; fidelity
explains why compression can lose after reach is saturated; trust limits what addressable memory
can deliver during rollout. They do not establish universal laws or a unique continuous crossing.
The final confirmatory analysis is preregistered for a new hash-pinned holdout and includes
false-positive-sensitive, component and coverage endpoints.

## 1. Introduction

### 1.1 The empirical problem

Hierarchical video models are often motivated by a plausible story: coarse state changes slowly,
fine state renders detail, and a hierarchy should therefore retain longer history efficiently.
That story is not yet an answer. A model can appear better because it sees more frames, receives a
different prediction target, uses more inference work, or is evaluated on a metric that ignores
its hallucinations. Conversely, a hierarchy can appear useless on a benchmark where every
relevant object remains visible to a short context.

This article begins with three questions rather than a claim that hierarchy must win:

1. **Reach:** when does the supplied context contain the past that determines a re-entering event?
2. **Fidelity:** when reach is available, how much task-relevant information survives the package?
3. **Trust:** when information is stored, does repeated self-fed generation preserve its utility?

The experimental object is context packaging. “Hierarchy” denotes one family of packages—multiple
rates, resolutions or depths—rather than the conclusion. This distinction lets a flat window, a
spatially coarse stream, and a temporally subsampled full-resolution stream be compared as design
choices under explicitly reported budgets.

### 1.2 Why aggregate evaluation is insufficient

Aggregate MSE on sparse synthetic video is dominated by background. MSE inside only the
ground-truth foreground mask has the opposite blind spot: content hallucinated elsewhere is not
penalized. Empty foreground frames also change its meaning if they silently fall back to
full-frame MSE. Identity metrics based on left-to-right component order fail when objects cross,
and analyses that retain only frames scorable by both arms select easy survivors.

We therefore separate populations and estimands. The core report includes full-frame,
non-empty-GT-foreground, background and union-mask errors; foreground precision and recall;
component misses, extras and overlap; and scoring coverage. Persistent-object studies use
generator-provided identities and one-to-one assignment. No single score is asked to represent
fidelity, hallucination, localization and coverage simultaneously.

### 1.3 Research questions and predictions

**RQ1 — Reach.** For a fixed context-token budget, is the package advantage concentrated on frames
whose relevant evidence lies inside the longer package's reach but outside the shorter package's
reach? The effect should be larger in a pair-derived reach-differential band than in generic
horizon bins. A result outside this band would weaken the mechanism even if an aggregate
comparison remained positive.

**RQ2 — Fidelity.** Once both packages reach the relevant past, does higher spatial detail recover
the advantage? We expect tested inversions as a flat full-resolution window approaches the same
temporal span, without assuming monotonicity or a unique crossing between sampled budgets.

**RQ3 — Package shape.** Are temporal subsampling and spatial coarsening interchangeable ways to
buy reach? Not necessarily: static off-screen content may favour sparse full-resolution storage;
dynamically evolving or lower-detail content may favour another package. The present testbed
locates one operating point rather than proving data-independent ordering.

**RQ4 — Trust.** Does information usable under teacher forcing remain usable during self-fed
rollout? Rollout may introduce a reliability boundary distinct from addressability. Its location
should vary with horizon and refresh/corruption interventions.

### 1.4 Contributions

The fourth article contributes:

1. a low-cost procedural instrument with exact geometric observability and re-entry labels;
2. tested rate-versus-resolution packages at explicit context-token budgets;
3. pair-specific reach-differential analysis derived from comparator geometry;
4. lower- and upper-regime inversions reported as tested configurations, not interpolated laws;
5. seed-resolved evidence that the generation head changes package contrasts;
6. a separation between addressable memory and self-fed reliability;
7. an untouched-confirmation protocol with false-positive-sensitive and coverage endpoints;
8. explicit reporting of negative, confounded and seed-sensitive outcomes.

Energy-based modelling and stochastic reversibility remain dissertation strands and supplementary
context; they are not required to make this article's central claim.

## 2. Controlled instrument

### 2.1 Panning-world generator

A 64-pixel viewport moves over a wider toroidal world containing background texture and digit
sprites at fixed world coordinates. Recorded actions and viewport positions determine when each
world location was last visible. Under the primary loopmom operating point, a directed traversal
returns relevant content after approximately 48 frames.

The generator provides exact **geometric observability**: whether the world region needed for the
current re-entry appeared within a candidate context span. This is not called
information-theoretic necessity because shared MNIST priors can make unseen glyph content
statistically guessable. A stronger necessity experiment would use paired counterfactual worlds
that are identical over the short context but differ at re-entry.

### 2.2 Context packages and the B1 confound

Every package is described by context temporal stride, context spatial resolution, slots and
reach, target stride and slots, auxiliary losses, context/target/total tokens, head and sampler
allocation. The legacy B1 stride-2 arm couples context stride to coarse target stride. It therefore
compares a complete stride-2 configuration to a short-flat configuration. The prepared causal
control keeps four coarse and four fine target slots, loss weights and target stride fixed while
changing only the 512-token context package.

### 2.3 Cost accounting

“Matched compute” is avoided unless total work is genuinely matched. Comparison tables list
context tokens, target tokens, total sequence length, parameters, examples, optimizer updates,
batch size, EMA, per-step and total NFE, runtime and peak memory. The B16x30k and B8x60k regimes
process approximately the same examples but differ in updates, gradient noise, grouping and
effective EMA horizon; c2 is an optimizer-regime robustness control.

### 2.4 Development and confirmation populations

The existing test pool informed qualitative monitoring, arm selection and theory development. It
provides development evidence, not an untouched final holdout. Confirmation uses a newly
generated, hash-pinned set rejected by training and monitoring loaders. The analysis, thresholds,
reach bands, table shells and interpretation branches are frozen before access.

## 3. Endpoints and inference

### 3.1 Complete frame metrics

For every frame record we save full-frame MSE; GT-foreground MSE only on non-empty frames;
GT-background and union-mask MSE; foreground precision, recall and F1; component counts, misses,
extras, IoU and centroid error. Empty-GT frames are a named stratum and never inherit a differently
defined full-frame value under a foreground label.

### 3.2 Persistent-object metrics

Object claims require stable generator IDs, glyph classes, trajectories, boxes and overlaps.
Predicted components are assigned jointly with a frozen Hungarian cost. Misses receive a
trajectory penalty, extras remain visible, and identity accuracy is intention-to-score over all
labelled GT objects. Matched-only performance is diagnostic; coverage is co-primary.

### 3.3 Pair-specific reach analysis

For reaches $r_s < r_l$, the mechanistic band is $r_s < \Delta t \le r_l$, restricted to
non-empty, geometrically answerable records. For `hier-min` versus `flat-min`, it is
`14 < dt <= 48`; for stride-2 B1 versus `flat-short`, `8 < dt <= 48`. Current results in these
bands are post-hoc. The same geometry-derived definitions are primary on untouched confirmation.

### 3.4 Statistical protocol

Effects are paired by clip and horizon; confidence intervals resample clips rather than
autocorrelated frames. Reports show each seed, record/clip counts and coverage. Opposite
seed-specific effects are heterogeneity even when their mean is near zero. Equivalence requires a
justified smallest effect of interest and reliable measurement; otherwise a null interval is
inconclusive.

## 4. Development evidence

### 4.1 Reach-differential effects

On non-empty answerable records, `hier-min` versus `flat-min` yields +0.0880 [0.0668, 0.1083] at
seed 0 and +0.0809 [0.0621, 0.0998] at seed 1 in `14 < dt <= 48`. Stride-2 B1 versus
`flat-short` yields +0.0675 [0.0526, 0.0828] and +0.0709 [0.0544, 0.0871] in
`8 < dt <= 48`. These are the strongest current mechanistic observations, while their post-hoc
origin prevents a confirmatory label on the development pool.

### 4.2 Tested reach–fidelity frontier

At 896 context tokens the dense coarse family has 48-frame reach and beats the shorter flat
comparator under diffusion by +0.0399 and +0.0358 across seeds 0/1. At 3,072 tokens the flat
48-frame window wins across both seeds (−0.0131 and −0.0107). At B16x30k, 1,792 is the largest
tested budget favouring the spatial package and 2,432 the smallest tested budget favouring flat
full-resolution context. This bracket does not prove monotonicity, continuity or one crossing.

### 4.3 B1 configuration result

The stride-2 B1 configuration wins under diffusion by +0.0330 and +0.0338 across the completed
seeds. Under regression, seed 0 yields +0.0639 [0.0516, 0.0763]. This is strong evidence for the
whole configuration. It does not isolate context stride because target slots, target tokens,
auxiliary loss, sequence length and optimization difficulty also change. The article retains
configuration-level wording until the fixed-target control lands.

### 4.4 Temporal versus spatial packages and head interaction

Under diffusion, temporally subsampled full-resolution history beats the spatially coarse package
by +0.0197 and +0.0046 across two seeds. Under regression the package contrast is +0.0107 at seed 0
and −0.0104 at seed 1, both intervals excluding zero. Averaging them would hide a sign reversal
and is not equivalence. The head-induced shift points toward spatial packaging on both seeds,
which motivates—but does not yet establish—a formal head × package interaction.

### 4.5 Addressability and trust

At a longer-loop operating point, single-step teacher-forced retrieval succeeds when span covers
the loop, while pure self-fed rollout loses the advantage after roughly 24 frames in the decisive
programme. This shows addressability and rollout reliability can diverge. Because the decisive
rollout result is seed-scoped and protocol allocation changes, it is a measured boundary. A
title-level trust claim requires another seed or horizon/task plus refresh/corruption diagnostics.

## 5. Boundaries and positioning

The study does not establish a universal hierarchy advantage, a unique token crossing, a pure B1
context-stride effect, regression equivalence, information-theoretic necessity, natural-video
generalization, a trust law, or valid legacy Phase C object conclusions. Moving-MNIST 10-to-10 and
TaxiBJ are descriptive boundary/competence anchors rather than equivalence or re-entry evidence.

Prior work already studies slow latent state, compressed trajectory memory, retrieval, memory
benchmarks and autoregressive mismatch [CW-VAE; FDM; TECO; recent spatial-memory world models;
WorldPack; MBench; CaR/SceneFly; MemLearner; GIM-World; Self Forcing]. The novelty is narrower: an
analytic observability instrument, matched-budget rate-versus-resolution tests, measured
inversions, pair-specific reach bands, and separation of addressable from self-fed memory.

A strong follow-up would use paired counterfactual worlds identical over the short visible context
but differing in an off-screen texture/barcode that determines re-entry. Crossing static/dynamic
hidden content with low/high spatial detail, plus classical pose-indexed and learned retrieval
baselines, would test whether the preferred package follows the information structure.

## 6. Conclusion

The project is a controlled search for boundaries, not proof that one architecture always wins.
Current evidence supports three scoped propositions: effects are largest where only the longer
package reaches the relevant past; full-resolution fidelity can recover after reach is saturated;
and self-fed generation can make addressable memory unreliable. Context packaging is the
intervention; reach, fidelity and trust organize the explanation.

The highest-value remaining test is a one-time untouched confirmation using endpoints that expose
hallucinations, misses, extras and coverage. The next causal control holds B1's target interface
fixed. The next trust study varies horizon and refresh/corruption directly. Results that fail these
tests will narrow the account rather than being explained away; that outcome-independent
structure is part of the contribution.

## Required pre-submission inserts

- [ ] integrate seed-2 trio and c2 with regime-correct wording;
- [ ] insert untouched-confirmation and metric-decomposition results;
- [ ] decide whether trust has sufficient replication for the title;
- [ ] decide whether the regression interaction merits seed 2+;
- [ ] add the fixed-target B1 control or retain configuration-only wording;
- [ ] replace reference keys with verified primary citations;
- [ ] generate configuration/cost, claim/evidence and prior-art tables from artifacts;
- [ ] move EBT, reversibility and B3 detail to supplement/dissertation;
- [ ] complete the evidence archive and reproducibility appendix.
