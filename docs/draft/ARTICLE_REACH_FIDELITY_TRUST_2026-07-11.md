# Reach and Fidelity in Budgeted Video Memory:
# A Controlled Context-Packaging Study

> **Canonical fourth-article direction, updated 2026-07-15.** This document changes neither the
> dissertation topic nor the subject matter of the three existing articles. It reorganizes the
> final article around the questions the experiments can actually answer. Current-pool analyses
> remain development evidence; the separately frozen E2 protocol has now been evaluated once and
> supplies the confirmation clauses explicitly labelled below.

## Abstract

A bounded-context video predictor must decide not only how much history to retain, but how to
represent it. More temporal reach can be purchased by reducing spatial fidelity or temporal rate.
We study this as a controlled **context-packaging** problem. The manipulated factors are temporal
sampling rate, spatial resolution and package depth; the main explanatory questions are whether a
package reaches the memory-relevant past and preserves sufficient fidelity. We separately test
whether an apparent self-fed reliability boundary is robust across training seeds.

We construct a procedural panning-world instrument in which foreground content exits and re-enters
the viewport and its geometric observability is known from recorded camera motion. A shared
backbone consumes flat or multistream context packages at tested token budgets. Development-set
experiments locate a reach–fidelity frontier rather than assuming that hierarchy should win. A
long-reach spatial package strongly outperforms shorter packages when the relevant evidence lies
inside its 48-frame reach but outside theirs: the post-hoc pair-specific effects are +0.0880 and
+0.0809 for the 14-to-48-frame band across two seeds. At larger tested budgets, full-resolution
context recovers and can become preferable. A stride-2 package wins at 512 tokens across diffusion
seeds, but the legacy implementation also changes its target interface, so this is reported as a
configuration effect pending a fixed-target control. A five-seed factorial finds a stable
temporal-package advantage under diffusion, but sign-changing package effects under regression
and no training-seed-robust directional head interaction. In a preregistered three-seed rollout
study, the early hierarchy advantage is positive at every seed, but a late inversion observed at
seed 0 reverses to a late hierarchy advantage at seeds 1 and 2. The proposed trust boundary is
therefore not replicated.

These results support a scoped account: context reach explains where memory can help, while
fidelity explains why compression can lose after reach is saturated. Addressability and self-fed
reliability remain distinct constructs, but the present data do not locate a stable trust
boundary. They do not establish universal laws or a unique continuous crossing.
The central reach–fidelity comparisons were confirmed once on a hash-pinned untouched holdout,
with false-positive-sensitive, component and coverage endpoints. All claims remain scoped to the
procedural static-world testbed, frozen representation and tested training regimes.

## 1. Introduction

### 1.1 The empirical problem

Hierarchical video models are often motivated by a plausible story: coarse state changes slowly,
fine state renders detail, and a hierarchy should therefore retain longer history efficiently.
That story is not yet an answer. A model can appear better because it sees more frames, receives a
different prediction target, uses more inference work, or is evaluated on a metric that ignores
its hallucinations. Conversely, a hierarchy can appear useless on a benchmark where every
relevant object remains visible to a short context.

This article begins with two primary explanatory questions and one robustness question rather
than a claim that hierarchy must win:

1. **Reach:** when does the supplied context contain the past that determines a re-entering event?
2. **Fidelity:** when reach is available, how much task-relevant information survives the package?
3. **Self-fed reliability:** does an apparent late failure of stored information replicate across
   training seeds?

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

**RQ4 — Self-fed reliability.** Does the seed-0 early-benefit/late-inversion pattern replicate
under the same long-loop protocol? The preregistered prediction is an early positive and late
negative hierarchy effect in at least one new seed; a failed replication demotes the boundary
rather than triggering post-hoc localization experiments.

### 1.4 Contributions

The fourth article contributes:

1. a low-cost procedural instrument with exact geometric observability and re-entry labels;
2. tested rate-versus-resolution packages at explicit context-token budgets;
3. pair-specific reach-differential analysis derived from comparator geometry;
4. lower- and upper-regime inversions reported as tested configurations, not interpolated laws;
5. a five-seed package factorial that separates a stable diffusion result from regression and
   cross-head heterogeneity;
6. an outcome-independent negative replication showing that the proposed late trust inversion is
   training-seed-specific under this protocol;
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
`14 < dt <= 48`; for stride-2 B1 versus `flat-short`, `8 < dt <= 48`. These definitions were
post-hoc on the development pool and were frozen as primary before the untouched E2 evaluation.

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

The five-seed factorial gives a clean result under diffusion and a boundary under regression:

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | −0.0089 [−0.0151, −0.0025] |
| 1 | +0.0046 [+0.0007, +0.0084] | −0.0104 [−0.0146, −0.0062] | −0.0150 [−0.0199, −0.0104] |
| 2 | +0.0111 [+0.0070, +0.0151] | −0.0208 [−0.0270, −0.0150] | −0.0319 [−0.0401, −0.0245] |
| 3 | +0.0075 [+0.0037, +0.0114] | +0.0248 [+0.0181, +0.0323] | +0.0173 [+0.0106, +0.0245] |
| 4 | +0.0060 [+0.0027, +0.0092] | −0.0166 [−0.0229, −0.0108] | −0.0226 [−0.0297, −0.0160] |

All five diffusion effects favor temporally sparse full-resolution context. Their across-training-
seed mean is +0.0098, SD 0.0060, with t95 [+0.0023, +0.0173]. Regression instead has two
temporal and three spatial material winners; its mean is −0.0024 with t95 [−0.0266, +0.0217].
The interaction also changes sign, with mean −0.0122 and t95 [−0.0353, +0.0109]. Consequently,
the preregistered robust negative head-shift rule fails. The defensible conclusion is that the
diffusion package result is stable across the five tested seeds, whereas package utilization
under regression and cross-head generalization are training-seed-sensitive. This is not
equivalence and not evidence for a directional head law. Clip-cluster intervals condition on
each trained model; the t intervals use training seeds as units. Unlike E2, this factorial uses
the existing adaptively used evaluation pool and is measured development evidence, not a new
untouched confirmation.

### 4.5 Addressability and self-fed reliability: a negative replication

At a longer-loop operating point, single-step teacher-forced retrieval succeeds when span covers
the loop. Seed 0 then shows an early hierarchy benefit followed by a late inversion during pure
self-fed rollout. E5a preregistered that horizon pattern and repeated the same pair at two new
training seeds. Positive effects below favor the 224-frame-reach hierarchy:

| Training seed | Early h1–24 effect [95% CI] | Late h61–256 effect [95% CI] |
|---:|---:|---:|
| 0 | +0.0204 [+0.0133, +0.0279] | −0.0064 [−0.0095, −0.0032] |
| 1 | +0.0082 [+0.0035, +0.0133] | +0.0074 [+0.0038, +0.0112] |
| 2 | +0.0230 [+0.0150, +0.0315] | +0.0087 [+0.0035, +0.0139] |

The early advantage is positive and CI-clean at all three seeds. Neither new seed reproduces the
required late-negative direction; both instead show a positive, CI-clean late hierarchy effect.
The registered verdict is **NOT_REPLICATED**, so the gated horizon/refresh/corruption programme is
not authorized. This result preserves the conceptual distinction between addressability and
self-fed reliability while rejecting a stable approximately-24-frame trust boundary in the
current evidence.

### 4.6 Untouched confirmation

On the frozen E2 holdout, `hier-min` versus `flat-min` yields +0.0980 [0.0788, 0.1179] and
+0.0914 [0.0736, 0.1098] in `14 < dt <= 48` across seeds 0/1. B1 versus `flat-short` yields
+0.0692 [0.0543, 0.0845] and +0.0743 [0.0588, 0.0899] in `8 < dt <= 48`. Whole-population,
union/full-frame, recall/F1, missed-component and coverage endpoints corroborate the signs.

The simultaneous frontier family confirms hierarchy at 896 and 1,152 tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. Thus the evidence
supports a tested reach–fidelity phase structure without claiming a smooth or unique crossing.

## 5. Boundaries and positioning

The study does not establish a universal hierarchy advantage, a unique token crossing, a pure B1
context-stride effect, regression equivalence, information-theoretic necessity, natural-video
generalization, a trust law, or valid legacy Phase C object conclusions. Moving-MNIST 10-to-10 and
TaxiBJ are descriptive boundary/competence anchors rather than equivalence or re-entry evidence.

Prior work already studies slow latent state, compressed trajectory memory, retrieval, memory
benchmarks and autoregressive mismatch [CW-VAE; FDM; TECO; recent spatial-memory world models;
WorldPack; MBench; CaR/SceneFly; MemLearner; GIM-World; Self Forcing]. The novelty is narrower: an
analytic observability instrument, matched-budget rate-versus-resolution tests, measured
inversions, pair-specific reach bands, and an outcome-independent replication test of self-fed
reliability.

A strong follow-up would use paired counterfactual worlds identical over the short visible context
but differing in an off-screen texture/barcode that determines re-entry. Crossing static/dynamic
hidden content with low/high spatial detail, plus classical pose-indexed and learned retrieval
baselines, would test whether the preferred package follows the information structure.

## 6. Conclusion

The project is a controlled search for boundaries, not proof that one architecture always wins.
Current evidence supports two central scoped propositions: effects are largest where only the
longer package reaches the relevant past, and full-resolution fidelity can recover after reach is
saturated. Context packaging is the intervention; reach and fidelity organize the confirmed
explanation. Self-fed reliability remains an important question, not a third empirical law.

The one-time untouched confirmation supports the central reach–fidelity account. The five-seed
factorial strengthens the diffusion temporal-package result while rejecting a training-seed-
robust directional head shift; regression remains a sign-changing scope boundary rather than an
equivalence result. Because the article retains configuration-level B1 wording, the fixed-target
control is not required for its present causal scope. E5a failed its preregistered late-inversion
replication rule, so trust is removed from the title and the gated E5b diagnostic programme is not
run. The three-seed early benefit and seed-0-only late failure are reported together. This
outcome-independent narrowing is part of the contribution.

## Required pre-submission inserts

- [x] integrate seed-2 trio and c2 with regime-correct wording;
- [x] insert untouched-confirmation and metric-decomposition results;
- [x] remove trust from the title after E5a returned `NOT_REPLICATED`;
- [x] complete the five-seed E4a/E4b factorial; retain stable diffusion packaging and demote the
      directional head-shift claim;
- [ ] add the fixed-target B1 control or retain configuration-only wording;
- [ ] replace reference keys with verified primary citations;
- [ ] generate configuration/cost, claim/evidence and prior-art tables from artifacts;
- [ ] move EBT, reversibility and B3 detail to supplement/dissertation;
- [ ] complete the evidence archive and reproducibility appendix.
