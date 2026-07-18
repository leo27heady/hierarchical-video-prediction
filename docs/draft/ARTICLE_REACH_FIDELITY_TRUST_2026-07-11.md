# When Does Hierarchical Video Memory Pay?
## Reach–Fidelity Trade-offs under Fixed Context Budgets

<!-- canonical-id: binding fourth-article manuscript -->

**Keywords:** long-horizon video prediction; context compression; hierarchical latent models;
memory; observability; re-entry; diffusion transformer; reproducible evaluation.

## Abstract

A bounded-context video predictor must choose not only how much history to retain, but how to
represent it. We frame temporal sampling rate, spatial resolution and package depth as a controlled
**context-packaging** design space. A procedural panning-world instrument provides exact geometric
observability when content leaves and re-enters the viewport, while a shared diffusion-transformer
backbone consumes flat or multistream packages at explicit token budgets.

On a hash-pinned untouched holdout, the hierarchy wins when it alone reaches the relevant past:
the registered 14-to-48-frame reach-band effects are +0.0980 and +0.0914 across two training
seeds. A simultaneous frontier confirms hierarchy at 896 and 1,152 context tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. In the frozen latent
interface, 48 fine frames cost 3,072 tokens, whereas 48 coarse frames plus a two- or six-frame fine
tail cost 896 or 1,152 tokens. Hierarchical packaging therefore provides the same nominal
48-frame reach with 2.7–3.4 times fewer context tokens, while the upper-budget reversal shows the
price of discarded detail.

The main directions persist across a fourfold sampling-work range, three sampler seeds and an
independently trained codec, although the 2,432-token inversion is less precise under alternate
sampler or codec conditions. A five-seed factorial finds a stable temporal-package advantage under
diffusion but sign-changing regression and cross-head effects. A preregistered long-rollout study
replicates the early hierarchy benefit but not a seed-0 late inversion, rejecting a stable trust
boundary. Finally, a protected persistent-object Moving-MNIST study yields a mixed external
boundary: hierarchy better preserves support and trajectory continuity, whereas flat context
better preserves digit identity; foreground-only error is unresolved.

The resulting prescription is conditional rather than architectural: buy enough reach to include
the memory-relevant past, then preserve the fidelity required by the task. Hierarchy is most useful
when a flat full-resolution window cannot afford that reach; flat context becomes preferable when
fine detail, rather than temporal access, is limiting.

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

1. a low-cost re-entry instrument with exact geometric observability, complete error metrics and
   pair-derived reach bands;
2. an untouched, simultaneous fixed-budget frontier showing where hierarchical reach pays and
   where flat full-resolution fidelity recovers;
3. package-shape, prediction-head, sampler-allocation and codec interventions that separate stable
   diffusion directions from training-seed and upper-frontier precision boundaries;
4. explicit-access and persistent-object boundary studies that distinguish geometric reach,
   support/continuity, identity fidelity and reliable reuse;
5. outcome-independent reporting of a failed trust-boundary replication, a confounded B1
   configuration result and all protected negative or mixed outcomes.

Energy-based modelling and stochastic reversibility remain dissertation strands and supplementary
context; they are not required to make this article's central claim.

### 1.5 Relation to prior work

Clockwork VAEs introduced hierarchies whose upper latent sequences update at slower rates and
demonstrated long video prediction over sequences up to 1,000 frames [Saxena et al., 2021].
Dynamic latent hierarchies, TECO, MilliVid and flexible diffusion conditioning likewise pursue
long-horizon generation through hierarchical state, latent temporal consistency, coarse-to-fine
rollout or variable history [Zakharov et al., 2023; Yan et al., 2023; Chandratreya et al., 2026;
Harvey et al., 2022]. These works motivate temporal abstraction, but they do not provide the
matched package-by-budget intervention or exact pair-derived observability bands used here.

Recent work makes it especially important not to claim that memory-focused evaluation or
retrieval is absent. WorldPack combines compressed trajectory packing with retrieval and evaluates
spatial consistency on LoopNav [Oshima et al., 2025]. CaR/SceneFly, MemLearner and GIM-World
study compression or learned retrieval under annotated camera motion and occlusion
[Peng et al., 2026; Yu et al., 2026; Wei et al., 2026]. MBench and MemoBench explicitly benchmark
long-term entity/environment/causal memory or disappearance and reappearance [Zhang et al., 2026;
Chen et al., 2026]. Self Forcing directly targets the mismatch between ground-truth-conditioned
training and self-fed autoregressive inference [Huang et al., 2025].

The present contribution is narrower and complementary. It treats temporal rate, spatial
resolution and depth as manipulable ways to package a bounded context; computes exact geometric
observability from a controlled generator; and tests where longer reach helps and where lost
fidelity reverses that benefit. The pose-indexed mosaic reference receives raw pixels and exact
pose and is therefore reported separately from token-budget comparisons. Cross-paper numerical
ranking is not attempted because datasets, context, decoders, horizons, samplers and endpoints are
not protocol-equivalent.

**Table 1 — Protocol positioning relative to representative long-memory video work.**

| Work | Memory mechanism | Matched package budget | Exact observability | Explicitly self-fed evaluation | Role here |
|---|---|---|---|---|---|
| CW-VAE | clocked hierarchical latent state | no | no | yes | temporal-hierarchy lineage |
| FDM | flexible sparse/long-range diffusion conditioning | no | no | yes | conditioning lineage |
| TECO | compressed latent sequence + temporal transformer | no | no | yes | long-consistency lineage |
| MilliVid | coarse-to-fine hierarchical latent rollout | no | no | yes | modern hierarchy comparator |
| WorldPack | trajectory packing + retrieval | no | no | yes | compressed-memory comparator |
| CaR / MemLearner / GIM-World | learned or geometry-conditioned retrieval | no | annotated or learned geometry | yes | modern retrieval comparators |
| MBench / MemoBench | benchmark instruments | n/a | task-dependent | model-dependent | memory-evaluation context |
| Self Forcing | rollout-distribution training | no | no | explicitly targeted | self-fed mismatch intervention |
| This work | flat, spatial and temporal context packages | yes where stated | analytic LoopMom labels | yes | controlled reach–fidelity study |

This table compares experimental roles, not scores. In particular, the later pose-indexed mosaic
receives raw pixels and exact pose, so it is visually separated from the matched-budget neural
comparisons.

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

![LoopMom geometry and registered reach band](../figures/fig15_loopmom_observability_schematic.png)

**Figure 1.** The 64-pixel viewport moves over a 192-pixel toroidal static world. Saved positions
make geometric observability exact. The primary reach analysis evaluates records whose relevant
content was observed 15–48 frames earlier: available to `hier-min`, unavailable to `flat-min`.
The pose mosaic is visually separated because it also receives exact future pose and raw stored
pixels.

### 2.2 Context packages and the B1 confound

Every package is described by context temporal stride, context spatial resolution, slots and
reach, target stride and slots, auxiliary losses, context/target/total tokens, head and sampler
allocation. The legacy B1 stride-2 arm couples context stride to coarse target stride. It therefore
compares a complete stride-2 configuration to a short-flat configuration. We permanently retain
that configuration-level interpretation in this article; a fixed-target control would be required
only to revive a pure causal context-stride claim.

![Frozen codec, shared predictor and context packages](../figures/fig16_context_packaging_architecture.png)

**Figure 2.** The predictor backbone and frozen two-level codec are shared while the context is
packaged as contiguous fine frames, dense coarse history plus a recent fine tail, or sparse
full-resolution anchors plus a recent tail. Reach and fidelity describe consequences of the
package; they are not separately trained modules.

**Table 2 — Exact interface accounting for the central configurations.**

| Arm | Package | Reach | Context tokens | Target tokens | Total tokens |
|---|---|---:|---:|---:|---:|
| `flat-min` | 14 contiguous fine frames | 14 | 896 | 320 | 1,216 |
| `hier-min` | 48 dense coarse + 2 fine | 48 | 896 | 320 | 1,216 |
| `flat-short` | 8 contiguous fine frames | 8 | 512 | 320 | 832 |
| `hier-s2-min` | 24 stride-2 coarse + 2 fine | 48 | 512 | 288 | 800 |
| `flat-budget` | 18 contiguous fine frames | 18 | 1,152 | 320 | 1,472 |
| `hier-long` | 48 dense coarse + 6 fine | 48 | 1,152 | 320 | 1,472 |
| `strided-flat` | 11 stride-4 fine anchors + 6 fine | 48 | 1,088 | 320 | 1,408 |
| `flat-c2` / `hier-c2` | contiguous fine / dense coarse + fine | 38 / 48 | 2,432 | 320 | 2,752 |

The `strided-flat` package is near-matched, not exactly matched, to `hier-long`: it uses 64 fewer
context tokens, a deficit that favors the temporal-package control. B1 changes 32 target tokens
as well as the context, so its outcome remains a configuration effect.

### 2.3 Package, predictor and estimand

Let a context package be

$$
\mathcal P_t=\{(K_\ell,q_\ell)\}_{\ell=1}^{L},
$$

where $K_\ell$ is the set of historical lags retained at representation level $\ell$ and
$q_\ell$ is the number of tokens used by one slot at that level. Its context cost and nominal
reach are

$$
B_{\mathrm{ctx}}(\mathcal P)=\sum_{\ell=1}^{L}|K_\ell|q_\ell,\qquad
R(\mathcal P)=\max_{\ell,k\in K_\ell} k .
$$

This accounting makes reach exact but does not pretend that fidelity is one scalar: spatial
resolution, codec distortion and temporal sparsity preserve different information. In the frozen
interface $q_{\mathrm{fine}}=64$ and $q_{\mathrm{coarse}}=16$. Thus 48 fine slots cost 3,072
tokens, whereas 48 coarse slots plus two or six recent fine slots cost 896 or 1,152.

The common predictor models a target latent block $z_{t+1:t+m}$ conditional on the packaged
history and camera actions,

$$
p_\theta\!\left(z_{t+1:t+m}\mid \mathcal P_t,u_{t+1:t+m}\right).
$$

For the diffusion head, training minimizes

$$
\mathcal L_{\mathrm{diff}}=
\mathbb E_{z,\epsilon,\tau}
\left\|\epsilon-\epsilon_\theta(z_\tau,\tau,\mathcal P_t,u)\right\|_2^2 ,
$$

while the regression control predicts the clean target latents directly. The principal paired
effect is always reported as comparator loss minus candidate loss, so positive values favor the
registered candidate. For a long package $\mathcal P_l$ and shorter comparator $\mathcal P_s$,
the reach-specific population is

$$
\mathcal D_{\mathrm{reach}}=
\{i:R(\mathcal P_s)<\Delta t_i\le R(\mathcal P_l),\
\text{answerable}_i,\ \text{foreground}_i\ne\varnothing\}.
$$

The diffusion and deterministic DDIM sampling conventions follow Ho et al. (2020) and Song et al.
(2021); the latent-token transformer follows the DiT design family (Peebles and Xie, 2023).

### 2.4 Cost accounting

“Matched compute” is avoided unless total work is genuinely matched. Comparison tables list
context tokens, target tokens, total sequence length, parameters, examples, optimizer updates,
batch size, EMA, per-step and total NFE, runtime and peak memory. The B16x30k and B8x60k regimes
process approximately the same examples but differ in updates, gradient noise, grouping and
effective EMA horizon; c2 is an optimizer-regime robustness control.

### 2.5 Development and confirmation populations

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

The need for this suite is empirical rather than stylistic. On the exact archived 100-clip,
964-target CW-VAE Moving-MNIST population, a deterministic all-black predictor scores 0.7681
SSIM / 13.941 dB, compared with 0.5708 / 12.918 for the trained best-of-5 CW-VAE curve. Because
only the model's mean curves survived, this is a hash-pinned same-population metric diagnostic
without paired inference—not a budget-matched model ranking. It shows why sparse-background
full-frame metrics cannot carry the memory claim.

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

Sampling robustness is evaluated separately from training-seed robustness. E7c freezes four
already-supported seed-0 diffusion contrasts and five variants before evaluation: NFE 8/16/32 at
sampler seed 0 and NFE 16 at sampler seeds 1/2. A 10,000-replicate studentized max-statistic
clip-cluster bootstrap covers the resulting 20-effect family. This checkpoint-only design can
test sensitivity to inference allocation and randomness; it cannot establish robustness to
training seed, representation seed or another dataset family.

## 4. Results

### 4.1 Untouched confirmation of reach and the fixed-budget frontier

On the frozen E2 holdout, `hier-min` versus `flat-min` yields +0.0980 [0.0788, 0.1179] and
+0.0914 [0.0736, 0.1098] in the registered `14 < dt <= 48` band across seeds 0/1. The B1
configuration versus `flat-short` yields +0.0692 [0.0543, 0.0845] and +0.0743
[0.0588, 0.0899] in `8 < dt <= 48`. Whole-population, union/full-frame, recall/F1,
missed-component and coverage endpoints corroborate the signs. The concentration of the benefit
inside the pair-derived reach band supports reach as the mechanism rather than horizon alone.

The single simultaneous frontier family confirms hierarchy at 896 and 1,152 tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. The largest tested
family-wise-confirmed hierarchy point is therefore 1,152 tokens and the smallest tested confirmed
flat point is 2,432. No value between them is inferred. The memory arithmetic is substantial:
the confirmed hierarchical packages provide nominal 48-frame reach with 896 or 1,152 tokens,
compared with 3,072 tokens for a 48-frame fine-only window. This is a 2.7–3.4-fold context-token
reduction for reach, not a claim of equal information or equal error at every operating point.

![Untouched frontier and context-token arithmetic](../figures/article_confirmed_frontier.png)

**Figure 3.** Left: untouched E2 effects under one studentized max-statistic clip-cluster family;
positive favors hierarchy. Right: exact cost of 48 fine slots versus coarse history with a recent
fine tail. The figure marks tested outcomes only and draws no interpolated crossing.

### 4.2 Development evidence that motivated the confirmation

On non-empty answerable development records, `hier-min` versus `flat-min` yields +0.0880
[0.0668, 0.1083] at seed 0 and +0.0809 [0.0621, 0.0998] at seed 1 in `14 < dt <= 48`.
Stride-2 B1 versus `flat-short` yields +0.0675 [0.0526, 0.0828] and +0.0709
[0.0544, 0.0871] in `8 < dt <= 48`. These post-hoc observations motivated the frozen E2
analysis; they are not relabelled as confirmatory. Development comparisons also suggested an
intermediate hierarchy region and an upper flat recovery. E2 preserved the lower and upper
directions but correctly demoted the 1,792-token point to unresolved.

### 4.3 B1 configuration result

The stride-2 B1 configuration wins under diffusion by +0.0330 and +0.0338 across the completed
seeds. Under regression, seed 0 yields +0.0639 [0.0516, 0.0763]. This is strong evidence for the
whole configuration. It does not isolate context stride because target slots, target tokens,
auxiliary loss, sequence length and optimization difficulty also change. The article retains
configuration-level wording permanently within its present scope.

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

![Five-seed head and package effects](../figures/fig17_head_package_seed_heterogeneity.png)

**Figure 4.** Clip-bootstrap intervals are conditional on a trained model and are shown per seed.
The shaded interval and dashed mean instead summarize five training seeds. Diffusion consistently
favors temporal anchors; regression and the interaction change sign.

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

![Registered long-loop replication](../figures/fig18_trust_negative_replication.png)

**Figure 5.** The early hierarchy benefit is positive at all three seeds. The registered late
negative direction occurs only at seed 0 and reverses at both new seeds, producing the
`NOT_REPLICATED` verdict.

### 4.6 Sampler and inference-allocation robustness

E7c evaluates seven frozen seed-0 diffusion checkpoints on a separate purpose-locked LoopMom
robustness set. Four contrasts—896-token reach, 1,152-token interior reach–fidelity, the
near-matched 1,088-token temporal-versus-1,152-token spatial package shape, and the 2,432-token
flat-fidelity inversion—are crossed with five frozen NFE/sampler variants. Positive effects favor
the registered candidate. The 64-token deficit favors the temporal control; this package-shape
comparison is not described as exactly token matched.

All 20 point estimates retain their expected direction. The 896 reach effects range from +0.0465
to +0.0482; the 1,152 interior effects from +0.0140 to +0.0201; the package-shape effects from
+0.0221 to +0.0292; and the 2,432 inversion effects from +0.0077 to +0.0122. The simultaneous
family-wise lower bound is positive for every variant of the first three contrasts and for the
inversion at NFE 8/16/32 with sampler seed 0. At NFE 16, the inversion remains positive but is
family-wise unresolved under sampler seed 1 (+0.0098 [−0.0001,+0.0196]) and sampler seed 2
(+0.0077 [−0.0013,+0.0167]). The registered verdict is therefore
**DIRECTIONALLY_ROBUST**, not `ROBUST_CI_CLEAN`.

Non-empty foreground, full-frame, background and union-mask errors favor the registered candidate
in all 20 variants, as do recall and F1. Component-count diagnostics are not uniformly favorable,
so the result is not described as improvement on every endpoint. This study rules out a simple
explanation based on the default NFE-16 allocation or one sampling-noise realization for the
observed directions. It does not establish universal sampler invariance: it is checkpoint-only,
uses seed-0 predictors, one frozen codec and one procedural dataset family.

![Sampler and NFE robustness](../figures/fig19_sampler_nfe_robustness.png)

**Figure 6.** All 20 point estimates preserve their registered direction. Open circles mark the
two 2,432-token upper-inversion effects whose simultaneous family intervals include zero under
alternate sampler seeds.

### 4.7 Robustness to an independently trained latent codec

E7a changes the entire latent representation by training a second codec seed, recomputing
codec-specific normalization and retraining the same seven seed-0 diffusion predictors. It is
therefore a whole-codec intervention—not an isolated test of reconstruction error, latent scale
or one normalization component. The new 500×200 LoopMom set is purpose-locked and shared by all
four original-codec controls and four alternate-codec contrasts. The eight effects form one
studentized max-statistic clip-cluster family.

| Codec | 896 reach | 1,152 interior | 1,088 vs 1,152 shape | 2,432 inversion |
|---|---:|---:|---:|---:|
| Original | +0.0494 [0.0339, 0.0649] | +0.0283 [0.0169, 0.0397] | +0.0253 [0.0163, 0.0343] | +0.0150 [0.0065, 0.0235] |
| Alternate | +0.0521 [0.0374, 0.0669] | +0.0439 [0.0310, 0.0569] | +0.0112 [0.0039, 0.0184] | +0.0079 [−0.0006, 0.0165] |

All eight point estimates retain the registered direction. Seven simultaneous lower bounds are
positive; only the alternate-codec 2,432-token inversion includes zero. The registered verdict is
therefore **CODEC_DIRECTIONALLY_ROBUST**. The alternate codec strengthens the lower/interior
reach effects but descriptively shrinks the package-shape and upper-inversion effects. Because
codec seed is not an inferential replicate, those changes locate sensitivity without identifying
its latent-space cause.

The upper comparison is metric-sensitive in precision, not reversed: non-empty foreground MSE
still favors flat by +0.0097 [0.0019, 0.0177], while full-frame, background, union,
precision/F1 and component-count intervals are mixed. No third codec, predictor seed, arm or
endpoint was selected after observation.

![Alternate-codec robustness](../figures/fig20_alternate_codec_robustness.png)

**Figure 7.** Original- and alternate-codec effects under one simultaneous eight-effect family.
All point estimates preserve direction; the alternate-codec upper inversion is the sole interval
crossing zero.

### 4.8 Privileged pose-indexed raw-memory reference

E8a asks how much error remains when access is made explicit. It builds a frozen world-coordinate
mosaic from raw warm-up pixels and queries it using exact saved viewport positions. It receives
information unavailable to the neural predictors: raw pixel storage and all future query poses.
It is therefore an access reference, not a token-matched architecture comparison, learned
retrieval baseline or natural-video result.

| Registered effect | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| Raw K14 minus raw K48 | +0.0723 | [+0.0548, +0.0898] |
| Raw K38 minus raw K48 | +0.0210 | [+0.0096, +0.0325] |
| `hier-min` minus raw K48 | +0.1219 | [+0.1031, +0.1407] |
| `hier-long` minus raw K48 | +0.1368 | [+0.1190, +0.1545] |
| `strided-flat` minus raw K48 | +0.1134 | [+0.0944, +0.1324] |
| `flat-c2` minus raw K48 | +0.1310 | [+0.1122, +0.1498] |

The K48 mosaic attains foreground MSE 0.0494, full-frame MSE 0.0027 and foreground recall
0.8438, versus foreground MSE 0.1628–0.1862 for the four registered neural references. Every
neural-to-K48 point estimate and simultaneous lower bound is positive, giving the preregistered
verdict **RAW_ACCESS_CEILING_CI_CLEAN**. The raw-span diagnostics are also positive: K48 improves
over both K14 and K38. The K38 saturation condition is not met. Median stored world coverage
increases from 80 columns at K6 to 154 at K48; mean future-crop observed coverage increases from
0.454 to 0.809. Observed pixels are reproduced exactly, while unobserved columns use the frozen
training-only row prior.

This closes a useful explanatory gap. Large error remains in the learned self-fed systems even
when their context can geometrically reach the relevant past; exact addressing and lossless
storage can remove much of it on this static world. The result does not show that hierarchy is
unnecessary or that a raw mosaic is deployable: the mosaic has privileged poses, no token budget,
no learned dynamics, no decoder loss and no self-fed state. It instead separates the potential
value of explicit access from the measured neural package frontier.

![Privileged pose-indexed raw memory](../figures/fig21_pose_indexed_raw_memory.png)

**Figure 8.** Left: stored-world and future-crop coverage as the raw warm-up span increases.
Right: the registered simultaneous family. All six lower bounds are positive; the four circular
neural-to-K48 effects determine the clean ceiling verdict.

### 4.9 External boundary: persistent-object long Moving-MNIST

E6 tests whether the reach–fidelity account survives the recognizable Moving-MNIST setting
[Srivastava et al., 2015], with
persistent digit identities, collisions and 300-frame self-fed prediction. The comparison uses
three training seeds, a 48-frame warm-up, matched 1,152-token flat and hierarchical packages, an
official train/test glyph split, generator-saved object identity and trajectory, one-to-one
Hungarian assignment, explicit missed/extra-object accounting, and a protected 100-clip holdout.
Positive effects below favor hierarchy. All 15 intervals belong to one preregistered
studentized max-statistic clip-cluster family.

| Seed | Foreground MSE | Union MSE | Trajectory error | Identity accuracy | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.0036 [−0.0074,+0.0146] | +0.0102 [+0.0019,+0.0185] | +2.4786 [+0.2130,+4.7443] | −0.0762 [−0.1106,−0.0418] | +0.0304 [+0.0016,+0.0591] |
| 1 | −0.0031 [−0.0127,+0.0065] | +0.0122 [+0.0052,+0.0192] | +1.4223 [−1.0646,+3.9092] | −0.0561 [−0.0934,−0.0188] | +0.0159 [−0.0163,+0.0482] |
| 2 | −0.0024 [−0.0131,+0.0083] | +0.0169 [+0.0093,+0.0245] | +4.1325 [+1.5271,+6.7380] | −0.0647 [−0.0955,−0.0338] | +0.0498 [+0.0161,+0.0834] |

The absolute values are important because both packages remain far from solved:

| Seed | Arm | Union MSE ↓ | Trajectory error ↓ | Identity accuracy ↑ | Object coverage ↑ |
|---:|---|---:|---:|---:|---:|
| 0 | flat / hierarchy | 0.6861 / 0.6759 | 42.19 / 39.71 | 0.214 / 0.138 | 0.619 / 0.650 |
| 1 | flat / hierarchy | 0.6901 / 0.6779 | 40.20 / 38.77 | 0.244 / 0.188 | 0.646 / 0.662 |
| 2 | flat / hierarchy | 0.6931 / 0.6763 | 43.79 / 39.65 | 0.222 / 0.158 | 0.599 / 0.649 |

The registered verdict is **MIXED_BOUNDARY**. Union-mask error favors hierarchy on every seed.
Trajectory and coverage favor hierarchy cleanly on seeds 0 and 2 and remain directionally
positive but unresolved on seed 1. Missed objects are lower and full-frame/background error is
lower for hierarchy at every seed as secondary descriptive evidence. Conversely, identity
accuracy favors flat context with simultaneous intervals wholly below zero at every seed.
Foreground-only error is unresolved and changes sign.

This is a scientifically useful division of labor. The coarse stream better preserves spatial
support—where an object is, whether it remains represented, and how its trajectory continues—but
loses fine glyph information needed to preserve what the object is. Because identity uses a fixed
ground-truth object denominator, the result is not explained by selecting only detected survivors.
It is consistent with the article's reach–fidelity account, but it is not a replication of
LoopMom's exact observability mechanism or evidence of natural-video transfer.

![Persistent-object long Moving-MNIST boundary](../figures/fig22_phasec_v2_boundary.png)

**Figure 9.** Registered E6 simultaneous effects. Positive favors hierarchy. The support and
continuity endpoints separate from identity fidelity, producing the preregistered mixed verdict.

## 5. Reproducibility and evidence availability

Every confirmatory or robustness dataset has a purpose string, generation seed, split indices and
SHA-256 hashes in a committed manifest. Opening a protected dataset requires the matching
purpose-specific access log. Each model or evaluation cell has a content-addressed `RunSpec`,
immutable metric artifact, checkpoint provenance, source commit, declared GPU-hour ceiling and
cloud run. The evidence index `artifacts/rft_claim_artifact_index.json` binds the manuscript's
central claims to frozen result JSON files; `docs/draft/RFT_GENERATED_TABLES.md` is regenerated
from those artifacts rather than transcribed by hand.

Training checkpoints store model, optimizer and EMA state. New-regime runs additionally record
Python, NumPy and Torch RNG state; older resumed runs are described as distributionally
reproducible rather than trajectory-identical where those states were absent. CPU gates check
dataset hashes, arm interfaces, batch-index and fixed-seed numerical parity where legacy and
refactored paths must be combined. W&B is a cloud mirror and health record, not the source of
truth: conclusions are recomputed from immutable local NPZ/JSON artifacts. GPU-hour accounting is
wall-clock ownership time, including evaluation, pilots, failures and retries.

The repository archive contains the generator, training and evaluation code, frozen protocols,
authorization gates, exact command wrappers, access logs, generated tables and figure scripts.
Large datasets and checkpoints are identified by cryptographic hashes even when distributed
separately. Smoke tests cannot overwrite canonical result names, and a completed study is not
accepted unless its independent completion validator passes.

## 6. Boundaries and positioning

The study does not establish a universal hierarchy advantage, a unique token crossing, a pure B1
context-stride effect, regression equivalence, information-theoretic necessity, natural-video
generalization, a trust law, or valid legacy Phase C object conclusions. Moving-MNIST 10-to-10 and
TaxiBJ are descriptive boundary/competence anchors rather than equivalence or re-entry evidence.
The corrected persistent-object long Moving-MNIST study is valid but remains an external
synthetic boundary: it supports neither a universal hierarchy ranking nor LoopMom-mechanism or
natural-video generalization.
E7a supports whole-codec directional robustness conditional on predictor seed 0; it does not
establish component-level codec causality or universal representation invariance.
E8a is a privileged static-world access reference. Because it receives exact future poses and
raw warm-up pixels without a matched token or inference budget, it cannot support a model-ranking,
efficiency or natural-generalization claim.

Prior work already studies slow latent state, compressed trajectory memory, retrieval, memory
benchmarks and autoregressive mismatch [CW-VAE; FDM; TECO; recent spatial-memory world models;
WorldPack; MBench; CaR/SceneFly; MemLearner; GIM-World; Self Forcing]. The novelty is narrower: an
analytic observability instrument, matched-budget rate-versus-resolution tests, measured
inversions, pair-specific reach bands, an outcome-independent replication test of self-fed
reliability, and a frozen simultaneous inference-robustness family.

A strong follow-up would use paired counterfactual worlds identical over the short visible context
but differing in an off-screen texture/barcode that determines re-entry. Crossing static/dynamic
hidden content with low/high spatial detail, plus classical pose-indexed and learned retrieval
baselines, would test whether the preferred package follows the information structure.

## 7. Conclusion

The project is a controlled search for boundaries, not proof that one architecture always wins.
Current evidence supports two central scoped propositions: effects are largest where only the
longer package reaches the relevant past, and full-resolution fidelity can recover after reach is
saturated. Context packaging is the intervention; reach and fidelity organize the confirmed
explanation. Self-fed reliability remains an important question, not a third empirical law.

The one-time untouched confirmation supports the central reach–fidelity account. The five-seed
factorial strengthens the diffusion temporal-package result while rejecting a training-seed-
robust directional head shift; regression remains a sign-changing scope boundary rather than an
equivalence result. E7c further shows that all four central diffusion directions survive the
registered fourfold NFE range and sampler-seed variations as point estimates. Three contrasts are
simultaneous-CI clean throughout; the upper flat-fidelity inversion is directionally retained but
family-wise unresolved at two alternate sampler seeds. Because the article retains
configuration-level B1 wording, the fixed-target control is not required for its present causal
scope. E5a failed its preregistered late-inversion
replication rule, so trust is removed from the title and the gated E5b diagnostic programme is not
run. The three-seed early benefit and seed-0-only late failure are reported together. This
outcome-independent narrowing is part of the contribution.

The independently trained codec preserves the same four directions. The lower reach, interior and
package-shape contrasts remain family-wise clean, while the upper inversion again becomes a
precision boundary rather than a sign reversal. Together, E7c and E7a support a robust
reach–fidelity account while making the upper-budget inversion the appropriately qualified edge
of the evidence.

The pose-indexed reference then exposes a complementary gap: explicit lossless access to observed
content substantially outperforms every registered neural reference, with a clean simultaneous
family. This does not replace the learned-model comparison; it shows that reach alone is not
enough when addressing, compression and self-fed generation remain imperfect. The final account
is sharpened by the persistent-object boundary: hierarchy is better at retaining spatial support
and continuity, while flat full-resolution context is better at retaining digit identity.
The operational prescription is therefore conditional: buy enough reach to include the relevant
past, preserve the fidelity the task needs, and distinguish having information in context from
being able to address and reuse it reliably.

## References

- Chandratreya, I. P., Charatan, D., Van Hoorick, B., Zakharov, S., Guizilini, V., Isola, P.,
  and Sitzmann, V. (2026). *MilliVid: Hierarchical Latents for Long-Range Consistency in Video
  Generation*. arXiv:2606.09056.
- Chen, H., Zhou, K., Hua, H., et al. (2026). *MemoBench: Benchmarking World Modeling in
  Dynamically Changing Environments*. arXiv:2606.27537.
- Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C., and Wood, F. (2022). Flexible
  Diffusion Modeling of Long Videos. *NeurIPS 35*.
- Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
  *NeurIPS 33*.
- Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. (2025). Self Forcing: Bridging the
  Train-Test Gap in Autoregressive Video Diffusion. *NeurIPS 2025*.
- Oshima, Y., Iwasawa, Y., Suzuki, M., Matsuo, Y., and Furuta, H. (2025). *WorldPack:
  Compressed Memory Improves Spatial Consistency in Video World Modeling*. arXiv:2512.02473.
- Peebles, W., and Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV 2023*.
- Peng, Z., Ma, J., Sun, H., et al. (2026). *Compression and Retrieval: Implicit Memory
  Retrieval for Video World Models*. arXiv:2606.23105.
- Saxena, V., Ba, J., and Hafner, D. (2021). Clockwork Variational Autoencoders. *NeurIPS 34*.
- Song, J., Meng, C., and Ermon, S. (2021). Denoising Diffusion Implicit Models.
  *ICLR 2021*.
- Srivastava, N., Mansimov, E., and Salakhutdinov, R. (2015). Unsupervised Learning of Video
  Representations using LSTMs. *ICML 2015*.
- Wei, Z., Guo, X., Li, X., et al. (2026). *Geometry-Aware Implicit Memory for Video World
  Models*. arXiv:2606.02436.
- Yan, W., Hafner, D., James, S., and Abbeel, P. (2023). Temporally Consistent Transformers for
  Video Generation. *ICML 2023*.
- Yu, J., Gao, J., Bai, J., et al. (2026). *MemLearner: Learning to Query Context Memory for
  Video World Models*. ECCV 2026; arXiv:2606.31734.
- Zakharov, A., Guo, Q., and Fountas, Z. (2023). *Long-horizon video prediction using a dynamic
  latent hierarchy*. arXiv:2212.14376.
- Zhang, S., Zhang, Z., Huang, S., et al. (2026). *MBench: A Comprehensive Benchmark on Memory
  Capability for Video World Models*. arXiv:2606.00793.
