# When Does Hierarchical Video Memory Pay?
## Reach–Fidelity Trade-offs under Fixed Context Budgets

<!-- canonical-id: binding fourth-article manuscript -->

**Keywords:** long-horizon video prediction; context compression; hierarchical latent models;
memory; observability; re-entry; diffusion transformer; reproducible evaluation.

## Abstract

A bounded-context video predictor must choose both how much history to retain and how to represent
it. We study temporal rate, spatial resolution and package depth as a controlled
**context-packaging** design space. A panning-world instrument provides exact geometric
observability at viewport re-entry, while a shared latent diffusion-transformer consumes flat or
multistream packages at explicit token budgets.

On a hash-pinned untouched holdout, hierarchy is most effective when it alone reaches the relevant
past: the registered 14-to-48-frame reach-band effects are +0.0980 and +0.0914 across two training
seeds. One simultaneous comparison family confirms hierarchy at 896 and 1,152 context tokens,
leaves 1,792 unresolved, and confirms flat full-resolution context at 2,432 and 3,072. In the
frozen latent interface, 48 fine frames cost 3,072 tokens, whereas 48 coarse frames plus a two- or
six-frame fine tail cost 896 or 1,152. The hierarchical packages therefore provide the same
nominal 48-frame addressable reach with 2.67–3.43 times fewer context tokens, without implying
equal information or equal prediction error.

The principal directions persist across sampler allocations and an independently trained codec,
with reduced precision for the upper inversion. A preregistered long-rollout study replicates the
early hierarchy benefit but not a seed-0 late inversion, rejecting a stable trust boundary. In
persistent-object long Moving-MNIST, hierarchy better preserves spatial support and trajectory
continuity, whereas flat context better preserves digit identity; both systems remain difficult in
absolute terms.

The result is conditional rather than architectural: first buy enough reach to include the
memory-relevant past, then preserve the fidelity required by the task. Hierarchy pays in the tested
reach-limited region; flat context recovers when fine detail, rather than temporal access, becomes
limiting.

## 1. Introduction

### 1.1 The empirical problem

Hierarchical video models are often motivated by a plausible story: coarse state changes slowly,
fine state renders detail, and a hierarchy should therefore retain longer history efficiently.
That story is not yet an answer. A model can appear better because it sees more frames, receives a
different prediction target, uses more inference work, or is evaluated on a metric that ignores
its hallucinations. Conversely, a hierarchy can appear useless on a benchmark where every
relevant object remains visible to a short context. Earlier recurrent, stochastic and latent video
predictors established the importance of temporal state and multi-modal futures
[Srivastava et al., 2015; Shi et al., 2015; Wang et al., 2017; Denton and Fergus, 2018;
Lee et al., 2018], but their reported scores do not by themselves identify whether a later failure
comes from insufficient temporal access, lossy state, or self-fed rollout.

This article begins with two primary explanatory questions and one robustness question rather
than a claim that hierarchy must win:

1. **Reach:** when does the supplied context contain the past that determines a re-entering event?
2. **Fidelity:** when reach is available, how much task-relevant information survives the package?
3. **Self-fed reliability:** does an apparent late failure of stored information replicate across
   training seeds?

The experimental object is context packaging. “Hierarchy” denotes one family of packages—multiple
rates, resolutions or depths—rather than the conclusion. This distinction lets a flat window, a
spatially coarse stream, and a temporally subsampled full-resolution stream be compared as design
choices under explicitly reported budgets. The controlled instrument is deliberately synthetic:
it is designed to identify a reach mechanism, not to establish natural-video superiority or a
production-scale world model.

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
Clocked recurrence, variational temporal abstraction, multiscale prediction and nested long-video
diffusion provide additional antecedents for representing different temporal rates
[Koutník et al., 2014; Kim et al., 2019; Villar-Corrales et al., 2022; Yin et al., 2023].
Outside video, Transformer-XL, compressive memory and memorizing transformers illustrate the
broader distinction between extending a flat attention window and retaining selected historical
state [Dai et al., 2019; Rae et al., 2020; Wu et al., 2022].

The predictor also belongs to a well-established latent diffusion and transformer lineage:
denoising diffusion supplies the stochastic objective [Ho et al., 2020], DDIM supplies the
deterministic finite-step sampler [Song et al., 2021], latent diffusion moves denoising into a
compressed representation [Rombach et al., 2022], and DiT supplies the token-transformer design
[Peebles and Xie, 2023]. VideoGPT, FDM and TECO illustrate related ways to represent or condition
long video sequences [Yan et al., 2021; Harvey et al., 2022; Yan et al., 2023]. These components
are antecedents, not claimed inventions of this article.

Recent work makes it especially important not to claim that memory-focused evaluation or
retrieval is absent. WorldPack combines compressed trajectory packing with retrieval and evaluates
spatial consistency on LoopNav [Oshima et al., 2025]. CaR/SceneFly, MemLearner and GIM-World
study compression or learned retrieval under annotated camera motion and occlusion
[Peng et al., 2026; Yu et al., 2026; Wei et al., 2026]. MBench and MemoBench explicitly benchmark
long-term entity/environment/causal memory or disappearance and reappearance [Zhang et al., 2026;
Chen et al., 2026]. Long-context state-space world models and explicit spatial-memory models offer
additional alternatives to retaining all history in a flat attention window
[Po et al., 2025; Wu et al., 2025]. Self Forcing directly targets the mismatch between
ground-truth-conditioned training and self-fed autoregressive inference [Huang et al., 2025],
continuing the broader exposure-bias line represented by scheduled sampling and sequence-level
training [Bengio et al., 2015; Ranzato et al., 2016].

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

![LoopMom geometry and registered reach band](../../figures/fig15_loopmom_observability_schematic.png)

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

![Frozen codec, shared predictor and context packages](../../figures/fig16_context_packaging_architecture.png)

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
\mathcal P_t=\{(\ell_i,s_i,n_i,a_i)\}_{i=1}^{L},
$$

where $\ell_i$ identifies a representation level, $s_i$ is temporal stride, $n_i$ is the number
of retained slots and $a_i$ is the lag of the most recent anchor in stream $i$. The lag set is
$K_i=\{a_i+j s_i:j=0,\ldots,n_i-1\}$. If a slot at level $\ell$ contains
$q_\ell=H_\ell W_\ell$ tokens, the package cost and nominal addressable reach are

$$
B_{\mathrm{ctx}}(\mathcal P)=\sum_{i=1}^{L}n_iq_{\ell_i},\qquad
R(\mathcal P)=\max_i\left[a_i+(n_i-1)s_i\right].
$$

This accounting makes reach exact but does not pretend that fidelity is one scalar: spatial
resolution, codec distortion and temporal sparsity preserve different information. In the frozen
interface $q_{\mathrm{fine}}=64$ and $q_{\mathrm{coarse}}=16$. Thus 48 fine slots cost 3,072
tokens, whereas 48 coarse slots plus two or six recent fine slots cost 896 or 1,152. If the target
contains $m_j$ slots at target level $j$, then
$B_{\mathrm{tgt}}=\sum_jm_jq_j$ and
$B_{\mathrm{total}}=B_{\mathrm{ctx}}+B_{\mathrm{tgt}}$. Claims about context efficiency refer to
$B_{\mathrm{ctx}}$; tables report all three quantities so they are not conflated with total
sequence length.

The common predictor models a target latent block $z_{t+1:t+m}$ conditional on the packaged
history and camera actions,

$$
p_\theta\!\left(z_{t+1:t+m}\mid \mathcal P_t,u_{t+1:t+m}\right).
$$

At diffusion time $\tau$, a clean target latent $z$ is perturbed as
$z_\tau=\alpha_\tau z+\sigma_\tau\epsilon$. The executed head uses the velocity target
$v_\tau=\alpha_\tau\epsilon-\sigma_\tau z$ and minimizes

$$
\mathcal L_v=
\mathbb E_{z,\epsilon,\tau}
\left\|v_\tau-v_\theta(z_\tau,\tau,\mathcal P_t,u)\right\|_2^2 ,
$$

following the stable diffusion parameterization described by Salimans and Ho (2022). The
regression control instead predicts the clean target latents directly. The principal paired
effect is always reported as comparator loss minus candidate loss, so positive values favor the
registered candidate. For a long package $\mathcal P_l$ and shorter comparator $\mathcal P_s$,
the reach-specific population is

$$
\mathcal D_{\mathrm{reach}}=
\{i:R(\mathcal P_s)<\Delta t_i\le R(\mathcal P_l),\
\text{answerable}_i,\ \text{foreground}_i\ne\varnothing\}.
$$

The diffusion schedule follows Ho et al. (2020) and Nichol and Dhariwal (2021), deterministic
sampling follows Song et al. (2021), and the latent-token transformer follows the DiT design
family (Peebles and Xie, 2023).

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
defined full-frame value under a foreground label. SSIM and PSNR are retained as conventional
image-reconstruction diagnostics [Wang et al., 2004], while perceptual and distributional metrics
such as LPIPS, FID and FVD motivate—but do not replace—the task-specific foreground, component and
re-entry endpoints used here [Zhang et al., 2018; Heusel et al., 2017;
Unterthiner et al., 2019].

The need for this suite is empirical rather than stylistic. On the exact archived 100-clip,
964-target CW-VAE Moving-MNIST population, a deterministic all-black predictor scores 0.7681
SSIM / 13.941 dB, compared with 0.5708 / 12.918 for the trained best-of-5 CW-VAE curve. Because
only the model's mean curves survived, this is a hash-pinned same-population metric diagnostic
without paired inference—not a budget-matched model ranking. It shows why sparse-background
full-frame metrics cannot carry the memory claim. OpenSTL and modern deterministic predictors
remain useful short-horizon competence references, but their aggregate benchmark protocols answer
a different question from the registered re-entry populations used here
[Gao et al., 2022; Tan et al., 2023].

### 3.2 Persistent-object metrics

Object claims require stable generator IDs, glyph classes, trajectories, boxes and overlaps.
Predicted components are assigned jointly with a frozen Hungarian cost [Kuhn, 1955]. Misses receive a
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
inconclusive. For clip $c$ with retained frame records $i\in c$, the paired record differences
$d_{ci}$ are summarized by

$$
\widehat\Delta=
\frac{\sum_c\sum_{i\in c}d_{ci}}{\sum_c n_c}.
$$

Each bootstrap replicate samples clips with replacement and retains all selected frames inside
each sampled clip. It therefore preserves within-rollout dependence rather than treating frames
as independent observations [Efron, 1979; Cameron et al., 2008]. These clip-cluster intervals
condition on a trained model. Training seeds are a separate uncertainty level and are displayed
individually or summarized with seeds—not clips—as units.

Sampling robustness is evaluated separately from training-seed robustness. E7c freezes four
already-supported seed-0 diffusion contrasts and five variants before evaluation: NFE 8/16/32 at
sampler seed 0 and NFE 16 at sampler seeds 1/2. A 10,000-replicate studentized max-statistic
clip-cluster bootstrap covers the resulting 20-effect family. This checkpoint-only design can
test sensitivity to inference allocation and randomness; it cannot establish robustness to
training seed, representation seed or another dataset family. Registered comparison families use
the maximum absolute studentized statistic in each replicate, yielding simultaneous intervals
that control the familywise error rate rather than interpreting many marginal intervals
independently [Westfall and Young, 1993; Romano and Wolf, 2005].

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

![Untouched frontier and context-token arithmetic](../../figures/article_confirmed_frontier.png)

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

![Five-seed head and package effects](../../figures/fig17_head_package_seed_heterogeneity.png)

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

![Registered long-loop replication](../../figures/fig18_trust_negative_replication.png)

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

![Sampler and NFE robustness](../../figures/fig19_sampler_nfe_robustness.png)

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

![Alternate-codec robustness](../../figures/fig20_alternate_codec_robustness.png)

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

![Privileged pose-indexed raw memory](../../figures/fig21_pose_indexed_raw_memory.png)

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

![Persistent-object long Moving-MNIST boundary](../../figures/fig22_phasec_v2_boundary.png)

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

This work is a controlled search for operating boundaries, not proof that one architecture always
wins. On the untouched LoopMom holdout, hierarchy is confirmed where a token-matched flat window
cannot address the re-entry past; at larger tested budgets, full-resolution fidelity recovers the
advantage. The tested packages reach 48 frames with 2.67–3.43 times fewer context tokens than a
dense fine-history package, but this is a reach-efficiency result rather than an equivalence claim.

The supporting studies narrow the scope rather than changing the conclusion. Temporal packaging is
stable across five diffusion training seeds but not under the regression head; sampler and codec
interventions retain the principal directions while reducing precision for the upper inversion;
and the proposed late trust boundary fails to replicate. Privileged pose-indexed raw access shows
that including information in context is not the same as addressing and reusing it reliably.
Persistent-object Moving-MNIST adds the complementary fidelity boundary: hierarchy better
preserves support and continuity, whereas flat context better preserves identity, and neither
system solves the task.

The practical rule is therefore conditional: buy enough reach to include the relevant past,
preserve the detail the task needs, and evaluate storage, addressing and self-fed reuse as
different failure modes. Within the tested regime, hierarchy buys reach and flat context buys
fidelity.

## References

- Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. (2015). Scheduled Sampling for Sequence
  Prediction with Recurrent Neural Networks. *NeurIPS 28*.
- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-Based Improvements for
  Inference with Clustered Errors. *Review of Economics and Statistics*, 90(3), 414–427.
- Chandratreya, I. P., Charatan, D., Van Hoorick, B., Zakharov, S., Guizilini, V., Isola, P.,
  and Sitzmann, V. (2026). *MilliVid: Hierarchical Latents for Long-Range Consistency in Video
  Generation*. arXiv:2606.09056.
- Chen, H., Zhou, K., Hua, H., et al. (2026). *MemoBench: Benchmarking World Modeling in
  Dynamically Changing Environments*. arXiv:2606.27537.
- Dai, Z., Yang, Z., Yang, Y., et al. (2019). Transformer-XL: Attentive Language Models Beyond a
  Fixed-Length Context. *ACL 2019*.
- Denton, E., and Fergus, R. (2018). Stochastic Video Generation with a Learned Prior.
  *ICML 2018*.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*,
  7(1), 1–26.
- Gao, Z., Tan, C., Wu, L., and Li, S. Z. (2022). SimVP: Simpler yet Better Video Prediction.
  *CVPR 2022*.
- Harvey, W., Naderiparizi, S., Masrani, V., Weilbach, C., and Wood, F. (2022). Flexible
  Diffusion Modeling of Long Videos. *NeurIPS 35*.
- Heusel, M., Ramsauer, H., Unterthiner, T., et al. (2017). GANs Trained by a Two Time-Scale
  Update Rule Converge to a Local Nash Equilibrium. *NeurIPS 30*.
- Ho, J., Jain, A., and Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
  *NeurIPS 33*.
- Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. (2025). Self Forcing: Bridging the
  Train-Test Gap in Autoregressive Video Diffusion. *NeurIPS 2025*.
- Kim, T., Ahn, S., and Bengio, Y. (2019). Variational Temporal Abstraction. *NeurIPS 32*.
- Koutník, J., Greff, K., Gomez, F., and Schmidhuber, J. (2014). A Clockwork RNN. *ICML 2014*.
- Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics
  Quarterly*, 2(1–2), 83–97.
- Lee, A. X., Zhang, R., Ebert, F., et al. (2018). Stochastic Adversarial Video Prediction.
  arXiv:1804.01523.
- Nichol, A., and Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models.
  *ICML 2021*.
- Oshima, Y., Iwasawa, Y., Suzuki, M., Matsuo, Y., and Furuta, H. (2025). *WorldPack:
  Compressed Memory Improves Spatial Consistency in Video World Modeling*. arXiv:2512.02473.
- Peebles, W., and Xie, S. (2023). Scalable Diffusion Models with Transformers. *ICCV 2023*.
- Peng, Z., Ma, J., Sun, H., et al. (2026). *Compression and Retrieval: Implicit Memory
  Retrieval for Video World Models*. arXiv:2606.23105.
- Po, R., Nitzan, Y., Zhang, R., et al. (2025). *Long-Context State-Space Video World Models*.
  arXiv:2505.20171.
- Rae, J. W., Potapenko, A., Jayakumar, S. M., and Lillicrap, T. P. (2020). Compressive
  Transformers for Long-Range Sequence Modelling. *ICLR 2020*.
- Ranzato, M., Chopra, S., Auli, M., and Zaremba, W. (2016). Sequence Level Training with
  Recurrent Neural Networks. *ICLR 2016*.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022). High-Resolution Image
  Synthesis with Latent Diffusion Models. *CVPR 2022*.
- Romano, J. P., and Wolf, M. (2005). Exact and Approximate Stepdown Methods for Multiple
  Hypothesis Testing. *Econometrica*, 73(4), 1237–1282.
- Salimans, T., and Ho, J. (2022). Progressive Distillation for Fast Sampling of Diffusion
  Models. *ICLR 2022*.
- Saxena, V., Ba, J., and Hafner, D. (2021). Clockwork Variational Autoencoders. *NeurIPS 34*.
- Shi, X., Chen, Z., Wang, H., et al. (2015). Convolutional LSTM Network: A Machine Learning
  Approach for Precipitation Nowcasting. *NeurIPS 28*.
- Song, J., Meng, C., and Ermon, S. (2021). Denoising Diffusion Implicit Models.
  *ICLR 2021*.
- Srivastava, N., Mansimov, E., and Salakhutdinov, R. (2015). Unsupervised Learning of Video
  Representations using LSTMs. *ICML 2015*.
- Tan, C., Li, S., Gao, Z., et al. (2023). OpenSTL: A Comprehensive Benchmark of Spatio-Temporal
  Predictive Learning. *NeurIPS 36 Datasets and Benchmarks*.
- Unterthiner, T., van Steenkiste, S., Kurach, K., et al. (2019). Towards Accurate Generative
  Models of Video: A New Metric and Challenges. arXiv:1812.01717.
- Villar-Corrales, A., Karapetyan, A., Boltres, A., and Behnke, S. (2022). MSPred: Video
  Prediction at Multiple Spatio-Temporal Scales with Hierarchical Recurrent Networks. *BMVC 2022*.
- Wang, Y., Long, M., Wang, J., Gao, Z., and Yu, P. S. (2017). PredRNN: Recurrent Neural
  Networks for Predictive Learning using Spatiotemporal LSTMs. *NeurIPS 30*.
- Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). Image Quality Assessment:
  From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*,
  13(4), 600–612.
- Wei, Z., Guo, X., Li, X., et al. (2026). *Geometry-Aware Implicit Memory for Video World
  Models*. arXiv:2606.02436.
- Westfall, P. H., and Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples and
  Methods for p-Value Adjustment*. Wiley.
- Wu, Y., Rabe, M. N., Hutchins, D., and Szegedy, C. (2022). Memorizing Transformers.
  *ICLR 2022*.
- Wu, T., Yang, S., Po, R., et al. (2025). *Video World Models with Long-term Spatial Memory*.
  arXiv:2506.05284.
- Yan, W., Zhang, Y., Abbeel, P., and Srinivas, A. (2021). VideoGPT: Video Generation using
  VQ-VAE and Transformers. arXiv:2104.10157.
- Yan, W., Hafner, D., James, S., and Abbeel, P. (2023). Temporally Consistent Transformers for
  Video Generation. *ICML 2023*.
- Yin, S., Wu, C., Yang, H., et al. (2023). NUWA-XL: Diffusion over Diffusion for eXtremely Long
  Video Generation. *ACL 2023*.
- Yu, J., Gao, J., Bai, J., et al. (2026). *MemLearner: Learning to Query Context Memory for
  Video World Models*. ECCV 2026; arXiv:2606.31734.
- Zakharov, A., Guo, Q., and Fountas, Z. (2023). *Long-horizon video prediction using a dynamic
  latent hierarchy*. arXiv:2212.14376.
- Zhang, S., Zhang, Z., Huang, S., et al. (2026). *MBench: A Comprehensive Benchmark on Memory
  Capability for Video World Models*. arXiv:2606.00793.
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. (2018). The Unreasonable
  Effectiveness of Deep Features as a Perceptual Metric. *CVPR 2018*.
