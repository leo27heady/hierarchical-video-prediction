# Article 5: integrated architecture feasibility and experiment plan

> Superseded publication framing (2 September 2026): this is no longer planned as a separate
> fifth article. Its valid architecture and gate material is now the integrated evidence block of
> one new article, together with the `A4_*` TRB hierarchy study. Current implementation and final
> two-dataset/two-seed decisions are authoritative in
> `INTEGRATED_OPENSTL_MODEL_AND_SINGLE_ARTICLE_PLAN_2026-09-02.md`. Legacy `A5_*` IDs are preserved
> only for provenance.

## 1. Objective

Determine whether a hierarchical predictor, reversible internal computation, and information-density-aware spatial routing can be combined correctly and whether the combination produces a meaningful end-to-end memory advantage without unacceptable predictive degradation on OpenSTL datasets.

The one-week goal is a rigorous go/no-go decision and a validated prototype. A guarantee of competitiveness is scientifically impossible before the runs. The plan maximizes the chance of discovering failure early and prevents a memory result from being confused with a quality result.

## 2. Core hypothesis

At comparable model capacity and training protocol, a fixed-budget hierarchical state with dynamic detail allocation and a reversible predictor core will lie on a better quality--memory Pareto frontier than a flat non-reversible counterpart.

Subhypotheses:

- H5.1: spatial/temporal hierarchy reduces retained inference state at equal reach.
- H5.2: reversible internal blocks reduce training activation memory while preserving the external predictor transition.
- H5.3: dynamic routing improves quality at a fixed detail-token quota, or preserves quality at a lower quota, compared with static allocation.
- H5.4: the full composition retains the individual advantages without instability or hidden overhead dominating the full peak.

Ratios such as 3--4x hierarchy or 2--3x reversibility are stretch targets, not acceptance assumptions. Their denominators must be explicit, and they must not be multiplied unless the measured end-to-end peak supports that multiplication.

## 3. Is the integration technically possible?

Yes, with architectural constraints.

### 3.0 Verdict on the current prototype

The current code proves partial compatibility, but it is not the final integrated architecture:

| Current implementation | Verified behavior | Consequence for Article 5 |
|---|---|---|
| `FixedRegionLayout` in `hvp_rev/fixed_region_revformer.py` | always 16 coarse plus 8 immutable region tokens per frame | stable fixed-width prototype, not dynamic allocation |
| `information_density_summary` | variance softmax weights cells inside each immutable region | density-weighted pooling, not token/byte reduction |
| `FixedRegionTokenizer` positions | learned temporal, coarse-slot, and region-slot embeddings | safe only because current regions never move; unsuitable for dynamic routes |
| `ReversibleTemporalCore` | reversible blocks inside a fixed token core | promising basis for same-forward internal reversibility |
| `ReversibleSecondOrderRollout` | wraps the predictor as a second-order additive recurrence | changes multi-step dynamics; not a memory-only substitute for ordinary rollout |
| `train_fixed_region_adapter.py` | repeatedly evaluates and selects on canonical Moving-MNIST test data | reported quality is development evidence only |
| `profile_integrated_reversible_openstl.py` | uses zero targets to profile reversible execution | memory/reconstruction diagnostic, not long-horizon forecast quality |

Thus the answer is: **the components are structurally combinable, but the current assembled prototype is not yet semantically correct for the proposed Article 5 claim.** The v2 design must retain the fixed-width idea and internal reversible blocks, replace static route identity with physical coordinate metadata, move routing outside the reversible boundary, and rebuild the evaluation split.

### 3.1 What can compose cleanly

- The hierarchy defines what representation is retained across space and time.
- The router decides which fine spatial details occupy a fixed quota.
- The fixed-budget packer converts variable semantic choices into a fixed-shape tensor plus coordinates and masks.
- Reversible transformer blocks process that tensor internally.
- The decoder combines a coarse full-frame base with fine residuals scattered using explicit coordinates.

### 3.2 What cannot be assumed

- A top-k router is not reversible.
- A variable semantic region cannot safely reuse a static slot positional embedding.
- Variable token counts padded to a maximum do not reduce ordinary dense-attention GPU memory.
- Turning on routing only at inference causes a representation distribution shift unless training included route variation or the router affects storage only after a route-robust predictor.
- The current second-order reversible rollout is not a drop-in memory-saving version of the ordinary predictor rollout.

### 3.3 Required positional design

Each token carries continuous `(x, y)`, temporal position/age, scale, and level. The baseline positional implementation should be factorized coordinate RoPE or Fourier features plus level embeddings. The following tests are mandatory:

- permutation equivariance when tokens and their metadata are permuted together;
- translation test under consistent image/token coordinate shift;
- identical-coordinate test across levels, which must remain distinguishable by scale/level;
- changing-route test, verifying that a fine token moved to another slot keeps the same prediction when metadata and content are unchanged;
- temporal-age test for context reordering and causal masking.

### 3.4 Route stability

Dynamic selection can churn between adjacent cells and cause flicker. Begin with fixed per-frame quota and one of:

- score exponential moving average;
- top-k hysteresis with an entry/exit margin;
- temporal consistency regularization;
- a small reserved quota for previously selected regions.

Measure selection Jaccard/churn, foreground coverage, route entropy, and boundary residual/flicker. If learned routing is unstable, the fallback is deterministic variance/motion routing with the same state contract, not fixed-region softmax aggregation mislabeled as compression.

## 4. Baselines and fairness

### 4.1 Internal matched baselines

The primary causal comparison uses the same encoder/decoder width, prediction target, training schedule, and approximately matched parameter count:

- `B0 Flat-Dense`: flat tokens, ordinary blocks, no dynamic routing.
- `B1 Hier-Static`: hierarchical retention, static detail allocation.
- `B2 Flat-Rev`: flat state, reversible internal blocks.
- `B3 Hier-Static-Rev`: hierarchy plus reversible core.
- `B4 Hier-Dynamic`: hierarchy plus dynamic routing, ordinary core.
- `B5 Full`: hierarchy plus dynamic routing plus reversible core.

### 4.2 External OpenSTL baselines

At minimum compare with robust representatives available in OpenSTL, such as ConvLSTM, SimVP, and TAU, subject to verified availability in the installed revision. Use identical dataset splits, input/output lengths, image sizes, evaluation preprocessing, and disclosed parameter/training differences.

External numbers copied from papers are context only. Claims of direct superiority require reruns or exactly compatible official checkpoints and protocols.

## 5. Datasets

Priority order:

1. **Moving MNIST:** integration debugging, long-horizon and controlled memory sweeps.
2. **TaxiBJ:** non-synthetic spatiotemporal data and a fast second-domain check.
3. **KTH:** natural video and motion; more expensive, used after the gates pass.

Article 5 final evidence should use OpenSTL datasets only for superiority claims. Synthetic TRB evidence stays in Article 4. Moving MNIST is synthetic in origin but is an OpenSTL benchmark; describe it accurately and do not call it natural video.

## 6. Experiment series

### A5-COR: component and state-contract correctness

Tests:

- encode/pack/unpack/decode shape and coordinate round trip;
- route permutation equivariance;
- causal-mask invariance;
- no overlap among train/validation/test samples;
- checkpoint selection never accesses canonical test data;
- exact or tolerance-bounded reconstruction of reversible blocks;
- output and gradient parity between ordinary and reversible-internal variants;
- deterministic stochastic-layer replay or stochastic layers disabled;
- memory meter calibrated against analytic tensor bytes.

Estimate: 1--1.5 engineering days; 2--6 GPU-hours.

### A5-PE: positional-encoding and moving-route validation

Compare:

- static slot embedding as a deliberate negative control;
- coordinate Fourier features plus level/time embeddings;
- coordinate RoPE plus level embedding.

Use moving-route unit tests and short Moving MNIST fits. Choose one method before full experiments. Expected winner is coordinate-aware encoding; selection is empirical.

Estimate: 0.5--1 day; 6--12 GPU-hours.

### A5-REV: reversible parity and memory sweep

For sequence lengths `10, 20, 40` and at least three block depths, compare ordinary versus reversible-internal execution.

Report:

- output/gradient error;
- convergence at a short fixed training budget;
- block-only activation memory;
- incremental CUDA allocated/reserved peak;
- full training-step peak including encoder, decoder, optimizer, and routing;
- wall-time penalty.

Pass target: numerical/optimization equivalence within declared tolerance and at least 1.7x full-step memory reduction in a block-dominated configuration. A 2--3x block-only gain is plausible; it is not guaranteed for the full model.

Estimate: 12--24 short runs; 8--24 GPU-hours.

### A5-HIER: hierarchy versus flat state

Import the locked Article 4 hierarchy allocation. Compare it to a flat state at:

- equal temporal reach;
- equal retained-state bytes;
- equal approximate compute, if attainable.

Run first on Moving MNIST, then TaxiBJ. Measure actual inference state and end-to-end inference peak. The 3--4x hypothesis is most plausible for token payload at equal reach, less so for full GPU peak.

Estimate: 6--12 configurations, initially seed 0; 12--30 GPU-hours.

### A5-ROUTE: static versus dynamic spatial allocation

At fixed fine-token quotas `K/full = 1.0, 0.5, 0.25, 0.125`, compare:

- uniform/static allocation;
- deterministic motion/variance allocation;
- learned density router;
- learned router with stability mechanism.

The coarse full-frame base is constant. Test routing during training and inference. An inference-only condition may be included as a negative/control experiment, not as the intended final method.

Metrics:

- predictive quality;
- retained-state bytes and end-to-end inference peak;
- foreground/motion coverage;
- selection churn/Jaccard;
- route entropy and utilization;
- boundary error/flicker;
- router/metadata overhead.

Pass condition: dynamic routing improves quality at fixed `K`, or permits a smaller `K` within the predeclared quality tolerance. If it only changes weights over all fixed regions, it fails the compression hypothesis.

Estimate: 16 seed-0 screening runs plus 6--12 confirmation runs; 20--55 GPU-hours.

### A5-ABL: factorial component ablation

The required six primary variants are B0--B5 above. Add only targeted interactions justified by screening:

- hierarchy x reversibility;
- hierarchy x dynamic routing;
- reversibility x dynamic routing;
- full integration.

Plot quality against three distinct memory axes rather than combining them:

1. retained inference state;
2. inference peak;
3. training-step peak.

The full model must stand out or lie on the Pareto envelope. If a component hurts both quality and its relevant memory axis, remove it from the final architecture and report the negative interaction.

Estimate: 6 variants x 2 datasets x 1 screening seed = 12 runs; confirm locked variants with 3--4 seeds after the week; 24--72 GPU-hours depending dataset and duration.

### A5-OSL: OpenSTL competitiveness

For Moving MNIST and TaxiBJ in the one-week sprint, compare locked B0/B5 with at least SimVP and one recurrent baseline under the same protocol. KTH begins only after B5 passes A5-COR, A5-REV, and A5-ROUTE.

Final article target:

- Moving MNIST, TaxiBJ, and KTH;
- four seeds for the matched internal baseline and full model;
- at least one reproducible seed or official compatible result for external baselines, with limitations explicit;
- identical resolution, sequence length, data split, and metric implementation.

One-week target:

- two seeds for B0/B5 on Moving MNIST;
- one or two seeds on TaxiBJ;
- KTH smoke test only if earlier gates pass by Day 5.

Estimate for final confirmation beyond the sprint: 48--120 GPU-hours for matched B0/B5 alone, plus 30--100 GPU-hours for external baselines and KTH, depending on hardware and schedules.

## 7. One-week calendar

| Day | Work | Exit criterion |
|---|---|---|
| 1 | implement/fix split discipline and state contract; A5-COR | no test leakage; all state tests pass |
| 2 | A5-PE and reversible internal-block parity | moving routes have valid positional semantics; gradients match |
| 3 | A5-REV memory sweep; build B0--B5 configs | reversible gate decided using full and component peaks |
| 4 | A5-HIER and A5-ROUTE Moving MNIST screens | hierarchy and routing each have a viable point |
| 5 | A5-ABL Moving MNIST; begin TaxiBJ B0/B5 | full composition reaches the provisional Pareto envelope |
| 6 | TaxiBJ and external baseline comparisons; optional KTH smoke | second-domain feasibility result available |
| 7 | locked evaluations, profiling repetitions, atomic plots/tables, go/no-go review | all claims trace to configs and manifests |

On one GPU, do not attempt all routing quotas and all external baselines. Prioritize A5-COR, A5-PE, A5-REV, B0--B5 on Moving MNIST, and B0/B5 plus SimVP on TaxiBJ. KTH and four-seed confirmation become follow-up work.

### 7.1 Executable queue on the local one-GPU host

The inspected machine has one RTX 4060 Ti with 16 GB VRAM. Because Days 1--3 include substantial implementation and debugging, the one-week plan reserves only about 55--80 usable GPU-hours.

| Queue | Unique work | GPU-hours |
|---|---:|---:|
| A5-COR | unit tests, tiny overfits, parity and meter calibration | 2--5 |
| A5-PE | three encodings, two short conditions | 3--8 |
| A5-REV | 3 horizons x 3 depths x ordinary/reversible profiles plus selected short fits | 6--15 |
| A5-ROUTE screen | four quotas x static/deterministic/learned on Moving MNIST, successively halved | 10--20 |
| A5-ABL | B0--B5, two Moving-MNIST seeds | 12--24 |
| A5-OSL gate | B0, B5, and SimVP on TaxiBJ, one seed; second seed only if time remains | 8--18 |
| **Total** | correctness plus approximately 30--45 model/profile jobs | **41--90** |

KTH is not in the mandatory one-week queue. It begins only if the full model passes Moving-MNIST and TaxiBJ gates early enough; otherwise a KTH smoke run would consume time without resolving the integration semantics.

### 7.2 Completion queue after the one-week sprint

A concluded Article 5 requires more than the feasibility week:

- freeze one coordinate positional encoding and one routing-stability mechanism;
- complete B0--B5 on Moving MNIST and TaxiBJ with four seeds for the primary B0/B1/B2/B4/B5 contrasts;
- run KTH for flat, hierarchy-only, reversible-only where meaningful, and full variants, with four seeds for the primary flat/full comparison;
- rerun at least SimVP, TAU, and ConvLSTM where feasible under the same OpenSTL data/evaluation contract;
- repeat memory profiles on an idle GPU and report allocated, reserved, retained-state, throughput, and wall time separately;
- lock test evaluation and generate seed-level statistics, atomic result figures, manifests, and claim ledger.

Expected additional compute after a successful week: approximately 220--700 GPU-hours. On the local GPU this is roughly 2--6 weeks of elapsed unattended training, plus 2--4 researcher weeks for architecture iteration, analysis, and paper/manuscript writing. A second comparable GPU approximately halves training wall time but not engineering or analysis time.

The one-week result is therefore a reliable architecture decision, not a guarantee of OpenSTL comparability. The only defensible way to become sure is to pass the matched B0/B5 and external-baseline gates; no architectural argument can substitute for those measurements.

## 8. Expected memory behavior

The mechanisms apply to different memory pools:

```text
full training peak
  = parameters + optimizer + gradients
  + encoder/decoder activations
  + predictor activations
  + retained hierarchical state
  + attention/workspace/router overhead
```

- Hierarchy and routing primarily reduce retained state and attention cost.
- Reversibility primarily reduces predictor activations during training.
- Router metadata and coarse-base tokens add overhead.
- Parameter/optimizer memory is largely unchanged.

Therefore a 3x state reduction and a 2x predictor-activation reduction do not imply a 6x full peak. Report a component-wise waterfall plus measured full peak. The win condition is a better Pareto point, not a multiplication of idealized ratios.

## 9. Risk register and fallback

| Risk | Early signal | Response |
|---|---|---|
| dynamic route causes flicker | high churn and boundary error | add hysteresis/EMA; use static/deterministic routing fallback |
| positional mismatch | slot permutation changes result | replace slot embedding with coordinate-aware encoding |
| reversible rollout changes dynamics | output parity fails after first step | keep external transition ordinary; move reversibility inside blocks |
| dense padding erases savings | CUDA peak unchanged as active `K` falls | use fixed smaller `K` per config or true packed sparse kernels |
| router collapses | entropy/utilization collapse | quota regularization, load balancing, temperature schedule |
| quality below OpenSTL | B5 dominated by B0/SimVP | report memory trade-off; narrow claim; improve backbone before scale-up |
| two weeks insufficient | gates pass but seeds incomplete | preserve locked configs and schedule 1--3 GPU-weeks confirmation |

## 10. Publication decision

Proceed to a full Article 5 only if:

- the state/position/reversible correctness gates pass;
- at least two components independently improve their relevant memory axis without unacceptable quality loss;
- the full composition is non-dominated or scientifically explains a meaningful interaction;
- a second OpenSTL dataset supports the direction.

If those conditions fail, do not force an integrated-superiority article. Publish the rigorous hierarchy investigation as Article 4, retain reversibility and routing as scoped component studies, and revise the architecture before further expensive training.
