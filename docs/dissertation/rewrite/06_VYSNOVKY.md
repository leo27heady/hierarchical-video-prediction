# General Conclusions

In the dissertation research, the actual scientific and applied problem of systems analysis,
mathematical modeling, and uncertainty resolution in complex multi-step systems of hierarchical
video data generation has been solved: methods of systems analysis, mathematical models, and
intelligent information technologies have been developed that improve the efficiency and the
temporal and spatial consistency of hierarchical video generation processes. The purpose of the
research has been achieved, and all seven of the set tasks have been fulfilled.

The resource contribution is best understood along two orthogonal memory axes. The hierarchical
context-packaging model reduces the context/state memory needed to reach task-relevant history:
distant frames can be stored at a cheaper spatial or temporal rate, while a short recent tail
retains detail. TempVerseFormer reduces the activation memory needed to optimize long temporal
computations by reconstructing intermediate states during the backward pass. These are
complementary dissertation results, not one experimentally integrated architecture and not a
claim that their savings multiply. Together they support one systems principle: **compress what
must be remembered; reconstruct what need not be stored**.

The main scientific and practical results of the dissertation research are as follows.

**1.** *(Task 1)* The process of hierarchical video data generation has been characterized as a
complex control system with inherent informational, structural-algorithmic, computational, and
target uncertainties; structural, hierarchical, and temporal decompositions of the system have
been performed, and the central systemic contradiction of the field has been identified — the
resource blow-up (quadratic attention compute, linearly growing key–value memory, linearly
growing training-activation memory) and the consistency degradation of flat long-context
modeling, whereby content that leaves the field of view is not restored consistently upon its
return. The analysis of existing approaches (Chapter 1) identified codebook-specific
optimization risks in discrete vector-quantized hierarchies and delimited the implemented scope:
hierarchical modeling in compact continuous hierarchical latent state spaces. Because the
repository contains no controlled VQ-versus-continuous experiment, no representation-family
superiority is claimed. Energy-based models are reviewed as literature and later characterized
experimentally at the tested scale.

**2.** *(Task 2)* The criteria apparatus of systems analysis of generative processes has been
improved by introducing the criteria of conceptual consistency (the ability to generate unknown
combinations), hierarchical consistency (alignment of structural levels), and trajectory
consistency, the latter operationalized through measurable autoregressive-rollout degradation
metrics stratified by an exactly computable re-entry hardness label (the novel-foreground
fraction $\nu_m$, eq. 3.18). The operationalized criterion proved decision-grade in practice: it
is what localized the hierarchical model's advantage to the hard re-entry strata and the
beyond-flat-horizon bins (Table 4.6), and what turned the boundary study's negative into an
exactly stated boundary rather than an unexplained failure.

**3.** *(Task 3)* A method for estimating the state of uncertainty of spatio-temporal data has
been developed through their adaptive decomposition based on entropy metrics (the V-BLT
architecture) with N-dimensional rotary positional embedding (RoPE-ND), which allows dynamic
allocation of computational resources to high-information zones of the scene and mathematical
alignment of coordinate systems between abstraction levels of hierarchical models. Per-pixel
byte-level entropy maps were shown to act as a reliable diagnostic indicator of complex
kinematics, and the RoPE-ND coordinate-scaling mechanism provides an instrument for cross-level
attention localization in architectures that require spatial coordinate transfer. The measured
Chapter 3 predictor uses learned level/role/frame embeddings in joint attention and therefore
does not depend on RoPE-ND; the two results are complementary instruments rather than one being
the implementation basis of the other.

**4.** *(Task 4)* A method for the computationally efficient optimization of multi-step iterative
systems has been developed based on reversible transformations for time-agnostic
backpropagation, ensuring constant activation-memory consumption independent of the temporal
horizon (Lemma 1: $O(T \cdot L \cdot D) \to O(L \cdot D)$), and extended to stochastic
multi-level recurrences via the exactly invertible store-the-noise recurrence: state
reconstruction is exact at numerical precision ($5.5 \times 10^{-15}$ in fp64 for the
deterministic coupling; $8.4 \times 10^{-13}$ for the stochastic scheme), and reverse-replay
gradients coincide with backpropagation through time (cosine similarity 1.000000). The measured
memory frontier confirms the practical payoff on a consumer-grade 16 GB device: classical BPTT
consumes 12.2 GB at rollout length $K = 40$ and fails with out-of-memory at $K = 120$, whereas
the developed method remains at ≈1.1–1.2 GB through $K = 160$, with gradient checkpointing a
comparable alternative (1.47–1.97 GB). The method's applicability domain has been delimited by
the three-axis reversibility taxonomy (depth — time — computation): the temporal-rollout axis
pays and is the resource foundation of long-horizon training; the inner-optimization-loop axis
provably cannot pay against checkpointing (measured saving 0.26% of peak at $K = 128$, analytic band ≈0.6–1.0%; crossover at
$K \approx 5{,}000$–$30{,}000$ inner steps, far beyond practical refinement depths) — a
characterized negative stated as a scoping theorem of the method. It has additionally been
established by measurement that invertibility does not imply stability (perturbation growth from
$333\times$ to $1.0\times$ over 100 steps purely as a function of the coupling Lipschitz bound),
so trajectory stability is a separate design requirement.

**5.** *(Task 5)* A hierarchical mathematical model of multi-level video generation in compact
hierarchical latent state spaces has been developed, supporting **multiple context streams at
different temporal rates and spatial resolutions** over one shared attention backbone (the
context-stream family of §3.1/§3.3.5). Long-range temporal memory is carried by a long-span
stream: the **coarse stream is the measured spatial implementation** (48-frame coarse context at
16 tokens per remembered frame versus the fine level's 64 — a 4× token and asymptotically ≈16×
attention-cost advantage per memory frame), and **strided full-resolution history is the
measured temporal implementation** (span 48 at 11 anchors; it beats the coarse form under the
diffusion head on all five tested training seeds, while regression package ordering is
seed-heterogeneous) — with unidirectional
inter-level context transfer (gradient
detachment at the representation/dynamics and past/future boundaries) and a per-frame
iterative-refinement (diffusion) generation core selected by controlled mechanism analysis. The
model demonstrably restores content re-entering the field of view beyond the horizon of a flat
context window at a matched context-token budget: on 152-frame autoregressive rollouts the
hierarchical model beats the token-matched flat control on two training seeds (seed 0: ALL
+0.0189 [+0.0121, +0.0264], with the win concentrated from horizon 25 onward and peaking on hard
re-entry bins, h 25–60 HARD +0.0344 [+0.0162, +0.0531]; seed 1: ALL +0.0354 [+0.0267, +0.0447],
winning every horizon-hardness cell). This specific reach comparison also replicates under the
regression head (every bin at both seeds, ALL +0.0334 [+0.0248, +0.0423] and +0.0501
[+0.0396, +0.0612]), and the underlying single-step retrieval
capability reduces the hard re-entry error by ≈13% on both seeds (+0.0135 [+0.0080, +0.0196] and
+0.0143 [+0.0083, +0.0208]), emerging only at the larger optimization budget (null at 6,000
steps, win at 20,000). The claim is honestly scoped: **the hierarchy is validated where its
package reaches memory-relevant history under the tested budget, while full-resolution fidelity
can recover at larger tested budgets**. At the primary operating point a flat model
paying 2.7× the tokens remains the outright Pareto frontier (0.1564), so hierarchy there is an
efficiency point, not a capability necessity; and at the capability operating point the
single-step span-threshold capability (memory span exceeding the effective loop
$L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed}(1 - p_{\mathrm{hold}}))$ enables retrieval
learning in 2/2 runs versus 0/4 below the threshold). In self-fed rollout, the hierarchy's early
benefit is positive and CI-clean at all three training seeds, but the late inversion appears only
at seed 0 and reverses to a positive late hierarchy effect at seeds 1–2. The registered E5a
verdict is NOT_REPLICATED; no general trust boundary is claimed.

**6.** *(Task 6)* An intelligent information technology and a software complex (Python/PyTorch)
have been created for the systems analysis of hierarchical video generation processes,
integrating the developed methods of adaptive entropy-based tokenization, resource-efficient
reversible optimization, and hierarchical generation, together with the purpose-built
memory-horizon looping testbed and its re-entry-stratified evaluation protocol as a first-class
measurement instrument: analytic (exactly computable) hardness labels from recorded control
tracks, foreground-masked per-image metrics with verified thresholds and decoder-set ceilings,
mandatory decoded-sample review, paired clip-level cluster bootstrap confidence intervals with
common random numbers (frame-iid intervals were measured to be 2–7× too narrow at rollout
autocorrelation +0.86), and per-operating-point reference floors and ceilings. The instrument
operationalizes the trajectory-consistency criterion into a statistically disciplined measurement
procedure, and the complete benchmark trains and evaluates on a single consumer-grade 16 GB GPU.

**7.** *(Task 7)* Comprehensive experimental verification of the developed methods and technology
has been conducted (Chapter 4): a matched-capacity, matched-NFE comparison of prediction
mechanisms established that the generation-head ranking is regime-dependent — diffusion wins the
single-step oracle-coverage regime on every column (foreground error 0.0512 versus 0.0664
overall; 0.1549 versus 0.1808 on the hard, genuinely generative stratum), regression wins the
deterministic best-of-1 rollout at equal sharpness (ALL delta CI [+0.0110, +0.0227]; a
pre-registered expectation refuted and reported as such) — and included the two-sided
characterized study of energy-based descent with the official implementation (landscape ordering
recipe-dependent: stably mis-ordered under one published recipe at 21/21 probes, correctly
ordered under the other at 15/15 probes from step 6,000; descent reachability failing under both
recipes and under the paper's own sampler — a capability boundary at 16 GB-reachable scale, not a
reproduction failure). The spatial-abstraction factorial isolated resolution from window length
and returned an economic, head-conditional verdict (coarse history costs the diffusion head
−0.0082 [−0.0115, −0.0050] at 42% token savings and is statistically free for the regression head,
+0.0008 [−0.0013, +0.0029]; window extension at fixed representation is worth −0.027 and dwarfs
the abstraction cost). The temporal-capability study at the second operating point closed with a
registered negative replication of the seed-0 late trust inversion (conclusion 5) and three
transferable methodology lessons (effective-loop span sizing; the autoencoder-reuse trap;
evaluation power against plausible effect sizes). Memory-consumption
measurements confirmed the constant-memory rollout capability (conclusion 4). The measured two-level spatial
package outperforms the token-budget-matched flat control in the tested lower/interior
memory-horizon regime (the shape comparison across packagings is conclusion 7's separate,
seed-resolved statement),
and the boundary conditions of every claim are stated explicitly. The published validation of
the reversibility method ([A2]: procedurally generated 2-D rotating shapes under controlled
temporal laws) is retained as its receipt; the bridge evaluation on the standard OpenSTL
Moving-MNIST suite is a separate measured dissertation experiment and returns the predicted
negative control — flat ≈ hierarchical (180.91 vs 181.29
aggregate pixel-sum MSE; autoencoder ceiling 3.30, so the tokenizer is not the limiter) on a
benchmark with no off-screen re-entry and hence no memory horizon to carry, exactly
complementing the operating-point-A win, with the aggregate-MSE gap to the deterministic
specialist regressors (22.06–29.80) disclosed and no leaderboard claim made. A separate valid
persistent-object 48→300 Moving-MNIST study resolves the long-rollout external boundary:
hierarchy improves union error at all three seeds and trajectory/coverage cleanly at two, while
flat context preserves digit identity at all three and foreground-only error is unresolved. This
`MIXED_BOUNDARY` result shows that spatial support/continuity and fine identity fidelity can favor
different packages. The shape study (§4.4)
established that the packaging axes are not interchangeable: the win band's lower edge follows
each family's measured configuration (896 tokens for the stride-1 coarse family, two seeds; the
stride-2 512-token configuration wins on both diffusion seeds but changes its target interface),
and temporally subsampled full-resolution history beats the coarse packaging under the diffusion
head on all five training seeds (effects +0.0046 to +0.0197), whereas regression has two temporal
and three spatial material winners and the head × package interaction changes sign — the findings
motivating the scoped empirical regularities of §4.8 (ledger:
`docs/CLAIM_LEDGER.md`), whose conclusion is a testable conditional design rule rather than universal
superiority: allocate context tokens to cover the task-relevant span with the least lossy family
that achieves it; spend the remainder on resolution and recency; add depth only where the data
contain a distinct slower factor. Extension to the remaining synthetic suites (2D rotations,
3D polycubes, MMNIST-CIFAR) is stated as **future work of the dissertation programme** (Chapter 4
§§4.6.3–4.6.5 carry the pre-registered protocols; per the shell end-state rule no projected
result is claimed here). Measured in full, by contrast, is the **token-budget frontier
completion** (hier-rich at 3,072 tokens loses to flat-long, −0.0131
[−0.0201, −0.0069] at seed 0 and −0.0107 [−0.0163, −0.0058] at seed 1 — a two-seed inversion —
at equal budget and training regime; flat-short at 512 beats hier-short,
−0.0036 [−0.0063, −0.0010] — the frontier inverts at both ends, an outcome stricter than the
pre-registered convergence-at-saturation expectation and reported as measured) and the
**measured lower edge of the win band** (a pre-registered geometric prediction confirmed the
same day: at 896 tokens — the cheapest budget whose 48-slot coarse span covers the ≈48-frame
loop, against a flat window of only 14 frames — the hierarchical arm wins every horizon ×
hardness cell, +0.0399 [+0.0299, +0.0501] at seed 0 and +0.0358 [+0.0267, +0.0454] at seed 1 —
a two-seed result, the largest hierarchical advantage of the campaign), which together give the efficiency
claim its final, tested-bracket form: **the simultaneous family confirms hierarchy at the tested
896- and 1,152-token lower/interior points, leaves 1,792 unresolved, and confirms flat fidelity at
2,432 and 3,072 tokens**. Thus 1,152 is the largest tested family-wise-confirmed hierarchy win and
2,432 the smallest tested confirmed flat win; monotonicity, a unique crossing and interpolation
between tested budgets are not claimed. A separate frozen E7c study retained the expected
point-estimate sign in all 20 NFE/sampler variants, with 18 simultaneous lower bounds positive;
the upper inversion remains family-wise unresolved at alternate sampler seeds 1/2, so the result
is directionally robust rather than universally sampler-invariant. The independent-codec E7a
study retained all four registered directions: 896 reach +0.0521 [0.0374,0.0669], 1,152 interior
+0.0439 [0.0310,0.0569], package shape +0.0112 [0.0039,0.0184], and the 2,432 inversion +0.0079
[−0.0006,0.0165]. Hence representation robustness is directional for this whole-codec
intervention, while the upper edge remains family-wise precision-limited. The separate E8a
explicit-access reference reaches foreground MSE 0.0494 with raw K48 memory; the four registered
neural-minus-K48 effects range from +0.1134 to +0.1368 and all simultaneous lower bounds are
positive. Its `RAW_ACCESS_CEILING_CI_CLEAN` verdict shows that exact addressing and lossless
storage can remove much of the remaining static-world error, but the exact future poses,
raw-pixel payload and unmatched budget prohibit a neural-model or efficiency ranking.
The E6 result is therefore not averaged into one architecture score: it is retained as the
external reach–fidelity boundary showing what coarse long memory preserves and what it loses.

Beyond the numbered results, the dissertation deliberately records its **characterized
negatives** as scientific results of equal epistemic rank: (i) the energy-based generation core
is a two-sided characterized negative at 16 GB-reachable scale (ordering versus reachability
dissociate); (ii) reversibility on the inner-optimization-loop axis cannot pay against gradient
checkpointing (0.26% measured saving — the three-axis taxonomy closes this axis by analysis and
measurement); (iii) spatial abstraction is not an intrinsically better representation — its
verdict is economic and head-conditional; and (iv) the **combined-optimum readout** (regression
head + coarse history + long window: 0.1682 at 1,152 tokens; at the short window, 0.2008 at 672
tokens versus the diffusion flat control's 0.2038 at 1,152) is reported as a *measured arm
ordering* within the evaluated campaign, **not** as a factorial-interaction claim.

**The practical significance** of the obtained results lies in the development of a complex of
algorithms and software-information tools implementing the developed intelligent information
technology for the systems analysis of hierarchical video generation processes (Python/PyTorch;
repository modules for the hierarchical model, the reversible recurrence, and the testbed
generator). Unique synthetic spatio-temporal datasets have been created for testing generation
and optimization systems: 2D shapes, 3D polycubes, and the looping memory-horizon testbed with
recorded action tracks and analytic hardness labels (3,000 training clips × 100 frames; 500 test
clips × 200 frames at the primary operating point). The developed technology has been implemented
in the practical activities of ORNETIKA LLC (implementation act of October 14, 2025) and in the
educational process of VNTU (implementation act of September 19, 2025), per the acts covering the
reversible-transformer method and entropy-based tokenization. The proposed technology can be
applied to state estimation and prediction of future states of complex systems in technical,
medical, and environmental monitoring.

**Prospects for further research** follow directly from the measured boundary conditions rather
than from speculation. First, **self-fed reliability across tasks and seeds**: single-step
span-threshold retrieval exists, but the proposed late rollout inversion is training-seed-sensitive
rather than a replicated boundary. A future study, if pursued, must be newly preregistered across
multiple tasks, horizons and training seeds; it should compare teacher-forced, self-fed and
controlled-refresh conditions instead of treating E5b as localization of an established
boundary. Distribution-matching rollout objectives remain a possible intervention, with the
constant-memory reversible method of Chapter 2 providing an implementation path [REF:
Self-Forcing-class rollout training]. Second, the **extension
of the memory-horizon protocol to real video**: transferring the re-entry-stratified evaluation
methodology from the analytic testbed family to natural-video data of the KTH class (where
hardness labels must be estimated rather than computed exactly), building on the completed
OpenSTL bridge as the connectivity anchor and executing the future-work protocols of §4.6.
These directions require no revision of the present results: they extend a boundary currently
stated only for its measured testbeds and configurations.

---
