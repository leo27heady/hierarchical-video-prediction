# 06 — ВИСНОВКИ (rewrite, phase 1)

> **Status:** English working text of the full ВИСНОВКИ replacement (the old draft's ВИСНОВКИ was
> a literal "TEXT" stub); the Ukrainian pass translates it mechanically. Genre spots marked
> **[UA-PASS: …]**, author-input spots **[AUTHOR: …]**, citation spots **[REF: …]**.
> **Sources of truth:** `docs/dissertation/rewrite/00_FRONT_MATTER.md` (the 7 results and 4
> novelty items — the 7 numbered conclusions below mirror the front-matter results 1:1 against
> the 7 tasks of `01_VSTUP.md`; the two files must stay in sync), `docs/draft/CH4_RESULTS.md`
> (audited numbers, including the §4.5 PAIR OUTCOME of 2026-07-06 — the boundary study closed
> RED; the honest-scoping sentences below carry the trustworthy-memory boundary),
> `docs/SPATIAL_FACTORIAL.md` (incl. the FRONTIER COMPLETION pre-registration),
> `docs/draft/T7_BRIDGE.md` (the MEASURED OpenSTL bridge — authoritative for the bridge numbers
> in result 7), `docs/draft/VALIDITY_AUDIT.md`,
> `docs/dissertation/rewrite/03_ROZDIL2.md` (reversibility receipts), `05_ROZDIL4.md`
> (chapter-level conclusions these generalize), `docs/dissertation/DISS_DIGEST.md` §6 («ВИСНОВКИ —
> 7 results restated against the 7 rewritten tasks; honest scoping») and §8 author decisions.
> **Genre note:** ВИСНОВКИ opens with the standard resolution formula, gives the numbered main
> results mapped to the tasks, then practical significance and prospects. No new claims appear
> here — every number is a restatement of Chapters 2–4.

---

# ВИСНОВКИ

*(EN: General conclusions.)*

In the dissertation research, the actual scientific and applied problem of systems analysis,
mathematical modeling, and uncertainty resolution in complex multi-step systems of hierarchical
video data generation has been solved: methods of systems analysis, mathematical models, and
intelligent information technologies have been developed that improve the efficiency and the
temporal and spatial consistency of hierarchical video generation processes. The purpose of the
research has been achieved, and all seven of the set tasks have been fulfilled. [UA-PASS: opening
formula «У дисертаційній роботі розв'язано актуальну науково-прикладну задачу…; мету дослідження
досягнуто, усі поставлені задачі виконано.»]

The main scientific and practical results of the dissertation research are as follows.

**1.** *(Task 1)* The process of hierarchical video data generation has been characterized as a
complex control system with inherent informational, structural-algorithmic, computational, and
target uncertainties; structural, hierarchical, and temporal decompositions of the system have
been performed, and the central systemic contradiction of the field has been identified — the
resource blow-up (quadratic attention compute, linearly growing key–value memory, linearly
growing training-activation memory) and the consistency degradation of flat long-context
modeling, whereby content that leaves the field of view is not restored consistently upon its
return. The analysis of existing approaches (Chapter 1) — including the author's own experimental
finding that hierarchical video models over discrete vector-quantized token spaces are unstable
to train at practically reachable scale — substantiated the chosen direction: hierarchical
modeling in compact continuous hierarchical latent state spaces, with energy-based models
reviewed as literature and later characterized experimentally.

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
kinematics, and the RoPE-ND coordinate-scaling mechanism provides the mathematical basis of
cross-level attention localization — the inter-level context-signal transfer instrument of the
hierarchical model.

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
measured temporal implementation** (span 48 at 11 anchors; matches or exceeds the coarse form
under the diffusion head at seed 0 [EXPECTED-B2-s1: replication queued]) — with unidirectional
inter-level context transfer (gradient
detachment at the representation/dynamics and past/future boundaries) and a per-frame
iterative-refinement (diffusion) generation core selected by controlled mechanism analysis. The
model demonstrably restores content re-entering the field of view beyond the horizon of a flat
context window at a matched context-token budget: on 152-frame autoregressive rollouts the
hierarchical model beats the token-matched flat control on two training seeds (seed 0: ALL
+0.0189 [+0.0121, +0.0264], with the win concentrated from horizon 25 onward and peaking on hard
re-entry bins, h 25–60 HARD +0.0344 [+0.0162, +0.0531]; seed 1: ALL +0.0354 [+0.0267, +0.0447],
winning every horizon-hardness cell), the result is mechanism-agnostic (the regression twin
pairing wins every bin at both seeds, ALL +0.0334 [+0.0248, +0.0423] and +0.0501
[+0.0396, +0.0612]), and the underlying single-step retrieval
capability reduces the hard re-entry error by ≈13% on both seeds (+0.0135 [+0.0080, +0.0196] and
+0.0143 [+0.0083, +0.0208]), emerging only at the larger optimization budget (null at 6,000
steps, win at 20,000). The claim is honestly scoped: **the hierarchy is validated where the
memory horizon demands it *and* where the memory contents remain trustworthy** — within the
continuously refreshed recent window of the primary operating point, or near a ground-truth
anchor at the capability operating point — while at the primary operating point a flat model
paying 2.7× the tokens remains the outright Pareto frontier (0.1564), so hierarchy there is an
efficiency point, not a capability necessity; and at the capability operating point the
single-step span-threshold capability (memory span exceeding the effective loop
$L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed}(1 - p_{\mathrm{hold}}))$ enables retrieval
learning in 2/2 runs versus 0/4 below the threshold) does not yet convert to a self-fed rollout
capability across a 213-frame loop (all pre-registered rollout gate cells null over 256 frames).
[UA-PASS: «ієрархія підтверджена там, де цього вимагає горизонт пам'яті і де вміст пам'яті
залишається достовірним».]

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
has been conducted (Chapter 4): a matched-capacity, matched-compute comparison of prediction
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
the abstraction cost). The temporal-capability boundary study at the second operating point
closed under its binding pre-committed rule with the trustworthy-memory boundary statement
(conclusion 5) and three transferable methodology lessons (effective-loop span sizing; the
autoencoder-reuse trap; evaluation power against plausible effect sizes). Memory-consumption
measurements confirmed the constant-memory rollout capability (conclusion 4). The measured two-level spatial
package outperforms the token-budget-matched flat control exactly in the memory-horizon regime
(the shape comparison across packagings is conclusion 7's separate, seed-gated statement),
and the boundary conditions of every claim are stated explicitly. The published validations of
the reversibility method ([A2]: 2D/3D rotations, Moving MNIST, MMNIST-CIFAR, TaxiBJ, KTH) are
retained as its receipts; the bridge evaluation on the standard OpenSTL Moving-MNIST suite is
**measured** and returns the predicted negative control — flat ≈ hierarchical (180.91 vs 181.29
aggregate pixel-sum MSE; autoencoder ceiling 3.30, so the tokenizer is not the limiter) on a
benchmark with no off-screen re-entry and hence no memory horizon to carry, exactly
complementing the operating-point-A win, with the aggregate-MSE gap to the deterministic
specialist regressors (22.06–29.80) disclosed and no leaderboard claim made. The shape study (§4.4)
established that the packaging axes are not interchangeable: the win band's lower edge follows
each family's token arithmetic (896 tokens for the stride-1 coarse family, two seeds; 512 for
the stride-2 family, +0.0330 [+0.0246, +0.0419] every cell at seed 0 [EXPECTED-B1-s1: queued]),
and temporally subsampled full-resolution history beats the coarse packaging under the diffusion
head at seed 0 (+0.0197 [+0.0143, +0.0252]; [EXPECTED-B2-s1/REG: seed and regression-head
resolution queued]) — the findings consolidating into the empirical laws of §4.8 (ledger:
`docs/CLAIM_LEDGER.md`), whose conclusion is a conditional design rule rather than universal
superiority: allocate context tokens to cover the task-relevant span with the least lossy family
that achieves it; spend the remainder on resolution and recency; add depth only where the data
contain a distinct slower factor. Extension to the remaining synthetic suites (2D rotations,
3D polycubes, MMNIST-CIFAR) is stated as **future work of the dissertation programme** (Розділ 4
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
claim its final, edge-measured form: **hierarchy wins the band of token budgets in which its
coarse span crosses the memory horizon while the flat window does not — approximately 896 to
≈2×10³ tokens (upper edge measured to lie in (1,792, 2,432)) at the primary operating
point, with the advantage peaking at the band's lower
edge — and inverts outside the band**, not dominance at every budget.

Beyond the numbered results, the dissertation deliberately records its **characterized
negatives** as scientific results of equal epistemic rank: (i) the energy-based generation core
is a two-sided characterized negative at 16 GB-reachable scale (ordering versus reachability
dissociate); (ii) reversibility on the inner-optimization-loop axis cannot pay against gradient
checkpointing (0.26% measured saving — the three-axis taxonomy closes this axis by analysis and
measurement); (iii) spatial abstraction is not an intrinsically better representation — its
verdict is economic and head-conditional; and (iv) the **combined-optimum readout** (regression
head + coarse history + long window: 0.1682 at 1,152 tokens; at the short window, 0.2008 at 672
tokens versus the diffusion flat control's 0.2038 at 1,152) is reported as a *measured arm
ordering* within the evaluated campaign, **not** as a factorial-interaction claim. [UA-PASS:
«охарактеризовані негативні результати як наукові результати рівного епістемічного рангу»;
«виміряне впорядкування конфігурацій, а не факторне твердження».]

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
medical, and environmental monitoring. **[AUTHOR: per decision Q5, no copyright certificate has
been filed — do not mention one; if a new registration is filed before defense, insert its title
and number here and in the front matter.]**

**Prospects for further research** follow directly from the measured boundary conditions rather
than from speculation. First, the **capability regime at scale**: the boundary study established
that single-step span-threshold retrieval exists at the capability operating point but does not
yet survive pure self-feeding across a 213-frame effective loop — the open question is whether
larger model/data scale, larger optimization budgets (retrieval was measured to be a
late-emerging capability), or training objectives that keep the memory contents trustworthy along
the rollout (distribution-matching rollout objectives, whose long-horizon training memory the
constant-memory reversible method of Chapter 2 makes affordable [REF: Self-Forcing-class rollout
training]) convert the single-step capability into a rollout capability. Second, the **extension
of the memory-horizon protocol to real video**: transferring the re-entry-stratified evaluation
methodology from the analytic testbed family to natural-video data of the KTH class (where
hardness labels must be estimated rather than computed exactly), building on the completed
OpenSTL bridge as the connectivity anchor and filling the [EXPECTED] transfer shells of §4.6. These directions require no revision of the
present results: they extend the measured boundary, which is stated exactly.

---

## Working notes (not part of the text)

- The 7 numbered conclusions mirror `00_FRONT_MATTER.md` «Main scientific and practical results»
  items 1–7 one-to-one (which in turn map 1:1 to the 7 tasks of `01_VSTUP.md`); wording drift
  between the three files must be resolved toward the front matter at the UA pass.
- Conclusion 5 carries the post-PAIR-OUTCOME scoping («and where the memory contents remain
  trustworthy») — this supersedes any earlier draft of the scoping sentence that mentioned only
  the memory horizon; the front-matter general-conclusions paragraph should be updated to match
  at the next front-matter pass. **[AUTHOR: confirm.]**
- All CIs are 95% paired clip-level cluster bootstrap unless the chapter text discloses otherwise
  (single-step gates: see §4.3.1 design-effect widening; §4.5.4 sample-iid disclosure).
- Deliberately absent per the rules: any capability headline for operating point B (the study
  closed RED); any factorial-interaction reading of the combined optimum; any new TaxiBJ/KTH
  *claim* (the §4.6 transfer shells are [EXPECTED]-marked plans under the Q4 run-or-strike
  disposition, not commitments — struck before defense if unfilled); any "competitive with the
  OpenSTL leaderboard on aggregate MSE" phrasing (refuted by measurement: 180.91 vs SimVP 26.69);
  the copyright-certificate sentence.
- Result 7's generalization sentence is dual-branch by construction: the [EXPECTED] and
  [FALLBACK] markers must be resolved (or the sentence softened to "planned") before defense,
  in sync with §§4.6.3–4.6.7 of `05_ROZDIL4.md`. The frontier clause is RESOLVED (measured
  2026-07-07, in sync with §4.3.6): inversion at both ends; win band [896, e) with BOTH
  edges measured — lower edge 896 two-seed (R-A), upper edge e bracketed to (1,792, 2,432) by
  R-F/R-G (2026-07-09/10); peak at the lower edge — edge-measured band wording.
