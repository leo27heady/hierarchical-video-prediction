# Front Matter

---

## 1. Title Page

- **Topic (UA, approved, verbatim — fixed):** «Системний аналіз та інтелектуальні технології
  консистентного моделювання процесів ієрархічної генерації відеоданих»
- **Topic (EN):** *Systems Analysis and Intelligent Technologies for Consistent Modeling of
  Hierarchical Video Data Generation Processes*
- Candidate: Kulyk Leonid Ruslanovych. Specialty 124 «System Analysis», field 12 «Information
  Technologies». Supervisor: Prof. O. B. Mokin, Dr. Sc. (Eng.). VNTU, Vinnytsia — 2026.
- UDC: 004.054:[004.032.26+004.85] **[AUTHOR: confirm the UDC classification and the title-page
  quotation against the approved institutional record.]**

---

## 2. Abstract


The purpose of the dissertation research is to improve the efficiency and the temporal and spatial
consistency of hierarchical video data generation processes through the development of systems
analysis methods, mathematical models, and intelligent information technologies. The dissertation
addresses the actual scientific and technical problem of systems analysis, mathematical modeling,
and uncertainty resolution in complex multi-step systems for the hierarchical generation of
spatio-temporal data.

The systems contribution separates two resource bottlenecks. Hierarchical multi-rate context
packaging reduces the state memory required to reach distant history; reversible temporal
optimization reduces the activation tape required to train long computations. In conceptual
form, these act on different terms of
$M_{\mathrm{system}}=M_{\mathrm{parameters}}+M_{\mathrm{optimizer}}+
M_{\mathrm{context/state}}+M_{\mathrm{activations}}+M_{\mathrm{transient}}$.
They are complementary dissertation strands rather than one causally integrated pipeline. Their
shared prescription is to compress what must be remembered and reconstruct what need not be
stored.

The introduction presents the relevance and justification of the research topic in the context of
systems analysis requirements; describes the purpose, object, subject, and tasks of the research;
states the scientific novelty; shows the connection of the dissertation with research programs;
states the practical significance of the results, their approbation and publications; and
describes the personal contribution of the applicant.

### Chapter-by-Chapter Summary

**Chapter 1** performs a systems analysis of hierarchical video data generation processes and the
criteria of their consistency. Generative video models are analyzed as open complex dynamic
control systems; structural, hierarchical, and temporal decompositions of the system state space
are performed, and the types of uncertainty (informational, structural-algorithmic, computational,
and target) are classified. The analysis of existing approaches establishes the central systemic
contradiction of the field: state-of-the-art video generators extend a *flat* (single-level)
spatio-temporal context, which leads to a rapid growth of attention memory and computational cost
with context length and, simultaneously, to the degradation of temporal consistency over long
horizons — content that leaves the field of view is not re-generated consistently upon return. It
is further shown, based on the author's own experimental findings, that hierarchical video models
over discrete (vector-quantized) token spaces are unstable to train at practically reachable
scale, which motivates hierarchical modeling in *compact continuous hierarchical latent state
spaces*. Energy-based models are reviewed as literature, with their resource barrier noted (their
capability boundary is later characterized experimentally in Chapter 4). The criteria apparatus is
improved by introducing new types of consistency (conceptual, hierarchical, trajectory), with
trajectory consistency operationalized through measurable autoregressive-rollout and
re-entry-stratified metrics. The research tasks are defined.

**Chapter 2** develops mathematical methods for uncertainty estimation and resource-efficient
modeling of multi-step video processes. A method for estimating spatio-temporal uncertainty based
on entropy metrics (the V-BLT architecture) is proposed, enabling adaptive decomposition of video
data. The mathematical apparatus of N-dimensional rotary positional embedding (RoPE-ND) is
developed to preserve the structural coherence of data, including the alignment of coordinate
systems across abstraction levels of hierarchical models. A method for optimizing the
computational resources of a complex system based on reversible transformations (TempVerseFormer)
is proposed, and the time-agnostic backpropagation algorithm is mathematically substantiated,
ensuring constant (independent of the optimization-horizon length) consumption of activation
memory. The chapter closes with the extension of this apparatus to stochastic multi-level
recurrences: an exactly invertible stochastic recurrence (the "store-the-noise" scheme) is proved
and measured — reverse-replay gradients coincide with backpropagation through time (cosine
similarity 1.000000), the invertibility error is at numerical precision ($5.5\times10^{-15}$ in
fp64; $8.4\times10^{-13}$ for the stochastic scheme), and activation memory in a $K$-step rollout
is $O(1)$ in $K$ — together with a three-axis applicability taxonomy (reversibility across depth,
time, and computation) that delimits where reversibility pays (long temporal rollouts) and where
it cannot (inner optimization loops, where the achievable saving versus gradient checkpointing is
0.26% measured, with an analytic band of 0.6–1.0%).

**Chapter 3** develops a mathematical model of multi-stream video sequence generation that
packages long-range context across temporal rate and spatial resolution. The state space is a compact two-level hierarchical latent
representation (a frozen two-level convolutional autoencoder interface with per-level decodable
coarse and fine latents); its coarse stream is the measured spatially compressed implementation
of long-history packaging, not the only possible carrier of the memory horizon. Per-frame generation uses an iterative refinement (denoising
diffusion) core, selected empirically by the matched-capacity, matched-NFE mechanism analysis of Chapter 4. The
inter-level transfer of context signals is realized within a single joint attention mechanism
with per-token level, role, and frame conditioning (generalizing the cross-level cross-attention
of the earlier design), with unidirectional control enforced by the information-design asymmetry
of the two levels and by gradient detachment at the representation and rollout boundaries; the
causality of the system is enforced by sequence construction and the autoregressive recursion;
the autonomous autoregressive video-prediction algorithm rolls the model out window-by-window
with a frozen pixel decoder. The chapter also gives the formal
definition of the *memory-horizon looping testbed* and its re-entry-stratified evaluation
protocol — a criterion-driven measurement instrument in which the hardness of every predicted
frame ("recoverable only from memory") is an exactly computable analytic label — and derives the
memory and complexity estimates of the rollout algorithm.

**Chapter 4** presents the experimental research and practical application of the developed
methods and technologies. The information base includes procedurally synthesized datasets (2D
shapes, 3D polycubes) used in the reversible-transformer studies, the purpose-built looping
memory-horizon testbed (the primary benchmark: a $192\times64$ toroidal world observed through a
panning $64\times64$ viewport, with 67% of the world off-screen at any instant and a re-entry loop
period of approximately 48 frames), and a measured bridge evaluation on the standard Moving MNIST
suite (canonical 10→10 protocol, n = 10,000; the model halves the persistence baseline's
perceptual error, and the flat-versus-hierarchical null on this memory-horizon-free benchmark
serves as the predicted negative control). A matched-capacity, matched-NFE comparison of
prediction mechanisms (regression,
diffusion, energy-based descent) on a shared backbone shows that regression averages over possible
futures while diffusion commits to sharp, correctly located content (foreground error 0.0512
versus 0.0664; on the hard, genuinely generative subset 0.1549 versus 0.1808), and includes a
two-sided capability study of Energy-Based Transformers using the official implementation — a
characterized negative result: one published recipe yields a stably mis-ordered energy landscape,
the other repairs the ordering, yet gradient descent from the prescribed initialization does not
reach structured outputs at the scale reachable in this work. On a hash-pinned untouched LoopMom
holdout, the registered pair-specific reach effects at the 896-token operating point are +0.0980
and +0.0914 across two training seeds, and the whole-population effects are +0.0474 and +0.0436.
A simultaneous max-statistic family confirms hierarchical packaging at 896 and 1,152 context
tokens, leaves 1,792 unresolved, and confirms flat full-resolution context at 2,432 and 3,072
tokens. The frozen latent interface therefore supplies the same nominal 48-frame reach with
2.7–3.4 times fewer context tokens in the two confirmed hierarchical packages, while the
upper-budget reversals show that compression is not free. All 20 registered sampler/NFE point
estimates retain their direction, with 18 simultaneous lower bounds above zero; an independently
trained continuous codec retains all four registered directions, although the 2,432-token
inversion becomes family-wise unresolved. A privileged pose-indexed raw-memory reference reaches
foreground MSE 0.0494 and establishes a remaining explicit-access gap without constituting a
token-matched competitor. On the persistent-object 48→300 Moving-MNIST boundary, hierarchy
preserves spatial support and continuity better, whereas flat context preserves digit identity
better. The constant-activation-memory capability is demonstrated separately: classical
backpropagation through time exhausts GPU memory at rollout depth $K=120$, while the reversible
implementation remains at approximately 1.1–1.2 GB through $K=160$.
The practical application of the developed methods in the tasks of ORNETIKA LLC and in the
educational process of VNTU is described (per the existing implementation acts, which cover the
reversible-transformer method and entropy tokenization). [A2]'s published 2-D rotating-shape
validation is reported as the receipt of the reversibility method; [A1] supplies the 2-D/3-D
entropy-tokenization examples. Moving MNIST and TaxiBJ are separate dissertation bridges, while
MMNIST-CIFAR and KTH remain explicitly unexecuted extensions. In the valid persistent-object
48→300 Moving-MNIST bridge, hierarchy lowers union error on all three seeds and improves
trajectory/coverage cleanly on two, while flat context preserves digit identity on all three;
foreground-only error is unresolved. The result is reported as a mixed external boundary rather
than a universal model ranking.

The general conclusions present the main results of the dissertation research, confirming the
fulfillment of all the set tasks, with honest scoping: the hierarchy is validated where its reach
covers memory-relevant history under a bounded budget, while full-resolution fidelity can recover
at larger tested budgets. A seed-0 late self-fed inversion did not replicate at two new training
seeds and is therefore not stated as a general trust boundary. The energy-based mechanism is a
characterized negative; reversibility is scoped by its applicability taxonomy. The four central
reach–fidelity directions survive all registered NFE/sampler variants and an independently
trained continuous codec at the point-estimate level; the upper inversion is explicitly retained
as a family-wise precision boundary under alternate sampler and codec conditions. A separate
privileged pose-indexed raw-memory reference reaches foreground MSE 0.0494 and beats all four
registered neural references with positive simultaneous lower bounds, showing that geometric
reach and reliable explicit addressing are distinct while making no token-matched model claim.
The protected persistent-object long Moving-MNIST result further separates the axes: coarse
long-memory packaging preserves spatial support and continuity, whereas flat full-resolution
context preserves fine digit identity.

### 7 items, mapped 1:1 to the 7 tasks

1. Characterized the process of hierarchical video data generation as a complex control system
   with inherent informational, structural-algorithmic, computational, and target uncertainties;
   performed structural, hierarchical, and temporal decompositions of the system and identified
   the central systemic contradiction — the resource blow-up and consistency degradation of flat
   long-context modeling — that the subsequent chapters resolve. *(Task 1)*

2. Improved the criteria apparatus of systems analysis of generative processes by introducing the
   criteria of conceptual, hierarchical, and trajectory consistency, and operationalized
   trajectory consistency through measurable autoregressive-rollout metrics stratified by an
   exactly computable re-entry hardness label. *(Task 2)*

3. Developed a method for estimating the state of uncertainty of spatio-temporal data through
   their adaptive decomposition based on entropy metrics (V-BLT) with N-dimensional rotary
   positional embedding, which allows dynamic allocation of computational resources to
   high-information zones of the scene and mathematical alignment of coordinate systems between
   abstraction levels of hierarchical models. *(Task 3)*

4. Developed a method for the computationally efficient optimization of multi-step iterative
   systems based on reversible transformations for time-agnostic backpropagation, ensuring
   constant activation-memory consumption independent of the temporal horizon; extended it to
   stochastic multi-level recurrences via an exactly invertible stochastic recurrence
   (store-the-noise; gradients coincide with backpropagation through time), and delimited its
   applicability domain by a three-axis taxonomy. *(Task 4)*

5. Developed a mathematical model of multi-stream video generation in compact hierarchical
   latent state spaces, supporting context streams at different temporal rates and spatial
   resolutions over a shared attention backbone, with unidirectional context transfer (gradient
   detachment at the representation and rollout boundaries) and a per-frame iterative-refinement
   generation core selected by controlled mechanism analysis; measured spatial and temporal
   packages restore content beyond a contiguous flat window at bounded token budgets. *(Task 5)*

6. Created an intelligent information technology and a software complex (Python/PyTorch) for the
   systems analysis of hierarchical video generation processes, integrating the developed methods
   of adaptive tokenization, resource-efficient optimization, and hierarchical generation,
   together with the purpose-built memory-horizon looping testbed and its re-entry-stratified
   evaluation protocol as a first-class measurement instrument. *(Task 6)*

7. Conducted comprehensive experimental verification of the developed methods and technology:
   a matched-capacity, matched-NFE comparison of prediction mechanisms (including the
   two-sided characterized study of energy-based descent), single-step and 152-frame
   autoregressive-rollout evaluations with clip-clustered paired-bootstrap confidence intervals, and
   memory-consumption measurements; the developed system outperforms the token-budget-matched
   flat control at the tested lower/interior points where its package alone reaches
   memory-relevant history, while tested larger budgets expose a full-resolution-fidelity
   inversion; the boundary conditions of every claim are stated explicitly. *(Task 7)*

The main results of the dissertation research are published in 10 scientific works: 3 articles in
professional journals of Ukraine in specialty 124; 1 article indexed in the Scopus scientometric
database; and 6 conference abstracts. The final bibliographic details of the fourth article and
the six conference abstracts require the applicant's publication records.

### Scientific Novelty

1. **For the first time**, a methodology for the systems analysis and evaluation of long-range
   memory in hierarchical video generation processes was developed — a purpose-built
   memory-horizon testbed whose re-entry hardness labels are exactly computable from recorded
   control tracks, with a re-entry-stratified evaluation protocol and a matched-context-token-budget
   comparison design — and it was demonstrated with its help that **long-span multi-stream
   context packaging** — a hierarchical allocation of a fixed context-token budget across
   spatial resolution and temporal rate, of which the coarse-level-memory paradigm of
   hierarchical latent video models (§1.3.2, adopted prior art) is the measured spatial form
   and strided full-resolution history the measured temporal form — restores content that re-enters the field of view beyond the horizon of
   a flat context window at the same context-token budget, with tested lower/interior and upper
   regimes bracketed by explicit token arithmetic; unlike existing evaluation practice, which aggregates
   prediction error without isolating memory-dependent content, the developed methodology makes
   the memory contribution of the packaging directly measurable and localizes the improvement of
   temporal consistency to registered reach-differential populations.

2. **For the first time**, a method for the computationally efficient optimization of processes in
   multi-step iterative systems was developed based on reversible transformations for
   time-agnostic backpropagation, which, unlike classical backpropagation through time, ensures
   constant memory consumption regardless of the depth of the temporal horizon; the method is
   extended to stochastic multi-level recurrences via an exactly invertible stochastic recurrence
   with noise storage, and its applicability domain is delimited by a three-axis taxonomy of
   reversibility (across depth, time, and computation).

3. **Improved** the criteria apparatus of systems analysis of the results of generative models by
   introducing the criteria of conceptual consistency (the ability to generate unknown
   combinations), hierarchical consistency (alignment of structural levels), and trajectory
   consistency, the latter operationalized through autoregressive-rollout degradation metrics
   stratified by re-entry hardness, which allows a more comprehensive and measurable assessment of
   the adequacy of complex generative systems.

4. **Improved** the method for assessing the state of uncertainty of spatio-temporal data through
   their adaptive decomposition using N-dimensional rotary positional embedding and per-pixel
   entropy maps, which allows dynamic allocation of computational resources during the analysis of
   complex visual scenes.

### implementation acts — kept as-is, they match reality

- practical activities of **ORNETIKA LLC** (Kyiv): the developed information technology was used
  in designing intelligent systems for the analysis of spatio-temporal video data and anomaly
  detection in complex technical environments; the resource-efficient methods (TempVerseFormer)
  and adaptive entropy tokenization substantially reduced the computational cost of modeling long
  temporal sequences (implementation act of October 14, 2025);
- the **educational process of VNTU** in the disciplines *Information Technologies for Monitoring
  and Analysing the State of Complex Systems* and *Internet of Things and Intelligent Data
  Analysis* for PhD
  students of specialty 124 (implementation act of September 19, 2025).

### Practical significance

The practical significance lies in the development of a complex of algorithms and software tools
implementing the developed intelligent information technology for the systems analysis of
hierarchical video generation processes (Python/PyTorch; repository modules for the hierarchical
model, the reversible recurrence, and the testbed generator). Unique synthetic spatio-temporal
datasets were created for testing generation and optimization systems: 2D shapes, 3D polycubes,
and the looping memory-horizon testbed with recorded action tracks and analytic hardness labels
(3,000 training clips × 100 frames; 500 test clips × 200 frames). The proposed technology can be
applied to state estimation and prediction of future states of complex systems in technical,
medical, and environmental monitoring.

### Keywords

systems analysis; complex system; mathematical modeling; intelligent information technology;
decision making; neural network; deep learning; artificial intelligence; video data generation;
**hierarchical video generation**; **compact hierarchical latent state spaces**; **memory
horizon**; **temporal consistency**; **context packaging**; **reversible transformations**;
**diffusion refinement core**; autoregressive prediction; state estimation; forecasting;
consistency.

---

## 3. Applicant's Publications

The following stable labels are used throughout the dissertation:

- **[A1]** — Kulyk L. R., Mokin O. B. «Scaling Video Prediction with Spatio-Temporal Patches»,
  *Visnyk of Vinnytsia Politechnical Institute*, 2025, No. 5, pp. 129–139,
  DOI 10.31649/1997-9266-2025-182-5-129-139.
- **[A2]** — Kulyk L. «Temporal modeling with reversible transformers» (TempVerseFormer),
  *Neural Computing and Applications*, 2026, Vol. 38, article 30,
  DOI 10.1007/s00521-025-11707-3, sole author, Scopus-indexed.
- **[A3]** — Kulyk L. R., Mokin O. B. «Methods for Ensuring Consistent Generation in Diffusion
  Models», *Visnyk of Vinnytsia Politechnical Institute*, 2024, No. 4, pp. 75–85,
  DOI 10.31649/1997-9266-2024-175-4-75-85.
- **[A4]** — the fourth article (in finalization): a controlled causal study of long-span
  context packaging (rate × resolution × depth) for long-horizon video prediction — the
  memory-horizon testbed, the matched-token-budget frontier with measured edges, the shape study,
  the matched-capacity, matched-NFE mechanism study including the EBT characterized negative, and
  the supporting reversibility theory. **[AUTHOR: supply target venue and status.]**
- Six conference theses. **[AUTHOR: supply the six verified bibliographic entries.]**

---

## 4. Symbols and Abbreviations

| Abbreviation | Expansion |
|---|---|
| VNTU | Vinnytsia National Technical University |
| AE | Autoencoder |
| AR | Autoregressive |
| BPTT | Backpropagation Through Time |
| CI | Confidence Interval |
| CRN | Common Random Numbers |
| DDIM | Denoising Diffusion Implicit Models |
| DDPM | Denoising Diffusion Probabilistic Models |
| DiT | Diffusion Transformer |
| EBM | Energy-Based Model |
| EBT | Energy-Based Transformer |
| EMA | Exponential Moving Average |
| fg-MSE | Foreground Mean Squared Error |
| GPU | Graphics Processing Unit |
| LPIPS | Learned Perceptual Image Patch Similarity |
| MCMC | Markov Chain Monte Carlo |
| MSE | Mean Squared Error |
| NFE | Number of Function Evaluations |
| OOM | Out of Memory |
| RoPE-ND | N-dimensional Rotary Positional Embedding |
| SSIM | Structural Similarity Index Measure |
| V-BLT | Video Byte-Latent Transformer |
| VAE | Variational Autoencoder |

---

## 5. Table of Contents

- Introduction
- Chapter 1. Systems Analysis of Hierarchical Video Generation and Its Consistency Criteria
- Chapter 2. Mathematical Methods for Uncertainty Estimation and Resource-Efficient Modelling of
  Multi-Step Video Processes
- Chapter 3. A Hierarchical Mathematical Model for Video Generation with Long Multi-Stream Context
- Chapter 4. Experimental Evaluation and Practical Application
- General Conclusions
- References
- Appendices: publications, implementation acts, experiment configurations, and the
  OpenSTL/Moving-MNIST bridge

<div style="page-break-after: always;"></div>

# Introduction

## Justification and Relevance

The rapid development of information technologies and artificial-intelligence methods over the
last decade has produced new classes of mathematical models capable of solving complex problems of
analysis, prediction, and synthesis of multidimensional data. A special place among them is
occupied by generative video models, which have evolved from simple statistical tools into highly
complex multi-level systems capable of forming high-quality spatio-temporal video data. Such
systems are critically important for many applied fields: from medical diagnostics and
environmental monitoring to the prediction of the state of technical systems and the construction
of simulators for autonomous control.

However, despite significant progress, the control of the generation process remains
insufficiently formalized from the standpoint of systems analysis methodology. Modern
architectures — diffusion and autoregressive video models — are mostly treated as "black-box"
deep-learning algorithms. Operating under high structural, informational, and target uncertainty,
these systems are prone to generating anomalies — spatial, logical, or temporal inconsistencies —
which makes their outputs unreliable for decision-making. The use of single-level (flat) latent
spaces mixes the levels of abstraction of the visual scene (semantic composition, geometry,
texture) within one representation, which leads to the loss of hierarchical control over the
generation process.

The dominant engineering answer to the consistency problem — extending a *flat* spatio-temporal
context window — runs into an objective systemic barrier. First, the computational cost and memory
of the attention mechanism grow rapidly with the number of context tokens (quadratically in
computation, linearly in the key–value cache), so long flat contexts are resource-prohibitive
precisely at the horizons where they are most needed. Second, even where a long flat context is
affordable, the temporal consistency of long generated sequences degrades: content that leaves the
field of view is not restored consistently upon its return, and autoregressive error accumulation
(exposure bias) destabilizes the trajectory of the generation process. Third, the identification
of parameters of long-horizon models by backpropagation through time requires activation memory
that grows linearly with the number of steps of the modeled process, which makes deep training
trajectories unattainable on practically available computing systems.

These are two different memory bottlenecks and the dissertation treats them separately. For
systems analysis, peak memory is decomposed conceptually as

$$
M_{\mathrm{system}} =
M_{\mathrm{parameters}} + M_{\mathrm{optimizer}} +
M_{\mathrm{context/state}} + M_{\mathrm{activations}} + M_{\mathrm{transient}} .
$$

The hierarchical context-packaging model of Chapters 3–4 reduces
$M_{\mathrm{context/state}}$ for a required temporal reach by storing distant history at a lower
spatial rate or a lower temporal rate. The reversible TempVerseFormer method of Chapter 2 reduces
$M_{\mathrm{activations}}$ for long temporal computation by reconstructing intermediate states
during the backward pass. These are complementary resource axes, not one integrated mechanism and
not evidence that their savings automatically multiply. The common design principle is:
**compress what must be remembered; reconstruct what need not be stored**.

A natural remedy is the hierarchical decomposition of the latent state space. Discrete
vector-quantized hierarchies are an important reviewed family, but they also introduce codebook
and assignment-specific optimization risks; this dissertation does not present a controlled
VQ-versus-continuous comparison. The implemented direction is therefore stated constructively:
hierarchical modeling of video generation processes in **compact continuous hierarchical latent
state spaces**, in which multiple context streams can package long-range information at
different temporal rates and spatial resolutions, context signals are transferred
unidirectionally (with gradient detachment), per-frame generation is performed by an iterative
refinement (denoising diffusion) core selected by controlled mechanism analysis, and the memory
barrier of long-horizon activation-memory growth is bounded by reversible transformations that
make backpropagation time-agnostic within their measured applicability domain. Chapter 1
explicitly marks the representation evidence boundary; Chapter 4 tests robustness to a second
continuous codec seed rather than claiming representation-family universality. That intervention
preserves all four registered directions; three contrasts remain simultaneous-CI clean, while
the alternate-codec 2,432-token inversion remains positive but family-wise unresolved. A
separately gated exact-pose raw-memory reference then measures the remaining explicit-access gap:
its 48-frame mosaic attains foreground MSE 0.0494 and all four registered neural-minus-reference
simultaneous lower bounds are positive. Because this reference receives raw pixels and exact
future poses, it is reported as a privileged ceiling rather than a token-matched competitor.

Thus an objective systemic contradiction arises between the necessity of ensuring multi-level
consistency of video data over long time horizons (which requires long memory horizons and deep
hierarchies) and the strict resource constraints of modern computing systems (the attention-memory
blow-up of flat contexts and the linear growth of training memory with horizon length). Resolving
this contradiction requires the creation of new mathematical methods of decomposition,
resource-efficient control, and intelligent information technologies of systems analysis, which
determines the high relevance of this dissertation research.

## Relationship to Research Programmes, Plans and Topics

The dissertation was carried out in accordance with the research plans of Vinnytsia National
Technical University, in particular within the state-budget research topics:

- Research topic No. **[AUTHOR: research-project number 1]**, *Development of Information
  Technologies and Mathematical Methods for Intelligent Analysis of Spatio-Temporal Data in
  Complex Systems* (2021–2023), in
  which the applicant developed the method of adaptive estimation of spatio-temporal uncertainty
  and entropy-based tokenization of video data;
- Research topic No. **[AUTHOR: research-project number 2]**, *Systems Analysis and Modelling of
  Multi-Step Processes under Resource Constraints and Uncertainty* (2024–2026), in which the applicant
  developed the method of computational-resource optimization based on reversible transformations
  and the **hierarchical generative model of video data**. **[AUTHOR: verify both approved
  project titles and the applicant's role against the official project reports.]**

## Aim and Research Tasks

**The purpose of the dissertation research** is to improve the efficiency and the temporal and
spatial consistency of hierarchical video data generation processes through the development of
systems analysis methods, mathematical models, and intelligent information technologies.

The dissertation's components serve **one backbone question**:
*how can hierarchical decomposition, temporal abstraction, and resource-aware rollout improve
the consistency of long-horizon video generation under bounded memory and computation?* Each
contribution has a defined role in answering it: adaptive entropy-based decomposition and
RoPE-ND (Chapter 2) — how to represent uncertain visual information efficiently; reversible
transformations and time-agnostic backpropagation (Chapter 2) — how to train and roll out long
processes within memory limits; the multi-stream hierarchical predictor (Chapter 3) — the
demonstrated system; the memory-horizon testbed and re-entry-stratified protocol (Chapters 3–4)
— the causal instrument for measuring whether context memory actually matters; the frontier and
shape experiments (Chapter 4) — the evidence for how to allocate a fixed context budget; the
energy-based study (Chapter 4) — the measured mechanism-selection negative.

For the fourth article, context packaging is the manipulated design space. **Reach** asks whether
the package contains the memory-relevant past; **fidelity** asks what useful information survives
compression or subsampling; reliable reuse under self-fed generation is treated as a robustness
question because the proposed late trust boundary did not replicate. The terminal empirical
summary is conditional: **hierarchy buys reach; flat context buys fidelity**.

To achieve this purpose, the following scientific and practical tasks had to be solved:

1. Analyze the state and problems of generative video-data models as complex systems; classify
   the types of uncertainty and identify the systemic causes of anomalies (inconsistencies) in
   the generation process, including the resource and consistency limitations of flat
   long-context modeling.

2. Improve the criteria apparatus of systems analysis of generative processes by extending the
   taxonomy of consistency types (conceptual, hierarchical, trajectory) and operationalizing
   trajectory consistency through measurable rollout metrics stratified by re-entry hardness.

3. Develop a method for estimating the state of uncertainty of spatio-temporal data through their
   adaptive decomposition using entropy metrics and the mathematical apparatus of
   multidimensional rotary positional embedding.

4. Develop a method for optimizing the computational resources of multi-step iterative models
   that ensures constant, process-length-independent consumption of memory on the basis of
   reversible mathematical transformations; extend it to stochastic multi-level recurrences and
   delimit its domain of applicability.

5. Develop a mathematical model of multi-stream video generation in compact hierarchical latent
   state spaces, supporting context streams at different temporal rates and spatial resolutions
   over a shared attention backbone, with unidirectional transfer of context signals (gradient
   detachment at the representation and rollout boundaries) and a per-frame iterative-refinement
   generation core selected by controlled mechanism analysis.

6. Create an intelligent information technology and a software complex for the systems analysis
   of hierarchical video generation processes, integrating the developed methods of adaptive
   tokenization, resource optimization, and hierarchical generation, together with a
   purpose-built memory-horizon testbed and re-entry-stratified evaluation protocol as its
   measurement instrument.

7. Conduct comprehensive experimental studies and approbation of the developed methods and
   technology on synthetic and open datasets — including a matched-capacity, matched-NFE
   comparison of prediction mechanisms — and compare their efficiency and consistency with
   modern analogues.

## Object and Subject of Research

**The object of research** is the processes of systems analysis, mathematical modeling, and
control of complex multi-level systems of video data generation under structural, informational,
and computational uncertainty.

**The subject of research** is the mathematical models, methods, and intelligent information
technologies of systems analysis, uncertainty estimation, criteria-based evaluation, and control
in hierarchical generative systems.

## Research Methods

To solve the set tasks, a complex apparatus of modern science was used: the methodology of systems
analysis for the decomposition of generative processes and the classification of uncertainty and
consistency types; probability theory, mathematical statistics, and information theory for entropy
estimation and the assessment of spatio-temporal uncertainty of data; the theory of stochastic
processes for the mathematical description of the iterative refinement (denoising diffusion)
generation core, with the Markov-chain formalism applied insofar as autoregressive generation
constitutes a Markovian state transition and, within the experimental mechanism study, for the
description of the gradient-descent (Langevin) sampling of energy-based models; optimization
methods and matrix/tensor algebra for the formalization of training objectives and the development
of reversible transformations and attention mechanisms; machine- and deep-learning methods,
including backpropagation, for the identification of parameters of the mathematical models;
elements of decision-making theory for the construction of criteria-based decision procedures —
the pre-registered decision rules of the experimental program and the re-entry-stratified
evaluation protocol, which operationalizes the acceptance or rejection of model configurations
as a formal criteria-based choice; and methods of statistical inference (paired bootstrap
confidence intervals, common random numbers) for the experimental evaluation protocol.
- In **[A1]** (hierarchy/BLT-adaptation article), the applicant proposed and implemented the
  method of adaptive decomposition of spatio-temporal data based on entropy metrics (the V-BLT
  architecture) with multidimensional rotary positional embedding, and the procedural synthesis
  of video datasets. The two-stage V-BLT architecture (local byte-level encoder/decoder around a
  global latent transformer) is the applicant's **first realization of the hierarchical
  adaptation of video generation** — the two-level decomposition whose mature, memory-carrying
  form Chapter 3 develops; the dissertation's hierarchical line thus runs [A1] (two-stage
  entropy-adaptive hierarchy) → [A4] (two-level latent hierarchy with coarse-level long-range
  memory).
- In **[A2]** (TempVerseFormer, Springer *Neural Computing and Applications* 2026, Scopus, sole
  author), the applicant developed and mathematically substantiated the algorithm of
  time-agnostic backpropagation based on reversible transformations, which resolved the
  computational uncertainty of long-horizon training. The journal is described only as
  Scopus-indexed; no unverified quartile claim is used.
- In **[A3]** (consistency-criteria article), the applicant improved the criteria apparatus for
  the evaluation of complex generative systems by formalizing the notion of «conceptual
  consistency».
- In **[A4]** (the fourth article, in finalization) and within the dissertation research, the
  applicant developed the hierarchical latent video generation model with long-range coarse-level
  memory; constructed the memory-horizon looping testbed and the re-entry-stratified evaluation
  protocol; conducted the matched-capacity, matched-NFE study of prediction mechanisms,
  including the two-sided capability study of energy-based transformers (a characterized negative
  result); developed the theory of the exactly invertible stochastic recurrence with the
  three-axis reversibility taxonomy; and performed all complex experimental studies.

## Validation and Presentation of Results

The main provisions, results, and conclusions of the dissertation were reported and discussed at
6 scientific conferences: **[AUTHOR: verify every event below against the applicant's records and
bind each event to the corresponding conference publication.]**

- 50th Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2021;
- 51st Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2022;
- *Problems of Higher Mathematical Education: Contemporary Challenges*, VNTU, Vinnytsia, 2022;
- 52nd Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2023;
- 53rd Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2024;
- Dynamical System Modeling and Stability Investigation (DSMSI-2023), Kyiv, 2023.

## Publications

Based on the dissertation topic, 10 scientific works were published, including: 3 articles in
professional journals of Ukraine included in the Ministry of Education and Science list for
specialty 124 ([A1], [A3], [A4]);
1 article in an international scientific journal indexed by the Scopus scientometric database
([A2]); and 6 abstracts of reports at international and all-Ukrainian scientific-practical
conferences. **[AUTHOR: supply the venue/status of [A4] and the six conference entries, then
confirm that the 3+1+6 count is correct.]**

## Dissertation Structure and Length

The dissertation consists of an introduction, four chapters, conclusions to each chapter, general
conclusions, a list of references, and appendices.

- **Chapter 1** performs the systems analysis of hierarchical video generation processes and the
  criteria of their consistency: generative video models as complex dynamic control systems;
  structural, hierarchical, and temporal decomposition; the classification of uncertainty types;
  the analysis of existing approaches (flat long-context diffusion and autoregressive generators
  and their resource/consistency barrier; hierarchical latent modeling as the remedy; the
  instability of discrete hierarchical video models; energy-based models as reviewed literature);
  the improved consistency-criteria apparatus; and the statement of the research tasks.
- **Chapter 2** develops the mathematical methods for uncertainty estimation and
  resource-efficient modeling of multi-step video processes: the entropy-based V-BLT method,
  RoPE-ND, reversible transformations with temporal chaining (TempVerseFormer), time-agnostic
  backpropagation, and the exactly invertible stochastic recurrence with the three-axis
  reversibility taxonomy.
- **Chapter 3** develops the multi-stream mathematical model of video sequence generation with
  context packaged across temporal rate and spatial resolution in compact hierarchical latent state spaces: the two-level
  latent state space, the iterative-refinement generation core, joint inter-level attention with
  unidirectional control (information-design asymmetry plus gradient detachment at the
  representation and rollout boundaries), the autoregressive rollout algorithm with its memory
  and complexity estimates, and the formal definition of the memory-horizon testbed with the
  re-entry-stratified evaluation protocol.
- **Chapter 4** presents the experimental research and practical application: the information
  base; the matched-capacity, matched-NFE mechanism study (including the two-sided characterized study of
  energy-based descent); the single-step and 152-frame autoregressive-rollout (after a 48-frame
  ground-truth warm-up) evaluations of the hierarchical
  model; the memory-consumption measurements; and the implementations.

The total volume of the work is **[AUTHOR: final page count]** pages; the main part comprises
**[AUTHOR: final main-text page count]** pages, with the final figure and table counts obtained
mechanically during typesetting. The bound scientific bibliography currently contains 79
external sources. The appendices contain copies of the implementation acts, the preregistration
trail, and the software and experiment-configuration map of the developed information technology.

<div style="page-break-after: always;"></div>

# Chapter 1 — Systems Analysis of Hierarchical Video Generation and Its Consistency Criteria

## 1.1 Generative video models as complex control systems

The development of artificial-intelligence and machine-learning methods over the last decade has
produced new classes of mathematical models capable of solving complex problems of data analysis,
prediction, and synthesis. A special place among them is occupied by generative models, which have
transformed from simple statistical tools into extremely complex multi-level information systems.
Modern generative models — in particular those operating on spatio-temporal video data — hold
billions of parameters and demand enormous computational resources [1]. Yet despite impressive visual results, the control of the generation
process in such models remains insufficiently formalized. For the most part these models are
treated as "black-box" deep-learning algorithms, which complicates guaranteeing the stability of
their operation, the predictability of their results, and their resource efficiency [2, 3].

To address these problems it is necessary to change the paradigm of perceiving generative
algorithms and to consider them as open complex dynamic systems. In this formulation the process
of video data generation is not merely a mathematical transformation of information but a
purposeful process of changing the state of a complex system under control actions and the
influence of the external environment. Iterative generative models — diffusion models [4], autoregressive systems, and energy-based models employing Markov chain Monte Carlo —
fall squarely under the definition of a complex dynamic control system.

A complex system controlling the process of video data generation is characterized by input
actions, a state space, and output data. The input actions of such a system are the initial noise
distribution, which plays the role of an entropy source, and a set of conditioning vectors (for
example, textual descriptions, semantic masks, previous frames of the video sequence, or recorded
control/action tracks), which play the role of control signals. The state space is the
multidimensional latent or pixel space in which the system resides at every discrete moment of
time. The output data is the final generated object, which must satisfy the specified criteria of
quality and consistency. The scheme of a modern generative video model as an example of an open
complex dynamic control system is shown in Fig. 1.1.

![Video generation as an open dynamic control system](../../figures/fig30_control_system.png)

**Figure 1.1 — Video generation as an open dynamic control system.** Noise acts as an entropy
source, previous frames and actions as control signals, and autoregressive feedback makes output
quality a property of the complete self-fed trajectory. Source: author’s schematic.

In accordance with classical systems theory [5], the object
under study is characterized by a large number of interrelated elements, nonlinear interactions,
emergence, and a hierarchical structure. In the context of computer vision the elements of the
system are individual pixels or latent features (tokens). Their interaction is provided by
attention mechanisms and convolutional operations, which model spatio-temporal correlations.
Nonlinearity is achieved through nonlinear activation functions in the artificial neural networks
that form the core of the system.

An important characteristic of such a system is the iterative nature of its state transitions.
Unlike generative adversarial networks (GAN) [6] or variational
autoencoders (VAE) [7], which attempt to map an input noise vector to an
output image in a single computation step, iterative models such as denoising diffusion
probabilistic models (DDPM) [4] decompose this process into a multitude of discrete steps.
This approach fundamentally changes the mathematical nature of the system: the state evolves in
time (or in the pseudo-time of optimization steps), and every next state depends on the previous
one, the current parameters, and the control action.

Mathematically, the evolution of the state of such a system is described by a recurrent equation
of Markov type:

$$x_t = f(x_{t-1}, u_t, \theta, \xi_t) \tag{1.1}$$

where $x_t$ is the system state (a multidimensional tensor of video data or its latent
representations) at step $t$; $u_t$ is the vector of control actions (generation conditions);
$\theta$ is the parameter matrix of the system (the neural network weights); $\xi_t$ is the
stochastic component (the random number generator); and $f$ is the nonlinear transition function
describing the law of state change.

The presence of the stochastic component $\xi_t$ indicates that the system operates under
uncertainty. Every generation step is a probabilistic process, and the system produces not a
deterministic result but a sample from a specified multidimensional probability distribution.
According to research on complex systems under risk and uncertainty [8, 9], such operating conditions make the system
extremely sensitive to initial conditions and create the threat of anomalous states. In
generative models this manifests as artifacts in images or violations of physical logic in video.

A complex system always possesses a hierarchical structure [10],
where every level describes the object with a certain degree of detail or abstraction. In video
models this hierarchy is expressed with particular clarity. At the **macro level — the highest
level of abstraction** — the system operates with semantic concepts (global style, the mutual
arrangement of objects, scene composition). At the **meso level**, structural contours and
geometry are determined. At the **micro level — the lowest level of abstraction, the highest
level of detail** — high-frequency textures, lighting, and individual pixels are formed.

It is the hierarchical structure that gives rise to such a systemic property as emergence: a
meaningful image arises only thanks to complex spatial and temporal correlations between elements
at different levels of the hierarchy. If the system controlling the generation process cannot
ensure coherence between these levels, the emergent property is destroyed (for example, an object
receives a correct texture imposed on a distorted global shape). Hence the task of ensuring
stable generation reduces to the task of ensuring the stability of a complex hierarchical system.

### 1.1.1 Generation as an optimal-control problem

Traditional deep-learning approaches attempt to solve the stability problem extensively, by
increasing the number of parameters $\theta$ and the volume of the training sample. This path,
however, has physical limits and leads to exponential growth of computational complexity. A need
therefore arises for mathematical methods of optimal control of the generation trajectory.

Consider the generation process as an optimal-control problem. At the initial moment $t = 0$ the
system state $x_0$ is fully chaotic (a pure-noise distribution), corresponding to the state of
maximal informational uncertainty. At the final moment $t = T$ the system must arrive at a state
$x_T$ belonging to the target distribution of real video data (a state of high internal
structural organization).

The formalization of this process requires a systemic criteria apparatus. In the present research
the quality of the terminal state and of the trajectory leading to it is described by a
**consistency criterion functional** $J(x, u)$, which quantitatively expresses the degree of
non-correspondence between the current state $x$ and the target concept under conditions $u$.
Generation then becomes a constrained optimization problem over the state space:

$$x^{*} = \arg\min_{x} J(x, u) \tag{1.2}$$

subject to the state-transition law (1.1). The smaller the value of the criterion, the more
consistent and realistic the generated object.

![Multi-step generation on a consistency landscape](../../figures/fig31_consistency_landscape.png)

**Figure 1.2 — Conceptual multi-step generation on the landscape of a consistency criterion
$J(x,u)$.** The trajectory moves from a high-criterion noise state toward a structured
low-criterion region. This is a systems-level abstraction; only energy-based models instantiate
the landscape as an explicit learned scalar energy. Source: author’s schematic.

In the context of the intelligent information technology being developed, it is important to
clearly delimit the roles of two fundamental system concepts: the **global consistency criterion**
and **entropy**. The global criterion (1.2) is trajectory-level: it is a functional the modeling
system strives to minimize along the entire multi-step generation trajectory in order to reach
integral consistency of the object; in this dissertation it is operationalized not by a learned
scalar energy but by the measurable consistency criteria of §1.4 — in particular the
re-entry-stratified rollout metrics of §1.4.5, which the hierarchical model of Chapter 3 is built
to optimize. Entropy, by contrast, acts as a **local diagnostic indicator**: it quantifies the
level of structural informational uncertainty at specific spatio-temporal coordinates of the
data. Entropy maps allow the system to "understand" where complex moving contours are located and
where there is static background, and accordingly to adaptively redistribute limited
computational resources (this apparatus is realized in the V-BLT subsystem in Chapter 2).
Together these two system concepts form a closed loop of analysis and control.

Controlling this system runs into **computational uncertainty** on two fronts. First, at the
stage of parameter identification (training): since the transition function is realized by a deep
neural network, identifying the system parameters along a long chain of iterations requires
storing all intermediate states (activations) in operating memory, so the memory of
backpropagation through time grows linearly with the trajectory length. This creates a resource
barrier that rules out long control trajectories on practically available computing systems.
Second, at the stage of operation (inference): extending the temporal context of a flat attention
mechanism inflates its key–value memory linearly and its computation quadratically with context
length (analyzed quantitatively in §1.3.1). Overcoming the first front requires special methods
of structural optimization — in particular the introduction of reversible transformations that
reduce the memory-consumption complexity (developed in Chapter 2); overcoming the second requires
hierarchical decomposition of the state space with a compact long-memory level (developed in
Chapter 3).

### 1.1.2 Applied domains

Beyond its theoretical significance, understanding generative video models as complex control
systems has enormous practical importance for various applied fields. The tasks solved by such
systems intertwine closely with the objects of study of the specialty "System Analysis".

The first important application domain is medical imaging and the development of intelligent
systems for supporting clinical decision-making. Complex multidimensional spatial data (MRI, CT)
often require reconstruction, super-resolution, or cross-modal generation to minimize the
radiation load on the patient. In this domain any generation anomaly (a "hallucination" in the
form of non-existent tumors, or the loss of real anatomical structures) has fatal consequences,
so the requirement of absolute structural consistency is critical.

The second domain is ecological and technical monitoring using remote sensing data. Satellite
data often contain gaps due to cloud cover or technical failures. Generative systems are used for
spatial or temporal interpolation of missing data and for predicting the development of
ecological catastrophes (for example, the spread of forest fires). In this case the generated
intermediate states must strictly obey the laws of physics, thermodynamics, and the topology of
the terrain.

The third domain is the creation of educational and training simulators (for example, for
autonomous vehicle control systems). Training robots requires millions of hours of synthetic
video data with diverse scenarios (including extreme ones). The requirement imposed on the
generative system is long-term temporal coherence: objects cannot suddenly disappear, deform over
time, or violate the logic of physical interactions. In particular, an object that leaves the
camera's field of view must reappear in a consistent state and position when the camera returns
to it — a requirement that, as will be shown in §1.3.1 and §1.4.5, directly demands a long memory
horizon from the modeling system.

The analysis of the listed practical domains shows that modern generative systems are subject to
the following strict systemic requirements:

1. High visual quality and detail of the target data distribution;
2. Controllability of the system and the ability to respond to control actions;
3. Hierarchical representations without loss of coupling between levels;
4. Absence of spatial, temporal, and logical anomalies — including over long horizons, when
   content leaves and re-enters the field of view;
5. Resource efficiency under constraints of computational memory;
6. Interpretability of the process.

Summarizing, it can be stated that generative video models have reached the stage of evolution at
which their further development by empirical architecture search exhausts its potential. What is
required is the treatment of these algorithms as complex systems controlling dynamic processes,
which demands the development of a corresponding mathematical apparatus: methods of
decomposition, identification of uncertainty types, introduction of new consistency criteria, and
creation of algorithms for optimal resource control. The following subsections of this research
are devoted precisely to these questions.

## 1.2 Decomposition of the video generation system

The principal instrument of systems analysis in the study of complex systems is the method of
decomposition — the division of an integral system into interrelated subsystems, elements, and
couplings, with the aim of simplifying analysis and revealing deep properties and sources of
uncertainty. For the multidimensional system controlling the process of video data generation it
is expedient to apply several orthogonal kinds of decomposition: structural (functional),
hierarchical (semantic), and temporal.

### 1.2.1 Structural decomposition

Structural (functional) decomposition treats the generative system as a set of functional
information-processing blocks. In most modern architectures, irrespective of their mathematical
basis, three basic macro-subsystems are distinguished [11].

The first subsystem is the **Encoder** — the block of condition preprocessing and encoding. Its
function is to transform "raw" control signals (for example, a textual description, semantic
masks, or the pixels of context video frames) into a formalized multidimensional feature space.

The second subsystem, the **Iterative Generator / Denoiser**, is the central block of iterative
transformation. It is the mathematical core of the system, directly executing the
state-transition equation and performing optimization or denoising steps in the feature space.
This subsystem is responsible for eliminating the entropy of the initial state and forming a
consistent structure.

The third subsystem, the **Decoder**, is the block of decoding and reconstruction, which
translates the optimized latent state of the system back into the space of human perception (the
pixel space of an image or video stream).

![Structural decomposition of the generation system](../../figures/fig32_system_decomposition.png)

**Figure 1.3 — Structural decomposition of the video-generation system.** The encoder and
decoder define the representation interface; the central predictor carries the controlled
state transition and iterative refinement. Source: author’s schematic.

Of greatest interest for systems analysis is the central subsystem, since it is where the
state-control process takes place and the principal work of overcoming uncertainty is
concentrated. Its detailed study requires the deeper kinds of decomposition that follow.

### 1.2.2 Hierarchical decomposition of the state space

Complex systems have a pronounced hierarchical nature [10].
Accordingly, the state space of the central generation subsystem cannot be "flat" — it must be
decomposed into macro, meso, and micro levels.

At the **macro level** (the level of highest abstraction) the system operates with semantic
vectors of low spatial resolution but high semantic saturation. Here the questions of conceptual
composition are decided: the presence of objects, their mutual arrangement, and correspondence to
the specified condition.

At the **meso level** of abstraction (medium resolution) the system forms structural contours,
geometry, and the spatial relations of object parts.

At the **micro level** (high resolution) the questions of local pixel organization are decided:
the coherence of textures, lighting, and micro-details.

The problem of many modern generative systems — for example, standard diffusion models built over
variational autoencoders [12, 7] — is that they operate in a
single-level latent space, ignoring the necessity of explicit hierarchical decomposition [13, 14, 15]. This produces an internal control conflict in which macro- and
micro-decisions are taken simultaneously. The mixing of levels is a direct source of generation
anomalies: a correct object texture may be imposed on a distorted global shape.

![Hierarchical decomposition of visual state](../../figures/fig33_visual_hierarchy.png)

**Figure 1.4 — Hierarchical decomposition of visual state.** Macro denotes the highest
abstraction and lowest spatial rate; micro denotes the pixel-proximal level and highest detail.
Source: author’s schematic.

The systems approach requires the development of mathematical models of explicit hierarchical
decomposition with mechanisms of strict coherence between these orthogonal levels of
representation. This decomposition is the core of the present dissertation's topic. Moreover — a
property central to the direction developed in Chapters 2–4 — an explicitly decomposed hierarchy
allows the levels to run at *different temporal rates and context lengths*: a compact upper
(coarse) level with a low information rate can afford to retain a long temporal context at the
same memory budget for which a flat single-level representation affords only a short one, and can
therefore serve as the carrier of the system's long-range temporal memory (quantified in §1.3.1
and developed in Chapter 3).

### 1.2.3 Temporal decomposition

The third necessary kind of stratification is temporal decomposition. When modeling systems that
generate video data, time must be considered in two entirely different, orthogonal dimensions,
which creates the so-called "two-dimensional temporal grid".

The first dimension concerns the internal **pseudo-time** of the generation process — the
partition of the continuous process of state change into discrete refinement steps
$k = 1, \dots, K$ *per generated frame*: the steps of the reverse diffusion process, the token
steps of an autoregressive decoder, or the descent steps of an energy-based sampler. Every such
step is a micro-transition of the system in phase space, during which the total entropy of the
state decreases and its consistency increases.

The second dimension is introduced upon the transition to processing video sequences and reflects
the **physical time axis** $t = 1, \dots, T$, where every discrete slice is a separate frame. The
complexity of system control lies in the fact that the generative model must simultaneously
ensure stability in both temporal dimensions: the system state must evolve correctly along the
optimization axis $k$ for every individual frame, and at the same time all generated frames must
be spatio-temporally coherent along the physical axis $t$, forming smooth motion kinematics.

The physical axis, in turn, gives rise to the key quantitative characteristic of this
dissertation: the **memory horizon** — the maximal temporal distance $\Delta t$ at which
information about the past state of the modeled process can still influence the generated present
state. For a system whose central subsystem attends to a flat context window of $w$ frames, the
memory horizon is bounded by $w$; events separated by more than $w$ frames (for example, content
that left the field of view and returns later) are informationally inaccessible to it.

![Physical time and refinement pseudo-time](../../figures/fig34_temporal_grid.png)

**Figure 1.5 — Two-dimensional temporal decomposition of video generation.** Horizontal
transitions advance physical frame time, vertical transitions refine each predicted state, and
the sliding context window bounds directly addressable history. Source: author’s schematic.

### 1.2.4 Classification of uncertainty types

The performed decomposition of the complex system permits a classification of the kinds of
uncertainty arising in it. In accordance with the theory of systems analysis of complex systems
under risk [8, 9] and system methodology [8, 9], uncertainties are classified by source and nature. In the video generation system
under study, four principal kinds of uncertainty are distinguished: informational,
structural-algorithmic, computational, and target uncertainty.

**Informational uncertainty** is conditioned by the very nature of the iterative generative
process, which begins from a state of pure noise (the state of maximal entropy). Moreover,
informational uncertainty is inevitably present in the control actions. For example, the textual
description "a car drives along a road" or a coarse semantic mask is always incomplete: it
specifies only boundary macro-conditions, leaving a multitude of micro-details (the shape of the
headlights, the texture of the asphalt, the lighting) undescribed. The task of the complex system
is the correct and plausible resolution of this uncertainty during generation. If the system
completes the missing information incorrectly, "hallucinations" and anomalies arise, rendering
the result inconsistent [16].

**Structural-algorithmic uncertainty** is connected with the rigid limitations of the
mathematical apparatus itself (the neural network). Models always have finite capacity and use
architectural inductive biases. A separate and extremely destructive manifestation of this
uncertainty is the problem of spatio-temporal tokenization. Standard transformer architectures
(ViT, ViViT) partition the image into a regular grid of fixed-size patches, ignoring the fact
that the visual signal has non-uniform information density.

![Fixed-grid tokenization and information density](../../figures/fig35_fixed_grid_tokenization.png)

**Figure 1.6 — Fixed-grid tokenization under non-uniform information density.** Equal-size
patches assign equal token cost to static background and to high-information moving contours.
Source: author’s schematic.

As follows from the principle of optimal control, a uniform distribution of resources across
monotonous background and complex moving objects is suboptimal. This gives rise to structural
uncertainty: computational resources are lacking where they are genuinely needed and are wasted
where there are no objects or dynamics. Solving this problem requires the creation of adaptive
tokenization systems based on the measurement of local entropy (developed in Chapter 2).

**Computational uncertainty** is the most critical for iterative multi-step models. It arises on
two fronts, corresponding to the two temporal axes of §1.2.3. On the training front (the
identification of model parameters), the backpropagation-through-time algorithm (BPTT) requires
storing in operating memory all intermediate states for every step of the modeled process; since
the state space of visual data is enormous, the linear growth $O(N \cdot K)$ of memory
requirements quickly exceeds the physical limits of the hardware. The easiest "solution" — cutting
the optimization trajectories down to a few steps — renders the system incapable of modeling
long-term dynamics. On the operation front, the flat attention context over the physical time
axis inflates the key–value memory linearly and the computation quadratically with the number of
context tokens (§1.3.1, eq. (1.7)), so the memory horizon of a flat system is capped by the
hardware precisely in the regimes where a long horizon is required. The task of radically
reducing memory consumption — for gradient computation along deep trajectories and for context
retention along long horizons — is a critically necessary condition for the viability of
multi-step generative systems.

**Target uncertainty** arises when the criterion optimized by the model during training does not
coincide with the global goal during generation. For example, diffusion models optimize the local
error of noise prediction at one isolated step, but during generation these steps are chained
into a long sequence in which errors accumulate. The absence of a single global objective
functional of the form (1.2) makes the control process "blind" to the final result.

The performed multidimensional decomposition of the video generation control system and the
classification of the identified kinds of uncertainty form a clear methodological basis for the
further research. Systems analysis shows incontrovertibly that to ensure high generation
consistency and to solve the anomaly problem it is insufficient to improve individual
architectural elements empirically.

What is required is a comprehensive mathematical approach comprising: methods of adaptive
entropy-based state estimation for resolving spatial uncertainty; resource-saving algorithms
based on reversible computation for overcoming computational uncertainty; and the creation of a
new **hierarchical control apparatus over compact continuous hierarchical latent state spaces**
for eliminating structural and target uncertainty.

The next subsection conducts a detailed critical analysis of existing approaches to the
mathematical modeling of visual data — diffusion, hierarchical latent and autoregressive, and
energy-based models — with the aim of identifying their strengths and their fundamental systemic
limitations in the context of the requirements formulated above.

## 1.3 Analysis of existing approaches

To solve the posed problems of systems analysis and to construct an intelligent information
technology of video data generation, a thorough critical review of the existing mathematical
approaches to modeling complex multimodal spatio-temporal distributions is necessary. The modern
scientific landscape offers several competing paradigms, each with its own architectural
features, advantages, and fundamental systemic limitations. This subsection reviews three classes
of models **in the order dictated by the argument of this dissertation**: (i) diffusion models —
the state of the art in generation quality, whose systemic limitation is the monolithic flat
long-context attention; (ii) hierarchical latent and autoregressive approaches — the remedy
direction, within which the choice between discrete and continuous latent spaces is examined;
(iii) energy-based models — reviewed as literature together with the resource-efficiency methods
they motivated. The focus criteria are systemic consistency and resource control.

### 1.3.1 Diffusion models: generation quality and the flat long-context barrier

Denoising diffusion probabilistic models (DDPM) [4] and their non-Markovian
generalizations (DDIM) [17] are today the dominant standard in high-quality image and
video generation. Unlike traditional generative approaches such as GAN [6], which attempt
to map random noise directly onto the data, diffusion models implement a more effective and
mathematically grounded iterative strategy.

The mathematical basis of the diffusion approach rests on concepts of non-equilibrium
thermodynamics. The modeling process is decomposed into two mutually inverse processes: forward
and reverse diffusion.

The forward diffusion stage is fixed and consists in the gradual addition of Gaussian noise to
the original "clean" data sample $x_0$, progressively driving it to the state of random noise
$x_T$. The process is modeled as a Markov chain:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t \mathbf{I}\right) \tag{1.3}$$

where $\beta_t$ is a predetermined deterministic schedule of noise variance at step $t$. A
mathematical advantage of this construction is the closed-form availability of the noised state
at an arbitrary step directly from the original state:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t)\mathbf{I}\right) \tag{1.4}$$

![Forward diffusion](../../figures/fig36_forward_diffusion.png)

**Figure 1.7 — Forward diffusion.** Progressive Gaussian corruption destroys the information
structure of a sample from $x_0$ toward $x_T$. Source: author’s schematic.

At the reverse diffusion stage the system learns to invert this process, denoising the corrupted
sample and restoring the data. Unlike forward diffusion, the reverse process constitutes a
complex computational problem solved by deep neural networks:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\right) \tag{1.5}$$

In practice the model is optimized to predict the added noise $\epsilon$ via the simplified loss:

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\,\bigl\|\epsilon - \epsilon_\theta\bigl(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\bigr)\bigr\|^2\,\right] \tag{1.6}$$

![Reverse diffusion](../../figures/fig37_reverse_diffusion.png)

**Figure 1.8 — Reverse diffusion.** A learned denoising field iteratively restores structure from
noise. The sequence is conceptual and does not depict an experimental sample. Source: author’s
schematic.

This "forward–reverse" paradigm makes it possible to generate images of extremely high quality:
the decomposition of denoising into steps naturally decomposes the learning problem and
simplifies the model's acquisition of features. Classical pixel-space diffusion models, however,
face colossal memory deficits, which led to latent diffusion models (LDM) [12] operating in
the compressed space of a variational autoencoder. The architectural evolution then displayed a
stable trend away from convolutional U-Nets toward transformers: Diffusion Transformers (DiT)
[18] demonstrated that processing noised images as sequences of patches scales better, and
in the video domain this approach evolved into systems of the Sora class [19], operating on spatio-temporal latent patches and generating video up to a minute long.

![Latent diffusion architecture](../../figures/fig38_latent_diffusion.png)

**Figure 1.9 — Latent-space diffusion.** Spatial compression reduces the per-step denoising cost,
but it does not by itself solve the temporal-memory allocation problem. Source: author’s
schematic.

Despite their dominant generation quality, modern diffusion video systems possess systemic
limitations that are the direct object of this dissertation's analysis. **The paradigm-level
strengths of iterative refinement are not in question here** — on the contrary, Chapter 4's
matched-capacity, matched-NFE mechanism study confirms empirically that an iterative refinement (diffusion)
core commits to sharp, correctly located content where a regression core averages over futures;
the per-frame generation core of the model developed in Chapter 3 is itself a diffusion core,
selected by exactly that controlled analysis. The limitations lie elsewhere — in *how the
temporal context is represented and scaled*.

The first limitation is the **flat, single-level latent space**. The latent space of the
underlying autoencoder is single-level; the system is forced to solve macro-problems (global
composition) and micro-problems (textures) simultaneously within one tensor, which precludes
independent control of abstraction levels and, per §1.2.2, is a direct source of hierarchical
inconsistency.

The second limitation is the **resource blow-up of flat long-context attention**. To lengthen the
memory horizon, a flat video transformer must lengthen its spatio-temporal token context. For a
context of $T_w$ frames tokenized at $N_s$ spatial tokens per frame, the attention context length
is $L = T_w \cdot N_s$, and the principal costs of the attention mechanism scale as

$$C_{\text{compute}} = O(L^2 \cdot D) = O(T_w^2 N_s^2 D), \qquad
M_{\text{KV}} = O(L \cdot D) = O(T_w N_s D) \tag{1.7}$$

where $D$ is the model width: computation grows quadratically and key–value cache memory linearly
with the context length. Doubling the memory horizon of a flat system quadruples its attention
computation and doubles its inference memory — the resource cost is paid *at every generation
step, for every frame*, precisely at the horizons where long memory is most needed. This is the
central resource contradiction of flat long-context modeling.

The third limitation is the **degradation of long-horizon temporal consistency** even where a
long flat context is affordable. Empirical evaluations of long-video generators and world models
show that content which leaves the field of view is not restored consistently upon its return,
and that consistency metrics degrade with the distance to the relevant past [20, 21, 22, 23]. In systems-analysis terms, the flat
context window imposes a hard memory horizon (§1.2.3): information older than the window is not
merely expensive — it is absent from the system state. Related to this is **target uncertainty**:
during training the model optimizes the local error of one isolated step, while during generation
errors accumulate along the whole autoregressive chain — the phenomenon of exposure bias [24, 25].

![Flat long-context barrier](../../figures/fig39_flat_long_context_barrier.png)

**Figure 1.10 — Flat long-context resource and re-entry barrier.** Panel A is schematic rather
than a hardware benchmark: key–value memory grows linearly and attention work quadratically with
sequence length. Panel B shows why a low-rate long-memory stream can remain addressable after the
relevant content has left a short flat window. Source: author’s schematic.
Source: NEW (draw)

### 1.3.2 Hierarchical latent approaches and autoregressive prediction

The remedy direction for the contradictions of §1.3.1 is formed by approaches that (i) treat
video prediction autoregressively along the physical time axis and (ii) decompose the latent
state space instead of flattening it. This subsection reviews the components of that direction:
patch-based autoregressive transformers, discrete-token models, and hierarchical latent models —
and motivates, by the author's own measurements, the choice of *continuous* hierarchical latent
spaces made in this work.

**Patch-based autoregressive transformers.** This line of work originates in the transfer of
natural-language-processing principles to computer vision. The standard approach, popularized by
the Vision Transformer (ViT) [26] and its video adaptations ViViT [27] and Video
Swin Transformer [28], partitions the video into a grid of non-overlapping
spatio-temporal patches that are linearized and fed to a transformer. The principal advantage of
transformer models is their capacity to model long-range dependencies between distant regions of
the video; their principal cost is exactly the quadratic attention complexity (1.7). Although
Video Swin partially mitigates it with local attention windows, the fundamental limitation —
a single-level latent space with fixed patching — remains. Fixed-grid tokenization is a form of
rigid, content-blind discretization: static low-information background and dynamic scene regions
receive equal computational budgets, which §1.2.4 identified as a source of
structural-algorithmic uncertainty. An analogous problem in language modeling was resolved by the
Byte Latent Transformer (BLT) [29], which groups bytes into latent patches of variable size
governed by information density (entropy). Transferring this paradigm to the visual modality
resolves the rigid-tokenization component of structural uncertainty; the V-BLT architecture
developed in this dissertation realizes that transfer and is presented in Chapter 2. *(Preserved
from old §1.3.2.)*

**Discrete-token autoregressive models.** A prominent branch of autoregressive video modeling
operates over *discrete* latent spaces: a vector-quantized autoencoder (VQ-VAE) [30] compresses frames into grids of codebook indices, over which an autoregressive
transformer learns the video distribution — the VideoGPT line [31] and its successors,
including long-video architectures with compressed memory [15]. The attraction of this paradigm is genuine: discrete tokens give a clean autoregressive
likelihood, natural compatibility with language-model tooling, and strong compression.

Discrete bottlenecks also introduce specific optimization risks—straight-through assignment,
codebook underutilization, dead codes and level collapse—that are documented in the VQ
literature [30, 31]. The implemented fourth-article system
therefore uses **compact continuous hierarchical latent state spaces**, in which both levels are
end-to-end differentiable and independently decodable. This is an implementation choice, not an
empirical conclusion that continuous representations are universally superior: the present
repository contains no controlled hierarchical-VQ-versus-continuous comparison and makes no such
claim. Representation sensitivity is instead tested within the continuous-codec family by the
pre-registered alternate-codec-seed study in Chapter 4.

![Representation interfaces and empirical scope](../../figures/fig56_representation_scope.png)

**Figure 1.11 — Representation interfaces and the evidence boundary of this dissertation.**
Discrete VQ models are reviewed as an important autoregressive family; the controlled experiments
use a continuous two-level codec. The dashed path denotes a comparison not performed here.
Source: author’s schematic.

**Hierarchical latent video models.** The direct remedy for the flat-latent limitation is the
class of models whose latent state space is explicitly multi-level, with distinct temporal or
spatial rates per level. Early representatives use multi-rate recurrent latents — clockwork
variational hierarchies [13] — where upper levels tick more slowly and thereby
retain longer temporal context. The modern form of this idea in the diffusion family is the
hierarchical latent video diffusion model of the MilliVid class [32]: a compact hierarchy of continuous latents (a low-rate *coarse* level and
a high-rate *fine* level, each independently decodable to pixels), a diffusion core that denoises
the levels jointly with cross-level attention, and autoregressive rollout along physical time.
The systemic significance of this spatial construction follows directly from eq. (1.7): because the
coarse level carries far fewer tokens per frame than the fine level, a *heterogeneous* context —
long in coarse tokens, short in fine tokens — covers a memory horizon many times longer than a
flat context of the same total token budget. The hierarchy converts the quadratic cost of horizon
extension into a near-linear one along the cheap level, making the coarse level one economical
carrier of long-range temporal memory. Temporal subsampling provides a distinct route to the same
span and is compared directly in Chapter 4. *(Chapter 3 develops the spatial form with a two-level
continuous latent interface and unidirectional context transfer; Chapter 4 treats it as one point
in the broader context-packaging design space.)*

![Context-package comparison](../../figures/fig40_context_package_comparison.png)

**Figure 1.12 — Context packages allocate equal capacity differently across reach and
fidelity.** Flat contiguous, spatially hierarchical and temporally strided packages feed the same
predictor family; the exact experimental token counts and target interfaces are reported in
Chapters 3–4. Source: author’s schematic.
Formally, the hierarchical factorization of the video distribution replaces the flat conditional
$p_\theta(x_t \mid x_{t-w:t-1})$ with a level-decomposed conditional of the form

$$p_\theta\!\left(z^{c}_{t}, z^{f}_{t} \,\middle|\, z^{c}_{t-w_c:t-1},\; z^{f}_{t-w_f:t-1},\; u_t\right), \qquad w_c \gg w_f \tag{1.8}$$

where $z^{c}$ and $z^{f}$ are the coarse- and fine-level latent states, $w_c$ and $w_f$ their
context windows, and $u_t$ the control signals; the asymmetry $w_c \gg w_f$ at a matched token
budget $w_c N_s^{c} + w_f N_s^{f} \approx \mathrm{const}$ is what extends the memory horizon
without the flat blow-up (1.7).

**Methodology for comparing hierarchical configurations.** The canonical comparison methodology
of this class is the Clockwork-VAE benchmark
[13]: one configuration table over several datasets and metrics, quality-retention
curves over long open-loop rollouts, and level-content probes (an upper level following its
prior while lower levels see the context). The dissertation adopts this *format* in §4.4 while
tightening three controls the original lacks — a systems-analysis point in its own right. First,
capacity: the compared CW-VAE-class models differ in parameter count (their RSSM baseline 5M vs
CW-VAE 12M; only the NoTmpAbs ablation is matched), whereas in the token-stream architecture of
Chapter 3 depth adds tokens to a shared backbone with only per-level I/O heads (+1.4%
parameters, disclosed) — the depth axis is backbone-capacity neutral. Second, context cost: a recurrent state has no token-budget axis, so
hierarchy-versus-flat comparisons in the recurrent family cannot hold context cost fixed; the
token interface makes every configuration's cost an exact count, enabling matched-budget
comparison. Third, metrics: the CW-VAE evaluation aggregates SSIM/PSNR over full frames, and the
author's own measurement (§3.4.3) shows a constant all-black frame outperforming a trained
CW-VAE on such aggregates on sparse-foreground video — hence the foreground-masked,
re-entry-stratified protocol of §3.4. It is also noted that the CW-VAE paper's depth evidence is
an information-allocation (KL-per-level) analysis rather than a prediction-quality dose–response;
the quality-per-depth cells are measured directly in §4.4.5. The temporal-abstraction relatives —
VTA [33] (whose lower level resets when the upper level ticks) and the clockwork recurrent
lineage [34] — complete the review of hierarchy *shapes*:
spatial and temporal abstraction, depth and budget allocation are four axes of the design space.
Which form of hierarchy pays and where is therefore a measurement question (§4.4), not a
postulate.

Summarizing the hierarchical-latent review: this paradigm inherits the generation quality of the
diffusion core (§1.3.1) while remedying its two systemic limitations — the flat latent space and
the flat context window — and, per the author's measured finding, its *continuous* form is the
one trainable at practically reachable scale. It is therefore adopted as the base paradigm of
this dissertation.

### 1.3.3 Energy-based models and resource efficiency

The third class of the review is that of energy-based models (EBM). The EBM paradigm originates
in statistical physics [35, 36]. Its basic principle is that the probability of a data configuration $x$ is defined by
the Gibbs/Boltzmann distribution:

$$p_\theta(x) = \frac{\exp\left(-E_\theta(x)\right)}{Z(\theta)} \tag{1.9}$$

where $E_\theta(x)$ is a scalar energy function parameterized by a neural network and $Z(\theta)$
is the partition function. The lower the energy, the higher the probability of the state and the
more consistent it is deemed.

An important theoretical bridge connects EBM to diffusion models through score matching [37, 38]: the score function approximated by a diffusion model is the negative gradient of
an (implicit) energy,

$$\nabla_x \log p(x) = -\nabla_x E(x) \tag{1.10}$$

From the standpoint of systems analysis the attraction of this paradigm is evident: the energy is
a single explicit optimization criterion; the system always moves toward equilibrium; and the
energy value serves as a universal indicator of anomalies in a generated frame — if an artifact
is generated, the local energy grows. Transformer-parameterized realizations of this idea —
Energy-Based Transformers (EBT) [39] — extend it to autoregressive
video modeling and add a systemically appealing property: the number of descent steps (1.11) is a
natural "think-longer" control knob, allowing the system to spend more computation on harder
predictions.

However, the broad application of EBM is critically limited by computational uncertainty. To
train such a model, backpropagation must be threaded through $K$ MCMC iterations of the forward
pass; BPTT memory grows as $O(K \cdot L)$, where $L$ is the network depth, and since every MCMC
iteration requires a gradient with respect to the input (double backward), memory overflow occurs
rapidly [40, 41], precluding deep optimization trajectories for
high-quality video data.

Approaches to this barrier constitute the resource-efficiency toolbox of multi-step systems.
Gradient checkpointing [42, 43] trades computation for memory by storing
only a subset of activations and recomputing the rest, reducing memory to the order of the square
root of the trajectory length. Neural ODEs [44] offer constant memory through
adjoint integration of differential equations but suffer from numerical instability. The most
promising direction is discrete reversible neural networks — RevNet [45] and its
transformer form RevViT [46] — which restore activations mathematically instead of
caching them. It is this apparatus of reversible computation, applied to removing the resource
constraints of long-horizon *temporal* modeling, that forms the core of the TempVerseFormer
method presented in Chapter 2 and underlies the constant-memory autoregressive rollout of the
hierarchical model of Chapter 3.

An honest forward reference completes this review. In the present research the energy-based
mechanism was not adopted a priori but *examined experimentally* under matched conditions: the
matched-capacity, matched-NFE mechanism study of Chapter 4 (§4.2) includes a two-sided
capability study of Energy-Based Transformers, run with the official published implementation and
both published training recipes at the largest scale reachable on the available hardware. Its
outcome is a **characterized negative** at that scale: one published recipe converges to a stably
*mis-ordered* energy landscape (the mean of the data attains lower energy than the true target);
the other recipe repairs the ordering — the true target becomes the energy minimum among probed
candidates — yet gradient descent from the prescribed initialization still does not reach
structured outputs, and inference-time computation does not buy the capability back; meanwhile a
diffusion core of roughly half the parameter count, on the same latents and protocol, commits to
sharp and correctly located predictions. The study is scoped explicitly: it tests a capability
the EBT publication does not claim at a scale it did not run, and contradicts none of the
published claims. For this dissertation its systemic significance is that the choice of the
per-frame generation core in Chapter 3 is made *by controlled measurement, not by paradigm
preference* — and the measurement selects diffusion.

Summarizing the critical analysis of existing approaches: diffusion models deliver the highest
generation quality, but their monolithic flat realization runs into the long-context resource
blow-up (1.7) and the memory-horizon consistency degradation; patch-based transformer models
capture global context well but require entropy-based adaptation of their rigid tokenization;
hierarchical latent models remedy the flat-latent and flat-context limitations, and in their
continuous form are trainable at reachable scale (whereas the author's experiments found
hierarchical VQ configurations unstable); energy-based models offer an attractive explicit
criterion but are resource-barred in training, and their descent mechanism is experimentally
characterized as a negative at this work's reachable scale (Chapter 4). The synthesis defining
this dissertation is therefore: **a hierarchical continuous-latent video model with a diffusion
refinement core, whose coarse level carries the long memory horizon (Chapter 3), supported by
entropy-adaptive decomposition (Chapter 2), reversible-computation resource optimization
(Chapter 2), and a criterion-driven measurement instrument (Chapters 3–4)**.

## 1.4 Consistency criteria and methods

In the theory of complex systems one of the fundamental notions is the stability of a system
[47] — its capacity to maintain a specified state or target
trajectory under external disturbances and internal stochasticity. In the context of generative
video models the notion of stability is most closely tied to the term "consistency" (coherence,
non-contradiction). Traditionally, the evaluation of generation results in machine learning
reduces to the subjective criterion of visual attractiveness or to statistical metrics of
proximity to the training sample, such as FID [48]. From the standpoint of systems
analysis such an approach is severely limited.

Systems analysis requires the transition from abstract aesthetic evaluation to a formalized
criteria apparatus. The generative model is treated as a system transforming the input
information signal of the condition $u$ into the output state $x$. If the state $x$ contains
internal logical, spatial, or physical contradictions, it is classified as anomalous.
Accordingly, **consistency is the criterion of the absence of anomalies in the generated state of
the complex system**; ensuring consistency becomes synonymous with ensuring the reliability and
stability of the information technology of generation.

Since video data have a deep multidimensional structure, consistency cannot be described by one
scalar parameter. It is a complex systemic phenomenon requiring an extended taxonomy. This
subsection analyzes the classical kinds of consistency — spatial, object, and temporal —
integrates the notion of conceptual consistency, and formalizes hierarchical and trajectory
consistency, which are critically important for hierarchical multi-step generative systems.
### 1.4.1 Spatial and object consistency

The basic level of coherence for any visual signal is **spatial consistency**. This criterion
determines the structural integrity of the generated frame at the level of geometry and the
mutual arrangement of elements. Spatial consistency guarantees that objects in the image have
correct proportions, that boundaries are sharp, and that perspective and lighting obey uniform
physical laws across the whole scene.

The next level is **object consistency**. In many practical tasks the system must generate a
series of states of one and the same object (from different viewpoints, under different
lighting), preserving its unique identifying features. Modern generative algorithms approach this
task in several ways.

The simplest method of achieving object consistency in diffusion models is generation by the
image-to-image approach [49]: an existing reference image is used as
the starting point to steer the generation toward a specified result. The diffusion model
transfers the information of the image into its latent space, subtly influencing the direction of
the process.

![Image-to-image conditioning](../../figures/fig41_image_to_image.png)

**Figure 1.13 — Image-to-image conditioning.** Increasing noising strength expands variation but
weakens direct preservation of the reference identity. Source: author’s schematic.

This approach, however, has rigid systemic limitations — the difficulty of precise control: a
higher degree of noising of the reference increases variability and loses the consistency of fine
details (textures, scratches).

A deeper method is fine-tuning of the model itself by approaches of the DreamBooth family [50] and its low-rank variation LoRA [51]. This method uses a set of reference
images of a specific object to adjust the internal representation (the weight coefficients
$\theta$) of the model. Although DreamBooth can secure the consistency of specific features (for
example, facial features or the texture of a particular object), it is computationally heavy and
works exclusively within the "learned" object.

For enforcing strict spatial consistency of shape there exists the ControlNet architecture [52], which injects a special control signal (depth maps, contours, segmentation masks)
into the reverse diffusion process. Methods of the ControlNet type perform the function of hard
boundary control of the admissible state space: the input image acts as a guiding form
constraining the evolution of the system.

![ControlNet conditioning](../../figures/fig42_controlnet.png)

**Figure 1.14 — Structural control with a ControlNet-class branch.** A trainable branch maps
edges, depth or segmentation into residual constraints for a frozen diffusion backbone. Source:
author’s schematic after the ControlNet architecture [52].

Despite the effectiveness of ControlNet, it is not a complete solution for object consistency,
since it focuses only on the form ("how") and not on the properties ("what"). A fully autonomous
system must be able to maintain object identity through internal control mechanisms, without
mandatory external structural prompts. *(This requirement — identity maintenance by internal
memory rather than external prompts — is exactly the long-memory requirement that the coarse
level of the Chapter 3 model serves.)*

### 1.4.2 Temporal consistency

Upon the transition from static images to video data the key criterion becomes **temporal
consistency**. It requires that the sequence of generated states $x_1, x_2, \dots, x_T$ form a
logical and smooth video stream. Anomalies of temporal consistency manifest as flicker, sudden
background color changes, or deformation of objects during motion.

The mathematically strict local requirement of temporal consistency is formalized through
optical flow or a warping function $W$: if the vector field of motion between frames $t-1$ and
$t$ is known, the generated state $x_t$ must maximally correspond to the warped previous state
$W(x_{t-1})$. The local temporal-consistency error is defined as:

$$L_{\text{temp}} = \left\|\left(x_t - W(x_{t-1})\right)\right\|^2 \odot M \tag{1.12}$$

where $M$ is a binary mask excluding occlusion regions.

It must be emphasized, however, that the warp criterion (1.12) is **local**: it couples only
adjacent frames and therefore certifies only short-range temporal consistency. It is entirely
blind to the long-range regime identified in §1.3.1: content that leaves the field of view exits
the domain of the mask $M$, and its consistent restoration upon re-entry many frames later is not
constrained by any chain of local warp terms — it is constrained only by the memory of the
generating system. Long-range temporal consistency is therefore operationalized separately, as
part of trajectory consistency (§1.4.5), through metrics stratified by re-entry hardness. For
ensuring temporal consistency in the developed information technology, the following mechanisms
are used: multidimensional positional encoding 3D-RoPE, block-causal attention masks, and
temporal chaining mechanisms (Chapter 2), together with the coarse-level long memory of the
hierarchical model (Chapter 3).

### 1.4.3 Style and conceptual consistency

Besides object consistency, complex systems must maintain a unified **stylistic** appearance of a
series of generated data. Modern approaches such as StyleAligned [53] solve this
problem through intervention in the attention mechanism (shared attention): target images receive
information about the reference image during generation, copying its global stylistic
characteristics (color palettes, brushwork) without interfering with spatial geometry.

![Shared-attention style conditioning](../../figures/fig43_shared_attention.png)

**Figure 1.15 — Style consistency through shared attention.** Reference attention features are
shared across distinct target trajectories while their content conditions remain separate.
Source: author’s schematic after StyleAligned [53].

For a more thorough and qualitative analysis of generative models, within this dissertation
research — published in **[A3]** (the consistency-criteria article) — the notion of **conceptual
consistency** was formalized.

Marr's multi-stage account of visual representation — progressing from image evidence toward
structured descriptions of the scene [54] — motivates a broad engineering analogy, but does not itself prescribe
the taxonomy used here. The dissertation's narrower claim is that the traditional division into
style and object alone is incomplete. Conceptual consistency is the capacity of the generative system for
compositionality and extrapolation. It evaluates the ability of the model not merely to reproduce
existing objects but to generate entirely new visual ideas (out-of-distribution), never
encountered during training, while strictly observing the logic of the control conditions $u$.

For example, generating the image "a skyscraper built of skyscrapers" requires the system to
perfectly disentangle the vectors of "shape" and "material". An anomaly of conceptual consistency
is the loss of compositionality — when the system ignores part of the textual condition or
generates objects side by side instead of combining them.

As the metric for evaluating conceptual consistency, the cosine similarity between the semantic
vectors of the generated image and the textual condition in the joint space of a contrastive
model (CLIP score) [55] is proposed. Let $E_I$ be the visual encoder and $E_T$ the textual
encoder; then the measure of conceptual consistency is defined as:

$$C_{\text{concept}}(x, u) = \frac{E_I(x) \cdot E_T(u)}{\left\|E_I(x)\right\| \left\|E_T(u)\right\|} \tag{1.13}$$

High values of this indicator testify that the system has successfully resolved the target
uncertainty and learned the deep causal relations of the visual space.

### 1.4.4 Hierarchical consistency

For the transition to systems of a higher order of complexity — multi-level generative models —
this research introduces the notion of **hierarchical consistency**, developed within the
construction of the hierarchical generation model of Chapter 3. This criterion follows directly
from the principles of systems analysis of multi-level objects.

Hierarchical consistency is defined as the strict informational and structural coherence between
the representations of one and the same visual object at different levels of abstraction: the
macro level (semantics), the meso level (structure), and the micro level (pixels or textures). An
anomaly of hierarchical consistency is a situation of internal system conflict: for example, the
macro signal forms the contour of a brick building while at the micro level the system generates
organic wood patterns.

Ensuring this property requires the introduction of an inter-level coherence operator. From the
standpoint of system control, the essential requirement is that the junior (fine/detailed) level
must not be able to contradict the senior (coarse/abstract) level's decisions at long range. In
the Chapter 3 model this requirement is satisfied by two complementary mechanisms (§3.1.2–3.1.3).
First, by **information-design asymmetry**: within a prediction step both levels are denoised
jointly in one shared attention pass, but only the coarse level's history extends over the full
memory horizon — long-range structure can therefore *originate only from the coarse channel*,
and the fine level physically lacks the inputs to contradict it at long range; the control is
unidirectional in the information-theoretic sense (coarse → fine for long-horizon signals), not
in the scheduling sense. Second, by **gradient detachment at the system boundaries** where it is
exact and load-bearing: between the frozen representation subsystem and the prediction core (the
state coordinate system is a constant of the control problem), and between past and future along
the autoregressive rollout (previously generated context enters as a constant, excluding
destructive gradient interference between successive windows).

### 1.4.5 Trajectory consistency

The last — and, for multi-step generative systems, fundamental — criterion is **trajectory
consistency**: the stability of the generation process itself as a trajectory of the complex
system. In the original, single-frame reading it concerned the stability of the iterative
refinement process in pseudo-time $k$; in the present research it is generalized and
operationalized on the physical time axis, where the dominant practical instability of video
generation lives: the degradation of **autoregressive rollout**, in which the system consumes its
own outputs as context and errors compound along $t$ (the realization of the target uncertainty
of §1.2.4 and the exposure bias of §1.3.1).

A trajectory-consistent system must satisfy two requirements. First, **bounded degradation**: the
prediction error along an autoregressive rollout must grow in a controlled manner, without
divergence or collapse of the generated content. Second — the requirement that separates a
long-memory system from a flat one — **consistent re-entry**: content that left the field of view
must be restored correctly when it returns, at any temporal distance within the modeled process.

The operationalization of this criterion, developed in Chapters 3–4, stratifies rollout error by
an exactly computable hardness label. For a generated frame $x_t$ with context window $C$, let
the **novel-foreground fraction** $\nu_t(C) \in [0, 1]$ be the fraction of the frame's foreground
content that was informationally inaccessible from $C$ (in the purpose-built testbed of Chapter 3
this quantity is computed *exactly* from the recorded control-action track, rather than
estimated). Trajectory consistency is then measured as the family of stratified degradation
curves

$$\mathcal{E}_{b}(h) = \mathbb{E}\!\left[\, e\!\left(x_{t+h},\, \hat{x}_{t+h}\right) \,\middle|\, \nu_{t+h} \in b \,\right] \tag{1.14}$$

— the expected prediction error $e$ at rollout horizon $h$ within each hardness bin $b$ (easy:
$\nu = 0$; medium: small $\nu$; hard re-entry: substantial $\nu$) — rather than as a single
aggregate number. This stratification is what makes the criterion *diagnostic*: a flat-context
system and a long-memory hierarchical system can be indistinguishable on aggregate error while
differing qualitatively on the hard re-entry bin beyond the flat window's horizon; conversely, an
anomaly of trajectory consistency (divergence, mean-collapse, identity loss at re-entry)
localizes to specific bins and horizons.

For energy-based mechanisms the trajectory criterion has a specialized classical diagnostic in
pseudo-time: the divergence of the sampling chain (1.11) is monitored by the **energy gap**
between the initial and final states of the descent,

$$\Delta E = \frac{E(x_0) - E(x_K)}{E(x_0)} \tag{1.15}$$

which must be strictly positive, and by the ordering of the trained landscape (whether the true
target attains lower energy than degenerate candidates). Finally, the practical attainability of long
consistent trajectories depends on resolving the computational uncertainty of §1.2.4: removing
the memory limits on trajectory depth by the time-agnostic backpropagation developed in **[A2]**
and Chapter 2.

![Trajectory consistency operationalization](../../figures/fig44_trajectory_consistency.png)

**Figure 1.16 — Conceptual operationalization of trajectory consistency.** The schematic predicts
separation specifically on memory-relevant re-entry records beyond a short window; Chapter 4
tests this prediction with exact generator-derived observability labels and paired rollouts.
Source: author’s schematic; curves are illustrative, not measured.

Systems analysis shows that none of the kinds of consistency can be considered in isolation:
they form an interdependent hierarchy of criteria. For example, without trajectory consistency it
is impossible to attain high spatial and temporal consistency of a long generated sequence; and
without hierarchical consistency, the macro-level memory that trajectory consistency relies on at
re-entry cannot be trusted by the lower levels.

Classical approaches — adding a ControlNet or a LoRA — attempt to solve these problems locally,
by patching individual criteria with external mechanisms. The systems approach instead requires
an architecture in which the observance of the consistency criteria follows from the structure of
the process itself: **an explicit multi-stream decomposition of context across temporal rate and
spatial resolution with unidirectional context transfer** — so that hierarchical consistency is enforced by
construction (stop-gradient control, §1.4.4), long-range temporal and trajectory consistency are
supplied with the memory they informationally require (§§1.3.1–1.3.2), and the whole is trainable
within realistic resource constraints thanks to reversible computation (Chapter 2). Constructing
and validating exactly such an architecture is the subject of Chapters 3 and 4.

## 1.5 Conclusions and statement of the research tasks

In the first chapter of the dissertation a thorough systems analysis of the processes of
hierarchical video data generation has been carried out. On the basis of complex-systems theory,
generative video models have been characterized as open dynamic multi-level control systems
operating under substantial structural, informational, computational, and target uncertainty.

The decomposition of the generation system revealed the key problems of contemporary approaches.
The analysis of diffusion models showed that, while they define the state of the art in
generation quality, their monolithic realization suffers from two systemic defects: a
single-level (flat) latent space that mixes abstraction levels and loses hierarchical control,
and a flat spatio-temporal attention context whose computation grows quadratically and whose
memory grows linearly with context length — so that extending the memory horizon of a flat system
is resource-prohibitive precisely where it is most needed, and the temporal consistency of long
sequences degrades: content leaving the field of view is not restored consistently upon
re-entry. In addition, the divergence between the local training objective and the long
autoregressive generation trajectory (exposure bias) undermines long-horizon stability.

The review of patch-based transformer architectures demonstrated their advantages in capturing
global contextual dependencies but revealed the critical problem of irrational computational
resource allocation due to fixed spatio-temporal tokenization, which substantiates the necessity
of methods for adaptive decomposition of visual data based on measuring their local entropy
(developed in Chapter 2).

The study of hierarchical latent approaches established the remedy direction: an explicitly
multi-level latent state space in which a compact upper (coarse) level, carrying few tokens per
frame, can retain a long temporal context at a fixed token budget and thereby serve as the
carrier of the long-range memory of the process. Discrete-token hierarchies remain an important
reviewed alternative with codebook-specific optimization risks. The implemented and evaluated
scope of this dissertation is a **compact continuous hierarchical latent state space**; because
no controlled VQ-versus-continuous receipts are present, representation-family superiority is
not claimed.

The examination of energy-based models (EBM) recorded their systemic attraction — an explicit
scalar optimization criterion doubling as an anomaly indicator — alongside their fundamental
computational barrier: gradient identification through long MCMC trajectories with linear memory
growth. In this dissertation the energy-based mechanism is treated as reviewed literature whose
capability boundary at reachable scale is characterized experimentally in Chapter 4; the survey
of resource-efficiency methods it motivated (checkpointing, Neural ODE, reversible networks)
identified reversible transformations as the most promising apparatus, which Chapter 2 develops
into the time-agnostic backpropagation method and its extension to stochastic multi-level
recurrences.

Furthermore, the chapter extended and improved the criteria apparatus for evaluating the results
of generative models. The notion of consistency was reconceived as the systemic property of the
absence of anomalies and the stability of the generation trajectory. To the classical criteria
(spatial, object, temporal consistency) new kinds were added: conceptual, hierarchical, and
trajectory consistency — the last operationalized through autoregressive-rollout degradation
metrics stratified by an exactly computable re-entry hardness label, which converts a previously
declarative criterion into a measurable instrument of systems analysis.

The identified systemic contradiction — between the necessity of ensuring multi-level consistency
of video data over long time horizons (which requires long memory horizons and deep hierarchies)
and the strict resource constraints of modern computing systems (the attention-memory blow-up of
flat contexts and the linear growth of training memory with horizon length) — determines the
relevance of this dissertation research and forms its purpose.

**The purpose of the dissertation research** is to improve the efficiency and the temporal and
spatial consistency of hierarchical video data generation processes through the development of
systems analysis methods, mathematical models, and intelligent information technologies.

To achieve this purpose and to resolve the identified systemic contradictions, the following
research tasks must be solved:

1. Analyze the state and problems of generative video-data models as complex systems; classify
   the types of uncertainty and identify the systemic causes of anomalies (inconsistencies) in
   the generation process, including the resource and consistency limitations of flat
   long-context modeling *(performed in Chapter 1)*.

2. Improve the criteria apparatus of systems analysis of generative processes by extending the
   taxonomy of consistency types (conceptual, hierarchical, trajectory) and operationalizing
   trajectory consistency through measurable rollout metrics stratified by re-entry hardness
   *(the taxonomy is constructed in Chapter 1; the operational metrics and their measurement
   instrument are developed in Chapters 3–4)*.

3. Develop a method for estimating the state of uncertainty of spatio-temporal data through their
   adaptive decomposition using entropy metrics and the mathematical apparatus of
   multidimensional rotary positional embedding.

4. Develop a method for optimizing the computational resources of multi-step iterative models
   that ensures constant, process-length-independent consumption of memory on the basis of
   reversible mathematical transformations; extend it to stochastic multi-level recurrences and
   delimit its domain of applicability.

5. Develop a mathematical model of multi-stream video data generation in compact hierarchical
   latent state spaces, supporting context streams at different temporal rates and spatial
   resolutions over a shared attention backbone, with unidirectional context-signal transfer
   and a per-frame iterative-refinement generation core selected by controlled mechanism analysis.

6. Create an intelligent information technology and a software complex for the systems analysis
   of hierarchical video generation processes, integrating the developed methods of adaptive
   tokenization, resource optimization, and hierarchical generation, together with a
   purpose-built memory-horizon testbed and re-entry-stratified evaluation protocol as its
   measurement instrument.

7. Conduct comprehensive experimental studies and approbation of the developed methods and
   technology on synthetic and open datasets — including a matched-capacity, matched-NFE
   comparison of prediction mechanisms — and compare their efficiency and consistency with
   modern analogues.

The solution of tasks 3–7 is the subject of the following chapters of the dissertation. Chapter 2
presents the mathematical apparatus of adaptive data tokenization and computational-resource
optimization; Chapter 3 develops the hierarchical mathematical model with long-range coarse-level
memory and the memory-horizon measurement instrument; Chapter 4 presents the experimental
research and practical application.

The results of this chapter are published in the author's work [A3] and in the conference
publications to be identified in the final author-supplied publication list.

---

<div style="page-break-after: always;"></div>

# Chapter 2 — Mathematical Methods for Uncertainty Estimation and Resource-Efficient Modelling of Multi-Step Video Processes

## 2.1 Entropy-Based Estimation of Spatio-Temporal Video Uncertainty

As established in the previous chapter during the decomposition of the complex system, the
processes of hierarchical video data generation operate under a high level of multidimensional
uncertainty. To build an effective control system for such processes, it is necessary to develop
a mathematical apparatus for resolving the informational, structural-algorithmic, and
computational uncertainty at every stage of information processing. One of the fundamental
problems that generates a significant share of the structural-algorithmic and computational
uncertainty is the problem of inefficient decomposition of the input spatio-temporal data.

In classical systems analysis, the decomposition of a complex system must reflect its internal
information saturation and structure. However, the traditional approaches to the mathematical
modeling of video data, popularized by architectures of the Vision Transformer (ViT) type
[26] and their spatio-temporal extensions ViViT
[27] and Video Swin Transformer [28],
are based on a rigid, a-priori-fixed partition of the image or video stream into a regular grid
of non-overlapping patches of fixed size, e.g., 16×16 pixels.

From the standpoint of optimal control and resource-allocation theory, such an approach is
severely suboptimal. The visual signal, especially video data, is characterized by an extremely
non-uniform distribution of information density. Large areas of the frame are often occupied by a
static, low-information background (monotonous sky, walls), while the key dynamics and
high-frequency details (moving objects, complex kinematics) are concentrated in small local
zones. Using a fixed grid forces the system to distribute its limited computational resource
uniformly across all regions of the frame, wasting it on the processing of redundant information.

To solve this problem, it is necessary to move from a rigid geometric decomposition to an
adaptive decomposition of the state space that is driven by the data themselves. The conceptual
foundation for this transition is the new paradigm of byte-level language models that emerged in
natural language processing, in particular the Byte Latent Transformer (BLT)
[29]. These systems abandoned the paradigm of fixed token vocabularies
in favor of dynamic grouping of elementary information units (bytes) into latent patches whose
size is inversely proportional to the local complexity of the signal.

![Entropy-driven byte patching](../../figures/fig45_blt_variable_patching.png)

**Figure 2.1 — Conceptual operating principle of entropy-driven byte patching.** Long
low-entropy regions are grouped into larger latent patches, whereas locally surprising regions
receive finer allocation. The values are schematic, not empirical measurements. Source:
author’s schematic after BLT [29].

Within the dissertation research, the results of which are published in **[A1]**, the hypothesis
was put forward that transferring the principles of adaptive entropy-based tokenization to the
visual modality opens fundamental opportunities for optimizing and redistributing the
computational resources of a generative system. On the basis of this approach, the author
developed the method for estimating spatio-temporal uncertainty and a new mathematical
architecture — the Vision Byte Latent Transformer (V-BLT).

Working at the level of "raw" bytes of video data, where each pixel of an RGB channel is
represented by a discrete integer from 0 to 255, allows the system to avoid the loss of
information that is unavoidable when heuristic pre-compression or vector-quantization (VQ-VAE)
algorithms are used [30]. The developed V-BLT architecture is
a three-tier hierarchical information-transformation system consisting of a Local Encoder, a
Global Latent Transformer, and a Local Decoder. In the arc of this dissertation, V-BLT is the
author's **first realization of the hierarchical adaptation of video generation** — a two-stage
decomposition in which local byte-level stages surround a global latent stage and an entropy
criterion governs the allocation of capacity between them; the mature form of this hierarchical
decomposition, in which the upper level carries the long-range temporal memory of the process,
is developed in Chapter 3.

![Vision Byte Latent Transformer architecture](../../figures/fig46_vblt_architecture.png)

**Figure 2.2 — Vision Byte Latent Transformer architecture.** Local entropy-guided aggregation
and byte classification surround the global time-causal latent transformer. Source: author’s
schematic of the [A1] method.

The first subsystem — the **Local Encoder** — performs the function of primary adaptive
aggregation. The input is a video sequence as a multidimensional tensor of raw bytes. The task of
the Local Encoder is the dynamic grouping of locally correlated bytes into compact latent
representations in the form of spatio-temporal patches $Z = \{z_1, z_2, \dots, z_M\}$.
Mathematically it is realized as a lightweight convolutional-transformer network capable of
determining the optimal boundaries of information aggregation.

The second subsystem — the **Global Latent Transformer** — is the computational core and is
responsible for modeling the long-term dynamics in the compressed space. Since this block
performs autoregressive prediction for time-series modeling, its architecture requires a specific
design of the attention-masking system in order to avoid structural uncertainty in the form of
information leakage from the future [56], while
fully resolving the spatial uncertainty within the current frame.

For this purpose, a spatio-temporal attention-masking matrix is used. Let the latent space have
dimensionality $T' \times S'$, where $T'$ is the number of time steps and $S'$ is the number of
spatial patches per frame. The attention matrix $M$ is computed so that for any token $z_i$ from
frame $t_i$, the attention weights toward a token $z_j$ from frame $t_j$ obey the rule:

$$
M_{i,j} =
\begin{cases}
0, & \text{if } t_j \le t_i \\
-\infty, & \text{if } t_j > t_i
\end{cases}
\tag{2.1}
$$

This strictly time-causal but spatially fully-connected mask guarantees the physical and logical
plausibility of the generation process.

The third subsystem — the **Local Decoder** — projects the predicted latent macro-states back
into the micro-space of image bytes. The key mathematical innovation of the developed method,
which distinguishes it from standard pixel-regression approaches of the MSE type, is the use of a
classification model for byte generation. At the output, the decoder forms a tensor that yields,
for each pixel, not a scalar brightness value but a full probability distribution over the 256
possible byte states.

Formally, for each byte (sub-pixel) $x_i$ the system computes a logit vector
$L_i \in \mathbb{R}^{256}$. The probability estimate that the byte takes a specific discrete
value $k \in \{0, \dots, 255\}$ is determined by the softmax activation function:

$$
p(x_i = k) = \frac{\exp L_{i,k}}{\sum_{j=0}^{255} \exp L_{i,j}}
\tag{2.2}
$$

Such a mathematical formulation transforms the problem of minimizing the mean squared error
(MSE), which by its nature is prone to smoothing and to the loss of high-frequency details, into
a classification problem using the cross-entropy loss function [57]. This makes it possible to correctly model multimodal distributions:
for example, the contour of a moving object may shift one pixel to the left or to the right with
equal probability, which cannot be described by a single mean value.

The main advantage of the classification approach for systems analysis lies in the possibility of
direct mathematical computation of a measure of spatio-temporal uncertainty. Having a discrete
probability distribution for each pixel, the classical apparatus of Shannon information theory is
applied. The information-entropy measure $H_i$ for each predicted pixel $i$ is computed as:

$$
H_i = -\sum_{k=0}^{255} p(x_i = k)\,\log_2 p(x_i = k)
\tag{2.3}
$$

The entropy value lies in the range from 0 (absolute determinism and confidence of the system in
its prediction) to 8 bits (maximum informational uncertainty, a uniform distribution). By
computing this value for all pixels of a frame, the generative system forms a two-dimensional
per-pixel entropy map.

Entropy maps are a direct indicator of the state of structural uncertainty and serve as a
powerful diagnostic instrument. Empirical studies, the results of which are presented in **[A1]**
and are considered in detail in Chapter 4, confirm that the highest entropy values systematically
concentrate on the contours of fast-moving objects and in occlusion zones with complex
kinematics, whereas the static, predictable background exhibits entropy rapidly approaching zero.
Figures 2.3 and 2.4 show prediction examples for a simple 2D and a complex 3D sequence
respectively, where the first row contains the expected frames of the sequence, the second the
frames predicted by the V-BLT model, and the third the entropy maps corresponding to the
predicted frames (black indicates high entropy, white low entropy). A key aspect of the V-BLT
architecture is its ability to operate on raw bytes, which makes it possible to compute per-pixel
entropy for every prediction. Entropy in this context is a measure of the model's uncertainty:
the higher the entropy, the less "confident" the model is in its prediction for the given pixel.

![V-BLT prediction and entropy maps for a rotating 2D shape](../../figures/fig59_vblt_2d_entropy_published.png)

**Figure 2.3 — V-BLT prediction and entropy maps for a rotating 2D shape.** Rows show,
respectively, ground truth, V-BLT predictions and per-pixel entropy maps over five future frames.
The first prediction has the largest shape error and the broadest high-entropy region; subsequent
predictions recover the rotating contour while entropy concentrates near its boundary. Source:
Kulyk and Mokin, *Scaling Video Prediction with Spatio-Temporal Patches*, published Fig. 7,
DOI 10.31649/1997-9266-2025-182-5-129-139.

![V-BLT prediction and entropy maps for a rotating 3D polycube](../../figures/fig60_vblt_3d_entropy_published.png)

**Figure 2.4 — V-BLT prediction and entropy maps for a rotating 3D polycube.** Rows show ground
truth, V-BLT predictions and per-pixel entropy maps. The first low-context forecast is visibly
degraded; later frames preserve the coarse polycube structure, while uncertainty remains strongest
along moving and newly exposed boundaries. Source: Kulyk and Mokin, *Scaling Video Prediction with
Spatio-Temporal Patches*, published Fig. 8,
DOI 10.31649/1997-9266-2025-182-5-129-139.

From the standpoint of complex-system control, the entropy map is an ideal feedback signal. It
allows the problem of structural-algorithmic uncertainty to be solved by dynamically forming
latent patches of variable size. Low-entropy regions (background) can be aggregated into large
spatial blocks, saving the $O(L^2)$ resources of the global attention mechanisms. High-entropy
regions (motion, fine details) can be split into small patches, concentrating the maximum
computational power of the transformer exactly on the zones where the uncertainty is highest.

It is important to note the place of this instrument within the overall system developed in this
dissertation. The per-pixel entropy is a *local* measure of informational complexity: it answers
the question "where exactly is prediction hardest for the system". It thereby complements the
*global* consistency criteria of the generation process — the hierarchical- and
trajectory-consistency criteria introduced in Chapter 1, which for the hierarchical generation
model developed in Chapter 3 are operationalized in Chapter 4 through autoregressive-rollout
degradation metrics stratified by re-entry hardness. The local entropy signal governs the
allocation of computational resources within a frame, while the global criteria evaluate the
consistency of the generated process as a whole.

The successful substantiation and approbation of the entropy-based method for estimating
spatio-temporal uncertainty opens the path to building fully resource-efficient video-generation
models. However, the process of transforming a multidimensional spatio-temporal data array into a
linear sequence of patches for transformer processing carries a serious mathematical problem —
the loss of structural topology. The next subsection is devoted to solving this problem by
adapting and integrating multidimensional rotary positional encoding.

## 2.2 Structural Coherence through Multidimensional Positional Encoding

During the transformation of a multidimensional spatio-temporal information array into a form
suitable for processing by the model, the primary topology of the complex system is destroyed.
Therefore, an important step is to ensure structural coherence before feeding the data into the
system.

The multidimensional tensor of visual data, mathematically described as
$X \in \mathbb{R}^{T \times H \times W \times C}$, where $T$ is the temporal dimension, $H$ and
$W$ are the spatial coordinates, and $C$ the information channels, necessarily undergoes a
linearization operation. As a result, a one-dimensional sequence of latent vectors
$Z = \{z_1, z_2, \dots, z_L\}$ is formed, with total sequence length $L = T \cdot H \cdot W$.
This change of dimensionality is a necessary condition for applying global self-attention
mechanisms, but it completely erases the information about the mutual spatio-temporal arrangement
of the elements.

In the theory of complex systems, such a loss of information is classified as a critical growth
of structural uncertainty. The transformer architecture is, by its mathematical nature, invariant
to permutations of the elements of a set. This means that without additional control signals the
system cannot distinguish whether two latent patches were physically adjacent horizontally within
one frame, or belonged to entirely different frames of the video sequence in time. The absence of
a precise coordinate anchoring makes it impossible to model adequate kinematics and geometry of
the visual scene.

The traditional approach to preserving positional information is absolute positional encoding
[56]. This method adds a special coordinate
vector $p_i$ to each latent token $z_i$. However, absolute encoding has fundamental drawbacks in
this application: it destroys translation invariance, which is a natural property of the visual
world, since the system must recognize and generate an object equally well regardless of its
absolute coordinates in the frame.

An alternative approach is relative positional encoding (Relative Positional Bias)
[58, 28], where coordinate information is
added directly to the attention-weight matrix in the form of biases. Although this method models
relative distances more correctly, it is accompanied by a significant growth of computational
complexity and requires $O(L^2)$ memory for storing multidimensional bias matrices, which
directly contradicts one of the main tasks of this research — the optimization of resource
consumption.

To overcome these systemic contradictions, within the development of the information technology
it is proposed to apply the mathematical apparatus of Rotary Positional Embedding (RoPE)
[59] with its subsequent generalization to N-dimensional space. The
RoPE method combines the advantages of absolute encoding — the absence of quadratic memory costs
for storing positions — with the translation-invariance property of relative encoding. The core
idea is to represent the state vectors of the system as complex numbers and to apply to them an
orthogonal rotation operation in the complex space.

Let the attention subsystem operate on query vectors $q$ and key vectors $k$ of dimensionality
$D$. For a one-dimensional sequence, the position of element $m$ is encoded by multiplying the
vector by an orthogonal rotation matrix $R_{\Theta,m}$. Applying this operation to the query and
key vectors before computing their scalar product makes the attention weight a function of their
relative distance only. In the complex representation this is proved by the following equation:

$$
\left\langle R_{\Theta,m}\,q,\; R_{\Theta,n}\,k \right\rangle
= \operatorname{Re}\!\left[ \left(q\,e^{im\theta}\right) \left(k\,e^{in\theta}\right)^{*} \right]
= \operatorname{Re}\!\left[ q\,k^{*}\,e^{i(m-n)\theta} \right]
\tag{2.4}
$$

As the equation shows, the absolute coordinates $m$ and $n$ cancel algebraically, leaving only
their difference $m - n$. This mathematically guarantees that the attention system reacts
exclusively to the relative arrangement of elements.

However, the original RoPE method [59] was developed exclusively for
one-dimensional sequences such as text and cannot be applied directly to spatio-temporal video
data without losing the multidimensional topology. To solve this problem, within the research
underlying the results of **[A1]**, the method of N-dimensional rotary positional embedding
(RoPE-ND) [59] is used, which allows the system to
simultaneously account for the mutual arrangement of elements along several independent
coordinate axes.

The essence of the generalization is the decomposition of the latent token's feature space into
$N$ independent subspaces, each responsible for its own physical coordinate. For video data the
dimensionality of the coordinate system is $N = 3$: the time axis $t$, the vertical axis $y$, and
the horizontal axis $x$. Accordingly, the state vectors $q$ and $k$ of total dimensionality $D$
are split into three equal non-overlapping segments of dimensionality $d = D/3$.

For the mathematical rotation operation in the complex plane to be possible, the dimensionality
of each subspace must be an even number. This yields a hard architectural requirement on the
generative system: the total dimensionality of the latent feature space $D$ must be divisible by
$2N$, i.e., divisible by 6 in the case of video data.

For each of the three coordinates, an independent set of basis rotation frequencies $\theta_j$ is
formed. The frequencies are typically defined by a geometric progression:

$$
\theta_j = 10000^{-2j/d}, \qquad j \in \{0, \dots, d/2 - 1\}
\tag{2.5}
$$

Let a visual token of the system be located at the spatio-temporal point with physical
coordinates $(t_c, y_c, x_c)$. The rotation angles for each segment are computed as the product
of the coordinate and the corresponding frequencies: $t_c\theta_j$, $y_c\theta_j$, $x_c\theta_j$.
The resulting unitary transformation of the whole state vector $q$ is formed as the concatenation
of three independently rotated sub-vectors:

$$
q'_{\text{rotated}} = \left[\; R^{d}_{\Theta,t_c} q_t \;\big\|\; R^{d}_{\Theta,y_c} q_y \;\big\|\; R^{d}_{\Theta,x_c} q_x \;\right]
\tag{2.6}
$$

where $R^{d}_{\Theta,\text{pos}}$ denotes the $d \times d$ rotation matrix for the corresponding
axis. The same operation is applied to the key vector $k$.

![N-dimensional rotary position embedding](../../figures/fig47_rope_nd.png)

**Figure 2.5 — N-dimensional rotary positional embedding.** Independent rotations encode
horizontal, vertical and temporal coordinates so attention depends on relative displacement
along all three axes. Source: author’s schematic of the [A1] method.

Applying such a multidimensional transformation restores the structural coherence of the complex
system. When the attention-score matrix between two arbitrary tokens in the video sequence is
computed, the scalar product decomposes into three independent sums, each depending exclusively
on the coordinate difference along the corresponding axis:

$$
q'^{\top}_{\text{rotated}}\, k'_{\text{rotated}} = f(\Delta t, \Delta y, \Delta x)
\tag{2.7}
$$

where $\Delta t = t_q - t_k$, $\Delta y = y_q - y_k$, $\Delta x = x_q - x_k$. This proves that
the system mathematically identifies not the absolute position of objects in space but their
relative spatio-temporal topology, which is critically important for modeling physical motion.

**Systemic significance of the method for hierarchical architectures.** Beyond the plain modeling
of temporal dynamics in patch-based systems, the RoPE-ND method is a universal instrument and
plays a decisive role in ensuring *hierarchical consistency* in the hierarchical video generation
model developed in Chapter 3. In multi-level architectures, a problem of resolving structural
uncertainty arises between different levels of abstraction: for example, how to correctly
transfer information between a fine level with a dense token grid (e.g., $32 \times 32$ tokens)
and a coarse level with a sparser grid (e.g., $16 \times 16$ tokens).

For the correct routing of cross-attention signals, the coordinate systems of these levels must
be mathematically aligned. The RoPE-ND method offers an elegant systemic solution through affine
scaling of the coordinates. A parent coarse-level token physically covers a cluster of several
child fine-level tokens; after the coordinates of both levels are brought to a common scale, the
attention mechanism is forced to center precisely on the geometrically relevant zones. This
eliminates the risk of chaotic mixing of semantic features between distant objects and ensures
that the global composition governs exclusively its own local details — the mathematical basis of
the unidirectional inter-level context-signal transfer developed in Chapter 3.

From the standpoint of resource efficiency, the computational complexity of applying RoPE-ND is
strictly linear, $O(D)$, with respect to the feature-space dimensionality, and it requires no
additional memory for storing large bias matrices. This makes the developed mathematical
apparatus an ideal candidate for integration into architectures with time-agnostic
backpropagation, where memory is the most valuable resource. The description of such an
architecture and the corresponding optimization method is given in the following subsections.

## 2.3 Resource Optimization for Multi-Step Temporal Modelling

The improvement of the mathematical methods for uncertainty estimation (V-BLT) and structural
coherence (RoPE-ND) considered in the previous subsections substantially raises the quality of
spatio-temporal data modeling. However, the practical implementation of iterative generative
models inevitably runs into the exhaustion of available computational resources as a fundamental
systemic constraint. From the standpoint of systems analysis, the effectiveness of controlling
any process is determined not only by the adequacy of the mathematical model but also by the
feasibility of its implementation under strict resource constraints.

In the modeling of multi-step temporal processes, the main bottleneck is not so much the
computational throughput of the processors (the number of floating-point operations, FLOPs) as
the amount of operating memory of the graphics accelerators [60, 61]. This problem arises from the architectural features of
modern artificial neural networks and their optimization algorithms.

The standard Backpropagation Through Time (BPTT) algorithm [62] is the basis
for identifying the parameters of dynamic systems. To compute the control gradients, the
algorithm requires the mandatory storage of all intermediate states of the system — the
activations — at every layer and every time step during the forward pass.

Mathematically, the amount of operating memory $M$ required to store the full computation graph
is linearly dependent on the depth of the temporal unrolling and the number of layers in the
system:

$$
M = O(N \cdot T \cdot L \cdot D)
\tag{2.8}
$$

where $N$ is the dimensionality of the spatial state (the number of tokens per frame), $T$ the
number of time steps or optimization iterations, $L$ the number of model layers, and $D$ the
dimensionality of the latent feature space. Since for video data the spatial parameter $N$ is
extremely large, modeling long sequences $T$ very quickly leads to the exhaustion of the
available physical memory (Out of Memory, OOM).

The presence of such computational uncertainty forces various heuristic compromises. The most
common approaches are truncated backpropagation (Truncated BPTT) [63] and
gradient checkpointing [42, 43]. Truncated
backpropagation artificially limits the gradient horizon — e.g., optimizing over only the last 5
frames — which inevitably loses the memory of long-range cause-and-effect relations. The
checkpointing method reduces memory consumption (to $O(\sqrt{T} \cdot L \cdot D)$ with the
standard segment schedule; see Table 2.1), but the price is a significant slowdown of training
due to redundant recomputation.

For the effective functioning of the generative model, a fundamental revision of its architecture
is necessary. The systemic task is to develop a mathematical method of organizing the
computations that would provide constant memory consumption, $O(1)$ with respect to the number of
time steps $T$, without losing information about global long-range dependencies.

To solve this systemic problem, the dissertation research developed the Temporal Reversible
Transformer (TempVerseFormer) architecture. The developed architecture and the mathematical
method of computational-resource optimization were published by the author as the sole author in
an international scientific journal indexed by the Scopus scientometric database (**[A2]**).

The method is based on the transition from traditional directed acyclic computation graphs to
discrete reversible architectures. The core idea of reversible computation is to build
state-transition functions that are bijective (one-to-one). If the transition function from layer
$l$ to layer $l+1$ has an exact analytic inverse, the system can restore its previous state
purely from the current state, without storing it in memory during the forward pass.

The introduction of the reversibility principle into deep neural networks was previously studied
in continuous models of the Neural ODE type [44], but these suffer
from numerical instability during integration. Discrete reversible models, e.g., RevViT
[45], have shown high stability. The
scientific novelty of TempVerseFormer lies in combining structural reversibility at the level of
individual transformer layers with an iterative temporal-chaining mechanism for modeling
multi-step processes.

The structural decomposition of the reversible block in the TempVerseFormer architecture is based
on splitting the input information flow into two independent branches. Let the system operate on
a latent state of total dimensionality $D$. At the input of the reversible block, this state is
forcibly split into two mutually exclusive sub-vectors $x_1$ and $x_2$. The functioning of the
reversible block is described by a system of two sequential nonlinear transformations using the
attention mechanism and a multilayer perceptron (MLP).

The mathematical formalization of the forward pass through reversible block $l$ is described by
the following system of algebraic equations.

Update equation of the first branch:

$$
y_1^{(l)} = x_1^{(l)} + \operatorname{Attention}\!\left(x_2^{(l)}\right)
\tag{2.9}
$$

Update equation of the second branch:

$$
y_2^{(l)} = x_2^{(l)} + \operatorname{MLP}\!\left(y_1^{(l)}\right)
\tag{2.10}
$$

This specific algebraic structure guarantees that only $x_2$ — which remains unchanged during the
first stage — is used to compute the additive update of $x_1$. Accordingly, the input information
can be reconstructed absolutely without loss by performing the operations in reverse order and
replacing addition with subtraction.

Reverse reconstruction of the second branch:

$$
x_2^{(l)} = y_2^{(l)} - \operatorname{MLP}\!\left(y_1^{(l)}\right)
\tag{2.11}
$$

Reverse reconstruction of the first branch:

$$
x_1^{(l)} = y_1^{(l)} - \operatorname{Attention}\!\left(x_2^{(l)}\right)
\tag{2.12}
$$

![Forward and reverse information flow in the reversible block](../../figures/fig53_reversible_block_published.png)

**Figure 2.6 — Forward and reverse information flow in a reversible transformer block.** The
additive coupling structure permits exact reconstruction of block inputs from its outputs.
Source: author’s published TempVerseFormer project asset [A2].

For the reversible blocks to solve the task of controlling generation in time, they are
integrated into a higher structural level of the system through the **temporal chaining**
mechanism. Instead of concatenating all video frames into a single enormous tensor — which leads
to the quadratic memory collapse $O((N \cdot T)^2)$ in the attention mechanisms — the system
moves to an iterative autoregressive paradigm of state generation.

The temporal chaining mechanism works as follows. The system receives an initial context set
$V_{\text{ctx}}$. At step $t$, the system applies the transition functional — a stack of $L$
reversible blocks — and generates the time-shifted sequence $V_{\text{out}}$. This output tensor
is fed back directly and becomes the input $V_{\text{in}}$ for the next generation step $t+1$.
Mathematically this is described by the recurrence relation:

$$
V_t = \Phi(V_{t-1}, \theta)
\tag{2.13}
$$

where $\Phi$ is the transition functor built exclusively from algebraically reversible
transformer blocks.

![Temporal chaining mechanism](../../figures/fig54_temporal_chaining_published.png)

**Figure 2.7 — Temporal chaining for autoregressive time-series modeling.** The same reversible
transition is applied along physical time while the realized state is passed to the next step.
Source: author’s published TempVerseFormer project asset [A2].

Since the functor $\Phi$ consists of reversible elements, the entire multi-step trajectory
$V_1 \to V_2 \to \dots \to V_T$ preserves the property of global reversibility. Knowing only the
final state $V_T$, the system has the mathematical ability to reconstruct, step by step, all
intermediate values without storing them in memory.

To focus the computational resource of the reversible transformer exclusively on the analysis of
complex temporal dynamics, the global architecture of the TempVerseFormer information technology
uses an auxiliary spatial-compression subsystem — a variational autoencoder (VAE). Its function
is to resolve the spatial redundancy of pixels: each video frame is projected into a compact
latent vector before processing by the reversible blocks. After generation is completed, the
latent states are unfolded back by the VAE decoder. This architectural modularity separates the
task of spatial compression from the task of temporal control.

![TempVerseFormer architecture](../../figures/fig55_tempverseformer_architecture_published.png)

**Figure 2.8 — Global TempVerseFormer architecture.** A spatial VAE interfaces the video frames
with the reversible temporal core. Source: author’s published TempVerseFormer project asset [A2].

The integration of reversible blocks into a single system with temporal chaining creates an
extremely powerful mathematical instrument: structural reversibility guarantees that information
is not lost. Yet the main systemic challenge remains — converting this theoretical structural
reversibility into an effective training algorithm for computing the control gradients of the
loss function. Exactly how the reversibility property is exploited by the algorithm to remove the
computational uncertainty is considered formally in the next subsection.

## 2.4 Time-Agnostic Backpropagation through Reversible Transformations

Solving the problem of computational uncertainty — the critical deficit of operating memory in
the optimization of iterative and temporal systems — requires a transition from classical
training algorithms to new mathematical paradigms. The traditional BPTT method
[62] is the de-facto standard for identifying the parameters of dynamic
systems modeled by neural networks, but its mathematical architecture contains a fundamental
deficiency from the standpoint of resource control.

The BPTT algorithm is based on the chain rule of differentiation of a composite function. To
compute the control gradient of the loss function $L_{\text{loss}}$ with respect to the weight
parameters $\theta$ of the system at the current time step, one must know not only the derivative
of the state at the next step but also the exact values of the internal activations of the system
at every step. Mathematically this means that the forward pass must be accompanied by the caching
of absolutely all intermediate vectors of the multidimensional state space, for all steps, in the
operating memory of the computing device.

In the modeling of complex spatio-temporal video data, this requirement leads to a catastrophic
growth of resource consumption. The control of the system reaches a dead end: to increase the
temporal consistency of generation, the optimization trajectory $T$ must be lengthened, but
lengthening the trajectory exhausts the memory and makes the gradient computation impossible.

Within this dissertation research, the method of **time-agnostic backpropagation** was developed
to resolve this systemic contradiction. The method is based on integrating the mathematical
apparatus of discrete reversible transformations directly into the gradient-computation process
along the temporal chain.

The main idea of the developed method is a change in the paradigm of computation-graph
construction. Instead of statically caching all system states during the forward pass, they are
reconstructed dynamically during the backward pass. This reconstruction is mathematically exact
and requires no additional training or approximation, since it relies on the algebraic properties
of the inverse transition functions $R^{-1}$ built into the architecture of the reversible
blocks.

The training strategy of the complex generative system under this algorithm splits into two
phases.

**Phase one — the forward pass.** The system sequentially computes the evolution of the state.
For every time step $t$ from $1$ to $T$ and every layer $l$ from $1$ to $L$, the forward
reversible transformation $R$ is applied to the current inputs $(x_1, x_2)$. After the outputs
$(y_1, y_2)$ are computed, the system immediately deletes all internal activations from memory,
keeping only the result. Upon reaching the final step $T$, the system stores only one final
macro-state in memory.

![Activation-free forward evolution](../../figures/fig48_activation_free_forward.png)

**Figure 2.9 — Activation-free forward evolution.** Intermediate temporal states are discarded;
the terminal state, parameters and stochastic replay information are retained. Source: author’s
algorithmic schematic.

**Phase two — the reverse pass.** This phase is the conceptual core of the method. The reverse
pass starts with the computation of the gradient of the loss function with respect to the final
state of the system, $\nabla_{y_T} L_{\text{loss}}$. Then, for every time step $t$ from $T$ down
to $1$, the algorithm moves in the reverse direction: it reconstructs the layer inputs by the
inverse transformations (2.11)–(2.12), locally rebuilds the computation graph of a single
transition, computes the local gradients, and immediately destroys the local graph.

![Reverse local replay](../../figures/fig49_reverse_local_replay.png)

**Figure 2.10 — Reverse-pass local replay.** Each state is reconstructed, its local graph is
rebuilt, gradients are accumulated, and the graph is released before moving to the preceding
time step. Source: author’s algorithmic schematic.

![Two-phase time-agnostic backpropagation](../../figures/fig50_time_agnostic_backprop.png)

**Figure 2.11 — Two-phase time-agnostic backpropagation.** A tape-free forward phase is followed
by exact reverse reconstruction and local gradient accumulation. Source: author’s algorithmic
schematic.

For a rigorous confirmation of the effectiveness of the developed method, we formulate and prove
a lemma on the optimization of operating-memory consumption.

**Lemma 1 (on the computational complexity of time-agnostic backpropagation).** *Let a discrete
dynamic system be given with a trajectory length of $T$ steps, where the transition functor is
parameterized by a reversible neural network with $L$ layers and per-layer state dimensionality
$D$. Then the peak amount of operating memory required to compute the gradients of the objective
function with respect to the system parameters is $O(L \cdot D)$ and is constant, $O(1)$, with
respect to the length $T$ of the temporal trajectory.*

**Proof.**

1. Assume the dimensionality of the state tensor at one layer is $D$. Under classical BPTT the
   algorithm caches all inputs for every layer $l \in \{1, \dots, L\}$ and every time step
   $t \in \{1, \dots, T\}$; hence the peak memory is
   $M_{\text{BPTT}} = T \cdot L \cdot D = O(T \cdot L \cdot D)$.

2. The proposed algorithm stores, during the forward pass, only the final state at step $T$. The
   memory for this is $O(D)$. Activations for $t < T$ and $l < L$ are deleted immediately after
   they are used for the next step. Hence the forward-pass memory does not depend on $T$.

3. During the reverse pass the algorithm processes the steps sequentially: $t = T, T-1, \dots, 1$.
   At the current step $t$ the algorithm reconstructs the inputs and unrolls the local
   computation graph of a single transition. The size of this graph is $O(L \cdot D)$.

4. Immediately after the gradients $\nabla_{\theta} L_{\text{loss}}$ and
   $\nabla_{x_{t-1}} L_{\text{loss}}$ are computed, this local graph is completely destroyed and
   its memory freed before the algorithm proceeds to step $t-1$.

5. Therefore, the maximum number of tensors simultaneously resident in the system memory at any
   moment equals the size of one local graph, $O(L \cdot D)$, plus a constant volume for the
   gradient-accumulation buffer. Since the temporal parameter $T$ does not enter this expression,
   the peak memory consumption is mathematically $O(1)$ with respect to the sequence length.
   $\blacksquare$

For comparative analysis, Table 2.1 presents the theoretical complexity estimates of the
developed method against the existing approaches.

**Table 2.1 — Theoretical estimate of computational complexity in multi-step optimization**

| Optimization method | Memory consumption | Operation count | Gradient exactness |
|---|---|---|---|
| Standard BPTT | $O(T \cdot L \cdot D)$ | $C$ | exact |
| Gradient checkpointing | $O(\sqrt{T} \cdot L \cdot D)$ | $\approx 1.5\,C$ | $\pm 10^{-6}$ |
| Developed method (time-agnostic BP) | $O(L \cdot D) \sim O(1)$ in $T$ | $2\,C$ | $\pm 10^{-6}$ |

![GPU memory consumption versus prediction horizon](../../figures/fig57_tempverse_memory_published.png)

**Figure 2.12 — GPU memory consumption versus prediction horizon.** Published TempVerseFormer
measurement at batch size 32. The reversible temporal model remains near 0.9 GB over 1–1,024
prediction steps, whereas the two non-reversible transformer variants grow to approximately
160 GB in the reported extrapolated/extended measurement and the LSTM grows more slowly. The
logarithmic vertical scale must be retained when typesetting. Source: Kulyk, *Temporal modeling
with reversible transformers*, Fig. 13, author preprint (CC BY 4.0), DOI
10.21203/rs.3.rs-6293520/v1; version of record DOI 10.1007/s00521-025-11707-3.

![Training time versus prediction horizon](../../figures/fig58_tempverse_time_published.png)

**Figure 2.13 — Training time per step versus prediction horizon.** Published comparison at batch
size 32 and 1–1,024 prediction steps. Time grows with the prediction horizon for all four
architectures; TempVerseFormer pays a compute premium relative to the matched non-reversible
transformer while preserving the qualitatively different memory-scaling behavior in Fig. 2.12.
The logarithmic vertical scale must be retained when typesetting. Source: Kulyk, *Temporal
modeling with reversible transformers*, Fig. 14, author preprint (CC BY 4.0), DOI
10.21203/rs.3.rs-6293520/v1; version of record DOI 10.1007/s00521-025-11707-3.

As Fig. 2.12 shows, measured temporal-model memory remains nearly constant even at large
prediction depths. Figure 2.13 also makes the trade-off explicit rather than cost-free:
TempVerseFormer is slower per step than the matched TempFormer over the tested horizons, and all
methods become slower as the horizon grows. Within this published configuration, the developed
method is the only compared transformer whose activation-memory curve is practically independent
of trajectory length. Algebraic reconstruction during the reverse pass therefore exchanges
additional computation for the critically scarce resource — GPU memory — and makes horizons of
hundreds or thousands of steps operationally possible when the non-reversible activation tape
would exceed the device budget.

In addition, the proposed architecture enables a unique **selective backpropagation** strategy.
In many control problems the initial steps of a process are noisy, while the most informative
steps are the last ones. The method allows the system to run the forward pass to full length and
then reconstruct activations and compute gradients only for a narrow target segment
$[T - K_{\text{opt}},\; T]$, ignoring the rest of the chain. This focuses the optimization on the
relevant portions of the trajectory with zero memory cost for the skipped stages.

### Extension to Stochastic Recurrences and Limits of Reversible Computation

The time-agnostic backpropagation method was substantiated above for *deterministic* transition
functors (2.13). The hierarchical video generation model developed in Chapter 3, however, is a
*stochastic* dynamic system: its state transitions include an injected noise component, both as a
generative modeling choice and within its iterative-refinement generation core. For the developed
resource-optimization method to serve as the training foundation of that model, it must be
extended to stochastic recurrences, and the boundaries of its applicability must be established.
This section performs both steps; all numerical values are measured properties of the software
implementation (the measurement methodology and the experimental environment are described in
Chapter 4).

**(i) The exactly invertible stochastic recurrence (store-the-noise).** Consider the stochastic
generalization of the temporal chain (2.13), in which the latent state $h_t$ of the modeled
process evolves as

$$
h_t = F(h_{t-1}) + \sigma\,\epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I),
\tag{2.14}
$$

where $F$ is a transition operator built from reversible (additive-coupling) blocks, $\sigma$ is
the noise scale, and $\epsilon_t$ is the noise sample of step $t$. Transformation (2.14), taken
alone, is not invertible: the noise irreversibly injects new information. However, if the noise
samples $\epsilon_t$ drawn during the forward pass are *stored* — the store-the-noise principle,
extending the reversible recurrent networks of [64] — the
recurrence becomes exactly invertible. Equations (2.14)–(2.15) state the principle in its
simplest, *external-additive* form, in which the stored noise is subtracted before the inverse
transition is applied:

$$
h_{t-1} = F^{-1}\!\left(h_t - \sigma\,\epsilon_t\right).
\tag{2.15}
$$

The construction **implemented and measured in this dissertation** is the more general
*conditioning-injected* form: within a coupled two-branch state $(s_1, s_2)$, the stored noise
enters the coupling functions as a conditioning argument,

$$
(y_1, y_2) \;=\; \Bigl(s_1 + F(s_2, \epsilon_t),\;\; s_2 + G\bigl(y_1\bigr)\Bigr),
\tag{2.16}
$$

whose closed-form inverse subtracts the couplings in reverse order with the *same* stored
conditioning replayed:

$$
s_2 = y_2 - G(y_1), \qquad s_1 = y_1 - F(s_2, \epsilon_t).
\tag{2.16'}
$$

The additive-coupling inverse is exact for an *arbitrary* conditioning argument, so (2.16')
holds verbatim for any signal injected this way (noise, actions, level context); the
external-additive scheme (2.14)–(2.15) is recovered as the special case in which the noise
enters only as a final additive term. All numerical properties quoted below are measured on the
conditioning-injected construction (2.16)–(2.16'), which is the one realized in the software
complex. The memory cost of storing the noise sequence grows only as $O(T \cdot D_{\epsilon})$ —
one noise tensor per step, with no layer factor $L$ and no computation-graph factor — and is
negligible against the $O(T \cdot L \cdot D)$ activation graph that BPTT would store (confirmed
by the direct measurement in part (iii): the total memory of the reversible method stays flat).

The following properties of this construction were verified numerically on the implemented
hierarchical multi-rate recurrent core (software complex of the dissertation; measurement details
in Chapter 4):

- **Exact invertibility of the deterministic coupling:** maximum state-reconstruction error
  $5.5 \times 10^{-15}$ (fp64); for the Lipschitz-constrained coupling in fp32,
  $3.1 \times 10^{-5}$.
- **Exact invertibility of the stochastic recurrence:** with the stored noise replayed, the
  maximum reconstruction error over the trajectory is $8.4 \times 10^{-13}$; replaying *wrong*
  noise destroys the reconstruction (error of order $10^{1}$) — i.e., the stored noise is
  exactly the information-theoretic price of inverting a stochastic step.
- **Gradient equivalence:** the gradients obtained by reverse-replay (reconstructing states
  backward by (2.15) and rebuilding local graphs, as in Lemma 1) coincide with the gradients of
  full BPTT with cosine similarity $1.000000$ — the extension preserves the exact-gradient
  property of the deterministic method.

Thus the time-agnostic backpropagation of §2.4 transfers to stochastic multi-level recurrences
without loss of exactness, which is precisely the form required by the hierarchical generation
model of Chapter 3 (Task 4 of the research in its extended formulation).

**(ii) Invertibility does not imply stability: the Lipschitz characterization.** A separate
systemic question is whether the reversibility of the transition operator stabilizes the
long-horizon behavior of the modeled process. The answer, established by direct measurement, is
negative — and this delimits what the developed method may and may not be credited with. All
recurrences in Table 2.2 are *exactly invertible* (the additive coupling has unit Jacobian
determinant); they differ only in the spectral-norm bound imposed on the coupling function $F$.
The measured perturbation growth $\lVert \delta_K \rVert / \lVert \delta_0 \rVert$ of a
free-running rollout over $K = 100$ steps spans four orders of magnitude:

**Table 2.2 — Perturbation growth of exactly invertible recurrences over $K = 100$ free-running
steps as a function of the Lipschitz bound on the coupling**

| Coupling constraint | $\lVert \delta_{100} \rVert / \lVert \delta_0 \rVert$ | Regime |
|---|---|---|
| unconstrained | $333\times$ | explodes |
| spectral norm, $\operatorname{Lip}(F) \le 1$ | $26{,}600\times$ | **still explodes** |
| spectral norm, $\operatorname{Lip}(F) \le 0.1$ | $1.5\times$ | bounded |
| spectral norm, $\operatorname{Lip}(F) \le 0.02$ | $1.0\times$ | tightly bounded |

The mechanism is elementary: the additive-coupling step $s \mapsto s + F(s)$ has a step-Lipschitz
constant

$$
\operatorname{Lip}(\text{step}) \approx 1 + \operatorname{Lip}(F) \;\ge\; 1,
\tag{2.17}
$$

i.e., the additive structure is inherently non-contractive — constraining $F$ to
$\operatorname{Lip}(F) \le 1$ is *not* sufficient (the growth factor $26{,}600\times$ in
Table 2.2), and bounded long-horizon dynamics require a *small* coupling Lipschitz constant
($\operatorname{Lip}(F) \lesssim 0.1$). This is the stability analysis of invertible networks
[65, 66] instantiated
for temporal recurrences and confirmed by measurement. The systemic conclusion: **invertibility
is a memory-and-correctness property, not a stability property.** In the developed technology,
reversibility is claimed exclusively as the constant-memory enabler of long-horizon training with
exact gradients; the stability of the modeled trajectory is a separate design requirement,
enforced by the Lipschitz constraint on the coupling (this requirement informs the architecture
of Chapter 3).

![Measured reversible memory frontier](../../figures/fig51_reversible_memory_frontier.png)

**Figure 2.14 — Measured memory frontier of the stochastic recurrent core.** Standard BPTT
reaches 12.2 GB at $K=40$, spills at $K=80$ and cannot run at $K=120$ on the 16 GB device;
checkpointing grows slowly, while reverse replay stays near 1.1–1.2 GB through $K=160$. Source:
repository Panel-A measurements (`hvp_rev/mem_frontier.py`,
`docs/SESSION_JOURNEY_AND_EBT_PLAN.md`, §2.2).

**(iii) The constant-memory autoregressive-rollout frontier (measured).** The practical payoff of
the extended method is measured on the real training loop of the implemented stochastic
hierarchical recurrent core (three levels, batch 32, Adam optimizer, NVIDIA RTX 4060 Ti 16 GB;
the full environment is specified in Chapter 4, §4.1.4). Table 2.3 gives the peak training memory as a function of the
free-running rollout length $K$:

**Table 2.3 — Measured peak training memory (MB) vs rollout length $K$: BPTT, gradient
checkpointing, and the developed reversible reverse-replay on the same model**

| $K$ | BPTT | Gradient checkpointing | Developed method (reversible) |
|---:|---:|---:|---:|
| 40 | 12,162 | 1,471 | 1,091 |
| 80 | 23,305\* | 1,637 | 1,125 |
| 120 | OOM | 1,801 | 1,152 |
| 160 | OOM | 1,967 | 1,176 |

\* At $K = 80$, BPTT exceeds the physical GPU memory and spills into shared host RAM, slowing a
training step by a factor of $\approx 60$ (from 0.74 s at $K = 40$ to 43.8 s); at $K = 120$ it
fails with a hard out-of-memory error.

The frontier confirms Lemma 1 and its stochastic extension empirically: BPTT consumes 12.2 GB
already at $K = 40$ and becomes infeasible at $K = 120$, while the developed method remains
essentially flat ($\approx$1.1–1.2 GB through $K = 160$), making long-horizon rollout training
feasible on a consumer-grade 16 GB device. **The honest reading of Table 2.3, however, must
include the checkpointing column:** gradient checkpointing also remains feasible at all measured
$K$ (1.47–1.97 GB, growing slowly). The correct claim for the developed method is therefore a
**constant-memory capability relative to vanilla BPTT** — with strictly exact reconstruction and
gradients equal to BPTT — and an approximate **Pareto equivalence with gradient checkpointing**
(comparable memory order and comparable recomputation overhead), *not* a memory victory over
checkpointing. This scoping is stated deliberately: it is what the three-axis analysis below
turns into a general applicability boundary.

**(iv) The three-axis taxonomy of reversible computation and its no-go result.** The reversible
mechanisms developed and reviewed in this chapter can be organized along three orthogonal axes of
the computation graph on which reversibility can be exploited to avoid storing states:

1. **The depth axis** — reversibility across the layers of a network at a single step (RevNet and
   its descendants [45]): the
   literature baseline on which the developed blocks build.
2. **The temporal axis** — reversibility along the physical time of the modeled process
   (temporal chaining of §2.3–2.4 and the stochastic extension (2.14)–(2.16)). On this axis
   reversibility **pays**: the stored quantity that BPTT accumulates is the $O(T \cdot L \cdot D)$
   activation trajectory itself, exactly what the inverse coupling reconstructs; the measured
   result is the frontier of Table 2.3.
3. **The computational axis** — reversibility along the *inner optimization loop*: the $K$
   pseudo-time steps of iterative refinement (gradient descent, MCMC/Langevin sampling, or a
   diffusion sampler) executed *inside* one generation step. The original draft of this
   dissertation asserted that the graph isomorphism between the temporal chain $T$ and the inner
   chain $K$ makes time-agnostic backpropagation the foundation for differentiating through such
   loops. The isomorphism of the graph *structure* is real, but the memory *accounting* is
   opposite, and on this axis reversibility **provably cannot pay** against checkpointing.

The accounting for the third axis is as follows. Differentiating through a $K$-step
gradient-descent (or Langevin) inner loop requires, at each reverse step, a Hessian-vector
product and mixed second-order terms — i.e., rebuilding the differentiable graph of
$\nabla E$ (a `create_graph`-class second-order graph) from a fresh forward pass, *regardless of
how the trajectory state at that step was obtained*. Reversibility can eliminate only the storage
of the trajectory states themselves. **The direct four-way measurement on the implemented system**
(naive full-graph / truncated / checkpointed / reversible differentiation of the inner loop,
$K \in \{4, 8, 32, 128\}$, micro-batch 12, fp32) gives: the naive full graph consumes 18.1 GB
already at $K = 4$ (spilling to shared memory) and exhausts device memory at $K = 8$; the
truncated, checkpointed, and reversible variants all remain flat at 5.282–5.296 GB through
$K = 128$; and the reversible variant saves **14 MB = 0.26% of the peak** relative to
checkpointing, at approximately twice the wall-clock time (recomputation). The per-step descent
state $(y, v)$ is 0.117 MB, so checkpointing's total state storage at $K = 128$ is 15.0 MB =
0.28% of its measured peak — an upper bound on what reversibility could ever reclaim on this
axis. The analytic accounting band for the same quantity is $\approx 0.6$–$1.0\%$, and the
crossover at which trajectory states would begin to dominate the second-order graph lies at
$K \approx 5{,}000$–$30{,}000$ inner steps, far beyond any practically meaningful refinement
depth. The underlying storage arithmetic — counting the bits per step that an optimizer
trajectory consumes versus the volume contraction of the update map — is due to
[67], **applied** here to the video-generation inner loop, not claimed as a new theorem.
Moreover, plain gradient descent is not closed-form invertible at all (inversion requires the
momentum-buffer scheme of Maclaurin et al., whose kilobyte-scale state itself demonstrates that
trajectory storage was never the binding constraint), and the reversible-sampler variants for
diffusion inner loops are already published [68, 69, 70], with large-scale practice choosing checkpointing
or truncation precisely because state $\ll$ graph [71, 72].

![Axes of reversible computation](../../figures/fig52_reversibility_axes.png)

**Figure 2.15 — Three axes of reversible computation and their measured applicability.**
Reconstruction pays along network depth and temporal recurrence. In the tested inner-optimization
loop the model/optimizer base state dominates: reversible replay saves only 14 MB (0.26% of peak)
relative to checkpointing while approximately doubling wall time. Source: author’s taxonomy from
`logs_rhcp/mem_panel.md`.

The resulting **applicability boundary** of the developed resource-optimization method is stated
as the closing result of this chapter: *reversibility pays on recurrences whose transition
operator is the invertible coupling itself — the temporal rollout of the modeled video process —
and cannot pay on inner optimization loops, whose reverse step is dominated by the reconstruction
of a per-step derivative graph; the latter regime belongs to gradient checkpointing.* This
scoping replaces the earlier $T \leftrightarrow K$ isomorphism claim with a two-sided,
measurement-backed delimitation: it simultaneously establishes where the method is the correct
tool — as the **enabling instrument** for constant-memory rollout-through-time training, whose
feasibility Chapter 4 demonstrates by direct memory measurement (the validated Chapter 3 model
itself is trained single-window and does not presuppose it) — and closes, on the record, the
axis where it is not.

## 2.5 Chapter Conclusions

In the second chapter of the dissertation, a complex of mathematical methods was developed and
substantiated for resolving the structural, informational, and computational uncertainties in
multi-step video generation systems. The main results are as follows.

1. In accordance with **Task 3** of the dissertation research, a method for estimating the
   spatio-temporal uncertainty of video data was developed through the transition from rigid
   patch tokenization to adaptive entropy-based decomposition (the V-BLT architecture). It was
   established that computing the information entropy at the level of raw bytes yields per-pixel
   entropy maps that act as a reliable diagnostic indicator of complex kinematics and can be used
   for the optimal redistribution of the system's computational resources.

2. To ensure the structural coherence of data under spatial linearization, the mathematical
   apparatus of N-dimensional rotary positional embedding (RoPE-ND) was developed. It was proven
   that the method algebraically guarantees the preservation of the multidimensional
   spatio-temporal topology inside the attention mechanisms. A coordinate-scaling mechanism was
   developed that provides an instrument for precise spatial localization of cross-level
   cross-attention between abstraction levels. This is a level-appropriate instrument for
   architectures that require spatial coordinate transfer; the measured Chapter 3 predictor
   instead uses learned level/role/frame embeddings in joint attention, so its results do not
   depend on RoPE-ND.

3. In accordance with **Task 4**, the fundamental problem of computational uncertainty — the
   critical deficit of operating memory in BPTT optimization of long processes — was resolved.
   The TempVerseFormer mathematical architecture was developed on the basis of discrete
   reversible transformer blocks joined by the temporal-chaining mechanism.

4. The lemma on the computational complexity of the time-agnostic backpropagation algorithm was
   formulated and proven: the algebraic reconstruction of local graphs during the reverse pass
   reduces the peak memory consumption from $O(T \cdot L \cdot D)$ to $O(L \cdot D)$ — a constant
   with respect to the trajectory length — while preserving gradient exactness.

5. The time-agnostic backpropagation method was **extended to stochastic multi-level
   recurrences** via the exactly invertible store-the-noise recurrence: with stored noise, state
   reconstruction is exact (measured error $8.4 \times 10^{-13}$) and the reverse-replay
   gradients coincide with BPTT (cosine similarity $1.000000$). The measured memory frontier of
   the real training loop confirms the practical payoff — BPTT consumes 12.2 GB at rollout length
   $K = 40$ and fails (OOM) at $K = 120$, whereas the developed method remains at
   $\approx$1.1–1.2 GB through $K = 160$, with gradient checkpointing remaining a comparable
   alternative (1.47–1.97 GB) — i.e., the method provides a constant-memory capability relative
   to vanilla BPTT and approximate Pareto equivalence with checkpointing, which is stated
   honestly. The **applicability domain** of reversible computation was delimited by the
   three-axis taxonomy (depth — time — computation): the temporal-rollout axis pays and forms the
   resource foundation for the long-horizon autoregressive training of the hierarchical video
   generation model of Chapter 3; the inner-optimization-loop axis provably cannot pay against
   checkpointing (a saving of $\approx 0.2\%$ at $K = 128$, with the state-versus-graph
   accounting applied after Maclaurin et al.). It was additionally established by measurement
   that **invertibility does not imply stability**: exactly invertible recurrences span
   perturbation-growth regimes from $333\times$ to $1.0\times$ over 100 steps purely as a
   function of the Lipschitz bound on the coupling, so trajectory stability is a separate design
   requirement (a Lipschitz constraint), which is taken into account in the architecture of
   Chapter 3.

The results of this chapter are published in the author's works **[A1]** (§§2.1–2.2), **[A2]**
(§§2.3–2.4), and the extension to stochastic recurrences with the three-axis applicability
taxonomy in **[A4]**. These stable labels are bound mechanically to the final numbered publication
list.

---

<div style="page-break-after: always;"></div>

# Chapter 3 — A Hierarchical Mathematical Model for Video Generation with Long Multi-Stream Context

The preceding chapters performed the decomposition of the video-generation system and developed
the mathematical instruments for uncertainty estimation (entropy-based adaptive decomposition,
RoPE-ND) and for resource-efficient multi-step optimization (reversible transformations,
time-agnostic backpropagation). This chapter synthesizes these results into the central
contribution of the dissertation: a hierarchical mathematical model of video-sequence generation
in a compact continuous hierarchical latent state space with **long-span multi-stream context**:
the demonstrated system carries long-range temporal memory in a long-span context stream — the
coarse (spatially downsampled) stream being its measured spatial implementation within the
multi-stream family of §3.1/§3.3.5 — context signals are
transferred between abstraction levels within a single joint attention mechanism, and per-frame
generation is performed by an iterative-refinement (denoising diffusion) core whose selection is
justified empirically by the matched-capacity and matched-NFE mechanism study of Chapter 4.

Two methodological commitments frame the chapter. First, the model is described exactly as
implemented and verified in the software complex (module `hvp_rev/rhvd.py` of the developed
system; the design was pre-registered before the experimental campaign): every architectural
constant quoted below is the constant of the working system, not a proposal. Second, alongside
the generative model itself, the chapter formalizes the **measurement instrument** — a
purpose-built memory-horizon testbed with a re-entry-stratified evaluation protocol (§3.4). Under
the systems-analysis methodology of specialty 124, the instrument is a deliverable of equal rank
with the model: it is what operationalizes the trajectory-consistency criterion of Chapter 1
(§1.4.5) into an exactly computable, statistically disciplined measurement procedure.

## 3.1 Formalization of the Hierarchical Latent State Space

The chapter's organizing object, defined here once and used throughout (full formalization in
§3.3.5): a **context-stream family**. Each context stream is a triple
$s_i = (r_i,\; k_i,\; n_i)$ — spatial resolution $r_i$, temporal stride $k_i$, and slot count $n_i$, and a system configuration is a set of such streams
consumed by one shared attention backbone. In this vocabulary: a **flat** configuration is a
single fine-resolution stride-1 stream; a **temporal (rate) hierarchy** has two or more tick
rates at any resolution; a **resolution hierarchy** has two or more spatial resolutions; a
**deep hierarchy** has three or more streams; the **context budget** counts context tokens only,
with total sequence length and compute reported separately (§4.1). The two-level
coarse-plus-fine model developed next is the family's per-level-decodable resolution form — the
dissertation's demonstrated system, not the family's definition.

### 3.1.1 Two-Level State Decomposition and Frozen Autoencoder

Chapter 1 (§1.2.2) established the hierarchical decomposition of the visual scene state into
levels of abstraction as the core systems-analysis remedy for the anomalies of flat latent
spaces. The developed model instantiates this decomposition with $L = 2$ levels, which is the
minimal configuration that exhibits the phenomenon under study (a low-rate upper level able to
carry memory that the detailed lower level cannot afford), while remaining fully trainable and
evaluable on a single consumer-grade GPU — a deliberate scoping decision consistent with the
resource constraints formalized in Chapter 1.

The state of the video process at frame $t$ is represented not by the pixel frame
$x_t \in \mathbb{R}^{64 \times 64}$ but by a pair of latent codes produced by a shared
convolutional encoder $\mathcal{E}$:

$$
z_t \;=\; \bigl(z^{\mathrm{c}}_t,\; z^{\mathrm{f}}_t\bigr) \;=\; \mathcal{E}(x_t),
\qquad
z^{\mathrm{c}}_t \in \mathbb{R}^{16 \times 4 \times 4},
\qquad
z^{\mathrm{f}}_t \in \mathbb{R}^{16 \times 8 \times 8},
\tag{3.1}
$$

where $z^{\mathrm{c}}_t$ is the **coarse** (macro-level) code — 16 channels on a $4{\times}4$
spatial grid, 256 numbers per frame — and $z^{\mathrm{f}}_t$ is the **fine** (micro-level) code —
16 channels on an $8{\times}8$ grid, 1{,}024 numbers per frame. Both codes pass through a
$\tanh$ nonlinearity and are therefore bounded. The coarse code is, by construction and by
training incentive, the low-rate "layout" description of the scene (global composition and camera
position); the fine code carries texture and stroke detail. In the macro/meso/micro vocabulary of
§1.2.2, the coarse level plays the macro role and the fine level the micro role; the meso level is
absorbed — a two-level pyramid is the minimal sufficient instantiation for the claims under test.

The defining structural property of the autoencoder — adopted from the hierarchical latent video
modeling line of work [32, 13, 15] in minimal form — is **per-level decodability**: each level is *independently* decodable to
the full frame by its own decoder,

$$
\hat{x}^{\mathrm{c}}_t = \mathcal{D}_{\mathrm{c}}\bigl(z^{\mathrm{c}}_t\bigr),
\qquad
\hat{x}^{\mathrm{f}}_t = \mathcal{D}_{\mathrm{f}}\bigl(z^{\mathrm{f}}_t\bigr),
\tag{3.2}
$$

so that neither level is a mere residual of the other: the coarse code alone suffices to
reconstruct a (blurred but correctly composed) frame, and the fine code alone suffices to
reconstruct a detailed frame. This property is enforced during autoencoder training by
random-level dropout: at each training step exactly one of the two reconstruction losses
(L1 + MSE) is supervised, with the coarse branch up-weighted ($\times 4$) so that the low-rate
level is not ignored. The trained autoencoder has 1.85 M parameters and reaches a validation
reconstruction error of 0.0020 (MSE, fine level) on the testbed data of §3.4.

Two systemic decisions about this autoencoder follow the unidirectional-control principle
(§3.1.3) and are stated here explicitly:

1. **The autoencoder is frozen.** It is trained once, on the training clips only, and never
   fine-tuned afterwards; during the training and operation of the predictive core its parameters
   are algorithmically excluded from the gradient graph. The latent space is therefore a *fixed
   coordinate system* for the entire study — every predictive mechanism compared in Chapter 4
   operates in literally the same state space, which is a precondition for a valid matched-NFE
   comparison. This decision retains, in the new architecture, exactly the systemic function that
   the old design assigned to its frozen semantic encoder: invariance of the state coordinates
   during generation.

2. **The latent interface is standardized.** The raw $\tanh$-bounded codes are small in scale
   (per-channel standard deviations average 0.121 for the coarse and 0.162 for the fine level).
   Every latent channel $c$ of every level $\ell \in \{\mathrm{c}, \mathrm{f}\}$ is z-scored with
   statistics computed **on the training set only**:

$$
\tilde{z}^{(\ell)}_{t}[c] \;=\; \frac{z^{(\ell)}_{t}[c] - \mu^{(\ell)}_{c}}{\sigma^{(\ell)}_{c}},
\qquad
\sigma^{(\ell)}_{c} \ge 10^{-4},
\tag{3.3}
$$

   and all held-out data are standardized with the *same* constants. This step is part of the
   model definition, not an implementation detail: the diffusion process of §3.2.3 injects noise
   at unit scale, and an unstandardized latent (scale $\approx 0.12$–$0.16$) silently mis-scales
   the entire noise schedule — the initial noise sample is then 4–7× off-scale relative to the
   data manifold. Chapter 4 (§4.2) documents that this single scaling error is capable of
   reversing mechanism-level experimental conclusions; the standardization (3.3) is therefore
   elevated to a protocol requirement. In the remainder of the chapter, $z$ denotes the
   standardized latents.

The pixel decoders $\mathcal{D}_{\mathrm{c}}, \mathcal{D}_{\mathrm{f}}$ participate in the system
only downstream of the predictive core and only through the frozen interface — the old chapter's
"detached pixel decoder" principle survives here in a strictly stronger form (the decoder is not
merely gradient-detached but entirely frozen): pixel-space rendering can never destabilize the
optimization of the dynamics core, and all pixel-space metrics of §3.4.3 are measured through one
and the same fixed decoder for every compared system.

**Figure 3.1 — Model architecture and context-packaging alternatives.**
*Content:* input frames → frozen two-level conv-AE (shared encoder trunk; coarse
$16{\times}4{\times}4$ and fine $16{\times}8{\times}8$ taps); training-only latent
standardization; the shared WindowDiT generation core consuming a selected flat, spatial or
temporal package together with actions; a jointly predicted four-frame latent window; and the
frozen fine decoder. The lower panel reports exact package accounting and makes the 1,088-token
temporal versus 1,152-token spatial comparison explicitly near-matched rather than exact.
*Source:* **`docs/figures/fig16_context_packaging_architecture.png`** and vector PDF; generated
deterministically by `scripts/make_rft_core_schematics.py` from the frozen interface.

### 3.1.2 Cross-Level Context Transfer: from Cross-Attention to Joint Attention

If the two levels of the hierarchy were optimized in complete isolation, the generated states
would contradict each other — the coarse level could place scene content at one position while
the fine level renders texture consistent with another. Guaranteeing the integrity of the
multi-level system requires a strict information channel between the levels; Chapter 1 (§1.4.4)
introduced *hierarchical consistency* as the corresponding criterion. The question is the
mechanism of this **inter-level context-signal transfer**.

The classical mechanism, described in the earlier design iterations of this work, is cross-level
cross-attention: queries formed from the child level, keys and values projected from the parent
level, with positional coordinate scaling to align the two spatial resolutions. The developed
model retains the *function* of this mechanism but generalizes its *form*: instead of a directed
pair of attention maps between separate per-level networks, all tokens of both levels — context
and target alike — are placed into **one joint self-attention sequence** of a single shared
transformer (§3.2.1). Within every attention layer, each fine token attends to all coarse tokens
(and vice versa), with the level identity, temporal position, and role of every token supplied to
the network as additive embeddings and as per-token conditioning of the adaptive normalization
(§3.2.2). Cross-level transfer is thus realized as a special case of joint attention: the
attention sub-blocks between fine queries and coarse keys are exactly the cross-attention of the
earlier design, but they are learned jointly with the within-level blocks rather than imposed as
a separate module.

This generalization has three systemic consequences:

1. **Bidirectional consistency coupling at generation time.** Both levels of the target window
   are denoised *jointly, in one pass, under a shared noise level* (§3.2.3). The fine level
   conditions on the coarse level's current (partially denoised) state and the coarse level
   conditions on the fine level's — at every one of the refinement steps. Hierarchical
   consistency is therefore enforced continuously along the generation trajectory, not once at a
   hand-off between stages.

2. **Asymmetric information budgets without asymmetric wiring.** The long memory enters the
   sequence only as coarse tokens (48 frames × 16 tokens), while the fine level contributes only
   a short window (6 frames × 64 tokens). The *architecture* is symmetric; the *information
   design* is what makes the coarse level the carrier of long-range memory (§3.3.4). This is the
   precise sense in which "the coarse level carries the memory horizon" is a designed property of
   the developed model rather than an emergent accident.

3. **No hand-crafted spatial masks.** The old design constrained the child-to-parent attention by
   a hard $2{\times}2$ receptive-field mask routed by scaled positional coordinates. In the
   developed model, the spatial correspondence between a $4{\times}4$ coarse cell and its
   $2{\times}2$ block of $8{\times}8$ fine cells is available to the network through the learned
   per-level spatial position embeddings, but it is not imposed as a hard constraint; at the
   adopted resolutions the joint attention learns the routing directly. Thus the hard
   $2{\times}2$ mask of the superseded 32×32 design is explicitly not part of the validated
   4×4/8×8 implementation. RoPE-ND remains the scaled-coordinate instrument developed for the
   [A1] system in Chapter 2.

### 3.1.3 One-Way Control with Detached Gradient Paths

The old chapter postulated a strict *sequential* ordering of level generation (macro level first,
then meso, then micro, each receiving the previous level's output as a gradient-detached
constant) — and simultaneously, in its own algorithmic listing, executed the levels in the
opposite order. The resolution adopted in this dissertation is to state what the validated
implementation actually does and to re-scope the unidirectionality principle to the boundaries
where it is both real and load-bearing:

**Within a prediction step, the levels are generated jointly, not sequentially.** As established
in §3.1.2, one attention pass denoises both levels of the 4-frame target window simultaneously.
There is no macro-first or micro-first ordering *inside* the step, and the dissertation makes no
such claim. What survives of the top-down intuition is the **information-design asymmetry**: the
coarse level is the only level whose history extends over the full memory horizon, so at
generation time the long-range structure of the prediction can only originate from the coarse
context — the fine level physically lacks the inputs to contradict it at long range. The control
is unidirectional in the information-theoretic sense (long-horizon signals flow coarse → fine
because only the coarse channel transports them), not in the scheduling sense.

**Gradient detachment survives at two system boundaries**, where it is exact and deliberate:

1. **Between the representation subsystem and the prediction core.** The autoencoder is frozen
   (§3.1.1): no gradient of any prediction objective ever reaches $\mathcal{E}$,
   $\mathcal{D}_{\mathrm{c}}$, or $\mathcal{D}_{\mathrm{f}}$. The state coordinate system is a
   fixed constant of the control problem. This is the direct descendant of the old design's
   Detached-KV principle — the parent representation is a constant with respect to the child's
   optimization — applied at the representation/dynamics boundary where it demonstrably matters.

2. **Between past and future along the rollout.** At every autoregressive step (§3.3.1), the
   context buffers contain previously generated latents that enter the network as constants:
   during training no gradient propagates from the current window's loss into earlier windows
   (training is single-window by design, §3.2.3), and during generation the sampled window is
   appended to the buffers via an implicit detachment (sampling runs under no-grad). Destructive
   interference between the optimization of successive windows — the failure mode the old
   chapter's unidirectional control was designed to prevent — is thereby excluded structurally.

This formulation combines stop-gradient discipline, constant parent context and prevention of
bidirectional gradient interference. The implemented and validated system generates both levels
jointly in a single attention pass, with the coarse level distinguished by *what it remembers*,
not by a separate execution phase.

## 3.2 Generative Core: Windowed Latent Diffusion

The per-frame generation mechanism of the developed model is an iterative-refinement (denoising
diffusion) core. This choice is not aesthetic: Chapter 4 (§4.2) selects it by a matched-capacity,
matched-capacity and matched-NFE comparison of three mechanisms sharing this chapter's backbone — direct
regression, denoising diffusion, and energy-based gradient descent — in which diffusion commits
to sharp, correctly located content where regression hedges toward the conditional mean and the
energy-based alternative fails in a two-sided, characterized manner. This chapter presents the
selected core; the selection evidence belongs to Chapter 4.

### 3.2.1 Token Sequence and Component-Wise Conditioning

The prediction problem is windowed. Let $t$ be the first frame to predict, $W = 4$ the window
length, $H_{\mathrm{c}}$ the coarse context length and $w_{\mathrm{f}}$ the fine context length
(system configuration: $H_{\mathrm{c}} = 48$, $w_{\mathrm{f}} = 6$; §3.3.4 introduces the control
configurations). One prediction step consumes:

- the coarse context $\;C^{\mathrm{c}} = \bigl(z^{\mathrm{c}}_{t-H_{\mathrm{c}}}, \ldots,
  z^{\mathrm{c}}_{t-1}\bigr)$ — the last 48 coarse latents;
- the fine context $\;C^{\mathrm{f}} = \bigl(z^{\mathrm{f}}_{t-w_{\mathrm{f}}}, \ldots,
  z^{\mathrm{f}}_{t-1}\bigr)$ — the last 6 fine latents (the fine context frames are by
  construction the *last* $w_{\mathrm{f}}$ frames of the coarse window, so the two contexts are
  temporally aligned and satisfy $w_{\mathrm{f}} \le H_{\mathrm{c}}$);
- the recorded control actions $a_{t}, \ldots, a_{t+W-1}$, $a \in \{-1, 0, +1\}$ (§3.4.1), where
  $a_{t+j}$ is the action that moves the process from frame $t{+}j{-}1$ to frame $t{+}j$;
- the noisy target window $\;y = \bigl(y^{\mathrm{c}}_{t..t+3},\, y^{\mathrm{f}}_{t..t+3}\bigr)$
  at noise level $\tau$ (during generation — the current iterate of the sampler).

Every latent map is linearized into spatial tokens (one token per spatial cell, its 16 channels
linearly projected to the model width $d$): a coarse frame contributes $4 \times 4 = 16$ tokens,
a fine frame $8 \times 8 = 64$ tokens. The full input of one prediction step is the single joint
sequence

$$
S \;=\;
\underbrace{\bigl[\,\text{coarse ctx: } 48 \times 16\,\bigr]}_{768}
\;\Vert\;
\underbrace{\bigl[\,\text{fine ctx: } 6 \times 64\,\bigr]}_{384}
\;\Vert\;
\underbrace{\bigl[\,\text{targets: } 4 \times (16 + 64)\,\bigr]}_{320}
\;=\; 1472 \ \text{tokens},
\tag{3.4}
$$

of which $48 \cdot 16 + 6 \cdot 64 = 1152$ are context tokens — the context-token budget that
the matched-budget flat control of §3.3.4 is required to equal exactly.

Each token carries four additive identity embeddings: a learned spatial position embedding (one
table per level), a **level** embedding ($\mathrm{c}/\mathrm{f}$), a **role** embedding
(context/target), and a **frame-index** embedding. Frame indices are absolute within the step:
context frames occupy indices $0 \ldots 47$ (fine context frames share the indices $42 \ldots 47$
of their coarse counterparts), and the four target frames occupy indices $48 \ldots 51$ — the
frame-embedding table has 52 entries. On top of the identity embeddings, every token receives a
per-token conditioning vector $c$ used by the adaptive normalization of every block:

$$
c \;=\;
e_{\mathrm{level}} + e_{\mathrm{frame}} + e_{\mathrm{role}} +
\begin{cases}
e_{\tau}(0), & \text{context tokens (clean, noise level } 0\text{)},\\[2pt]
e_{\tau}(\tau) + e_{\mathrm{act}}(a_{t+j}), & \text{target tokens of frame } t{+}j,
\end{cases}
\tag{3.5}
$$

where $e_{\tau}(\cdot)$ is a sinusoidal-MLP embedding of the continuous noise level and
$e_{\mathrm{act}}$ is a learned embedding of the action alphabet. Two properties of (3.5) are
essential. First, context tokens are conditioned at noise level 0 — the network is explicitly
told which part of the sequence is clean evidence and which is the object of refinement; the
prediction loss is computed on target tokens only. Second, the action conditioning is
**per-target-frame**: frame $t{+}j$ receives exactly its own action $a_{t+j}$, so the network
never needs to disentangle a summed multi-step action code — each generated frame is bound to
the control signal that produced it.

**Figure 3.2 — Token layout for one prediction step.**
The 1,472-token sequence contains 768 coarse-context tokens, 384 recent fine-context tokens and
four 80-token coarse-plus-fine targets. The bracket identifies the 1,152-token context budget
matched by the central contiguous-fine control; the lower strips show the additive token
embeddings and per-target action/noise conditioning.
*Source:* `docs/figures/fig22_token_sequence_layout.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`, reflecting
`hvp_rev/rhvd.py::WindowDiT._ctx_tokens/forward`.

### 3.2.2 Transformer Backbone with Adaptive Normalization

The sequence (3.4) with conditioning (3.5) is processed by a stack of $N_{\mathrm{blk}} = 8$
pre-norm transformer blocks of width $d = 384$ with 6 attention heads and MLP expansion ratio 4
(22.18–22.19 M trainable parameters for the two-level arms; the 0.01 M difference is due to the
package-specific input/output interface). Attention is full (every token attends to every token): the causality of the
overall system is enforced by sequence construction and by the autoregressive recursion (§3.3.2),
not by an attention mask inside the step.

Each block applies **adaLN-Zero conditioning** [18]: the per-token
conditioning vector $c$ is mapped (SiLU → linear) to six modulation signals
$(\gamma_1, \beta_1, g_1, \gamma_2, \beta_2, g_2)$, and the block computes

$$
\operatorname{mod}(h; \beta, \gamma) = h \,(1 + \gamma) + \beta,
\tag{3.6}
$$

$$
h \leftarrow h + g_1 \cdot \operatorname{Attn}\bigl(\operatorname{mod}(\operatorname{LN}(h);
\beta_1, \gamma_1)\bigr),
\qquad
h \leftarrow h + g_2 \cdot \operatorname{MLP}\bigl(\operatorname{mod}(\operatorname{LN}(h);
\beta_2, \gamma_2)\bigr),
\tag{3.7}
$$

with the modulation projection zero-initialized, so every block starts as the identity — the
standard stable initialization for a denoiser trained as a residual refiner. Queries and keys are
RMS-normalized per head. After the final block, only the 320 target tokens are read out: each
level has its own final adaptive layer-norm (again zero-initialized) and its own zero-initialized
linear head projecting $d \to 16$ channels, after which the tokens are reshaped back into the
latent geometry — $(W, 16, 4, 4)$ for the coarse and $(W, 16, 8, 8)$ for the fine level. The
backbone, tokenizer, and conditioning are shared verbatim with the alternative mechanisms of the
Chapter 4 mechanism study; only the output interpretation and the training objective differ —
this is what "matched capacity" means operationally in this work.

### 3.2.3 v-Parameterization

Why a *window* of $W = 4$ frames rather than one frame at a time? The reason is
rollout-consistency of the training distribution. The autoregressive rollout of §3.3 appends
*all* frames it generates to the context buffers; if training predicted only a single frame at
offset $k = 4$ (the design of the preliminary single-step harness), the rollout context would
consist of stride-4 samples while the training context consisted of consecutive frames — a
train/rollout distribution mismatch built into the protocol itself. Predicting a window of $W$
*consecutive* frames per step and advancing the recursion by exactly $W$ keeps the buffers
consecutive at all times: the layout of every context the model sees at rollout is identically
distributed with the contexts it saw in training.

Let $z_0 = \bigl(z^{\mathrm{c}}_{t..t+3},\, z^{\mathrm{f}}_{t..t+3}\bigr)$ denote the clean
(standardized) target-window latents. The forward diffusion process corrupts them with Gaussian
noise $\varepsilon \sim \mathcal{N}(0, I)$ at a noise index
$\tau \sim \mathrm{U}\{1, \ldots, T\}$, $T = 1000$, *shared across the window's frames and both
levels within a training sample* (one $\tau$ per sample — the coupling that ties the two levels
and four frames to a common refinement trajectory):

$$
y \;=\; \sqrt{\bar\alpha_\tau}\, z_0 \;+\; \sqrt{1 - \bar\alpha_\tau}\,\varepsilon,
\tag{3.8}
$$

with the cosine noise schedule [4]

$$
\bar\alpha_\tau \;=\; \frac{f(\tau)}{f(0)},
\qquad
f(\tau) \;=\; \cos^{2}\!\Bigl(\frac{\tau/T + s}{1 + s} \cdot \frac{\pi}{2}\Bigr),
\qquad
s = 0.008,
\qquad
\bar\alpha_\tau \in [10^{-4},\, 0.9999].
\tag{3.9}
$$

The network is trained in the **v-parameterization** [73]: the regression target is the velocity

$$
v \;=\; \sqrt{\bar\alpha_\tau}\,\varepsilon \;-\; \sqrt{1 - \bar\alpha_\tau}\, z_0,
\tag{3.10}
$$

which interpolates between noise prediction (at low noise) and sign-flipped data prediction (at
high noise) and is well-conditioned across the whole schedule. Denoting by
$v_\theta\bigl(y,\, C^{\mathrm{c}},\, C^{\mathrm{f}},\, \tau,\, a\bigr)$ the two-level output of
the backbone (§3.2.2), the training objective is the unweighted sum of the two levels'
mean-squared v-errors:

$$
\mathcal{L}(\theta)
\;=\;
\mathbb{E}_{\,z_0,\,\tau,\,\varepsilon}
\Bigl[
\bigl\lVert v^{\mathrm{c}}_\theta - v^{\mathrm{c}} \bigr\rVert^2_2
\;+\;
\bigl\lVert v^{\mathrm{f}}_\theta - v^{\mathrm{f}} \bigr\rVert^2_2
\Bigr].
\tag{3.11}
$$

Equal weighting is meaningful only because of the standardization (3.3), which puts both levels'
latents at unit per-channel variance; without it the loss would be silently dominated by whichever
level's raw scale happened to be larger.

Optimization protocol (constants of the validated system): AdamW, learning rate
$3 \cdot 10^{-4}$, weight decay $10^{-4}$; linear warmup over the first 2{,}000 steps to a
constant learning rate (restart-safe by design — training is resumable from atomic checkpoints);
gradient-norm clipping at 1.0; batch size 16–32; 30{,}000 training steps per system arm; an
exponential moving average (EMA) of the parameters with decay 0.999 maintained throughout, with
*all* evaluation performed on the EMA weights. Training uses clean (ground-truth) contexts only —
no self-generated context injection in the base protocol; the pre-registered design reserves
context-noise augmentation as a single budgeted fallback. It was not activated: the observed
deep-horizon sharpness-retention failure is reported as a boundary result rather than repaired by
an adaptive retraining branch.

Training-noise/teacher-forcing scope note for the defense: the long-horizon capability claimed in
Chapter 4 is achieved *without* backpropagation through the rollout — the model is trained on
single windows with clean contexts, and its long memory is an inference-time property of the
information design (§3.1.2), not of a long-horizon training objective. Where a rollout-through-
time objective would be required, the constant-memory machinery of Chapter 2 is the enabling
instrument; the two contributions compose but do not presuppose each other.

### 3.2.4 DDIM

Generation inverts the diffusion with a deterministic DDIM sampler [17] over $N_{\mathrm{DDIM}} = 16$ steps on a rounded-linear sub-grid
$\tau_1 = T > \tau_2 > \cdots > \tau_{16} \ge 1$ of the training schedule. Starting from pure
noise $y^{(1)} \sim \mathcal{N}(0, I)$ over the whole two-level window, each step evaluates the
network once and converts its v-output into the implied clean-data and noise estimates:

$$
\hat{z}_0 \;=\; \sqrt{\bar\alpha_{\tau_i}}\, y^{(i)} \;-\; \sqrt{1 - \bar\alpha_{\tau_i}}\,
v_\theta\bigl(y^{(i)}, \cdot\bigr),
\qquad
\hat{\varepsilon} \;=\; \sqrt{1 - \bar\alpha_{\tau_i}}\, y^{(i)} \;+\;
\sqrt{\bar\alpha_{\tau_i}}\, v_\theta\bigl(y^{(i)}, \cdot\bigr),
\tag{3.12}
$$

$$
y^{(i+1)} \;=\; \sqrt{\bar\alpha_{\tau_{i+1}}}\; \hat{z}_0 \;+\;
\sqrt{1 - \bar\alpha_{\tau_{i+1}}}\; \hat{\varepsilon},
\qquad
y^{(N_{\mathrm{DDIM}}+1)} \;:=\; \hat{z}_0 \ \ (\text{final step}),
\tag{3.13}
$$

i.e. the deterministic ($\eta = 0$) DDIM update; the sampler's only stochasticity is the initial
noise draw, which the evaluation protocol controls with common random numbers (§3.4.4). The
computational price of generation is thus fixed: 16 network evaluations per 4-frame window — 4
network function evaluations per generated frame.

### 3.2.5 Regression Control Head

For the mechanism claims of Chapter 4 (hypothesis H2: the generative mechanism earns its keep at
rollout), the system admits a drop-in **regression** variant on the identical backbone: the noisy
target inputs are replaced by zeros, the noise level is fixed at $\tau \equiv 0$, the output is
interpreted directly as $\hat z_0$, and the objective is the smooth-L1 error on both levels. Its
"sampling" is a single forward pass. Architecture, tokenization, conditioning, context design,
parameter count, optimizer, and evaluation protocol are all shared with the diffusion core — the
two systems differ in the training objective and nothing else. The chapter records this variant
as part of the model family because the honest form of the dissertation's mechanism claim is
comparative, and the comparison is meaningful only under this strict parity.

## 3.3 Autoregressive Rollout with Ring-Buffer Memory

The developed technology operates as a fully **autonomous** autoregressive predictor: its
inputs are past video frames (through the latent interface) and the recorded low-dimensional
control track; no exogenous modality (in particular, no text) steers the generation. The control
track is not a side channel of convenience: in the systems framing of Chapter 1 the actions *are*
part of the observed state of the control system, and Chapter 4 (§4.2) documents that withholding
them forces every mechanism to hedge over the control-marginal future and manufactures a spurious
blur verdict. The salvaged autonomy principle of the old chapter therefore survives in a
sharpened form: autonomous with respect to semantic side information, conditioned on the recorded
control signal.

### 3.3.1 Rollout Recurrence

The state of the rollout between prediction steps is the pair of **ring buffers**

$$
B^{\mathrm{c}}_n = \bigl(\hat z^{\mathrm{c}}_{t_n - H_{\mathrm{c}}}, \ldots,
\hat z^{\mathrm{c}}_{t_n - 1}\bigr),
\qquad
B^{\mathrm{f}}_n = \bigl(\hat z^{\mathrm{f}}_{t_n - w_{\mathrm{f}}}, \ldots,
\hat z^{\mathrm{f}}_{t_n - 1}\bigr),
\tag{3.14}
$$

holding the last $H_{\mathrm{c}} = 48$ coarse and the last $w_{\mathrm{f}} = 6$ fine latents,
where $\hat z$ denotes ground-truth latents during warm-up and self-generated latents thereafter.
The recursion is:

**Algorithm 3.1 — Autoregressive rollout with ring-buffer memory** *(implemented as
`hvp_rev/rhvd.py::rollout_eval`)*

> **Input:** ground-truth frames $x_{0..H_{\mathrm{w}}-1}$ for warm-up ($H_{\mathrm{w}} = 48$),
> recorded actions $a_{1..T}$, frozen AE $(\mathcal{E}, \mathcal{D})$, trained EMA weights
> $\theta$, horizon $T$.
> **Output:** predicted frames $\hat x_{H_{\mathrm{w}}..T-1}$.
> 1. Encode and standardize the warm-up frames; initialize $B^{\mathrm{c}}, B^{\mathrm{f}}$ with
>    their last $H_{\mathrm{c}}$ and $w_{\mathrm{f}}$ latents; $t_0 \leftarrow H_{\mathrm{w}}$.
> 2. **for** $n = 0, 1, \ldots, \lfloor (T - H_{\mathrm{w}})/W \rfloor - 1$ **do**
> 3. &nbsp;&nbsp; read the actions $a_{t_n}, \ldots, a_{t_n+W-1}$ of the current window;
> 4. &nbsp;&nbsp; sample the window by DDIM (§3.2.4):
>    $(\hat z^{\mathrm{c}}_{t_n..t_n+3},\, \hat z^{\mathrm{f}}_{t_n..t_n+3})
>    \leftarrow \mathrm{DDIM}_{16}\bigl(v_\theta;\, B^{\mathrm{c}}_n,\, B^{\mathrm{f}}_n,\,
>    a_{t_n..t_n+3}\bigr)$;
> 5. &nbsp;&nbsp; append all $W$ predicted latent pairs to the buffers; drop the oldest $W$
>    entries of each (ring update); $t_{n+1} \leftarrow t_n + W$;
> 6. &nbsp;&nbsp; decode predictions for evaluation:
>    $\hat x_{t_n+j} = \mathcal{D}_{\mathrm{f}}\bigl(\hat z^{\mathrm{f}}_{t_n+j}\bigr)$,
>    $j = 0..3$ (through un-standardization and the frozen fine decoder).
> 7. **end for**
> On the 200-frame evaluation clips of §3.4: warm-up 48 frames, then
> $(200 - 48)/4 = 38$ rollout steps, i.e. 152 self-generated frames per clip — more than three
> full loop periods of the testbed world.

Because the training windows are consecutive and the recursion advances by exactly $W$, the
buffer contents at every rollout step are distributionally identical to training contexts
(§3.2.3) — the property that makes clean-context training sufficient. The warm-up length is
**universal across all compared systems** (48 frames even for arms whose context is shorter, e.g.
the flat control warms its 18-frame buffer from ground-truth frames 30–47): every arm starts
generating at the same frame of the same clip, so the per-(clip, horizon) records of §3.4 pair
exactly across arms.

**Figure 3.3 — Ring buffers for autoregressive rollout.**
After the common 48-frame ground-truth warm-up, each step combines the fixed 48-slot coarse and
6-slot fine buffers with four actions, samples four latent pairs with DDIM-16, and appends/evicts
four positions. The stored state is 18,432 latent numbers and does not grow with rollout length.
*Source:* `docs/figures/fig23_rollout_ring_buffer.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`, reflecting `hvp_rev/rhvd.py::rollout_eval`.

### 3.3.2 Temporal Routing and Block-Causal Semantics

The old chapter's prediction skeleton promised two routing mechanisms: a block-causal attention
mask (full attention within a frame, causal across frames) over a jointly optimized past+future
token tensor, and 3D-RoPE relative spatio-temporal positional coding. The developed system
realizes the *functions* of both, by different and simpler means, and the dissertation states the
mapping explicitly rather than letting the old vocabulary stand:

- **Block-causality holds at window granularity and is enforced by construction, not by a
  mask.** Within one prediction step, attention over the sequence (3.4) is full — and
  legitimately so: every token in the sequence belongs either to the observed past (context) or
  to the window currently being generated, whose frames are refined *jointly* as one block under
  a shared noise level. Across steps, the recursion of Algorithm 3.1 makes information flow
  strictly causal: frames of window $n{+}1$ are generated conditioned on (at most) windows
  $\le n$ through the buffers, and no token of any future window exists at step $n$. Viewed over
  an entire rollout, the effective attention pattern is exactly block-causal with block size
  $W = 4$ — realized structurally, which eliminates both the mask bookkeeping and any
  possibility of future-information leakage (there is no future present to leak).

- **Temporal position is encoded by learned absolute frame-index embeddings** (52-entry table,
  §3.2.1) rather than rotary relative coding. Because every prediction step re-indexes its
  context to the fixed range $0 \ldots 47$ and its targets to $48 \ldots 51$, the absolute
  indices *are* relative to the window, so the scheme is translation-invariant across the rollout
  by construction.
  RoPE-ND and these learned indices are therefore presented as level-appropriate instruments:
  RoPE-ND for the variable-patch, cross-resolution [A1] system in Chapter 2, and fixed learned
  indices for the re-indexed window used by the validated hierarchical predictor. No unmeasured
  RoPE-ND retrofit is attributed to the final model.

### 3.3.3 Memory and Computational Complexity

**Rollout memory is constant in the horizon.** The complete inter-step state of Algorithm 3.1 is
the buffer pair (3.14):

$$
M_{\mathrm{roll}}
\;=\;
H_{\mathrm{c}} \cdot \bigl|z^{\mathrm{c}}\bigr| \;+\; w_{\mathrm{f}} \cdot
\bigl|z^{\mathrm{f}}\bigr|
\;=\;
48 \cdot 256 \;+\; 6 \cdot 1024
\;=\; 18{,}432 \ \text{numbers}
\;=\; O(1) \ \text{in the rollout length } T,
\tag{3.15}
$$

plus the (constant) transient activations of one network evaluation. Generation compute is
linear in the horizon with a fixed per-frame constant — $N_{\mathrm{DDIM}} / W = 4$ network
evaluations per generated frame — and, crucially, the per-step attention cost never grows with
how long the system has been rolling, because the sequence length (3.4) is fixed. This replaces
the old chapter's inconsistent claims (its §3.4 derived a per-image memory reduction
$O(K \cdot T \cdot L \cdot D) \to O(T \cdot L \cdot D)$ while its §3.5 asserted
$O(K \cdot L \cdot D) \to O(D)$ for the same algorithm) with a single derivation that is true of
the implemented system: there is no inner optimization chain $K$ to reverse at all — sampling
runs gradient-free — and the memory object that matters for long-horizon operation, the rollout
state (3.15), is constant by design.

**Training memory is single-window and horizon-independent.** The training graph spans one
network evaluation over 1,472 tokens and $N_{\mathrm{blk}} = 8$ blocks; no gradient crosses window
boundaries (§3.1.3). The stored residual-state term is
$O(N_{\mathrm{blk}} S d)$, while a naïve attention implementation can additionally materialize
$O(N_{\mathrm{blk}}S^2)$ attention-score storage. Memory-efficient or fused attention changes
that kernel-dependent term but not the model's $O(N_{\mathrm{blk}}S^2d)$ attention compute.
Accordingly, peak VRAM is treated as an empirical implementation property. The justified
invariance is to the number of rollout windows; memory still depends on token count $S$ and hence
on the selected package.

**Token-budget economics of the hierarchy.** The systemic argument for the two-level design is a
counting argument. Let $H$ be the number of past frames a model can consult ("effective memory
frames"). A flat single-level model pays $64$ tokens per remembered frame; the hierarchical model
pays $16$ tokens per remembered frame beyond its short fine window:

$$
S^{\mathrm{flat}}_{\mathrm{ctx}}(H) \;=\; 64\,H,
\qquad
S^{\mathrm{hier}}_{\mathrm{ctx}}(H) \;=\; 16\,H \;+\; 64\,w_{\mathrm{f}}
\quad (w_{\mathrm{f}} \ \text{fixed}),
\tag{3.16}
$$

an asymptotic 4× token advantage per memory frame, hence (attention being quadratic in sequence
length within each of the $N_{\mathrm{blk}}$ layers, $O(S^2 d)$) an asymptotic ≈16× advantage in
attention cost for the same memory horizon. At the system's operating point the concrete numbers
are: hier-long consults 48 frames for 1152 context tokens; a flat model consulting the same 48
frames pays $48 \cdot 64 = 3072$ context tokens — 2.67× the tokens and, at full sequence lengths
(1472 vs 3392 tokens), a ≈5.3× ratio in per-layer attention cost. Conversely, a flat model
*constrained to hier-long's token budget* can afford only $1152 / 64 = 18$ frames of memory —
which is precisely the matched-budget control of §3.3.4. The claim is about **hierarchical
context packaging whose benefit is mediated by reach**, not an intrinsic superiority of coarse
features and not context length treated as a free variable: at equal budget,
the flat alternative's horizon (18 frames) falls short of the testbed's loop period (≈48 frames,
§3.4.1), while the hierarchical allocation covers it.

**Disposition of the reversible inner loop (salvage of old §3.3).** The earlier design stage of
this work proposed integrating the reversible-transformation apparatus of Chapter 2 *inside* the
per-frame generation loop ("Reversible MCMC": reconstructing the activations of a $K$-step
gradient-descent sampling chain instead of storing them). The final system contains no such
component, for two reasons established in this dissertation. First, the three-axis reversibility
taxonomy of Chapter 2 (§2.4) proves that the inner-optimization-loop axis cannot outperform
gradient checkpointing there: the chain's states must be retained for exact reversal, which *is*
checkpointing under another name — the axis is closed by analysis, not abandoned by neglect.
Second, the mechanism study of Chapter 4 removed the premise: the energy-descent core whose
$K$-step chains motivated the machinery was itself the characterized-negative mechanism, and the
selected diffusion core requires no gradient flow through its sampling recursion (3.12)–(3.13) at
all. Reversibility remains a contribution of this dissertation on the axis where it pays — the
temporal-rollout axis of Chapter 2 — and this paragraph is the honest record of where it does
not.

### 3.3.4 System Configurations and Matched-Budget Controls

The information design of §3.1.2 turns context allocation into an experimental variable. The
model family is parameterized by $(H_{\mathrm{c}}, w_{\mathrm{f}}, \text{core})$, and the
dissertation's claims are comparative statements over the following pre-registered
configurations (all sharing the backbone of §3.2.2, the training protocol of §3.2.3, and the
evaluation protocol of §3.4):

**Table 3.1 — System configurations (arms) and context-token accounting.**

| Configuration | Coarse ctx $H_{\mathrm{c}}$ | Fine ctx $w_{\mathrm{f}}$ | Context tokens | Core | Systemic role |
|---|---|---|---|---|---|
| **hier-long** | 48 | 6 | $48 \cdot 16 + 6 \cdot 64 = 1152$ | diffusion | the developed system: coarse memory ≥ loop period |
| hier-short | 6 | 6 | $6 \cdot 16 + 6 \cdot 64 = 480$ | diffusion | memory ablation (hierarchy without the horizon) |
| flat-budget | 0 | 18 | $18 \cdot 64 = 1152$ | diffusion | **token-budget-matched flat control** (equal cost, 18-frame horizon) |
| flat-long | 0 | 48 | $48 \cdot 64 = 3072$ | diffusion | window-matched flat exhibit (equal horizon, 2.67× budget) |
| reg-long | 48 | 6 | 1152 | regression | mechanism control (H2), §3.2.5 |
| persistence / AE round-trip | — | — | — | — | floor / ceiling references (§3.4.4) |

The flat configurations set $H_{\mathrm{c}} = 0$: coarse *context* tokens are absent, while the
coarse *target* tokens are retained, so the latent interface, the output heads, and the loss
(3.11) remain identical across all arms — the flat arm differs from the hierarchical one in
exactly one respect, the absence of coarse memory. The decisive comparison is hier-long versus
flat-budget: both pay 1152 context tokens per step, but the hierarchical allocation stretches
those tokens over a 48-frame horizon while the flat allocation reaches only 18 frames. The
pre-registered prediction — validated in Chapter 4 — is that the two systems separate exactly
where the testbed's re-entry events lie beyond the 18-frame flat horizon, and are otherwise
comparable; if they had *not* separated, the honest conclusion recorded in the design memo was
"context length, not hierarchy," and the dissertation would have reported it as such.

### 3.3.5 Generalization to a Family of Hierarchical Configurations

The token-stream design of §3.2.1 generalizes beyond the two-level, stride-1 family of Table 3.1
without any change to the backbone. A configuration is a set of context streams, each
parameterized by the triple **(spatial resolution, temporal stride, number of frames)**: the
coarse stream may tick at
stride $k_t > 1$ (one slot per $k_t$ frames, block-end anchors, $W/k_t$ target slots per
prediction window so the slot buffer keeps phase for any $k_t$ dividing $W$); a full-resolution
stream may itself be strided (temporal abstraction without a resolution change); and a third,
**top** stream at $2{\times}2$ (4 tokens per frame) extends the pyramid to depth 3 at factor 2
per level ($2{\times}2 / 4{\times}4 / 8{\times}8$). The top latent is a third tap of the frozen
autoencoder trained on the **frozen trunk** (§3.1.1), so the two lower levels remain bit-identical
and every measurement of Chapter 4 stays valid across the extension. Two systemic properties
distinguish this family from recurrent hierarchies of the CW-VAE class: (i) **depth is
backbone-capacity neutral** — all streams share one attention backbone; a third level adds only
per-level input/output parameters (+0.30 M on the 22.19 M predictor, +1.4%) and the frozen
autoencoder tap (1.25 M), measured and disclosed; and (ii)
every configuration's context cost is an exact token count, so families are comparable at
**matched budget** — the property that licenses the shape comparisons of §4.4 (simplified
configurations: `hier-s2-min` = 24 slots at stride 2 + 2 fine = 512 tokens; `strided-flat` = 11
full-resolution slots at stride 4 + 6 fine = 1,088; `hier3` = 48 top + 24 coarse + 9 fine =
1,152).

## 3.4 Memory-Horizon Testbed and Re-Entry-Stratified Evaluation

A model whose defining property is a *memory horizon* cannot be validated on benchmarks that do
not exercise one. On Moving-MNIST-class data nothing ever leaves the frame and returns: every
future frame is either fully determined by a short context or permanently unknowable, so there
exists no *hard-but-computable-with-memory* regime in which a long-memory mechanism could
demonstrate its value — this project's own null results on such data (Chapter 1, §1.3; Chapter 4)
trace to exactly this degeneracy. Existing memory-oriented benchmarks (Memory Maze and
TECO-Minecraft [20, 15], and the recent long-term spatial-memory evaluation protocols for world
models [21, 23, 22]) pose genuine
long-horizon problems but derive their re-entry events from 3-D egomotion in rendered
environments, where determining *which* pixels of a target frame were knowable from a given
context requires geometry processing at best — and their model scales exceed a single-GPU
budget. The instrument constructed here is the minimal fully analytic member of this family:
hardness labels are computed **exactly** from the recorded control track, and the complete
benchmark trains and evaluates on one 16 GB consumer GPU.

### 3.4.1 Cyclic World with Inertial Panning

Each clip is generated (module `scripts/gen_looping.py`) from a static **world**: a
$192 \times 64$ texture on a horizontal torus, composed of a smooth background (a sum of four
sinusoids with integer wavenumbers, hence seamlessly wrapping; intensities in $[0.15, 0.5]$) and
three MNIST digits max-composited at fixed world positions. A $64 \times 64$ **viewport** pans
horizontally; the observed clip is the sequence of viewport crops, so two thirds of the world
(128 of 192 columns, 67%) is off-screen at every instant.

The pan is a **momentum action walk**: at each frame the camera holds with probability
$p_{\mathrm{hold}}$, otherwise continues its heading with probability $p_{\mathrm{cont}}$ and
reverses otherwise (measured on the shipped data: $p_{\mathrm{hold}} \approx 0.10$,
$p_{\mathrm{cont}} \approx 0.90$). The executed action $a_t \in \{-1, 0, +1\}$ is recorded, and
the viewport advances by $4 a_t$ pixels:

$$
x^{\mathrm{vp}}_t \;=\; \bigl(x^{\mathrm{vp}}_{t-1} + 4\,a_t\bigr) \bmod 192 .
\tag{3.17}
$$

The momentum is the essential design element. A symmetric random walk diffuses too slowly to
circumnavigate the torus within a clip, so off-screen content would be either still inside a
short context window (determined) or never seen again (unknowable) — reproducing the degenerate
Moving-MNIST regime. Directed motion cycles the torus with loop period $\approx 192/4 = 48$
frames: a digit that exits the viewport re-enters roughly 48 frames later at a world position
fixed since frame 0. A model with memory spanning the loop period can predict the re-entry
exactly; a short-context model cannot — and 48 frames is precisely the coarse horizon
$H_{\mathrm{c}}$ of the developed system (§3.3.4), while the token-matched flat control's
18-frame horizon falls short of it. The testbed thereby *instantiates* the memory-horizon regime
as a controlled experimental variable.

Datasets (fixed, generated once, shared by every experiment): `loopmom_train` — 3{,}000 clips ×
100 frames (split by clip into 2{,}700 train / 300 validation); `loopmom_test` — 500 clips ×
200 frames, providing a held-out pool of 250 evaluation clips whose 200-frame duration supports
the multi-loop rollouts of Algorithm 3.1 (rollout evaluation uses 100 of them per arm). Test
latents are standardized with the training statistics (3.3).

**Figure 3.4 — LoopMom geometry and analytic reach band.**
The 64-pixel viewport traverses a 192-pixel toroidal static world. Saved positions make geometric
observability exact; the highlighted 14-to-48-frame band contains evidence available to the
long-reach package but outside the compared short flat window. The separate pose-mosaic inset is
marked as a privileged reference because it receives exact pose and raw stored pixels.
*Source:* `docs/figures/fig15_loopmom_observability_schematic.png`; deterministic source
`scripts/make_rft_core_schematics.py`.

### 3.4.2 Exact Re-Entry Difficulty Labels

Because the viewport trajectory (3.17) is a deterministic function of the recorded actions, the
set of world columns covered by any context window is computable in closed form — hardness is an
analytic label, not a heuristic. For a target frame $t$ and a reference window of the $m$
preceding frames, let $\mathrm{cov}_m(t) \subset \{0, \ldots, 191\}$ be the union of world
columns visible in frames $t-m, \ldots, t-1$, and let $F(t)$ be the set of world columns occupied
by the target frame's sprite pixels (detected by the foreground threshold of §3.4.3). The
**novel-foreground fraction** is

$$
\nu_m(t) \;=\; \frac{\bigl|\, F(t) \setminus \mathrm{cov}_m(t) \,\bigr|}{\bigl|\, F(t) \,\bigr|}
\;\in\; [0, 1],
\tag{3.18}
$$

the fraction of the target sprite that is geometrically unobserved in that context. It is an exact
observability label for the generator, not information-theoretic necessity: a shared digit prior
can still support a statistical guess. The protocol fixes the reference window at $m = 6$ frames for *all* systems regardless
of their actual context length ("novel-fg-at-6"), so every arm is stratified on the same hardness
axis, and defines three strata: **easy** ($\nu_6 = 0$ — the sprite was fully visible in the short
window), **mid** ($0 < \nu_6 \le 0.1$), and **hard re-entry** ($\nu_6 > 0.1$ — a substantial part
of the sprite is not geometrically visible in the short window). The single-step mechanism study of Chapter 4 uses
the analogous *novel-strip* label (the same construction restricted to the newly revealed leading
strip of the pan — the only genuinely generative region of a single predicted frame). This
stratification is what turns the trajectory-consistency criterion of §1.4.5 into a measurement:
the dissertation's headline quantities are error deltas *within the hard re-entry stratum* as a
function of rollout horizon.

### 3.4.3 Decoded Frame, Foreground and Object Metrics

Aggregate pixel metrics are excluded by protocol. On the archived CW-VAE Moving-MNIST population
(100 clips, 36 context frames, 964 open-loop targets), a deterministic all-black prediction
scores 0.7681 SSIM and 13.941 dB, while the trained CW-VAE's best-of-5-by-SSIM curve scores
0.5708 and 12.918 dB. The source dataset, archived curve and evaluator are hash-pinned in
`artifacts/cwvae_black_baseline/results.json`; record-level CW-VAE values were not retained, so
this is a same-population metric diagnostic without paired inference, not a budget-matched model
comparison. A benchmark on which the empty image wins its full-frame metrics is not evidence
about dynamics. All metrics below are therefore computed on decoded pixels through the frozen
fine decoder and reported with foreground, background, union, full-frame and component-sensitive
views.

**Table 3.4-A — Same-population Moving-MNIST metric sanity diagnostic.**

| Open-loop horizon | CW-VAE SSIM | All-black SSIM | CW-VAE PSNR, dB | All-black PSNR, dB |
|---|---:|---:|---:|---:|
| +1–48 | 0.6338 | 0.7679 | 13.944 | 13.943 |
| +49–192 | 0.5665 | 0.7679 | 12.625 | 13.942 |
| +193–964 | 0.5677 | 0.7682 | 12.909 | 13.941 |

- **Foreground MSE (fg-MSE).** Predictions and ground truth are decoded to $[0,1]$; the error is
  averaged over pixels where the ground-truth frame exceeds the threshold $0.5$. The threshold is
  testbed-calibrated and verified: the background texture occupies ≈0.18–0.41 while sprite
  strokes reach ≈0.99, so 0.5 isolates the digits (a conventional 0.1 threshold would classify
  the entire panning background as foreground).

- **Foreground sharpness (fg-sharp).** Per-image Laplacian energy on that image's foreground
  mask: $\sum (\nabla^2 \hat x)^2$ weighted by the mask, normalized by mask area, averaged over
  images. Deliberately *not* a batch-variance statistic: batch variance rewards inter-sample
  diversity, whereas per-image Laplacian energy rewards per-frame detail — which is what
  distinguishes a committed prediction from a mean-hedged blur.

- **Rollout records.** During Algorithm 3.1, every generated frame contributes a record
  (clip, horizon $h = t - H_{\mathrm{w}} + 1 \in \{1, \ldots, 152\}$, hardness $\nu_6$, fg-MSE,
  fg-sharp, and the persistence reference for the same frame). Analyses bin horizons as 1–8,
  9–24, 25–60, 61+ — the 9–24 bin brackets the flat control's 18-frame horizon boundary, and
  25–60 brackets the loop period.

- **Decoded-sample review** is a first-class protocol element: every evaluation saves panels of
  decoded frames (last context | prediction | ground truth) from the same samples that produced
  the scores. Twice during this research, scalar metrics concealed a qualitative failure (a slow
  fade to the mean; a blank-output collapse) that one glance at the panels exposed; the protocol
  therefore mandates the panels alongside the numbers.

### 3.4.4 Statistical Protocol, References and Boundaries

All system-versus-system comparisons are **paired**: evaluation draws use fixed seeds so every
arm sees identical (clip, target) pairs; stochastic samplers are re-seeded per batch with common
random numbers; and rollout records pair exactly across arms by (clip, horizon) thanks to the
universal 48-frame warm-up (§3.3.1). Headline deltas are reported with 95% confidence intervals
from a paired **clip-cluster** bootstrap (10{,}000 resamples), so all horizons belonging to one
clip are resampled together. The untouched frontier and protected robustness families use a
studentized max-statistic clip-cluster procedure for simultaneous coverage across their
registered contrasts. Intervals are computed within each stated hardness×horizon population;
single-step evaluation pools are sized at ≈768 samples
so the hard stratum retains power, and rollout evaluation covers 100 clips × 152 generated frames
= 15{,}200 records per arm. The headline arms (hier-long, flat-budget) are replicated over 2
seeds. Where a best-of-$N$ statistic appears it is an oracle *coverage* diagnostic (selection by
pixel-space fg-MSE against ground truth) and is always reported alongside the mean over $N$;
headline tables use best-of-1.

Every table is bracketed by references measured under the identical pipeline: the **persistence
floor** (copy the last context frame's fine latent, decode) scores fg-MSE 0.2367 / fg-sharp
0.0515 at the 4-frame offset — any model below that error has learned at least the
action-conditioned warp; and the **sharpness ceiling set by the frozen decoder**: raw
ground-truth frames measure fg-sharp 0.4785, but their autoencoder round-trip measures 0.2412,
so all model sharpness is read against ≈0.24 (roughly half the raw sharpness is spent in the
autoencoder before any prediction model acts), and a decoded output below ≈0.08 contains no
digit structure at all (the empirically calibrated collapse threshold used throughout the
mechanism study).

**Table 3.2 — Testbed and evaluation protocol summary.**

| Component | Value | Notes |
|---|---|---|
| World / viewport | 192×64 torus / 64×64 crop | 128 columns (67%) off-screen at any time |
| Content | background texture + 3 MNIST digits | fixed world positions; max-composited |
| Pan process | momentum walk, 4 px/frame | $p_{\mathrm{hold}} \approx 0.10$, $p_{\mathrm{cont}} \approx 0.90$ (measured); actions $\{-1,0,+1\}$ recorded |
| Loop period | ≈ 48 frames | $192/4$; sets the memory horizon; $= H_{\mathrm{c}}$ of hier-long |
| Train / test | 3,000×100 / 500×200 frames | clip-disjoint; test standardized with train statistics |
| Latent interface | frozen 2-level conv-AE, 1.85 M | coarse 16 ch @ 4×4, fine 16 ch @ 8×8, per-level decodable; fine val recon MSE 0.0020 |
| Standardization | per-channel z-score, per level | training-set statistics only; eq. (3.3) |
| Error metric | fg-MSE, threshold 0.5 | decoded pixels; background ≈0.18–0.41, sprite ≈0.99 |
| Sharpness metric | per-image masked Laplacian energy | GT 0.4785; AE round-trip ceiling 0.2412; collapse < ≈0.08 |
| Hardness axis | $\nu_6$ — novel-fg vs 6-frame window, exact | easy $=0$; mid $(0, 0.1]$; hard $> 0.1$; eq. (3.18) |
| Rollout | 48-frame GT warm-up; 38 steps × 4 frames | 152 generated frames/clip; 100 clips; horizons binned 1–8 / 9–24 / 25–60 / 61+ |
| Statistics | paired clip-cluster 95% CI; CRN for same target shapes | 10,000 resamples; simultaneous max-statistic families where registered; ≈768-sample single-step pools; 15,200 rollout records/arm; training seeds reported separately |
| Floors / ceilings | persistence fg-MSE 0.2367 / fg-sharp 0.0515 | k = 4 offset; identical pipeline for every arm |

Scope and limitations of the instrument are stated plainly, as required of a measurement
deliverable: motion is one-dimensional and piecewise-deterministic given the action, so the
generative burden is concentrated in the re-entry events by design; world content is static in
world coordinates, so the testbed does not probe memory of *dynamic* off-screen state; all
conclusions are at 64×64 grayscale and single-digit complexity; and the frozen decoder's
sharpness ceiling caps every downstream comparison. The analytic hardness labels that justify
the design would not transfer unchanged to 3-D egomotion benchmarks, which are positioned as the
natural next rung (Chapter 4 and General Conclusions).

## 3.5 Chapter Conclusions

1. In accordance with **Task 5** of the dissertation research, a mathematical model of
   multi-stream video-data generation in a compact continuous hierarchical latent state space
   has been developed. It supports streams with different temporal rates and spatial resolutions
   over one shared attention backbone. Its measured spatial implementation decomposes the state
   (3.1) into a coarse
   (macro) level — 16 channels at 4×4, independently decodable, standardized per channel — and a
   fine (micro) level — 16 channels at 8×8 — over a frozen two-level autoencoder. In that
   implementation the coarse stream economically packages long-range history: it is the stream
   whose context extends over the full 48-frame memory horizon, at a cost
   of 16 tokens per remembered frame against the fine level's 64 (eq. 3.16, a 4× per-frame and
   asymptotically ≈16× attention-cost advantage).

2. The inter-level transfer of context signals has been formalized as joint self-attention over
   a single token sequence of both levels (eq. 3.4–3.5), generalizing the cross-level
   cross-attention of the earlier design and enforcing hierarchical consistency continuously
   along the refinement trajectory (both levels denoised jointly under a shared noise level).
   The unidirectional-control principle with gradient detachment has been retained at the two
   boundaries where it is exact and load-bearing — between the frozen representation subsystem
   and the prediction core, and between past and future along the autoregressive recursion —
   while the internally contradictory strict level-ordering claim of the earlier formulation has
   been withdrawn in favor of the validated joint-generation formulation (§3.1.3).

3. The per-frame generation core has been formalized as window diffusion in the standardized
   latent space: v-parameterized (3.10) denoising of 4-frame two-level windows under a cosine
   schedule (3.9) with per-target-frame action conditioning, trained on single windows (3.11)
   and sampled by 16-step deterministic DDIM (3.12)–(3.13) at a fixed cost of 4 network
   evaluations per generated frame. The window length equals the recursion advance, which makes
   the training and rollout context distributions identical by construction — the property that
   permits clean-context training without rollout-time distribution shift of the context layout.
   A capacity-matched regression variant on the identical backbone (§3.2.5) completes the model
   family for the mechanism claims whose evidence is presented in Chapter 4.

4. The autoregressive rollout algorithm with ring-buffer memory (Algorithm 3.1) has been
   developed and its resource profile derived: the inter-step state is constant at
   $48 \cdot 256 + 6 \cdot 1024 = 18{,}432$ numbers (3.15) independently of the rollout horizon,
   per-step cost is fixed by the constant 1472-token sequence, and block-causality holds at
   window granularity by construction rather than by masking (§3.3.2). The re-derivation (§3.3.3)
   removes the complexity-claim inconsistency of the earlier chapter version, and the earlier
   reversible-inner-loop proposal is disposed of honestly: by the three-axis taxonomy of
   Chapter 2 that axis cannot outperform checkpointing, and the selected diffusion core requires
   no gradient through its sampler — reversibility remains a contribution on the temporal-rollout
   axis where it pays.

5. In accordance with **Task 6**, the measurement instrument of the intelligent information
   technology has been formalized: the momentum looping memory-horizon testbed (3.17) with
   exactly computable re-entry hardness labels (3.18), foreground-masked per-image metrics with
   verified thresholds and decoder-set ceilings, mandatory decoded-sample review, and a paired
   bootstrap protocol with common random numbers and a universal warm-up that pairs every record
   across systems. The instrument operationalizes the trajectory-consistency criterion of
   Chapter 1 into an analytic, statistically disciplined measurement procedure, and its scope
   limits are stated as part of the deliverable.

6. The complete pre-registered experimental design has been fixed within the model chapter
   (Table 3.1): the developed system (hier-long), the memory ablation (hier-short), the
   token-budget-matched flat control (flat-budget, 1152 tokens = 18-frame horizon), the
   window-matched flat exhibit (flat-long, 2.67× budget), and the regression control (reg-long) —
   with the decisive comparison, hier-long versus flat-budget at equal token cost, constructed so
   that the hierarchy's memory horizon (48 frames ≥ loop period) and the flat horizon (18 frames
   < loop period) straddle the testbed's re-entry structure. The experimental verification of the
   pre-registered hypotheses on this design — including the honest fallback conclusions specified
   in advance — constitutes §§4.2–4.4 of Chapter 4.

The results of this chapter form the methods and model section of [A4]. Its final numbered
bibliographic position and any related conference theses are publication-administration fields,
not scientific dependencies of the chapter.

---

<div style="page-break-after: always;"></div>

# Chapter 4 — Experimental Evaluation and Practical Application

This chapter performs the comprehensive experimental verification required by **Task 7** of the
dissertation research: the developed methods and the intelligent information technology are
evaluated on the testbed family formalized in §3.4, under the pre-registered protocol of
Table 3.1/Table 3.2, with every quoted comparison paired, stratified by the exact re-entry
hardness label, and interval-estimated by the clip-level cluster bootstrap. The order of the
sections is deliberate. The generation-head study (§4.2) is settled first, because every
subsequent table depends on knowing how much of a system arm's score is owed to its head rather
than to its memory design. The matched-token-budget memory study (§4.3) is the headline result of
the dissertation. The spatial-abstraction factorial (§4.4) isolates the factor that every
hierarchical arm otherwise confounds. The temporal-capability boundary study at the second
operating point (§4.5) closes the capability line of the experimental program with an honestly
bounded negative and the methodology lessons it produced. The generalization study (§4.6) then
carries the configuration selected by §§4.2–4.4 onto the standard and synthetic benchmark
suites — its measured components include the standard Moving-MNIST negative-control bridge and
the corrected persistent-object long Moving-MNIST external boundary. §4.7 documents the
implementation of the results,
and §4.8 concludes the chapter.

Two standing conventions of the chapter, fixed by the validity audit of the experimental campaign
and stated once here: (i) the long-horizon evaluation protocol is a **48-frame ground-truth
warm-up followed by 152 autoregressive frames** (38 scheduler steps × 4 frames) — all text says
"152 AR frames", never "200-frame rollout"; (ii) all confidence intervals quoted in this chapter
are 95% paired **clip-level cluster bootstrap** intervals unless a disclosure states otherwise
(§4.1.5).

## 4.1 Data, Hardware and Measurement Protocol

### 4.1.1 Testbed Family: Two Operating Points

All experiments of §§4.2–4.5 run on the memory-horizon testbed family of §3.4 at two operating
points.

**Operating point A (primary)** is the momentum looping world *loopmom* formalized in §3.4.1: a
$192 \times 64$ toroidal world observed through a panning $64 \times 64$ viewport at 4 px/frame
($x^{\mathrm{vp}}_t = (x^{\mathrm{vp}}_{t-1} + 4a_t) \bmod 192$; measured
$p_{\mathrm{hold}} \approx 0.10$, $p_{\mathrm{cont}} \approx 0.90$), so ~67% of the world is
off-screen at any instant and off-screen content re-enters with a loop period of ≈48 frames.
Datasets: 3,000 training clips × 100 frames and 500 test clips × 200 frames, clip-disjoint
(§3.4.1); the frozen two-level autoencoder interface, standardization, hardness labels, metrics,
and floors are exactly those of §§3.1.1 and 3.4.2–3.4.4.

**Operating point B (capability regime)** is *hgrid3*: the same generator scaled to a 768-px
world (12 digits, speed 4 px/frame, near-ballistic circulation), whose **effective loop** is
$L_{\mathrm{eff}} \approx 213$ frames (eq. 4.1, §4.5.2) — deliberately beyond any affordable flat
context window, which is what makes it the *capability* rather than *efficiency* operating point.
Operating point B carries its own frozen autoencoder (fine reconstruction MSE 0.0024, coarse
0.0059) and its own re-measured references (Table 4.1); its regime properties are verified before
any model run: median re-entry gap 192 frames, 204 beyond-48 re-entry frames per clip, warm-up
world-coverage 88.5%, and viewport-crop uniqueness 0.004–0.010 (the positional-landmark property
whose destruction is diagnosed in §4.5.1).

**Table 4.1 — Reference floors and ceilings, measured under the identical pipeline per operating
point (never mixed across operating points).**

| Reference | Op. point A (loopmom) | Op. point B (hgrid3) |
|---|---:|---:|
| Persistence floor, fg-MSE (single-step, k = 4) | 0.2367 | 0.2381 |
| Persistence floor, fg-MSE (rollout protocol) | 0.2552 | — |
| Persistence fg-sharp (k = 4) | 0.0515 | — |
| AE round-trip fg-MSE ceiling | — | 0.0108 |
| AE round-trip sharpness ceiling | 0.2412 | 0.2084 |
| Raw ground-truth sharpness | 0.4785 | 0.4218 |
| Collapse threshold (no digit structure below) | ≈0.08 | ≈0.08 |
| Frozen AE reconstruction (fine / coarse MSE) | 0.0020 / — | 0.0024 / 0.0059 |

**Figure 4.A — LoopMom geometry and the registered reach band.**
*Content:* toroidal world, moving viewport, persistent digits and the pair-derived
`14 < dt <= 48` population. The privileged pose-indexed raw-memory reference is drawn
separately from token-budget-matched neural packages.
*Source:* `docs/figures/fig15_loopmom_observability_schematic.png` and vector PDF; generated
deterministically from the frozen E2 geometry by `scripts/make_rft_core_schematics.py`.

### 4.1.2 [A2]

The experimental validation of the reversible-transformer method and time-agnostic
backpropagation (§§2.3–2.4) was published in [A2] on procedurally synthesized **2-D rotating
shapes** at $64 \times 64$ resolution. The suite controls the temporal law—constant, changing,
oscillating, interrupted, and combined rotation—and therefore isolates the effect of prediction
depth and temporal discontinuity. The published regime uses a 12-frame context, grows the
prediction depth from one to five frames during 30,000 training steps, evaluates 2,000 independently
generated sequences, and trains all compared models with Adam at learning rate $10^{-5}$ on one
NVIDIA RTX 4060 Ti 16 GB. The memory-scaling measurements use batch 32. These values replace the
unsupported old-draft claims of AdamW with cosine decay, 100–300 epochs, and A100 80 GB hardware.

The **3-D polycube** and entropy-tokenization material belongs to [A1], not [A2]. Moving MNIST,
MMNIST-CIFAR, TaxiBJ, and KTH are likewise not reported as [A2] experiments: Moving MNIST and
TaxiBJ are separate dissertation bridges (§§4.6.2 and 4.6.5), whereas MMNIST-CIFAR and KTH remain
explicitly unexecuted extensions. This separation preserves the content and scope of the existing
publications instead of retroactively enlarging them.

### 4.1.3 Bridge to the OpenSTL/Moving-MNIST Benchmark

**[MEASURED 2026-07-06 — full results, table, and reading in §4.6.2 (Table 4.8).]** For
continuity with the literature, the developed stack is additionally evaluated on the standard
OpenSTL Moving-MNIST suite [74]. The pre-registered expectation, recorded before the
runs: mid-pack aggregate MSE/SSIM against the OpenSTL leaderboard — aggregate metrics on a
≈93%-black-background benchmark reward mean prediction (this project's own measurement: a
constant all-black prediction outperforms a trained CW-VAE [13] on aggregate SSIM/PSNR,
§3.4.3) — together with competitive perceptual (LPIPS) behavior and an autoencoder round-trip
ceiling row that decomposes decoder error from dynamics error. The purpose of the bridge is
connectivity to the standard suite, not a leaderboard claim; the memory-horizon claims of this
chapter cannot be tested on Moving MNIST at all, for the degeneracy reasons established in §3.4.
The measured outcome (§4.6.2) confirmed the connectivity, ceiling-row, and
perceptual-versus-persistence clauses, **refuted the "mid-pack aggregate MSE" clause** (180.91
against the deterministic specialists' 22.06–29.80 — reported as a refutation, per the standing
convention of §4.2.3), and returned the predicted flat ≈ hierarchical null as the negative
control of the memory-horizon claim.

### 4.1.4 Hardware and Software Environment

All experiments of the [A4] campaign (§§4.2–4.5) were performed on a single consumer-grade GPU
(NVIDIA GeForce RTX 4060 Ti, 16 GB) — a deliberate constraint of the research design: the
resource-efficiency claims of Chapter 2 are made *for* this class of hardware, and the complete
benchmark (data generation, autoencoder training, all system arms, all evaluations) reproduces on
one such device. The software complex is implemented in Python/PyTorch (§4.7; the module map is
given in the appendix **[AUTHOR: appendix letter]**). The memory-consumption measurements quoted
in Chapter 2 (Table 2.3: BPTT 12.2 GB at rollout length $K = 40$ and OOM at $K = 120$ versus the
developed method's ≈1.1–1.2 GB through $K = 160$; the inner-loop panel behind the 0.26%-measured no-go
result) were performed in this same environment, on the implemented stochastic hierarchical
recurrent core, and constitute part of the experimental program of this chapter even though their
tables live with the theory they verify.

Training protocol of the rollout arms (operating point A): 30,000 optimizer steps, batch 16,
bf16, EMA decay 0.999, evaluation on EMA weights (§3.2.3); the flat-long arm is the disclosed
exception (§4.3.4). The DiT backbone of every arm is the §3.2.2 configuration (d = 384, depth 8,
6 heads; 22.19 M parameters).

### 4.1.5 Statistical Protocol and Threats to Validity

The evaluation protocol is the §3.4.4 instrument, hardened by a six-surface adversarial validity
audit of the campaign (data/splits, rollout protocol, gate protocol, energy-model protocol,
autoencoder provenance, metrics/statistics; 24 findings, 2 confirmed major, both fixed and
recomputed the same day). The audit's material corrections and standing disclosures are part of
the protocol definition and are stated here once:

1. **Cluster-robust intervals.** Rollout records are 100 clips × 152 consecutive AR frames with
   lag-1 autocorrelation +0.86; frame-iid bootstrap intervals are 2–7× too narrow. All quoted CIs
   are therefore paired clip-level cluster bootstrap intervals. Under the corrected intervals,
   three secondary bin verdicts of the campaign flipped to null; **every headline verdict
   survived**. Records whose ground-truth frame contains no foreground (3,263 of 15,200 = 21.5%,
   where fg-MSE would silently fall back to full-frame MSE) are excluded from all stratified bins
   and reported as their own background-only row.

2. **Corrected hardness anchoring.** An audit-confirmed bug in the mechanism-study hardness
   labels (the target-anchored coverage window counted the k − 1 = 3 bridge frames as covered,
   mislabeling 11–14% of hard samples as easy; hard fraction 0.19 buggy → 0.27 correct) was fixed
   and Table 4.2 recomputed from the saved checkpoints with a single raw-ground-truth metric
   reference for all rows. The memory-gate and rollout evaluations always used the correct
   anchor. All hardness labels are computed from ground-truth geometry, independently of any
   model and identically across arms.

3. **Metric scope disclosures.** fg-MSE scores only ground-truth-sprite pixels; hallucinated
   off-sprite content is invisible to it — it is therefore never interpreted alone but jointly
   with foreground sharpness, decoded-sample review (§3.4.3), and the complete-metric evidence
   artifacts, which include full-frame, union-mask and false-positive-sensitive endpoints.
   Per-horizon *sharpness* series are chunk-batch means duplicated per record — valid for overall
   means and trend curves, never for per-bin or per-clip readings; every sharpness-versus-horizon
   statement in this chapter is so scoped.

4. **Pairing and replication conventions.** All arm-versus-arm comparisons share evaluation
   clips and horizons; sampling noise is common only when target tensor shapes match. Cross-shape
   cells retain record pairing but not elementwise CRN (unbiased, with less variance reduction).
   "2 seeds" for the single-step
   memory gate means two *training* seeds sharing one CRN evaluation set (by design, for exact
   pairing). At operating point B the evaluation adds the pre-registered
   **dt-since-last-visible** stratifier (per-record time since the target region was last in
   view, computed from the recorded trajectory; bins dt ∈ (48, 96], (96, 192], > 192, crossed
   with hardness) and a warm-up-SEEN restriction, so that gates are read only on re-entries of
   content that was answerable by construction.

5. **External validity.** The testbed is purpose-built and synthetic; digit assets are shared
   across splits (texture familiarity only — re-entry memory requires clip-specific trajectory
   content, which does not transfer across clips), while trajectories and world layouts are
   split-disjoint. All claims are scoped to the memory-horizon regime the testbed
   operationalizes; the OpenSTL bridge (§4.1.3, measured in §4.6.2) anchors short-horizon
   behavior on a standard benchmark. Generation seeds of the shipped dataset files were not recorded (measured
   properties are stated as measured; a frame-hash duplication check found no cross-file clip
   duplication).

## 4.2 Matched-Compute Mechanism Study: Selecting the Generative Core

The claim defended in this section is deliberately conditional: **which prediction head wins
depends on the inference regime; there is no one best head.** Three results establish this — the
single-step coverage study (§4.2.2), its rollout reversal (§4.2.3), and the two-sided study of
the energy-based head (§4.2.4). The section thereby supplies the empirical selection evidence
promised in §3.2 (the diffusion core) *and* the honest boundary of that selection.

**Figure 4.1 — Three prediction heads in the shared context testbed.**
Direct regression predicts one target latent, diffusion predicts a denoising velocity and commits
through DDIM sampling, while the energy head learns a scalar landscape traversed by candidate
descent. The diagram separates objective and inference from the shared context question.
*Source:* `docs/figures/fig24_generation_objectives.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`.

### 4.2.1 Shared Mechanism Testbed

All Part-A predictors are transformers of one shape (width 256, 6 blocks, 8 heads) over one token
layout — 16 coarse plus 64 fine tokens per frame with joint attention over both levels' tokens
(the §3.1.2 conditioning mechanism) — with parameter counts 7.27 M (regression), 7.53 M
(diffusion), and 7.28 M (energy-based): the differences are the mechanism-specific heads and
embeddings only. Context is 6 frames of both levels; the target is the frame k = 4 steps after
the last context frame; the four bridging actions are embedded through the same conditioning path
in every arm. Training is single-step and leakage-free (ground-truth context and target; no
prediction fed back); batch 48, fp32; the regressor trains 6,000 steps (its loss plateaus at
0.096), the diffusion arm 8,000 steps. Latents are standardized per channel and per level with
training-set statistics — the §3.1.1 protocol requirement, and a finding in its own right: with
raw latents (per-channel scale ≈0.12 coarse / ≈0.16 fine) the diffusion arm *underperformed* the
regressor, because unit-scale noise buried the signal mid-schedule and sampling initialized 4–7×
off-scale; the one-line standardization fix reversed the mechanism ranking. The failure mode is
silent — losses fall, samples look plausible, nothing errors — which is why standardization is
elevated to a protocol requirement rather than an implementation detail.

Inference budgets are matched in network function evaluations (NFE), not step counts: one DDIM
step = 1 forward pass; one energy-descent step = forward + gradient ≈ 2 forward passes; one
regression sample = 1 forward pass (evaluation keeps dropout 0.1 active, so N stochastic passes
yield N samples — MC-dropout, making best-of-N comparable across arms). Best-of-N selection is
performed in decoded pixel space on foreground MSE against ground truth — an **oracle coverage
diagnostic** (§3.4.4), always reported alongside the mean over N.

### 4.2.2 Single-Step Oracle-Coverage Regime: Diffusion Wins Every Column

Table 4.2 reports the corrected mechanism comparison (single-step, k = 4, best-of-8, NFE 16;
six evaluation batches of 48; hard split = novel-strip > 0, covering 27% of samples; references
per Table 4.1).

**Table 4.2 — Matched-capacity head comparison, single-step (corrected hardness anchor, single
raw-GT reference).**

| Head | fg-MSE (all) | easy | HARD | mean-over-8 | fg-sharp |
|---|---:|---:|---:|---:|---:|
| Regression (MC-dropout) | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| Diffusion (v-pred, DDIM) | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** |
| Persistence floor | 0.2367 | — | — | — | 0.0515 |

**Table 4.3 — Matched-NFE control (best-of-8 fg-MSE).**

| NFE per sample | Regression | Diffusion |
|---:|---:|---:|
| 4 | 0.0680 | 0.0546 |
| 8 | 0.0680 | 0.0514 |
| 16 | 0.0680 | 0.0503 |

Three observations. First, both heads are far below the persistence floor on every column: the
action-conditioned warp is learned by everything, as expected where the easy region is
deterministic given the action. Second, the separation concentrates exactly where the hardness
stratification says it should: on the easy bin diffusion's advantage is 0.011 absolute (0.0232 →
0.0120), on the genuinely generative HARD bin it is 0.026 (0.1808 → 0.1549), and diffusion is
sharper (0.1886 vs 0.1591 against the 0.2412 ceiling). Third, the advantage is not an
inference-budget artefact (Table 4.3): the regressor is budget-flat at 0.0680 while diffusion
already leads at NFE 4 and improves monotonically. The regressor's mean-over-8 ≈ best-of-8 gap is
near zero — it has *averaged over* the possible futures rather than distributing probability mass
across them; diffusion's best-of-N gap (0.0640 → 0.0512) is the coverage signature of a head that
commits to distinct hypotheses per sample. This is a controlled replication of the
"regression averages, sampling-based generation commits" finding of the stochastic
video-prediction literature [75, 76, 77] under matched capacity, matched conditioning, matched inference
budget, and exact hardness stratification — and its protocol scope is the point: best-of-8
selection is an oracle coverage regime.

**Figure 4.2 — Single-step regression and diffusion predictions.**
Each row shows last context, prediction and ground truth on identical inputs. The panels are
descriptive examples of the registered single-step regime; the aggregate and paired results,
not a selected row, determine the comparison.
*Source:* `docs/figures/fig29_mechanism_single_step_samples.png`, composed without pixel editing
from `logs_rhcp/step1_mech_clean_{reg,diff}.png` by
`scripts/make_dissertation_missing_figures.py`.

### 4.2.3 Rollout Reversal: Regression Wins Best-of-1

The pre-registered expectation for the rollout campaign was that the regressor, rolled out
autoregressively, would fade toward the mean while diffusion held sharpness. The measured outcome
**refutes** this expectation, and the reversal is reported as such, not reframed: under the
deterministic best-of-1 rollout protocol (§4.1), **the regression arm beats the hierarchical
diffusion arm at every horizon band and every hard bin at equal sharpness** — reg-long vs
hier-long: fg-sharp 0.0741 vs 0.0729; ALL fg-MSE 0.1682 vs 0.1848; direct paired cluster CI on
the ALL delta [+0.0110, +0.0227]. The same holds at the matched flat context: reg-flat 0.2015 vs
flat-budget (diffusion) 0.2038.

The two results reconcile without contradiction once the protocols are stated side by side: the
Table 4.2 diffusion win is single-step **best-of-8** — oracle coverage switched on; the rollout
result is **best-of-1** — coverage off, and every sampled hypothesis is fed back and compounded
over 152 frames. Diffusion wins the coverage regime; regression wins the deterministic best-of-1
rollout regime. Any claim of the form "head X is better" that omits the regime is, on this
evidence, wrong.

### 4.2.4 Two-Sided Energy-Head Study: A Characterized Negative Result

The third candidate head is energy-based descent: Energy-Based Transformers [39] train a scalar energy $E(\text{context}, \text{candidate})$ — reviewed in
§1.3.3 as a global consistency criterion — and generate by gradient descent on the candidate. The
mechanism is attractive for this dissertation's problem precisely because descent steps are a
natural "think-longer" knob for hard re-entry frames. The study below is *two-sided* in that it
decomposes the observed failure into two separable capabilities: whether the trained energy
landscape **orders** candidates correctly (target below mean below noise), and whether descent
from the prescribed initialization can **reach** a structured output. The headline result is that
these dissociate.

**Setup: the official implementation, imported unchanged.** This dissertation's own
reimplementation of the mechanism, trained in the §4.2.1 harness, collapsed to a blank near-mean
output; a reimplementation failure is uninformative, so the study removes the author's code from
the loop entirely. The control module of the software complex imports the reference repository's
model classes unmodified (RMSNorm blocks, rotary positions, causal autoregressive attention, the
prepended descent-step time-embedding token, the bias-free linear energy head, the causal replay
buffer), and transcribes the reference MCMC descent control flow line-for-line (random-noise
initialization, per-step step-size randomization, Langevin noise, gradients with respect to the
pre-projection frame embeddings, single-graph and truncated variants per recipe). Two adaptations
were unavoidable and are flagged: (i) *frame-as-token* — the reference flattens each frame's
latent into one token via a learned down-projection; the identical scheme is applied to the
concatenated standardized latents of §3.1.1 ($16 \cdot 4 \cdot 4 + 16 \cdot 8 \cdot 8 = 1{,}280$
dimensions), with descent performed in the original 1,280-d space exactly as in the reference;
(ii) an additive learned *action embedding* on the candidate tokens (the reference video task is
action-free; this testbed is action-conditioned). One recipe number is re-tuned: the descent step
size $\alpha$, whose published values are calibrated to the reference latent space's gradient
scale, is recalibrated to this latent scale by the reference's own movement principle
($\alpha \cdot |\nabla E| \cdot n_{\text{steps}} \approx$ the initialization-to-target distance).

Both published training recipes are run at the paper's "xxs" video shape — width 384, 6 layers, 6
heads, FFN multiplier 4, context 16 frames; 14.66 M parameters in the frame-token instantiation —
with the paper's scheduler shape, peak learning rate $4 \times 10^{-4}$, weight decay 0.01,
gradient clip 1.0, effective batch 128 (micro-batch 32 × 4 accumulation), 20,000 optimizer steps,
fp32. The **S2 recipe** uses a single training landscape with randomized extra final-landscape
steps, truncated second-order graph, Langevin noise, a replay buffer, and fixed recalibrated
$\alpha = 15.97$; the **S1 recipe** (the configuration behind the paper's published video
generations) uses two descent steps with detach between them, a second-order graph on every step,
no replay, no training-time Langevin noise, and a learnable $\alpha$ (recalibrated initialization
23.95). Every 1,000 steps, held-out probes record: the energies $E(\text{target})$,
$E(\text{mean latent})$, $E(\text{random noise})$ on the final landscape; the cold-descent
output's latent distance to target $d_{\mathrm{tgt}}$ (the mean-to-target distance is 0.964 in
the same units); and the decoded output's foreground sharpness (collapse threshold ≈0.08; ceiling
≈0.24). For S1, the paper's own inference recipe is additionally run at each probe (best-of-4 by
minimum energy, 4 descent steps, inference $\alpha/4$, Langevin noise, probed at both the
current-$\alpha$ and initial-$\alpha$ readings).

**Figure 4.3 — Energy head and gradient-descent generation loop.**
The tested head scores a 1,280-dimensional per-frame candidate together with context and actions.
Generation repeatedly updates the candidate using its energy gradient and the registered noise
rule. This makes landscape ordering and reachability by descent separately testable.
*Source:* `docs/figures/fig25_ebt_generation_loop.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`.

**The S2 recipe: a stably mis-ordered landscape.** The S2 run converges (training loss settles at
≈0.597 from roughly step 7,000 to the end) — and its landscape is stably wrong. At **every one of
the 21 probes** from step 0 to step 20,000, $E(\text{mean}) < E(\text{target})$: the batch-mean
latent has strictly lower energy than the true target, with a stable gap of ≈0.6–1.2 energy units
from step 3,000 onward. Cold descent behaves accordingly: $d_{\mathrm{tgt}}$ stays flat at
1.26–1.29 (against the mean's own 0.964 — the output is no closer to the target than the mean
is), and decoded foreground sharpness never leaves the collapse regime (0.008–0.014 across all
probes, versus the 0.08 structure threshold). Twenty thousand converged steps at the paper's
shape, batch semantics, and scheduler do not repair the ordering; the objection that the earlier
negative was an under-training artefact is thereby closed for this recipe. One disclosure: the
batch-mean latent has roughly eight times lower norm than an individual target, so a norm
preference could in principle carry part of the ordering; two probe features rebut a pure norm
artefact — $E(\text{random noise})$ sits far above both throughout, and the S1 recipe flips the
*same* probe to the correct ordering, which a norm effect alone cannot do.

**The S1 recipe: ordering repaired, generation still out of reach.** The S1 run is the
informative one. Its loss falls further (0.361 at 20,000 steps, still improving), the learnable
$\alpha$ drifts upward (23.95 → 33.27), and — unlike S2 — the landscape *becomes correctly
ordered*: $E(\text{target})$ is strictly lowest at **all fifteen consecutive probes from step
6,000 to 20,000**, with a target-below-mean gap ranging from +0.22 to +3.09 energy units across
those probes (the exact flip-onset step rests on one fixed 64-sample probe batch; the
fifteen-consecutive-probe persistence is the claim, not the onset). By the ordering criterion
that indicted S2, S1's energy is *right*. Generation still never materializes: cold descent from
the prescribed random-noise initialization stays at $d_{\mathrm{tgt}} \approx 1.12$–$1.17$, with
foreground sharpness creeping from 0.011 to a maximum of 0.035 — a real but small trend that
remains a factor of ≈2.3 below the structure threshold and ≈7 below the decoder ceiling. The
paper's own inference recipe does no better: $d_{\mathrm{tgt}} \approx 1.19$–$1.24$ and fg-sharp
0.010–0.015 at every probe and both $\alpha$ readings (final probe: 1.207 / 0.0127 at the
current-$\alpha$ reading; 1.233 / 0.0120 at the init-$\alpha$ reading). A correctly ordered
landscape, probed under the publication's own sampler, produces no structured output at this
scale.

**Decomposition: ordering × reachability.** Combining the two runs with the positive-control
probes gives a clean two-factor decomposition. *(a) Landscape ordering is recipe-dependent:* S2
never orders the landscape correctly (21/21 mean-lowest probes at converged loss); S1 orders it
correctly from step 6,000 onward (15/15 target-lowest). The choice between the two published
training recipes decides whether the energy function even prefers the truth. *(b) Descent
reachability from noise fails under both, and is not bought back by inference-time compute.* In
positive-control experiments on the same official model (copy and shift tasks), the trained
landscape is correct by every probe — on copy, $E(\text{target}) = -65.6$ vs mean $-57.2$ vs
noise $-14.8$, and an energy-conditioning matrix identifies the matching context for 8/8 targets
with an off-diagonal gap of +26.4 — yet cold descent still lands blank (fg-sharp 0.030 on copy,
0.016 on shift). Sweeping cold-descent steps 2 → 128 leaves $d_{\mathrm{tgt}}$ flat
(1.131–1.149); warm-starting *at the target* and descending does not hold it (at the operating
step size the output drifts to $d_{\mathrm{tgt}} 0.41 \to \approx 0.55$, settling in a shallow
basin between target and mean — the target is at best a weak attractor along the descent
dynamics); the minimum-energy sample among 16 cold descents remains collapsed (fg-sharp 0.0145);
and a persistence-initialized descent also fails to move toward the target ($d_{\mathrm{tgt}}$
0.835, fg-sharp 0.046, versus 0.348 and 0.088 for a plain regressor on the same task). The two
factors are separable because S1 fixes (a) while leaving (b) intact: what is observed at
16 GB-reachable scale is not "the energy model learns the wrong criterion" simpliciter — with the
generation recipe it learns a usefully ordered criterion whose minima are nonetheless unreachable
by the prescribed sampler from the prescribed initialization. This is consistent with theoretical
accounts of entropy collapse in gradient-descent sampling [78], cited as context being instantiated, not as a discovered mechanism.

**Scope and non-claims.** This study tests a capability the EBT publication does not claim at a
scale it did not run: the paper's video-section claims concern validation-loss scaling and
energy-as-uncertainty, not sharp generation, and its published video generations come from a
larger model class trained on natural video through a large pretrained autoencoder on multi-GPU
hardware. Nothing here is a reproduction failure, and nothing here contradicts the published
claims; published small-model positives in other domains [79] are
likewise untouched. What the study establishes is a **capability boundary with a mechanism
attached**: at the scales reachable in this work (Table 4.4), the energy-descent head does not
convert even a correctly ordered landscape into committed video predictions — while the diffusion
control at roughly half the parameter count (7.53 M vs 14.66 M), on the same frozen latents and
the same standardization, shows recognizable digit structure by 1,500 optimizer steps and at
convergence commits to sharp, correctly located digits (Table 4.2). Whatever resource the
energy sampler is missing at this scale, diffusion does not miss it. Residual limitations are
stated: the scaled runs remain a factor of 3–5 below the reference step budgets and 2 below its
batch size; the S1 sharpness trend (0.011 → 0.035), while far below structure, is non-zero, and a
longer run could in principle continue it — the claim is scoped accordingly.

**Table 4.4 — Configuration gap between this study's fair-scale runs and the reference training
configurations (the extrapolation-risk record).**

| | This study (S2 / S1 arms) | Reference S2 | Reference S1 |
|---|---|---|---|
| Model shape | xxs: d 384, 6L, 6H, FFN 4× (14.66 M with 1,280-d frame tokens) | xxs | xxs |
| Context | 16 frames | 16 frames | 16 frames |
| Steps (warm-up) | 20,000 (5,000) | 60,000 (10,000) | 100,000 (10,000) |
| Effective batch | 128 fresh + 64 replay / 128 fresh | 256 fresh + replay | 256 fresh |
| Step size $\alpha$ | fixed 15.97 / learnable, init 23.95 (recalibrated to latent scale) | fixed 3,000 | learnable 30,000 |
| Data | synthetic 64×64 grayscale; 2,700 clips × 100 frames | natural video, 224×224, pretrained-AE latents | same |
| Hardware | 1× RTX 4060 Ti 16 GB, fp32 | 4 GPUs/node | 4 GPUs/node |

**Figure 4.4 — Energy-landscape ordering under two recipes.**
S1 changes to the desired target-below-mean ordering after approximately 6,000 updates; S2 keeps
the batch mean below the target through 20,000 updates. Both series include the random-candidate
reference.
*Source:* `docs/figures/fig26_ebt_landscape_ordering.png`, parsed deterministically from
`logs_rhcp/ebt_scaled_{s1,s2}.log` by `scripts/make_dissertation_missing_figures.py`.

**Figure 4.5 — Foreground sharpness after cold energy descent.**
Both recipes remain below the preregistered 0.08 structure threshold and far below the
approximately 0.24 decoder reference, even after S1 learns the desired energy ordering.
*Source:* `docs/figures/fig27_ebt_descent_sharpness.png`, parsed deterministically from
`logs_rhcp/ebt_scaled_{s1,s2}.log` by `scripts/make_dissertation_missing_figures.py`.

### 4.2.5 Generative-Core Conclusion

The head ranking is **regime-dependent**: diffusion for the single-step best-of-N coverage
regime, regression for the deterministic best-of-1 rollout regime, energy-based descent for
neither at 16 GB-reachable scale (a two-sided characterized negative: ordering recipe-dependent,
reachability failing under both recipes and under the paper's own sampler). The memory studies
that follow therefore use **both** viable heads for the central matched-budget memory comparison
(§4.3.5), while §4.4 estimates and reports package preference separately by head and training
seed. The central reach result is not tied to one head; the broader ordering of temporal and
spatial packages is explicitly head- and seed-conditional.

## 4.3 Single-Step Memory Gate and Matched-Budget Rollout Study

This section tests the central pre-registered prediction of Table 3.1: at an equal
context-token budget, does a hierarchical allocation (long coarse memory + recent fine window)
beat a flat allocation when only the former reaches memory-relevant history? The measured studies
then bracket the tested budgets where that prediction holds and where full-resolution fidelity
instead wins; they do not assume a continuous or unique transition.

### 4.3.1 Preliminary Gate: Recall Emerges Only after Sufficient Optimization

Before any rollout machinery, the single-step memory gate asked whether a long *coarse* context
helps at all: coarse context 48 versus 6 at fixed fine context 6 (1,232 vs 560 tokens including
the target slots), best-of-8, stratified by the exact hardness label. The result is an
emergence-with-budget pattern: at 6,000 steps the HARD-bin delta is null on both training seeds
(+0.0014 [−0.0004, +0.0033] and +0.0008 [−0.0009, +0.0025]); at 20,000 steps it is a decisive win
on both (+0.0135 [+0.0080, +0.0196] and +0.0143 [+0.0083, +0.0208]; hard-bin absolute error
0.1053 → 0.0918 on seed 0 — a ≈13% relative reduction), with sharpness tied (≈0.188 both arms):
the win is retrieval, not sharpness. Four disclosures attach: "2 seeds" means two training seeds
sharing one CRN evaluation set (§4.1.5); these evaluation windows are near-independent rather
than fully clustered, so the audit's worst-case design-effect correction widens the 20k hard-bin
CIs to ≈[+0.005, +0.022] (seed 0) and ≈[+0.006, +0.023] (seed 1) — the win survives; **each
budget is its own complete, fully annealed training schedule** (a 6,000-step run and a
20,000-step run, budget-matched within each contrast), so the compared quantity is the
optimization budget with schedule shape a bundled covariate — and the 6,000-step arms were the
better-converged endpoints of their own schedules, which works against a mid-anneal objection;
and the 20,000-step budget was a **disclosed post-null follow-up** — the originally registered
read at 6,000 steps returned its null, which is on the record, and the extension was then run
and replicated on both seeds, so the claim rests on the twice-replicated 20k result, not on a
selected read. The methodological moral carries through the whole chapter:
**retrieval-from-memory requires a large optimization budget, and short-budget ablations (6,000
steps here) would have spuriously killed it.**

**Figure 4.6 — Emergence of single-step memory benefit with optimization budget.**
The HARD-bin effect is unresolved at 6,000 updates for both seeds and positive with intervals
excluding zero at 20,000 updates for both seeds. This is an optimization-budget result, not a
claim about asymptotic convergence.
*Source:* `docs/figures/fig28_memory_gate_optimization.png`, parsed deterministically from the
four `logs_rhcp/mem_gate_full_s{0,1}{,_t20k}.log` files by
`scripts/make_dissertation_missing_figures.py`.

### 4.3.2 Rollout Matrix

Table 4.5 gives the full arm ordering under the rollout protocol (ALL-row fg-MSE over 152 AR
frames after the 48-frame ground-truth warm-up, seed 0; arms per Table 3.1).

**Table 4.5 — Rollout arm ordering, operating point A.**

| Arm | Context | Ctx tokens | ALL fg-MSE | fg-sharp | Note |
|---|---|---:|---:|---:|---|
| flat-long (diff) | 48 fine frames | 3072 (2.7×) | **0.1564** | **0.0996** | B = 8 × 60k sample-matched (caveat, §4.3.4) |
| reg-long | 48c + 6f, regression | 1152 | 0.1682 | 0.0741 | |
| hier-long (diff) | 48c + 6f | 1152 | 0.1848 | 0.0729 | THE system arm |
| reg-flat | 18 fine frames, regression | 1152 | 0.2015 | 0.0512 | |
| flat-budget (diff) | 18 fine frames | 1152 | 0.2038 | 0.0582 | token-matched flat control |
| hier-short (diff) | 6c + 6f | 480 | 0.2260 | 0.0392 | memory ablation |
| persistence | — | — | 0.2552 | — | floor |

Three seed-0 regularities are visible in the ordering, each defended separately below: (i) the
48-frame coarse-plus-fine arm beats the shorter contiguous-fine control under both tested heads
at this seed; (ii) regression wins the deterministic best-of-1 rollout (§4.2.3); and (iii) the
full-resolution 48-frame window wins outright at 2.7× the tokens (§4.3.4). The first observation
replicates at seed 1 for this particular long-versus-short comparison, but it must not be promoted
to a head-independent law. The complete five-seed temporal-versus-spatial factorial (§4.4.3)
instead finds stable package direction under diffusion and sign-changing direction under
regression. The defensible claim is therefore configuration- and head-scoped: reach helps where
the shorter control cannot address the relevant past, while the preferred way to package that
reach is not universally mechanism-agnostic.

**Figure 4.7 — Error and sharpness by rollout horizon.**
*Content:* foreground MSE on all frames, foreground MSE on HARD re-entry frames, and
foreground-sharpness retention, with persistence and frozen-decoder references.
*Source:* `docs/figures/fig1_claim1_horizon_curves.png`; deterministic source
`scripts/make_figures.py`.

### 4.3.3 Main Two-Seed Pair: Advantage beyond the Flat Memory Horizon

Table 4.6 is the headline of the dissertation: hier-long versus flat-budget at equal 1,152
context tokens, the full horizon × hardness matrix (Δ = flat − hier; positive = hierarchy
better).

**Table 4.6 — hier-long vs flat-budget (seed 0), 152 AR frames after the 48-frame GT warm-up,
paired per-frame fg-MSE, clip-level cluster bootstrap; empty-GT-foreground records (21.5%)
excluded from stratified bins.** Sharpness 0.0729 vs 0.0582; persistence floor 0.2552.

| Horizon | Hardness | n | flat-budget | hier-long | Δ | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| h 1–8 | easy | 434 | 0.0440 | 0.0419 | +0.0022 | [−0.0040, +0.0089] | null |
| h 1–8 | HARD | 150 | 0.1435 | 0.1359 | +0.0075 | [−0.0164, +0.0327] | null |
| h 9–24 | easy | 919 | 0.1216 | 0.1189 | +0.0028 | [−0.0075, +0.0133] | null |
| h 9–24 | mid | 45 | 0.1661 | 0.1624 | +0.0038 | [−0.0121, +0.0182] | null |
| h 9–24 | HARD | 282 | 0.1800 | 0.1728 | +0.0071 | [−0.0110, +0.0260] | null |
| h 25–60 | easy | 2286 | 0.2210 | 0.1989 | +0.0221 | [+0.0080, +0.0368] | **WIN** |
| h 25–60 | mid | 79 | 0.2218 | 0.2036 | +0.0182 | [−0.0049, +0.0437] | null |
| h 25–60 | HARD | 550 | 0.2554 | 0.2209 | **+0.0344** | [+0.0162, +0.0531] | **WIN** |
| h 61+ | easy | 5610 | 0.3060 | 0.2788 | +0.0272 | [+0.0166, +0.0382] | **WIN** |
| h 61+ | mid | 209 | 0.3224 | 0.2842 | +0.0382 | [+0.0178, +0.0616] | **WIN** |
| h 61+ | HARD | 1353 | 0.3073 | 0.2759 | **+0.0314** | [+0.0202, +0.0431] | **WIN** |
| bg-only | | 3263 | 0.0067 | 0.0047 | +0.0020 | [+0.0013, +0.0026] | WIN |
| **ALL** | | 15200 | 0.2038 | 0.1848 | **+0.0189** | [+0.0121, +0.0264] | **WIN** |

On seed 0 the matrix reads as the claim in one table: **null in every bin below horizon 25; a
CI-excluding-zero win in every easy/HARD bin from h 25–60 onward** (the sparse mid cell, n = 79,
is null at h 25–60 and a win at h 61+). The advantage appears exactly where the memory horizon
demands it — h ≈ 24 is half a loop period, past which the required horizon exceeds what 18 flat
frames carry — and nowhere else.

**Seed-1 replication (pre-registered before running).** The same pairing retrained at seed 1 wins
**in every cell**: ALL +0.0354 [+0.0267, +0.0447]; HARD h 25–60 +0.0622; sharpness 0.088 vs
0.057. Two-seed status of the headline: **CONFIRMED** (seed 0: ALL +0.0189 [+0.0121, +0.0264]).
The honest nuance is the onset boundary: it is seed-variable — seed 0 is null below h 25 while
seed 1 wins already from h 1–8, and the hier-long arm itself moves 0.1848 → 0.1690 across seeds
over a stable flat baseline. The robust two-seed claim is therefore **the win and its mid-horizon
concentration, not the exact onset frame**; the "no advantage before h 25" phrasing is a seed-0
observation, retained only as such.

**Per-horizon behavior and the deep-horizon disclosure.** hier-long stays below the persistence
floor at *every* horizon out to 152 frames (more than three loop periods); its error saturates at
0.221–0.222 over the last 70 frames rather than diverging — bounded drift. Sharpness, however,
fades monotonically for **all** arms (hier-long 0.2023 at h 1–4 → 0.0323 at h 105–152; batch-mean
series, §4.1.5): the pre-registered sharpness-retention clause of gate G2b (> 0.5× single-step
sharpness at depth) is **not met** at h ≥ 81, and this is stated plainly as the deep-horizon
limitation rather than tuned away. The blur ceiling is owned by the frozen decoder (AE round-trip
0.2412), not by the dynamics.

**Figure 4.8 — Effect map by horizon and difficulty.**
*Content:* horizon × hardness map of Δ(flat − hierarchy), seeds 0 and 1 side by side. The
horizon-gated onset is seed-0-specific; the common feature is the positive effect and
mid-horizon peak rather than one exact onset.
*Source:* `docs/figures/fig2_claim1_winmap.png`; deterministic source
`scripts/make_figures.py`.

**Figure 4.9 — Qualitative slices of autoregressive rollout.**
*Content:* identical ground truth and seed-1 `hier-long`/`flat-budget` predictions at registered
horizons. Both models become blurred at long horizon; the figure illustrates a modest average
advantage, not a universal per-frame recovery or a claim that one named arm uniquely
hallucinates.
*Source:* `docs/figures/fig7_samples_claim1.png`, from the saved rollout checkpoints.

### 4.3.4 flat-long: the 2.7×-Token Pareto Reference

The flat-long arm (48 full-resolution frames, 3,072 tokens) beats every other arm on both error
and sharpness: ALL 0.1564, fg-sharp 0.0996; against flat-budget its ALL delta is +0.0473
[+0.0362, +0.0592]. This is reported with the same prominence as the headline, because it is what
keeps the claim honest: **at operating point A, hierarchy is an efficiency point, not a
capability necessity.** The hierarchical arm recovers most of the long-memory benefit at ≈37% of
the context tokens; a party willing to pay 2.7× buys the frontier outright. (Protocol caveat,
disclosed wherever flat-long is quoted: it trained at B = 8 × 60k steps — sample-matched to
B = 16 × 30k, but with a different EMA horizon and gradient-noise scale than the other arms.) The
regime where no affordable flat window can cover the loop at all is exactly what operating
point B exists to test (§4.5).

### 4.3.5 Prediction-Head Robustness and Its Boundary

The preregistered two-seed gate asked whether the specific `reg-long` versus token-matched
`reg-flat` comparison reproduced the diffusion reach result. It did: `reg-long` beats `reg-flat`
in every bin at both tested seeds (ALL +0.0334 [+0.0248, +0.0423] at seed 0 and +0.0501
[+0.0396, +0.0612] at seed 1; hard-bin wins at every horizon, including h 1–8 HARD +0.0447;
sharpness 0.0741 versus 0.0512 at seed 0). The regression pairing has no null prelude, so the
seed-0 diffusion onset near h = 25 is not a general property of the memory geometry.

This is evidence that one long-coarse-versus-short-fine reach advantage is not solely an artifact
of the diffusion head. It is **not** evidence that package preference is mechanism-agnostic. The
complete five-seed temporal-versus-spatial factorial in §4.4.3 gives two temporal and three
spatial regression winners and a sign-changing head × package interaction. Accordingly, the
dissertation reports the two-seed replication as a scoped robustness result and rejects the
broader head-independent wording.

### 4.3.6 Completing the Token-Budget Frontier with Matched Upper and Lower Partners

The arm matrix of Table 4.5 leaves the two extreme token budgets without same-budget partners:
flat-long (3,072 tokens) has no hierarchical arm at its budget, and hier-short (480 tokens) has
no flat control at its. Two frontier-completion arms close the grid,
pre-registered before launch (the FRONTIER COMPLETION block of the factorial pre-registration):
**hier-rich** (48 coarse + 36 fine frames = 3,072 tokens, trained B = 8 × 60,000 steps —
sample-matched to flat-long's disclosed regime, so the pairing inherits flat-long's §4.3.4
protocol caveat *symmetrically*) and **flat-short** (8 fine frames = 512 tokens ≈ hier-short's
480, standard B = 16 × 30,000). Diffusion heads first; the regression twins are optional.

The reading was fixed in advance and is quoted verbatim from the pre-registration: "at 3072
tokens the window is SATURATED at 48 frames by the T=100 train clips — hierarchy cannot buy more
window, so per the factorial's abstraction cost the honest expectation is hier-rich ~<= flat-long
(CONVERGENCE at the frontier top). The efficiency claim's correct form is 'the hierarchy curve
reaches a given quality at fewer tokens and converges with flat where the window saturates' — a
frontier-from-below, not dominance at every budget. At the low end, hier-short vs flat-short
tests the frontier's bottom."

**Table 4.6a — Token-budget frontier of operating point A (letter-suffixed to avoid renumbering
the audited table register; final numbering at typeset). Measured anchor rows from Tables
4.5/4.6; frontier-completion cells measured 2026-07-07 (`logs_rhcp/frontier_top.log`,
`logs_rhcp/frontier_bottom.log`); lower-edge pair measured 2026-07-07
(`logs_rhcp/frontier_low.log`, pre-registered the same morning before the runs).**

| Token budget | Flat arm | ALL fg-MSE | Hierarchical arm | ALL fg-MSE | Paired Δ (flat − hier), cluster CI |
|---|---|---:|---|---:|---|
| ≈3,072 | flat-long | 0.1564 | hier-rich | 0.1695 | **−0.0131 [−0.0201, −0.0069]** (flat wins; seed 1 REPLICATED: −0.0107 [−0.0163, −0.0058]) |
| 1,152 | flat-budget | 0.2038 | hier-long | 0.1848 | +0.0189 [+0.0121, +0.0264] (Table 4.6) |
| **896** | flat-min (14 f) | 0.2106 | hier-min (48c+2f) | **0.1706** | **+0.0399 [+0.0299, +0.0501]** (hier wins EVERY cell; seed 1 REPLICATED: +0.0358 [+0.0267, +0.0454], every cell) |
| ≈480–512 | flat-short | 0.2224 | hier-short | 0.2260 | **−0.0036 [−0.0063, −0.0010]** (flat wins) |

**Figure 4.10 — Confirmed token-budget frontier.**
*Content:* ALL fg-MSE versus context-token budget at operating point A, one 6-point curve per
family (flat: flat-short → flat-min → flat-budget → flat-c1 → flat-c2 → flat-long; hierarchical:
hier-short → hier-min → hier-long → hier-c1 → hier-c2 → hier-rich). Mark the tested
family-wise-confirmed hierarchy points at 896/1,152, the unresolved 1,792 point, and the confirmed
flat points at 2,432/3,072. Draw the geometric span threshold as an explanatory annotation and
the persistence floor (0.2552); do not shade an interpolated win band.
*Source:* `docs/figures/article_confirmed_frontier.png`, generated from the untouched E2
simultaneous family in `artifacts/confirmation_e2/results.json`. The historical development
`fig9_frontier.png` is retained for provenance but is not used for confirmation.

The measurement resolved the pre-registered dual branch to a **third outcome, stricter than the
pre-registered convergence expectation, and reported as measured under the no-soft-reads rule**
(the default-to-FAIL discipline of §4.5 applied to wording). At the top budget hier-rich does
not converge with flat-long — it loses CI-clean at equal tokens and equal training regime
(−0.0131 [−0.0201, −0.0069], a small uniform loss across essentially every horizon × hardness
bin); at the bottom budget flat-short beats hier-short (−0.0036 [−0.0063, −0.0010], concentrated
at h ≤ 60, null beyond). The frontier therefore *inverts at both ends*: this lands on the first
fallback branch at the top (the abstraction cost does not vanish at the top budget — the −0.0131
loss is of the same kind as the diffusion-head abstraction cost of §4.4) and refutes the
matched-budget law at the bottom, where the 6–8-frame window is below the memory-relevance scale
and coarse slots are redundancy rather than memory. The mechanism reading: hierarchy pays where
its coarse tokens buy a memory window that (a) exceeds what fine tokens afford and (b) matters;
at window saturation (48 frames, clip-bounded) coarse slots merely displace full-resolution
frames, and below the memory-relevance threshold they duplicate it. Sharpness follows the window,
not the hierarchy: hier-rich 0.0898 versus flat-long 0.0996, hier-short 0.0392 versus flat-short
0.0432 — at matched saturation the hierarchy does **not** retain sharpness better.

This mechanism reading makes a quantitative prediction, which was pre-registered and then
measured the same day (the lower-edge row of Table 4.6a): the win must switch on at the cheapest
budget whose coarse span covers the ≈48-frame loop — 48 × 16 coarse tokens plus a minimal fine
tail, ≈896 tokens, a budget at which a flat model affords only a 14-frame window. The measured
pair confirms the prediction, and more strongly than expected: hier-min beats flat-min in
**every horizon × hardness cell** (overall +0.0399 [+0.0299, +0.0501] — the largest hierarchical
advantage measured in the campaign), with the visibility-stratified gates showing the reach
signature exactly where the mechanism operates (dt ≤ 48 HARD +0.0750; dt 49–96 HARD +0.0282;
dt 97–192, beyond the 48-frame span, null). Two structural conclusions follow. First, the win
smallest tested dense-family hierarchy win coincides with the geometric span-threshold arithmetic
written before the run: the next smaller dense-family point favors flat, while hierarchy wins at
896. This is a tested ordering rather than a continuous switch. Second, the **largest tested
dense-family advantage occurs at 896 tokens**: reducing the
budget removes the flat model's window from the memory horizon faster than the hierarchical
abstraction cost grows — the efficiency argument is strongest exactly where budgets are
tightest, and hier-min at 896 tokens (0.1706) reaches a quality the flat curve attains only at
substantially larger budgets in the development curve. The untouched family leaves 1,792
unresolved and confirms flat at 2,432, so no equal-quality multiplier or interpolated crossing is
claimed.
The pre-registered seed-1 replication confirmed the point: every cell WIN again (overall
+0.0358 [+0.0267, +0.0454]) with the same visibility-stratified reach signature — the 896-token
896-token point is therefore a two-seed result (seed-mean Δ ≈ +0.0379, the largest two-seed effect
of the campaign), alongside the two-seed 1,152-token pairing of §4.3.3. The §4.3
efficiency claim takes its final, tested-bracket form: **the simultaneous confirmation family
supports hierarchy at 896/1,152, leaves 1,792 unresolved, and supports flat fidelity at
2,432/3,072; no interval, exact edge or unique crossing is inferred.**

## 4.4 Forms of Hierarchical Abstraction: Spatial, Temporal and Depth

A hierarchy in a latent video predictor is not one construction but a **family of forms**. A
context-token budget can be packaged along four axes:
**spatial abstraction** (every frame, at lower resolution), **temporal abstraction** (full
resolution, at a coarser tick), **depth** (how many nested levels), and **allocation** (the split
between long-span and recent-detail streams). Each shape buys memory span at a different token
price and pays a different information cost. Throughout, hierarchy means multi-rate and/or
multi-RESOLUTION context — independent axes: a configuration can be hierarchical in tick rate
while single-resolution (the §4.4.3 strided control — a two-rate temporal hierarchy in the
clockwork/CW-VAE sense, whose levels are likewise temporal) or hierarchical in resolution while
single-rate (dense coarse arms); where the two depth counts differ, both are reported
(resolution depth / temporal-rate depth). The systems-analysis question is therefore not whether
hierarchy helps universally, but **which form pays, where, and why**. Three cells of this grid
are measured in earlier sections (allocation — §4.3.6; the spatial factorial — §4.4.1 below; the
slow-clock boundary — §4.5); the remaining cells (temporal family §4.4.2, span control §4.4.3,
depth §4.4.5) follow the pre-registered program of `SPATIAL_FACTORIAL.md` (final block), and
§4.4.5 assembles the grid into one comparison table in the manner of the Clockwork VAE benchmark
[13] — the format adopted, the controls tightened (Chapter 1, §1.3; fourth article):
matched token budgets, backbone-capacity-neutral depth (levels add tokens to a shared
backbone; per-level I/O heads +0.30 M / +1.4% on the 22.19 M predictor + the frozen 1.25 M AE
tap, disclosed), foreground-masked metrics.

**Table 4.7-A — What each packaging axis buys and costs.**

| Axis | Form | What it buys | Information or causal cost | Evidence |
|---|---|---|---|---|
| spatial | dense 4×4 coarse stream, stride 1 | 4× reach per token; per-level decodability; dense temporal coverage | −0.008 under diffusion and approximately null under regression at fixed window | §4.4.1, measured |
| temporal | full-resolution anchors, stride 4 | 4× reach per token while retaining detail at anchors | blind intervals between anchors; no cheap dense decodable level | §4.4.3: temporal package wins under diffusion on all five seeds; regression is heterogeneous |
| spatial × temporal | coarse stream, stride 2 | 8× nominal reach per token; cheapest tested loop coverage | couples both information costs and the legacy target-interface change | §4.4.2: +0.0330/+0.0338 at 512 tokens under diffusion |
| depth | additional 2×2 stream | a third representational scale | target-loss allocation, optimization load and top-stream floor are inseparable | §4.4.5: −0.0196 versus two levels at fixed budget, seed 0; appendix-grade |
| allocation | spend on reach before recency/detail | confirmed lower/interior hierarchy operating points | sacrifices full-resolution fidelity after reach is saturated | E2: hierarchy at 896/1,152; 1,792 unresolved; flat at 2,432/3,072 |

### 4.4.1 Spatial Factorial: an Economical, Head-Dependent Verdict

Every hierarchy arm of §4.3 confounds two factors: the coarse spatial resolution of the history,
and the longer window its cheaper tokens buy (an accepted external-review finding). The factorial
of this section isolates the resolution factor: a window × history-resolution grid at operating
point A (shared frozen AE, seed 0, protocol identical to the rollout campaign so all existing
records pair exactly), holding recency equal — every coarse-history arm keeps the 6 most recent
fine frames, exactly the hier-long design. The hypothesis under test, in its original form:
coarse history is a *more abstract, less distracted* memory representation.

**Table 4.7 — Window × history-resolution factorial (fg-MSE ALL over 152 AR frames, cluster CIs;
context tokens in parentheses).**

| Window | Full-res history | Coarse history | Paired Δ (full − coarse) |
|---|---|---|---|
| 18 f, diffusion | flat-budget 0.2038 (1152) | spat18 0.2120 (672) | **−0.0082 [−0.0115, −0.0050] LOSS** (short-horizon-concentrated, null by h > 120; sharp 0.058 vs 0.050) |
| 18 f, regression | reg-flat 0.2015 (1152) | spat18-reg 0.2008 (672) | **+0.0008 [−0.0013, +0.0029] null** (h 61–120 HARD +0.0047 WIN; tiny h 1–8 loss) |
| 48 f, diffusion | flat-long 0.1564 (3072) | hier-long 0.1848 (1152) | −0.0284 (from Tables 4.5/4.6; 2.7× token gap) |

**Abstraction sufficiency at the fixed 18-frame window (read G-S1) — head-dependent.** For the
diffusion head, coarse history is a small, real, short-horizon-concentrated cost: −0.0082 ALL at
42% token savings. For the regression head — the better rollout head (§4.2.3) — coarse history is
**statistically free**: +0.0008 [−0.0013, +0.0029], null ALL at the same 42% savings, with the
long-horizon HARD bin slightly *better* (h 61–120 HARD +0.0047). The "intrinsically better
abstraction" hypothesis is therefore **not supported**; the supported form of the claim is
**economic and head-conditional**: spatial abstraction is lossy compression whose token savings
buy the memory window, and under the regression head the loss term is ≈0.

**Interaction (read G-S2).** The abstraction cost per saved token is similar at both windows for
the diffusion head (≈0.007–0.008 per 480 tokens); there is no evidence the cost shrinks as the
window grows.

**Window at fixed representation (read G-S3).** hier-long 0.1848 (W48, 1,152 tokens) versus
spat18 0.2120 (W18, 672 tokens): extending the window via coarse tokens buys −0.027 — an order
that dwarfs the abstraction cost, the same window-ordering law as Table 4.5.

**Within-family readout (measured arm ordering, not a factorial claim).** spat18-reg (672
tokens) at 0.2008 matches or beats the diffusion flat control flat-budget (1,152 tokens) at
0.2038 — regression + coarse history at 58% of the tokens. At the long window, the complete
five-seed factorial below separates a stable diffusion package effect from regression and
cross-head training-seed heterogeneity.

**Honest link to the prior spatial-gate null.** This project's earlier per-pixel spatial-gate
study on Moving MNIST found that an upper spatial level engaged unstably and added no quality — a
clean null for spatial abstraction as an *intrinsic* representational win. The factorial is
consistent with that history rather than overturning it: isolated from the window confound,
coarse history is not intrinsically better (it costs under diffusion and is at best free under
regression); its value is what its cheapness buys. The temporal-only corner (strided
full-resolution history), declared an omission in the factorial's pre-registration, is filled at
operating point A by the span control of §4.4.3.

**Figure 4.11 — Spatial factorial by prediction head.**
*Content:* the 2×2 factorial as a diagram: axes window (18/48) × history resolution (full/
coarse), each cell annotated with arm name, token count, ALL fg-MSE; paired deltas drawn on the
edges (resolution edges: −0.0082 diff / +0.0008 reg; window edge: −0.027 at fixed coarse
representation), with the combined-optimum path highlighted.
*Source:* `docs/figures/fig4_claim3_factorial.png`; deterministic source
`scripts/make_figures.py`. The figure does not promote the combined arm ordering to a
training-seed-robust head interaction.

### 4.4.2 Temporal Family and the Moving Tested Boundary

Span-per-token provides an explanatory calculation, not an exact edge law. The dense coarse
configuration first favors hierarchy at the tested 896-token point. The stride-2 coarse
configuration (`hier-s2-min`, 24 slots + 2 fine) purchases the same 48-frame reach at **512
tokens** and wins on both development seeds (+0.0330/+0.0338) and on the untouched E2 holdout
(+0.0340/+0.0381). Its reach-band effects are also positive on both E2 seeds. This is strong
evidence for the complete stride-2 configuration and is consistent with the span calculation.

It is not a pure context-stride intervention: the legacy arm also changes target slots,
target-token count, auxiliary supervision and total sequence length. Therefore the result is not
worded as an exactly moved lower edge. A future stride dose-response, including any 320-token
point, would require a fixed target interface and a new preregistration; it is not part of the
current closure portfolio. The measured regression seed-0 B1 result (+0.0639
[+0.0516,+0.0763]) supports the configuration but does not remove this confound.
**Figure 4.13:** `docs/figures/fig13_family_edges.png`; the caption treats B1 as a configuration
comparison and does not draw a continuous edge or unmeasured point.

### 4.4.3 Coarse Level or Reach Alone? Sparse Full-Resolution Control

The control `strided-flat` keeps NO coarse level: 11 full-resolution slots at stride 4 (span 48)
+ 6 recent fine = 1,088 tokens — 64 short of the 1,152 pairing, a deficit that *favors the
control*. The complete five-seed factorial returns the following package effects (positive means
the temporal package wins; negative interaction means regression shifts toward spatial
coarsening):

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | **−0.0089 [−0.0151, −0.0025]** |
| 1 | +0.0046 [+0.0007, +0.0084] | −0.0104 [−0.0146, −0.0062] | **−0.0150 [−0.0199, −0.0104]** |
| 2 | +0.0111 [+0.0070, +0.0151] | −0.0208 [−0.0270, −0.0150] | **−0.0319 [−0.0401, −0.0245]** |
| 3 | +0.0075 [+0.0037, +0.0114] | +0.0248 [+0.0181, +0.0323] | **+0.0173 [+0.0106, +0.0245]** |
| 4 | +0.0060 [+0.0027, +0.0092] | −0.0166 [−0.0229, −0.0108] | **−0.0226 [−0.0297, −0.0160]** |

Diffusion favors temporal subsampling on all five seeds (mean +0.0098, SD 0.0060, t95 [+0.0023,
+0.0173]). Regression has two temporal and three spatial material winners (mean −0.0024, SD
0.0195, t95 [−0.0266, +0.0217]). The head interaction changes sign and has mean −0.0122 with t95
[−0.0353, +0.0109], so the preregistered robust negative-shift rule fails. The clip-cluster
intervals condition on each trained model; the t intervals use training seeds as units. The
measured statement is that, at this operating point, the diffusion head consistently uses
temporally sparse full-resolution packaging, whereas the regression head is training-seed
dependent—not that one package is universally preferred or that a directional head law holds.

All cells in this factorial use the existing adaptively used evaluation pool; E4a/E4b is
seed-resolved development evidence, not a second untouched confirmation.

**Figure 4.12 — Head × package heterogeneity across training seeds.**
*Content:* five seed-resolved diffusion, regression and interaction effects. Clip-bootstrap
intervals condition on each trained model; the shaded t intervals use the five training seeds as
the inferential units.
*Source:* `docs/figures/fig17_head_package_seed_heterogeneity.png`, generated directly from
`artifacts/e4b_head_package/results.json`.

Three binding scope qualifiers (mirror of
the article §4.4.3): (1) **static world** — nothing changes between stride-4 anchors, so the
testbed is maximally kind to temporal subsampling; on data with inter-anchor dynamics the stride
aliases what a dense stream tracks—this is an operating-point result, not a general law;
(2) the abstraction cost is a property of this autoencoder's 4×4
layout code (frozen-AE discipline fixes the choice); (3) the winning control is itself a
hierarchy in the RATE sense (two tick rates, one resolution — the clockwork quadrant), so the
measured statement is that the multi-rate form beats the multi-resolution form at this operating
point under diffusion, not that hierarchy is unnecessary.

### 4.4.4 What Each Level Preserves: Blinding Probes

Rerun the trained arm's full 152-frame rollout with one context stream **blinded** (zeroed in
standardized latent space) at every step. **MEASURED (2026-07-10): asymmetric essentiality.** Blind-coarse = the predicted
retrieval signature (hard re-entry +0.073 vs h1–8 +0.022; ALL 0.2236 past flat-budget) ✓;
blind-fine = renderer collapse, expectation partially wrong (h1–8 0.2427; sharp 0.0086 below
the structural threshold; ALL below persistence — brittleness-dominated per the binding OOD
caveat); blind-top (hier3) = +0.0035 ALL only — the top stream carries almost nothing usable,
independent convergent evidence for the §4.4.5 mechanism reading. Report-only. Binding
interpretation caveat (per the
pre-registered amendment): a zeroed stream is an out-of-distribution input, so the collapse
magnitude bounds the stream's contribution from below confounded with OOD brittleness from
above — the probe licenses “stream removed → capability lost,” not a precise information
accounting. **Figure 4.14:** `docs/figures/fig14_level_blinding.png`; measured
eval-only blinding curves with the out-of-distribution qualification retained.

### 4.4.5 Consolidated Comparison: Depth, Heads and the Assembled Grid

**Table 4.7-B — Consolidated context-shape grid (152-step autoregressive rollout; `ALL`
foreground MSE, lower is better).**

| Context budget | Resolution / temporal depth | Temporal factor | Spatial factor | Head | Configuration | Reach | ALL fg-MSE |
|---:|---|---:|---:|---|---|---:|---:|
| 512 | 1 / 1 | 1 | 1 | diffusion | `flat-short` | 8 | 0.2215 (2 seeds) |
| 512 | 2 / 2 | 2 | 2 | diffusion | `hier-s2-min` | 48 | 0.1881 (2 seeds) |
| 512 | 2 / 2 | 2 | 2 | regression | `hier-s2-min-reg` | 48 | 0.1566 (1 seed) |
| 896 | 1 / 1 | 1 | 1 | diffusion | `flat-min` | 14 | 0.2104 (2 seeds) |
| 896 | 2 / 1 | 1 | 2 | diffusion | `hier-min` | 48 | 0.1725 (2 seeds) |
| 1,088 | 1 / 2 | 4 | 1 | diffusion | `strided-flat` | 48 | 0.1644 (2 seeds) |
| 1,088 | 1 / 2 | 4 | 1 | regression | `strided-flat-reg` | 48 | 0.1590 (5 seeds) |
| 1,152 | 1 / 1 | 1 | 1 | diffusion | `flat-budget` | 18 | 0.2042 (3 seeds) |
| 1,152 | 2 / 1 | 1 | 2 | diffusion | `hier-long` | 48 | 0.1762 (3 seeds) |
| 1,152 | 1 / 1 | 1 | 1 | regression | `reg-flat` | 18 | 0.2009 (5 seeds) |
| 1,152 | 2 / 1 | 1 | 2 | regression | `reg-long` | 48 | 0.1592 (5 seeds) |
| 1,152 | 3 / 1 | 1 | 2×2 | diffusion | `hier3` | 48 | 0.2044 (1 seed) |
| 1,792 | 1 / 1 | 1 | 1 | diffusion | `flat-c1` | 28 | 0.1876 (1 development seed) |
| 1,792 | 2 / 1 | 1 | 2 | diffusion | `hier-c1` | 48 | 0.1799 (1 development seed; E2 simultaneous result unresolved) |
| 2,432 | 1 / 1 | 1 | 1 | diffusion | `flat-c2` | 38 | 0.1703 (B16×30k) |
| 2,432 | 2 / 1 | 1 | 2 | diffusion | `hier-c2` | 48 | 0.1820 (B16×30k) |
| 3,072 | 1 / 1 | 1 | 1 | diffusion | `flat-long` | 48 | 0.1572 (B8×60k; 2 seeds) |
| 3,072 | 2 / 1 | 1 | 2 | diffusion | `hier-rich` | 48 | 0.1691 (B8×60k; 2 seeds) |

*Notes.* Multi-seed values are means; exact seed effects and intervals appear in the corresponding
subsections. `hier-s2-min` also changes the target interface and is therefore a whole-configuration
effect. `strided-flat` has 1,088 context tokens and `hier-long` 1,152: the comparison is near
matched, not exact. The machine-generated grid with sequence length, throughput, VRAM and
sharpness is stored in `docs/CWSTYLE_TABLE.md`.

The depth row `hier3` (2×2/4×4/8×8 factor-2 pyramid,
48 top + 24 coarse + 9 fine = the same 1,152 tokens) tests a task with a single memory scale and a
protocol-capped 48-frame history. **MEASURED (2026-07-10, s0) — the pre-registered ALTERNATIVE (loss) branch:** hier3 ALL
**0.2044** / sharp 0.0564 (class-lowest): CI-clean loss to hier-long (**−0.0196
[−0.0248, −0.0146]**), null vs flat-budget (−0.0007) — the third stream erases the two-stream
win at fixed budget. Three inseparable disclosed mechanisms: loss-weight dilution (per the
amendment; the class-lowest sharpness is its predicted signature), fixed-step optimization
load, and the top-stream information floor (2×2 = 32-px cells vs ~20-px digits, with the coarse
stream cut 48→24 to pay for the top). It is reported as a depth-plus-loss-allocation probe that
costs approximately 0.02 at fixed budget, not as evidence that depth is intrinsically harmful;
the depth hypothesis
stands (single-scale data cannot reward a third scale — consistent with the per-pixel-gate
null, the §4.5 boundary, CW-VAE's two-levels-sufficed observation). Disposition:
appendix-grade; a weight-matched follow-up would be required for any causal depth statement.
**Figure 4.15:** `docs/figures/fig12_cwstyle_curves.png`, degradation curves over the
152-frame rollout for the measured configuration grid.

## 4.5 Operating Point B

Operating point A proves *efficiency at matched budget*: the loop (≈48 frames) sits inside the
affordable flat window (48 frames), so flat-long can always buy the result with tokens. The
boundary study asks the capability question on the pre-registered design frozen before any
launch (the `docs/HGRID_DESIGN.md` preregistration trail is included in the reproducibility
appendix): in a regime where
**no affordable flat window covers the memory horizon** (hgrid3, effective loop ≈196–213 frames),
does temporal abstraction — a strided, slow-rate coarse level — deliver what flat context cannot?
The seed-0 study closed with a **negative rollout outcome under a binding pre-committed decision
rule**. The registered E5a repetition then retained the genuine single-step positive and the early
self-fed benefit, but did not reproduce the late inversion at either new training seed. This
section therefore reports a negative replication and three diagnosed methodology lessons, not a
trust boundary or capability law. The §4.3 matched-budget
result is unaffected under every branch of this study, and remains the headline of the
dissertation.

### 4.5.1 Aliasing Episode: Diagnosed Failure, Not a Discarded Run

The first dataset iteration at the wide operating point (*hgrid*: world 768 px, with background
wavenumbers scaled ×4 so that the frozen loopmom autoencoder could be reused) produced a
dead-flat null at the 20,000-step single-step pre-check (HARD −0.0008 [−0.0027, +0.0012]; all dt
bands null) and a CI-clean **LOSS** at the budget-matched 40,000-step extension (HARD −0.0025
[−0.0042, −0.0008]; ALL −0.0012; dt > 192 −0.0026): more optimization made the unaddressable
memory slots a pure attention cost. The cause is established, not speculative: the ×4 wavenumber
scaling made the background **exactly 192-px periodic**, so the four aliased viewport positions
are pixel-identical — the crop test measures pairwise MSE **0.000000** between them. Retrieval
lost the absolute-position landmark it plausibly used at operating point A (where unit-wavenumber
components made positions unique). The autoencoder-reuse constraint *caused* the failure and was
dropped (rule adopted: a fresh autoencoder per testbed variant; within-grid comparability needs
only one shared autoencoder per campaign). The second iteration (hgrid2, a DC brightness cue) was
aborted mid-run in favor of the mechanistically stronger variant; two position-probe attempts
from that period failed their own positive controls and are **retracted on the record**. The
third iteration, **hgrid3** (unscaled background with unit wavenumber forced), restores position
uniqueness (crop-uniqueness 0.004–0.010, versus hgrid2's 0.0006–0.003) and passes the full
regime verification of §4.1.1.

### 4.5.2 Geometry: Memory Reach versus Effective Loop Period

The pre-check arm carries 48 coarse slots at stride 4 — a nominal memory span of 192 frames. The
relevant horizon, however, is not $\mathrm{world}/\mathrm{speed} = 192$ but the **effective
loop**

$$
L_{\mathrm{eff}}
\;=\;
\frac{\mathrm{world}}{\mathrm{speed} \cdot (1 - p_{\mathrm{hold}})}
\;\approx\;
\frac{768}{4 \cdot 0.9}
\;\approx\; 213 \ \text{frames},
\tag{4.1}
$$

because hold frames advance time without advancing the pan. The 48-slot arm's 192-frame span
therefore falls *short* of its own target loop (≈196–213 measured) — a geometry error inherited
from the operating-point-A intuition, and the motivation for the span-224 arm of §4.5.4.

### 4.5.3 Recall Exists: Extended Evaluation and Internal Control

The 4×-widened paired re-evaluation of the same hgrid3 models (evaluation batches 24 → 96;
pre-registered before running; both reads disclosed) flips the underpowered nb = 24 nulls to
CI-clean wins for the strided arm — **48-slot stride-4 (reach 195 frames): HARD +0.0016
[+0.0007, +0.0025] WIN; REACHABLE-HARD +0.0021 [+0.0008, +0.0036] WIN; dt 49–96 +0.0032 WIN; ALL
+0.0005 WIN** — while the contiguous-48 positive-control arm (operating point A's exact winning
design, but with reach only 51 frames here) is **null in every cell**. The effect appears exactly
where the memory reach covers the re-entry interval and vanishes where it cannot: a reach
dose-response, serving as the internal control. Two readings attach. First, beyond-flat-window
retrieval on the position-unique wide world **exists** but is ≈7× smaller than at operating
point A (+0.002 vs +0.0135 absolute; ≈2% vs ≈13% relative) — consistent with the span shortfall
of §4.5.2 diluting it and with the harder addressing problem at this scale. Second, the nb = 24
pre-checks were underpowered for effects of this magnitude, not wrong about the aliased hgrid
(whose 40k read was a CI-clean LOSS; the aliasing verdict stands).

### 4.5.4 Reach-Threshold Hypothesis and Preregistered Single-Step Test

The pre-approved span-fix shot — 56 slots at stride 4, span 224 > loop, 40,000 steps, evaluation
width 96 — returned, on a single seed: **HARD +0.0273; REACHABLE-HARD +0.0363; beyond-227 null**,
with the effect confined to the hard-reachable cells (easy, mid, and beyond-reach all null).
Taken at face value this is the threshold signature — an order-of-magnitude jump once the span
exceeds the effective loop, localized exactly where reach predicts — but it was explicitly
recorded as *"threshold HYPOTHESIS — systematic-or-lucky, undetermined"* (single seed, no
same-pool 48-vs-56 comparison, measured against the weakest baseline). The resolution rule was
written before the disambiguation runs, converged with two external reviewers, and was binding;
it is quoted verbatim in the pre-registration appendix: R1 (replication on an independent seed,
CI > 0), R2 (reach causality: the strided arm must beat a contiguous arm with *identical* token
and slot counts, reach 227 vs 59, CI > 0), R3 (informative, non-gating: the 48-slot arm at
seed 1), with PASS = R1 AND R2 gating the decisive rollout pair, any uncovered outcome defaulting
to FAIL, and FAIL closing the study with no further capability spend.

**Single-step outcomes (the characterized positive).** R1 passed: the span-224 arm beats its
baseline on an independent seed — REACHABLE-HARD +0.0134 [+0.0107, +0.0161], HARD +0.0089
[+0.0069, +0.0109] — at 2.7× smaller magnitude than seed 0 (+0.0363 / +0.0273; seed mean
≈ +0.025 reachable): replicable but seed-noisy. R2 passed: the strided arm beats the contiguous
same-token control at both seeds (differential-reach band +0.0374 [+0.0334, +0.0416] seed 0;
+0.0144 [+0.0118, +0.0171] seed 1). R3: the span-192 arm stays ≈flat at both seeds (+0.0004;
+0.0011), and the contiguous arms never help at either seed. The mechanism *attribution* is
seed-dependent and is not over-read: at seed 1 the win concentrates in the reach-differential
bands exactly as a per-sample reach-causal account predicts (shared +0.0053 vs differential
+0.0181; versus the contiguous control, shared null −0.0005 vs differential +0.0144), but at
seed 0 it is uniform across shared and differential bands (+0.0343 vs +0.0366) — a circuit-level
rather than per-sample-reach signature. The reliable two-seed single-step statement is therefore:
**a memory span exceeding the effective loop reliably enables retrieval learning (2/2 span > loop
runs CI-positive versus 0/4 span ≤ loop runs at ≈0), while the magnitude and per-sample usage
pattern of the learned circuit vary by seed.** Disclosure: the single-step gate CIs in this
subsection are sample-iid, not clip-clustered; the load-bearing cells survive a conservative 2–3×
design-effect widening, and a clip-clustered recomputation is reported alongside. The
teacher-forced caveat is stated where it belongs: the single-step study conditions on
ground-truth context, and the rollout pair of §4.5.5 is the real test.

### 4.5.5 Decisive Rollout Pair and Registered Negative Replication

The pre-registered decisive pair (the span-224 strided arm versus the long-context diffusion
baseline on hgrid3; warm-up 224 frames; uniform B = 8 × 40k training) returned **RED**: over 256
self-fed rollout frames the pre-registered gate cells are null — dt 49–96 HARD +0.0020
[−0.0071, +0.0111]; dt 97–192 −0.0028 [−0.0088, +0.0035]; dt > 192 −0.0018 [−0.0065, +0.0027] —
so the seed-0 programme did not establish rollout capability on its registered gate. Its horizon
profile nevertheless suggested an early hierarchy benefit followed by a late inversion.

E5a preregistered this early-positive/late-negative pattern and repeated the same pair at training
seeds 1 and 2. Positive effects favor the hierarchy: seed 0 is +0.0204
[+0.0133,+0.0279] at h 1–24 and −0.0064 [−0.0095,−0.0032] at h 61–256; seed 1 is +0.0082
[+0.0035,+0.0133] and +0.0074 [+0.0038,+0.0112]; seed 2 is +0.0230
[+0.0150,+0.0315] and +0.0087 [+0.0035,+0.0139]. Thus the early benefit is CI-clean at all
three seeds, but neither new seed reproduces the late-negative direction. The registered verdict
is **NOT_REPLICATED**, and the gated dose-response is not authorized. Addressability and self-fed
reliability remain distinct constructs, but the present study does not establish a stable
approximately-24-frame trust boundary. The §4.3 matched-budget result is unaffected.

**Figure 4.16 — Registered negative replication of the self-fed boundary.**
*Content:* seed-resolved paired Δ (strided − baseline) versus rollout horizon with cluster-CI
bands: early hierarchy benefit at seeds 0–2, late inversion at seed 0 only, and positive late
effects at seeds 1–2; the registered primary windows are highlighted.
*Source:* `docs/figures/fig18_trust_negative_replication.png`, generated from
`artifacts/e5a_trust_replication/results.json` and independently validated E5a manifests.

### 4.5.6 Methodological Lessons from the Boundary Study

1. **The effective-loop formula.** A memory mechanism's span must be sized against
   $L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed} \cdot (1 - p_{\mathrm{hold}}))$ (eq. 4.1),
   not $\mathrm{world}/\mathrm{speed}$: hold frames consume slots without advancing the loop. The
   48-slot arm's 192-frame span silently fell ≈2–10% short of its own target (≈196–213),
   plausibly diluting the effect it was built to detect.
2. **The autoencoder-reuse trap.** Reusing a frozen autoencoder across testbed variants forced a
   background transformation (×4 wavenumber scaling) that made the world exactly
   viewport-periodic and destroyed the positional landmarks retrieval depends on — a dataset-side
   confound that manifested as a *mechanism* failure (memory slots scoring as pure attention
   cost). Rule adopted: a fresh autoencoder per testbed variant; share one autoencoder only
   within a campaign.
3. **Evaluation power.** The nb = 24 pre-checks were adequately powered for operating-point-A
   magnitudes (+0.0135) but not for the ≈+0.002 effects of the harder regime; the pre-registered
   4× widening flipped nulls to CI-clean wins on the *same* checkpoints. Reading a null requires
   a power argument against the plausible effect size, not just a confidence interval spanning
   zero — and both the underpowered and the widened reads are disclosed.

## 4.6 Generalization across Standard and Synthetic Datasets

The studies of §§4.2–4.5 select a configuration and delimit it on the purpose-built testbed
family; this section carries that configuration outward — to the standard benchmark of the field
and to the synthetic suites on which the reversible-transformation method was published ([A2],
§4.1.2) — so that the dissertation's two experimental programs meet on shared ground. Scope
discipline, stated once: **§§4.6.2 and 4.6.5 are measured**; the 2D-rotation, 3D-polycube,
MMNIST-CIFAR and KTH extensions in §§4.6.3, 4.6.4 and 4.6.6 are future protocols and contribute
no projected number or performance claim.

### 4.6.1 Selected Configuration by Inference Regime

The recipe transferred below is not a preference but the measured selection of §§4.2–4.4, and it
is regime-dependent, exactly as §4.2.5 concluded. For the **deterministic best-of-1
autoregressive rollout** — the deployment regime of every long-horizon claim — the selected head
is **regression** (it wins the rollout reversal, §4.2.3, and the memory win replicates under it,
§4.3.5), combined with a context package selected to cover the relevant span with the least
measured information loss. The coarse-plus-fine package (long coarse memory + recent fine
window) is the replicated spatial option, whose coarse history is statistically free under this head
(+0.0008 [−0.0013, +0.0029], Table 4.7), at the **largest affordable window** — window extension
at fixed representation dwarfs the abstraction cost (−0.027 versus −0.008, read G-S3). The
strongest replicated spatial recipe is reg-long (seed mean 0.1593 at 1,152 tokens, §4.4), while
the cross-family temporal option remains seed/head-gated. For the **single-step best-of-N coverage regime**, the selected head is **diffusion** (it
wins every column at matched capacity and NFE, Table 4.2), with the same hierarchical context
rule. Transfers below use the rollout recipe as primary and report the diffusion head where
sampling diversity is part of the question; where a measured transfer predates this formulation,
its actual arm configuration is disclosed in place (§4.6.2 used the diffusion head, best-of-1).

### 4.6.2 OpenSTL Moving-MNIST Bridge: Measured Negative Control

The bridge announced in §4.1.3 is measured. Protocol: the canonical OpenSTL 10→10 suite on the
canonical test set (n = 10,000 sequences), with bit-verified reimplementations of the OpenSTL
metric conventions (pixel-sum MSE/MAE) and a faithful port of the standard sequence generator; a
fresh two-level autoencoder trained for this benchmark (per the §4.5.6 rule: a fresh autoencoder
per testbed variant); the unmodified hierarchical DiT backbone of §3.2.2 (22.19 M parameters),
trained B = 16 × 15,000 steps, bf16, evaluated best-of-1 with three autoregressive window steps
(4 + 4 + 2 frames). Two arms: **mm-flat** (10 fine frames of context, 640 tokens) and
**mm-hier** (10 coarse + 6 fine, 544 tokens).

**Table 4.8 — OpenSTL Moving-MNIST bridge (10→10, canonical test set, n = 10,000; pixel-sum
MSE/MAE per the OpenSTL convention).**

| Arm | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mm-flat (10 fine frames, 640 tokens) | 180.91 | 229.94 | 0.7625 | 13.55 | **0.1320** |
| mm-hier (10 coarse + 6 fine frames, 544 tokens) | 181.29 | 230.45 | 0.7607 | 13.54 | 0.1389 |
| AE round-trip ceiling | 3.30 | 20.79 | 0.9917 | 30.93 | 0.0062 |
| Persistence | 286.28 | 337.89 | 0.6528 | 11.56 | 0.2533 |

Leaderboard reference (deterministic specialist regressors at ≈200 epochs, OpenSTL
documentation): ConvLSTM-S 29.80, PredRNN 23.97, PredRNN++ 22.06, SimVP+gSTA-S 26.69, TAU 24.60 aggregate MSE [74] (range parity with article Table 5.1; the strongest specialist, PredRNN++ 22.06, sets the quoted floor).

Three readings, in the order of their epistemic weight.

1. **The protocol anchor is achieved.** The harness reproduces the standard benchmark protocol
   exactly (canonical test sequences, bit-verified metric conventions), and the autoencoder
   ceiling row (3.30 against the arms' ≈181) shows the tokenizer is not the limiter: the gap to
   the specialists is owned by the dynamics model and objective, not the representation.

2. **The absolute aggregate numbers are modest, with disclosed reasons — and the "mid-pack
   aggregate MSE" clause of the §4.1.3 pre-registered expectation is refuted and reported as
   such.** The measured 180.91 stands against the specialists' 22.06–29.80: the evaluated arm is
   a 22 M stochastic latent-diffusion *generalist* at ≈15,000 optimizer steps, evaluated
   best-of-1, against deterministic pixel-regressors trained roughly an order of magnitude longer
   and optimized exactly for the pixel-sum-MSE aggregate — an aggregate that structurally
   punishes sampling variance (§3.4.3: on this ≈93%-black benchmark, aggregate metrics reward
   mean prediction). Accordingly, **no statement of the form "competitive with, or beating, the
   OpenSTL leaderboard on aggregate MSE" appears anywhere in this dissertation** — the
   measurement refutes it. What does hold: the model halves persistence's LPIPS (0.1320 vs
   0.2533) and beats persistence on every metric of the table.

3. **The flat ≈ hier null (Δ ≈ 0.4 aggregate MSE ≈ 0) is the measured short-protocol boundary,
   not a failure.** Moving MNIST has no off-screen re-entry at 10→10, so this protocol does not
   expose the re-entry-defined demand isolated at operating point A. It does not prove that
   persistent identity/velocity is irrelevant over long rollout; a valid persistent-object study
   would test that separately.
   Thus the bridge supplies a short/no-re-entry boundary observation, while the matched-budget
   hierarchy win is two-seed CI-clean (§4.3.3). The question "when does hierarchy pay" receives
   its negative control from the standard benchmark itself.

**Historical Phase C preregistration (2026-07-10; before any result).** The original
long-horizon design registered `mmL-flat` and `mmL-hier` at seeds 0/1/2, a 48-frame warm-up and
approximately 300 self-fed frames. Its continuity primary was paired foreground-MSE: replicated
practical equivalence required every seed's complete clip-cluster interval to lie inside ±0.005,
with horizon bins secondary. The design also proposed digit identity, trajectory and count
endpoints with preliminary margins ±0.02 accuracy, ±1 pixel and ±0.05 digit/frame. A sound
positive hierarchical effect was to be examined and reported. **The study did not launch.**

**Validity amendment (2026-07-11; before launch).** The registered foreground-MSE question is
retained narrowly. The legacy factor evaluator inferred identity from horizontal component order,
merged overlaps, used many-to-one trajectory matching and intersected frames scorable in both
arms; those choices cannot support identity, trajectory, coverage or equivalence conclusions. The
corrected hybrid protocol stores persistent object/glyph IDs, official train/test glyph splits,
boxes and per-object trajectories; uses one-to-one Hungarian assignment; penalizes misses and
extras; reports coverage; validates the digit classifier on rendered, overlapping and
autoencoder-reconstructed crops; and justifies native-unit margins. The original registration and
this amendment are both retained in the scientific record. The conditional protocol is documented
in `docs/PHASE_C_V2_PROTOCOL_2026-07-11.md`; no result or figure enters the dissertation until its
CPU, classifier and protocol-freeze gates pass.

### 4.6.2a Persistent-object long Moving-MNIST: a measured external boundary

The corrected E6 protocol passed every CPU, codec, classifier, production-crop, data-integrity and
protocol-freeze gate before protected evaluation. The frozen comparison used `mmL-flat` and
`mmL-hier` at training seeds 0/1/2, a 48-frame ground-truth warm-up, 300 self-fed target frames and
100 protected test clips. Both arms used the same 1,152 context-token budget, predictor regime,
codec, sampler and evaluation geometry. The generator saved persistent object and source-glyph
identity, boxes and trajectories; the evaluator used one-to-one Hungarian assignment, explicit
missed/extra-object penalties and intention-to-score coverage. The official train/test glyph split
and classifier-domain floors were verified before the holdout was opened.

**Table 4.8a — E6 persistent-object long Moving-MNIST.** Positive effects favor hierarchy; all
intervals belong to one simultaneous 95% max-statistic clip-cluster family over the 15
co-primary effects.

| Seed | Foreground MSE | Union MSE | Trajectory error, px | Identity accuracy | Object coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.0036 [−0.0074,+0.0146] | +0.0102 [+0.0019,+0.0185] | +2.4786 [+0.2130,+4.7443] | −0.0762 [−0.1106,−0.0418] | +0.0304 [+0.0016,+0.0591] |
| 1 | −0.0031 [−0.0127,+0.0065] | +0.0122 [+0.0052,+0.0192] | +1.4223 [−1.0646,+3.9092] | −0.0561 [−0.0934,−0.0188] | +0.0159 [−0.0163,+0.0482] |
| 2 | −0.0024 [−0.0131,+0.0083] | +0.0169 [+0.0093,+0.0245] | +4.1325 [+1.5271,+6.7380] | −0.0647 [−0.0955,−0.0338] | +0.0498 [+0.0161,+0.0834] |

The preregistered verdict is **MIXED_BOUNDARY**, not equivalence and not a null result. Union-mask
error favors hierarchy on all three seeds. Trajectory and coverage favor hierarchy cleanly on
seeds 0 and 2 and are directionally positive but unresolved on seed 1. Missed-object rate,
full-frame error, background error and foreground F1 also improve for hierarchy at all seeds as
secondary descriptive endpoints. Digit identity, however, favors flat context with simultaneous
intervals wholly below zero at all three seeds; foreground-only MSE is unresolved and changes
sign.

This division is consistent with the dissertation's systems interpretation. The coarse stream
retains spatial support economically—where an object is, whether it persists and how its motion
continues—but discards fine glyph information needed to preserve what the object is. Flat
full-resolution storage retains that identity fidelity better. Thus E6 extends, but does not
universally generalize, the LoopMom reach–fidelity frontier. It is a controlled synthetic external
boundary, not evidence of natural-video transfer.

![E6 persistent-object boundary](../../figures/fig22_phasec_v2_boundary.png)

**Figure 4.22 —** Persistent-object long Moving-MNIST simultaneous effects. Positive favors
hierarchy; support/continuity and fine identity fidelity separate.

### 4.6.3 Synthetic [A1]/[A2] Testbeds: 2D Rotations and 3D Polycubes

Two distinct published sources motivate this future bridge. [A2] supplies the $64 \times 64$
procedural 2-D rotation suite and its controlled temporal laws; [A1] supplies the V-BLT
2-D-polygon and 3-D-polycube sequences with entropy maps. Neither publication evaluates the
Chapter 3 reach–fidelity model. A future transfer study would therefore ask whether the selected
hierarchical configuration retains accuracy and consistency on these source-faithful suites,
using each publication's exact split, metrics, horizon and baselines. No such transfer has been
executed, so no result table or projected advantage is inserted here. The published examples
remain documented in Chapter 2, Figures 2.3–2.4.

### 4.6.4 Moving-MNIST-CIFAR

The MMNIST-CIFAR variant replaces the black background with natural-image texture, removing the
background degeneracy that makes aggregate metrics mean-seeking on standard Moving MNIST
(§3.4.3) while keeping the same no-re-entry kinematics — so the pre-registered expectation of
the §4.6.2 negative control (flat ≈ hier) carries over, now with more informative aggregate
metrics. This extension is not required to establish the present reach–fidelity result and has
not been executed; no numerical outcome is implied.

### 4.6.5 TaxiBJ: Measured Real Spatio-Temporal Forecasting Data

TaxiBJ is the one *real-data* suite of the transfer set: Beijing taxi in/out-flow grids,
$32 \times 32 \times 2$ (two flow channels), 4 → 4 frame prediction under the standard OpenSTL
protocol, scored by the benchmark's own metric implementation invoked verbatim (the
normalization convention — data rescaled to $[0,1]$ by the benchmark's loader — was verified in
the OpenSTL source before any model number was read). Instead of the anticipated dedicated
32-px model variant, the transfer used the **exact-upsample equivalent** (disclosed, registered
before the runs): frames nearest-upsampled ×2 through the *unchanged* 64-px two-level
autoencoder (2-channel input) and WindowDiT stack of Chapter 3, predictions downsampled back by
the exactly inverse 2×2 mean before scoring — an information-identical formulation that keeps
the architectural constants of §3.2 literally unchanged across every dataset of the study.
Regime: fresh autoencoder (12,000 steps), four arms at batch 64 × 15,000 steps (≈48 epochs,
comparable to the leaderboard's 50-epoch protocol), best-of-1, unconditional, seed 0.

**Table 4.9 — TaxiBJ 4→4, MEASURED (n = 500 canonical test pairs; OpenSTL convention;
leaderboard rows = OpenSTL's own 50-epoch retraining, locally verified).**

| Arm | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| **Flat, regression (256 ctx tokens)** | **0.3642** | **14.96** | 0.9781 | **39.83** |
| Hierarchical, regression (128 ctx tokens) | 0.4170 | 15.44 | 0.9772 | 39.64 |
| Hierarchical, diffusion (128 ctx tokens) | 0.4986 | 18.37 | 0.9692 | 37.87 |
| Flat, diffusion (256 ctx tokens) | 0.5152 | 18.44 | 0.9686 | 37.85 |
| AE round-trip ceiling (2-ch) | 0.0192 | 4.48 | 0.9987 | 50.92 |
| Persistence | 3.8268 | 48.63 | 0.7853 | 30.20 |
| TAU (OpenSTL, 50 ep) | 0.3108 | 14.93 | 0.9848 | 39.74 |
| MIM (OpenSTL, 50 ep) | 0.3110 | 14.96 | 0.9847 | 39.65 |
| PredRNN (OpenSTL, 50 ep) | 0.3194 | 15.31 | 0.9838 | 39.51 |
| SimVP+gSTA-S (OpenSTL, 50 ep) | 0.3246 | 15.03 | 0.9844 | 39.71 |
| PhyDNet (OpenSTL, 50 ep) | 0.3622 | 15.53 | 0.9828 | 39.46 |

The measured readings, stated per metric and without a leaderboard-superiority headline (the
§4.2.3 rule): the regression arm of the *unchanged generalist stack* lands **inside the
leaderboard band on MAE (14.96 — tying MIM for second-best of the quoted rows) and on PSNR
(39.83 — above every quoted row)**, at the band's lower edge on MSE (0.3642, 0.55% above
PhyDNet), and below the band on SSIM (0.9781 vs 0.9828–0.9848). Persistence is beaten ≈10.5×;
the autoencoder ceiling (0.0192, 16× below the best specialist row) shows the frozen tokenizer
is not the limiter — the pre-registered decoder-capacity fallback branch is measured-moot. The
regression-over-diffusion best-of-1 ordering replicates on real data (0.364/0.417 vs
0.515/0.499), and the no-memory-horizon boundary replicates as pre-registered: at half the
context tokens the hierarchical regression arm pays −0.053 MSE with faster per-horizon
degradation (4-frame protocol → coarse history is pure token economy). Disclosure: single seed
(a transfer row, not a claim-bearing contrast, per the campaign's two-seed rule).

### 4.6.6 KTH-64 — Future Extension

KTH (human actions, grayscale, evaluated at 64 × 64) is a natural-video extension named in the
research prospects; it is not an [A2] result. It is retained as a future external-validity test,
not as an optional empty result shell.

### 4.6.7 Boundaries of the Measured Generalization Claim

The measured generalization claim is deliberately narrow. On OpenSTL Moving MNIST the package
comparison is a no-re-entry negative control and does not match specialist aggregate MSE. On
TaxiBJ the unchanged generalist stack is competitive on some reported metrics, but the
hierarchical package loses to the larger flat regression context and only one training seed was
tested. The remaining synthetic and natural-video suites are future work. Accordingly, **a
claim of beating (or matching) the OpenSTL leaderboard on aggregate MSE is never made** — Table
4.8 refutes it (180.91 versus SimVP's 26.69), and the honest statement of that comparison is the
disclosed-generalist reading of §4.6.2.

### 4.6.8 Independent Confirmation on an Untouched LoopMom Holdout

The original test population informed qualitative monitoring and configuration selection.
Accordingly, the central reach claims were evaluated once on the newly generated, hash-pinned
holdout `loopmom-confirmation_holdout-20260712-e033abee6e04`. Before first access, the protocol
froze 26 checkpoint evaluations, two primary pairs, their geometric reach bands, the complete
metric suite and one studentized max-statistic clip-cluster family for 12 correlated frontier
contrasts. No model was trained or selected on this holdout.

For `hier-min` versus `flat-min`, the registered `14 < dt <= 48` effects are +0.0980
[0.0788, 0.1179] at seed 0 and +0.0914 [0.0736, 0.1098] at seed 1. For the B1
configuration versus `flat-short`, the `8 < dt <= 48` effects are +0.0692
[0.0543, 0.0845] and +0.0743 [0.0588, 0.0899]. Whole-population, union-mask,
full-frame, recall/F1, missed-component and coverage endpoints corroborate both directions.
B1 still changes its target interface and is therefore a whole-configuration result.

The simultaneous frontier confirms hierarchy at 896 and 1,152 context tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. Thus 1,152 is the
largest tested family-wise-confirmed hierarchy point and 2,432 the smallest tested confirmed flat
point. No monotonic curve, unique crossing or interpolation is inferred. In exact memory
arithmetic, 48 fine frames cost 3,072 context tokens, while 48 coarse frames plus two or six recent
fine frames cost 896 or 1,152. The hierarchy therefore provides the same nominal 48-frame reach
with 2.7–3.4 times fewer context tokens. This is a reach-efficiency statement, not a claim of
equal retained information or equal performance at every budget.

![Untouched frontier and exact context-token arithmetic](../../figures/article_confirmed_frontier.png)

**Figure 4.18 — Untouched reach–fidelity frontier and memory arithmetic.** Positive effects favor
hierarchy. All intervals in the left panel belong to one simultaneous family; the right panel
shows exact token counts for nominal 48-frame reach.
*Source:* `artifacts/confirmation_e2/results.json`; generated by
`scripts/make_rft_result_figures.py`.

### 4.6.9 Robustness to Sampling Budget and DDIM Randomness

E7c separates a property of the trained checkpoints from dependence on one default diffusion
allocation. Four frozen seed-0 contrasts—896-token reach, the 1,152-token interior point, the
near-matched 1,088-token temporal versus 1,152-token spatial package shape, and the 2,432-token
flat-fidelity inversion—were crossed with NFE 8/16/32 at sampler seed 0 and NFE 16 at sampler
seeds 1/2. All 20 effects were covered by one preregistered simultaneous family.

All 20 point estimates retain the registered direction and 18 simultaneous lower bounds exceed
zero. The reach, interior and package-shape contrasts are family-wise clean for every variant.
The upper inversion remains positive but is unresolved at sampler seed 1
(+0.0098 [−0.0001,+0.0196]) and seed 2 (+0.0077 [−0.0013,+0.0167]). The registered verdict is
therefore `DIRECTIONALLY_ROBUST`, not universal sampler invariance. Non-empty foreground,
full-frame, background and union-mask error, recall and F1 agree in all 20 variants; component
counts are not uniformly favorable.

![Sampler and NFE robustness](../../figures/fig19_sampler_nfe_robustness.png)

**Figure 4.19 — Robustness to NFE and sampler randomness.** The two open upper-inversion points
have simultaneous intervals that include zero.
*Source:* `artifacts/e7c_sampler_nfe/results.json`.

### 4.6.10 Robustness to an Independently Trained Latent Representation

E7a is a whole-codec intervention: a second codec seed is trained independently, normalization is
recomputed, and the same seven seed-0 diffusion predictors are retrained in the new latent space.
It does not isolate codec weights, reconstruction error, latent scale or normalization. Four
original-codec and four alternate-codec effects share one new purpose-locked holdout and one
simultaneous family.

**Table 4.10 — Robustness of the four registered directions to an independent codec.**

| Codec | 896 reach | 1,152 interior | 1,088 vs 1,152 shape | 2,432 inversion |
|---|---:|---:|---:|---:|
| Original | +0.0494 [0.0339, 0.0649] | +0.0283 [0.0169, 0.0397] | +0.0253 [0.0163, 0.0343] | +0.0150 [0.0065, 0.0235] |
| Alternate | +0.0521 [0.0374, 0.0669] | +0.0439 [0.0310, 0.0569] | +0.0112 [0.0039, 0.0184] | +0.0079 [−0.0006, 0.0165] |

All eight point estimates retain direction and seven simultaneous lower bounds exceed zero. The
sole unresolved interval is the alternate-codec 2,432-token inversion. The verdict is
`CODEC_DIRECTIONALLY_ROBUST`. Codec seed is not an inferential replicate, so descriptive effect
changes do not identify a component-level latent-space cause.

![Alternate-codec robustness](../../figures/fig20_alternate_codec_robustness.png)

**Figure 4.20 — Whole-codec robustness.** The open alternate-codec upper-inversion interval
includes zero.
*Source:* `artifacts/e7a_codec_robustness/results.json`.

### 4.6.11 Privileged Pose-Indexed Explicit-Access Reference

E8a separates having relevant history from addressing and reusing it. A fixed world-coordinate
mosaic is built from raw warm-up pixels and queried at the exact saved future viewport positions.
Unobserved columns use a row prior computed only from training data. The mosaic receives raw
pixels and future poses, has no token-matched budget and is not learned; it is therefore a
privileged access reference, not an architecture or efficiency competitor.

**Table 4.11 — Simultaneous E8a effects for pose-indexed raw memory.**

| Registered contrast | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| Raw K14 minus raw K48 | +0.0723 | [+0.0548, +0.0898] |
| Raw K38 minus raw K48 | +0.0210 | [+0.0096, +0.0325] |
| `hier-min` minus raw K48 | +0.1219 | [+0.1031, +0.1407] |
| `hier-long` minus raw K48 | +0.1368 | [+0.1190, +0.1545] |
| `strided-flat` minus raw K48 | +0.1134 | [+0.0944, +0.1324] |
| `flat-c2` minus raw K48 | +0.1310 | [+0.1122, +0.1498] |

K48 reaches foreground MSE 0.0494, full-frame MSE 0.0027 and foreground recall 0.8438, versus
foreground MSE 0.1628–0.1862 for the four neural references. All neural-minus-K48 lower bounds
are positive, yielding `RAW_ACCESS_CEILING_CI_CLEAN`. Median stored world columns rise from 80 at
K6 to 154 at K48, and mean observed future-crop coverage from 0.454 to 0.809. The result shows
that reach alone does not guarantee reliable access; it does not rank the privileged mosaic
against deployable neural models.

![Privileged pose-indexed raw memory](../../figures/fig21_pose_indexed_raw_memory.png)

**Figure 4.21 — Pose-indexed raw-memory reference.** Coverage grows with the stored span and all
six registered K48 effects have positive simultaneous lower bounds.
*Source:* `artifacts/e8a_pose_mosaic/results.json`.

## 4.7 Implementation and Use of the Research Results

The results of the dissertation research have been implemented in:

- the **practical activities of ORNETIKA LLC** (Kyiv): the developed information technology was
  used in designing intelligent systems for the analysis of spatio-temporal video data and
  anomaly detection in complex technical environments; the resource-efficient methods (the
  TempVerseFormer reversible-transformer method) and adaptive entropy-based tokenization
  substantially reduced the computational cost of modeling long temporal sequences
  (implementation act of October 14, 2025);
- the **educational process of VNTU** in the disciplines *Information Technologies for Monitoring
  and Analysing the State of Complex Systems* and *Internet of Things and Intelligent Data
  Analysis* for PhD
  students of specialty 124 (implementation act of September 19, 2025).

Copies of the implementation acts are given in the appendices **[AUTHOR: assign the final
appendix letters and insert the signed act copies]**.

## 4.8 Chapter Conclusions

1. In accordance with **Task 7** of the dissertation research, comprehensive experimental studies
   of the developed methods and the intelligent information technology were conducted on the
   two-operating-point testbed family (§4.1) under an audit-hardened measurement protocol:
   paired clip-level cluster bootstrap confidence intervals (frame-iid intervals were shown to be
   2–7× too narrow at lag-1 autocorrelation +0.86), exact analytic hardness stratification with
   an audit-corrected anchoring, common random numbers, per-operating-point reference floors and
   ceilings (Table 4.1), and full disclosure of every correction and caveat (§4.1.5). The [A2]
   receipt datasets (2D/3D rotations, Moving MNIST, MMNIST-CIFAR, TaxiBJ, KTH) are retained as
   the published validations of the reversibility method; the OpenSTL bridge is **measured**
   (§4.6.2, Table 4.8): the protocol anchor is achieved, and the pre-registered "mid-pack
   aggregate MSE" clause is refuted and reported as such.

2. The matched-capacity and matched-NFE mechanism study (§4.2) established that **the generation-head ranking is
   regime-dependent**: the diffusion core wins the single-step oracle-coverage regime on every
   column at matched capacity and NFE (fg-MSE 0.0512 vs 0.0664 overall; 0.1549 vs 0.1808 on the
   hard, genuinely generative stratum), while the regression core wins the deterministic
   best-of-1 rollout at equal sharpness (ALL delta CI [+0.0110, +0.0227]) — a refutation of the
   pre-registered fade expectation, reported as such. No statement of the form "one best head"
   survives the chapter; the memory claims were therefore verified under both viable heads.

3. The energy-based head was characterized by the two-sided capability study (§4.2.4) with the
   official implementation and both published recipes at 20,000 converged steps: landscape
   ordering is recipe-dependent (S2 stably mis-ordered, 21/21 probes; S1 correctly ordered,
   15/15 probes from step 6,000), while descent reachability fails under both recipes and under
   the paper's own inference procedure ($d_{\mathrm{tgt}} \approx 1.12$–$1.29$ against the mean's
   0.964; foreground sharpness ≤ 0.035 against the 0.08 structure threshold). The result is a
   **characterized negative** — a capability boundary at 16 GB-reachable scale with the
   configuration gap stated (Table 4.4), not a reproduction failure — and it empirically
   justifies the diffusion selection of Chapter 3 jointly with conclusion 2.

4. The headline claim of the dissertation was confirmed (§4.3): **at an equal context-token
   budget (1,152 tokens), the hierarchical long-coarse-memory model beats the flat control on
   152-frame autoregressive rollouts on two seeds and under both heads** — seed 0 ALL +0.0189
   [+0.0121, +0.0264] with the win concentrated from h ≥ 25 on hard re-entry bins (h 25–60 HARD
   +0.0344 [+0.0162, +0.0531]); seed 1 ALL +0.0354 [+0.0267, +0.0447], winning every cell;
   regression pairing ALL +0.0334 seed 0 and +0.0501 seed 1 — two-seed. The single-step memory
   gate showed the
   underlying retrieval capability emerging late in training (null at 6,000 steps, ≈13% hard-bin
   win at 20,000 on both seeds). The claim is honestly bounded: the onset horizon is
   seed-variable (the robust claim is the win and its mid-horizon concentration); sharpness fades
   for all arms at depth (the G2b retention clause is not met at h ≥ 81; the ceiling is owned by
   the frozen decoder); and flat-long at 2.7× the tokens is the outright Pareto frontier
   (0.1564) — at operating point A, hierarchy is an efficiency point, not a capability necessity.

5. The spatial-abstraction factorial (§4.4) returned an **economic, head-conditional** verdict.
   At the fixed 18-frame window, coarse history costs diffusion −0.0082 [−0.0115, −0.0050] at
   42% token savings and is statistically free for regression (+0.0008 [−0.0013, +0.0029]); its
   value is what the savings buy. At 48-frame reach, temporal full-resolution anchors beat spatial
   coarsening under diffusion on all five seeds, while regression has two temporal and three
   spatial material winners. The head × package interaction changes sign and its five-seed t95
   crosses zero. The robust conclusion is a stable diffusion package result plus regression and
   cross-head training-seed heterogeneity, not a directional head law or one universally optimal
   package. The separate E7c inference study (§4.6.9) preserved the expected point-estimate
   direction in all 20 registered NFE/sampler variants, with 18 simultaneous lower bounds above
   zero. The 2,432-token inversion is family-wise unresolved at alternate sampler seeds 1/2;
   therefore the result is directionally robust rather than universally sampler-invariant.
   The independent-codec study E7a (§4.6.10) likewise preserves all four point-estimate
   directions: the 896 reach, 1,152 interior and package-shape contrasts remain
   simultaneous-CI clean, while the alternate-codec 2,432 inversion is +0.0079
   [−0.0006, 0.0165]. Thus the reach–fidelity account is directionally robust to both inference
   allocation/randomness and one whole-codec intervention, while its upper tested edge remains
   the precision-limited boundary. The privileged explicit-access study E8a (§4.6.11) further
   gives a clean simultaneous reference: raw K48 foreground MSE is 0.0494, and all four
   registered neural-minus-K48 lower bounds are positive. This is evidence that geometric reach
   and reliable addressing are distinct, not a token-matched model comparison.

6. The temporal-capability study at operating point B (§4.5) closed with a characterized
   single-step positive and a **registered negative replication of the proposed late trust
   boundary**: retrieval beyond the flat window exists on the position-unique wide world and is
   reach-causal (dose-response with the internal contiguous control), a memory span exceeding the
   effective loop $L_{\mathrm{eff}}$ (eq. 4.1) reliably enables single-step retrieval learning
   (2/2 span > loop runs CI-positive vs 0/4 span ≤ loop). The early rollout benefit is positive
   at all three training seeds, whereas the late inversion appears only at seed 0 and reverses at
   seeds 1–2. No stable trust boundary is therefore claimed. The negative replication, together
   with the three methodology lessons (effective-loop sizing; the autoencoder-reuse trap;
   evaluation power), constitutes the honest delimitation of this mechanism.

7. The generalization study (§4.6) carries the measured per-regime configuration selection —
   regression head for the deterministic best-of-1 rollout, diffusion for the single-step
   coverage regime, hierarchical coarse-plus-fine context at the largest affordable window —
   onto the standard and synthetic suites. Its measured component, the OpenSTL Moving-MNIST
   bridge (Table 4.8), achieves the protocol anchor (bit-verified standard metrics; the
   autoencoder ceiling 3.30 shows the tokenizer is not the limiter) and returns the **predicted
   negative control**: flat ≈ hierarchical (180.91 vs 181.29 aggregate MSE) on a benchmark with
   no off-screen re-entry — hierarchy pays where the memory horizon exists (§4.3) and is a null
   where it does not, while the aggregate-MSE gap to the deterministic specialists (22.06–29.80)
   is disclosed as the refutation of the pre-registered "mid-pack" clause, and no leaderboard
   claim is made. The valid persistent-object long Moving-MNIST study (§4.6.2a, Table 4.8a)
   returns a separate **mixed boundary**: hierarchy improves union error at all three seeds and
   trajectory/coverage cleanly at two, while flat context preserves digit identity at all three
   and foreground-only error is unresolved. It therefore supports a support/continuity-versus-
   identity-fidelity interpretation rather than universal superiority. TaxiBJ is a measured
   real-data transfer (§4.6.5); the unexecuted 2D-rotation,
   3D-polycube, MMNIST-CIFAR and optional KTH-64 protocols are retained only as future-work
   designs and contribute no projected result to this conclusion. The token-budget frontier
   completion (§4.3.6; hier-rich at 3,072 tokens, flat-short at 512) is **measured**: the
   frontier inverts at both ends (hier-rich loses to flat-long −0.0131 [−0.0201, −0.0069] at
   equal budget and regime; flat-short beats hier-short −0.0036 [−0.0063, −0.0010]) — an outcome
   stricter than the pre-registered convergence expectation, reported as measured. The final
   bounded efficiency claim is therefore discrete and empirical: hierarchy is family-wise
   confirmed at the tested 896- and 1,152-token points, unresolved at 1,792, and full-resolution
   flat context is confirmed at 2,432 and 3,072; no untested interval or unique crossing is
   inferred.

8. The results of the research have been implemented in the practical activities of ORNETIKA LLC
   and in the educational process of VNTU (§4.7, per the implementation acts).

The results of this chapter have been published in [A4] (the mechanism, memory, factorial, and
boundary studies) and are connected to [A2] through the published receipt summarized in §4.1.2.
The stable labels are bound mechanically to the final numbered publication list.

9. The chapter's findings consolidate into **scoped empirical regularities and explicit
   boundaries** (single source of truth: `docs/CLAIM_LEDGER.md`; every statement names its
   testbed, head, representation and seed level): **(L1) reach** — long-span packaging wins at
   tested lower/interior budgets when it contains the relevant re-entry past and the comparator
   does not; **(L2) fidelity inversion** — the simultaneous family confirms hierarchy at
   896/1,152, leaves 1,792 unresolved and confirms flat at 2,432/3,072, without interpolation;
   **(L3) stride-family configuration** — the 512-token stride-2 arm wins strongly but changes
   its target interface; **(L4) package shape** — sparse full-resolution history wins under
   diffusion on five seeds, while regression and the cross-head interaction are
   training-seed-heterogeneous; **(L5) depth — hypothesis, not law** — a
   depth-plus-objective-allocation probe on single-scale data; **(L6) self-fed reliability —
   negative replication, not law** — the seed-0 late inversion is not reproduced at seeds 1–2;
   and **(F4) inference robustness** — E7c retains all 20 NFE/sampler effect directions, with
   18 simultaneous lower bounds positive and the upper inversion unresolved at alternate sampler
   seeds 1/2; **(F5) representation robustness** — E7a retains all four alternate-codec point
   directions, with three family-wise clean contrasts and the upper inversion again
   precision-limited; **(F6) explicit-access reference** — E8a gives a clean privileged
   raw-memory ceiling while remaining outside token-matched model ranking; and **(X3) external
   persistent-object boundary** — E6 separates hierarchy-favoring spatial support/continuity from
    flat-favoring digit identity under one valid protected simultaneous family. Together, these
    confirmed regularities and explicit boundaries form a scoped, falsifiable theory of context
    allocation—the main scientific conclusion of the chapter.

---

<div style="page-break-after: always;"></div>

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
constant-memory reversible method of Chapter 2 providing an implementation path [80]. Second, the **extension
of the memory-horizon protocol to real video**: transferring the re-entry-stratified evaluation
methodology from the analytic testbed family to natural-video data of the KTH class (where
hardness labels must be estimated rather than computed exactly), building on the completed
OpenSTL bridge as the connectivity anchor and executing the future-work protocols of §4.6.
These directions require no revision of the present results: they extend a boundary currently
stated only for its measured testbeds and configurations.

---

<div style="page-break-after: always;"></div>

# References

External sources are numbered by first appearance in the assembled dissertation.

1. Liu Y., Zhang K., Li Y., Yan Z., Gao C., Chen R., Yuan Z., Huang Y., Sun H., Gao J., He L., Sun L. Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arXiv:2402.17177, 2024.
2. Lipton Z. C. The Mythos of Model Interpretability. *Communications of the ACM* 61(10), 2018.
3. Rudin C. Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. *Nature Machine Intelligence* 1, 2019.
4. Ho J., Jain A., Abbeel P. Denoising Diffusion Probabilistic Models. *NeurIPS 33*, 2020.
5. von Bertalanffy L. *General System Theory: Foundations, Development, Applications*. George Braziller, 1968.
6. Goodfellow I., Pouget-Abadie J., Mirza M., et al. Generative Adversarial Networks. *NeurIPS 27*, 2014.
7. Kingma D. P., Welling M. Auto-Encoding Variational Bayes. *ICLR*, 2014.
8. Згуровський М. З., Панкратова Н. Д. *Основи системного аналізу: підручник*. Київ: Видавнича група BHV, 2007. 544 с. ISBN 978-966-552-153-2.
9. Zgurovsky M. Z., Pankratova N. D. *System Analysis: Theory and Applications*. 1st ed. Springer Berlin, Heidelberg, 2007. XXII, 447 p. DOI: 10.1007/978-3-540-48880-4.
10. Mesarović M. D., Macko D., Takahara Y. *Theory of Hierarchical, Multilevel Systems*. Academic Press, 1970.
11. Bond-Taylor S., Leach A., Long Y., Willcocks C. G. Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models. *IEEE TPAMI* 44(11), 2022.
12. Rombach R., Blattmann A., Lorenz D., Esser P., Ommer B. High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*, 2022.
13. Saxena V., Ba J., Hafner D. Clockwork Variational Autoencoders. *NeurIPS 34*, 2021.
14. Zakharov A., Guo Q., Fountas Z. Long-Horizon Video Prediction Using a Dynamic Latent Hierarchy. arXiv:2212.14376, submitted 2022, revised 2023.
15. Yan W., Hafner D., James S., Abbeel P. Temporally Consistent Transformers for Video Generation (TECO). *ICML*, 2023.
16. Ji Z., Lee N., Frieske R., et al. Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys* 55(12), 2023.
17. Song J., Meng C., Ermon S. Denoising Diffusion Implicit Models. *ICLR*, 2021.
18. Peebles W., Xie S. Scalable Diffusion Models with Transformers (DiT). *ICCV*, 2023.
19. OpenAI (Brooks T., Peebles B., et al.). Video Generation Models as World Simulators (Sora technical report), 2024.
20. Pasukonis J., Lillicrap T., Hafner D. Evaluating Long-Term Memory in 3D Mazes (Memory Maze). arXiv:2210.13383, 2022.
21. Po R., Nitzan Y., Zhang R., Chen B., Dao T., Shechtman E., Wetzstein G., Huang X. Long-Context State-Space Video World Models. arXiv:2505.20171, 2025.
22. Wu T., Yang S., Po R., Xu Y., Liu Z., Lin D., Wetzstein G. Video World Models with Long-term Spatial Memory. arXiv:2506.05284, 2025.
23. Oshima Y., Iwasawa Y., Suzuki M., Matsuo Y., Furuta H. WorldPack: Compressed Memory Improves Spatial Consistency in Video World Modeling. arXiv:2512.02473, 2025.
24. Ranzato M., Chopra S., Auli M., Zaremba W. Sequence Level Training with Recurrent Neural Networks (exposure bias). *ICLR*, 2016.
25. Bengio S., Vinyals O., Jaitly N., Shazeer N. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. *NeurIPS 28*, 2015.
26. Dosovitskiy A., Beyer L., Kolesnikov A., et al. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT). *ICLR*, 2021.
27. Arnab A., Dehghani M., Heigold G., et al. ViViT: A Video Vision Transformer. *ICCV*, 2021.
28. Liu Z., Ning J., Cao Y., et al. Video Swin Transformer. *CVPR*, 2022.
29. Pagnoni A., Pasunuru R., Rodriguez P., et al. Byte Latent Transformer: Patches Scale Better Than Tokens. arXiv:2412.09871, 2024.
30. van den Oord A., Vinyals O., Kavukcuoglu K. Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 30*, 2017.
31. Yan W., Zhang Y., Abbeel P., Srinivas A. VideoGPT: Video Generation using VQ-VAE and Transformers. arXiv:2104.10157, 2021.
32. Chandratreya I. P., Charatan D., Van Hoorick B., Zakharov S., Guizilini V., Isola P., Sitzmann V. MilliVid: Hierarchical Latents for Long-Range Consistency in Video Generation. arXiv:2606.09056, 2026.
33. Kim T., Ahn S., Bengio Y. Variational Temporal Abstraction. *NeurIPS 32*, 2019.
34. Koutník J., Greff K., Gomez F., Schmidhuber J. A Clockwork RNN. *Proceedings of Machine Learning Research* 32(2):1863–1871, ICML, 2014.
35. LeCun Y., Chopra S., Hadsell R., Ranzato M., Huang F. A Tutorial on Energy-Based Learning. In *Predicting Structured Data*, MIT Press, 2006.
36. Ackley D. H., Hinton G. E., Sejnowski T. J. A Learning Algorithm for Boltzmann Machines. *Cognitive Science* 9(1), 1985.
37. Hyvärinen A. Estimation of Non-Normalized Statistical Models by Score Matching. *JMLR* 6, 2005.
38. Song Y., Ermon S. Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS 32*, 2019.
39. Gladstone A., Nanduru G., Islam M. M., Han P., Ha H., Chadha A., Du Y., Ji H., Li J., Iqbal T. Energy-Based Transformers are Scalable Learners and Thinkers. arXiv:2507.02092, 2025.
40. Du Y., Mordatch I. Implicit Generation and Modeling with Energy-Based Models. *NeurIPS 32*, 2019.
41. Nijkamp E., Hill M., Zhu S.-C., Wu Y. N. On Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model. *NeurIPS 32*, 2019.
42. Chen T., Xu B., Zhang C., Guestrin C. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174, 2016.
43. Griewank A., Walther A. Algorithm 799: revolve — An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation. *ACM TOMS* 26(1), 2000.
44. Chen R. T. Q., Rubanova Y., Bettencourt J., Duvenaud D. Neural Ordinary Differential Equations. *NeurIPS 31*, 2018.
45. Gomez A. N., Ren M., Urtasun R., Grosse R. B. The Reversible Residual Network: Backpropagation Without Storing Activations. *NeurIPS 30*, 2017.
46. Mangalam K., Fan H., Li Y., et al. Reversible Vision Transformers. *CVPR*, 2022.
47. Khalil H. K. *Nonlinear Systems* (3rd ed.). Prentice Hall, 2002.
48. Heusel M., Ramsauer H., Unterthiner T., et al. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID). *NeurIPS 30*, 2017.
49. Saharia C., Chan W., Chang H., et al. Palette: Image-to-Image Diffusion Models. *SIGGRAPH*, 2022.
50. Ruiz N., Li Y., Jampani V., et al. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. *CVPR*, 2023.
51. Hu E. J., Shen Y., Wallis P., et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*, 2022.
52. Zhang L., Rao A., Agrawala M. Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet). *ICCV*, 2023.
53. Hertz A., Voynov A., Fruchter S., Cohen-Or D. Style Aligned Image Generation via Shared Attention. *CVPR*, 2024.
54. Marr D. *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*. W. H. Freeman, 1982.
55. Radford A., Kim J. W., Hallacy C., et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*, 2021.
56. Vaswani A., Shazeer N., Parmar N., et al. Attention Is All You Need. *NeurIPS 30*, 2017.
57. Goodfellow I., Bengio Y., Courville A. *Deep Learning*. MIT Press, 2016.
58. Shaw P., Uszkoreit J., Vaswani A. Self-Attention with Relative Position Representations. *NAACL*, 2018.
59. Su J., Ahmed M., Lu Y., et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing* 568, 2024 (arXiv:2104.09864, 2021).
60. Rajbhandari S., Rasley J., Ruwase O., He Y. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC20*, 2020.
61. Korthikanti V., Casper J., Lym S., McAfee L., Andersch M., Shoeybi M., Catanzaro B. Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198, 2022.
62. Werbos P. J. Backpropagation Through Time: What It Does and How to Do It. *Proceedings of the IEEE* 78(10), 1990.
63. Williams R. J., Peng J. An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories (truncated BPTT). *Neural Computation* 2(4), 1990.
64. MacKay M., Vicol P., Ba J., Grosse R. B. Reversible Recurrent Neural Networks. *NeurIPS 31*, 2018.
65. Behrmann J., Grathwohl W., Chen R. T. Q., Duvenaud D., Jacobsen J.-H. Invertible Residual Networks (i-ResNet). *ICML*, 2019.
66. Behrmann J., Vicol P., Wang K.-C., Grosse R., Jacobsen J.-H. Understanding and Mitigating Exploding Inverses in Invertible Neural Networks. *AISTATS*, 2021.
67. Maclaurin D., Duvenaud D., Adams R. P. Gradient-Based Hyperparameter Optimization Through Reversible Learning. *ICML*, 2015.
68. Wallace B., Gokul A., Ermon S., Naik N. End-to-End Diffusion Latent Optimization Improves Classifier Guidance (DOODL). *ICCV*, 2023.
69. Wallace B., Gokul A., Naik N. EDICT: Exact Diffusion Inversion via Coupled Transformations. *CVPR*, 2023.
70. Zhuang J., Dvornek N. C., Tatikonda S., Duncan J. S. MALI: A Memory-Efficient and Reverse-Accurate Integrator for Neural ODEs. *ICLR*, 2021.
71. Prabhudesai M., Goyal A., Pathak D., Fragkiadaki K. Aligning Text-to-Image Diffusion Models with Reward Backpropagation (AlignProp). arXiv:2310.03739, 2023.
72. Clark K., Vicol P., Swersky K., Fleet D. J. Directly Fine-Tuning Diffusion Models on Differentiable Rewards (DRaFT). *ICLR*, 2024.
73. Salimans T., Ho J. Progressive Distillation for Fast Sampling of Diffusion Models. *ICLR*, 2022.
74. Tan C., Li S., Gao Z., et al. OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning. *NeurIPS 36 (D&B)*, 2023.
75. Denton E., Fergus R. Stochastic Video Generation with a Learned Prior (SVG). *ICML*, 2018.
76. Lee A. X., Zhang R., Ebert F., et al. Stochastic Adversarial Video Prediction (SAVP). arXiv:1804.01523, 2018.
77. Alonso E., Jelley A., Micheli V., et al. Diffusion for World Modeling: Visual Details Matter in Atari (DIAMOND). *NeurIPS 37*, 2024.
78. Häggbom M., Karlsmark M., Andén J. Mean-Field Microcanonical Gradient Descent. arXiv:2403.08362, 2024.
79. Davies T., Huang Y., Gladstone A., Liu Y., Chen X., Ji H., Liu H., Hu L. EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities. arXiv:2510.27545, 2025.
80. Huang X., Li Z., He G., Zhou M., Shechtman E. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. *NeurIPS 2025* (spotlight); arXiv:2506.08009.
