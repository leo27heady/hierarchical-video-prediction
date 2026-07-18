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
