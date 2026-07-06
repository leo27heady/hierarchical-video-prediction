# 01 — ВСТУП (rewrite, phase 1)

> **Status:** English working text of the full ВСТУП replacement; the Ukrainian pass translates it
> mechanically. Genre spots marked **[UA-PASS: …]**, author-input spots **[AUTHOR: …]**.
> **Sources of truth:** `docs/dissertation/DISS_DIGEST.md` (esp. §3 items 10–14, §6, §8 decisions
> Q1–Q7), `docs/draft/ARTICLE_SKELETON.md`, `docs/draft/CH2_TESTBED.md`,
> `docs/draft/CH3_MECHANISM.md`. Structure follows the committee-expected order of the original
> draft: актуальність → зв'язок з НДР → мета і задачі → об'єкт/предмет → методи → новизна →
> практичне значення → особистий внесок → апробація → публікації → структура.

---

## Обґрунтування вибору теми дослідження (Justification / relevance)

The rapid development of information technologies and artificial-intelligence methods over the
last decade has produced new classes of mathematical models capable of solving complex problems of
analysis, prediction, and synthesis of multidimensional data. A special place among them is
occupied by generative video models, which have evolved from simple statistical tools into highly
complex multi-level systems capable of forming high-quality spatio-temporal video data. Such
systems are critically important for many applied fields: from medical diagnostics and
environmental monitoring to the prediction of the state of technical systems and the construction
of simulators for autonomous control. *(Register and application list preserved from the original
draft.)*

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
trajectories unattainable on practically available computing systems. [UA-PASS: «експозиційне
зміщення (exposure bias)» kept with the EN term in parentheses.]

A natural remedy is the hierarchical decomposition of the latent state space. However, the
author's own experimental investigations show that hierarchical video models built over
*discrete* (vector-quantized) token spaces are unstable to train at practically reachable
computational scale. This motivates the direction developed in this dissertation: hierarchical
modeling of video generation processes in **compact continuous hierarchical latent state
spaces**, in which the upper (coarse) level is a low-rate code that carries the long-range
temporal memory of the process, context signals are transferred between levels unidirectionally
(with gradient detachment), per-frame generation is performed by an iterative refinement
(denoising diffusion) core selected by controlled mechanism analysis, and the memory barrier of
long-horizon training is removed by reversible transformations that make backpropagation
time-agnostic. **[AUTHOR: cross-reference where the VQ-instability finding will be documented in
Chapter 1 (own experiments); DISS_DIGEST Q1 authorizes using it as motivation.]**

Thus an objective systemic contradiction arises between the necessity of ensuring multi-level
consistency of video data over long time horizons (which requires long memory horizons and deep
hierarchies) and the strict resource constraints of modern computing systems (the attention-memory
blow-up of flat contexts and the linear growth of training memory with horizon length). Resolving
this contradiction requires the creation of new mathematical methods of decomposition,
resource-efficient control, and intelligent information technologies of systems analysis, which
determines the high relevance of this dissertation research.

## Зв'язок роботи з науковими програмами, планами, темами

The dissertation was carried out in accordance with the research plans of Vinnytsia National
Technical University, in particular within the state-budget research topics:

- Research topic No. **[AUTHOR: НОМЕР НДР 1]** «Розробка інформаційних технологій та математичних
  методів інтелектуального аналізу просторово-часових даних у складних системах» (2021–2023), in
  which the applicant developed the method of adaptive estimation of spatio-temporal uncertainty
  and entropy-based tokenization of video data;
- Research topic No. **[AUTHOR: НОМЕР НДР 2]** «Системний аналіз і моделювання багатокрокових
  процесів в умовах ресурсних обмежень та невизначеності» (2024–2026), in which the applicant
  developed the method of computational-resource optimization based on reversible transformations
  and the **hierarchical generative model of video data**. **[AUTHOR: per DISS_DIGEST item 13,
  verify this wording against the actual НДР report text — the old draft said «ієрархічну
  енергетичну модель генерації», which must not survive.]**

## Мета і задачі дослідження

**The purpose of the dissertation research** is to improve the efficiency and the temporal and
spatial consistency of hierarchical video data generation processes through the development of
systems analysis methods, mathematical models, and intelligent information technologies.
[UA-PASS: «Метою дисертаційного дослідження є підвищення ефективності та просторової і часової
консистентності процесів ієрархічної генерації відеоданих шляхом створення методів системного
аналізу, математичних моделей та інтелектуальних інформаційних технологій.» Note: no
discrete-space commitment, per decisions Q1/Q2.]

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

5. Develop a hierarchical mathematical model of multi-level video data generation in compact
   hierarchical latent state spaces, in which the upper (coarse) level carries the long-range
   temporal memory of the process, with unidirectional inter-level transfer of context signals
   (gradient detachment between abstraction levels) and a per-frame iterative-refinement
   generation core selected by controlled mechanism analysis. *(Rewritten per DISS_DIGEST item 11
   + decision Q1: continuous-latent formulation, no Markov-optimization/energy commitment.)*

6. Create an intelligent information technology and a software complex for the systems analysis
   of hierarchical video generation processes, integrating the developed methods of adaptive
   tokenization, resource optimization, and hierarchical generation, together with a
   purpose-built memory-horizon testbed and re-entry-stratified evaluation protocol as its
   measurement instrument.

7. Conduct comprehensive experimental studies and approbation of the developed methods and
   technology on synthetic and open datasets — including a matched-capacity, matched-compute
   comparison of prediction mechanisms — and compare their efficiency and consistency with
   modern analogues.

## Об'єкт і предмет дослідження

**The object of research** is the processes of mathematical modeling and hierarchical generation
of multidimensional video data as complex systems under structural and computational uncertainty.
*(«ітеративної» → «ієрархічної»; no discrete-space framing.)*

**The subject of research** is the mathematical models, methods, and intelligent information
technologies of systems analysis, uncertainty estimation, and control in hierarchical generative
systems.

## Методи дослідження

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
including backpropagation, for the identification of parameters of the mathematical models; and
methods of statistical inference (paired bootstrap confidence intervals, common random numbers)
for the experimental evaluation protocol. *(Rewritten per DISS_DIGEST item 12: the old
«теорія марковських процесів та стохастичні диференціальні рівняння для моделювання динаміки
Ланжевена та MCMC-керування» clause is trimmed; Langevin/MCMC survive only as mechanism-study
vocabulary.)*

## Наукова новизна отриманих результатів

1. **For the first time**, a methodology for the systems analysis and evaluation of long-range
   memory in hierarchical video generation processes was developed — a purpose-built
   memory-horizon testbed whose re-entry hardness labels are exactly computable from recorded
   control tracks, with a re-entry-stratified evaluation protocol and a matched-context-token-budget
   comparison design — and it was demonstrated with its help that a hierarchical latent state
   space whose upper (coarse) level carries the long-range temporal memory of the process
   (developing the coarse-level-memory paradigm of hierarchical latent video models, §1.3.2)
   restores content that re-enters the field of view beyond the horizon of a flat context window
   at the same context-token budget; unlike existing evaluation practice, which aggregates
   prediction error without isolating memory-dependent content, the developed methodology makes
   the memory contribution of the hierarchy directly measurable and localizes the improvement of
   temporal consistency exactly in the regime where the memory horizon demands it.

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

*(Wording identical to 00_FRONT_MATTER.md §2 — keep the two files in sync. Item 1 re-scoped
2026-07-07 (audit fix): «вперше» now carried by the testbed methodology + matched-budget
demonstration, with the coarse-memory architecture credited to adopted prior art per §1.3.2 —
supersedes the DISS_DIGEST §6 wording. No "guarantees" language anywhere — the old novelty item
1's «математично гарантує» claim is replaced by the empirically demonstrated, exactly scoped
claim.)*

## Положення, що виносяться на захист

**[AUTHOR: confirm whether the VNTU spec-124 council template expects an explicit
defended-theses list; if the council treats the scientific-novelty items as the defended
propositions, replace this block with one sentence stating exactly that.]**

The following propositions are submitted for defense:

1. The hierarchical organization of a compact latent state space, in which the upper (coarse)
   level carries the long-range temporal memory of the process, improves the temporal
   consistency of long generated video sequences at a matched context-token budget exactly in
   the regime where the task's memory horizon exceeds the affordable flat context window — and
   this regime is identified, measured, and bounded by the developed memory-horizon evaluation
   methodology with re-entry-stratified analytic hardness labels.

2. Reversible transformations enable time-agnostic backpropagation with constant activation
   memory in the length of the modeled process; the method extends to stochastic multi-level
   recurrences through exact noise storage (gradients coinciding with backpropagation through
   time), and its domain of applicability is delimited by the three-axis reversibility taxonomy:
   the temporal-rollout axis pays, while the inner-optimization-loop axis cannot pay against
   gradient checkpointing.

3. The consistency-criteria apparatus extended with conceptual, hierarchical, and trajectory
   consistency — the latter operationalized through autoregressive-rollout degradation metrics
   stratified by re-entry hardness — provides a measurable, statistically disciplined decision
   procedure for the acceptance or rejection of configurations of complex generative systems.

4. The adaptive decomposition of spatio-temporal data based on entropy metrics with
   multidimensional rotary positional embedding allows dynamic allocation of computational
   resources to high-information zones of complex visual scenes and mathematically aligns the
   coordinate systems of the abstraction levels of hierarchical models.

## Практичне значення отриманих результатів

The practical significance lies in the development of a complex of algorithms and
software-information tools implementing the developed intelligent information technology for the
systems analysis of hierarchical video generation processes (the reversible-transformer method,
entropy-based adaptive tokenization, the hierarchical generation model, and the memory-horizon
evaluation instrument; Python/PyTorch). Unique synthetic spatio-temporal datasets were created for
testing generation and optimization systems: 2D shapes, 3D polycubes, and the looping
memory-horizon testbed with recorded action tracks and exactly computable hardness labels. The
developed technology was implemented in the practical activities of ORNETIKA LLC (implementation
act of October 14, 2025) and in the educational process of VNTU (implementation act of September
19, 2025). **[AUTHOR: per decision Q5 no copyright certificate has been filed — drop the
certificate sentence, or file under a new title without "HVEBT" and insert the number.]**

## Особистий внесок здобувача

All theoretical and experimental results submitted for defense were obtained by the applicant
personally. *(Canonical article labels [A1]–[A4] per DISS_DIGEST §2; the old draft's «Стаття 1–4»
numbering was inconsistent and is abandoned. [AUTHOR: bind [A1]–[A4] to the final numbered
publication list.])*

- In **[A1]** (hierarchy/BLT-adaptation article), the applicant proposed and implemented the
  method of adaptive decomposition of spatio-temporal data based on entropy metrics (the V-BLT
  architecture) with multidimensional rotary positional embedding, and the procedural synthesis
  of video datasets.
- In **[A2]** (TempVerseFormer, Springer Neural Computing and Applications 2025, Scopus, sole
  author), the applicant developed and mathematically substantiated the algorithm of
  time-agnostic backpropagation based on reversible transformations, which resolved the
  computational uncertainty of long-horizon training. **[AUTHOR: confirm "Q1" quartile claim or
  soften to "Scopus-indexed".]**
- In **[A3]** (consistency-criteria article), the applicant improved the criteria apparatus for
  the evaluation of complex generative systems by formalizing the notion of «conceptual
  consistency».
- In **[A4]** (the fourth article, in finalization) and within the dissertation research, the
  applicant developed the hierarchical latent video generation model with long-range coarse-level
  memory; constructed the memory-horizon looping testbed and the re-entry-stratified evaluation
  protocol; conducted the matched-capacity, matched-compute study of prediction mechanisms,
  including the two-sided capability study of energy-based transformers (a characterized negative
  result); developed the theory of the exactly invertible stochastic recurrence with the
  three-axis reversibility taxonomy; and performed all complex experimental studies. *(Per
  DISS_DIGEST item 14 — replaces the old «формалізовано ієрархічну енергетичну модель HVEBT»
  wording.)*

## Апробація результатів дисертації

The main provisions, results, and conclusions of the dissertation were reported and discussed at
6 scientific conferences: **[AUTHOR: verify this list — it was inherited from the old draft; the
DSMSI-2023 participation is flagged in DISS_DIGEST Q3 as unverified, and all six [X] thesis
citations must resolve to real theses]**

- L науково-технічна конференція підрозділів ВНТУ, Вінниця, 2021;
- LI науково-технічна конференція підрозділів ВНТУ, Вінниця, 2022;
- «Проблеми вищої математичної освіти: виклики сучасності», ВНТУ, Вінниця, 2022;
- LII науково-технічна конференція підрозділів ВНТУ, Вінниця, 2023;
- LIII науково-технічна конференція підрозділів ВНТУ, Вінниця, 2024;
- Dynamical System Modeling and Stability Investigation (DSMSI-2023), Київ, 2023.

## Публікації

Based on the dissertation topic, 10 scientific works were published, including: 3 articles in
professional journals of Ukraine included in the МОН list for specialty 124 ([A1], [A3], [A4]);
1 article in an international scientific journal indexed by the Scopus scientometric database
([A2]); and 6 abstracts of reports at international and all-Ukrainian scientific-practical
conferences. *(Per decision Q3 the list is unchanged for now.)* **[AUTHOR: replace the donor
(Isaienkov) bibliography wholesale; resolve all [X] placeholders; confirm that [A4] will be
published in a UA professional journal by defense so the 3+1+6 count holds.]**

## Структура та обсяг дисертації

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
- **Chapter 3** develops the hierarchical mathematical model of video sequence generation with
  long-range coarse-level memory in compact hierarchical latent state spaces: the two-level
  latent state space, the iterative-refinement generation core, cross-level attention with
  unidirectional gradient-detached control, block-causal spatio-temporal routing, the
  autoregressive rollout algorithm with its memory and complexity estimates, and the formal
  definition of the memory-horizon testbed with the re-entry-stratified evaluation protocol.
- **Chapter 4** presents the experimental research and practical application: the information
  base; the matched-compute mechanism study (including the two-sided characterized study of
  energy-based descent); the single-step and 152-frame autoregressive-rollout (after a 48-frame
  ground-truth warm-up) evaluations of the hierarchical
  model; the memory-consumption measurements; and the implementations.

The total volume of the work is **[AUTHOR: pages]** pages of typewritten text; the main part
comprises **[AUTHOR: pages]** pages, **[AUTHOR: N]** figures, and **[AUTHOR: N]** tables. The list
of references contains **[AUTHOR: N]** items. The appendices contain copies of the implementation
acts and fragments of the software code of the developed information technology. *(The old
draft's «185 сторінок / 145 джерел / 45 рисунків / 12 таблиць» were placeholders — recount after
assembly; do not carry them over.)*
