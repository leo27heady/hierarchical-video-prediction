# Introduction

## Relevance and Justification of the Research Topic

Video generation and prediction are multi-step processes in which spatial appearance, object
identity, motion, scene geometry, and control signals must remain mutually consistent over time.
Recurrent prediction, stochastic latent modelling, transformer-based generation, and diffusion
have progressively improved short-horizon visual quality
[1] [2] [3] [4].
Long-horizon use, however,
introduces a systems problem: the generator must preserve information that may disappear from the
current viewport, decide which detail should remain accessible, and propagate learning signals
through an increasingly long computation.

From the viewpoint of systems analysis, a generative video model is a complex dynamic system with
partially observed state, interacting representation levels, stochastic transitions, bounded
resources, and multiple performance criteria. The relevant uncertainties are not only
probabilistic. They include informational uncertainty about hidden content, structural uncertainty
about the appropriate decomposition of state, algorithmic uncertainty about the generation
mechanism, computational uncertainty about feasible horizons, and target uncertainty about which
notion of consistency the task requires
[5, 6, 7]
[8, 9, 10, 11].

A common engineering response is to enlarge a flat spatio-temporal context. Standard dense
self-attention has quadratic arithmetic in sequence length, while stored attention state and
key–value caches grow with the number of retained tokens; memory-efficient implementations reduce
data movement but do not make the choice of retained history irrelevant
[12, 13]
[14, 15, 16, 17, 18, 19]. Long-context video models
therefore face an allocation problem: a fixed state budget can preserve many low-detail or sparse
historical observations, or fewer dense high-detail observations
[20] [21]
[22, 23, 24].

Long self-fed rollout creates a second difficulty. A model trained predominantly on
ground-truth-conditioned histories is evaluated on its own imperfect outputs; errors alter later
inputs and can accumulate over time [25, 26]
[27]. Extending the training horizon by ordinary backpropagation through time also stores
intermediate activations whose memory grows with the differentiated process length
[28] [29]. Gradient checkpointing trades recomputation for reduced
activation storage, while reversible networks reconstruct intermediate states instead
[30, 31]
[32].

These pressures are often described collectively as a “memory problem,” but they act on different
terms. The dissertation uses the conceptual decomposition

$$
M_{\mathrm{system}} =
M_{\mathrm{parameters}} +
M_{\mathrm{optimizer}} +
M_{\mathrm{context/state}} +
M_{\mathrm{activations}} +
M_{\mathrm{transient}}.
$$

Hierarchical or multirate context packaging acts primarily on
\(M_{\mathrm{context/state}}\): it changes how much historical state can be addressed under a
fixed context budget. Reversible temporal computation acts primarily on
\(M_{\mathrm{activations}}\): it reconstructs intermediate states during reverse replay instead
of retaining the full activation tape. These mechanisms are complementary but separately
measured. The dissertation does not assume that their savings multiply, and the final video
predictor is not presented as an end-to-end combination of every dissertation strand. The shared
systems principle is:

> **Compress what must be remembered; reconstruct what need not be stored.**

The state-memory question also contains an information trade-off. A coarse, subsampled, or
hierarchical package may extend addressable temporal reach but discard fine detail. A dense flat
package may preserve identity and texture but cover a shorter history. Existing hierarchical and
memory-augmented video models demonstrate that long history can be represented in several ways
[33] [34]
[20, 35, 36] [37], but cross-paper
score comparisons do not isolate whether a difference is caused by reach, fidelity, target
interface, sampler allocation, or evaluation population.

This dissertation therefore treats context packaging as a controlled design space and introduces
a memory-horizon measurement instrument in which geometric observability at viewport re-entry is
known exactly. The resulting empirical rule is deliberately conditional:

> **Hierarchy buys reach; flat context buys fidelity.**

The hierarchy is confirmed at tested budgets where a flat full-resolution window cannot address
the relevant past. Flat context recovers at larger tested budgets and better preserves digit
identity in a persistent-object boundary study. This is not a universal architecture ranking; it
locates operating regions and failure modes.

The relevance of the research follows from the need for methods that jointly:

1. decompose a complex video-generation process into measurable state, resource, and consistency
   components;
2. retain memory-relevant history under bounded context/state resources;
3. reduce the activation cost of long temporal optimization;
4. distinguish storage, addressing, fidelity, and self-fed reuse experimentally; and
5. provide an auditable software and evidence chain for claims about complex generative systems.

## Relationship to Research Programmes, Plans and Topics

The dissertation was carried out in accordance with the research plans of Vinnytsia National
Technical University, including the following state-budget research topics:

- Research topic No. **[AUTHOR: research-project number 1]**, *Development of Information
  Technologies and Mathematical Methods for Intelligent Analysis of Spatio-Temporal Data in
  Complex Systems* (2021–2023), within which the applicant developed adaptive estimation of
  spatio-temporal uncertainty and entropy-based tokenization of video data.
- Research topic No. **[AUTHOR: research-project number 2]**, *Systems Analysis and Modelling of
  Multi-Step Processes under Resource Constraints and Uncertainty* (2024–2026), within which the
  applicant developed reversible temporal optimization and the hierarchical video-generation
  model.

**[AUTHOR: verify both approved project titles, numbers, dates, and the applicant’s role against
the official project reports.]**

## Aim and Research Tasks

**The aim of the dissertation** is to improve the resource efficiency and temporal and spatial
consistency of hierarchical video-data generation processes through systems-analysis methods,
mathematical models, and intelligent information technologies that separate context/state memory,
activation memory, and task-dependent fidelity.

The following seven tasks are solved:

1. Analyze video-data generation as a complex dynamic system; classify its informational,
   structural-algorithmic, computational, and target uncertainties; and identify the resource and
   consistency limitations of flat long-context modelling.

2. Improve the criteria apparatus for generative processes by extending the taxonomy of
   conceptual, hierarchical, and trajectory consistency and by operationalizing trajectory
   consistency through measurable self-fed rollout and re-entry-stratified endpoints.

3. Develop a method for estimating spatio-temporal uncertainty through adaptive entropy-based
   decomposition and multidimensional rotary positional encoding.

4. Develop a reversible method for optimizing long multi-step temporal processes, extend it to
   stochastic multi-level recurrences, and delimit its applicability relative to
   backpropagation through time and gradient checkpointing.

5. Develop a mathematical model of multi-stream latent video generation whose context streams
   retain history at different temporal rates and spatial resolutions over a shared
   joint-attention backbone.

6. Create a research software toolkit and intelligent information technology for systems
   analysis of hierarchical video generation, including adaptive representation, reversible
   optimization, context packaging, provenance controls, the memory-horizon testbed, and
   re-entry-stratified evaluation. The modules are integrated at the methodology and software
   environment level; no claim is made that every module forms one measured end-to-end predictor.

7. Experimentally evaluate the developed methods on controlled and open datasets; compare
   generation mechanisms and context packages under declared budgets; measure memory use,
   uncertainty, consistency, and robustness; and determine the operating regions and limitations
   of the resulting methods.

The tasks correspond directly to the principal results and General Conclusions. The fourth
article addresses the context-packaging part of Tasks 5–7; [A1], [A2], and [A3] support the
representation, reversible-optimization, and consistency-criteria strands.

## Object and Subject of Research

**The object of research** is the processes of systems analysis, mathematical modelling, and
control of complex multi-level video-data generation systems under structural, informational, and
computational uncertainty.

**The subject of research** is the mathematical models, methods, criteria, algorithms, and
intelligent information technologies for uncertainty estimation, resource-efficient temporal
optimization, hierarchical context representation, and consistency evaluation in generative
video systems.

## Research Methods

The research uses:

- systems-analysis and hierarchical-decomposition methods to define the system boundary,
  interacting state levels, uncertainty classes, resource terms, and evaluation criteria;
- information theory and entropy estimation for adaptive spatio-temporal decomposition;
- probability theory and stochastic-process models for diffusion, autoregressive rollout, and
  uncertainty analysis;
- tensor algebra, optimization, attention mechanisms, and deep learning for model construction
  and parameter identification;
- reversible computation, backpropagation through time, and checkpointing analysis for temporal
  activation-memory optimization;
- common random numbers, paired clip-level cluster bootstrap, training-seed summaries, and
  simultaneous max-statistic inference for registered comparison families;
- criteria-based decision rules, protected holdouts, content-addressed run specifications, and
  immutable artifacts for outcome-independent experimental control.

The statistical sampling unit is the clip. Frames remain inside a resampled clip because rollout
errors are temporally dependent. Clip-cluster intervals condition on a trained model; training
seeds form a separate uncertainty level. Null intervals are not interpreted as equivalence without
a justified smallest effect of interest.

## Scientific Novelty

The scientific novelty is scoped to the developed methods and controlled demonstrations rather
than to the invention of hierarchical video models as a class.

1. **For the first time, a controlled methodology for systems analysis of memory-dependent
   re-entry in budgeted video prediction is developed.** It combines a purpose-built
   memory-horizon testbed with exact geometric observability, pair-specific reach-differential
   populations, complete false-positive-sensitive endpoints, matched context-token comparisons,
   protected holdouts, and simultaneous clip-cluster inference. Unlike aggregate evaluation, the
   methodology isolates records for which the longer package can address relevant history that
   the comparator cannot.

2. **A fixed-budget operating-region account of hierarchical video memory is established for the
   tested system.** Hierarchical packages are confirmed at 896 and 1,152 context tokens, the
   1,792-token point is unresolved, and flat full-resolution packages are confirmed at 2,432 and
   3,072. At equal nominal 48-frame reach, the confirmed hierarchical packages use 2.67–3.43
   times fewer context tokens. The contribution is the controlled reach–fidelity measurement and
   tested bracket, not a universal hierarchy law or a unique continuous crossing.

3. **For the first time in the dissertation’s temporal-optimization line, an exactly reversible
   stochastic recurrence with stored noise and a three-axis applicability taxonomy is developed
   and experimentally verified.** It extends time-agnostic reverse replay to the tested
   stochastic multi-level process while distinguishing reversibility across depth, time, and
   computation and identifying checkpointing as the relevant practical comparator.

4. **The criteria apparatus for generative systems is improved** by formalizing conceptual,
   hierarchical, and trajectory consistency and connecting trajectory consistency to
   self-fed-rollout, object-support, re-entry, identity, miss/extra, and coverage endpoints.

5. **The entropy-based adaptive decomposition method is improved** through N-dimensional rotary
   positional encoding and spatio-temporal entropy maps, enabling resource allocation to
   high-information regions while preserving multi-axis positional structure.

## Practical Significance

The practical result is a research software toolkit and evidence-controlled workflow that
implements:

- entropy-guided adaptive decomposition and RoPE-ND experiments;
- reversible temporal blocks, reverse replay, and memory measurement;
- flat, hierarchical, and temporally sparse context packages;
- a shared latent diffusion-transformer predictor and regression/energy controls;
- the LoopMom memory-horizon generator, persistent-object Moving-MNIST generator, and complete
  frame/object metrics;
- fail-closed experiment authorization, content-addressed run specifications, dataset-purpose
  logging, immutable evidence artifacts, and generated tables.

The toolkit supports research and education in the modelling of long spatio-temporal processes.
Its present evidence establishes controlled synthetic and benchmark boundaries; it does not by
itself establish deployment effectiveness in medical, environmental, or safety-critical
monitoring.

The reported implementation acts concern the reversible-transformer and entropy-tokenization
strands:

- ORNETIKA LLC, Kyiv, implementation act dated 14 October 2025;
- the VNTU educational process for the disciplines *Information Technologies for Monitoring and
  Analysing the State of Complex Systems* and *Internet of Things and Intelligent Data Analysis*,
  implementation act dated 19 September 2025.

The final submission should reproduce the wording of those acts exactly in the appendices.

## Reliability and Validation of the Results

Reliability is supported by analytical derivations, controlled synthetic instruments, protected
and purpose-locked holdouts, multiple training seeds where registered, common-random-number paired
evaluation, clip-cluster resampling, simultaneous inference, independent codec and sampler
interventions, negative-replication reporting, hash-pinned datasets and artifacts, and executable
CPU validation.

The central hierarchy result is limited to the tested representation, predictor family, budgets,
and datasets. LoopMom is a mechanistic synthetic instrument rather than natural-video evidence.
Persistent-object Moving-MNIST is a valid external synthetic boundary, not a replication of
LoopMom’s exact geometry. The reversible-memory result is independently measured and is not
claimed as the cause of the context-package effects.

## Personal Contribution of the Applicant

The applicant’s contribution is separated by publication strand:

- In **[A1]**, the applicant developed and implemented entropy-based adaptive decomposition of
  spatio-temporal data, RoPE-ND, and the corresponding video-data experiments.
- In **[A2]**, a sole-author article, the applicant developed and mathematically substantiated
  TempVerseFormer and time-agnostic backpropagation based on reversible transformations.
- In **[A3]**, the applicant developed the consistency-criteria treatment used in the
  dissertation’s systems-analysis framework.
- In **[A4]** and the associated dissertation programme, the applicant developed the controlled
  context-packaging study, hierarchical latent predictor, LoopMom instrument, pair-specific
  reach populations, persistent-object evaluation, experiment-control system, and the complete
  reach–fidelity evidence chain.
- The exactly reversible stochastic recurrence and applicability taxonomy are dissertation
  developments connected to, but separately scoped from, the published TempVerseFormer result.

For co-authored works, the final institutional version must reproduce the formally approved
contribution statements.

## Approbation and Publications

The main provisions and results were reported at six scientific conferences:

- 50th Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2021;
- 51st Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2022;
- *Problems of Higher Mathematical Education: Contemporary Challenges*, VNTU, Vinnytsia, 2022;
- 52nd Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2023;
- 53rd Scientific and Technical Conference of VNTU Departments, Vinnytsia, 2024;
- Dynamical System Modeling and Stability Investigation (DSMSI-2023), Kyiv, 2023.

**[AUTHOR: verify every event, date, contribution, and corresponding publication against the
applicant’s records.]**

The dissertation currently records four article strands—[A1], [A2], [A3], and the fourth article
[A4]—plus six conference theses. **[AUTHOR: supply the target venue/status of [A4], the six
conference entries, and the institutionally approved publication count and categories.]**

## Dissertation Structure and Volume

The dissertation consists of an Introduction, four chapters, chapter conclusions, General
Conclusions, references, and appendices.

- **Chapter 1** analyses video generation as a complex hierarchical system, classifies its
  uncertainties and consistency criteria, reviews flat, hierarchical, memory-augmented, and
  energy-based approaches, and states the research tasks.
- **Chapter 2** develops entropy-based adaptive decomposition, RoPE-ND, reversible temporal
  transformations, time-agnostic backpropagation, the stochastic reversible recurrence, and the
  applicability boundary relative to checkpointing.
- **Chapter 3** develops the compact two-level latent state, joint-attention multi-stream
  predictor, v-parameterized window diffusion, autoregressive rollout, token-cost model, LoopMom
  instrument, complete endpoints, and clip-cluster inference.
- **Chapter 4** evaluates mechanism selection, untouched reach confirmation, the tested
  reach–fidelity frontier, package shape, robustness, negative replication, explicit access, the
  persistent-object boundary, practical implementation, and limitations.

The final page, figure, table, reference, and appendix counts must be inserted after institutional
typesetting. **[AUTHOR: supply the final pagination and confirm the appendix set.]**
