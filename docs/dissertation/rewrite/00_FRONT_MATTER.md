# 00 — FRONT MATTER (rewrite, phase 1)

> **Status:** English working text for the Ukrainian dissertation front matter. The Ukrainian pass
> translates this mechanically; genre-specific phrasing spots are marked **[UA-PASS: …]** and
> author-input spots are marked **[AUTHOR: …]**.
> **Sources of truth:** `docs/dissertation/DISS_DIGEST.md` (§§3, 5, 6, 8 — author decisions Q1–Q7
> override everything), `docs/draft/ARTICLE_SKELETON.md`, `docs/draft/CH2_TESTBED.md`,
> `docs/draft/CH3_MECHANISM.md`. Every number below is traceable to those files; none are invented.
> **Convention:** one canonical abstract text is written once (§2). The UA АНОТАЦІЯ is its
> translation; the EN ABSTRACT is this text as-is. This prevents UA/EN drift.

---

## 1. Title page (replacement fields)

- **Topic (UA, approved, verbatim — fixed):** «Системний аналіз та інтелектуальні технології
  консистентного моделювання процесів ієрархічної генерації відеоданих»
- **Topic (EN):** *Systems Analysis and Intelligent Technologies for Consistent Modeling of
  Hierarchical Video Data Generation Processes*
- Candidate: Kulyk Leonid Ruslanovych. Specialty 124 «System Analysis», field 12 «Information
  Technologies». Supervisor: Prof. O. B. Mokin, Dr. Sc. (Eng.). VNTU, Vinnytsia — 2026.
- UDC: 004.054:[004.032.26+004.85] **[AUTHOR: confirm the UDC still fits the new topic (it was
  chosen for the old energy-based framing); also fix the stray opening «-quote typo before
  "праця на правах рукопису" on the current title page].**
- All occurrences of the old topic wording («…інформаційні технології… ітеративної генерації…»)
  are replaced by the approved wording above, including the degree line inside the abstract.

---

## 2. АНОТАЦІЯ (UA) / ABSTRACT (EN) — canonical replacement text

> [UA-PASS: the UA version opens with the standard formula «Кулик Л. Р. …
> — Кваліфікаційна наукова праця на правах рукопису. Дисертація на здобуття наукового ступеня
> доктора філософії за спеціальністю 124 «Системний аналіз». — Вінницький національний технічний
> університет, Вінниця, 2026.» The EN version mirrors it as in the current draft.]

The purpose of the dissertation research is to improve the efficiency and the temporal and spatial
consistency of hierarchical video data generation processes through the development of systems
analysis methods, mathematical models, and intelligent information technologies. The dissertation
addresses the actual scientific and technical problem of systems analysis, mathematical modeling,
and uncertainty resolution in complex multi-step systems for the hierarchical generation of
spatio-temporal data. [UA-PASS: «розкриття невизначеності» is the spec-124 idiom for "uncertainty
resolution".]

The introduction presents the relevance and justification of the research topic in the context of
systems analysis requirements; describes the purpose, object, subject, and tasks of the research;
states the scientific novelty; shows the connection of the dissertation with research programs;
states the practical significance of the results, their approbation and publications; and
describes the personal contribution of the applicant.

### Chapter-by-chapter summary (new structure per DISS_DIGEST §6)

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
0.26% measured, with an analytic band of 0.6–1.0%). [UA-PASS: «часонезалежне зворотне розповсюдження помилки», «оборотні
перетворення», «збереження шуму».]

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
memory and complexity estimates of the rollout algorithm. [UA-PASS: «грубий (верхній) рівень із
довготривалою пам'яттю», «циклічний тестовий стенд», «стратифікація за складністю повторного
входження».]

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
reach structured outputs at the scale reachable in this work. For the developed hierarchical
model, single-step evaluation shows that spatially compressed long history reduces the error on hard
re-entry frames by approximately 13% (two seeds, confidence intervals excluding zero), a
capability that emerges only under a larger optimization budget (null at 6,000 steps, win at
20,000; budget-matched schedules); in 152-frame autoregressive rollouts (after a 48-frame
ground-truth warm-up) the hierarchical model beats a flat model matched in context-token budget
precisely beyond the horizon of the flat context window (null at every horizon-hardness bin below
horizon 25, a win with confidence interval excluding zero at every bin from horizon 25 onward),
and the overall window ordering confirms that longer memory windows strictly improve rollout error
(foreground error 0.1848 versus 0.2038 versus 0.2260 for the 48-frame coarse-memory, token-matched
flat, and short-window arms respectively). The constant-memory rollout capability is demonstrated
by measurement: classical backpropagation through time exhausts GPU memory at rollout depth
$K=120$, while the reversible implementation remains at approximately 1.1–1.2 GB through $K=160$.
The practical application of the developed methods in the tasks of ORNETIKA LLC and in the
educational process of VNTU is described (per the existing implementation acts, which cover the
reversible-transformer method and entropy tokenization). [A2]'s published 2-D rotating-shape
validation is reported as the receipt of the reversibility method; [A1] supplies the 2-D/3-D
entropy-tokenization examples. Moving MNIST and TaxiBJ are separate dissertation bridges, while
MMNIST-CIFAR and KTH remain explicitly unexecuted extensions.

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

### Main scientific and practical results (7 items, mapped 1:1 to the 7 tasks)

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
database; 6 conference abstracts. **[AUTHOR: per decision Q3 the publication list is unchanged for
now; the donor (Isaienkov) list in the current draft must still be replaced wholesale with the
real list — see §3 below.]**

### Scientific novelty (the 4 rewritten items — canonical wording)

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
   temporal consistency to registered reach-differential populations. [UA-PASS:
   «Вперше розроблено методологію … яка, на відміну від існуючих …, забезпечує …» genre formula.
   Re-scoped 2026-07-07: the «вперше» is carried by the testbed methodology + matched-budget
   demonstration (genuinely unclaimed), NOT by the coarse-memory architecture itself, which
   §1.3.2 cites as adopted prior art (CW-VAE/MilliVid-class) — this removes the audit-flagged
   self-contradiction.]

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
   the adequacy of complex generative systems. [UA-PASS: «удосконалено».]

4. **Improved** the method for assessing the state of uncertainty of spatio-temporal data through
   their adaptive decomposition using N-dimensional rotary positional embedding and per-pixel
   entropy maps, which allows dynamic allocation of computational resources during the analysis of
   complex visual scenes.

### Usefulness of the obtained results (implementation acts — kept as-is, they match reality)

- practical activities of **ORNETIKA LLC** (Kyiv): the developed information technology was used
  in designing intelligent systems for the analysis of spatio-temporal video data and anomaly
  detection in complex technical environments; the resource-efficient methods (TempVerseFormer)
  and adaptive entropy tokenization substantially reduced the computational cost of modeling long
  temporal sequences (implementation act of October 14, 2025);
- the **educational process of VNTU** in the disciplines «Інформаційні технології моніторингу та
  аналізу стану складних систем» and «Інтернет речей та інтелектуальний аналіз даних» for PhD
  students of specialty 124 (implementation act of September 19, 2025).

*(Note: per DISS_DIGEST item 38, the acts mention only TempVerseFormer and entropy tokenization —
no claims about the hierarchical model are added here.)*

### Practical significance

The practical significance lies in the development of a complex of algorithms and software tools
implementing the developed intelligent information technology for the systems analysis of
hierarchical video generation processes (Python/PyTorch; repository modules for the hierarchical
model, the reversible recurrence, and the testbed generator). Unique synthetic spatio-temporal
datasets were created for testing generation and optimization systems: 2D shapes, 3D polycubes,
and the looping memory-horizon testbed with recorded action tracks and analytic hardness labels
(3,000 training clips × 100 frames; 500 test clips × 200 frames). The proposed technology can be
applied to state estimation and prediction of future states of complex systems in technical,
medical, and environmental monitoring. **[AUTHOR: per decision Q5 no copyright certificate has
been filed — either delete the certificate sentence from the current draft or file a new
registration under a title without "HVEBT" (e.g., «Комплекс програм для ресурсоефективної
ієрархічної генерації відеоданих») and insert its number.]**

### Keywords (replacement list)

systems analysis; complex system; mathematical modeling; intelligent information technology;
decision making; neural network; deep learning; artificial intelligence; video data generation;
**hierarchical video generation**; **compact hierarchical latent state spaces**; **memory
horizon**; **temporal consistency**; **context packaging**; **reversible transformations**;
**diffusion refinement core**; autoregressive prediction; state estimation; forecasting;
consistency.

*(Dropped per instructions: «енергетичні моделі», «марковські ланцюги», and all VQ/discrete-space
terms; added the six new-topic terms in bold.)* [UA-PASS: ієрархічна генерація відеоданих;
компактні ієрархічні приховані простори станів; горизонт пам'яті; часова консистентність;
довготривала пам'ять грубого рівня; оборотні перетворення; дифузійне ядро ітеративного
уточнення.]

---

## 3. СПИСОК ПУБЛІКАЦІЙ ЗДОБУВАЧА — replacement note

The entire publication list in the current draft (items [1]–[10]) belongs to Ya. O. Isaienkov (the
donor template) and must be replaced wholesale with the applicant's real list. Per decision Q3 the
real list itself is unchanged for now; canonical article labels used throughout the rewrite:

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
  the matched-capacity, matched-NFE mechanism study including the EBT characterized negative, and the
  reversibility theory (non-core supporting material per the 2026-07-10 triage). **[AUTHOR: target venue and status by
  defense.]**
- 6 conference theses. **[AUTHOR: identify all six; DSMSI-2023 participation needs verification
  per DISS_DIGEST Q3.]**

---

## 4. ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ ТА СКОРОЧЕНЬ — replacement list

The current list is a GAN-era donor template; replace with the abbreviations actually used:

| Abbreviation | Expansion |
|---|---|
| ВНТУ | Вінницький національний технічний університет |
| AE | Autoencoder (автокодувальник) |
| AR | Autoregressive (авторегресійний) |
| BPTT | Backpropagation Through Time |
| CI | Confidence Interval (довірчий інтервал) |
| CRN | Common Random Numbers |
| DDIM | Denoising Diffusion Implicit Models |
| DDPM | Denoising Diffusion Probabilistic Models |
| DiT | Diffusion Transformer |
| EBM | Energy-Based Model |
| EBT | Energy-Based Transformer |
| EMA | Exponential Moving Average |
| fg-MSE | Foreground Mean Squared Error (середньоквадратична помилка переднього плану) |
| GPU | Graphics Processing Unit |
| LPIPS | Learned Perceptual Image Patch Similarity |
| MCMC | Markov Chain Monte Carlo (у розділі механізмного дослідження) |
| MSE | Mean Squared Error |
| NFE | Number of Function Evaluations |
| OOM | Out of Memory |
| RoPE-ND | N-dimensional Rotary Positional Embedding |
| SSIM | Structural Similarity Index Measure |
| V-BLT | Video Byte-Latent Transformer |
| VAE | Variational Autoencoder |

**[AUTHOR: extend after the chapter rewrites settle; the УМОВНІ ПОЗНАЧЕННЯ list must contain only
symbols actually used in the final text.]**

---

## 5. ЗМІСТ (table of contents skeleton — new chapter titles per DISS_DIGEST §6)

- ВСТУП
- РОЗДІЛ 1. Системний аналіз процесів ієрархічної генерації відеоданих та критеріїв їх
  консистентності
- РОЗДІЛ 2. Математичні методи оцінювання невизначеності та ресурсоефективного моделювання
  багатокрокових відеопроцесів
- РОЗДІЛ 3. Ієрархічна математична модель генерації відеопослідовностей із довготривалою
  пам'яттю грубого рівня
- РОЗДІЛ 4. Експериментальне дослідження та практичне застосування
- ВИСНОВКИ · СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ · ДОДАТКИ (публікації; акти впровадження; конфігурації
  експериментів; міст до OpenSTL/MMNIST)

*(Section-level TOC is regenerated after the chapter rewrites; appendices А/Б of the donor draft —
dental X-ray content — are removed entirely.)*
