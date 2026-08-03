# Chapter 1 — Systems Analysis of Hierarchical Video Generation and Its Consistency Criteria

## 1.1 Generative video models as complex control systems

The development of artificial-intelligence and machine-learning methods over the last decade has
produced new classes of mathematical models capable of solving complex problems of data analysis,
prediction, and synthesis. A special place among them is occupied by generative models, which have
transformed from simple statistical tools into extremely complex multi-level information systems.
Large generative models — including systems operating on spatio-temporal video data — can contain
billions of parameters and require substantial computational resources [REF: survey of
large-scale generative video models]. Despite impressive visual results, their long-horizon
state, resource, and failure behaviour is often difficult to characterize from aggregate output
quality alone. Their learned components can be difficult to interpret, which complicates
analysis of stability, predictability, and resource efficiency [REF: black-box critique /
interpretability of deep generative models].

To address these problems it is necessary to change the paradigm of perceiving generative
algorithms and to consider them as open complex dynamic systems. In this formulation the process
of video data generation is not merely a mathematical transformation of information but a
purposeful process of changing the state of a complex system under control actions and the
influence of the external environment. Iterative generative models — diffusion models [REF: DDPM,
Ho et al.], autoregressive systems, and energy-based models employing Markov chain Monte Carlo —
can usefully be analysed as complex dynamic systems with state, inputs, outputs, feedback, and
resource constraints.

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

In accordance with classical systems theory [REF: classical systems theory textbook], the object
under study is characterized by a large number of interrelated elements, nonlinear interactions,
emergence, and a hierarchical structure. In the context of computer vision the elements of the
system are individual pixels or latent features (tokens). Their interaction is provided by
attention mechanisms and convolutional operations, which model spatio-temporal correlations.
Nonlinearity is achieved through nonlinear activation functions in the artificial neural networks
that form the core of the system.

An important characteristic of such a system is the iterative nature of its state transitions.
Unlike common GAN [REF: GAN, Goodfellow et al.] or VAE
[REF: VAE, Kingma & Welling] decoders, which synthesize an output through one feed-forward
inference path, iterative models such as denoising diffusion probabilistic models (DDPM)
[REF: DDPM] explicitly decompose sampling into many state transitions.
This approach fundamentally changes the mathematical nature of the system: the state evolves in
time (or in the pseudo-time of optimization steps), and every next state depends on the previous
one, the current parameters, and the control action.

Mathematically, the evolution of the state of such a system can be written as a controlled Markov
transition:

$$z_{s+1} = f_\theta(z_s, a_s, c, \xi_s) \tag{1.1}$$

where $z_s$ is the system state (a multidimensional tensor of video data or its latent
representation) at a generic transition index $s$; $a_s$ is an admissible decision or scheduled
control action; $c$ is fixed exogenous conditioning; $\theta$ denotes the learned parameters; $\xi_s$ is a
stochastic input; and $f_\theta$ is the nonlinear state-transition map. The generic index $s$ is
used here deliberately. In the temporal decomposition below, $t$ is reserved for physical video
time and $k$ for refinement or sampling pseudo-time.

When the stochastic component $\xi_s$ is present, the transition samples from a conditional
distribution; deterministic samplers correspond to a fixed or zero stochastic term. Systems
methodology under risk and uncertainty [REF: systems-under-risk methodology; REF: system
methodology monograph] motivates tracking how initial state, conditioning, sampling, and feedback
affect later outputs. In generative video, failures can appear as visual artifacts, temporal
discontinuities, or violations of the intended scene dynamics.

Complex systems can be decomposed into interacting levels of detail or abstraction
[REF: hierarchical systems theory]. For video generation, a useful analytical decomposition
distinguishes a **macro level** of scene composition and global relations, a **meso level** of
geometry and object structure, and a **micro level** of high-frequency appearance, lighting, and
pixels. This is a modelling choice used to expose cross-level consistency requirements; it is not
a claim that every implemented model contains three explicit modules.

Meaningful video depends on spatial and temporal relations across these descriptive scales. A
failure of cross-scale coherence may, for example, place plausible texture on an inconsistent
shape. Hierarchical consistency is therefore one component of stable generation, alongside
temporal, trajectory, conceptual, and object-level criteria developed later in the chapter.

### 1.1.1 Generation as an optimal-control problem

Increasing model capacity and training data can improve generation, but it also raises training
and inference cost and does not by itself identify which state or trajectory constraint is
violated. This motivates a complementary control formulation of the generation trajectory.

Consider the generation path through the analytical lens of an optimal-control problem. At the
initial control step $s=0$, the state $z_0$ may be sampled from a noise distribution. At the final
step $s=S$, the induced state $z_S$ should satisfy the target and conditioning requirements. This
is a systems-level abstraction: a trained sampler usually executes a learned transition under a
fixed schedule rather than solving a new optimal-control problem online for every sample.

The formalization of this process requires a systemic criteria apparatus. Let
$\mathbf z=(z_0,\ldots,z_S)$ be an admissible state trajectory and
$\mathbf a=(a_0,\ldots,a_{S-1})$ its control sequence under fixed condition $c$. A
**consistency criterion functional** can combine terminal and path costs. The corresponding
constrained problem optimizes the controls and their induced trajectory, not an unconstrained
terminal tensor:

$$
(\mathbf z^{*},\mathbf a^{*}) \in
\arg\min_{\mathbf z,\mathbf a}
J(\mathbf z,\mathbf a;c)
=\arg\min_{\mathbf z,\mathbf a}
\left[\Phi(z_S;c)+\sum_{s=0}^{S-1}\ell_s(z_s,a_s;c)\right]
\quad\text{subject to}\quad
z_{s+1}=f_\theta(z_s,a_s,c,\xi_s),\;
(z_s,a_s)\in\mathcal Z\times\mathcal A.
\tag{1.2}
$$

Equation (1.2) is conditional on a fixed realization of $\boldsymbol\xi$; a stochastic-control
formulation instead minimizes the corresponding expected cost (or optimizes a feedback policy).
Smaller values indicate lower measured inconsistency under the chosen functional; they do not by
themselves establish perceptual realism or make every evaluation endpoint a training objective.

![Multi-step generation on a consistency landscape](../../figures/fig31_consistency_landscape.png)

**Figure 1.2 — Conceptual multi-step generation on the landscape of a consistency criterion.**
The trajectory moves from a high-criterion noise state toward a structured
low-criterion region. This is a systems-level abstraction; only energy-based models instantiate
the landscape as an explicit learned scalar energy. Source: author’s schematic.

In the context of the intelligent information technology being developed, it is important to
clearly delimit the roles of two fundamental system concepts: the **global consistency criterion**
and **entropy**. The global criterion (1.2) is trajectory-level: it is a functional used to assess
or design the entire multi-step generation trajectory; in this dissertation it is operationalized
not by a learned scalar energy but by the measurable consistency criteria of §1.4—in particular the
re-entry-stratified rollout metrics of §1.4.5, which evaluate the hierarchical model of Chapter 3
without being its direct training loss. Entropy, by contrast, acts as a **local diagnostic indicator**: the predictive
entropy of a model-provided categorical distribution measures how dispersed that distribution is
at specific spatio-temporal coordinates. In the reported V-BLT setting, entropy maps highlight
locations that the model predicts less decisively and can therefore guide
non-uniform allocation of limited computational resources (Chapter 2). Predictive entropy is not
automatically a calibrated estimate of data uncertainty and can mix ambiguity, model error, and
miscalibration [REF: uncertainty decomposition; REF: calibration of modern neural networks]. It
is therefore used here as an allocation proxy, not as evidence that the system "understands" the
scene.

Controlling this system runs into **computational uncertainty** on two fronts. First, at the
stage of parameter identification (training): a vanilla implementation of backpropagation through
a long neural computation retains the intermediate activations needed by the reverse pass, so its
activation tape grows with trajectory length. Checkpointing and reversible replay change this
storage–recomputation schedule; Chapter 2 measures their operating regions rather than treating
one method as universally required.
Second, at the stage of operation (inference): extending the temporal context of a flat attention
mechanism inflates its key–value memory linearly and its computation quadratically with context
length (analyzed quantitatively in §1.3.1). Chapter 2 studies reversible replay as one
storage–recomputation policy for the first front, alongside checkpointing. Chapter 3 studies
hierarchical and sparse context packages as alternatives for the second front. The interventions
act on different memory terms and are measured separately.

### 1.1.2 Applied domains

Beyond its theoretical significance, understanding generative video models as complex control
systems has enormous practical importance for various applied fields. The tasks solved by such
systems intertwine closely with the objects of study of the specialty "System Analysis".

The first important application domain is medical imaging and the development of intelligent
systems for supporting clinical decision-making. Complex multidimensional spatial data (MRI, CT)
can require reconstruction, super-resolution, or cross-modal generation. Distribution-matching
reconstruction can add or suppress diagnostically relevant structures, so generated outputs need
task-specific validation, uncertainty reporting, and clinical oversight rather than an assumed
guarantee of structural faithfulness [REF: medical image reconstruction hallucination risk].

The second domain is ecological and technical monitoring using remote sensing data. Satellite
data can contain gaps due to cloud cover or technical failures, motivating spatial or temporal
imputation and forecasting. In applications such as fire-spread analysis, physical and terrain
constraints are important evaluation or conditioning information; their satisfaction must be
measured for the particular task and is not implied by visual plausibility alone [REF:
physics-informed and uncertainty-aware remote-sensing imputation].

The third domain is the creation of educational and training simulators (for example, for
autonomous vehicle control systems). Training robots requires millions of hours of synthetic
video data with diverse scenarios (including extreme ones). The requirement imposed on the
generative system is long-term temporal coherence: objects cannot suddenly disappear, deform over
time, or violate the logic of physical interactions. In particular, an object that leaves the
camera's field of view should reappear in a consistent state and position when the camera returns
to it. In a controlled setting where the missing state cannot be inferred from recent frames
alone, this requires retained or retrievable evidence spanning the relevant absence (§1.3.1 and
§1.4.5).

The analysis of the listed practical domains shows that modern generative systems are subject to
the following strict systemic requirements:

1. High visual quality and detail of the target data distribution;
2. Controllability of the system and the ability to respond to control actions;
3. Representations that preserve the cross-scale information needed by the task, whether learned
   implicitly or encoded through an explicit hierarchy;
4. Measured control of spatial, temporal, and logical anomalies — including over long horizons,
   when content leaves and re-enters the field of view;
5. Resource efficiency under constraints of computational memory;
6. Interpretability of the process.

Summarizing, empirical architecture search alone does not expose which state, resource, or
trajectory constraint causes a failure. A complementary treatment of generative algorithms as
complex dynamic systems therefore motivates a corresponding mathematical apparatus: methods of
decomposition, identification of uncertainty types, explicit formalization and operationalization
of consistency criteria, and
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
basis, three basic macro-subsystems are distinguished [REF: encoder–generator–decoder
architecture survey].

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

Many complex systems admit useful descriptions at several spatial or semantic scales
[REF: hierarchical systems theory]. For video generation, an explicit macro/meso/micro
decomposition is therefore a testable **inductive bias and resource-allocation strategy**, not a
logical requirement: a single latent grid can learn multiscale features, while an explicit
hierarchy makes the allocation between coarse reach and fine detail directly controllable.

At the **macro level** (the level of highest abstraction) the system operates with semantic
vectors of low spatial resolution but high semantic saturation. Here the questions of conceptual
composition are decided: the presence of objects, their mutual arrangement, and correspondence to
the specified condition.

At the **meso level** of abstraction (medium resolution) the system forms structural contours,
geometry, and the spatial relations of object parts.

At the **micro level** (high resolution) the questions of local pixel organization are decided:
the coherence of textures, lighting, and micro-details.

Many modern generative systems — for example, diffusion models built over variational
autoencoders [REF: LDM/VAE latent-space diffusion] — operate on one explicit latent grid. Such a
model can still learn multiscale internal features, but it offers no explicit interface for
assigning different context horizons or token rates to those scales. Joint coarse and fine
decisions can also conflict in practice; the brick-texture/wood-shape example below illustrates a
failure mode, not a causal diagnosis that follows from using a flat grid.

![Hierarchical decomposition of visual state](../../figures/fig33_visual_hierarchy.png)

**Figure 1.4 — Hierarchical decomposition of visual state.** Macro denotes the highest
abstraction and lowest spatial rate; micro denotes the pixel-proximal level and highest detail.
Source: author’s schematic.

The systems approach motivates mathematical models of explicit hierarchical decomposition with
mechanisms that **bias and measure** coherence between these levels of representation. Such a
model cannot guarantee coherence by structure alone. This decomposition is the core of the
present dissertation's topic. Moreover — a
property central to the direction developed in Chapters 2–4 — an explicitly decomposed hierarchy
allows the levels to run at *different temporal rates and context lengths*: a compact upper
(coarse) level with a low information rate can afford to retain a long temporal context at the
same memory budget for which a flat single-level representation affords only a short one, and can
therefore serve as the carrier of the system's long-range temporal memory (quantified in §1.3.1
and developed in Chapter 3).

### 1.2.3 Temporal decomposition

A third useful analytical stratification is temporal decomposition. When modeling systems that
generate video data, two distinct axes should be separated, creating a "two-dimensional temporal
grid" for the analysis that follows.

The first dimension concerns the internal **pseudo-time** of the generation process — the
partition of the continuous process of state change into discrete refinement steps
$k = 1, \dots, K$ *per generated frame*: the steps of the reverse diffusion process, the token
steps of an autoregressive decoder, or the descent steps of an energy-based sampler. Every such
step is a micro-transition of the system in phase space. The algorithm is designed to move toward
a more probable or lower-objective state, but monotonic entropy reduction or consistency
improvement is not guaranteed for every realized trajectory.

The second dimension is introduced upon the transition to processing video sequences and reflects
the **physical time axis** $t = 1, \dots, T$, where every discrete slice is a separate frame. The
complexity of system control lies in the fact that the generative model must simultaneously
ensure stability in both temporal dimensions: the system state must evolve correctly along the
optimization axis $k$ for every individual frame, and at the same time all generated frames must
be spatio-temporally coherent along the physical axis $t$, forming smooth motion kinematics.

The physical axis, in turn, gives rise to a key quantitative characteristic of this dissertation:
the **explicit context reach** — the maximal temporal distance $\Delta t$ represented directly by
an item in the configured context package. For a system whose central subsystem receives only a
flat window of $w$ preceding frames, this package-addressable reach is bounded by $w$. An event
older than the window (for example, content that left the field of view and returns later) has no
explicit source-frame token in that interface. This is not an information-theoretic
inaccessibility claim: evidence may persist indirectly in recent observations, learned weights,
or another recurrent or retrieved state if the architecture supplies one.

![Physical time and refinement pseudo-time](../../figures/fig34_temporal_grid.png)

**Figure 1.5 — Two-dimensional temporal decomposition of video generation.** Horizontal
transitions advance physical frame time, vertical transitions refine each predicted state, and
the sliding context window bounds directly addressable history. Source: author’s schematic.

### 1.2.4 Classification of uncertainty types

The performed decomposition of the complex system permits a classification of the kinds of
uncertainty arising in it. In accordance with the theory of systems analysis of complex systems
under risk [REF: systems-under-risk methodology] and system methodology [REF: system methodology
monograph], uncertainties are classified by source and nature. In the video generation system
under study, four principal kinds of uncertainty are distinguished: informational,
structural-algorithmic, computational, and target uncertainty.

**Informational uncertainty** is conditioned by the very nature of the iterative generative
process, which begins from a state of pure noise (the state of maximal entropy). Moreover,
informational uncertainty is commonly present in the control actions. For example, the textual
description "a car drives along a road" or a coarse semantic mask is usually incomplete: it
specifies only boundary macro-conditions, leaving a multitude of micro-details (the shape of the
headlights, the texture of the asphalt, the lighting) undescribed. The task of the complex system
is the correct and plausible resolution of this uncertainty during generation. If the system
completes the missing information incorrectly, "hallucinations" and anomalies arise, rendering
the result inconsistent [REF: hallucination phenomenology in generative models].

**Structural-algorithmic uncertainty** is connected with limitations and inductive biases of the
mathematical apparatus itself (the neural network). Implemented models have finite capacity. A
separate manifestation of this
uncertainty is the problem of spatio-temporal tokenization. Standard transformer architectures
(ViT, ViViT) partition the image into a regular grid of fixed-size patches, ignoring the fact
that the visual signal has non-uniform information density.

![Fixed-grid tokenization and information density](../../figures/fig35_fixed_grid_tokenization.png)

**Figure 1.6 — Fixed-grid tokenization under non-uniform information density.** Equal-size
patches assign equal token cost to static background and to high-information moving contours.
Source: author’s schematic.

Uniform allocation can be inefficient when large low-variation regions and small high-detail
moving regions have different representational requirements. This motivates—not logically
requires—adaptive tokenization based on local entropy, developed and evaluated in Chapter 2.

**Computational uncertainty** is the most critical for iterative multi-step models. It arises on
two fronts, corresponding to the two temporal axes of §1.2.3. On the training front, a vanilla
autograd realization of backpropagation through time stores intermediate activations for the
unrolled process; the resulting horizon-dependent activation tape can exceed available hardware
memory. Checkpointing and reversible replay reduce this term through recomputation rather than
changing the underlying gradient objective. Truncating the differentiated
trajectory removes gradient paths beyond the selected horizon and may therefore impair learning
of long-range effects. On the operation front, the flat attention context over the physical time
axis inflates the key–value memory linearly and the computation quadratically with the number of
context tokens (§1.3.1, eq. (1.7)), so the package-addressable reach of a dense flat interface is
capped by its token and hardware budget precisely in the regimes where longer explicit history is
required. Reducing memory consumption
for gradient computation along deep trajectories and for context retention along long horizons
is therefore an important design objective for multi-step generative systems.

**Target uncertainty** arises when the criterion optimized by the model during training does not
coincide with the global goal during generation. For example, diffusion models optimize the local
error of noise prediction at one isolated step, but during generation these steps are chained
into a long sequence in which errors accumulate. This local-versus-trajectory mismatch can leave
terminal rollout quality only indirectly optimized.

The performed multidimensional decomposition of the video generation control system and the
classification of the identified kinds of uncertainty form a clear methodological basis for the
further research. This decomposition shows why improving one architectural element or one
aggregate metric need not resolve the complete consistency problem.

The dissertation therefore investigates a combined methodological toolkit comprising: methods of adaptive
entropy-based state estimation for resolving spatial uncertainty; resource-saving algorithms
based on reversible computation for overcoming computational uncertainty; and the creation of a
new **hierarchical control apparatus over compact continuous hierarchical latent state spaces**
for measuring and reducing structural and target uncertainty.

The next subsection conducts a detailed critical analysis of existing approaches to the
mathematical modeling of visual data — diffusion, hierarchical latent and autoregressive, and
energy-based models — with the aim of identifying their strengths and their fundamental systemic
limitations in the context of the requirements formulated above.

## 1.3 Analysis of existing approaches

To solve the posed problems of systems analysis and to construct an intelligent information
technology of video data generation, a thorough critical review of the existing mathematical
approaches to modeling complex multimodal spatio-temporal distributions is necessary. The modern
scientific landscape offers several competing paradigms, each with its own architectural
features, advantages, and limitations. This subsection reviews three classes
of models **in the order dictated by the argument of this dissertation**: (i) diffusion models —
one major route to high generation quality, with costly dense long-context variants; (ii)
hierarchical latent and autoregressive approaches — a possible remedy direction, within which the
choice between discrete and continuous latent spaces is examined;
(iii) energy-based models — reviewed as literature together with the resource-efficiency methods
they motivated. The focus criteria are systemic consistency and resource control.

### 1.3.1 Diffusion models: generation quality and the flat long-context barrier

Denoising diffusion probabilistic models (DDPM) [REF: DDPM] and deterministic/non-Markovian
samplers such as DDIM [REF: DDIM] are major approaches to high-quality image and video
generation. Unlike a GAN generator [REF: GAN], which maps noise to data through one learned
forward path, diffusion models use an explicitly iterative denoising process.

The mathematical basis of the diffusion approach rests on concepts of non-equilibrium
thermodynamics. The modeling process is decomposed into forward corruption and a learned reverse
process. In equations (1.3)–(1.6), $k$ denotes diffusion pseudo-time; it must not be confused with
the physical video-frame index $t$ introduced in §1.2.3.

The forward diffusion stage is fixed and consists in the gradual addition of Gaussian noise to
the original "clean" data sample $x_0$, progressively driving it to the state of random noise
$x_K$. The process is modeled as a Markov chain:

$$q(x_k \mid x_{k-1}) = \mathcal{N}\!\left(x_k;\; \sqrt{1-\beta_k}\, x_{k-1},\; \beta_k \mathbf{I}\right) \tag{1.3}$$

where $\beta_k$ is a predetermined deterministic schedule of noise variance at step $k$. A
mathematical advantage of this construction is the closed-form availability of the noised state
at an arbitrary step directly from the original state:

$$q(x_k \mid x_0) = \mathcal{N}\!\left(x_k;\; \sqrt{\bar\alpha_k}\, x_0,\; (1-\bar\alpha_k)\mathbf{I}\right) \tag{1.4}$$

![Forward diffusion](../../figures/fig36_forward_diffusion.png)

**Figure 1.7 — Forward diffusion.** Progressive Gaussian corruption destroys the information
structure of a sample from $x_0$ toward $x_K$. Source: author’s schematic.

At the reverse diffusion stage the system learns to invert this process, denoising the corrupted
sample and restoring the data. Unlike forward diffusion, the reverse process constitutes a
complex computational problem solved by deep neural networks:

$$p_\theta(x_{k-1} \mid x_k) = \mathcal{N}\!\left(x_{k-1};\; \mu_\theta(x_k, k),\; \Sigma_\theta(x_k, k)\right) \tag{1.5}$$

In practice the model is optimized to predict the added noise $\epsilon$ via the simplified loss:

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, k}\left[\,\bigl\|\epsilon - \epsilon_\theta\bigl(\sqrt{\bar\alpha_k}\,x_0 + \sqrt{1-\bar\alpha_k}\,\epsilon,\; k\bigr)\bigr\|^2\,\right] \tag{1.6}$$

![Reverse diffusion](../../figures/fig37_reverse_diffusion.png)

**Figure 1.8 — Reverse diffusion.** A learned denoising field iteratively restores structure from
noise. The sequence is conceptual and does not depict an experimental sample. Source: author’s
schematic.

The forward–reverse construction supports high-quality synthesis by turning sampling into a
sequence of denoising decisions. Pixel-space diffusion is expensive at high spatial resolution;
latent diffusion models (LDM) [REF: LDM] reduce this cost by operating in a compressed
autoencoder space. Diffusion Transformers (DiT) [REF: DiT] provide a transformer alternative to
convolutional denoisers, and video systems such as Sora [REF: Sora technical report] operate on
spatio-temporal latent patches at substantially larger data and model scales.

Modern video diffusion and token generators extend this lineage through spatio-temporal
denoising, latent video diffusion, large-scale video pretraining, variable-length token
generation, and masked video transformers [REF: modern video diffusion and token generation].
Interactive and autonomous world models further shift evaluation from isolated clips toward
action-conditioned state prediction and control [REF: interactive and autonomous world models].
These systems establish the scale and practical importance of the problem, but they do not remove
the need to measure which historical information is retained under a bounded state interface.

![Latent diffusion architecture](../../figures/fig38_latent_diffusion.png)

**Figure 1.9 — Latent-space diffusion.** Spatial compression reduces the per-step denoising cost,
but it does not by itself solve the temporal-memory allocation problem. Source: author’s
schematic.

Despite their generation quality, dense-context diffusion video systems retain resource and
long-horizon limitations that are the direct object of this dissertation's analysis. **The paradigm-level
strengths of iterative refinement are not in question here**. Chapter 4 selects diffusion for the
stochastic generative frontier because of its single-window coverage and sharp commitment under
the controlled mechanism study. The same study also shows that regression is stronger in the
deterministic best-of-1 self-fed rollout. The mechanism decision is therefore task-dependent; the
context question studied later is *how temporal evidence is represented and scaled* within the
selected generative evaluation programme.

One limitation of a **flat, single-level latent interface** is that global composition and local
appearance share one token field. This does not make inconsistency inevitable, but it prevents
explicit budget allocation and separate control across representation levels.

The second limitation is the **resource blow-up of flat long-context attention**. To lengthen the
explicit history span of a flat video transformer without another memory mechanism, its
spatio-temporal token context must grow. For a
context of $T_w$ frames tokenized at $N_s$ spatial tokens per frame, the attention context length
is $L = T_w \cdot N_s$, and the principal costs of the attention mechanism scale as

$$C_{\text{compute}} = O(L^2 \cdot D) = O(T_w^2 N_s^2 D), \qquad
M_{\text{KV}} = O(L \cdot D) = O(T_w N_s D) \tag{1.7}$$

where $D$ is the model width. For a standard dense attention layer, attention arithmetic grows
quadratically with context length; when past keys and values are cached, cache storage grows
linearly. Kernel improvements can reduce data movement and constants but do not remove the need
to choose which historical tokens remain in the state interface.

The third limitation is the **degradation of long-horizon temporal consistency** even where a
long flat context is affordable. Empirical evaluations of long-video generators and world models
show that content which leaves the field of view is not restored consistently upon its return,
and that consistency metrics degrade with the distance to the relevant past [REF: long-term
spatial-memory benchmarks for video world models; REF: memory-augmented world-model evaluation;
REF: geometry-grounded long-video consistency probes]. In systems-analysis terms, the flat
context window imposes a hard bound on explicit package reach (§1.2.3): source-frame tokens older
than the window are absent from that context interface. This does not rule out indirect evidence
in recent frames, model weights, or a separate state mechanism. Related to this is **target uncertainty**:
during training the model optimizes the local error of one isolated step, while during generation
errors accumulate along the whole autoregressive chain — the phenomenon of exposure bias [REF:
exposure bias in sequence generation].

![Flat long-context barrier](../../figures/fig39_flat_long_context_barrier.png)

**Figure 1.10 — Flat long-context resource and re-entry barrier.** Panel A is schematic rather
than a hardware benchmark: key–value memory grows linearly and attention work quadratically with
sequence length. Panel B shows why a low-rate long-memory stream can remain addressable after the
relevant content has left a short flat window. Source: author’s schematic.

### 1.3.2 Hierarchical latent approaches and autoregressive prediction

The remedy direction for the contradictions of §1.3.1 is formed by approaches that (i) treat
video prediction autoregressively along the physical time axis and (ii) decompose the latent
state space instead of flattening it. This subsection reviews the components of that direction:
patch-based autoregressive transformers, discrete-token models, and hierarchical latent models —
and motivates, by the author's own measurements, the choice of *continuous* hierarchical latent
spaces made in this work.

**Patch-based autoregressive transformers.** This line of work originates in the transfer of
natural-language-processing principles to computer vision. The standard approach, popularized by
the Vision Transformer (ViT) [REF: ViT] and its video adaptations ViViT [REF: ViViT] and Video
Swin Transformer [REF: Video Swin], partitions the video into a grid of non-overlapping
spatio-temporal patches that are linearized and fed to a transformer. The principal advantage of
transformer models is their capacity to model long-range dependencies between distant regions of
the video; their principal cost is exactly the quadratic attention complexity (1.7). Although
Video Swin partially mitigates it with local attention windows, the fundamental limitation —
a single-level latent space with fixed patching — remains. Fixed-grid tokenization is a form of
rigid, content-blind discretization: static low-information background and dynamic scene regions
receive equal computational budgets, which §1.2.4 identified as a source of
structural-algorithmic uncertainty. An analogous problem in language modeling was resolved by the
Byte Latent Transformer (BLT) [REF: BLT], which groups bytes into latent patches of variable size
governed by information density (entropy). Transferring this paradigm to the visual modality
resolves the rigid-tokenization component of structural uncertainty; the V-BLT architecture
developed in this dissertation realizes that transfer and is presented in Chapter 2.

**Discrete-token autoregressive models.** A prominent branch of autoregressive video modeling
operates over *discrete* latent spaces: a vector-quantized autoencoder (VQ-VAE) [REF: VQ-VAE, van
den Oord et al.] compresses frames into grids of codebook indices, over which an autoregressive
transformer learns the video distribution — the VideoGPT line [REF: VideoGPT] and its successors,
including long-video architectures with compressed memory [REF: TECO / long-video discrete-token
models]. The attraction of this paradigm is genuine: discrete tokens give a clean autoregressive
likelihood, natural compatibility with language-model tooling, and strong compression.

Discrete bottlenecks also introduce specific optimization risks—straight-through assignment,
codebook underutilization, dead codes and level collapse—that are documented in the VQ
literature [REF: VQ optimization/codebook collapse]. The implemented fourth-article system
therefore uses **compact continuous hierarchical latent state spaces**, in which both levels are
end-to-end differentiable and independently decodable. This is an implementation choice, not an
empirical conclusion that continuous representations are universally superior: the present
repository contains no controlled hierarchical-VQ-versus-continuous comparison and makes no such
claim. Representation sensitivity is instead tested within the continuous-codec family by the
pre-registered alternate-codec-seed study in Chapter 4.

Hierarchical latent variables are themselves a broad prior lineage, including deep
vector-quantized and variational hierarchies [REF: hierarchical latent representation
foundations]. The dissertation therefore assigns novelty to its systems methodology and
controlled video-memory demonstration, not to the general idea of arranging latent variables at
multiple levels.

![Representation interfaces and empirical scope](../../figures/fig56_representation_scope.png)

**Figure 1.11 — Representation interfaces and the evidence boundary of this dissertation.**
Discrete VQ models are reviewed as an important autoregressive family; the controlled experiments
use a continuous two-level codec. The dashed path denotes a comparison not performed here.
Source: author’s schematic.

**Hierarchical latent video models.** The direct remedy for the flat-latent limitation is the
class of models whose latent state space is explicitly multi-level, with distinct temporal or
spatial rates per level. Early representatives use multi-rate recurrent latents — clockwork
variational hierarchies [REF: Clockwork VAE] — where upper levels tick more slowly and thereby
retain longer temporal context. The modern form of this idea in the diffusion family is the
hierarchical latent video diffusion model of the MilliVid class [REF: MilliVid-class hierarchical
latent video diffusion]: a compact hierarchy of continuous latents (a low-rate *coarse* level and
a high-rate *fine* level, each independently decodable to pixels), a diffusion core that denoises
the levels jointly with cross-level attention, and autoregressive rollout along physical time.
The systemic significance of this spatial construction follows directly from eq. (1.7): because the
coarse level carries far fewer tokens per frame than the fine level, a *heterogeneous* context —
long in coarse tokens, short in fine tokens — covers a memory horizon many times longer than a
flat context of the same total token budget. Hierarchy therefore reduces the rate at which token
count grows with temporal reach and can cover more frames at fixed $L$; it does **not** change the
$O(L^2D)$ asymptotic cost of dense self-attention in the retained token count. The coarse level is
an economical carrier of long-range temporal evidence because its per-frame token slope is
smaller. Temporal subsampling provides a distinct route to the same
span and is compared directly in Chapter 4. *(Chapter 3 develops the spatial form with a two-level
continuous latent interface, asymmetric long-range history allocation, and bidirectional joint
attention between target levels; Chapter 4 treats it as one point
in the broader context-packaging design space.)*

Recent work broadens the design space beyond hierarchical latent state. WorldPack compresses and
retrieves trajectory memory; CaR/SceneFly and MemLearner study implicit or learned retrieval;
GIM-World introduces geometry-aware memory; and MBench and MemoBench construct explicit
long-memory evaluation tasks [REF: recent long-memory retrieval and benchmark work]. These papers
make architecture-level novelty untenable for generic “video memory.” The dissertation’s novelty
is instead the controlled package-by-budget intervention, exact LoopMom observability,
pair-specific reach populations, tested lower and upper inversions, and outcome-independent
simultaneous inference.

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
budget $w_c N_s^{c} + w_f N_s^{f} \approx \mathrm{const}$ is what extends the explicit context reach
without the flat blow-up (1.7).

**Methodology for comparing hierarchical configurations.** The canonical comparison methodology
of this class is the Clockwork-VAE benchmark
[REF: CW-VAE]: one configuration table over several datasets and metrics, quality-retention
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
VTA [REF: VTA / Variational Temporal Abstraction] (whose lower level resets when the upper level ticks) and the clockwork recurrent
lineage [REF: Clockwork RNN, Koutník et al. 2014] — complete the review of hierarchy *shapes*:
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
in statistical physics [REF: statistical-physics foundations of EBM; REF: Boltzmann-distribution
modeling]. Its basic principle is that the probability of a data configuration $x$ is defined by
the Gibbs/Boltzmann distribution:

$$p_\theta(x) = \frac{\exp\left(-E_\theta(x)\right)}{Z(\theta)} \tag{1.9}$$

where $E_\theta(x)$ is a scalar energy function parameterized by a neural network and $Z(\theta)$
is the partition function. Under a calibrated model, lower energy corresponds to higher model
density; it does not automatically imply semantic or temporal consistency.

An important theoretical bridge connects EBM to diffusion models through score matching [REF:
score matching]: the score function approximated by a diffusion model is the negative gradient of
an (implicit) energy,

$$\nabla_x \log p(x) = -\nabla_x E(x) \tag{1.10}$$

For the deterministic conditional descent mechanism evaluated later in this dissertation, the
sampling pseudo-trajectory is written with its own state $y_k$:

$$
y_{k+1}=y_k-\eta_k\nabla_y E_\theta(y_k;c),
\qquad k=0,\ldots,K-1,
\tag{1.11}
$$

where $c$ is the conditioning input and $\eta_k$ is the prescribed step size. Equation (1.11)
describes the tested gradient-descent sampler; it is not a claim that all EBM inference or
training uses this deterministic chain.

From the standpoint of systems analysis, the attraction is that the energy supplies an explicit
scalar criterion and its input gradient defines an inference direction. This does not guarantee
convergence, calibrated probabilities, or anomaly detection: those properties depend on the
learned landscape and sampler. Transformer-parameterized realizations such as Energy-Based
Transformers (EBT) [REF: Energy-Based Transformers] expose the number of descent steps as an
inference-compute control. This dissertation evaluates that mechanism for conditional video
prediction without attributing that task or result to the original EBT publication.

The particular **differentiable-through-descent** formulation studied here has an important
computational limitation. If a training loss is differentiated through all $K$ updates of
(1.11), the reverse pass must differentiate the input gradient $\nabla_yE_\theta$ and therefore
evaluate Hessian–vector and mixed parameter–state derivative terms (often called “double
backward”). A vanilla unroll retains a layer-by-step graph whose activation storage grows with
$K$ and network depth [REF: EBM training-cost analyses]. This cost characterizes the tested
unrolled-descent formulation; it is not an inherent requirement of every EBM objective, sampler,
or training algorithm.

Approaches to this barrier constitute the resource-efficiency toolbox of multi-step systems.
Gradient checkpointing [REF: gradient checkpointing] trades computation for memory by storing
only a subset of activations and recomputing the rest; a balanced segment schedule can achieve a
square-root memory trade-off in trajectory length. Neural ODE adjoints [REF: Neural ODE] can avoid
retaining the full forward tape but introduce reconstruction and numerical-accuracy concerns.
Discrete reversible neural networks—RevNet [REF: RevNet] and RevViT [REF: RevViT]—instead
reconstruct layer inputs algebraically. This apparatus forms the core of TempVerseFormer in
Chapter 2. It is **not** used to train the final Chapter 3 hierarchical predictor through its
rollout; the two resource strands are synthesized methodologically and measured separately.

A scoped forward reference completes this review. In the present research the energy-based
mechanism was not adopted a priori but *examined experimentally* under controlled conditions: the
matched-capacity regression–diffusion comparison and explicit NFE-sensitivity study of Chapter 4
(§4.2) are reported separately from a two-sided capability study of Energy-Based Transformers,
run with the official published implementation and
both published training recipes at the largest scale reachable on the available hardware. Its
outcome is a **characterized negative** at that scale: one published recipe converges to a stably
*mis-ordered* energy landscape (the mean of the data attains lower energy than the true target);
the other recipe repairs the ordering — the true target becomes the energy minimum among probed
candidates — yet gradient descent from the prescribed initialization still does not reach
structured outputs, and inference-time computation does not buy the capability back. This EBM
evidence comes from a separate 14.66 M-parameter official-code capability study, not from a third
matched-capacity cell. In the shared-interface comparison, the closely matched 7.53 M diffusion
and 7.27 M regression cores instead show regime-dependent strengths: diffusion gives better
best-of-8 stochastic coverage, whereas regression gives better deterministic best-of-1 rollout.
The study is scoped explicitly: it tests a capability
the EBT publication does not claim at a scale it did not run, and contradicts none of the
published claims. For this dissertation its systemic significance is that the choice of the
per-frame generation core in Chapter 3 is made *by controlled measurement, not by paradigm
preference* — and the measurement selects diffusion.

Summarizing the critical analysis of existing approaches: diffusion models are a strong
generation mechanism, but a dense flat realization incurs rapidly growing long-context cost;
fixed-patch transformers can capture global context, while entropy-guided adaptation offers one
way to allocate their tokens non-uniformly; hierarchical latent models make cross-scale and
cross-rate allocation explicit; and energy-based models offer an attractive scalar criterion but
did not provide usable descent generation at the scale tested in Chapter 4. These observations
define an **integrated systems methodology with evidentiary separation**. Chapter 2 studies how
representational capacity can follow an uncertainty proxy and how long differentiation tapes can
be reconstructed. Chapter 3 independently studies how retained context can be packaged across
scales and rates, while Chapters 3–4 turn the Chapter 1 criteria into a measurement instrument.
Only the components explicitly specified in Chapter 3 belong to the executed predictor; the
other strands share the resource problem and decision framework, not an unmeasured architectural
dependency.

## 1.4 Consistency criteria and methods

In the theory of complex systems one of the fundamental notions is the stability of a system
[REF: stability of complex systems] — its capacity to maintain a specified state or target
trajectory under external disturbances and internal stochasticity. In the context of generative
video models the notion of stability is most closely tied to the term "consistency" (coherence,
non-contradiction). Traditionally, the evaluation of generation results in machine learning
reduces to the subjective criterion of visual attractiveness or to statistical metrics of
proximity to the training sample, such as FID [REF: FID]. From the standpoint of systems
analysis such an approach is severely limited.

Systems analysis requires the transition from abstract aesthetic evaluation to a formalized
criteria apparatus. The generative model is treated as a system transforming the input
information signal of the condition $u$ into the output state $x$. If the state $x$ contains
internal logical, spatial, or physical contradictions, it is classified as anomalous.
Accordingly, **consistency is operationalized as the absence or bounded severity of specified
contradictions in the generated state and trajectory**. It is related to, but not synonymous with,
software reliability or dynamical stability.

Since video data have a deep multidimensional structure, consistency cannot be described by one
scalar parameter. It is a complex systemic phenomenon requiring an extended taxonomy. This
subsection analyzes the classical kinds of consistency — spatial, object, and temporal —
integrates the notion of conceptual consistency, and formalizes hierarchical and trajectory
consistency, which are critically important for hierarchical multi-step generative systems.
### 1.4.1 Spatial and object consistency

The basic level of coherence for any visual signal is **spatial consistency**. This criterion
determines the structural integrity of the generated frame at the level of geometry and the
mutual arrangement of elements. Its endpoints concern object proportions, boundaries,
perspective, and lighting coherence; no single endpoint guarantees all of these properties.

The next level is **object consistency**. In many practical tasks the system must generate a
series of states of one and the same object (from different viewpoints, under different
lighting), preserving its unique identifying features. Modern generative algorithms approach this
task in several ways.

The simplest method of achieving object consistency in diffusion models is generation by the
image-to-image approach [REF: image-to-image diffusion]: an existing reference image is used as
the starting point to steer the generation toward a specified result. The diffusion model
transfers the information of the image into its latent space, subtly influencing the direction of
the process.

![Image-to-image conditioning](../../figures/fig41_image_to_image.png)

**Figure 1.13 — Image-to-image conditioning.** Increasing noising strength expands variation but
weakens direct preservation of the reference identity. Source: author’s schematic.

This approach, however, has rigid systemic limitations — the difficulty of precise control: a
higher degree of noising of the reference increases variability and loses the consistency of fine
details (textures, scratches).

A deeper method is fine-tuning of the model itself by approaches of the DreamBooth family [REF:
DreamBooth] and its low-rank variation LoRA [REF: LoRA]. This method uses a set of reference
images of a specific object to adjust the internal representation (the weight coefficients
$\theta$) of the model. Although DreamBooth can secure the consistency of specific features (for
example, facial features or the texture of a particular object), it is computationally heavy and
works exclusively within the "learned" object.

For enforcing strict spatial consistency of shape there exists the ControlNet architecture [REF:
ControlNet], which injects a special control signal (depth maps, contours, segmentation masks)
into the reverse diffusion process. Methods of the ControlNet type perform the function of hard
boundary control of the admissible state space: the input image acts as a guiding form
constraining the evolution of the system.

![ControlNet conditioning](../../figures/fig42_controlnet.png)

**Figure 1.14 — Structural control with a ControlNet-class branch.** A trainable branch maps
edges, depth or segmentation into residual constraints for a frozen diffusion backbone. Source:
author’s schematic after the ControlNet architecture [REF: ControlNet].

Despite the effectiveness of ControlNet, it is not a complete solution for object consistency,
since it focuses on supplied structure rather than persistent internal identity. In autonomous
prediction, retaining or retrieving earlier evidence is one possible mechanism when identity is
not recoverable from the recent view. Chapter 3 tests coarse long-span context as such a carrier;
Chapter 4 also records the boundary where fine full-resolution context preserves identity better.

### 1.4.2 Temporal consistency

Upon the transition from static images to video data the key criterion becomes **temporal
consistency**. It requires that the sequence of generated states $x_1, x_2, \dots, x_T$ form a
logical and smooth video stream. Anomalies of temporal consistency manifest as flicker, sudden
background color changes, or deformation of objects during motion.

One local diagnostic of temporal consistency uses optical flow or a warping operator $W$: where
the correspondence from frame $t-1$ to frame $t$ is valid, the generated state $x_t$ should agree
with the warped previous state. Let $M_t(p)\in\{0,1\}$ select valid, non-occluded pixels $p$, and
let $W_{t-1\rightarrow t}$ warp into frame $t$. A normalized masked pixelwise error is

$$
L_{\text{temp}}(t)=
\frac{\displaystyle\sum_{p}M_t(p)
\left\|x_t(p)-W_{t-1\rightarrow t}(x_{t-1})(p)\right\|_2^2}
{\displaystyle\sum_{p}M_t(p)+\varepsilon},
\qquad \varepsilon>0.
\tag{1.12}
$$

The denominator makes values comparable across different valid-mask areas and $\varepsilon$
handles a frame pair with no valid correspondence.

It must be emphasized, however, that the warp criterion (1.12) is **local**: it tests one
adjacent-frame correspondence condition and does not certify even short-range temporal quality as
a whole. It does not directly evaluate the long-range regime identified in §1.3.1: content that
leaves the field of view exits the valid correspondence mask, so correct re-entry many frames
later is not observed by the intervening local warp terms. Re-entry then depends on whatever
context, recurrent state, learned dynamics, or prior carries the relevant evidence. Long-range
temporal consistency is therefore operationalized separately, as
part of trajectory consistency (§1.4.5), through metrics stratified by re-entry hardness. The
dissertation studies several level-appropriate mechanisms: RoPE-ND and reversible temporal
chaining in the separate [A1]/[A2] strands of Chapter 2, and learned window-relative embeddings,
block-causal autoregressive recursion, and coarse long-span context in the executed Chapter 3
predictor. Their shared role is to expose or manage temporal structure; they are not all
components of one implemented architecture.

### 1.4.3 Style and conceptual consistency

Besides object consistency, complex systems must maintain a unified **stylistic** appearance of a
series of generated data. Modern approaches such as StyleAligned [REF: StyleAligned] solve this
problem through intervention in the attention mechanism (shared attention): target images receive
information about the reference image during generation, copying its global stylistic
characteristics (color palettes, brushwork) without interfering with spatial geometry.

![Shared-attention style conditioning](../../figures/fig43_shared_attention.png)

**Figure 1.15 — Style consistency through shared attention.** Reference attention features are
shared across distinct target trajectories while their content conditions remain separate.
Source: author’s schematic after StyleAligned [REF: StyleAligned].

Article **[A3]** is the published basis for the dissertation-specific consistency-criteria
apparatus. Its contribution is the explicit systems taxonomy and operationalization used here,
not priority over earlier notions of semantic, conceptual, compositional, hierarchical, or
temporal consistency. Within that scoped apparatus, **conceptual consistency** denotes the
criterion defined below.

Marr's multi-stage account of visual representation — progressing from image evidence toward
structured descriptions of the scene [REF: multi-stage visual representation (image evidence
toward a 3-D description)] — motivates a broad engineering analogy, but does not itself prescribe
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
model (CLIP score) [REF: CLIP] is proposed. Let $E_I$ be the visual encoder and $E_T$ the textual
encoder; then the measure of conceptual consistency is defined as:

$$C_{\text{concept}}(x, u) = \frac{E_I(x) \cdot E_T(u)}{\left\|E_I(x)\right\| \left\|E_T(u)\right\|} \tag{1.13}$$

High values of this indicator show alignment in the contrastive embedding space between the
generated image and the textual condition. CLIP score is therefore used as a **semantic-alignment
proxy**; it neither identifies causal relations nor guarantees compositional understanding, and
must be interpreted together with task-specific and human evaluation [REF: CLIP-score validity
and limitations].

### 1.4.4 Hierarchical consistency

For multi-level generative models, this dissertation operationalizes **hierarchical consistency**
as a distinct criterion within the [A3]-based systems taxonomy and the Chapter 3 model. The claim
is the dissertation-specific definition, placement, and measurement role of the criterion—not
that the general idea of agreement across representational levels originates here.

Hierarchical consistency is defined as the strict informational and structural coherence between
the representations of one and the same visual object at different levels of abstraction: the
macro level (semantics), the meso level (structure), and the micro level (pixels or textures). An
anomaly of hierarchical consistency is a situation of internal system conflict: for example, the
macro signal forms the contour of a brick building while at the micro level the system generates
organic wood patterns.

An explicit inter-level communication mechanism can bias a model toward this property, but it
cannot guarantee it: representation loss, imperfect attention, and decoding error can still make
the levels disagree. In the Chapter 3 model, **information-design asymmetry** assigns long-span
history only to coarse tokens, while both target levels are denoised jointly in shared attention
(§3.1.2–3.1.3). Thus long-range observed evidence enters through the coarse channel and can
condition fine prediction. The fine target can nevertheless conflict with the coarse target, so
hierarchical consistency remains an evaluated criterion rather than a structural theorem.
Gradient detachment at the frozen representation boundary and between autoregressive windows
stabilizes the optimization interface; it does not itself establish semantic or structural
agreement between levels.

### 1.4.5 Trajectory consistency

The last — and, for multi-step generative systems, fundamental — criterion is **trajectory
consistency**: the stability of the generation process itself as a trajectory of the complex
system. In the original, single-frame reading it concerned the stability of the iterative
refinement process in pseudo-time $k$; in the present research it is generalized and
operationalized on the physical time axis, where the dominant practical instability of video
generation lives: the degradation of **autoregressive rollout**, in which the system consumes its
own outputs as context and errors compound along $t$ (the realization of the target uncertainty
of §1.2.4 and the exposure bias of §1.3.1).

The dissertation evaluates trajectory consistency through two requirements. First, **bounded
degradation**: prediction error along an autoregressive rollout should grow in a controlled manner,
without divergence or collapse of the generated content. Second, **consistent re-entry**: within
the evaluated horizon, content that left the field of view should be restored correctly when it
returns. Re-entry records whose source views lie outside a short flat package are especially
diagnostic of whether a longer-span context mechanism is useful; they do not make all flat
architectures information-theoretically incapable.

The operationalization of this criterion, developed in Chapters 3–4, stratifies rollout error by
an exactly computable hardness label. For a generated frame $x_t$ with explicit context package
$C$, let the **novel-foreground fraction** $\nu_t(C) \in [0, 1]$ be the fraction of target
foreground whose underlying world columns are absent from the views represented directly in
$C$. In the purpose-built testbed of Chapter 3 this geometric package-novelty quantity is computed
*exactly* from the recorded control-action track rather than estimated. It does not assert that
the model could possess no indirect information about those columns. Trajectory consistency is
then measured as the family of stratified degradation
curves

$$\mathcal{E}_{b}(h) = \mathbb{E}\!\left[\, e\!\left(x_{t+h},\, \hat{x}_{t+h}\right) \,\middle|\, \nu_{t+h} \in b \,\right] \tag{1.14}$$

— the expected prediction error $e$ at rollout horizon $h$ within each hardness bin $b$ (easy:
$\nu = 0$; medium: small $\nu$; hard re-entry: substantial $\nu$) — rather than as a single
aggregate number. This stratification is what makes the criterion *diagnostic*: a flat-context
system and a long-memory hierarchical system can be indistinguishable on aggregate error while
differing qualitatively on the hard re-entry bin beyond the flat window's horizon; conversely, an
anomaly of trajectory consistency (divergence, mean-collapse, identity loss at re-entry)
localizes to specific bins and horizons.

For energy-based mechanisms the trajectory criterion has a specialized diagnostic in
pseudo-time: the progress of the descent chain (1.11) is monitored by the **energy gap**
between the initial and final states of the descent,

$$\Delta E = E_\theta(y_0;c) - E_\theta(y_K;c) \tag{1.15}$$

This raw difference is invariant to an additive energy offset but is not compared across
independently trained energy gauges. It is reported within a fixed model together with the more
direct ordering diagnostic: whether the true target attains lower energy than degenerate
candidates.
If an objective is differentiated through a long trajectory, its practical feasibility also
depends on the activation-storage policy. Chapter 2 compares reversible replay with BPTT and
checkpointing for that setting. The Chapter 3 predictor is instead trained on individual windows;
its measured long-horizon behavior does not depend on reversible rollout training.

![Trajectory consistency operationalization](../../figures/fig44_trajectory_consistency.png)

**Figure 1.16 — Conceptual operationalization of trajectory consistency.** The schematic predicts
separation specifically on memory-relevant re-entry records beyond a short window; Chapter 4
tests this prediction with exact generator-derived observability labels and paired rollouts.
Source: author’s schematic; curves are illustrative, not measured.

The criteria interact but are not interchangeable. A frame can be spatially plausible while
identity is wrong, or retain identity while drifting in trajectory. This motivates reporting a
suite of criteria rather than reducing consistency to one aggregate score.

ControlNet and LoRA address specific conditioning or adaptation objectives. The systems approach
used here complements such mechanisms by making the state interface, context budget, temporal
reach, and evaluation population explicit. Chapters 3–4 construct and evaluate a multi-stream
context model; Chapter 2 measures reversible activation-memory savings separately. The
dissertation does not claim that stop-gradient transfer or reversibility guarantees every form of
consistency.

## 1.5 Conclusions and statement of the research tasks

In the first chapter of the dissertation a thorough systems analysis of the processes of
hierarchical video data generation has been carried out. On the basis of complex-systems theory,
generative video models have been characterized as open dynamic multi-level control systems
operating under substantial structural, informational, computational, and target uncertainty.

The decomposition of the generation system revealed the key problems of contemporary approaches.
The analysis of diffusion models showed that dense flat realizations have two limitations relevant
to this dissertation: a single-level latent interface provides no explicit allocation between
abstraction levels, and a flat spatio-temporal attention context whose computation grows
quadratically and whose
memory grows linearly with context length. This makes dense extension increasingly costly, while
long-video studies report consistency failures when content leaves and later re-enters the field
of view. In addition, the divergence between the local training objective and the long
autoregressive generation trajectory (exposure bias) undermines long-horizon stability.

The review of patch-based transformer architectures demonstrated their advantages in capturing
global contextual dependencies but also identified a possible inefficiency of fixed
spatio-temporal tokenization on non-uniform signals. This motivates the entropy-guided adaptive
decomposition developed in Chapter 2; it does not imply that fixed patches are always inferior.

The study of hierarchical latent approaches established the remedy direction: an explicitly
multi-level latent state space in which a compact upper (coarse) level, carrying few tokens per
frame, can retain a long temporal context at a fixed token budget and thereby serve as the
carrier of the long-range memory of the process. Discrete-token hierarchies remain an important
reviewed alternative with codebook-specific optimization risks. The implemented and evaluated
scope of this dissertation is a **compact continuous hierarchical latent state space**; because
no controlled VQ-versus-continuous receipts are present, representation-family superiority is
not claimed.

The examination of energy-based models (EBM) recorded their systemic attraction—an explicit
scalar optimization criterion and input-gradient inference direction—alongside a computational
barrier of the differentiable-through-descent formulation tested here: differentiating a loss
through a long input-gradient trajectory requires second-order derivative operations and a
horizon-dependent vanilla computation graph. In this dissertation the energy-based mechanism is
treated as reviewed literature whose
capability boundary at reachable scale is characterized experimentally in Chapter 4; the survey
of resource-efficiency methods it motivated (checkpointing, Neural ODE, reversible networks)
identified reversible transformations as one promising apparatus, which Chapter 2 develops
into the time-agnostic backpropagation method and its extension to stochastic multi-level
recurrences.

Furthermore, the chapter organized established consistency concerns into the dissertation-specific
[A3] systems taxonomy and supplied explicit operational roles for conceptual, hierarchical, and
trajectory consistency. The claim is this formalization and operationalization, not priority over
the general terms. Trajectory consistency is measured through autoregressive-rollout degradation
metrics stratified by an exactly computable re-entry hardness label, converting a broad systems
requirement into a diagnostic instrument.

The identified systemic contradiction is between the demand for consistency over longer
processes and finite resources for retaining context and differentiating through computation.
Explicit hierarchy and reversible replay are two candidate interventions on different resource
terms; neither is asserted to be universally necessary. This contradiction determines the
relevance of the dissertation and forms its purpose.

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

4. Develop a method for optimizing the activation-tape resources of multi-step iterative models
   through reversible mathematical transformations; characterize its horizon scaling, extend it
   to stochastic multi-level recurrences with explicit replay-information accounting, and delimit
   its domain of applicability relative to checkpointing.

5. Develop a mathematical model of multi-stream video data generation in compact hierarchical
   latent state spaces, supporting context streams at different temporal rates and spatial
   resolutions over a shared attention backbone, with explicit long-range context allocation and
   a per-frame iterative-refinement generation core selected by controlled mechanism analysis.

6. Create an intelligent information technology and a software complex for systems analysis of
   hierarchical video generation processes: a common methodology connecting adaptive
   representation, activation-resource optimization, context packaging, and a purpose-built
   memory-horizon measurement instrument, while retaining the experimentally verified boundary
   between separately measured components and the executed predictor.

7. Conduct comprehensive experimental studies and approbation of the developed methods and
   technology on synthetic and open datasets — including a matched-capacity regression–diffusion
   comparison with explicit inference-work accounting — and compare their efficiency and consistency with
   modern analogues.

The chapter therefore hands forward two resource questions. Chapter 2 asks **what information
deserves representational capacity** and **what part of a long differentiation tape must be stored
rather than reconstructed**. Chapter 3 asks **which past state must remain addressable at
inference time and how it should be packaged under a fixed context budget**. Chapter 4 then keeps
the evidence for these questions separate while comparing their operating regions. This is the
systems connection among the publication strands; it is not a claim that every method is executed
inside one model.

The results of this chapter are published in the author's work [A3] and in the conference
publications to be identified in the final author-supplied publication list.

---
