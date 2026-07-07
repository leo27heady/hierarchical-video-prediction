# 03 — РОЗДІЛ 2 (rewrite, phase 1)

> **Status:** English working text of the full РОЗДІЛ 2 replacement; the Ukrainian pass translates
> it mechanically. The Ukrainian original (`scratchpad/diss/diss_full.md`, РОЗДІЛ 2) is untouched.
> Genre spots are marked **[UA-PASS: …]**, author-input spots **[AUTHOR: …]**, unresolved
> literature citations **[REF: …]**.
> **Sources of truth:** `docs/dissertation/DISS_DIGEST.md` (§1 РОЗДІЛ 2 TOC, §3 items 22–25, §6
> НОВИЙ РОЗДІЛ 2, §8 decisions Q1/Q2/Q5), `docs/dissertation/rewrite/01_VSTUP.md` (fixed
> vocabulary), `docs/THEORY_CHAPTER_D.md` (T1–T4 measured numbers),
> `docs/draft/ARTICLE_SKELETON.md` §5 + T6a/T6b (frontier table, taxonomy),
> `docs/FINAL_ARTICLE_REVIEW_2026-07-02.md` §1-Q3 (inner-loop accounting). Every number below is
> traceable to those files; none are invented.
> **What changed vs the original chapter (per DISS_DIGEST):** §2.1 HVEBT-energy contrast
> retargeted (item 22); §2.2 "system significance for HVEBT" retargeted to the hierarchical model
> of Chapter 3 (item 23); §2.4 closing bridge («Зв'язок із марковською оптимізацією в енергетичних
> моделях», the T↔K isomorphism over-claim) **replaced** by the new closing section (item 24);
> §2.5 conclusions rewritten (item 25); Table 2.1 checkpointing entry fixed to
> $O(\sqrt{T}\cdot L\cdot D)$. Everything else survives faithfully (~90% per §6). "HVEBT" naming
> is dropped everywhere (decision Q5). Numbering 2.1–2.5 is preserved; new equations continue
> from (2.14); new figures continue from Fig. 2.14.

---

# РОЗДІЛ 2 — МАТЕМАТИЧНІ МЕТОДИ ОЦІНЮВАННЯ НЕВИЗНАЧЕНОСТІ ТА РЕСУРСОЕФЕКТИВНОГО МОДЕЛЮВАННЯ БАГАТОКРОКОВИХ ВІДЕОПРОЦЕСІВ

*(EN: Chapter 2 — Mathematical Methods for Uncertainty Estimation and Resource-Efficient
Modeling of Multi-Step Video Processes. Title per DISS_DIGEST §6; the old title «Математичне
моделювання та розкриття невизначеності у багатокрокових генеративних системах» is superseded.
[UA-PASS: keep the spec-124 idiom «розкриття невизначеності» inside the body text].)*

## 2.1 Метод оцінювання просторово-часової невизначеності відеоданих на основі ентропійних показників

*(EN: Method for estimating the spatio-temporal uncertainty of video data based on entropy
metrics.)*

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
[REF: Dosovitskiy et al. 2021, ViT] and their spatio-temporal extensions ViViT
[REF: Arnab et al. 2021, ViViT] and Video Swin Transformer [REF: Liu et al. 2022, Video Swin],
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
[REF: Pagnoni et al. 2024, BLT]. These systems abandoned the paradigm of fixed token vocabularies
in favor of dynamic grouping of elementary information units (bytes) into latent patches whose
size is inversely proportional to the local complexity of the signal.

> **[FIG 2-1 — PLACEHOLDER]**
> **Content:** Illustration of the conceptual operating principle of the Byte Latent Transformer
> (BLT): entropy-driven variable-size patching of a byte stream.
> **Source:** KEEP (original draft, Рисунок 2.1).

Within the dissertation research, the results of which are published in **[A1]**, the hypothesis
was put forward that transferring the principles of adaptive entropy-based tokenization to the
visual modality opens fundamental opportunities for optimizing and redistributing the
computational resources of a generative system. On the basis of this approach, the author
developed the method for estimating spatio-temporal uncertainty and a new mathematical
architecture — the Vision Byte Latent Transformer (V-BLT).

Working at the level of "raw" bytes of video data, where each pixel of an RGB channel is
represented by a discrete integer from 0 to 255, allows the system to avoid the loss of
information that is unavoidable when heuristic pre-compression or vector-quantization (VQ-VAE)
algorithms are used [REF: van den Oord et al. 2017, VQ-VAE]. The developed V-BLT architecture is
a three-tier hierarchical information-transformation system consisting of a Local Encoder, a
Global Latent Transformer, and a Local Decoder. In the arc of this dissertation, V-BLT is the
author's **first realization of the hierarchical adaptation of video generation** — a two-stage
decomposition in which local byte-level stages surround a global latent stage and an entropy
criterion governs the allocation of capacity between them; the mature form of this hierarchical
decomposition, in which the upper level carries the long-range temporal memory of the process,
is developed in Chapter 3.

> **[FIG 2-2 — PLACEHOLDER]**
> **Content:** Overall architecture of the Vision Byte Latent Transformer (V-BLT): Local Encoder →
> Global Latent Transformer → Local Decoder.
> **Source:** KEEP (original draft, Рисунок 2.2).

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
information leakage from the future [REF: information leakage in causal sequence modeling], while
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
a classification problem using the cross-entropy loss function [REF: cross-entropy
classification objective]. This makes it possible to correctly model multimodal distributions:
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

> **[FIG 2-3 — PLACEHOLDER]**
> **Content:** Example of V-BLT operation on a 2D shape: ground-truth frames / predicted frames /
> per-pixel entropy maps.
> **Source:** KEEP (original draft, Рисунок 2.3).

> **[FIG 2-4 — PLACEHOLDER]**
> **Content:** Example of V-BLT operation on a 3D polycube: ground-truth frames / predicted
> frames / per-pixel entropy maps.
> **Source:** KEEP (original draft, Рисунок 2.4).

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
consistency of the generated process as a whole. *(Replaces the old contrast with the "global
energy criterion of HVEBT" per DISS_DIGEST item 22.)*

The successful substantiation and approbation of the entropy-based method for estimating
spatio-temporal uncertainty opens the path to building fully resource-efficient video-generation
models. However, the process of transforming a multidimensional spatio-temporal data array into a
linear sequence of patches for transformer processing carries a serious mathematical problem —
the loss of structural topology. The next subsection is devoted to solving this problem by
adapting and integrating multidimensional rotary positional encoding.

## 2.2 Забезпечення структурної когерентності даних за допомогою багатовимірного позиційного кодування

*(EN: Ensuring the structural coherence of data by means of multidimensional positional
encoding.)*

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
[REF: absolute positional encoding, Vaswani et al. 2017]. This method adds a special coordinate
vector $p_i$ to each latent token $z_i$. However, absolute encoding has fundamental drawbacks in
this application: it destroys translation invariance, which is a natural property of the visual
world, since the system must recognize and generate an object equally well regardless of its
absolute coordinates in the frame.

An alternative approach is relative positional encoding (Relative Positional Bias)
[REF: relative positional bias, e.g. Shaw et al. 2018 / Swin], where coordinate information is
added directly to the attention-weight matrix in the form of biases. Although this method models
relative distances more correctly, it is accompanied by a significant growth of computational
complexity and requires $O(L^2)$ memory for storing multidimensional bias matrices, which
directly contradicts one of the main tasks of this research — the optimization of resource
consumption.

To overcome these systemic contradictions, within the development of the information technology
it is proposed to apply the mathematical apparatus of Rotary Positional Embedding (RoPE)
[REF: Su et al. 2021/2024, RoPE] with its subsequent generalization to N-dimensional space. The
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

However, the original RoPE method [REF: Su et al., RoPE] was developed exclusively for
one-dimensional sequences such as text and cannot be applied directly to spatio-temporal video
data without losing the multidimensional topology. To solve this problem, within the research
underlying the results of **[A1]**, the method of N-dimensional rotary positional embedding
(RoPE-ND) [REF: RoPE-ND / multi-axis rotary embedding] is used, which allows the system to
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

> **[FIG 2-5 — PLACEHOLDER]**
> **Content:** Visualization of the N-dimensional rotary positional embedding (RoPE-ND) for
> ensuring spatio-temporal coherence.
> **Source:** KEEP (original draft, Рисунок 2.5).

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
and a coarse level with a sparser grid (e.g., $16 \times 16$ tokens). *(Retargeted per
DISS_DIGEST item 23: the old text bound this role to the cascaded energy system "HVEBT"; the
mechanism itself — coordinate scaling for cross-level attention — survives unchanged and now
serves the inter-level context-signal transfer of the hierarchical model of Chapter 3.)*

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

## 2.3 Метод оптимізації обчислювальних ресурсів складної системи при моделюванні багатокрокових часових процесів

*(EN: Method for optimizing the computational resources of a complex system in the modeling of
multi-step temporal processes.)*

The improvement of the mathematical methods for uncertainty estimation (V-BLT) and structural
coherence (RoPE-ND) considered in the previous subsections substantially raises the quality of
spatio-temporal data modeling. However, the practical implementation of iterative generative
models inevitably runs into the exhaustion of available computational resources as a fundamental
systemic constraint. From the standpoint of systems analysis, the effectiveness of controlling
any process is determined not only by the adequacy of the mathematical model but also by the
feasibility of its implementation under strict resource constraints.

In the modeling of multi-step temporal processes, the main bottleneck is not so much the
computational throughput of the processors (the number of floating-point operations, FLOPs) as
the amount of operating memory of the graphics accelerators [REF: GPU memory as the binding
constraint in sequence-model training]. This problem arises from the architectural features of
modern artificial neural networks and their optimization algorithms.

The standard Backpropagation Through Time (BPTT) algorithm [REF: Werbos 1990, BPTT] is the basis
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
common approaches are truncated backpropagation (Truncated BPTT) [REF: truncated BPTT] and
gradient checkpointing [REF: Chen et al. 2016, gradient checkpointing]. Truncated
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
in continuous models of the Neural ODE type [REF: Chen et al. 2018, Neural ODE], but these suffer
from numerical instability during integration. Discrete reversible models, e.g., RevViT
[REF: Mangalam et al. 2022, RevViT; Gomez et al. 2017, RevNet], have shown high stability. The
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

> **[FIG 2-6 — PLACEHOLDER]**
> **Content:** Scheme of information flows during the forward (left) and reverse (right) pass
> through the reversible transformer block.
> **Source:** KEEP (original draft, Рисунок 2.6).

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

> **[FIG 2-7 — PLACEHOLDER]**
> **Content:** Temporal chaining mechanism for autoregressive time-series modeling.
> **Source:** KEEP (original draft, Рисунок 2.7).

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

> **[FIG 2-8 — PLACEHOLDER]**
> **Content:** Global architecture of the TempVerseFormer information technology (VAE spatial
> compression + reversible temporal core).
> **Source:** KEEP (original draft, Рисунок 2.8).

The integration of reversible blocks into a single system with temporal chaining creates an
extremely powerful mathematical instrument: structural reversibility guarantees that information
is not lost. Yet the main systemic challenge remains — converting this theoretical structural
reversibility into an effective training algorithm for computing the control gradients of the
loss function. Exactly how the reversibility property is exploited by the algorithm to remove the
computational uncertainty is considered formally in the next subsection.

## 2.4 Математичне обґрунтування часонезалежного зворотного розповсюдження помилки на основі оборотних перетворень

*(EN: Mathematical substantiation of time-agnostic backpropagation based on reversible
transformations.)*

Solving the problem of computational uncertainty — the critical deficit of operating memory in
the optimization of iterative and temporal systems — requires a transition from classical
training algorithms to new mathematical paradigms. The traditional BPTT method
[REF: Werbos 1990, BPTT] is the de-facto standard for identifying the parameters of dynamic
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

> **[FIG 2-9 — PLACEHOLDER]**
> **Content:** Forward-pass algorithm of TempVerseFormer (activation-free forward evolution).
> **Source:** KEEP (original draft, Рисунок 2.9).

**Phase two — the reverse pass.** This phase is the conceptual core of the method. The reverse
pass starts with the computation of the gradient of the loss function with respect to the final
state of the system, $\nabla_{y_T} L_{\text{loss}}$. Then, for every time step $t$ from $T$ down
to $1$, the algorithm moves in the reverse direction: it reconstructs the layer inputs by the
inverse transformations (2.11)–(2.12), locally rebuilds the computation graph of a single
transition, computes the local gradients, and immediately destroys the local graph.

> **[FIG 2-10 — PLACEHOLDER]**
> **Content:** Reverse-pass algorithm of TempVerseFormer (reconstruct → local graph → gradient →
> free).
> **Source:** KEEP (original draft, Рисунок 2.10).

> **[FIG 2-11 — PLACEHOLDER]**
> **Content:** Flowchart of the time-agnostic backpropagation algorithm (two phases).
> **Source:** KEEP (original draft, Рисунок 2.11).

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

*(The checkpointing memory entry is corrected from the original draft's $O(T \cdot L \cdot D)$ to
the standard $O(\sqrt{T} \cdot L \cdot D)$ segment schedule, per DISS_DIGEST §5; with the
recursive schedule it is $O(\log T)$ memory at $O(T \log T)$ compute — [REF: Chen et al. 2016;
Griewank & Walther, treeverse].)*

> **[FIG 2-12 — PLACEHOLDER]**
> **Content:** GPU operating-memory consumption vs optimization-trajectory length for the
> developed architecture compared with analogues ([A2] measurement).
> **Source:** KEEP (original draft, Рисунок 2.12).

> **[FIG 2-13 — PLACEHOLDER]**
> **Content:** Wall-clock time of one optimization iteration vs trajectory length ([A2]
> measurement).
> **Source:** KEEP (original draft, Рисунок 2.13).

As Fig. 2.12 shows, memory consumption remains near-constant even at large prediction depths,
while the computational and time overhead, given the memory gain, is moderate (Fig. 2.13). The
developed method is the only approach that achieves complete independence of memory consumption
from the length of the temporal trajectory. Although the algebraic reconstruction during the
reverse pass requires additional computational operations, this trade-off is highly favorable
from the standpoint of systems analysis: shifting the load from the critically scarce resource —
GPU memory — onto the abundant one — compute cores — allows the mathematical models to scale to
horizons of hundreds and thousands of steps.

In addition, the proposed architecture enables a unique **selective backpropagation** strategy.
In many control problems the initial steps of a process are noisy, while the most informative
steps are the last ones. The method allows the system to run the forward pass to full length and
then reconstruct activations and compute gradients only for a narrow target segment
$[T - K_{\text{opt}},\; T]$, ignoring the rest of the chain. This focuses the optimization on the
relevant portions of the trajectory with zero memory cost for the skipped stages.

### Узагальнення на стохастичні рекурентності та межі застосовності оборотних обчислень

*(EN: Extension to stochastic recurrences and the applicability boundary of reversible
computation. This closing section **replaces** the original bridge «Зв'язок із марковською
оптимізацією в енергетичних моделях» — the claim that the graph isomorphism between the temporal
chain $T$ and a $K$-step MCMC/Langevin optimization chain makes time-agnostic backpropagation the
"critical foundation" of an energy-based cascade. Per DISS_DIGEST item 24, that bridge was an
over-claim: the isomorphism of the graph *structure* is valid, but the memory accounting on the
inner-optimization axis is opposite — see part (iv) below, which proves it rigorously. The
replacement turns the bridge into a scoping result: it establishes exactly where reversible
computation pays and where it provably cannot.)*

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
extending the reversible recurrent networks of [REF: MacKay et al. 2018, Reversible RNN] — the
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
[REF: Behrmann et al. 2019, i-ResNet; Behrmann et al. 2021, numerical invertibility] instantiated
for temporal recurrences and confirmed by measurement. The systemic conclusion: **invertibility
is a memory-and-correctness property, not a stability property.** In the developed technology,
reversibility is claimed exclusively as the constant-memory enabler of long-horizon training with
exact gradients; the stability of the modeled trajectory is a separate design requirement,
enforced by the Lipschitz constraint on the coupling (this requirement informs the architecture
of Chapter 3).

> **[FIG 2-14 — PLACEHOLDER]**
> **Content:** Memory-frontier panel: peak GPU memory vs rollout length $K$ for the three
> training regimes of the same stochastic hierarchical recurrent core — BPTT (12.2 GB at
> $K{=}40$, host-RAM spill at $K{=}80$, OOM at $K{=}120$), gradient checkpointing (1.47–1.97 GB),
> reversible reverse-replay (flat $\approx$1.1–1.2 GB through $K{=}160$). Log-scale memory axis;
> annotate the OOM boundary and the 16 GB device ceiling.
> **Source:** NEW — plot from the measured Panel-A table (`hvp_rev/mem_frontier.py`; numbers in
> `docs/SESSION_JOURNEY_AND_EBT_PLAN.md` §2.2 = ARTICLE_SKELETON T6b; figure F9 of the article
> skeleton). No new runs required.

**(iii) The constant-memory autoregressive-rollout frontier (measured).** The practical payoff of
the extended method is measured on the real training loop of the implemented stochastic
hierarchical recurrent core (three levels, batch 32, Adam optimizer, 16 GB GPU; environment
details in Chapter 4 **[AUTHOR: confirm the §4.1 hardware description — RTX 4060 Ti 16GB — and
cross-reference it here]**). Table 2.3 gives the peak training memory as a function of the
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
   its descendants [REF: Gomez et al. 2017, RevNet; Mangalam et al. 2022, RevViT]): the
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
[REF: Maclaurin et al. 2015, gradient-based hyperparameter optimization through reversible
learning], **applied** here to the video-generation inner loop, not claimed as a new theorem.
Moreover, plain gradient descent is not closed-form invertible at all (inversion requires the
momentum-buffer scheme of Maclaurin et al., whose kilobyte-scale state itself demonstrates that
trajectory storage was never the binding constraint), and the reversible-sampler variants for
diffusion inner loops are already published [REF: DOODL, Wallace et al. 2023; EDICT,
Wallace et al. 2023; MALI, Zhuang et al. 2021], with large-scale practice choosing checkpointing
or truncation precisely because state $\ll$ graph [REF: AlignProp 2023; DRaFT 2023].

> **[FIG 2-15 — PLACEHOLDER]**
> **Content:** Three-axis taxonomy diagram of reversible computation: axis 1 depth (RevNet,
> literature), axis 2 time (temporal chaining + store-the-noise — the axis that pays; annotate
> with the Table 2.3 frontier), axis 3 computation / inner optimization loop (the axis that
> cannot pay; annotate with the MEASURED four-way panel: naive 18.1 GB spill at K=4 / OOM at
> K=8; truncate/checkpoint/reversible flat 5.28–5.30 GB through K=128; reversible saves 14 MB =
> 0.26% of peak vs checkpointing; analytic band ≈0.6–1.0%; crossover K ≈ 5,000–30,000).
> **Source:** NEW — drawing (figure F11 of the article skeleton; data `logs_rhcp/mem_panel.md`).

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

## 2.5 Висновки до розділу

*(EN: Conclusions to the chapter. Rewritten per DISS_DIGEST item 25: items 1–4 updated lightly —
retargeted forward references, corrected Table 2.1 — and item 5 replaced: the old «інтеграція до
MCMC-оптимізації усуває головний системний бар'єр… відкриваючи шлях до HVEBT» claim is
superseded by the store-the-noise extension + frontier + taxonomy scoping.)*

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
   developed that provides the mathematical basis for the precise spatial localization of
   cross-level cross-attention between the coarse and fine levels of hierarchical systems — the
   basis of the inter-level context-signal transfer in the hierarchical generation model of
   Chapter 3.

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
taxonomy in **[A4]**. **[AUTHOR: bind [A1], [A2], [A4] to the final numbered publication list;
the old draft's «[1], [2], [3], [4]» resolved into the donor bibliography and is abandoned.]**

---

## Figure inventory (old chapter → rewrite)

| Fig. | Content | Disposition |
|---|---|---|
| 2.1 | BLT concept illustration | KEEP |
| 2.2 | V-BLT architecture | KEEP |
| 2.3 | V-BLT on 2D shape + entropy maps | KEEP |
| 2.4 | V-BLT on 3D polycube + entropy maps | KEEP |
| 2.5 | RoPE-ND visualization | KEEP |
| 2.6 | Reversible block forward/reverse flows | KEEP |
| 2.7 | Temporal chaining mechanism | KEEP |
| 2.8 | TempVerseFormer global architecture | KEEP |
| 2.9 | Forward-pass algorithm | KEEP |
| 2.10 | Reverse-pass algorithm | KEEP |
| 2.11 | Time-agnostic backprop flowchart | KEEP |
| 2.12 | GPU memory vs trajectory length ([A2]) | KEEP |
| 2.13 | Iteration time vs trajectory length ([A2]) | KEEP |
| 2.14 | Memory-frontier panel (BPTT / checkpointing / reversible vs $K$) | **NEW** (from measured Panel-A numbers; skeleton F9) |
| 2.15 | Three-axis reversibility taxonomy diagram | **NEW** (drawing; skeleton F11) |

Tables: 2.1 (complexity — checkpointing entry **fixed** to $O(\sqrt{T} \cdot L \cdot D)$),
2.2 (**NEW** — Lipschitz/perturbation-growth characterization), 2.3 (**NEW** — measured memory
frontier).
