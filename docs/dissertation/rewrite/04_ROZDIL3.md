# 04 — РОЗДІЛ 3 (rewrite, phase 1)

> **Status:** English working text of the full РОЗДІЛ 3 replacement (the core model chapter that
> replaces the old HVEBT chapter wholesale); the Ukrainian pass translates it mechanically. Genre
> spots marked **[UA-PASS: …]**, author-input spots **[AUTHOR: …]**, citation spots **[REF: …]**.
> **Sources of truth:** `docs/dissertation/DISS_DIGEST.md` (§3 items 26–34, §6 «НОВИЙ РОЗДІЛ 3»,
> §8 decisions Q1/Q2/Q6/Q7), `docs/STAGE2_DESIGN.md` (pre-registered design + 4-frame-window
> addendum), `hvp_rev/rhvd.py` (THE implementation this chapter describes), `hvp_rev/hier_convae.py`
> (frozen 2-level conv-AE), `hvp_rev/faithful_nets.py` (tokenizer + adaLN-Zero blocks),
> `hvp_rev/step1_mech.py` (latent standardization, metrics), `docs/draft/CH2_TESTBED.md` (testbed
> formalization), `docs/dissertation/rewrite/01_VSTUP.md` (vocabulary).
> **Disposition of the old Ch.3 (per DISS_DIGEST §3):** §3.1.1 (energy criterion) and §3.2 (energy
> core, eqs. 3.6–3.14) are **CUT** here — their content moves, condensed, into the Ch.4 mechanism
> study (§4.2) as the description of what was tested. §3.3 (Reversible MCMC) is **CUT** as a
> contribution; one salvage paragraph survives in §3.3.3 below, pointing to the Ch.2 three-axis
> taxonomy's negative branch. §3.1.3 (cross-level cross-attention) and §3.1.4 (unidirectional
> stop-gradient control) are **SALVAGED and RETARGETED** in §§3.1.2–3.1.3. §3.4 (AR-prediction
> skeleton: block-causal masking, autonomy, per-frame hierarchy) is **SALVAGED** as the skeleton of
> §3.3, with the per-stage MCMC replaced by the actual generation procedure and every routing claim
> mapped honestly onto what `hvp_rev/rhvd.py` does. The macro-first vs micro-first ordering
> contradiction of the old chapter (digest item 30) is **FIXED** in §3.1.3. The
> O(K·L·D)→O(D) vs O(K·T·L·D)→O(T·L·D) inconsistency (digest item 34) is **FIXED** by re-deriving
> memory/complexity for the actual algorithm in §3.3.3. Old-figure dispositions: end of file.

---

# РОЗДІЛ 3 — ІЄРАРХІЧНА МАТЕМАТИЧНА МОДЕЛЬ ГЕНЕРАЦІЇ ВІДЕОПОСЛІДОВНОСТЕЙ ІЗ ДОВГОТРИВАЛИМ БАГАТОПОТОКОВИМ КОНТЕКСТОМ

> **[SUPERVISOR-GATE, packaging reframe 2026-07-10]:** proposed title above (multi-stream
> long-span context — survives the B2 measurement); prior title «…ІЗ ДОВГОТРИВАЛОЮ ПАМ'ЯТТЮ
> ГРУБОГО РІВНЯ» kept as fallback pending Prof. Mokin's sign-off. The coarse level remains
> the concrete stream type of §3.1.1; it is no longer the chapter-title-level principle.

*(EN: Hierarchical mathematical model of video-sequence generation with long-span
multi-stream context.)*

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
justified empirically by the matched-compute mechanism study of Chapter 4.

Two methodological commitments frame the chapter. First, the model is described exactly as
implemented and verified in the software complex (module `hvp_rev/rhvd.py` of the developed
system; the design was pre-registered before the experimental campaign): every architectural
constant quoted below is the constant of the working system, not a proposal. Second, alongside
the generative model itself, the chapter formalizes the **measurement instrument** — a
purpose-built memory-horizon testbed with a re-entry-stratified evaluation protocol (§3.4). Under
the systems-analysis methodology of specialty 124, the instrument is a deliverable of equal rank
with the model: it is what operationalizes the trajectory-consistency criterion of Chapter 1
(§1.4.5) into an exactly computable, statistically disciplined measurement procedure.

## 3.1 Формалізація ієрархічного прихованого простору станів

*(Formalization of the hierarchical latent state space.)*

The chapter's organizing object, defined here once and used throughout (full formalization in
§3.3.5): a **context-stream family**. Each context stream is a triple
$s_i = (r_i,\; k_i,\; n_i)$ — spatial resolution $r_i$, temporal stride $k_i$, and slot count $n_i$
[UA-PASS: «потік контекстних токенів»: роздільна здатність, часовий крок, кількість слотів], and a system configuration is a set of such streams
consumed by one shared attention backbone. In this vocabulary: a **flat** configuration is a
single fine-resolution stride-1 stream; a **temporal (rate) hierarchy** has two or more tick
rates at any resolution; a **resolution hierarchy** has two or more spatial resolutions; a
**deep hierarchy** has three or more streams; the **context budget** counts context tokens only,
with total sequence length and compute reported separately (§4.1). The two-level
coarse-plus-fine model developed next is the family's per-level-decodable resolution form — the
dissertation's demonstrated system, not the family's definition.

### 3.1.1 Дворівнева декомпозиція стану та заморожений автокодувальник

*(Two-level state decomposition and the frozen autoencoder.)*

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
modeling line of work [REF: MilliVid / hierarchical latent video diffusion; REF: CW-VAE; REF:
TECO] in minimal form — is **per-level decodability**: each level is *independently* decodable to
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
   operates in literally the same state space, which is a precondition for a valid matched-compute
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

**[FIG 3-1 — PLACEHOLDER]**
*Content:* overall architecture of the developed model: input frames → frozen two-level conv-AE
(shared encoder trunk; coarse $16{\times}4{\times}4$ and fine $16{\times}8{\times}8$ taps; two
independent decoders back to $64{\times}64$); standardization layer (3.3); the window-DiT
generation core (§3.2) consuming the coarse ring buffer (48 frames), the fine ring buffer (6
frames), the 4 recorded actions and the noise level $\tau$; predicted 4-frame window of latent
pairs; frozen fine decoder rendering the prediction. Freeze symbols on the AE; a dashed
"no-gradient" boundary between the core and the AE.
*Source:* **NEW** (draw from `hvp_rev/hier_convae.py` + `hvp_rev/rhvd.py`; no old figure depicts
this system).

### 3.1.2 Передача контекстних сигналів між рівнями: від перехресної уваги до спільної уваги

*(Inter-level context-signal transfer: from cross-attention to joint attention. — Salvaged from
old §3.1.3, retargeted per DISS_DIGEST item 29.)*

If the two levels of the hierarchy were optimized in complete isolation, the generated states
would contradict each other — the coarse level could place scene content at one position while
the fine level renders texture consistent with another. Guaranteeing the integrity of the
multi-level system requires a strict information channel between the levels; Chapter 1 (§1.4.4)
introduced *hierarchical consistency* as the corresponding criterion. The question is the
mechanism of this **inter-level context-signal transfer** [UA-PASS: «передача контекстних
сигналів між рівнями» — verbatim scope item of the approved topic].

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
   adopted resolutions the joint attention learns the routing directly. **[AUTHOR: the old text
   motivates the hard mask by quadratic-cost and leakage arguments at 32×32 resolution; at 4×4/8×8
   those arguments are moot. Confirm you are comfortable stating the hard mask is *dropped* (I
   recommend yes — it is the truth of the implementation), with the RoPE-ND scaled-coordinate
   mechanism remaining in Chapter 2 as the instrument developed for the A1 system.]**

### 3.1.3 Односпрямоване керування з відсіканням градієнтних зв'язків

*(Unidirectional control with gradient detachment. — Salvaged from old §3.1.4; the macro-first vs
micro-first contradiction of the old chapter is resolved here per DISS_DIGEST item 30.)*

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

This formulation preserves everything defensible in the salvaged mechanism (the stop-gradient
discipline, the constancy of parent context, the prevention of bidirectional gradient
interference) while eliminating the internal contradiction of the old text: the strict
macro-first sequencing claim is dropped because the system that was actually built, validated,
and measured generates both levels jointly in a single attention pass, with the coarse level
distinguished by *what it remembers*, not by *when it runs*.

## 3.2 Генеративне ядро: віконна дифузія у прихованому просторі

*(The generation core: window diffusion in latent space.)*

The per-frame generation mechanism of the developed model is an iterative-refinement (denoising
diffusion) core. This choice is not aesthetic: Chapter 4 (§4.2) selects it by a matched-capacity,
matched-compute comparison of three mechanisms sharing this chapter's backbone — direct
regression, denoising diffusion, and energy-based gradient descent — in which diffusion commits
to sharp, correctly located content where regression hedges toward the conditional mean and the
energy-based alternative fails in a two-sided, characterized manner. This chapter presents the
selected core; the selection evidence belongs to Chapter 4.

### 3.2.1 Токенна послідовність та покомпонентне кондиціонування

*(Token sequence and per-token conditioning.)*

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
the control signal that produced it. [UA-PASS: keep «покадрове кондиціонування керуючим
сигналом» for "per-target-frame action conditioning".]

**[FIG 3-2 — PLACEHOLDER]**
*Content:* token-sequence layout diagram for one prediction step: a horizontal bar of 1472 token
slots segmented as 768 coarse-context (48 frames × 16), 384 fine-context (6 frames × 64), and
4 × (16 + 64) target tokens; frame-index axis 0…51 annotated below; per-token embedding stack
(spatial + level + role + frame) and conditioning vector (noise level; per-frame action) shown as
layered strips; a bracket marking the 1152 context tokens equated to the flat control's budget.
*Source:* **NEW** (from `hvp_rev/rhvd.py::WindowDiT._ctx_tokens/forward`).

### 3.2.2 Трансформерний остов з адаптивною нормалізацією

*(The transformer backbone with adaptive normalization.)*

The sequence (3.4) with conditioning (3.5) is processed by a stack of $N_{\mathrm{blk}} = 8$
pre-norm transformer blocks of width $d = 384$ with 6 attention heads and MLP expansion ratio 4
(≈22 M parameters **[AUTHOR: confirm the exact parameter count from the rhvd training log line
`params=…M`]**). Attention is full (every token attends to every token): the causality of the
overall system is enforced by sequence construction and by the autoregressive recursion (§3.3.2),
not by an attention mask inside the step.

Each block applies **adaLN-Zero conditioning** [REF: DiT — Peebles & Xie, 2023]: the per-token
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

### 3.2.3 Віконна дифузійна цільова функція (v-параметризація)

*(The window-diffusion objective with v-parameterization.)*

Why a *window* of $W = 4$ frames rather than one frame at a time? The reason is
rollout-consistency of the training distribution. The autoregressive rollout of §3.3 appends
*all* frames it generates to the context buffers; if training predicted only a single frame at
offset $k = 4$ (the design of the preliminary single-step harness), the rollout context would
consist of stride-4 samples while the training context consisted of consecutive frames — a
train/rollout distribution mismatch built into the protocol itself. Predicting a window of $W$
*consecutive* frames per step and advancing the recursion by exactly $W$ keeps the buffers
consecutive at all times: the layout of every context the model sees at rollout is identically
distributed with the contexts it saw in training. [UA-PASS: «узгодженість розподілів навчання та
авторегресійного розгортання».]

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

with the cosine noise schedule [REF: improved DDPM — Nichol & Dhariwal, 2021]

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

The network is trained in the **v-parameterization** [REF: progressive distillation — Salimans &
Ho, 2022]: the regression target is the velocity

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
context-noise augmentation as a single budgeted fallback, and Chapter 4 reports that the fallback
was not needed [AUTHOR: confirm against the final G2b outcome].

Training-noise/teacher-forcing scope note for the defense: the long-horizon capability claimed in
Chapter 4 is achieved *without* backpropagation through the rollout — the model is trained on
single windows with clean contexts, and its long memory is an inference-time property of the
information design (§3.1.2), not of a long-horizon training objective. Where a rollout-through-
time objective would be required, the constant-memory machinery of Chapter 2 is the enabling
instrument; the two contributions compose but do not presuppose each other.

### 3.2.4 Детерміноване вибіркове відновлення (DDIM)

*(Deterministic DDIM sampling.)*

Generation inverts the diffusion with a deterministic DDIM sampler [REF: DDIM — Song, Meng,
Ermon, 2021] over $N_{\mathrm{DDIM}} = 16$ steps on a rounded-linear sub-grid
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

### 3.2.5 Регресійне контрольне ядро

*(The regression control core.)*

For the mechanism claims of Chapter 4 (hypothesis H2: the generative mechanism earns its keep at
rollout), the system admits a drop-in **regression** variant on the identical backbone: the noisy
target inputs are replaced by zeros, the noise level is fixed at $\tau \equiv 0$, the output is
interpreted directly as $\hat z_0$, and the objective is the smooth-L1 error on both levels. Its
"sampling" is a single forward pass. Architecture, tokenization, conditioning, context design,
parameter count, optimizer, and evaluation protocol are all shared with the diffusion core — the
two systems differ in the training objective and nothing else. The chapter records this variant
as part of the model family because the honest form of the dissertation's mechanism claim is
comparative, and the comparison is meaningful only under this strict parity.

## 3.3 Алгоритм авторегресійного розгортання з кільцевою буферною пам'яттю

*(The autoregressive rollout algorithm with ring-buffer memory. — Skeleton salvaged from old
§3.4; per-stage MCMC replaced by the actual generation procedure; digest item 33.)*

The developed technology operates as a fully **autonomous** autoregressive predictor: its
inputs are past video frames (through the latent interface) and the recorded low-dimensional
control track; no exogenous modality (in particular, no text) steers the generation. The control
track is not a side channel of convenience: in the systems framing of Chapter 1 the actions *are*
part of the observed state of the control system, and Chapter 4 (§4.2) documents that withholding
them forces every mechanism to hedge over the control-marginal future and manufactures a spurious
blur verdict. The salvaged autonomy principle of the old chapter therefore survives in a
sharpened form: autonomous with respect to semantic side information, conditioned on the recorded
control signal.

### 3.3.1 Рекурсія розгортання

*(The rollout recursion.)*

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
>
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
>
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

**[FIG 3-3 — PLACEHOLDER]**
*Content:* rollout ring-buffer schematic: timeline of a 200-frame clip with the 48-frame GT
warm-up shaded; the two ring buffers (48-slot coarse, 6-slot fine) drawn as circular/stack
registers above the timeline; one recursion step shown as: buffers + 4 actions → DDIM(16) → 4
new latent pairs → append/evict; the "advance by 4" pointer; annotation that buffers stay
consecutive (train = rollout layout) and that memory state is constant, $48 \cdot 256 + 6 \cdot
1024 = 18{,}432$ numbers, independent of rollout length.
*Source:* **NEW** (from `hvp_rev/rhvd.py::rollout_eval`).

### 3.3.2 Часова маршрутизація: чесне відображення блочно-каузального принципу

*(Temporal routing: the honest mapping of the block-causal principle. — Digest item 33.)*

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
  indices *are* relative to the window — the scheme is translation-invariant across the rollout
  by construction, which is the property the old text sought from relative positional coding.
  **[AUTHOR: RoPE-ND from [A1] is used in the A1 (V-BLT) system described in Chapter 2; the final
  hierarchical model uses learned frame embeddings as stated here. Decide whether to (a) present
  both as level-appropriate instruments (recommended: RoPE-ND where token counts and
  cross-resolution alignment demand it — Ch.2; learned indices where the window re-indexing makes
  them equivalent and cheaper — this chapter), or (b) unify by porting RoPE-ND into the final
  model, which would require a new training run and is NOT what the validated system does.]**

### 3.3.3 Аналіз пам'яті та обчислювальної складності

*(Memory and complexity analysis. — Re-derivation fixing the old chapter's inconsistency,
digest item 34; includes the token-budget economics of the hierarchy and the disposition of the
old Reversible-MCMC section, digest item 32.)*

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
network evaluation over 1472 tokens and $N_{\mathrm{blk}} = 8$ blocks; no gradient crosses window
boundaries (§3.1.3). Peak training memory is therefore $O(N_{\mathrm{blk}} \cdot S \cdot d)$ with
$S = 1472$ — independent of both the rollout horizon and the memory horizon $H_{\mathrm{c}}$'s
*temporal* extent (it depends on $H_{\mathrm{c}}$ only through the token count in $S$).

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
which is precisely the matched-budget control of §3.3.4, and the design reason the model's
long-memory claim is a claim about *hierarchy*, not merely about context length: at equal budget,
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

### 3.3.4 Конфігурації системи та контрольні варіанти зі зрівняним бюджетом

*(System configurations and matched-budget controls.)*

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
[UA-PASS: «зрівняний бюджет контекстних токенів» for "matched context-token budget".]

### 3.3.5 Узагальнення на сімейство ієрархічних конфігурацій

*(Generalization to a family of hierarchical configurations — the model basis of the §4.4 shape
study.)*

The token-stream design of §3.2.1 generalizes beyond the two-level, stride-1 family of Table 3.1
without any change to the backbone. A configuration is a set of context streams, each
parameterized by the triple **(роздільна здатність, часовий крок, кількість кадрів)** [UA-PASS:
«потік контекстних токенів», «часовий крок (проріджування)»]: the coarse stream may tick at
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
autoencoder tap (1.25 M), measured and disclosed [UA-PASS: «нейтральність глибини щодо ємності
остова»]; and (ii)
every configuration's context cost is an exact token count, so families are comparable at
**matched budget** — the property that licenses the shape comparisons of §4.4 (спрощені
конфігурації: `hier-s2-min` = 24 slots at stride 2 + 2 fine = 512 tokens; `strided-flat` = 11
full-resolution slots at stride 4 + 6 fine = 1,088; `hier3` = 48 top + 24 coarse + 9 fine =
1,152).

## 3.4 Тестовий стенд горизонту пам'яті та протокол оцінювання зі стратифікацією за повторним входженням

*(The memory-horizon testbed and the re-entry-stratified evaluation protocol — the
systems-analysis measurement instrument.)*

A model whose defining property is a *memory horizon* cannot be validated on benchmarks that do
not exercise one. On Moving-MNIST-class data nothing ever leaves the frame and returns: every
future frame is either fully determined by a short context or permanently unknowable, so there
exists no *hard-but-computable-with-memory* regime in which a long-memory mechanism could
demonstrate its value — this project's own null results on such data (Chapter 1, §1.3; Chapter 4)
trace to exactly this degeneracy. Existing memory-oriented benchmarks (Memory Maze [REF],
TECO-Minecraft [REF], and the recent long-term spatial-memory evaluation protocols for world
models [REF: arXiv 2505.20171; REF: arXiv 2512.02473; REF: arXiv 2506.05284]) pose genuine
long-horizon problems but derive their re-entry events from 3-D egomotion in rendered
environments, where determining *which* pixels of a target frame were knowable from a given
context requires geometry processing at best — and their model scales exceed a single-GPU
budget. The instrument constructed here is the minimal fully analytic member of this family:
hardness labels are computed **exactly** from the recorded control track, and the complete
benchmark trains and evaluates on one 16 GB consumer GPU.

### 3.4.1 Циклічний світ з інерційним панорамуванням

*(The momentum looping world.)*

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

**[FIG 3-4 — PLACEHOLDER]**
*Content:* testbed geometry and re-entry: (top) the 192×64 toroidal world strip with three digits
and the 64×64 viewport window marked, arrow showing the momentum pan and the wrap-around; (middle)
a row of consecutive observed frames showing a digit exiting the right edge and re-entering ~48
frames later; (bottom) timeline annotating loop period ≈48, the 6-frame short reference window,
and a hard re-entry event (target sprite columns never covered by the short window).
*Source:* **ADAPT** (from the article's testbed figure / `docs/draft/CH2_TESTBED.md` FIG ch2-1
materials; regenerate panels from `scripts/gen_looping.py` output).

### 3.4.2 Точні мітки складності повторного входження

*(Exact re-entry hardness labels.)*

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

the fraction of the target sprite that is information-theoretically unknowable from that
context. The protocol fixes the reference window at $m = 6$ frames for *all* systems regardless
of their actual context length ("novel-fg-at-6"), so every arm is stratified on the same hardness
axis, and defines three strata: **easy** ($\nu_6 = 0$ — the sprite was fully visible in the short
window), **mid** ($0 < \nu_6 \le 0.1$), and **hard re-entry** ($\nu_6 > 0.1$ — a substantial part
of the sprite is recoverable only from memory). The single-step mechanism study of Chapter 4 uses
the analogous *novel-strip* label (the same construction restricted to the newly revealed leading
strip of the pan — the only genuinely generative region of a single predicted frame). This
stratification is what turns the trajectory-consistency criterion of §1.4.5 into a measurement:
the dissertation's headline quantities are error deltas *within the hard re-entry stratum* as a
function of rollout horizon.

### 3.4.3 Метрики: маскований передній план, покадрово, після декодування

*(Metrics: foreground-masked, per-image, decoded.)*

Aggregate pixel metrics are excluded by protocol. In this project's own measurements on
Moving-MNIST-class data (≈93% black background), a constant all-black prediction outperformed a
trained CW-VAE [REF] on aggregate SSIM and PSNR — a benchmark on which the empty image wins is
not evidence about dynamics [AUTHOR: the appendix must include the citable table behind this
sentence, per the CH2_TESTBED provenance note]. All metrics below are computed on decoded pixels
(through the frozen fine decoder), under a foreground mask, at the per-image level.

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

### 3.4.4 Статистичний протокол, опорні рівні та межі

*(Statistical protocol, reference floors and ceilings.)*

All system-versus-system comparisons are **paired**: evaluation draws use fixed seeds so every
arm sees identical (clip, target) pairs; stochastic samplers are re-seeded per batch with common
random numbers; and rollout records pair exactly across arms by (clip, horizon) thanks to the
universal 48-frame warm-up (§3.3.1). Headline deltas are reported with 95% confidence intervals
from a paired bootstrap over per-record deltas (10{,}000 resamples, percentile intervals),
computed within each hardness×horizon bin; single-step evaluation pools are sized at ≈768 samples
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
| Statistics | paired bootstrap 95% CI; CRN for same target shapes | 10,000 resamples; ≈768-sample single-step pools; 15,200 rollout records/arm; 2 seeds on headline arms; cross-shape cells share records but not elementwise noise |
| Floors / ceilings | persistence fg-MSE 0.2367 / fg-sharp 0.0515 | k = 4 offset; identical pipeline for every arm |

Scope and limitations of the instrument are stated plainly, as required of a measurement
deliverable: motion is one-dimensional and piecewise-deterministic given the action, so the
generative burden is concentrated in the re-entry events by design; world content is static in
world coordinates, so the testbed does not probe memory of *dynamic* off-screen state; all
conclusions are at 64×64 grayscale and single-digit complexity; and the frozen decoder's
sharpness ceiling caps every downstream comparison. The analytic hardness labels that justify
the design would not transfer unchanged to 3-D egomotion benchmarks, which are positioned as the
natural next rung (Chapter 4, conclusions; ВИСНОВКИ).

## 3.5 Висновки до розділу

*(Conclusions to the chapter.)*

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

The results of this chapter have been published in [A4] **[AUTHOR: bind to the final numbered
publication list; add the relevant conference theses if any cover the hierarchical model or the
testbed]**.

---

## Dispositions of the OLD Ch.3 figures (working notes — not part of the chapter text)

| Old figure | Content (old) | Disposition |
|---|---|---|
| Рис. 3.1 (energy-landscape surface) | 3-D energy landscape with consistent states in minima, MCMC descent trajectory | **MOVE → Ch.4 §4.2** (mechanism-study setup); useful for explaining what the EBT study tested; do not use in Ch.3 |
| Рис. 3.2 (diffusion vs EBM objectives) | local noise-error vs global scalar criterion comparison | **MOVE → Ch.4 §4.2** as the mechanism-comparison framing figure (retitle: three objectives incl. regression) |
| Рис. 3.3 (HVEBT cascade block scheme) | frozen CLIP pyramid, 3 cascade MCMC stages, detached context arrows, text prompt, pixel decoder | **REPLACE** by **FIG 3-1** (new architecture); the detached-context arrow idea survives visually as the frozen-AE boundary; text-prompt input is gone (system is autonomous + action-conditioned) |
| Рис. 3.4 (energy-transformer core + scalar head) | reversible layer stack, scalar energy head, Σ | **MOVE → Ch.4 §4.2**, condensed, as the EBT-arm description (per digest item 31); the *backbone* half is superseded by FIG 3-1/3-2 |
| Рис. 3.6 (Reversible MCMC memory scheme) | forward/reverse passes with graph reconstruction along the K-chain | **CUT**; one paragraph in §3.3.3 records the disposition; if a figure is wanted for the taxonomy's negative branch it belongs in Ch.2 §2.4 |
| Рис. 3.7 (block-causal attention matrix) | green/red frame-block attention matrix, full-spatial + causal-temporal | **ADAPT → optional companion of FIG 3-2/3-3**: redraw as "effective block-causal pattern at window granularity over a rollout" if the defense wants the visual; otherwise cite the construction argument of §3.3.2 |
| Рис. 3.7-bis/3.8 (spatio-temporal data flow in AR prediction) | context (blue) vs target (orange) tensors, causal mask, 3D-RoPE axes inside MCMC loop | **REPLACE** by **FIG 3-3** (ring-buffer rollout); the context/target color convention is worth keeping in all new figures |

## Salvage map of OLD Ch.3 equations (working notes)

| Old eq. | Content | Disposition |
|---|---|---|
| (3.1)–(3.2) | $x^* = \arg\min_x E_\theta(x,u)$; Langevin step | **MOVE → Ch.4 §4.2** (setup of the mechanism study); referenced from Ch.1 §1.3.3 as literature |
| (3.3)–(3.5) | reversible-layer attention/cross-attention/MLP updates | superseded by (3.6)–(3.7) (adaLN-Zero blocks of the actual backbone); reversible blocks remain in Ch.2 |
| (3.6) | per-token scalar energy sum | **MOVE → Ch.4 §4.2** (EBT arm) |
| (3.7)–(3.14) | K-step Langevin training loop + SmoothL1+LPIPS loss | **MOVE → Ch.4 §4.2**, condensed (S1/S2 recipe description); LPIPS term does not survive (not used in the final study) [AUTHOR: verify against final Ch.4 text] |
| (3.15) | per-level Langevin recurrence | **MOVE → Ch.4 §4.2** |
| New (3.1)–(3.18) | as numbered in this file | replace the old chapter's equation register wholesale |
