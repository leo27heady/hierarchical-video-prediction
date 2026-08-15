# Chapter 3 — Hierarchical Context Model for Long-Horizon Video Prediction

Chapter 2 addressed two constraints that arise before long-range performance can even be tested: activation storage during temporal differentiation and the spatial allocation of representational capacity. It left a different question unresolved. Which observations remain available once prediction proceeds far beyond the initial input? That question belongs to inference state. A model may be trainable, spatially expressive and locally accurate, yet still fail when evidence for a re-entering object has already left its context window.

Architecturally, the isolation is equally explicit. The final predictor contains neither the reversible temporal model nor V-BLT and RoPE-ND, which operate at other system interfaces. Its representation is a frozen two-level continuous codec, its prediction core is a windowed diffusion transformer or regression control, and its fixed-size context buffers are updated with the model's own outputs. This separation permits an observed effect to be assigned to retained context without suggesting that the dissertation validated one jointly trained end-to-end architecture.

## 3.1 Retained context as the final model problem

### 3.1.1 From consistency requirements to memory reach

Long-horizon consistency requires more than a plausible continuation of the latest frame. Consider a digit or textured object that moves beyond the viewport. While it remains absent, several visually reasonable worlds are compatible with the current observation. When it returns, however, its position, extent and identity can be constrained by an earlier view. If that view is still available, the task is retrieval-conditioned prediction. Once it has left memory, the same target becomes a prior-driven guess.

Re-entry becomes decisive after the relevant appearance has left a dense fine-resolution window. A flat model can lengthen that window, but each additional fine frame has a fixed token cost. A multiresolution package instead keeps many older frames at lower spatial resolution and reserves detailed tokens for the recent past. The coarse history may then contain evidence that a token-matched flat window cannot reach. Compression also removes information, so mere inclusion does not guarantee a better prediction. The model must be tested where missing history and lost detail impose competing limitations.

![Re-entry after the last detailed observation leaves a short dense window](../../figures/restructured/ch3_fig3_1_reentry_memory.png)

**Figure 3.1 — Re-entry after the determining appearance leaves a dense context window.** *Source: dissertation-native conceptual schematic based on the LoopMom geometry specified in Section 3.5.*

### 3.1.2 Context-package design space

Let $t$ denote the first frame of the target window. A context package at that boundary is a finite set of streams

$$
\mathcal P_t=\left\{\mathcal S_i\right\}_{i=1}^{L},
\qquad
\mathcal S_i=(\ell_i,s_i,n_i,a_i),
\tag{3.1}
$$

$$
K_i=\{a_i+j s_i\mid j=0,\ldots,n_i-1\},
\qquad
A_i(t)=\{t-k\mid k\in K_i\}.
\tag{3.2}
$$

$$
B_{\mathrm{ctx}}(\mathcal P_t)=\sum_{i=1}^{L}n_i q_{\ell_i}.
\tag{3.3}
$$

$$
B_{\mathrm{tgt}}=\sum_j m_jq_j,
\qquad
B_{\mathrm{total}}=B_{\mathrm{ctx}}+B_{\mathrm{tgt}}.
\tag{3.4}
$$

Each quantity answers a different question. Context tokens measure retained state available to the prediction step; target tokens count the unknown latent cells estimated together; total tokens set the transformer sequence length and therefore affect training and inference work. Context efficiency alone cannot establish end-to-end memory, runtime or attention cost.

$$
R(\mathcal P_t)=\max_i\max K_i
=\max_i\bigl[a_i+(n_i-1)s_i\bigr].
\tag{3.5}
$$

Reach is nominal: it records the oldest supplied slot, while saying nothing about whether its contents are sufficient or actually used. In the executed dense hierarchical packages, the coarse stream covers the 48 immediately preceding frame positions. The temporally sparse control combines recent fine frames with older full-resolution anchors. Its registered “48-frame” label denotes the intended operating-span class, while configuration validation relies on the recorded anchors themselves.

![Flat, spatial-hierarchical and temporally sparse packages](../../figures/fig40_context_package_comparison.png)

**Figure 3.2 — Three ways to package retained video context.** Dense flat history preserves detail, spatial hierarchy preserves dense temporal coverage at lower per-frame cost, and sparse anchors preserve selected full-resolution times; the packages answer different information questions. *Source: author-generated schematic from the frozen package registry and predictor interface.*

Table 3.1 fixes the semantics used throughout the model and results chapters. “Level” names a representation without presuming semantic abstraction; “stride” names anchor spacing without implying that a token summarizes the intervening motion. In particular, a full-resolution anchor stored every four frames differs from a coarse token observed at every frame. Both extend temporal access, but they discard different information.

**Table 3.1 — Context-package fields and their interpretation.**

| Field | Symbol | Operational meaning | What it does not imply |
|---|---:|---|---|
| Representation level | $\ell_i$ | Frozen codec level and its spatial token grid | Semantic correctness or guaranteed cross-level agreement |
| Temporal stride | $s_i$ | Spacing between regular stream anchors | Temporal averaging, unless the encoder explicitly performs it |
| Retained slots | $n_i$ | Number of stored latent maps in the stream | Equal payload across levels |
| Most recent anchor lag | $a_i$ | Distance from the first target to the newest stored anchor | That all intervening frames are present |
| Anchor set | $A_i(t)$ | Exact historical frames represented by the stream | Successful retrieval from those frames |
| Tokens per slot | $q_{\ell_i}$ | Number of spatial cells at the selected level | Bytes, transformer width or full activation memory |
| Context cost | $B_{\mathrm{ctx}}$ | Sum of stored context tokens | Total sequence length or equal computation |
| Addressable reach | $R$ | Oldest supplied lag under the package geometry | Fidelity, information-theoretic necessity or reliable reuse |

### 3.1.3 Reach–fidelity hypothesis

Once packages are explicit, the central prediction can be stated without assuming that hierarchy is generally better. Let $Z_t^\star$ be the historical evidence relevant to a target at time $t$, and let $I_t(\mathcal P)$ indicate whether that evidence lies inside the supplied anchors. Let $Q_t(\mathcal P)$ denote the task-relevant information retained after encoding and packaging. The expected error can be organized conceptually as

$$
\mathcal R(\mathcal P)
=\mathbb E\!\left[
\mathcal L\mid I_t(\mathcal P)=0
\right]\Pr[I_t(\mathcal P)=0]
+\mathbb E\!\left[
\mathcal L\mid I_t(\mathcal P)=1,Q_t(\mathcal P)
\right]\Pr[I_t(\mathcal P)=1].
\tag{3.6}
$$

Equation (3.6) is an explanatory decomposition, not a fitted probabilistic model. Missing historical access drives the first term. The second contains representation loss, model error and self-fed degradation after the relevant past is nominally available. Increasing reach can reduce the first term. Preserving spatial detail can reduce part of the second. Neither change is guaranteed to dominate across all tasks.

![Conceptual reach-fidelity package axes](../../figures/restructured/ch3_fig3_3_reach_fidelity_axes.png)

**Figure 3.3 — Reach and fidelity are separate properties of a context package.** *Source: dissertation-native conceptual diagram; no empirical crossing is implied.*

## 3.2 Hierarchical latent state

### 3.2.1 Two-level frozen autoencoder

Package arithmetic becomes meaningful only after the representation levels are fixed. The implemented packages operate in a continuous latent space produced by a convolutional autoencoder. For an input frame $x_t\in[-1,1]^{1\times64\times64}$, a shared encoder trunk produces an $8\times8$ feature map. One $1\times1$ projection gives the fine code; a further stride-two encoder stage followed by another projection gives the coarse code:

$$
\mathcal E(x_t)=\left(z_t^{\mathrm c},z_t^{\mathrm f}\right),
\qquad
z_t^{\mathrm c}\in[-1,1]^{16\times4\times4},
\qquad
z_t^{\mathrm f}\in[-1,1]^{16\times8\times8}.
\tag{3.7}
$$

Both projections use a hyperbolic tangent, so the raw latent coordinates are bounded. A coarse frame contains $16\times4\times4=256$ coefficients arranged as 16 spatial tokens; a fine frame contains $16\times8\times8=1{,}024$ coefficients arranged as 64 tokens. It does not establish that the coarse code is four times less informative in any general sense.

Each level has its own decoder to the full image:

$$
\hat x_t^{\mathrm c}=\mathcal D_{\mathrm c}(z_t^{\mathrm c}),
\qquad
\hat x_t^{\mathrm f}=\mathcal D_{\mathrm f}(z_t^{\mathrm f}).
\tag{3.8}
$$

$$
\mathcal L_{\mathrm{AE}}=
\begin{cases}
4\bigl(\operatorname{MAE}(x,\hat x^{\mathrm c})+
\operatorname{MSE}(x,\hat x^{\mathrm c})\bigr), & b=\mathrm c,\\[2mm]
\operatorname{MAE}(x,\hat x^{\mathrm f})+
\operatorname{MSE}(x,\hat x^{\mathrm f}), & b=\mathrm f,
\end{cases}
\qquad \Pr(b=\mathrm c)=\Pr(b=\mathrm f)=\tfrac12.
\tag{3.9}
$$

The objective encourages both levels to reconstruct the entire frame. Terms such as “layout” and “detail” summarize their resolution difference; they are not supervised semantic labels. A coarse code may preserve identity information, and a fine code may encode global position. Their actual division of labour is an empirical property that later object-sensitive endpoints can expose.

$$
\tilde z_t^{\ell}[c]=
\frac{z_t^{\ell}[c]-\mu_c^{\ell}}
{\max(\sigma_c^{\ell},10^{-4})},
\qquad \ell\in\{\mathrm c,\mathrm f\}.
\tag{3.10}
$$

![Frozen two-level codec dimensions and token counts](../../figures/restructured/ch3_fig3_4_codec_dimensions.png)

**Figure 3.4 — Frozen two-level codec and latent tensor dimensions.** *Source: dissertation-native schematic of the implemented `HierConvAE` architecture.*

**Table 3.2 — Frozen latent-interface specification.**

| Property | Coarse level | Fine level | Consequence for the predictor |
|---|---:|---:|---|
| Latent shape per frame | $16\times4\times4$ | $16\times8\times8$ | Same channel width, different spatial grid |
| Spatial tokens per frame | 16 | 64 | Fine history costs four times as many context tokens per frame |
| Scalar coefficients per frame | 256 | 1,024 | Payload differs even before projection to transformer width |
| Pixel decoder | Independent coarse decoder | Independent fine decoder | Either code can be inspected as a full-frame reconstruction |
| Predictor input scaling | Per-channel training-set z-score | Per-channel training-set z-score | Noise scale and level-wise losses use comparable standardized coordinates |
| Training after codec freeze | None | None | Package comparisons cannot move the latent interface |
| Primary rollout rendering | Diagnostic only | Frozen fine decoder | Main frame metrics share one output path |

### 3.2.2 Stream construction and embeddings

A latent map becomes a token sequence by treating each spatial cell as one token. If $z_{t,h,w}^{\ell}\in\mathbb R^{16}$ is the channel vector at level $\ell$ and spatial position $(h,w)$, the shared content projection is

$$
e_{t,h,w}^{\ell}=W_{\mathrm{in}}z_{t,h,w}^{\ell}+b_{\mathrm{in}}+p_{h,w}^{\ell},
\qquad e_{t,h,w}^{\ell}\in\mathbb R^d,
\tag{3.11}
$$

$$
S_t=
\mathop{\Vert}_{i=1}^{L}
\mathop{\Vert}_{k\in K_i}
\operatorname{Tok}_{\ell_i}(\tilde z_{t-k}^{\ell_i})
\;\Vert\;
\mathop{\Vert}_{j=0}^{W-1}
\left[
\operatorname{Tok}_{\mathrm c}(y_{t+j}^{\mathrm c})
\Vert
\operatorname{Tok}_{\mathrm f}(y_{t+j}^{\mathrm f})
\right].
\tag{3.12}
$$

$$
h^{(0)}_r=e_r+e_{\mathrm{level}}(\ell_r)+e_{\mathrm{role}}(r_r)
+e_{\mathrm{frame}}(f_r),
\qquad r_r\in\{\mathrm{context},\mathrm{target}\}.
\tag{3.13}
$$

Arithmetic becomes concrete for the standard $48$-coarse plus $6$-fine configuration. Coarse history contributes $48\times16=768$ tokens. The recent fine tail contributes $6\times64=384$. Four targets contain $16+64=80$ tokens each. Thus

$$
|S_t|=
\underbrace{768+384}_{B_{\mathrm{ctx}}=1{,}152}
+\underbrace{4(16+64)}_{B_{\mathrm{tgt}}=320}
=1{,}472.
\tag{3.14}
$$

![Exact token sequence for one prediction step](../../figures/fig22_token_sequence_layout.png)

**Figure 3.5 — Token assembly before joint attention.** *Source: author-generated diagram from the implemented tokenization path.*

$$
c_r=e_{\mathrm{level}}(\ell_r)+e_{\mathrm{role}}(r_r)+e_{\mathrm{frame}}(f_r)
+
\begin{cases}
e_\tau(0), & r_r=\mathrm{context},\\
e_\tau(\tau)+e_a(u_{t+j}), & r_r=\mathrm{target\ at\ frame}\ t+j.
\end{cases}
\tag{3.15}
$$

Padding is unnecessary in the registered two-level arms because every stream length is fixed by its package. Consequently, a token's role and frame label cannot be inferred from a padding convention or an attention mask. They are explicit inputs. This becomes important when the same backbone is reused across package families: a fine token at lag 47 in the sparse stream and a fine token at lag 6 in the recent tail share the content projection, but their frame embeddings differ. The predictor can therefore learn that two equally detailed observations have different temporal relevance without assigning them different channel semantics.

### 3.2.3 Joint-attention context transfer

Coarse and fine tokens are processed together in one transformer sequence. At no point is a coarse prediction completed and handed to a separate fine stage: coarse context, fine context, noisy coarse targets and noisy fine targets coexist in every block. Each layer can therefore transfer information across levels and roles before either output head is read.

$$
Q=\operatorname{RMSNorm}(HW_Q),\quad
K=\operatorname{RMSNorm}(HW_K),\quad
V=HW_V,
\qquad
\operatorname{Attn}(H)=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
\tag{3.16}
$$

Queries and keys are normalized per head before scaled dot-product attention [161]. Multiple heads are concatenated and projected in the usual way [12]. Crucially, attention inside the current four-frame prediction step is full. There is no triangular mask that forces target frame $t+1$ to be generated before $t+2$, nor a parent–child mask that restricts fine cells to a predefined coarse cell. The predictor learns cross-level and within-window dependencies from token identity, conditioning and data.

$$
M^{\mathrm{roll}}_{nm}=
\begin{cases}
0, & m\le n,\\
-\infty, & m>n,
\end{cases}
\tag{3.17}
$$

Joint denoising still exposes only corrupted target candidates. Every target token supplied to a diffusion call belongs to the current noisy block; under regression, the corresponding position is zero. The four candidates are coordinated simultaneously while all observed information remains past-conditioned. Calling this interface “non-causal within block” would therefore be misleading.

![Joint attention within a block and structural rollout causality](../../figures/restructured/ch3_fig3_6_joint_attention.png)

**Figure 3.6 — Joint attention within a prediction block and structural causality across blocks.** *Source: dissertation-native schematic verified against the implemented `WindowDiT` forward path.*

### 3.2.4 Gradient boundaries and information asymmetry

Three boundaries determine which quantities can change during predictor training. Codec parameters are fixed; context latents are detached observations; and the training graph ends at the current target window. If $\theta$ denotes predictor parameters and $\phi$ denotes codec parameters, then

$$
\nabla_\phi\mathcal L_{\mathrm{pred}}=0,
\qquad
\frac{\partial\mathcal L_{n}}{\partial \hat z_{m}}=0
\quad\text{for }m<n,
\qquad
\nabla_\theta\mathcal L_n\ne0.
\tag{3.18}
$$

Decoded-pixel losses and auxiliary coarse-decoder teaching are absent from predictor training. Consequently, a package with more coarse slots cannot receive a different reconstruction-training signal merely because it exposes that pathway more often.

![Frozen, detached and trainable paths](../../figures/restructured/ch3_fig3_7_gradient_paths.png)

**Figure 3.7 — Frozen, detached and trainable paths.** Predictor gradients update token projections, embeddings, transformer blocks and output heads; they neither update the codec nor cross previous autoregressive windows. *Source: dissertation-native schematic of the implemented optimization graph.*

These boundaries make the package comparison cleaner, but they leave exposure bias intact [25, 26]. A longer package can gradually fill with corrupted self-generated values, so reliable operation must be tested under self-fed rollout and across training seeds. A future multi-window objective could use checkpointing or the reversible replay method of Chapter 2 to control activation storage. Such integration remains future work and contributes no evidence to the present model study.

## 3.3 Generative prediction core

### 3.3.1 Windowed latent-diffusion objective

With the package and latent interface fixed, the remaining model problem is to predict the next window. The predictor estimates a block of consecutive future latents conditioned on packaged history and recorded controls. With $W=4$, its conditional law is

$$
p_\theta\!\left(
z^{\mathrm c}_{t:t+W-1},z^{\mathrm f}_{t:t+W-1}
\mid \mathcal P_t,u_{t:t+W-1}
\right).
\tag{3.19}
$$

$$
y_\tau^{\ell}=\alpha_\tau z_0^{\ell}+\sigma_\tau\varepsilon^{\ell},
\qquad
\varepsilon^{\ell}\sim\mathcal N(0,I),
\qquad
\alpha_\tau=\sqrt{\bar\alpha_\tau},\quad
\sigma_\tau=\sqrt{1-\bar\alpha_\tau}.
\tag{3.20}
$$

Schedule coefficients follow the clipped cosine construction of Nichol and Dhariwal [149]:

$$
\bar\alpha_\tau=
\operatorname{clip}\!\left[
\frac{\cos^2\!\left(\frac{\tau/T+s}{1+s}\frac{\pi}{2}\right)}
{\cos^2\!\left(\frac{s}{1+s}\frac{\pi}{2}\right)},
10^{-4},0.9999
\right],
\qquad s=0.008.
\tag{3.21}
$$

$$
v_\tau^{\ell}=\alpha_\tau\varepsilon^{\ell}-\sigma_\tau z_0^{\ell},
\qquad \ell\in\{\mathrm c,\mathrm f\}.
\tag{3.22}
$$

Given both noisy target levels, the transformer also receives the clean context package, the normalized timestep and per-frame actions. Only target tokens are read out. Its loss is

$$
\mathcal L_v(\theta)=
\mathbb E\left[
\operatorname{MSE}\!\left(v_\theta^{\mathrm c},v_\tau^{\mathrm c}\right)
+
\operatorname{MSE}\!\left(v_\theta^{\mathrm f},v_\tau^{\mathrm f}\right)
\right].
\tag{3.23}
$$

![Diffusion, regression, and energy mechanism interfaces](../../figures/restructured/ch3_fig3_8_objective_interfaces.png)

**Figure 3.8 — Prediction-objective interfaces.** *Source: dissertation-native diagram of the implemented training objectives; diffusion formulation follows [2, 119, 149].*

Optimization uses AdamW [159] with learning rate $3\times10^{-4}$, weight decay $10^{-4}$, linear warm-up for 2,000 updates, gradient-norm clipping at 1.0 and exponential moving average decay 0.999. The principal regime uses batch 16 for 30,000 updates; a disclosed batch-8, 60,000-update regime processes approximately the same number of examples but is not identical in gradient noise, update count or effective EMA horizon. Training uses bfloat16 where supported, following the mixed-precision principle of retaining higher precision for sensitive operations [162]. Checkpoints contain model, optimizer and EMA state; later-format checkpoints also preserve random-number-generator state.

### 3.3.2 Transformer backbone and conditioning

The WindowDiT backbone contains eight pre-normalized transformer blocks, width $d=384$, six attention heads and an MLP expansion ratio of four. Its two-level variants contain approximately 22.18–22.19 million trainable parameters; the small difference comes from package-specific input and output interfaces. Every package comparison keeps backbone depth and width fixed. Model scale is therefore controlled while the context intervention remains visible.

Each block follows adaptive LayerNorm-Zero conditioning from the DiT family [4], building on feature normalization within each sample [160]. A modulation network maps $c_r$ from (3.15) to shift, scale and gate vectors for attention and the MLP. For a hidden token $h$,

$$
\operatorname{mod}(h;\beta,\gamma)=h\odot(1+\gamma)+\beta,
\tag{3.24}
$$

and one block updates

$$
\begin{aligned}
h'&=h+g_1\odot
\operatorname{Attn}\!\left(
\operatorname{mod}(\operatorname{LN}(h);\beta_1,\gamma_1)
\right),\\
h^{+}&=h'+g_2\odot
\operatorname{MLP}\!\left(
\operatorname{mod}(\operatorname{LN}(h');\beta_2,\gamma_2)
\right).
\end{aligned}
\tag{3.25}
$$

For the diffusion head, the block modulation projections are initialized to zero, so the residual blocks begin as identity maps in the adaLN-Zero convention. The regression control uses ordinary initialized block modulation rather than that denoiser-specific zero gating. In both modes, the final adaptive projections and output heads begin at zero. After the eighth block, only target positions are selected. Separate adaptive normalizations and linear heads map the coarse and fine target tokens back to 16 codec channels and reshape them into $4\times4$ and $8\times8$ latent maps.

![Complete hierarchical-context predictor](../../figures/fig16_context_packaging_architecture.png)

**Figure 3.9 — Complete predictor from frozen codes to a four-frame latent target.** The codec, WindowDiT and target interface remain fixed while only the retained context package changes in the principal frontier comparisons. *Source: author-generated schematic from the frozen codec and predictor implementation.*

### 3.3.3 DDIM sampling and inference work

Inference begins with Gaussian noise for every coarse and fine target latent in the four-frame block. Deterministic DDIM sampling [52] evaluates the denoiser on a rounded linear subgrid of the training schedule. Given a current iterate $y_i$ at schedule point $\tau_i$, the velocity output is converted to estimates of clean data and noise:

$$
\hat z_{0,i}=\alpha_{\tau_i}y_i-\sigma_{\tau_i}v_\theta(y_i,\mathcal P_t,u,\tau_i),
\qquad
\hat\varepsilon_i=\sigma_{\tau_i}y_i+\alpha_{\tau_i}v_\theta(y_i,\mathcal P_t,u,\tau_i).
\tag{3.26}
$$

$$
y_{i+1}=\alpha_{\tau_{i+1}}\hat z_{0,i}
+\sigma_{\tau_{i+1}}\hat\varepsilon_i,
\qquad i=1,\ldots,N_{\mathrm{DDIM}}-1,
\tag{3.27}
$$

Default evaluation sets $N_{\mathrm{DDIM}}=16$. One four-frame block then costs 16 network function evaluations, or four evaluations per generated frame. For a rollout with $N_{\mathrm{blk}}$ predicted blocks,

$$
\mathrm{NFE}_{\mathrm{roll}}=N_{\mathrm{DDIM}}N_{\mathrm{blk}}.
\tag{3.28}
$$

Candidate count is a separate multiplier. The headline package comparisons use one sampled rollout per registered sampler seed. In the mechanism-selection diagnostic, best-of-eight asks whether at least one of eight candidates covers a target well; it is an oracle coverage statistic, not ordinary deployment error. Eight diffusion candidates at 16 DDIM calls consume 128 predictor evaluations per record, whereas the regression diagnostic uses eight MC-dropout passes. Per-candidate NFE and total candidate-generation work are both reported because “best of eight” does not make the mechanisms compute-matched.

![Deterministic DDIM timeline and evaluation count](../../figures/restructured/ch3_fig3_10_ddim_timeline.png)

**Figure 3.10 — Deterministic DDIM timeline and function-evaluation accounting.** *Source: dissertation-native schematic of the implemented sampler.*

### 3.3.4 Regression control head

The regression variant uses the same tokenization, backbone dimensions, context package and target geometry. It replaces noisy targets with zeros, fixes the time input at zero, interprets the two output heads as clean latent predictions and minimizes

$$
\mathcal L_{\mathrm{reg}}(\theta)=
\operatorname{SmoothL1}(\hat z^{\mathrm c},z^{\mathrm c})
+\operatorname{SmoothL1}(\hat z^{\mathrm f},z^{\mathrm f}).
\tag{3.29}
$$

At ordinary evaluation the regression path is deterministic for fixed weights, inputs and arithmetic. Consequently, a single regression trajectory should not be described as a sample from the same generative family as DDIM. The control asks whether a one-pass conditional estimate can exploit the same packaged history, while the diffusion head asks whether iterative stochastic refinement changes that use.

This separation prevents a common interpretive shortcut. If diffusion exceeds regression on a best-of-eight statistic, the result may reflect multimodal coverage, iterative work, stochastic candidate count or their combination. If regression is stronger in a one-trajectory rollout, that does not invalidate diffusion. The valid conclusion is tied to the declared regime, candidate allocation and endpoint.

## 3.4 Autoregressive rollout and memory update

### 3.4.1 Ring-buffer recurrence

A one-block predictor becomes a long-horizon system only through recurrence. Each completed four-frame block enters the next context, and the process repeats. What persists between calls is the collection of fixed-size latent buffers specified by the active package; the transformer's activation graph ends with the call.

$$
B_{n+1}^{\ell}=
\operatorname{tail}_{N_\ell}\!\left(B_n^{\ell}\Vert\hat Z_n^{\ell}\right),
\qquad
t_{n+1}=t_n+W,
\qquad
(N_{\mathrm c},N_{\mathrm f})=(48,6).
\tag{3.30}
$$

The operator $\operatorname{tail}_{N}$ retains the newest $N$ slots after concatenation. With $W=4$, the fine buffer discards four of its six old values and appends four predictions; the coarse buffer performs the same update over 48 slots. Temporal-sparse and stride-reduced configurations use their declared anchor scheduler, so only target slots aligned to the stream's clock are returned to that stream. The dense frontier pairs, which carry the central reach–fidelity claim, update four coarse and four fine targets per block and keep their target interface identical.

**Algorithm 3.1 — Four-frame self-fed rollout with packaged latent memory**

> **Input:** warm-up frames $x_{0:47}$; controls $u_{48:T-1}$; frozen codec; training-set normalization; EMA predictor; package $\mathcal P$; sampler allocation.
> **Output:** decoded predictions $\hat x_{48:T-1}$ and aligned evaluation records.
> 1. Encode and standardize the warm-up frames, then initialize each stream from its declared anchors.
> 2. At boundary $t_n$, assemble ordered context tokens and read controls $u_{t_n:t_n+3}$.
> 3. Generate four coarse/fine latent pairs with DDIM or the regression control.
> 4. Commit generated slots through the package scheduler and evict obsolete anchors; retain no cross-window gradient graph.
> 5. Unstandardize and decode the fine latents, saving geometry labels and endpoint inputs with each frame.
> 6. Set $t_{n+1}=t_n+4$ and repeat to the declared horizon.

On the standard 200-frame LoopMom evaluation clip, the 48-frame warm-up is followed by 38 rollout blocks, giving 152 predicted frames. Later contexts therefore become entirely self-generated. Buffer capacity remains fixed, but the age and quality of its contents change. This is why a model may have nominal access to a long horizon and still lose reliable information during rollout.

![Ring-buffer update for the principal hierarchy](../../figures/fig23_rollout_ring_buffer.png)

**Figure 3.11 — Coarse and fine ring-buffer updates during self-fed rollout.** *Source: author-generated schematic of Algorithm 3.1 and the implemented rollout path.*

### 3.4.2 Anchor geometry and causal semantics

Buffer length is not enough to reconstruct historical meaning. Each stored slot must carry, or be deterministically assigned, the frame it represents. For stream $i$, slot $k$ at target boundary $t$ is anchored at

$$
\alpha_{i,k}(t)=t-a_i-k s_i,
\qquad
\lambda_{i,k}(t)=t-\alpha_{i,k}(t),
\qquad k=0,\ldots,n_i-1.
\tag{3.31}
$$

The same convention affects stride-reduced coarse context. `hier-s2-min` stores $t-47,t-45,\ldots,t-1$: 24 block-end anchors distributed across 48 frame positions, with oldest lag 47. Its purpose-locked protocol nevertheless uses the registered 48-span label and the predeclared $8<\Delta t\le48$ band. That comparison remains a whole-configuration test; it should not be used to infer a one-frame-precise reach threshold at the band's upper edge.

![Exact context-anchor geometry at the first target](../../figures/restructured/ch3_fig3_12_anchor_geometry.png)

**Figure 3.12 — Stream-anchor geometry at a four-frame prediction boundary.** *Source: dissertation-native schematic based on the package registry and batch constructor.*

Anchor geometry also explains the scope of the B1 stride-2 configuration. Its context uses 24 coarse slots at stride two and two recent fine frames, but the implemented target interface returns two coarse target slots instead of four. Target-token count and the action aggregation for coarse slots therefore change with context stride. A fixed-target version exists as a prepared specification only; because it was not executed, Chapter 4 retains whole-configuration wording.

### 3.4.3 Context, target and total cost

The fixed buffers make retained-state memory independent of rollout duration. For package $\mathcal P$ with $C_\ell=16$ codec coefficients per token and element size $b$ bytes, its contiguous latent payload is

$$
M_{\mathrm{payload}}(\mathcal P)=
bC_\ell B_{\mathrm{ctx}}(\mathcal P).
\tag{3.32}
$$

At float32, one token occupies 64 bytes; at float16 or bfloat16 it occupies 32. The principal 48-coarse plus 6-fine state therefore stores 18,432 latent coefficients, or 72 KiB in float32 and 36 KiB in a 16-bit format. A dense 48-frame fine history stores 49,152 coefficients, or 192 KiB in float32. These are exact array payloads, not measurements of Python containers, allocator behaviour or device peak memory.

**Table 3.3 — Exact package, token and context-payload accounting for the principal frontier and B1 interfaces.**

| Package | Exact context description | Nominal reach | Context / target / total tokens | Context latent payload, fp32 / bf16 | Context embeddings at width 384, bf16 | Comparison role |
|---|---|---:|---:|---:|---:|---|
| `flat-short` | 8 contiguous fine frames | 8 | 512 / 320 / 832 | 32 / 16 KiB | 0.375 MiB | B1 flat comparator |
| `hier-s2-min` | 24 stride-2 coarse + 2 recent fine | 48 | 512 / 288 / 800 | 32 / 16 KiB | 0.375 MiB | B1 whole configuration; target differs |
| `flat-min` | 14 contiguous fine frames | 14 | 896 / 320 / 1,216 | 56 / 28 KiB | 0.656 MiB | matched lower flat control |
| `hier-min` | 48 dense coarse + 2 recent fine | 48 | 896 / 320 / 1,216 | 56 / 28 KiB | 0.656 MiB | matched lower hierarchy |
| `flat-budget` | 18 contiguous fine frames | 18 | 1,152 / 320 / 1,472 | 72 / 36 KiB | 0.844 MiB | matched central flat control |
| `hier-long` | 48 dense coarse + 6 recent fine | 48 | 1,152 / 320 / 1,472 | 72 / 36 KiB | 0.844 MiB | principal spatial hierarchy |
| `strided-flat` | 11 older fine anchors + 6 recent fine | registered 48-span class | 1,088 / 320 / 1,408 | 68 / 34 KiB | 0.797 MiB | near-matched temporal package |
| `flat-c1` | 28 contiguous fine frames | 28 | 1,792 / 320 / 2,112 | 112 / 56 KiB | 1.313 MiB | matched intermediate flat control |
| `hier-c1` | 48 dense coarse + 16 recent fine | 48 | 1,792 / 320 / 2,112 | 112 / 56 KiB | 1.313 MiB | matched intermediate hierarchy |
| `flat-c2` | 38 contiguous fine frames | 38 | 2,432 / 320 / 2,752 | 152 / 76 KiB | 1.781 MiB | matched upper flat control |
| `hier-c2` | 48 dense coarse + 26 recent fine | 48 | 2,432 / 320 / 2,752 | 152 / 76 KiB | 1.781 MiB | matched upper hierarchy |
| `flat-long` | 48 contiguous fine frames | 48 | 3,072 / 320 / 3,392 | 192 / 96 KiB | 2.250 MiB | equal-reach dense reference |
| `hier-rich` | 48 dense coarse + 36 recent fine | 48 | 3,072 / 320 / 3,392 | 192 / 96 KiB | 2.250 MiB | matched top-budget hierarchy |

Self-attention work depends on total sequence length. Ignoring kernel constants, one block has an attention term proportional to

$$
\mathcal C_{\mathrm{attn}}(S,d)=O(S^2d),
\qquad
\frac{\mathcal C_{\mathrm{attn}}(S_1,d)}
{\mathcal C_{\mathrm{attn}}(S_2,d)}
\approx\left(\frac{S_1}{S_2}\right)^2.
\tag{3.33}
$$

![Exact token and latent-payload arithmetic for equal nominal reach](../../figures/restructured/ch3_fig3_13_token_costs.png)

**Figure 3.13 — Context, payload and total-token arithmetic at equal nominal 48-frame reach.** Hierarchical packages reduce retained context and total sequence length while the common target remains visible; the bars are exact interface counts, not performance measurements. *Source: derived from the frozen codec and package registry.*

The resource axis here is context/state memory, distinct from the activation memory studied in Chapter 2. During single-window training, activation memory depends on the arm's declared sequence length—800 to 3,392 tokens among the observed two-level configurations—while remaining independent of the number of rollout blocks evaluated later. Any joint saving would then require direct measurement.

## 3.5 Evaluation instruments and estimands

### 3.5.1 Why short benchmarks cannot identify the reach mechanism

Explicit packages make reach measurable in principle; an ordinary benchmark may still leave the mechanism unidentified. In canonical Moving-MNIST, digits move and collide within a $64\times64$ image, but the generator normally keeps them inside the frame by reflecting their trajectories at the boundary [124]. Future uncertainty and long rollout error are present. The protocol lacks a clean population in which an observed object leaves short context and later returns under geometry that makes longer-history availability computable.

Short benchmarks are therefore retained as competence and boundary instruments, not discarded. Canonical Moving-MNIST and TaxiBJ show whether a temporal predictor can operate on recognizable external protocols [1, 124, 126]. Persistent long Moving-MNIST adds stable object identities and a 300-frame self-fed horizon. Neither provides the exact visibility geometry needed for the central mechanism test. That role belongs to the purpose-built LoopMom generator.

Keeping these roles separate avoids two opposite errors. A synthetic mechanism dataset cannot replace evidence on established prediction tasks, while a standard benchmark cannot identify a memory advantage merely because its test sequence is long. Each table and claim therefore names the property it uses. The instrument was designed around one criterion: it must create records that are hard for a shorter package, answerable by a longer one, and labelable without inspecting model output. Achieving that criterion requires control over the world, viewport and trajectory rather than greater visual realism. The result is intentionally synthetic. Its strength is identification; its limitation is external scope.

### 3.5.2 LoopMom generator and re-entry geometry

Each LoopMom clip begins with a static world $\mathcal W\in[0,1]^{64\times192}$ whose horizontal dimension is toroidal. A smooth background is formed from four sinusoidal components with integer horizontal frequencies, normalized and scaled to the intensity interval $[0.15,0.50]$. Three MNIST glyphs are then placed at fixed world coordinates by pixelwise maximum composition. Horizontal glyph placement wraps around the torus; vertical placement keeps the 28-pixel source patch within the 64-pixel height. The selected world remains unchanged throughout a clip.

The observed frame is a $64\times64$ viewport. Let $r_t\in\{0,\ldots,191\}$ be its saved left-edge coordinate and $a_t\in\{-1,0,+1\}$ the recorded camera action. The executed motion and observation equations are

$$
r_t=(r_{t-1}+4a_t)\bmod192,
\qquad
x_t(y,x)=\mathcal W\!\left(y,(r_t+x)\bmod192\right).
\tag{3.34}
$$

![LoopMom world, viewport and registered reach band](../../figures/fig15_loopmom_observability_schematic.png)

**Figure 3.14 — LoopMom world geometry and analytic reach comparison.** Saved viewport positions make historical visibility computable, allowing a pair-specific test instead of a generic late-horizon comparison. *Source: author-generated schematic from the frozen generator and E2 configuration.*

Two dataset layers support development and protected evaluation. The common training set contains 3,000 clips of 100 frames; predictor training uses clip-disjoint training and validation indices. Evaluation sets contain 500 clips of 200 frames and are generated separately for development, untouched confirmation, sampler robustness, codec robustness and explicit-access analysis. Manifests record purpose, generator commit, random seed, source glyph file, configuration and SHA-256 hashes, consistent with dataset-documentation practice [165].

![Multi-frame disappearance and re-entry event](../../figures/restructured/ch3_fig3_15_reentry_storyboard.png)

**Figure 3.15 — Multi-frame disappearance and re-entry event.** The storyboard shows why late horizon alone is insufficient: the evaluator must know when target content was last visible and whether that observation falls inside each package. *Source: dissertation-native illustration derived from LoopMom geometry; not an empirical rollout.*

### 3.5.3 Exact observability and pair-specific reach populations

The evaluator derives visibility from saved viewport coordinates. For the $m$ frames before target time $t$, define the set of world columns covered by the viewport and the target foreground columns as

$$
\begin{aligned}
\operatorname{cov}_m(t)
&=\bigcup_{u=t-m}^{t-1}
\{(r_u+x)\bmod192\mid x=0,\ldots,63\},\\
F(t)&=\{(r_t+x)\bmod192\mid \max_y x_t(y,x)>0.5\},\\
N_m(t)&=F(t)\setminus\operatorname{cov}_m(t),
\qquad
\nu_m(t)=\frac{|N_m(t)|}{|F(t)|}.
\end{aligned}
\tag{3.35}
$$

Hardness alone leaves earlier observability unresolved. Let $v_t(c)$ be the latest time before $t$ at which world column $c$ was inside the viewport, and let $A_t=\{c\in N_6(t):v_t(c)\text{ exists}\}$. The answerability fraction and time-since-visible label are

$$
\rho_t=\frac{|A_t|}{|N_6(t)|},
\qquad
\Delta t_t=
\operatorname*{median}_{c\in A_t}\bigl[t-v_t(c)\bigr].
\tag{3.36}
$$

The explicit conventions are $\rho_t=0$ when $N_6(t)=\varnothing$ and $\Delta t_t=0$ when $A_t=\varnothing$. A record is called answerable when $|N_6(t)|>0$ and $\rho_t\ge0.5$. Thus an all-recent frame cannot enter a positive reach band, and a genuinely unseen re-entry is reported separately instead of being counted as evidence against a memory package.

$$
\mathcal D_{l:s}^{\mathrm{reach}}=
\left\{(c,t):
R(\mathcal P_s)<\Delta t_t\le R(\mathcal P_l),
\ \rho_t\ge0.5,
\ |F(t)|>0
\right\}.
\tag{3.37}
$$

For `hier-min` versus `flat-min`, the registered band is $14<\Delta t\le48$. The B1 whole-configuration comparison against `flat-short` uses $8<\Delta t\le48$. These bands were derived on the development instrument and frozen before the untouched confirmation set was opened. Other pairs receive their own limits; a generic “hard” label never substitutes for the reach difference of the actual arms.

![Pair-specific reach-differential population](../../figures/restructured/ch3_fig3_16_reach_population.png)

**Figure 3.16 — Pair-specific reach-differential population.** Only records whose relevant evidence lies outside the shorter package but inside the longer package identify the proposed reach mechanism. *Source: dissertation-native schematic of the registered geometric estimand.*

### 3.5.4 Complete frame and object metrics

No single endpoint is allowed to stand in for all forms of consistency. Let $p,g\in[0,1]^{H\times W}$ be a decoded prediction and target, and choose the population-specific threshold $\tau$. Define predicted, ground-truth and union masks by $P_\tau=\{p>\tau\}$, $G_\tau=\{g>\tau\}$ and $U_\tau=P_\tau\cup G_\tau$. With squared error $e=(p-g)^2$, the four pixel losses are

$$
\begin{aligned}
\operatorname{MSE}_{\mathrm{full}}&=\frac{\sum e}{HW},
&
\operatorname{MSE}_{\mathrm{fg}}&=\frac{\sum e\mathbf 1_{G_\tau}}{|G_\tau|},\\
\operatorname{MSE}_{\mathrm{bg}}&=\frac{\sum e\mathbf 1_{\neg G_\tau}}{|\neg G_\tau|},
&
\operatorname{MSE}_{\mathrm{union}}&=\frac{\sum e\mathbf 1_{U_\tau}}{|U_\tau|}.
\end{aligned}
\tag{3.38}
$$

Foreground sharpness is measured as per-image masked Laplacian energy, using the Laplacian focus-measure family [163]. SSIM and PSNR remain conventional reconstruction diagnostics [127]. Visual review is also compulsory: decoded prediction, target and last-context panels are saved from the same records used for numerical analysis. The panels diagnose collapse and qualitative mismatch; they are selected by a declared rule and do not replace inference.

Connected components add object-presence information to the frame endpoints [164]. For LoopMom, predicted and target components are paired by the Hungarian algorithm using cost $1-\operatorname{IoU}$ [125]. Zero-overlap assignments are rejected. If $M$ matches survive, missed objects equal $n_{\mathrm{GT}}-M$ and extras equal $n_{\mathrm{pred}}-M$. Mean overlap and centroid distance are calculated only for accepted matches and are always accompanied by miss, extra and coverage quantities. Otherwise, a model could improve a matched-only score by producing fewer objects.

**Table 3.4 — Complete endpoint suite, population and failure sensitivity.** Arrows indicate the conventional favorable direction; effect signs are harmonized separately for pairwise inference.

| Endpoint | Estimand population | Direction | Empty-case rule | Principal failure detected | Main limitation |
|---|---|---:|---|---|---|
| Full-frame MSE | every decoded frame | ↓ | always defined | global pixel error | background can dominate |
| GT-foreground MSE | non-empty $G_\tau$ | ↓ | undefined on empty GT | missed or distorted target content | ignores false foreground elsewhere |
| GT-background MSE | every frame | ↓ | zero only if denominator empty | hallucination outside GT support | can underweight mislocalized foreground |
| Union-mask MSE | every frame | ↓ | zero only if union empty | misses and extras in one support | threshold-dependent |
| Foreground precision / recall / F1 | every frame with explicit mask flags | ↑ | denominator conventions stated above | false positives / misses / balance | pixel overlap is not object identity |
| Masked Laplacian energy | non-empty GT foreground | ↑, interpreted against codec reference | undefined on empty GT | blur or collapse | can reward high-frequency artefacts |
| GT / predicted component count | every frame | equality | zero is valid | object survival and hallucination | connected components may split or merge |
| Missed / extra components | every frame | ↓ | explicit counts | omission / false objects | no persistent identity |
| Matched component IoU | accepted pairs | ↑ | diagnostic when no pairs | shape and overlap | survivor-selected unless coverage is shown |
| Matched centroid distance | accepted pairs | ↓ | diagnostic when no pairs | localization | survivor-selected unless misses are penalized |
| Object coverage | visible GT objects | ↑ | one when no GT object | detection survival | says nothing about class identity |
| Intention-to-score identity | visible labelled GT objects | ↑ | undefined when none are visible | persistent object identity, including misses | classifier- and assignment-dependent |
| Intention-to-score trajectory error | visible GT objects | ↓ | zero when no GT object | localization and missed objects | frame-diagonal miss penalty is task-defined |

### 3.5.5 Persistent-object and benchmark boundaries

LoopMom gives exact visibility labels, but its foreground components do not by themselves establish persistent glyph identity through collision and long self-fed motion. Training clips contain 100 frames; development and protected clips contain 348. After 48 warm-up frames, each model predicts 300 frames. Flat and hierarchical packages both use 1,152 context tokens, and three training seeds are evaluated.

$$
\begin{aligned}
c_{gd}&=\tfrac12\min\!\left(\frac{\lVert q_g-q_d\rVert_2}{D},1\right)
+\tfrac12(1-\operatorname{IoU}_{gd}),\qquad c_{gd}\le0.65,\\
\operatorname{coverage}&=\frac{M}{n_{\mathrm{GT}}},\\
\operatorname{traj}&=
\frac{\sum_{(g,d)\in\mathcal M}\lVert q_g-q_d\rVert_2+(n_{\mathrm{GT}}-M)D}
{n_{\mathrm{GT}}},\\
\operatorname{identity}&=
\frac{\#\{(g,d)\in\mathcal M:\operatorname{class}(g)=\operatorname{class}(d)\}}
{n_{\mathrm{GT,labelled}}}.
\end{aligned}
\tag{3.39}
$$

![Persistent-object assignment and identity tracking](../../figures/restructured/ch3_fig3_17_persistent_assignment.png)

**Figure 3.17 — Persistent-object assignment and identity tracking.** *Source: dissertation-native schematic of the frozen E6 evaluator.*

| Instrument | Context → prediction | Property examined | Valid role | Claim it cannot support |
|---|---:|---|---|---|
| LoopMom development | 48 → 152 | package design and geometry-derived strata | hypothesis formation and diagnostics | untouched confirmation |
| LoopMom untouched/protected sets | 48 → 152 | exact reach bands, frontier and named robustness studies | controlled mechanism confirmation | natural-video superiority |
| Persistent long Moving-MNIST | 48 → 300 | support, continuity, identity and trajectory | external synthetic boundary | replication of LoopMom's geometric reach mechanism |
| Canonical Moving-MNIST | protocol-specific short horizon | motion-prediction competence | benchmark check and negative control | hierarchy equivalence or re-entry benefit |
| TaxiBJ | OpenSTL protocol | spatio-temporal field prediction | external competence | long-memory object consistency |

**Table 3.5 — Evaluation datasets and their evidentiary roles.** A dataset change changes the scientific question; benchmark competence is not silently promoted to mechanism confirmation.

Changing the dataset changes the question. The persistent-object study tests whether the reach–fidelity interpretation remains useful when identity becomes explicit, while remaining an external synthetic boundary. Natural-video validation would require moving cameras, dynamic hidden content and controlled re-observation labels, or a benchmark whose annotations permit a comparable estimand.

### 3.5.6 Statistical protocol

Pairing is applied before resampling: the same clip and target horizon appear in both arms, and stochastic samplers use common random numbers for registered paired cells [169]. A lower-is-better endpoint uses comparator minus candidate; a higher-is-better endpoint uses candidate minus comparator, so positive effects consistently favour the registered candidate. Pairing reduces noise while leaving the comparison licence unchanged. A near-matched package remains near-matched, and a whole-configuration intervention remains confounded at component level.

$$
\begin{aligned}
\widehat\Delta_j
&=\frac{\sum_c\sum_{i\in c}d_{jci}}{\sum_c n_c},\\
\widehat\Delta_j^{*(b)}
&=\frac{\sum_{c\in C_b}\sum_{i\in c}d_{jci}}{\sum_{c\in C_b}n_c},\\
T_b^{\max}
&=\max_j\left|
\frac{\widehat\Delta_j^{*(b)}-\widehat\Delta_j}{\widehat s_j^*}
\right|,
\qquad
\mathrm{CI}_j^{\mathrm{sim}}
=\left[
\widehat\Delta_j-q_{0.95}(T^{\max})\widehat s_j^*,
\widehat\Delta_j+q_{0.95}(T^{\max})\widehat s_j^*
\right].
\end{aligned}
\tag{3.40}
$$

Verdicts follow the sign of the simultaneous interval together with the predeclared endpoint direction. An interval entirely above zero supports the registered candidate for that contrast; an interval entirely below zero supports the comparator; overlap with zero remains unresolved. Seed replication can strengthen or qualify that verdict, but thousands of within-clip frames cannot manufacture training-seed replication. This is why clip counts, record counts and model-seed counts appear as separate quantities in Chapter 4.

Equivalence is not inferred from an interval that crosses zero. It would require a justified smallest effect of interest in the endpoint's native units and reliable measurement under that margin. No such margin was frozen for the central programme, so unresolved means unresolved. Negative and mixed outcomes remain terminal evidence when their registered decision rules are satisfied; they do not trigger an unplanned search for a more favourable population.

## 3.6 Research hypotheses and chapter conclusions

### 3.6.1 Registered hypotheses and expected directions

The model and instruments above support five hypothesis families. They are stated before the results so that a later observation cannot redefine what the package was expected to do.

**H2 — Fidelity recovery.** As a dense flat package approaches the history span of the hierarchical package, preserving full-resolution evidence should become more valuable. The tested budget points may therefore move from hierarchy-favouring to unresolved and then flat-favouring outcomes. The hypothesis does not require monotonicity, a unique crossing or any conclusion between sampled budgets.

**H5 — Self-fed reliability and external boundary.** Nominal access need not remain trustworthy through a long rollout. A seed-specific early/late pattern therefore requires outcome-independent replication before it can become a boundary claim. On persistent-object long Moving-MNIST, support/trajectory and identity endpoints may separate because a coarse stream can preserve where an object is while losing detail about what it is.

**Table 3.6 — Hypothesis-to-estimand mapping fixed for Chapter 4.**

| Hypothesis | Primary intervention | Primary population or family | Expected informative pattern | Interpretation if not observed |
|---|---|---|---|---|
| H1 Reach differential | matched long hierarchy vs shorter flat package | answerable non-empty pair-specific reach band | positive candidate effect concentrated between reaches | reach account weakened for that pair |
| H2 Fidelity recovery | matched package pairs across sampled token budgets | one simultaneous frontier family | lower hierarchy region, possible unresolved middle, upper flat recovery | no interpolation; report sampled outcomes only |
| H3 Package shape | sparse fine anchors vs dense coarse history | near-matched package contrast, crossed with head and seed | stable direction only if seed and head evidence support it | package use is conditional or heterogeneous |
| H4 Robustness | sampler/NFE grid and independent codec | separately frozen simultaneous families | registered directions persist with disclosed precision | sensitivity localized to the changed interface |
| H5 Self-fed/external boundary | repeated long-loop seeds and persistent objects | fixed early/late contrasts; five object endpoints | replication required for a trust boundary; support may separate from identity | retain negative or mixed verdict without post-hoc rescue |

These hypotheses are conditional by construction. They identify operating regions for a particular frozen codec, predictor family and set of controlled tasks. Chapter 4 can broaden confidence through seeds and protected datasets, but it cannot turn that design into a universal ordering of hierarchical and flat video models.

### 3.6.2 Chapter conclusions

Chapter 3 has converted retained history from an informal architectural intuition into an explicit model variable. A package is defined by representation levels, temporal strides, retained slots and anchor geometry. Context, target and total tokens are counted separately. Reach then identifies which past observations can enter a prediction, while fidelity describes the task-relevant information that survives representation and sampling. The distinction leads to a conditional expectation: hierarchy can help where it uniquely includes the relevant past, whereas dense fine context can recover when both packages reach the event and detail becomes limiting.

At the systems level, the earlier methods remain complementary: spatial allocation and activation storage are controlled at interfaces outside the present experiment. With interfaces, hypotheses and estimands fixed, Chapter 4 can now answer the empirical question that motivated the design: where does longer hierarchical reach pay, and where does the fidelity retained by flat context become more valuable?
