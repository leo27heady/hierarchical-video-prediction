# Chapter 3 — Hierarchical Context Model for Long-Horizon Video Prediction

Chapter 2 addressed two constraints that arise before long-range performance can even be tested: activation storage during temporal differentiation and the spatial allocation of representational capacity. It left a different question unresolved. Which observations remain available once prediction proceeds far beyond the initial input? That question belongs to inference state. A model may be trainable, spatially expressive and locally accurate, yet still fail when evidence for a re-entering object has already left its context window.

Retained context is therefore the controlled variable of Chapter 3. Historical observations may differ in spatial resolution, temporal rate and recency, while a common predictor consumes every package through one joint-attention sequence. The central implementation keeps a long coarse history and a short fine-resolution tail; flat dense history and temporally sparse fine-resolution history serve as controls in the same vocabulary. *Reach* and *fidelity* describe consequences of the supplied package, not separately trained modules. Its package intervention and experimental programme are documented in [136].

Architecturally, the isolation is equally explicit. The final predictor contains neither the reversible temporal model nor V-BLT and RoPE-ND, which operate at other system interfaces. Its representation is a frozen two-level continuous codec, its prediction core is a windowed diffusion transformer or regression control, and its fixed-size context buffers are updated with the model's own outputs. This separation permits an observed effect to be assigned to retained context without suggesting that the dissertation validated one jointly trained end-to-end architecture.

An explicit package still cannot identify its own memory mechanism. The second half of the chapter therefore builds an instrument in which scene geometry determines the relevant history before model output is examined. LoopMom moves a viewport over a toroidal world, allowing an object to disappear and return after a shorter package has forgotten it. Complete frame and object endpoints protect the comparison against sparse-background and survivor-selection artefacts; paired clip-cluster inference respects dependence among frames from the same self-fed rollout. Chapter 4 reports the outcomes. Here the task is to fix what is compared, what is measured and what each comparison can establish.

## 3.1 Retained context as the final model problem

### 3.1.1 From consistency requirements to memory reach

Long-horizon consistency requires more than a plausible continuation of the latest frame. Consider a digit or textured object that moves beyond the viewport. While it remains absent, several visually reasonable worlds are compatible with the current observation. When it returns, however, its position, extent and identity can be constrained by an earlier view. If that view is still available, the task is retrieval-conditioned prediction. Once it has left memory, the same target becomes a prior-driven guess.

Benchmark design can conceal this difference. A short sequence may keep every relevant object continuously visible, while a full-frame average rewards a model that reproduces the background and omits the object. A recurrent state can persist nominally for hundreds of steps after its recoverable content has faded. Conversely, the required observation may remain in storage but be inaccessible to the predictor. Three claims must therefore remain separate: historical evidence was stored; the current predictor could address it; and self-fed generation reused it reliably. Explicit context construction controls the first two. Reliable reuse remains an empirical question.

Re-entry becomes decisive after the relevant appearance has left a dense fine-resolution window. A flat model can lengthen that window, but each additional fine frame has a fixed token cost. A multiresolution package instead keeps many older frames at lower spatial resolution and reserves detailed tokens for the recent past. The coarse history may then contain evidence that a token-matched flat window cannot reach. Compression also removes information, so mere inclusion does not guarantee a better prediction. The model must be tested where missing history and lost detail impose competing limitations.

![Re-entry after the last detailed observation leaves a short dense window](../../figures/restructured/ch3_fig3_1_reentry_memory.png)

**Figure 3.1 — Re-entry after the determining appearance leaves a dense context window.** The target object is absent from the recent fine history but remains represented in the longer package. The diagram defines a memory-relevant event; it does not assert that coarse representation preserves enough information to solve it. *Source: dissertation-native conceptual schematic based on the LoopMom geometry specified in Section 3.5.*

Sequence length by itself says little about this mechanism. The relevant variable is *addressable historical reach*: the oldest evidence that can enter the current prediction under a declared package and anchor rule. Reach belongs to the supplied interface, not to a model name, recurrent-state duration or nominal clip length. Accordingly, every stream is specified by its level, temporal spacing, retained slots and exact anchors before a comparison is called matched.

### 3.1.2 Context-package design space

Let $t$ denote the first frame of the target window. A context package at that boundary is a finite set of streams

$$
\mathcal P_t=\left\{\mathcal S_i\right\}_{i=1}^{L},
\qquad
\mathcal S_i=(\ell_i,s_i,n_i,a_i),
\tag{3.1}
$$

where $\ell_i$ identifies the representation level, $s_i$ is the temporal stride, $n_i$ is the retained slot count, and $a_i\ge 1$ is the lag of the most recent anchor. For a regular stream, the lag and frame-index sets are

$$
K_i=\{a_i+j s_i\mid j=0,\ldots,n_i-1\},
\qquad
A_i(t)=\{t-k\mid k\in K_i\}.
\tag{3.2}
$$

Calling a package “48 frames of context” is not precise enough. Two streams can contain the same number of slots yet sample different historical positions. Likewise, two packages can have the same oldest lag while retaining different events between the oldest and newest anchors. Irregular packages are represented by storing $A_i(t)$ directly; the regular tuple in (3.1) is only a compact special case. This formulation prevents anchor geometry from being hard-coded around one temporal stride.

If level $\ell$ has spatial dimensions $H_\ell\times W_\ell$, one frame at that level contributes $q_\ell=H_\ell W_\ell$ tokens. Context cost is then

$$
B_{\mathrm{ctx}}(\mathcal P_t)=\sum_{i=1}^{L}n_i q_{\ell_i}.
\tag{3.3}
$$

The target interface is counted separately. If target level $j$ contains $m_j$ slots, then

$$
B_{\mathrm{tgt}}=\sum_j m_jq_j,
\qquad
B_{\mathrm{total}}=B_{\mathrm{ctx}}+B_{\mathrm{tgt}}.
\tag{3.4}
$$

Each quantity answers a different question. Context tokens measure retained state available to the prediction step; target tokens count the unknown latent cells estimated together; total tokens set the transformer sequence length and therefore affect training and inference work. Context efficiency alone cannot establish end-to-end memory, runtime or attention cost.

For regular streams, the oldest supplied lag gives the package reach

$$
R(\mathcal P_t)=\max_i\max K_i
=\max_i\bigl[a_i+(n_i-1)s_i\bigr].
\tag{3.5}
$$

Reach is nominal: it records the oldest supplied slot, while saying nothing about whether its contents are sufficient or actually used. The exact anchor set remains the primary definition for implemented packages; $R$ is a compact summary used to derive comparison-specific populations. In the executed dense hierarchical packages, the coarse stream covers the 48 immediately preceding frame positions. The temporally sparse control combines recent fine frames with older full-resolution anchors. Its registered “48-frame” label denotes the intended operating-span class, while configuration validation relies on the recorded anchors themselves.

![Flat, spatial-hierarchical and temporally sparse packages](../../figures/fig40_context_package_comparison.png)

**Figure 3.2 — Three ways to package retained video context.** A dense flat package preserves full spatial detail at every retained time; a spatial hierarchy keeps long coarse history plus a recent fine tail; a temporally sparse package keeps selected full-resolution anchors plus the same recent tail. The 1,088-token temporal package is near-matched to the 1,152-token spatial package, not token-identical. All values refer to context tokens; target tokens are counted separately. *Source: author-generated schematic from the frozen package registry and predictor interface.*

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

One package vocabulary encompasses several experimental families without turning them into an undifferentiated hierarchy. A single fine stream at stride one is flat dense context. Coarse and fine streams with different spatial grids form a resolution hierarchy. Two fine streams at different temporal rates form temporal packaging without spatial hierarchy. A third codec level increases spatial depth. Those variants share a formal interface, but only declared pairs receive a matched or near-matched comparison licence in Chapter 4.

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

Several predicted operating regions follow. When a flat window ends before the relevant event but a coarse stream includes it, hierarchy should gain most strongly on records between the two reaches. Once both packages include the event, the advantage may narrow because coarse encoding no longer buys unique access. If identity or fine appearance is decisive, a full-resolution package may recover the advantage even at the same token budget. Temporal subsampling creates a third possibility: it can preserve fine appearance at selected times while omitting changes between anchors.

![Conceptual reach-fidelity package axes](../../figures/restructured/ch3_fig3_3_reach_fidelity_axes.png)

**Figure 3.3 — Reach and fidelity are separate properties of a context package.** Moving right can include older evidence; moving upward can preserve more task-relevant detail. The relevant-past threshold is defined by a task event, whereas package ordering above that threshold depends on what information the endpoint requires. *Source: dissertation-native conceptual diagram; no empirical crossing is implied.*

These predictions determine what Chapter 4 must show. A positive aggregate effect supports a reach claim only if it concentrates where the longer package alone can address the relevant past. A flat advantage at a larger budget remains compatible with the hypothesis when both packages reach the event and the endpoint rewards fine information. The experiment searches for operating regions among sampled configurations; it never interpolates a universal frontier between them.

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

Both projections use a hyperbolic tangent, so the raw latent coordinates are bounded. A coarse frame contains $16\times4\times4=256$ coefficients arranged as 16 spatial tokens; a fine frame contains $16\times8\times8=1{,}024$ coefficients arranged as 64 tokens. The fourfold token ratio is the arithmetic basis for retaining a longer coarse history under a fixed context budget. It does not establish that the coarse code is four times less informative in any general sense.

Each level has its own decoder to the full image:

$$
\hat x_t^{\mathrm c}=\mathcal D_{\mathrm c}(z_t^{\mathrm c}),
\qquad
\hat x_t^{\mathrm f}=\mathcal D_{\mathrm f}(z_t^{\mathrm f}).
\tag{3.8}
$$

Independent decodability prevents the coarse level from being only a residual that is meaningless without the fine code. During codec training, one branch is selected at each update. The selected reconstruction uses an $L_1$ plus mean-squared error, and the coarse branch is multiplied by four so that optimization cannot ignore the lower-token pathway:

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

After codec training, all encoder and decoder parameters are frozen. The predictor therefore receives a fixed coordinate system, and compared packages cannot improve by changing their representation during predictor training. Freezing also prevents pixel-space evaluation losses from leaking into the predictive model. Every decoded comparison uses the same fine decoder within a codec condition; the independently trained codec study later changes the whole frozen interface as a robustness intervention.

Raw latent channels have unequal scales. Diffusion noise would be miscalibrated if unit Gaussian noise were applied directly to coordinates whose standard deviation is much smaller than one. For each level and channel, training-set statistics are therefore fixed and applied to all later splits:

$$
\tilde z_t^{\ell}[c]=
\frac{z_t^{\ell}[c]-\mu_c^{\ell}}
{\max(\sigma_c^{\ell},10^{-4})},
\qquad \ell\in\{\mathrm c,\mathrm f\}.
\tag{3.10}
$$

Only training clips contribute to $\mu_c^{\ell}$ and $\sigma_c^{\ell}$. Held-out and protected populations reuse those constants. Before decoding, a predicted fine code is unstandardized with the matching codec statistics and passed through $\mathcal D_{\mathrm f}$. Normalization is therefore part of the frozen model interface and is applied identically during evaluation.

![Frozen two-level codec dimensions and token counts](../../figures/restructured/ch3_fig3_4_codec_dimensions.png)

**Figure 3.4 — Frozen two-level codec and latent tensor dimensions.** Both levels are independently decodable, but they allocate spatial tokens differently: 64 fine cells and 16 coarse cells per frame. Predictor gradients stop at the standardized latent interface. *Source: dissertation-native schematic of the implemented `HierConvAE` architecture.*

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

The codec is deliberately compact and task-specific, and its grayscale $64\times64$ operating point limits external validity. For the controlled question, however, it supplies two stable, independently interpretable token rates over the same observations. A more elaborate hierarchical codec could change compression loss and model capacity together, making the package mechanism harder to isolate.

### 3.2.2 Stream construction and embeddings

A latent map becomes a token sequence by treating each spatial cell as one token. If $z_{t,h,w}^{\ell}\in\mathbb R^{16}$ is the channel vector at level $\ell$ and spatial position $(h,w)$, the shared content projection is

$$
e_{t,h,w}^{\ell}=W_{\mathrm{in}}z_{t,h,w}^{\ell}+b_{\mathrm{in}}+p_{h,w}^{\ell},
\qquad e_{t,h,w}^{\ell}\in\mathbb R^d,
\tag{3.11}
$$

where $p_{h,w}^{\ell}$ is a learned level-specific spatial embedding. The same linear map $W_{\mathrm{in}}$ projects the 16 codec channels at both resolutions, while separate positional tables account for the $4\times4$ and $8\times8$ grids. The resulting coordinate interface is fixed-grid and learned directly at the two implemented resolutions.

For a package $\mathcal P_t$ and a target window of $W$ frames, tokenization yields one ordered sequence

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

Here $y$ denotes a noisy target latent during diffusion and a zero placeholder under the regression control. The stream order is deterministic: optional top-level context, coarse context, older fine anchors, recent fine context and then target groups. Ordering alone does not carry all identity information. Every token also receives learned level, role and frame-index embeddings:

$$
h^{(0)}_r=e_r+e_{\mathrm{level}}(\ell_r)+e_{\mathrm{role}}(r_r)
+e_{\mathrm{frame}}(f_r),
\qquad r_r\in\{\mathrm{context},\mathrm{target}\}.
\tag{3.13}
$$

Frame indices are expressed in the local coordinate system of the current prediction step. Under the main 48-frame package, context positions occupy the historical range and four target frames receive the next four indices. The same indexing convention is rebuilt at every rollout step. Its table therefore remains fixed in size as the absolute video time increases. Such re-indexing is useful operationally, though it should not be mistaken for strict temporal translation equivariance.

Arithmetic becomes concrete for the standard $48$-coarse plus $6$-fine configuration. Coarse history contributes $48\times16=768$ tokens. The recent fine tail contributes $6\times64=384$. Four targets contain $16+64=80$ tokens each. Thus

$$
|S_t|=
\underbrace{768+384}_{B_{\mathrm{ctx}}=1{,}152}
+\underbrace{4(16+64)}_{B_{\mathrm{tgt}}=320}
=1{,}472.
\tag{3.14}
$$

The six fine context frames refer to the same six most recent times already represented in the coarse stream. Their purpose is not to add six units of temporal reach. They enrich the recent end of a long, low-token history. A flat 18-frame control replaces both context streams with 18 consecutive fine maps, which also cost 1,152 context tokens. Target groups remain coarse plus fine in both arms, so the central dense comparisons preserve the 320-token target interface.

![Exact token sequence for one prediction step](../../figures/fig22_token_sequence_layout.png)

**Figure 3.5 — Token assembly before joint attention.** The standard spatial package contains 768 coarse-context tokens, 384 recent fine-context tokens and 320 target tokens. Content projection, identity embeddings and adaptive-normalization conditioning are distinct additions. The target interface is fixed for the main 896-, 1,152-, 1,792-, 2,432- and 3,072-token comparisons. *Source: author-generated diagram from the implemented tokenization path.*

Noise and action information are applied through a second, per-token conditioning path. Let $e_\tau(\tau)$ be the sinusoidal-MLP diffusion-time embedding and $e_a(u)$ the learned embedding of an action $u\in\{-1,0,+1\}$. The conditioning vector for token $r$ is

$$
c_r=e_{\mathrm{level}}(\ell_r)+e_{\mathrm{role}}(r_r)+e_{\mathrm{frame}}(f_r)
+
\begin{cases}
e_\tau(0), & r_r=\mathrm{context},\\
e_\tau(\tau)+e_a(u_{t+j}), & r_r=\mathrm{target\ at\ frame}\ t+j.
\end{cases}
\tag{3.15}
$$

Context is clean evidence and is marked with diffusion time zero. Each target frame receives the action that moves the process into that frame. In stride-reduced target configurations, the actions inside a represented temporal block are summed in embedding space. That interface co-changes with coarse target stride and is why the stride-2 B1 study is interpreted as a whole-configuration comparison, not a pure context-stride intervention. The main dense frontier avoids this issue by keeping four coarse and four fine target slots.

Padding is unnecessary in the registered two-level arms because every stream length is fixed by its package. Consequently, a token's role and frame label cannot be inferred from a padding convention or an attention mask. They are explicit inputs. This becomes important when the same backbone is reused across package families: a fine token at lag 47 in the sparse stream and a fine token at lag 6 in the recent tail share the content projection, but their frame embeddings differ. The predictor can therefore learn that two equally detailed observations have different temporal relevance without assigning them different channel semantics.

### 3.2.3 Joint-attention context transfer

Coarse and fine tokens are processed together in one transformer sequence. At no point is a coarse prediction completed and handed to a separate fine stage: coarse context, fine context, noisy coarse targets and noisy fine targets coexist in every block. Each layer can therefore transfer information across levels and roles before either output head is read.

For hidden states $H\in\mathbb R^{B\times S\times d}$, one attention head computes

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

In one operation, the full matrix permits four classes of exchange: context-to-target retrieval, coarse-to-fine transfer, fine-to-coarse correction and interaction among target positions. The first two are central to the memory hypothesis. A fine target cell can attend to an old coarse observation even though no fine observation was retained at that time; the same target can use recent fine tokens for local appearance. Conversely, a coarse target may use recent fine evidence before being committed to the long buffer. Because no head or layer is assigned a retrieval role in advance, attention weights alone cannot prove that retrieval occurred.

Causality is enforced at a different scale. At rollout step $n$, the sequence contains only context from completed steps and the current target block. A future block has not yet been instantiated. The effective relation over rollout blocks is

$$
M^{\mathrm{roll}}_{nm}=
\begin{cases}
0, & m\le n,\\
-\infty, & m>n,
\end{cases}
\tag{3.17}
$$

where $m$ indexes a block that could supply information to block $n$. Equation (3.17) describes structural causality across the unrolled system. Individual WindowDiT layers receive no corresponding block mask: within block $n$, all context and target tokens can attend to one another. Across blocks, generated values move forward only through the context buffers defined in Section 3.4.

Joint denoising still exposes only corrupted target candidates. Every target token supplied to a diffusion call belongs to the current noisy block; under regression, the corresponding position is zero. Clean target latents appear solely on the loss side during training, so the ground-truth value of a later frame is never available as context for an earlier one. The four candidates are coordinated simultaneously while all observed information remains past-conditioned. Calling this interface “non-causal within block” would therefore be misleading.

![Joint attention within a block and structural rollout causality](../../figures/restructured/ch3_fig3_6_joint_attention.png)

**Figure 3.6 — Joint attention within a prediction block and structural causality across blocks.** Cross-level transfer is learned through full attention over one token sequence. Temporal causality arises because later blocks do not exist until earlier predictions have been committed to the buffers. *Source: dissertation-native schematic verified against the implemented `WindowDiT` forward path.*

Joint attention provides an information path, not a consistency guarantee. A fine target can use long coarse evidence or ignore it, and coarse and fine outputs may disagree after decoding. Nor is one coarse cell hard-wired to four fine cells. Leaving that correspondence learned avoids a potentially incorrect geometric routing rule and places the burden of proof on the cross-level and object-sensitive evaluation in Chapter 4.

Hard routing is absent, and that has a cost. Attention scales with the square of the complete token sequence, and the model must learn level alignment from fixed positional tables and data. A hand-designed local map would be cheaper, but it would also pre-commit the experiment to a particular spatial correspondence. For the controlled package study, unrestricted transfer is preferable because it leaves historical availability—not an engineered parent–child connection—as the intended intervention. Section 3.4.3 reports the resulting sequence-length cost explicitly.

Package design, rather than transformer wiring, creates the asymmetry. A hierarchical arm supplies old observations only at the coarse level and recent observations at both levels. A flat arm supplies the selected history at fine resolution but retains the same coarse and fine target heads. Consequently, an effect can be interpreted as a property of what historical information was packaged, subject to the remaining codec and optimization qualifications. It cannot be described as evidence for a two-stage coarse-to-fine execution because that architecture was not run.

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

The second equality states the single-window credit-assignment rule. During training, contexts are ground-truth latents sampled from the dataset and treated as constants. During evaluation, previously generated latents occupy the buffers, but sampling runs without a gradient graph. The model is therefore exposed to self-fed values only at evaluation in the base protocol. Its long reach is an inference-time state-design property, not the consequence of backpropagating a loss through 152 predicted frames.

Trainable quantities include the shared latent projection, spatial tables, level/role/frame embeddings, action and diffusion-time encoders, adaptive-normalization modules, attention and MLP weights, and the level-specific output heads. The frozen set includes both codec branches and their channel statistics.

Decoded-pixel losses and auxiliary coarse-decoder teaching are absent from predictor training. Consequently, a package with more coarse slots cannot receive a different reconstruction-training signal merely because it exposes that pathway more often.

![Frozen, trainable, and detached gradient paths](../../figures/restructured/ch3_fig3_7_gradient_paths.png)

**Figure 3.7 — Frozen, detached and trainable paths.** Predictor gradients update token projections, embeddings, transformer blocks and output heads. They do not update the codec or propagate across autoregressive windows. *Source: dissertation-native schematic of the implemented optimization graph.*

These boundaries make the package comparison cleaner, but they leave exposure bias intact [25, 26]. A longer package can gradually fill with corrupted self-generated values, so reliable operation must be tested under self-fed rollout and across training seeds. A future multi-window objective could use checkpointing or the reversible replay method of Chapter 2 to control activation storage. Such integration remains future work and contributes no evidence to the present model study.

The same controls narrow the diagnosis of late degradation. With codec adaptation disabled and no gradient crossing rollout windows, failure belongs to the declared package, frozen representation and single-window-trained predictor under repeated self-feeding. Chapter 4 carries that interpretation into every verdict; it cannot identify which untested training intervention might have prevented the error.

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

Four-frame windows solve an interface problem. If training predicted only a distant single frame while rollout appended all intervening predictions, training and evaluation would present different temporal layouts. Here the model predicts frames $t$ through $t+3$ together and the rollout advances by four. Buffer shapes and local frame indices therefore agree between training and evaluation. Their values do not: clean ground-truth context is used for training, whereas later rollout steps consume self-generated latents.

Let $z_0=(z_0^{\mathrm c},z_0^{\mathrm f})$ denote a standardized clean target block. A diffusion index $\tau$ is sampled uniformly from $\{1,\ldots,T\}$ with $T=1000$, and one Gaussian-noise realization is drawn for each level. The same $\tau$ is used across the four frames and both levels in a sample:

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

Training uses the velocity parameterization [119]:

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

Each level-wise mean averages batch, channel, target-frame and spatial entries; the two means are then summed. Per-channel standardization in (3.10) is what makes equal level weighting defensible. Without it, a change in raw latent scale would also change the effective optimization emphasis and diffusion signal-to-noise ratio.

Only the target block is diffused. Corrupting context would create a different test—robustness to noisy memory—and would blur the distinction between reach and denoising difficulty. Clean context also matches the first rollout boundary exactly. Later boundaries are harder for another reason: their context contains model outputs, but those values remain fixed conditioning and receive no artificial diffusion time.

The coarse and fine noise tensors are sampled independently even though they share $\tau$. Copying the other head's corruption cannot solve either target, yet both heads operate at the same schedule signal-to-noise ratio. Their clean targets originate from the same frozen frame encoding. Summing the losses trains paired outputs that can both enter future context, while decoded equality and a deterministic coarse-to-fine relation remain unconstrained.

![Diffusion, regression, and energy mechanism interfaces](../../figures/restructured/ch3_fig3_8_objective_interfaces.png)

**Figure 3.8 — Prediction-objective interfaces.** A common diffusion time couples the target window and levels to one refinement stage, while level-specific noise is predicted through the velocity target. Context remains clean and incurs no loss. Regression is the capacity-aligned WindowDiT control. The energy route belongs to a separately scaled capability study and is not a third matched result cell. *Source: dissertation-native diagram of the implemented training objectives; diffusion formulation follows [2, 119, 149].*

Energy-based prediction uses a different contract. Context and candidate latents enter one transformer, candidate-token energies are summed to a scalar, and inference updates the candidate along the negative gradient of that scalar with a learned descent-step embedding. Training through the final update requires second-order differentiation. Although this mechanism uses related token roles, its official-code study has different scale, objective and inference dynamics. It is introduced here to define the choice evaluated later, then kept outside the matched regression–diffusion comparison.

Optimization uses AdamW [159] with learning rate $3\times10^{-4}$, weight decay $10^{-4}$, linear warm-up for 2,000 updates, gradient-norm clipping at 1.0 and exponential moving average decay 0.999. The principal regime uses batch 16 for 30,000 updates; a disclosed batch-8, 60,000-update regime processes approximately the same number of examples but is not identical in gradient noise, update count or effective EMA horizon. Training uses bfloat16 where supported, following the mixed-precision principle of retaining higher precision for sensitive operations [162]. Checkpoints contain model, optimizer and EMA state; later-format checkpoints also preserve random-number-generator state.

### 3.3.2 Transformer backbone and conditioning

The WindowDiT backbone contains eight pre-normalized transformer blocks, width $d=384$, six attention heads and an MLP expansion ratio of four. Its two-level variants contain approximately 22.18–22.19 million trainable parameters; the small difference comes from package-specific input and output interfaces. Every package comparison keeps backbone depth and width fixed. Model scale is therefore controlled while the context intervention remains visible.

Token count changes while width and depth stay constant. A larger package consequently receives more observations under the same parameter budget, yet incurs more attention work and may be harder to optimize because gradients coordinate a longer sequence. Parameter matching controls capacity; it cannot equalize training cost. Section 3.4.3 exposes the remaining token and payload difference.

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

Conditioning is token-specific rather than pooled once for the whole sequence. Historical tokens carry level, role and local-frame identity with the context-time code; each target group additionally carries its own action and current diffusion time. Adaptive shifts, scales and residual gates are produced from those vectors at every block. This preserves a shared transformer while allowing a coarse context token, a fine context token and a noisy fine target at the same local time to be treated differently.

![Complete hierarchical-context predictor](../../figures/fig16_context_packaging_architecture.png)

**Figure 3.9 — Complete predictor from frozen codes to a four-frame latent target.** The shared codec and WindowDiT remain fixed while flat, spatial-hierarchical and temporally sparse context packages change the retained-history interface. Reach and fidelity describe package consequences, not separate architectural blocks. *Source: author-generated schematic from the frozen codec and predictor implementation.*

Camera action is treated as observed control, not privileged semantic guidance. On LoopMom it determines viewport motion and is therefore part of the modeled dynamic system. Omitting it would force the predictor to average over futures generated by different control trajectories, confounding memory with unobserved action uncertainty. Persistent long Moving-MNIST supplies an all-zero action track through the same interface, preserving software compatibility without pretending that its objects are externally controlled.

Output asymmetry appears only in tensor shape. Coarse and fine heads read their own target slices, but both slices have already interacted with every context stream. No decoder sits inside the transformer, no pixel residual bypasses the latent heads, and no predicted coarse image is concatenated into a fine decoder. These details matter because a visually hierarchical diagram can otherwise suggest a serial model that the implementation does not contain.

### 3.3.3 DDIM sampling and inference work

Inference begins with Gaussian noise for every coarse and fine target latent in the four-frame block. Deterministic DDIM sampling [52] evaluates the denoiser on a rounded linear subgrid of the training schedule. Given a current iterate $y_i$ at schedule point $\tau_i$, the velocity output is converted to estimates of clean data and noise:

$$
\hat z_{0,i}=\alpha_{\tau_i}y_i-\sigma_{\tau_i}v_\theta(y_i,\mathcal P_t,u,\tau_i),
\qquad
\hat\varepsilon_i=\sigma_{\tau_i}y_i+\alpha_{\tau_i}v_\theta(y_i,\mathcal P_t,u,\tau_i).
\tag{3.26}
$$

For $\eta=0$, the next iterate is

$$
y_{i+1}=\alpha_{\tau_{i+1}}\hat z_{0,i}
+\sigma_{\tau_{i+1}}\hat\varepsilon_i,
\qquad i=1,\ldots,N_{\mathrm{DDIM}}-1,
\tag{3.27}
$$

and the final clean estimate is returned after the last evaluation. Determinism here means conditional determinism given the initial noise and model. Different initial draws can still produce different samples. Paired stochastic comparisons use common random numbers so that corresponding arms receive aligned noise streams.

Rounding a linear grid from training index 1,000 down to 1 gives the schedule points. A declared 16-step sample therefore makes exactly 16 calls; it does not hide an extra predictor evaluation for initialization or decoding. Unstandardization and the frozen fine decoder follow sampling and are not denoiser calls. The same distinction is used for regression, whose one predictor call per block is followed by the same rendering path.

Default evaluation sets $N_{\mathrm{DDIM}}=16$. One four-frame block then costs 16 network function evaluations, or four evaluations per generated frame. For a rollout with $N_{\mathrm{blk}}$ predicted blocks,

$$
\mathrm{NFE}_{\mathrm{roll}}=N_{\mathrm{DDIM}}N_{\mathrm{blk}}.
\tag{3.28}
$$

The standard 152-frame rollout contains 38 blocks and therefore uses 608 denoiser evaluations. Alternative 8- and 32-step samplers change this work explicitly; they are not relabelled as matched compute. The model parameters and context package can stay fixed while sampling allocation changes, which makes sampler robustness a distinct experimental question.

Candidate count is a separate multiplier. The headline package comparisons use one sampled rollout per registered sampler seed. In the mechanism-selection diagnostic, best-of-eight asks whether at least one of eight candidates covers a target well; it is an oracle coverage statistic, not ordinary deployment error. Eight diffusion candidates at 16 DDIM calls consume 128 predictor evaluations per record, whereas the regression diagnostic uses eight MC-dropout passes. Per-candidate NFE and total candidate-generation work are both reported because “best of eight” does not make the mechanisms compute-matched.

Common random numbers reduce paired Monte Carlo variation but do not make two learned conditional distributions identical. A sampler-seed robustness check changes only the initial diffusion noise after training. An NFE check changes the numerical path through the same learned velocity field. Neither substitutes for a new training seed, and neither establishes that a package effect is insensitive to every possible inference allocation.

![Deterministic DDIM timeline and evaluation count](../../figures/restructured/ch3_fig3_10_ddim_timeline.png)

**Figure 3.10 — Deterministic DDIM timeline and function-evaluation accounting.** Context and actions are reused at every refinement step. The sampler's stochastic variation comes from its initial noise; inference work scales with declared NFE rather than being hidden inside the term “one prediction.” *Source: dissertation-native schematic of the implemented sampler.*

### 3.3.4 Regression control head

The regression variant uses the same tokenization, backbone dimensions, context package and target geometry. It replaces noisy targets with zeros, fixes the time input at zero, interprets the two output heads as clean latent predictions and minimizes

$$
\mathcal L_{\mathrm{reg}}(\theta)=
\operatorname{SmoothL1}(\hat z^{\mathrm c},z^{\mathrm c})
+\operatorname{SmoothL1}(\hat z^{\mathrm f},z^{\mathrm f}).
\tag{3.29}
$$

Generation requires one network evaluation per four-frame block. The control is thus closely matched in backbone capacity but not in objective or inference work. Diffusion receives a corrupted target and a noise-level condition, predicts velocity and iterates; regression receives zero placeholders, predicts a conditional point estimate and returns once. These co-changes are intrinsic to the mechanism comparison and are reported rather than concealed.

At ordinary evaluation the regression path is deterministic for fixed weights, inputs and arithmetic. MC dropout is activated only for the named best-of-eight coverage diagnostic; it is not used in the principal self-fed rollout. Consequently, a single regression trajectory should not be described as a sample from the same generative family as DDIM. The control asks whether a one-pass conditional estimate can exploit the same packaged history, while the diffusion head asks whether iterative stochastic refinement changes that use.

Regression has two roles. It checks whether a package effect depends on the generative head, and it supplies a deterministic best-of-one operating point. It is not an inferior placeholder whose only purpose is to validate diffusion. Chapter 4 therefore evaluates head-by-package interaction across training seeds instead of assuming that a context package will be used identically by both objectives. Energy-based descent is examined there as a separately scaled capability study; it is not a third matched cell of the WindowDiT comparison.

This separation prevents a common interpretive shortcut. If diffusion exceeds regression on a best-of-eight statistic, the result may reflect multimodal coverage, iterative work, stochastic candidate count or their combination. If regression is stronger in a one-trajectory rollout, that does not invalidate diffusion. The valid conclusion is tied to the declared regime, candidate allocation and endpoint.

## 3.4 Autoregressive rollout and memory update

### 3.4.1 Ring-buffer recurrence

A one-block predictor becomes a long-horizon system only through recurrence. Each completed four-frame block enters the next context, and the process repeats. What persists between calls is the collection of fixed-size latent buffers specified by the active package; the transformer's activation graph ends with the call.

For the principal spatial hierarchy, let $B_n^{\mathrm c}$ hold 48 coarse latents and $B_n^{\mathrm f}$ hold six recent fine latents immediately before rollout block $n$. If the model generates the paired latent block $\hat Z_n^{\ell}=(\hat z_{t_n}^{\ell},\ldots,\hat z_{t_n+W-1}^{\ell})$, the dense update is

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

Before self-fed generation begins, every compared system receives a common 48-frame ground-truth warm-up. An arm with a shorter context takes the suffix or scheduled anchors required by its package, but it begins predicting at the same absolute frame as all other arms. This common start makes records pairable by clip and horizon. It also avoids giving a short-context arm an earlier and therefore easier prediction boundary.

The operational procedure is summarized below.

**Algorithm 3.1 — Four-frame self-fed rollout with packaged latent memory**

> **Input:** warm-up frames $x_{0:47}$; recorded controls $u_{48:T-1}$; frozen encoder and decoder; training-set normalization statistics; EMA predictor parameters; package $\mathcal P$; sampler allocation.  
> **Output:** decoded predictions $\hat x_{48:T-1}$ and aligned evaluation records.  
> 1. Encode all warm-up frames, standardize both latent levels and initialize every stream from its exact anchor set.  
> 2. Set the first target boundary to $t_0=48$.  
> 3. For rollout block $n=0,1,\ldots$, construct the ordered context tokens from the current stream buffers.  
> 4. Read the four controls $u_{t_n:t_n+3}$ and generate four coarse/fine latent pairs with DDIM or the regression control.  
> 5. Commit the generated slots to the package scheduler; evict slots that no longer belong to the next anchor set. No gradient graph is retained.  
> 6. Unstandardize and decode each predicted fine latent. Save the frame, clip identifier, horizon, geometry labels and all endpoint inputs before continuing.  
> 7. Advance $t_{n+1}=t_n+4$ and repeat until the declared horizon is reached.

On the standard 200-frame LoopMom evaluation clip, the 48-frame warm-up is followed by 38 rollout blocks, giving 152 predicted frames. Later contexts therefore become entirely self-generated. Buffer capacity remains fixed, but the age and quality of its contents change. This is why a model may have nominal access to a long horizon and still lose reliable information during rollout.

![Ring-buffer update for the principal hierarchy](../../figures/fig23_rollout_ring_buffer.png)

**Figure 3.11 — Coarse and fine ring-buffer updates during self-fed rollout.** Each call predicts four latent pairs, then appends and evicts four positions while preserving the 48-coarse/6-fine geometry. The layout matches the training interface; the values progressively move away from the clean-context training distribution. *Source: author-generated schematic of Algorithm 3.1 and the implemented rollout path.*

The recurrence uses the predicted fine latent for rendering and both predicted levels for future conditioning. Returning both levels is essential. A rollout that re-encoded its decoded fine frame at every step would introduce a different round-trip path and would no longer match the trained latent-target interface. The implemented buffers receive the model's direct latent outputs, while decoding remains an evaluation and visualization branch.

### 3.4.2 Anchor geometry and causal semantics

Buffer length is not enough to reconstruct historical meaning. Each stored slot must carry, or be deterministically assigned, the frame it represents. For stream $i$, slot $k$ at target boundary $t$ is anchored at

$$
\alpha_{i,k}(t)=t-a_i-k s_i,
\qquad
\lambda_{i,k}(t)=t-\alpha_{i,k}(t),
\qquad k=0,\ldots,n_i-1.
\tag{3.31}
$$

The lag $\lambda$ makes causal checks simple: every context anchor must be positive, while target anchors occupy the current block. Dense streams have stride one. A stride-$s$ coarse stream uses block-end anchors, and its target return contains $W/s$ coarse slots when $s$ divides the four-frame window. Older full-resolution anchors in the temporally sparse package are selected independently of the recent fine tail. Their positions are stored as a declared list, which avoids silently assuming that every historical stream uses stride four.

For the executed `strided-flat` arm, the older fine anchors are $t-47,t-43,t-39,t-35,t-31,t-27,t-23,t-19,t-15,t-11,t-7$, followed by the dense tail $t-6,\ldots,t-1$. The oldest stored observation is therefore at lag 47, while the registry assigns the package to a nominal 48-position span class. Chapter 4 preserves that registered label for continuity, but exact-anchor statements use the enumerated set. The distinction is small numerically and important methodologically: a span name must never override the actual tensor indices.

The same convention affects stride-reduced coarse context. `hier-s2-min` stores $t-47,t-45,\ldots,t-1$: 24 block-end anchors distributed across 48 frame positions, with oldest lag 47. Its purpose-locked protocol nevertheless uses the registered 48-span label and the predeclared $8<\Delta t\le48$ band. That comparison remains a whole-configuration test; it should not be used to infer a one-frame-precise reach threshold at the band's upper edge.

![Exact context-anchor geometry at the first target](../../figures/restructured/ch3_fig3_12_anchor_geometry.png)

**Figure 3.12 — Stream-anchor geometry at a four-frame prediction boundary.** Coarse and fine tokens can refer to the same historical frame, while temporally sparse anchors leave gaps. Controls are associated with target transitions rather than with the context as a single summed label. *Source: dissertation-native schematic based on the package registry and batch constructor.*

Anchor geometry also explains the scope of the B1 stride-2 configuration. Its context uses 24 coarse slots at stride two and two recent fine frames, but the implemented target interface returns two coarse target slots instead of four. Target-token count and the action aggregation for coarse slots therefore change with context stride. The configuration is scientifically useful as a tested package, yet it cannot identify a pure stride effect. A fixed-target version exists as a prepared specification only; because it was not executed, Chapter 4 retains whole-configuration wording.

The generic package definition deliberately does not inherit this implementation shortcut. A public geometry object should carry anchor lists or the four stream fields in (3.1), derive reach from them, and validate that the batch constructor returns exactly those times. This makes dense, stride-2 and sparse-fine packages instances of one contract. It also exposes the current registry helper's stride-four historical convention as a compatibility detail rather than a universal rule.
Within a target block, frames are denoised jointly and can exchange information. Across target blocks, Algorithm 3.1 supplies only committed past predictions. In practical terms, the model is causal at four-frame granularity. It is not frame-autoregressive inside the block. This trade-off lowers the number of outer rollout calls and lets the model coordinate short motion over four adjacent frames, while leaving any within-block order to learned joint attention.

Controls follow the same boundary. Action $u_{t+j}$ denotes the transition into target frame $t+j$; it is not attached to the last context frame as a summary of the whole block. When target slots are subsampled, the configured aggregation must be recorded alongside their anchors. Otherwise, two packages could appear to differ only in memory while actually conditioning their coarse targets on different control summaries.

### 3.4.3 Context, target and total cost

The fixed buffers make retained-state memory independent of rollout duration. For package $\mathcal P$ with $C_\ell=16$ codec coefficients per token and element size $b$ bytes, its contiguous latent payload is

$$
M_{\mathrm{payload}}(\mathcal P)=
bC_\ell B_{\mathrm{ctx}}(\mathcal P).
\tag{3.32}
$$

At float32, one token occupies 64 bytes; at float16 or bfloat16 it occupies 32. The principal 48-coarse plus 6-fine state therefore stores 18,432 latent coefficients, or 72 KiB in float32 and 36 KiB in a 16-bit format. A dense 48-frame fine history stores 49,152 coefficients, or 192 KiB in float32. These are exact array payloads, not measurements of Python containers, allocator behaviour or device peak memory.

Projection into transformer width is another materialization. A width-384 bfloat16 context embedding requires $768B_{\mathrm{ctx}}$ bytes. Targets, attention scores, Q/K/V tensors, MLP intermediates, parameters, optimizer state and training activations remain outside that count. Table 3.3 keeps these layers separate.

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

The equal-reach arithmetic yields two transparent ratios. Relative to 48 dense fine frames, 48 coarse frames plus six fine frames use $3{,}072/1{,}152=2.67$ times fewer context tokens; with a two-frame fine tail the factor is $3{,}072/896=3.43$. Their meaning stops at nominal reach efficiency. Equal information, equal error and proportional wall-clock reduction would each require separate evidence.

Self-attention work depends on total sequence length. Ignoring kernel constants, one block has an attention term proportional to

$$
\mathcal C_{\mathrm{attn}}(S,d)=O(S^2d),
\qquad
\frac{\mathcal C_{\mathrm{attn}}(S_1,d)}
{\mathcal C_{\mathrm{attn}}(S_2,d)}
\approx\left(\frac{S_1}{S_2}\right)^2.
\tag{3.33}
$$

For `flat-long` and `hier-long`, the total lengths are 3,392 and 1,472. Their idealized quadratic ratio is about 5.31, but realized runtime and peak memory depend on kernels, precision, batch size and intermediate storage. No end-to-end speedup is inferred from this expression unless a common profiling receipt exists.

![Exact token and latent-payload arithmetic for equal nominal reach](../../figures/restructured/ch3_fig3_13_token_costs.png)

**Figure 3.13 — Context, payload and total-token arithmetic at equal nominal 48-frame reach.** Hierarchical packages reduce retained context and total sequence length, while the common target interface remains visible. The bars are exact counts derived from tensor shapes, not performance measurements. *Source: derived from the frozen codec and package registry.*

The resource axis here is context/state memory, distinct from the activation memory studied in Chapter 2. During no-gradient sampling, the rollout stores fixed context plus transient work for the current network call. During single-window training, activation memory depends on the arm's declared sequence length—800 to 3,392 tokens among the observed two-level configurations—while remaining independent of the number of rollout blocks evaluated later. Extending training across many self-fed windows would reintroduce activation growth as a separate constraint. Any joint saving would then require direct measurement.

## 3.5 Evaluation instruments and estimands

### 3.5.1 Why short benchmarks cannot identify the reach mechanism

Explicit packages make reach measurable in principle; an ordinary benchmark may still leave the mechanism unidentified. In canonical Moving-MNIST, digits move and collide within a $64\times64$ image, but the generator normally keeps them inside the frame by reflecting their trajectories at the boundary [124]. Future uncertainty and long rollout error are present. The protocol lacks a clean population in which an observed object leaves short context and later returns under geometry that makes longer-history availability computable.

That distinction changes the interpretation of a null result. If recent frames already contain every visible object, a coarse long-memory stream may have no unique evidence to contribute. A model can still be compared for short-horizon competence, perceptual quality or accumulation of numerical error, but the comparison cannot identify whether hierarchy retrieves beyond a flat window. Conversely, a positive aggregate score on such a task might come from optimization, inductive bias or target-interface differences rather than reach.

Standard image metrics introduce a second problem. Sparse foreground allows background pixels to dominate full-frame MSE, PSNR and even structural similarity. A model that misses a digit can score surprisingly well if most pixels remain dark or easy to predict. Foreground-only error corrects that imbalance but ignores hallucinated content outside the ground-truth mask. Distributional metrics such as FID or FVD address other aspects of realism and diversity, yet they do not identify whether a particular re-entry was supported by available historical evidence [94, 129].

Short benchmarks are therefore retained as competence and boundary instruments, not discarded. Canonical Moving-MNIST and TaxiBJ show whether a temporal predictor can operate on recognizable external protocols [1, 124, 126]. Persistent long Moving-MNIST adds stable object identities and a 300-frame self-fed horizon. Neither provides the exact visibility geometry needed for the central mechanism test. That role belongs to the purpose-built LoopMom generator.

Keeping these roles separate avoids two opposite errors. A synthetic mechanism dataset cannot replace evidence on established prediction tasks, while a standard benchmark cannot identify a memory advantage merely because its test sequence is long. Dataset duration, object persistence and the availability of a geometry-derived counterfactual are different properties. Each table and claim therefore names the property it uses.
The instrument was designed around one criterion: it must create records that are hard for a shorter package, answerable by a longer one, and labelable without inspecting model output. Achieving that criterion requires control over the world, viewport and trajectory rather than greater visual realism. The result is intentionally synthetic. Its strength is identification; its limitation is external scope.

### 3.5.2 LoopMom generator and re-entry geometry

Each LoopMom clip begins with a static world $\mathcal W\in[0,1]^{64\times192}$ whose horizontal dimension is toroidal. A smooth background is formed from four sinusoidal components with integer horizontal frequencies, normalized and scaled to the intensity interval $[0.15,0.50]$. Three MNIST glyphs are then placed at fixed world coordinates by pixelwise maximum composition. Horizontal glyph placement wraps around the torus; vertical placement keeps the 28-pixel source patch within the 64-pixel height. The selected world remains unchanged throughout a clip.

The observed frame is a $64\times64$ viewport. Let $r_t\in\{0,\ldots,191\}$ be its saved left-edge coordinate and $a_t\in\{-1,0,+1\}$ the recorded camera action. The executed motion and observation equations are

$$
r_t=(r_{t-1}+4a_t)\bmod192,
\qquad
x_t(y,x)=\mathcal W\!\left(y,(r_t+x)\bmod192\right).
\tag{3.34}
$$

At every frame, 128 of the 192 world columns lie outside the viewport. The action process has momentum: with probability $p_{\mathrm{hold}}=0.1$ the camera does not move; otherwise it retains its current heading with probability $p_{\mathrm{cont}}=0.9$ and reverses with the remaining probability. A non-hold step moves four pixels. The nominal ballistic traversal time is $192/4=48$ frames, while realized disappearance and return times depend on holds and reversals. Saved positions, not the nominal period, determine every observability label.

Momentum is central to the test. An unbiased random walk tends to remain near its recent position over a tractable clip, creating too few full-world returns. Directed segments carry the viewport around the torus, so content can leave the visible crop and later re-enter. Because the world is static in world coordinates and controls are recorded, its earlier appearance remains a valid source of evidence when it returns. The task does not require guessing an unobserved off-screen change.

The generator saves the realized viewport position at every time step rather than asking the evaluator to replay pseudo-random motion. It also saves the action sequence used for conditioning. This removes a fragile dependency on random-number-library versions and makes visibility labels a deterministic function of the dataset artifact. A hash mismatch or missing pose track is consequently a data-integrity failure, not an invitation to approximate observability from rendered frames.

![LoopMom world, viewport and registered reach band](../../figures/fig15_loopmom_observability_schematic.png)

**Figure 3.14 — LoopMom world geometry and analytic reach comparison.** The viewport exposes one third of a static toroidal world at a time. Panel B illustrates the registered pair-derived interval in which evidence is available to the 48-frame hierarchical package but outside a 14-frame flat package. The pose-indexed mosaic is shown only as a privileged reference because it receives exact future pose and raw pixels. *Source: author-generated schematic from the frozen generator and E2 configuration.*

Two dataset layers support development and protected evaluation. The common training set contains 3,000 clips of 100 frames; predictor training uses clip-disjoint training and validation indices. Evaluation sets contain 500 clips of 200 frames and are generated separately for development, untouched confirmation, sampler robustness, codec robustness and explicit-access analysis. Manifests record purpose, generator commit, random seed, source glyph file, configuration and SHA-256 hashes, consistent with dataset-documentation practice [165]. Protected arrays cannot be opened through the evaluation path without the matching purpose and access log.

MNIST source glyphs follow declared split rules, and the source file is part of each manifest. The digit class is useful as controllable texture and identity, but LoopMom does not expose the class label to the predictor. What the predictor receives is the rendered crop, the frozen latent package and camera action. Privileged world coordinates and raw mosaic construction remain evaluator-only information except in the separately labelled explicit-access reference.

The standard rollout uses 100 evaluation clips from a frozen index range. It warms each arm with the first 48 frames and predicts the remaining 152. The clip duration covers several opportunities for departure and re-entry under the nominal loop scale. Because all arms receive the same frames, actions, prediction boundary and sampler noise where applicable, differences can be paired at the clip and horizon level.

![LoopMom-style observability and re-entry storyboard](../../figures/restructured/ch3_fig3_15_reentry_storyboard.png)

**Figure 3.15 — Multi-frame disappearance and re-entry event.** Recent visual continuity is broken during absence. At re-entry, the evaluator can determine whether the target's foreground columns were visible within each candidate package span. *Source: dissertation-native storyboard derived from the LoopMom generator; illustrative clip only.*

LoopMom concentrates the scientific burden on memory access by keeping world content static and motion one-dimensional. This deliberate narrowing excludes dynamic off-screen objects, three-dimensional occlusion, illumination changes and natural camera motion. Exact memory is not logically necessary either: an MNIST prior can still guess a plausible glyph that was never observed. The instrument offers geometric observability, a directly computable property whose value comes from identification rather than realism.

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

The threshold 0.5 separates glyph strokes from the generated background, whose intensity does not exceed 0.5 before quantization. Columns, not pixels, are the weighting unit. A wide stroke and a narrow stroke at the same horizontal location therefore contribute one visibility event. If $F(t)$ is empty, the frame is assigned to an explicit foreground-empty population rather than being interpreted through foreground error.

The common hardness axis uses $m=6$ for every arm. Frames with $\nu_6=0$ are easy relative to the recent window; $0<\nu_6\le0.1$ defines the middle stratum; and $\nu_6>0.1$ marks substantial re-entry. Because the same geometric reference is used throughout, stratification remains independent of each model's actual context length and no arm can redefine its own test population.

Hardness alone leaves earlier observability unresolved. Let $v_t(c)$ be the latest time before $t$ at which world column $c$ was inside the viewport, and let $A_t=\{c\in N_6(t):v_t(c)\text{ exists}\}$. The answerability fraction and time-since-visible label are

$$
\rho_t=\frac{|A_t|}{|N_6(t)|},
\qquad
\Delta t_t=
\operatorname*{median}_{c\in A_t}\bigl[t-v_t(c)\bigr].
\tag{3.36}
$$

The explicit conventions are $\rho_t=0$ when $N_6(t)=\varnothing$ and $\Delta t_t=0$ when $A_t=\varnothing$. A record is called answerable when $|N_6(t)|>0$ and $\rho_t\ge0.5$. Thus an all-recent frame cannot enter a positive reach band, and a genuinely unseen re-entry is reported separately instead of being counted as evidence against a memory package.

For a longer package $\mathcal P_l$ and shorter package $\mathcal P_s$, the mechanism-specific population is

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

Using the median time since visibility makes the record label robust to a few foreground columns with unusually old or recent coverage. It also means $\Delta t_t$ is a summary, not a claim that every novel column shares one age. The accompanying $\rho_t$ threshold guards the summary: at least half of the recent-window-novel columns must have an earlier observation before a frame is called answerable. Both quantities are saved, so alternative descriptive strata can be shown without changing the registered inferential population.

![Pair-specific reach-differential population](../../figures/restructured/ch3_fig3_16_reach_population.png)

**Figure 3.16 — Pair-specific reach-differential population.** The shaded interval contains records for which relevant foreground was seen within the longer package's reach but outside the shorter package's reach. Records beyond both reaches remain useful boundary cases but do not test the proposed reach mechanism. *Source: dissertation-native schematic of the registered geometric estimand.*

Concentration of an advantage in $\mathcal D_{l:s}^{\mathrm{reach}}$ would support the reach interpretation more strongly than an effect spread uniformly across arbitrary horizon bins. The inference stops short of information-theoretic necessity, perfect retrieval or natural-video generalization. Geometry establishes that evidence was available to one package and absent from the other under the defined interface; the neural model must still encode, address and preserve it through self-fed operation.

Three adjacent populations complete the diagnosis. Within both reaches, the comparison mainly probes representation and model use because neither package lacks nominal access. Between reaches, access differs by construction. Beyond both reaches, neither package has the determining observation and performance reflects priors, accidental similarity or other cues. Reporting these groups prevents a strong result in one region from being generalized to the entire rollout.

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

Each version exposes a different failure. Full-frame MSE retains global image error but can be dominated by background. Ground-truth-foreground MSE focuses on missed or distorted target content, yet it does not penalize a false object drawn elsewhere. Background MSE reveals such hallucination outside the target support. Union MSE scores every pixel occupied by either prediction or target and therefore responds to both omission and extra foreground.

Empty cases are part of the estimand. Foreground MSE and masked foreground sharpness are undefined when $|G_\tau|=0$ and are excluded only from populations explicitly labelled non-empty. They never fall back to full-frame values. Background and union errors are set to zero if their denominator is empty. Foreground precision $|P_\tau\cap G_\tau|/|P_\tau|$ and recall $|P_\tau\cap G_\tau|/|G_\tau|$ use value one when their own denominator is empty; F1 uses zero when precision plus recall is zero. The corresponding non-empty indicators remain in every record so that these conventions cannot be mistaken for observed foreground success.

Thresholds follow the generator and evaluation contract. LoopMom uses $\tau=0.5$ because the smooth background occupies lower intensities and digit strokes extend above this level. Connected components smaller than four pixels are discarded as isolated noise. The persistent-object boundary uses threshold 0.3 and a 30-pixel minimum, values frozen with its detector-reliability gates rather than borrowed from LoopMom.

Foreground sharpness is measured as per-image masked Laplacian energy, using the Laplacian focus-measure family [163]. Computing it within each image avoids confusing sample diversity with the sharpness of a single prediction. SSIM and PSNR remain conventional reconstruction diagnostics [127]. They are secondary because a sparse-background sequence can reward an empty or persistent frame. Visual review is also compulsory: decoded prediction, target and last-context panels are saved from the same records used for numerical analysis. The panels diagnose collapse and qualitative mismatch; they are selected by a declared rule and do not replace inference.

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

The endpoint suite is intentionally redundant in the statistical sense: several measures react to related errors. Its purpose is diagnostic completeness, not the creation of many independent claims. Registered families define which endpoints are co-primary and control multiplicity accordingly. Secondary metrics can qualify a verdict, but they cannot be selected after observation to reverse it.

### 3.5.5 Persistent-object and benchmark boundaries

LoopMom gives exact visibility labels, but its foreground components do not by themselves establish persistent glyph identity through collision and long self-fed motion. The external boundary therefore uses a corrected long Moving-MNIST generator with two persistent objects, official MNIST train/test glyph separation, saved object identifiers, source classes, boxes and trajectories. Training clips contain 100 frames; development and protected clips contain 348. After 48 warm-up frames, each model predicts 300 frames. Flat and hierarchical packages both use 1,152 context tokens, and three training seeds are evaluated.

Predicted components are not labelled by left-to-right order, which fails when objects cross. Each detection has an intensity-weighted centroid, bounding box and classifier label. For visible ground-truth object $g$ and detection $d$, assignment uses frame diagonal $D=\sqrt{H^2+W^2}$ and

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

Hungarian assignment minimizes the joint cost, after which pairs above 0.65 are rejected. Misses receive the frame-diagonal trajectory penalty and count as identity failures whenever the missed object has a label. Extras remain explicit false positives, though they are not inserted into the trajectory or identity denominator. When no GT object is visible, coverage is one and trajectory error zero; identity is undefined if no labelled object is visible. Matched-only trajectory and identity are saved for diagnosis, never substituted for the intention-to-score endpoints.

![Persistent-object assignment and identity tracking](../../figures/restructured/ch3_fig3_17_persistent_assignment.png)

**Figure 3.17 — Persistent-object assignment and identity tracking.** One-to-one assignment uses localization and overlap rather than horizontal order. Misses remain failures in the intention-to-score trajectory and identity endpoints. *Source: dissertation-native schematic of the frozen E6 evaluator.*

Canonical OpenSTL Moving-MNIST and TaxiBJ occupy a different role. They are useful tests of short-horizon competence and data-interface transfer, particularly for the temporal model introduced earlier, but they do not contain LoopMom's analytic re-entry labels. Table 3.5 prevents these datasets from being blended into one generalization claim.

**Table 3.5 — Evaluation datasets and their evidentiary roles.**

| Instrument | Context → prediction length | Property under examination | Memory demand | Valid role in this dissertation | Claim it cannot support |
|---|---:|---|---|---|---|
| LoopMom development | 48 → 152 | package design, mechanism development and geometry-derived strata | explicit disappearance and re-entry | hypothesis formation and diagnostic evidence | untouched confirmation |
| LoopMom untouched/protected sets | 48 → 152 | exact reach bands, frontier, sampler/codec robustness or privileged access according to purpose | saved poses and pair-specific observability | controlled mechanism confirmation or named robustness study | natural-video superiority |
| Persistent long Moving-MNIST | 48 → 300 | object support, continuity and identity through long self-fed motion | persistent identities but no LoopMom geometry | external synthetic boundary | replication of the LoopMom reach mechanism |
| Canonical Moving-MNIST | protocol-specific short context/target | motion-prediction competence | objects remain within the frame | benchmark competence and negative control | hierarchy equivalence or re-entry benefit |
| TaxiBJ | standard OpenSTL protocol | spatiotemporal field prediction | no object re-entry identity task | external competence | long-memory object consistency |

Changing the dataset changes the question. The persistent-object study tests whether the reach–fidelity interpretation remains useful when identity becomes explicit, while remaining an external synthetic boundary. Natural-video validation would require moving cameras, dynamic hidden content and controlled re-observation labels, or a benchmark whose annotations permit a comparable estimand.

### 3.5.6 Statistical protocol

Pairing is applied before resampling: the same clip and target horizon appear in both arms, and stochastic samplers use common random numbers for registered paired cells [169]. A lower-is-better endpoint uses comparator minus candidate; a higher-is-better endpoint uses candidate minus comparator, so positive effects consistently favour the registered candidate. Pairing reduces noise while leaving the comparison licence unchanged. A near-matched package remains near-matched, and a whole-configuration intervention remains confounded at component level.

The estimand in (3.40) is record-weighted within the frozen population: clips with more eligible frames contribute more records to the point estimate. Cluster resampling leaves that estimand unchanged; each selected clip contributes its complete eligible record set instead of being reduced to an equal-weighted clip mean. Variable numbers of re-entry frames make this distinction consequential, because the two summaries would otherwise appear interchangeable.

Frames from one autoregressive rollout are strongly dependent. Resampling them as independent observations would exaggerate the effective sample size. Let $d_{ci}$ be the aligned effect for record $i$ in clip $c$, let $n_c$ be the number of retained records in that clip, and let $C_b$ be a bootstrap sample of clip identifiers drawn with replacement. For a registered family $j=1,\ldots,J$, the estimators and simultaneous interval are

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

Each bootstrap replicate draws clips and retains every selected frame belonging to those clips. The primary analyses use 10,000 replicates. Studentization places contrasts with different native scales on a comparable statistic before the maximum absolute value is taken. The resulting critical value provides simultaneous 95% coverage for the frozen family instead of treating many marginal intervals as independent discoveries [120–123].

In (3.40), $\widehat s_j^*$ denotes the standard deviation of contrast $j$ across the complete set of bootstrap replicates; it is not re-estimated inside each replicate. The same scale converts the bootstrap deviations to $T_b^{\max}$ and expands the final interval by the family critical value.

Families are frozen by scientific decision, not by file convenience, consistent with prospective registration as a safeguard against undisclosed analytical flexibility [168]. The untouched frontier, sampler allocation, codec robustness and persistent-object endpoints form separate registered groups because they answer different questions. A max statistic is calculated only across the contrasts belonging to that group. Adding or removing an endpoint after seeing its sign would change the family and invalidate the advertised coverage.

Clip-cluster intervals condition on a trained model. Training seeds form a separate uncertainty level [170]; more records from one fitted model add no replication at that level. Seeds are shown individually or summarized with seeds—not clips—as the sampling units, and opposite signs count as heterogeneity even when their average is near zero. Sampler and codec seeds also represent different interventions: resampling initial diffusion noise cannot establish robustness to retraining, while one alternate codec cannot establish universal representation invariance.

Verdicts follow the sign of the simultaneous interval together with the predeclared endpoint direction. An interval entirely above zero supports the registered candidate for that contrast; an interval entirely below zero supports the comparator; overlap with zero remains unresolved. Seed replication can strengthen or qualify that verdict, but thousands of within-clip frames cannot manufacture training-seed replication. This is why clip counts, record counts and model-seed counts appear as separate quantities in Chapter 4.

Population filters are frozen with the contrast. Reach-band analyses require the pair-specific interval, answerability and non-empty foreground. Persistent-object co-primary endpoints use all visible labelled ground-truth objects through their intention-to-score denominators. Missing values are never intersected across arms in a way that retains only jointly successful frames. Record and clip counts, coverage and empty-population counts accompany every inferential table.

Equivalence is not inferred from an interval that crosses zero. It would require a justified smallest effect of interest in the endpoint's native units and reliable measurement under that margin. No such margin was frozen for the central programme, so unresolved means unresolved. Negative and mixed outcomes remain terminal evidence when their registered decision rules are satisfied; they do not trigger an unplanned search for a more favourable population.

## 3.6 Research hypotheses and chapter conclusions

### 3.6.1 Registered hypotheses and expected directions

The model and instruments above support five hypothesis families. They are stated before the results so that a later observation cannot redefine what the package was expected to do.

**H1 — Reach differential.** At the lower matched budgets, the longer hierarchical package should outperform its shorter flat comparator most strongly within $\mathcal D_{l:s}^{\mathrm{reach}}$. Whole-population support is useful, but concentration in the pair-derived band is the mechanistic prediction. Failure inside that band would weaken the reach account even if an aggregate endpoint happened to favour hierarchy.

**H2 — Fidelity recovery.** As a dense flat package approaches the history span of the hierarchical package, preserving full-resolution evidence should become more valuable. The tested budget points may therefore move from hierarchy-favouring to unresolved and then flat-favouring outcomes. The hypothesis does not require monotonicity, a unique crossing or any conclusion between sampled budgets.

**H3 — Package shape.** Temporally sparse fine-resolution anchors and spatially coarse dense history are not assumed to be interchangeable. Under diffusion, the static-world instrument may favour sparse full-resolution anchors because selected old observations retain glyph detail. The comparison is near-matched where its context counts differ, and its direction must be examined across training seeds and against the declared head.

**H4 — Robustness and representation dependence.** Principal package directions should be checked under alternative sampler allocations and an independently trained codec. Persistence of point-estimate direction supports robustness to the named intervention. It does not establish invariance to all samplers, predictor seeds, codecs or datasets, and an unresolved simultaneous interval lowers precision rather than proving equality.

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

The implemented predictor uses a frozen two-level convolutional codec with 16 coarse tokens and 64 fine tokens per frame. Both levels are independently decodable and standardized with training-only channel statistics. Context and target tokens share one joint-attention sequence, allowing cross-level exchange before either output head is read. Four target frames are estimated together; causality arises across rollout blocks because only completed predictions enter the fixed-size buffers. Predictor gradients stop at the codec and at window boundaries.

Windowed latent diffusion supplies the primary generative interface. Four consecutive coarse/fine targets share a diffusion time, use velocity parameterization and are sampled by deterministic DDIM with declared function-evaluation counts. A capacity-aligned regression head changes the objective and inference rule while retaining the package and backbone scale. Its presence lets the results chapter test whether package use depends on the generative mechanism instead of attributing every outcome to context alone.

Autoregressive operation is a fixed-state recurrence. The standard hierarchy retains 48 coarse and six fine latents, predicts four frames, commits them to both buffers and repeats without preserving a cross-window graph. Forty-eight fine frames require 3,072 context tokens; 48 coarse frames with two or six fine frames require 896 or 1,152. The corresponding 3.43- and 2.67-fold reductions describe nominal reach at the context interface. They do not assert equal information, equal error or a proportional reduction in end-to-end VRAM.

Measurement is equally important. LoopMom creates analytically observable disappearance and re-entry over a static toroidal world. Saved poses make answerability, time since visibility and pair-specific reach bands computable before model output is examined. Full-frame, foreground, background, union, component and persistent-object endpoints expose complementary failures. Resampling uses clips as units and retains their autocorrelated frames; registered contrast families receive simultaneous max-statistic intervals; training seeds remain a separate uncertainty level.

At the systems level, the earlier methods remain complementary: spatial allocation and activation storage are controlled at interfaces outside the present experiment. Chapter 3 isolates context/state memory and supplies the competing packages needed to test it. With interfaces, hypotheses and estimands fixed, Chapter 4 can now answer the empirical question that motivated the design: where does longer hierarchical reach pay, and where does the fidelity retained by flat context become more valuable?
