# Розділ 3 – Ієрархічна контекстна модель довгогоризонтного прогнозування відео

This chapter specifies two distinct implementations of a shared resource question. The controlled retained-context predictor isolates allocation of observed evidence. The OpenSTL-native predictor tests whether real spatial packing and reversible internal computation can coexist with useful prediction. Their codecs, objectives, positional encodings, and training protocols are not interchangeable. Neither implementation inherits the published numerical results of TempVerseFormer or V-BLT merely by using related principles.

## 3.1 Збережений контекст як проблема кінцевої моделі

### 3.1.1 Consistency requires access to relevant evidence

A scene region can leave the viewport before it leaves the world. When it returns, a predictor must either retrieve an earlier observation or reconstruct its appearance from a lossy state. A short dense window may contain excellent recent detail but none of the relevant observation. A hierarchy changes what evidence remains available; it does not ensure that the predictor uses that evidence correctly.

{{LEGACY_FIGURE:3.1}}

This mechanism distinguishes three axes. **Context age/reach** measures how far before the prediction origin an observed state is retained. **Prediction endpoint** is the four jointly predicted frames in the registered allocation study. **Autoregressive rollout horizon** counts predicted frames subsequently fed into later model calls. A long first axis does not establish stability on the third.

### 3.1.2 Context package, payload, and reach

A context package is an explicit set of physical identities rather than a dense tensor with entries set to zero. For a target beginning at frame index $s$, age $a=1$ denotes frame $s-1$. An identity contains a representation level, positive age, and spatial cell:

$$
\mathcal P=\{(\ell,a,j)\},\qquad t(\ell,a,j)=s-a<s.
\tag{3.P1}
$$

This exact identity contract is implemented by `AllocationPlan` and `PackedWindowBatcher`. The experiment fixes the package before seeing target pixels. Cell identities remain physical grid coordinates after gathering.

$$
N_{\mathrm{ctx}}=4|A_{\mathrm{top}}|+16|A_c|+|A_f|,
\qquad R(\mathcal P)=\max_{(\ell,a,j)\in\mathcal P}a.
\tag{3.P2}
$$

Here $A_f$ is a set of age–cell pairs, while $A_c$ and $A_{\mathrm{top}}$ contain complete coarse and top-level time positions. This is exact token accounting for the registered 2×2/4×4/8×8 grids, not a fitted memory law. Nominal reach $R$ says that some representation of an old frame is available; it does not imply full fine coverage of that frame.

![Which independent quantities describe a context package?](../../../figures/dissertation_v2/context_design_axes.svg)

Figure D3-design — Retained token count, oldest available age, and spatial detail are separate design coordinates. This is a non-empirical diagram; no curve implies a measured frontier.

### 3.1.3 The reach–fidelity trade-off

At a fixed payload, older coarse observations displace some recent or distributed fine detail. A possible benefit is access to a re-entering region; a possible cost is loss of identity or texture. The trade-off is explanatory, not an additive statistical decomposition of risk. The study therefore measures frame error, foreground behavior, and task-specific re-entry quantities separately. It does not estimate an unobservable “reach term” by subtraction from total error.

## 3.2 Ієрархічний прихований стан

### 3.2.1 Frozen codec and trainable predictor

The controlled study uses the frozen two-level convolutional codec and its hash-locked latent normalization. For 64×64 TRB frames it produces a 4×4 coarse grid and an 8×8 fine grid. The optional third-level construction adds a 2×2 stream. Codec reconstruction is a boundary condition shared across the paired predictor comparisons, not a learned response to each allocation policy.

$$
(z^c_t,z^f_t)=E_{\mathrm{frozen}}(x_t),\qquad
u_i=W_{\ell_i}\,\mathrm{norm}_{\ell_i}(z_{s-a_i,j_i})+e_i.
\tag{3.P3}
$$

The first relation defines cached latent observations; the second defines a trainable stream projection plus metadata embedding. Channel normalization is fitted on the training split, not recomputed on selected tokens or on holdout data. Exact channel counts, normalization hashes, stream shapes, and the source checkpoint are preserved in Appendix В's generated configuration registry.

![Where is the frozen boundary in the controlled predictor?](../../../figures/dissertation_v2/windowdit_contract.svg)

Figure D3-windowdit — Observed latent candidates are gathered on the host into the declared package before transfer to the shared predictor. Frozen codec/normalization and trainable projections/WindowDiT are distinct. Four future coarse/fine grids are denoised jointly; no future clean latent enters observed context. Exhaustive dimensions are in the adjacent appendix contract, not encoded as extra schematic panels.

### 3.2.2 Allocation identities and metadata

Flat retains complete 8×8 fine grids at the most recent ages. For the Linear-age and Geometric-age frontier arms, $B/32$ coarse positions consume half of the retained budget; the remaining half is fine detail distributed over those ages. Linear weights descend with age rank; geometric weights are $0.94^{a-1}$. Integer counts use capped largest-remainder allocation with a maximum of 64 cells per age and deterministic tie handling. The exact cell order is a farthest-point grid order, shifted by $(3a,5a)$ modulo eight. It is content independent and must not be described as a learned router.

At 1,792 tokens, Fine-heavy uses 28 coarse and 21 complete fine positions; Coarse-heavy uses 84 and seven; Reach-optimized uses 96 and four. Balanced recent uses 56 coarse positions and fills 896 fine tokens from the newest ages. Linear-age and Geometric-age use the same 56-position coarse support and fine total with different temporal distributions. Flat uses 28 complete fine positions. The labels name executable constructions, not qualitative judgments about their eventual quality.

Nested ownership allows coarse and fine representations at the same age. Exclusive ownership preserves declared payloads while relocating coarse ages away from fine support, if this fits inside the 96-frame history. Residual ownership changes the fine representation to a coarse-embedding residual. Thus nested/exclusive/residual are different representation packages; identical token counts do not make them a pure overlap intervention.

### 3.2.3 Joint attention and causal information flow

The predictor concatenates context and noisy target streams for joint attention. Target tokens may attend to other target tokens because the output is one jointly denoised future window, not a sequence of four causally generated frames. Context frames all precede the first target, and clean target latents are used only in the training objective. Actions supplied for TRB belong to its declared camera-motion conditioning contract; this extra information is not available in every video benchmark.

$$
\widehat v=F_\theta(U_{\mathcal P},z_\tau^{\mathrm{target}},\tau,a_{s:s+3}).
\tag{3.P4}
$$

This conditional predictor is implemented by the campaign's WindowDiT forward call. It defines the available inputs, not an assertion that attention weights establish causal use of distant evidence. Context-use interventions belong to the separate mechanism diagnostic.

### 3.2.4 Gradient boundary and information asymmetry

Cached observations and the codec are frozen. The trainable predictor receives a genuinely packed set of selected fine cells, their physical spatial indices, and chronological metadata. The target grids remain dense. Reducing context tokens therefore does not reduce all target, decoder, optimizer, or workspace costs. Training does not differentiate through a chain of generated future windows in the registered v2 matrix.

## 3.3 Ядро генеративного прогнозування

### 3.3.1 Executed latent diffusion objective

For the v-prediction parameterization, a clean target $z_0$ and independent noise $\epsilon$ define the noisy target and velocity target:

$$
z_\tau=\alpha_\tau z_0+\sigma_\tau\epsilon,\qquad
v_\tau=\alpha_\tau\epsilon-\sigma_\tau z_0,\qquad
\mathcal L=\mathbb E\sum_{\ell\in\{c,f\}}\operatorname{mean}_{\ell}\left(\widehat v_{\theta,\ell}-v_{\tau,\ell}\right)^2.
\tag{3.P5}
$$

Here $\widehat v_\theta=F_\theta(U_{\mathcal P},z_\tau,\tau,a)$. The loss sums coarse and fine per-stream MSE means, rather than using one unnormalized norm of concatenated targets; the optional top stream is context only. This is the implemented diffusion training target, not the primary model-selection metric. Selection uses decoded horizon-4 foreground MSE. Standard diffusion and Transformer construction are covered by [2–4, 12]; the earlier expanded derivation is retained in Appendix Е.

### 3.3.2 Shared Transformer backbone

The registered main runs share width 256, six blocks, and eight heads. They use a batch of 16, AdamW with learning rate 0.0003 and weight decay 0.0001, 2,000 warmup updates, EMA decay 0.999, and bfloat16 autocast with fp32 master weights. The mixed-precision normalization path is part of the pinned execution contract. These settings describe the current campaign, not universal recommended hyperparameters. The three-level extension adds a projection/stream interface and 1,280 parameters; its level comparison therefore reports this small but real capacity difference.

### 3.3.3 Sampling and bounded regression control

Validation uses DDIM with 16 function evaluations and common registered random-number handling. An inference sampling budget is different from the 30,000 optimizer-update budget. Regression and EBT controls belong to the historical predictor-mechanism study; their results justify retaining a bounded mechanism comparison, not changing the head midway through the new allocation campaign. Standard DDIM algebra and the regression objective are preserved in Appendix Е rather than repeatedly numbered here.

## 3.4 Авторегресивне розгортання та оновлення пам’яті

### 3.4.1 Four-frame update and ring buffers

In a self-fed evaluation, a model predicts a four-frame block, re-encodes or retains its predicted representation according to the evaluator, advances the origin by four, and selects the next package from the updated causal history:

$$
\widehat X_{s:s+3}=\mathrm{Predict}(\mathcal P_s),\qquad
S_{s+4}=\mathrm{Update}(S_s,\widehat X_{s:s+3}),\quad
\mathcal P_{s+4}=\mathrm{Gather}(S_{s+4}).
\tag{3.P6}
$$

This is the legacy/optional rollout interface. It is **not executed repeatedly by the 86-run v2 allocation endpoint**. The v2 development clips have length 100 and target start 96; their only target window is frames 96–99. Fresh confirmation clips have length 200 and target frames 196–199. All earlier context for those endpoint evaluations is observed, not generated.

![When does predicted content enter future context?](../../../figures/dissertation_v2/context_rollout_timeline.svg)

Figure D3-rollout — A single four-frame endpoint ends after one prediction. A self-fed rollout instead updates a causal buffer and makes another call. The lower path is legacy or separately authorized additional evidence; it is not another view of the v2 endpoint.

### 3.4.2 Anchors, causality, and retained cost

Ring-buffer storage order must never substitute for absolute/relative time identity. On every update the evaluator checks that all selected ages precede the new origin, that every requested age exists, and that target content is not copied into context before it has been predicted. Gradients are stopped between calls in the bounded rollout evaluation; this is not end-to-end BPTT across the full horizon.

$$
N_{\mathrm{attention}}=N_{\mathrm{ctx}}+N_{\mathrm{target}},\qquad
M_{\mathrm{payload}}=\sum_i \mathrm{numel}(s_i)\,\mathrm{bytes}(s_i).
\tag{3.P7}
$$

These exact counts cover the specified tensors only. Metadata, dense output grids, temporary encoder candidates, and allocator overhead must be included separately when measuring end-to-end memory. Attention work can change even when one named context component is fixed.

## 3.5 Інструменти оцінювання та оцінювані ефекти

### 3.5.1 Toroidal Re-entry Benchmark and observability

The public name is **Toroidal Re-entry Benchmark (TRB)**. `LoopMom` is retained only as a historical dataset/code identifier. A camera viewport moves over a static periodic world, making disappearance and re-entry exactly observable. This controlled environment tests memory access without claiming the semantic diversity or non-rigid motion of natural video.

{{LEGACY_FIGURE:3.14}}

For a world point $q$ and camera pose $p_t$, the observation is a viewport rendering of the periodic world. Visibility and package support are computed from geometry rather than from a model's prediction:

$$
x_t=\mathrm{Viewport}(W,p_t),\qquad
\mathcal V_{\mathcal P}(q,s)=\mathbf1\{q\text{ was visible at a retained age in }\mathcal P\}.
\tag{3.P8}
$$

This relation defines potential access, not retained identity accuracy. The exact wrap-around convention, occlusion cases, data hashes, and visibility implementation remain in the dataset/evaluation registry and the archived original definitions.

### 3.5.2 Historical pair-specific populations and the v2 confirmation frame mask

In historical E2, the reach-difference population is defined for a declared pair before inspecting their errors:

$$
\mathcal R_{H,F}=\{(q,s):\mathcal V_H(q,s)=1,\ \mathcal V_F(q,s)=0,\ q\text{ satisfies the registered re-entry rule}\}.
\tag{3.P9}
$$

Changing a package can change this historical population. Pair-specific effects therefore cannot be pooled as if every comparison evaluated identical pixels. The historical visibility figure below explains that distinction; it does not describe the v2 finalizer's mask.

{{LEGACY_FIGURE:3.16}}

The current v2 finalizer instead uses a fixed **96-frame/28-frame frame-level diagnostic** for every selected model. For each of four target horizons, it flags a frame if any currently visible sprite was visible somewhere in the preceding 96 observed frames and its last such observation occurred before the start of the most recent 28-frame reference window. Visibility uses overlap of sprite and viewport world-column supports, not predicted foreground pixels. This is one common clip/horizon mask, not package-specific fine/coarse coverage and not the world-point set in equation (3.P9).

The finalizer reports `mean_foreground_mse` averaged across all four horizons, `reentry_foreground_mse` averaged over flagged frames, and the non-re-entry counterpart. It saves per-clip/per-horizon arrays for explicit later analysis. These summary endpoints are **not** the development-selection `horizon4_foreground_mse`. A horizon-4 or package-specific confirmation analysis would require its own declared postprocessing contract; it is not silently claimed by this chapter. An empty re-entry set yields a missing result, not a zero-error observation.

### 3.5.3 Endpoints and object identity

The primary development statistic is the trainer's `horizon4_foreground_mse`: decoded error at the fourth predicted frame where the target intensity exceeds 0.5. Within each fixed validation batch it is foreground-pixel weighted; those batch values are then weighted by batch size. With four batches of 16 samples, this is the mean of four batch-level masked errors, not the mean of 64 independently normalized clip errors:

$$
m_{\mathrm{dev},4}=\frac{\sum_b n_b\left[\sum_{i,p\in b}M_{i,4,p}e_{i,4,p}/\max(1,\sum_{i,p\in b}M_{i,4,p})\right]}{\sum_b n_b},\qquad
M=\mathbf1\{y>0.5\}.
\tag{3.P11}
$$

Here $e$ is squared decoded pixel error, $n_b$ is batch size, and only horizon four enters the development criterion. The confirmation finalizer instead computes a masked mean separately for each clip/frame and then averages those per-frame errors over all four horizons or the flagged re-entry frames. Its weights and horizon population therefore differ. Pixel MSE, other horizons, SSIM, PSNR, diffusion loss, and latent loss remain secondary and cannot silently replace the development criterion. A foreground-conditioned error is not an object-identity score.

Historical E6 additionally uses persistent ground-truth object identities, a frozen one-to-one Hungarian assignment, object coverage, trajectory error, union-mask error, and identity accuracy. These custom definitions and edge cases remain in Appendix Д with the original implementation references. Coverage and identity must be reported independently: producing a plausible object in approximately the correct region can still be an identity failure.

### 3.5.4 Seeds, uncertainty, and selection

For each declared contrast let $d_s=m_{F,s}-m_{H,s}$ when lower metric values are better. The two-seed summary is:

$$
\bar d=(d_0+d_1)/2,\qquad \text{report }(d_0,d_1)\text{ as well as }\bar d.
\tag{3.P10}
$$

The mean is descriptive. Two training seeds do not characterize a broad seed distribution and do not justify a training-seed t interval. “Direction reproduced across two training seeds” is allowed only if both signs agree. Clip-clustered resampling preserves within-clip dependence; its interval is conditional on the trained checkpoint and is never relabelled as uncertainty over training runs. Historical multi-comparison intervals retain their original family and cannot be pooled with v2 results.

## 3.6 Дослідницькі гіпотези основного контекстного дослідження

### 3.6.1 Common protocol and 30k calibration

Every primary training cell uses 30,000 updates, seeds 0 and 1, and validation every 2,000 updates. Final-step development validation is primary; per-arm early stopping is not substituted after inspecting the curves. Two separate seed-0 calibration arms—Flat and Balanced recent at 1,792 tokens—were trained to 30k. They are two configurations, not two independent training seeds. The registered analysis tested a 2% remaining-improvement criterion together with stability of the ordering. Candidates from 8k through 24k did not meet the joint rule; ordering changed between 12k and 16k. “Ranking reversal” names that observed crossing, not an operation in the model code. The accepted 30k endpoint is bounded by the observed calibration range; it does not prove convergence beyond 30k or for every configuration.

### 3.6.2 Six-point context-budget frontier (A4-FRT)

The budgets are 320, 768, 1,152, 1,792, 2,432, and 3,072 retained tokens. At each budget Flat, Linear-age, and Geometric-age are trained with both seeds: 36 runs. Balanced recent is not a frontier arm. The question is how quality changes over an approximately tenfold token range. It is not a fixed-total-VRAM experiment because target and optimizer costs remain and attention sequence length changes. Plot actual numerical budget spacing rather than equally spaced categories.

### 3.6.3 Allocation at 1,792 tokens (A4-ALC)

Seven policies—Flat, Fine-heavy, Balanced recent, Coarse-heavy, Linear-age, Geometric-age, and Reach-optimized—share the budget. Three two-seed policy pairs reuse frontier checkpoints; four new pairs add eight runs. The question is which registered distribution is best on development validation, not which hand-selected visual package looks most plausible. Reused cells retain one identity in the registry and are not counted as independent replication.

### 3.6.4 Coarse-context expansion (A4-CEX)

Six recent full fine positions remain fixed while 0, 4, 8, 16, 32, or 64 exclusive older coarse positions are added. The resulting budgets are 384, 448, 512, 640, 896, and 1,408 tokens. Both seeds give 12 runs. This estimates the response to a compound increase in older coarse support, total budget, attention work, and reach. It does not isolate the causal effect of reach at fixed memory.

### 3.6.5 One, two, or three levels (A4-LVL)

At budget 1,792, the one-level Flat and selected two-level hierarchy are reused. Selection is restricted to eligible hierarchical ALC constructions for the dependency; Flat is a comparator, not an input to the three-level builder. The new three-level pair adds four top tokens at each of 96 ages, leaving 1,408 tokens allocated by the selected coarse:fine ratio with deterministic 16-token rounding. The new top stream and projection add 1,280 parameters. Thus this is a comparison of registered constructions at fixed payload, not a pure scalar intervention on level count. Only two new training runs are required.

### 3.6.6 Ownership/representation packages (A4-OVR)

Nested, exclusive, and residual packages use budget 1,792 and both seeds. One pair is reused; four runs are new. Exclusive support must fit inside the finite history and changes coarse ages. Residual fine embeddings change representation semantics. Report feasibility and the dependency-selected base policy. A result is attributed to the complete registered package, not to “overlap alone.”

### 3.6.7 Space–time detail allocation (A4-STS)

At budget 1,792 only, fine quotas of 64, 32, 16, and eight cells per selected fine time are crossed with recent, uniform, and geometric temporal placement. This gives 12 configurations and 24 runs. The exact number of fine times and coarse slots is generated under the fixed total payload. Spatial membership remains deterministic and content independent. The study tests how spatial savings can be reassigned to temporal coverage; it is not yet the integrated content-adaptive router.

### 3.6.8 Selection, sealed confirmation, and mechanism diagnostics

The programme contains 43 unique configurations and 86 training runs: 80 concrete first-wave runs and six dependency-resolved second-wave runs. The complete development matrix is the **registered development estimate**. After all final records pass provenance checks, a hash-bound selection ledger is written before opening a fresh 500-clip confirmation set. Only the exact selected roles receive **sealed-holdout confirmation** status. The full development frontier and the ALC/LVL/OVR/STS families are not thereby confirmed wholesale.

The current bounded A4-MEC output records target-query attention mass/entropy for one fixed clip (index 0) and a direct denoiser probe at diffusion time 0.5. The probe uses target latent inputs and does not reproduce a full noisy sampling trajectory or a causal context-ablation experiment. It is a **mechanism diagnostic**, not an independent quality claim or proof of useful old-context retrieval. Appendix В carries the full identifiers, resolved dependencies, hashes, and invariants. No new experiment is added by editing this chapter.

### 3.6.9 Selected-checkpoint autoregressive rollout: optional additional evidence

A separate inference-only study may evaluate the selected Flat and hierarchical checkpoints for both seeds on a newly generated, hash-locked long TRB set. It requires prior authorization and registration of warmup, horizon bands, clip count, sampling/NFE, common random numbers, re-entry eligibility, failure handling, and the exact causal update path. No checkpoint is retrained or reselected. A timing preflight precedes any runtime estimate. Until that study exists, legacy E5/E6 carry the long-rollout evidence, and the 86-run matrix remains a four-frame endpoint study.

## 3.7 Контрольоване інтеграційне розширення ієрархічної моделі

### 3.7.1 Composition hypotheses and the OpenSTL interface

The implemented OpenSTL-native model maps $(B,T_{in},C,H,W)$ to $(B,T_{out},C,H,W)$. It is trained end to end and does not reuse the controlled study's frozen codec or diffusion objective. The external protocols are Moving MNIST 10-to-10, one channel at 64×64, and TaxiBJ 4-to-4, two channels at 32×32 [1, 178]. Four long full-model runs are planned—two datasets, seeds 0 and 1—conditional on correctness, resource, quality, and authorization gates. Those runs alone cannot establish every component-quality ablation or a matched-baseline frontier.

The superseded fixed-region/packed-SimVP prototype is archived in Appendix Е. Fixed slots did not solve changing region identity; an early dynamic version still materialized dense fine state; repeated decode/re-encode was lossy; and second-order reversible rollout changed the predictive dynamics. The current model instead makes internal blocks reversible while retaining the same forward predictor in both execution modes.

### 3.7.2 Shared pyramid and genuinely packed state

A convolutional encoder generates stride-four fine candidates and stride-eight coarse coverage for every observed frame. For Moving MNIST these grids contain 256 and 64 cells per frame; for TaxiBJ, 64 and 16. The hierarchy gathers a global quota $K$ of fine tokens across the observed space–time volume before the Transformer:

$$
U=\operatorname{concat}(Z_c,\operatorname{gather}(Z_f,I_K)),\qquad
N_{\mathrm{packed}}=T_{in}(H/8)(W/8)+K.
\tag{3.I1}
$$

The exact retained core input has shape $(B,N_{\mathrm{packed}},D)$. This is a real gather, not a dense padded tensor masked inside attention. Dense encoder candidates still exist transiently and may contribute to the training graph. The decoder also keeps coarse-grid information. Consequently the compact payload counter is not the complete live model state or total GPU peak.

![How are packing, physical position, and reversible computation connected?](../../../figures/dissertation_v2/integrated_architecture.svg)

Figure D3-integrated — Encoder and router form an ordinary-autograd boundary; gathered tokens plus physical metadata enter the reversible core; future queries and the full-frame decoder form another ordinary boundary. Reversibility does not invert top-k selection or reconstruct discarded fine candidates.

### 3.7.3 Routing policies and route-aware training

The implemented density score standardizes fine–upsampled-coarse squared residual and adjacent-frame absolute change within each sample. The hybrid score adds a standardized learned scalar correction:

$$
r_i=\langle(z^f_i-\operatorname{up}(z^c)_i)^2\rangle_D,\quad
\delta_i=\langle|z^f_i-z^f_{\mathrm{previous}(i)}|\rangle_D,\quad
q_i=w_r\operatorname{stdz}(r)_i+w_\delta\operatorname{stdz}(\delta)_i+w_g\operatorname{stdz}(g_\theta(z^f))_i.
\tag{3.I2}
$$

Standardization uses population standard deviation with a numerical floor of $10^{-6}$. Change is zero for the first observed frame. There is no separate predictive-entropy input or V-BLT module in this score. The word “content” refers to the features from which these scores are computed.

Dynamic membership first reserves up to `min_detail_per_frame` cells in each frame, bounded by the quota, then fills the remainder by score. Selected indices are sorted to give a canonical output order. Equal-score top-k membership is not guaranteed stable across devices; sorting selected indices is not a tie-breaking proof. Static membership uses evenly spaced midpoints over the flattened candidate volume.

`static`, `density`, and `hybrid` are inference membership policies. `mixed` is a training exposure regime that samples static versus hybrid batches; evaluation resolves it to hybrid. A route-aware checkpoint may be evaluated under all three policies as a switch/distribution-shift diagnostic. Comparing static-trained and mixed-trained checkpoints requires separate training and cannot be inferred from a switch.

Top-k membership is discrete. In dynamic modes a learned embedding of selected scores is added to selected features, providing a gradient path through selected scores; it does not differentiate the membership decision. Because switching modes also changes this score-feature addition, its effect is not attributable to membership alone. Route entropy, temporal/spatial coverage, frame counts, and adjacent-frame membership churn are diagnostic endpoints. The current score margin uses an unconstrained sorted-score boundary; with per-frame reservations it is not the exact margin of constrained selection. Route stability across forecast calls and seam/flicker behavior require separate evaluation.

### 3.7.4 Physical position and hierarchy metadata

Packed slot number has no geometric meaning. For a cell at row $r$, column $c$ in a grid of height $h$ and width $w$, the top-left-origin convention uses $x=(c+1/2)/w$ and $y=(r+1/2)/h$, both strictly inside $(0,1)$. Scale is $(1/w,1/h)$, not the rank of a selected region. The last observed frame has time zero; older frames have negative time and future decoder queries have positive time, all divided by $\max(T_{in},T_{out},1)$. Level indices are coarse 0, fine 1, and future-query 2.

$$
p_i=W_p[\sin(\pi2^k\xi_i),\cos(\pi2^k\xi_i)]_{k=0}^{K_p-1}+e_{\ell_i},\quad
\xi_i=(x_i,y_i,t_i,s_{x,i},s_{y,i}).
\tag{3.I3}
$$

This is the implemented continuous Fourier encoding, **not RoPE-ND**. Normalization is determined by the physical grid and configured protocol, never recomputed from selected tokens. Changing resolution preserves normalized cell centres but changes scale. Changing horizons changes the time denominator and is therefore a protocol/distribution shift, not automatically validated extrapolation. The public forward method enforces the configured input shape; a different protocol needs a matching configuration and verification.

![Why can membership change without changing spatial identity?](../../../figures/dissertation_v2/physical_position_contract.svg)

Figure D3-position — Features and their x/y/time/scale/level metadata move together when packed order changes. The fp32 CPU decoder-permutation test uses absolute and relative tolerances of $3\times10^{-6}$. A deployed CUDA mixed-precision permutation tolerance is pending measurement, not inherited from this fp32 test.

### 3.7.5 Identical-forward reversible core and future decoder

Each core block splits feature channels and applies additive coupling:

$$
\begin{aligned}
y_1=x_1+F(x_2),&\quad y_2=x_2+G(y_1),\\
x_2=y_2-G(y_1),&\quad x_1=y_1-F(x_2).
\end{aligned}
\tag{3.I4}
$$

$F$ is normalized self-attention and $G$ is a normalized feed-forward module. The inverse is exact in real arithmetic; numerical reconstruction error is measured. Dropout is disabled in the reversible core. Ordinary autograd and reverse-reconstruction execution evaluate this same forward function. Routing, encoder, decoder, optimizer state, and dense target loss stay outside the reversible boundary.

Future queries combine the last observed coarse feature grid with future physical coordinates. Cross-attention retrieves the processed packed memory; convolutional upsampling produces a full-resolution residual added to the last input frame. This path predicts the configured output window directly. Internal depth reversibility is not a claim of a reversible temporal world state.

### 3.7.6 Verification gates and limits of current evidence

CPU tests cover true quota/coverage, route response, metadata permutation, fp32 decoder permutation, inverse reconstruction, core forward and parameter/input-gradient parity, bfloat16 CPU-autocast core parity, model shapes, checkpoint round trip, and an OpenSTL training/validation/optimizer smoke path. The whole-model static test checks output/input-gradient closeness and gradient connectivity; it does not itself compare every whole-model parameter gradient. That stronger dynamic/full-model gate remains explicit in Appendix Ж.

Recorded core fp32 tolerances are $2\times10^{-6}$ for outputs; input gradients use absolute $2\times10^{-5}$ and relative $2\times10^{-4}$; parameter gradients use $3\times10^{-5}$ and $3\times10^{-4}$. The double-precision inverse test uses $10^{-10}$. CPU bfloat16 parity has looser, separately specified tolerances; it does not certify CUDA bfloat16 behavior. No CPU test demonstrates competitive OpenSTL quality or a total-VRAM saving. Those result cells remain gated in Chapter 4.

## 3.8 Висновки до розділу

The controlled method fixes a codec, head, endpoint, and training schedule while varying exact context packages. Its primary question is the cost and utility of retaining distant observed evidence. The integrated method instead composes real spatial packing, physical metadata, route-aware exposure, and an identical-forward reversible core in an OpenSTL-native predictor. Chapter 4 separates development allocation evidence, selected confirmation, historical self-fed boundaries, correctness gates, component profiles, and matched external prediction results. This separation permits a useful negative result without changing the question after evaluation.
