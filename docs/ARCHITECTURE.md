# Architecture — VQ-free hierarchical single-step video predictor

This document is the comprehensive technical reference for `hvp`. It covers every component: the
continuous causal encoder/decoder, the latent hierarchy, the single-step cross-attention cascade, the
training objective and anti-collapse machinery, the metrics, and the design rationale. For the why and
the evidence trail see [PLAN_VQFREE_HIER_PREDICTOR.md](PLAN_VQFREE_HIER_PREDICTOR.md) and
[RESULTS_V1.md](RESULTS_V1.md).

---

## 0. Design principles

1. **No vector quantization.** Predict in a *continuous* latent. Rotation moves content smoothly, so an
   L2 metric is meaningful; the discrete composite-index head (a dead end in predecessor repos) is gone.
2. **Causal temporal downsampling.** A causal 3D CNN compresses `9 → 5 → 3` token-frames across stages
   while keeping the spatial pyramid `16 → 4 → 1`. Causality = left-only time padding `(K-1, 0)`.
3. **Single-step, not recursive.** Each stage predicts the next frame (teacher-forced `t→t+1`) in one
   forward pass; no envelope/rollout during training.
4. **Detached one-shot.** Encoder + decoder + cascade train in one run. The cascade predicts the
   encoder's **detached** latents, so the predictive loss never collapses the representation; the encoder
   is shaped only by reconstruction (+ a VICReg variance floor).
5. **Cross-attention hierarchy.** The coarse stage's hidden features condition the finer stage via
   causal **receptive-field-overlap** windows — the hierarchy pillar, and the part that earns the
   "hierarchical" name.
6. **Single source of truth.** `TopologyConfig` defines the downsample schedule; every per-stage spatial
   and temporal size is *derived*, so encoder, decoder, predictor, and masks never disagree.

The default operating resolution is **64×64**, `T=9`, three stages: **BOT 16²@T9 / MID 4²@T5 / TOP 1²@T3**.

---

## 1. Data

Synthetic rotating-shape clips rendered with the `simple-shape-dataset-toolbox` (PyVista). 3D polycubes
(2–6 cubes) rotating about the **Z axis only** (a deliberate v1 scope-lock — the apparent motion is a
1-D constant-angular-velocity signal; SO(3) is a later milestone). Each clip is `(C=3, T=9, H=64, W=64)`,
normalized to `[-1, 1]`. `T ≡ 1 (mod 4)` is required by the causal `9→5→3` chain.

`hvp/data/shape_video.py` reads pre-rendered `.npy` clips from a shared cache keyed by a config hash.
Because `mode` is part of the hash, the **train and val caches are rendered independently** → a true
cross-split (not an index split of one render). `scripts/gen_data_64.py` produces both.

---

## 2. Causal primitives (`hvp/nn/conv3d.py`)

### `CausalConv3d`
A `Conv3d` with **asymmetric left-only temporal padding** `t_pad = K-1` (and symmetric spatial padding).
Frame `t` therefore sees only frames `≤ t`. A temporal stride-2 maps `T → (T-1)//2 + 1` (e.g. `9→5→3`),
the `1 + s·t` convention shared with MAGVIT-v2 / Cosmos.

### `FrameNorm3d` (the causality-preserving norm)
**Critical subtlety:** a plain `nn.GroupNorm` over a 5-D `(B,C,T,H,W)` tensor normalizes over **T as
well**, which mixes information across time and *silently breaks causality* (perturbing the last frame
shifts every frame's statistics). `FrameNorm3d` reshapes `(B,C,T,H,W) → (B·T, C, H, W)`, applies
`GroupNorm2d` **per frame**, and reshapes back — statistics never cross the time axis. Autocast keeps
GroupNorm in fp32, so it is also AMP-stable (the WF-VAE concern). This is verified by
`test_encoder_bot_stage_is_causal`.

### `Upsampler3d`
Inverts the causal downsample: a conv expands channels by `∏ block_size`, `depth_to_space3d` redistributes
them over `(T,H,W)`, and for a temporal block `>1` the first `s-1` frames are dropped so `T → 2T-1`
(`3→5→9` on decode). Verified by the frame-count round-trip test.

---

## 3. Encoder (`hvp/tokenizer/encoder.py`)

A bottom-up causal 3D-CNN pyramid. From the 64×64 input it applies `ResBlock3d`s and stride-2
`CausalConv3d` downsamples following `ch_mult = (1,2,2,2,2,4,4)` (7 levels → 6 downsamples) and
`temporal_downsample = (F,F,T,F,T,F)`. Three **taps** are read off (`tap_after = (1,3,5)`), one per stage,
and each passes through a thin **latent head** (`FrameNorm → swish → 1×1 CausalConv`) to a continuous
latent:

| stage | tap (after ds #) | spatial | temporal | latent ch |
|---|---|---|---|---|
| BOT | 1 | 16×16 | 9 | 16 |
| MID | 3 | 4×4 | 5 | 8 |
| TOP | 5 | 1×1 | 3 | 8 |

There is **no quantizer** — the latent is the continuous head output. Latents are returned fine→coarse
`[z_bot, z_mid, z_top]`.

---

## 4. Decoder (`hvp/tokenizer/decoder.py`) + deep supervision

A **top-down fuse**: project the coarsest latent to `decode_width`, then repeatedly upsample (causal
temporal+spatial `Upsampler3d`) and add the next finer latent, down to the BOT grid, then render to
pixels (a final spatial upsample + ResBlock + `1×1 CausalConv` to RGB). Upsample block sizes are derived
from the per-stage size ratios (`top→mid` and `mid→bot` are `(2,4,4)`; `bot→pixels` is `(1,4,4)` at 64²).

**Deep supervision + coarse-first curriculum.** The decoder renders at each *depth*: depth-1 (TOP only),
depth-2 (TOP+MID), depth-3 (all), each at full resolution. The recon loss is the (weighted) mean over
active depths, and a curriculum ramps the active depth `1→2→3` (`active_depth_at`). This grounds **every**
stage in reconstruction — removing a stage hurts its depth's recon — which is what keeps the coarse
latents alive (high effective rank) and meaningful, a prerequisite for the predictor.

---

## 5. Attention stack (`hvp/attention/`)

- **`Rotary3D` — factorized 3D RoPE.** The head dim splits into three even chunks `(d_t, d_h, d_w)`
  (time-heavy); each chunk is rotated by its own axis position so attention sees **relative** `(t,h,w)`
  offsets — the motion-modeling lever (content can move across cells). cos/sin are built in fp32 and
  cached per `(T,H,W)`; rotation is applied **per chunk** (a global `rotate_half` would mix axes).
  Verified norm-preserving and relative-shift-invariant.
- **`SelfAttention` (SDPA).** PyTorch `scaled_dot_product_attention` (FlashAttention on CUDA). **qk_norm**
  (parameter-free RMSNorm on q,k before RoPE) bounds attention-logit magnitude under AMP — the NaN
  mitigation. Boolean `True=allowed` masks are passed straight through.
- **`temporal_causal_mask`.** Rule-based `(N,N)` bool: full attention **within** a frame, causal **across**
  frames (`t_j ≤ t_i`). Built once.
- **`SelfAttnBlock`.** Standard pre-norm transformer block (self-attn + MLP).

---

## 6. The single-step cascade (`hvp/predictor/`)

### Per-stage worker
Each stage is a small transformer over its latent grid: input projection + a learned per-cell spatial
bias, `n_layers` blocks of **causal-temporal + 3D-RoPE self-attention** (optionally a cross-attention
sublayer), a final norm, and a head. Heads: **`l2`** (deterministic, MSE) or **`gaussian`** (per-dim NLL).
Position `t` predicts frame `t+1`; the loss supervises positions `0..T-2` against `z[:, 1:]`.

### Cross-scale conditioning + the receptive-field masks (`masks.py`)
The cascade runs **coarse→fine** in one pass. TOP is predicted first; its hidden features condition MID
via **cross-attention**; MID's hidden conditions BOT. The child (finer, more frames) queries the parent
(coarser) hidden tokens under a **receptive-field-overlap mask**:

- **Spatial:** each child cell attends its single covering parent cell (`hc//ratio, wc//ratio`).
- **Temporal:** `causal_receptive_groups(child_T)` gives, for each parent frame `p`, the child frames it
  summarizes. This reproduces the intended windows exactly — for `top(T3)→mid(T5)`:
  `t0→{m0}, t1→{m0,m1,m2}, t2→{m2,m3,m4}` (i.e. `[pad pad m0], [m0 m1 m2], [m2 m3 m4]`).
- **Future-leak fix (essential):** a child frame `tc` may attend parent frame `p` **only if
  `max(RF(p)) ≤ tc`**. The naive "RF overlaps" rule (`isdisjoint`) connects a child to a parent whose
  receptive field includes frames *after* `tc`, importing the future; the `max(RF(p)) ≤ tc` rule is
  strictly causal. A golden test asserts zero leaking connections.

Masks are **rule-based and cached** as registered buffers (`CascadeMasks`), so they cost nothing per step
and move with the model. They are tiny (the largest, mid→bot, is `2304×80` bools).

### Cascade direction & detachment
`HierarchicalCascade.forward` iterates TOP→MID→BOT, passing each stage's hidden to the next finer stage.
The parent hidden is **detached by default** (`detach_parent`), so each stage is shaped by its own
predictive loss and the cascade only carries information forward — not collapse gradients. This is
verified by `test_conditioning_flows_down_not_up` (perturbing TOP changes BOT; perturbing BOT never
changes TOP/MID).

The **1×1 TOP degeneracy** noted in the plan is avoided because the TOP→MID cross-attention is over the
**temporal** RF window (up to 3 TOP frames per MID query), not a single key.

---

## 7. Objective (`hvp/objectives.py`)

```
total = recon_weight · recon                      # deep-supervised L1/L2 over active depths
      + vicreg_weight · Σ_s scale_s · var_hinge(z_s)   # per-dim std floor (γ=1): anti-collapse
      + vicreg_cov_weight · Σ_s scale_s · cov_pen(z_s) # off-diagonal covariance: anti dimensional-collapse
      + λ(step) · Σ_s w_s · MSE(ẑ_s, stopgrad(z_s)[:,1:])   # detached single-step prediction
```

- **Anti-collapse.** Predicting your own latent has a `Var(z)→0` global optimum. Three guards, all on by
  default: **stop-grad** on the predictive target (the encoder is never rewarded for shrinking itself), an
  **asymmetric predictor**, and a **VICReg variance hinge** (`max(0, γ-std)²`, weighted ~1 at TOP/MID,
  0.1 at BOT) plus a small covariance penalty. Reconstruction carries the bulk of the anti-collapse load;
  VICReg is the hard floor. **Loss→0 with falling effective rank is the *signature of collapse*, not
  success** — hence the effective-rank canary in eval.
- **λ schedule.** `λ` ramps `0 → lambda_pred` over `pred_warmup_steps`, keeping the run recon-dominant
  early so the encoder establishes a grounded code before prediction pressure arrives.
- **Per-stage weighting.** Predictive losses are per-element means (so BOT's 2304 cells don't drown TOP's
  1), then coarse stages are up-weighted (`TOP=2, MID=2, BOT=1`) — they carry the rotation and collapse
  first.

---

## 8. Training (`hvp/train.py`)

One-shot: a single optimizer over tokenizer + cascade. **bf16 autocast** (no GradScaler), grad-norm
clipping, a non-finite-loss step-skip guard. Periodic held-out validation logs recon PSNR/SSIM, per-stage
predictive VE (cascade **on** vs **off** = the hierarchy lift), effective rank, and pixel-space
single-step prediction (predicted vs persistence). Everything is logged to **wandb** (scalars + a
GT/Recon/Pred **image table** with `image_index` and `category` columns).

---

## 9. Metrics & gates

| metric | what it tells you |
|---|---|
| recon **PSNR / SSIM** (held-out) | tokenizer fidelity |
| **effective rank** per stage `(Σλ)²/Σλ²` | collapse canary — must stay near the latent dim |
| per-stage latent **std** | VICReg variance floor health |
| predictive **VE** (all / changed cells) | does the predictor beat persistence in latent space |
| **cascade lift** = VE(on) − VE(off) | does cross-attention conditioning actually help (the hierarchy) |
| pixel **pred PSNR/SSIM vs persistence** | the anti-blob proof (a blob loses to persistence) |
| grad norm, it/s, λ, active_depth | training health |

Gate sequence: **unit → G-recon → G0 (predictability) → G-stable → G-cascade → anti-blob**, then the
deferred **G-AR** (rollout) and **G-SO(3)**. See [RESULTS_V1.md](RESULTS_V1.md) for the passing numbers.

---

## 10. Config system (`hvp/config.py`)

Frozen dataclasses (`TopologyConfig`, `VAEConfig`, `PredictorConfig`, `DataConfig`, `OptimConfig`,
`RunConfig`, `Config`). `Config.from_yaml` supports `_base_` inheritance and dot-path CLI overrides
(`--set predictor.dim=256`), with **unknown-key rejection**. `TopologyConfig` is the single shape source;
`stage_spatial()` / `stage_temporal()` derive the per-stage grids.

---

## 11. Known limitations (v1)

- **L2 blur.** The deterministic MSE head averages over residual uncertainty → predicted frames track
  rotation but soften fine detail. The committed fix is a **GIVT-GMM distributional head** before SO(3).
- **BOT compute.** BOT self-attention is over 2304 tokens — ~the bulk of compute. **Windowed BOT
  attention** is the planned efficiency lever.
- **Single-step only.** Autoregressive rollout (and its exposure-bias drift) is the deferred **v1.5**
  frontier; the noised-conditioning hook is the next addition.
- **Z-axis rotation.** Deliberate scope-lock; SO(3) is a separate milestone tied to the distributional
  head + best-of-N eval.
