# PLAN — VQ-free hierarchical single-step video predictor (with causal temporal downsampling)

> **Status:** plan (2026-06-19). Target repo: `hierarchical-video-prediction`, branch `feat/vq-free-predictor`.
> Built from a deep 4-repo code read (repo1 `hierarchical-vq-transformers`, repo2 `SEED-Voken`,
> the working `feat/continuous-joint-blt` branch, the shape toolbox) + a 5-topic SOTA web survey
> + a 4-lens adversarial critique + synthesis. Companion evidence is on the
> `feat/continuous-joint-blt` branch (`README.md`, `docs/SESSION_SUMMARY_2026-06-19.md`).

---

## 0. TL;DR — the one thing to internalize first

**Your direction is right, and your own later work already half-proves it.** The thing you remember
as "not working" was SEED-Voken's *recursive multi-step* predictor. The **single-step continuous BLT
cascade you're describing already exists and works** on the sibling branch `feat/continuous-joint-blt`:

> **Teacher-forced MID token accuracy 0.824 vs persistence 0.476 (+0.348 on a true cross-split),
> 75.6 % on the cells that actually change. AR rollout does NOT blob-collapse.** (verified in that
> branch's `README.md` results table)

So the question is no longer *"can a single-step predictor escape the blob?"* — it can. The plan below
keeps that win and adds your four refinements as **independently-gated changes** rather than a rebuild:

| Your proposed change | Status vs the working branch | Plan treatment |
|---|---|---|
| **Drop VQ entirely (continuous AE)** | Predictor already targets the *continuous pre-quant* feature; LFQ is nearly vestigial to it | **Adopt.** Replace LFQ with a plain continuous bottleneck. Low risk. |
| **Keep temporal downsampling (causal conv, 9/5/3)** | The +0.348 win was at **1:1 temporal**; downsampling is the **genuinely untested** variable | **Adopt as configurable default, but gate on a cheap probe (G0) that re-measures MID predictability at 2–3 strides.** |
| **One-shot training (no frozen seam)** | The only direct test of predictive-gradient-into-encoder (α=0.3) **collapsed recon ~1.7×** | **Adopt one *run*, but with a DETACHED predictive target** (encoder shaped by recon; predictor on top). True joint is an opt-in escalation. |
| **Cross-attention conditioning w/ RF-overlap windows** | Current hierarchy uses *upsample-add into a 1×1 TOP* (helps only +0.013); cross-attention is **untested, not disproven** | **Adopt — but fix the future-leak in the as-built mask and widen/condition TOP properly** (it degenerates at 1×1). This is the real test of your hierarchy pillar. |

**The actual binding constraint for the dissertation is now AR rollout drift, not MID predictability.**
Stabilizing single-step training (your stated v1 goal) is largely a solved problem on the working
branch; the value of v1 here is (a) proving the **hierarchy pillar** (does cross-attention + temporal
downsampling genuinely help?) and (b) laying clean, anti-drift-ready foundations.

---

> **STATUS 2026-06-19: V1 BUILT — gates G-recon, G0, G-stable, G-cascade, anti-blob all PASS.**
> See [RESULTS_V1.md](RESULTS_V1.md). 26/26 tests; MID predictability 0.84 (downsampling worry
> falsified); cross-attn hierarchy lift +0.141/+0.076; single-step beats persistence +3.77 dB.

## 0.1 Decisions locked (2026-06-19)

| Fork | **Locked choice** | Consequence for this plan |
|---|---|---|
| **Codebase** | **Fresh rebuild** on `feat/vq-free-predictor` (clean slate) | Build the new `hvp/` from scratch. *Still port the verified assets* (§7): `conftest.py` RSS/OMP harness, config-tree pattern, the causal-conv transplant, and the `ContinuousStagePredictor`/`g0_premise` logic **as reference implementations**, re-implemented cleanly — don't reinvent the validated math, but don't carry the discrete machinery. |
| **Training** | **Detached-target one-shot** | One training run; encoder shaped by recon + VICReg; predictor on `stopgrad(z_next)`. True joint stays opt-in (G-joint). |
| **v1 priority** | **Hierarchy first, then AR** | v1 headline = prove temporal-downsampling + cross-attention help (G-cascade). AR drift is the explicit **v1.5** focus (G-AR) — bake the noised-conditioning + rollout-gap-logging hooks from day 0 so the pivot is a config flip, not a rewrite. |

Still open (defaults in §3.1 stand unless you say otherwise): per-stage head (MSE-behind-G0 vs GMM now),
Z-only scope lock, bottleneck (no-KL AE vs light-KL).

---

## 1. Does the proposal make sense? (verdict)

**Yes — directionally correct, with four corrections baked into the design below.**

1. **VQ-free is sound** and SOTA-supported. Continuous KL/AE video latents (Cosmos uses a *no-KL* 16-ch
   AE; MAR/GIVT pair continuous latents with distributional heads) beat VQ on reconstruction and remove
   the FSQ/LFQ composite-index pathology that the project already flagged as a dead end. The
   `IndexCodec` MSB-scramble blocker and per-bit-CE machinery become dead weight — good riddance.

2. **Single-step (not recursive) is correct** and already validated. repo1 was *already* single-step;
   SEED-Voken's recursive envelope was the thing that didn't work. Remove it.

3. **Your "BLT trains with one latent shift then one byte shift" mental model is HALF right.** Verified
   against the BLT paper (Pagnoni et al., arXiv:2412.09871): the *expensive coarse/global* step runs
   **once per patch** (correct — this is BLT's whole efficiency point and maps to "one expensive coarse
   temporal step per K fine frames"). But **vanilla BLT decodes the fine bytes autoregressively, one at
   a time**, not in a single parallel shot. The single-parallel-shot fine decode you're imagining is
   exactly what the *follow-up* "Fast BLT / BLT-D" (arXiv:2605.08044) had to *add*. **Implication:** a
   fixed-ratio, parallel per-stage shift is a *deliberate simplification* of BLT, not "what BLT does" —
   which is fine for v1; just don't claim faithfulness. It is also why the per-stage head choice matters
   (see §6): a parallel deterministic shift over uncertain futures is precisely the mode-averaging recipe.

4. **Keeping temporal downsampling is defensible but must be settled empirically, not by folklore.**
   The prior claim "repo1 worked *because* it was 1:1" is **folklore** (repo1 never measured
   prediction-vs-persistence; its viz panel passes for a pure copy model). And the 9/9/9 full-1:1 run
   did **not** restore same-cell predictability. So neither "downsampling is fatal" nor "1:1 rescues it"
   is supported. Keep it as the default (BLT amortization depends on it), make 1:1 the opt-out, and let
   **G0** pick the stride.

---

## 2. Corrected ground-truth evidence (supersedes the stale "blob" memo)

| Old claim (stale) | Corrected truth (from `feat/continuous-joint-blt` results) |
|---|---|
| "Coarse/MID tokens are unpredictable; predictor blobs even teacher-forced." | **False.** Learned attention + 3D-RoPE single-step predictor: MID +0.348, BOT +0.178 over persistence, cross-split. |
| "Flow/feature probes show MID is flat (VE ≈ −0.07)." | Those were **flow-only oracles** (copy/shift/rot90) — structurally unable to model in-place re-coding. A *learned* head predicts the re-coding. |
| "Joint continuous training is the fix for MID flatness." | The only direct test (α=0.3) **collapsed recon ~1.7× and did not raise predictability.** Predict-your-own-latent has a `Var(z)→0` optimum (SimSiam/BYOL/LeWorldModel). |
| "The blob is the unsolved problem." | The unsolved problem is **AR exposure-bias drift** (TF→AR gap +0.151; light noise + 1-step scheduled sampling did **not** close it). |
| "Cross-stage conditioning barely helps (+0.013), so hierarchy is weak." | That +0.013 is from **upsample-add into a degenerate 1×1 TOP**, *not* cross-attention with a ≥2×2 TOP. The hierarchy claim is **untested, not disproven.** |

---

## 3. Target architecture (decisions resolved — defaults, not menus)

```
ENCODER (causal-conv 3D, NO quantization)            PREDICTOR (single-step BLT cascade, 1 forward pass)
────────────────────────────────────────            ───────────────────────────────────────────────────
x [B,3,9,64,64] in [-1,1]                            per-stage continuous latent z_s (detached as pred target)
  │ CausalConv3D left-pad (K-1,0)                    predict TOP once (causal 3D-RoPE self-attn) → ẑ_top, h_top
  ▼ /4 spatial                                         │  condition (see §5) ──┐
 BOT  z 16×16 × C_bot=16  @ T=9                       predict MID once ◄────────┘ → ẑ_mid, h_mid
  ▼ /4 spatial, /2 temporal (causal)                   │  condition ───────────┐
 MID  z  4×4  × C_mid=8   @ T=5                       predict BOT once ◄────────┘ → ẑ_bot
  ▼ /4 spatial, /2 temporal (causal)                  decode(ẑ_bot) → predicted pixels (eval/logging)
 TOP  z  1×1  × C_top=8   @ T=3                       loss: per-stage MSE(ẑ_s[t], stopgrad(z_s[t+1]))
DECODER: causal-conv 3D upsample BOT(16×16@9)→64×64@9  + L_recon(decode(z_bot), x) + VICReg variance floor
(temporal_downsample is a decoupled List[bool])      within-frame full self-attn; block-causal across time
```

### 3.1 Resolved decisions

| Decision | **Default** | Why (grounded) |
|---|---|---|
| **Bottleneck** | **Plain continuous AE, no KL** (C: 16/8/8 for bot/mid/top) | Cosmos ships a no-KL 16-ch AE; KL collapse is a documented failure and a light-KL buys nothing on a low-dim deterministic task. Add light KL (~1e-5) only if warm-starting from an image VAE later. |
| **Predictive target** | **Detached** `stopgrad(z_next)` + **asymmetric predictor** (already so) + **VICReg per-dim variance hinge** (γ=1.0; weight 1.0 at TOP/MID, 0.1 at BOT) + small covariance penalty (~0.04) | SimSiam/BYOL: stop-grad is mandatory and gives only *implicit* variance reg; recon prevents *total* but not *dimensional* collapse; VICReg is the cheap *hard* floor. **Not** EMA/SIGReg (SIGReg hurts at low intrinsic dim — LeWorldModel). |
| **Predictive grad into encoder ("joint")** | **OFF.** Opt-in escalation, gated (G-joint), λ ramped 0→target over 5–10k steps | Only direct test collapsed recon 1.7× and raised nothing. Frozen/detached already gives +0.348. |
| **Per-stage head** | **v1 (Z-only): L2-MSE — but only after it clears the §4 constant-baseline probe.** Pre-SO(3): swap to **GIVT Gaussian-mixture (K=2–4) NLL** head | MSE provably regresses to the conditional mean → blur; Z-rotation is near-unimodal so MSE is tolerable *if gated*; SO(3) reintroduces depth/axis multimodality MSE *will* blob. GMM is the cheap distributional fix; MAR diffusion-MLP only if GMM underfits. |
| **Cross-scale conditioning** | **v1: cross-attention at MID/BOT (your design) with the leak fix (§5) + a ≥2×2 TOP, OR adaLN-Zero for the 1×1-TOP→MID hop.** | Cross-attention is the untested hierarchy test you want — but at a literal 1×1 TOP it degenerates to a single-key softmax (zero cross-gradient). Either widen TOP to ≥2×2, or condition the TOP→MID hop via adaLN-Zero (DiT ablation: adaLN ≥ cross-attn for low-cardinality/global signals) and keep cross-attention where spatial alignment matters (MID→BOT). |
| **Parent feature** | **Detached by default** (`detach_parent: true`); condition on the **predicted** coarse latent (cascade) but add **scale-level scheduled sampling** (anneal p(use GT coarse)=1→0) | Non-detached parent + predict-own-latent opens a collapse channel; GT→predicted curriculum closes the train/eval gap (train-what-you-test). |
| **Loss schedule** | **Phase A** recon-dominant (λ≤0.1 until recon plateaus & MID/TOP effective-rank healthy); **Phase B** ramp λ→0.25–0.5 over 5–10k steps. Per-stage L_pred = per-element **mean**, then up-weight coarse: **TOP=2, MID=2, BOT=1**. Invariant **L_recon ≥ λ·ΣL_pred** all run | Raw sum lets BOT (256 cells) drown TOP/MID (1, 16 cells) — the rotation-carrying stages that historically collapse first. |
| **RoPE / attention** | **Factorized 3D RoPE (t,h,w), computed fp32→cast; qk_norm ON; within-frame full self-attn, block-causal across frames** | qk_norm bounds attention-logit magnitude under AMP (directly mitigates the repo's known NaN); time is the AR/causal axis (NOVA/InfinityStar factorization). |
| **AMP / norm** | **bf16 (no GradScaler); LayerNorm (not FrameWiseGroupNorm) in enc/dec OR fp32 norm region; finite −1e4 SDPA mask (never −inf) + assert no all-masked rows; per-step non-finite-loss skip-guard** | SEED-Voken ships `debug_predictor_nan_step1.py` — a live wound. WF-VAE replaced GroupNorm→LayerNorm for causal-video stability. A fully-masked row + −inf → softmax NaN, possible at the stride-2 overlap windows. |

---

## 4. The cheapest falsification experiment (HARD pre-build gate — G0)

**Do not build the encoder/decoder/cascade first.** The decisive question for *your* new variable
(temporal downsampling + fully-continuous + one-shot) is: **does MID stay predictable at a downsampled
stride on a continuous latent?** Answer it in <1 GPU-hr before any multi-day build.

- **Reuse the working probe**, don't invent: `feat/continuous-joint-blt`'s `scripts/g0_premise.py` +
  `hvp/predictor/continuous.py::ContinuousStagePredictor` (the exact 3D-RoPE attention head that
  produced +0.348). Port its held-out-**by-clip** split + constant-baseline math.
- **2×2 screen** on cached pre-quant MID features: `{detached-recon-only latent, small-α joint latent}`
  × `{2–3 temporal strides on Z-only data}`. Train only the tiny head, a few hundred steps, on cached
  fp16 features.
- **Metric:** cross-split MID predictive lift over persistence **on changed cells** + a **collapse
  canary** (per-dim std + effective rank = (Σλ)²/Σλ²).
- **Decision rule:**
  - **MID lift ≥ 0.05 over persistence at the chosen stride** → MID is not the wall; proceed on the
    **detached** latent, lock that stride.
  - **Joint enabled only if its lift beats detached by a margin that survives a recon-non-regression
    check.** (Today α=0.3 fails this.)
  - **MID lift ≤ 0 with a small train/test gap, or effective rank crashes while L_pred→constant-baseline**
    → the single-step continuous-MSE plan reproduces the blob at that stride → reduce the stride (toward
    1:1) rather than tuning the conditioner. **STOP and re-pick topology before the build.**
- **Why decisive:** isolates the only genuinely new variable (downsampled continuous latent) against the
  project's own harness, settles the stride question in the same run, and detects collapse in minutes.
  **Loss→0 here is the SIGNATURE of collapse, not success — gate on effective rank, not nominal loss.**

---

## 5. Masking design (rule-based, built-once, cached — meets your explicit requirement)

Your overlapping-window spec is **exactly** SEED-Voken's causal receptive-field logic — verified:
`causal_receptive_groups(9, kernel=3, stride=2)` →
`m0→{b0}, m1→{b0,b1,b2}, m2→{b2,b3,b4}, m3→{b4,b5,b6}, m4→{b6,b7,b8}` =
your `[pad pad b0], [b0 b1 b2], [b2 b3 b4], …`. So the *rule* is correct and reusable. Two non-negotiable
fixes when porting it:

1. **Fix the future leak.** The as-built rule connects a child to *any* parent whose RF merely
   **overlaps** (`not child_frames.isdisjoint(parent_frames)`). Because a coarse frame's causal RF is
   wide, this imports content **after** the child's target (verified: BOT k=0, query frame 4 / target
   frame 5, attends a MID parent whose fine-RF = {4,5,6} — contains the target). **Replace with a
   strictly-causal rule** `max(parent_RF) <= max(child_RF)` and add a golden test asserting **no**
   future-leaking parent connection. (Teacher-forced numbers are inflated until this is fixed.)
2. **Keep it rule-based + cached** (as SEED-Voken already does via `PredictorMaskCache` registering
   `nn.Module` buffers — no per-step rebuild, moves with the model). Boolean masks; SDPA `True=allowed`
   polarity → convert to a **finite** additive bias (−1e4), never −inf. Memory is tiny
   (largest is BOT self-attn ≈ (9·256)² bools ≈ 5 MB; cross masks far smaller). Assert no fully-masked
   query rows.

**TOP degeneracy:** at a literal 1×1 TOP, the spatial cross-mask is all-ones → cross-attention attends a
single key (zero cross-gradient; SEED-Voken's `masks.py:31-38` documents this). Resolve by either
**(a) widening TOP to 2×2** (recommended — it's the actual hierarchy test and removes the degeneracy), or
**(b) conditioning TOP→MID via adaLN-Zero** (pooled TOP latent → per-channel scale/shift/gate, gate
zero-init) and reserving cross-attention for MID→BOT where spatial alignment is real.

---

## 6. SOTA grounding (what we borrow, cited)

- **BLT** (arXiv:2412.09871): coarse-rate global model + cheap fine expansion = the efficiency
  rationale. *Faithful* BLT decodes fine bytes AR; our parallel per-stage shift is a Fast-BLT-style
  simplification (arXiv:2605.08044) — adopt consciously.
- **MAR "diffusion loss"** (Li et al., NeurIPS 2024, arXiv:2406.11838) & **GIVT** (Tschannen et al.,
  ECCV 2024, arXiv:2312.02116): distributional continuous heads that model `p(x|z)` not its mean — the
  committed pre-SO(3) head swap (GIVT-GMM cheap; MAR diffusion-MLP strong).
- **Diffusion Forcing** (Chen et al., NeurIPS 2024, arXiv:2407.01392) + **GameNGen** (arXiv:2408.14837):
  per-frame independent noise / noised-history rollout = the cure for AR drift (v1.5 lever; bake the
  noised-conditioning hook in now).
- **V-JEPA / collapse theory** (SimSiam arXiv:2011.10566, BYOL 2006.07733, VICReg 2105.04906,
  LeWorldModel 2603.19312): the detach + asymmetric-predictor + VICReg-floor recipe of §3.1.
- **Causal video tokenizers** (MAGVIT-v2 2310.05737; Cosmos; WF-VAE/Open-Sora-Plan): causal left-pad
  `(K-1,0)`, the `1+s·t↔1+t` mapping + s−1 frame-drop on decode, decoupled temporal/spatial strides,
  LayerNorm-over-GroupNorm, no-KL 16-ch AE.
- **VAR / DiT / NOVA**: adaLN-Zero ≥ cross-attn for low-cardinality conditioning; in-context
  block-causal concat as a zero-param alternative.

---

## 7. Package layout + PORT / REWRITE / DROP

**One governing lineage: VQ-free continuous.** This makes the discrete codec/head dead weight.

```
hvp/
  config/        frozen dataclasses; TopologyConfig = single shape source; from_yaml/_base_/dot-override; unknown-key reject
  data/          shape_video.py (Z-only locked v1), tokens.py (feature cache, fp16, ckpt-sha-stamped)
  tokenizer/     encoder.py, decoder.py        (REWRITE fresh, causal-conv temporal downsample, no ch_mult)
  predictor/     stage.py (StagePredictor ABC), cascade.py (builder from TopologyConfig),
                 conditioning.py (cross-attn + adaLN-Zero, behind flags),
                 heads.py (l2 | gaussian-GMM), masks.py (rule-based, cached, causal-LE), rope.py
  attention/     sdpa.py (force SDPA, finite mask, qk_norm), rope.py (cached cos/sin fp32), blocks.py
  objectives.py  recon + per-stage detached MSE + VICReg + schedule
  train/         phase-as-config: fit_oneshot | (later) cache_features | fit_predictor
  probes/        continuous_predictability.py  (the §4 G0 gate)
  checkpoint.py
tests/           conftest.py (RSS watchdog + OMP/KMP guard), invariant/structural goldens
```

| Component | Action | Source / note |
|---|---|---|
| `tests/conftest.py` (OMP/MKL=1, KMP guard, single-thread torch, 12 GiB RSS watchdog) | **PORT VERBATIM** | HVP — Windows stability essential |
| `checkpoint.py` (top-k, cfg-as-dict, sha256 stamp) | **PORT VERBATIM** | HVP |
| Frozen-dataclass config tree + TopologyConfig single source | **PORT** | HVP |
| `attention/sdpa.py`, `rope.py` (cached cos/sin), `temporal_causal_mask` | **PORT** + apply finite-mask + qk_norm fixes | HVP |
| `ContinuousStagePredictor`, `HierarchicalPredictor`, `eval_rollout.py`, `commit_token_acc`/VE metrics, `g0_premise.py` | **PORT** (the working +0.348 core + harness) | HVP `feat/continuous-joint-blt` — **do not rebuild this** |
| CausalConv3D left-pad `time_pad=K-1` + `temporal_downsample: List[bool]`→`(t_stride,2,2)` + `1+s·t↔1+t` decode frame-drop | **TRANSPLANT ONLY THIS** | repo2 `improved_video_model.py` (verified ~:36). Validate the frame-count round-trip with a test. |
| Encoder/decoder body | **REWRITE FRESH** (HVP style, no ch_mult) | repo2 entangled w/ SQ-VAE/FSQ/ELBO; a subtly-wrong pad corrupts every latent |
| `FrameWiseGroupNorm` | **REWRITE → LayerNorm** (or fp32 region) | WF-VAE result |
| `masks.py` cross-mask rule | **REWRITE rule** (`max(parent)<=max(child)`); **keep** the cached-buffer mechanism | repo2 leak (§5) |
| Per-stage head | **REWRITE** (`l2` \| `gaussian`) | bare composite-index head was the blob |
| LFQ/FSQ/SQ-VAE/ELBO, `codec.py`, `tokenizer/quant.py`, per-bit-CE head, discrete bit-equality gates, `IndexCodec` | **DROP** | dead in a continuous repo |
| Recursive multi-step envelope/rollout, `reversible_block` + `rev_back_prop`, Lightning/class_path reflection | **DEFER behind flags** | reversibility is dissertation goal #3 |

**OOP:** `StagePredictor` ABC + `cascade.build_from(TopologyConfig)`; dataclass/NamedTuple IO
(`EncoderOut`, `StageMasks`); frozen-dataclass config with validation in `__post_init__`; masks in an
`nn.Module` buffer registry. Goldens are **invariant/structural** (decode∘encode PSNR floor;
no-future-leak mask assertion; persistence relations; CPU-deterministic atol/rtol) — *not* bit-exact
(no discrete decode contract in a continuous repo).

---

## 8. Phased build order (falsifiable gates)

- **G0 — Predictability + stride screen (PRE-BUILD, §4).** *Pass:* cross-split MID lift over persistence
  ≥ 0.05 on changed cells at a chosen stride, effective rank above floor. *Fail* → reduce stride / fix
  topology before building.
- **G-recon — Tokenizer round-trip.** *Pass:* decode∘encode PSNR ≥ floor **and** bit-exact `9→5→3→…→9`
  frame-count round-trip (`1+s·t↔1+t` with the s−1 drop).
- **G-stable — One-shot detached training.** *Pass:* recon converges; MID/TOP per-dim std + effective
  rank stay > 50 % of latent dim; no NaN over a full short run; L_pred beats the constant baseline at MID.
- **G-cascade — Cross-scale conditioning (your hierarchy test).** *Pass:* held-out MID/BOT lift above
  marginal **and** no future-leak (golden green); cascade adds measurable lift over independent stages
  (this is the real test of the +0.013 → "does the hierarchy actually help" question).
- **G-joint (opt-in) — Predictive-into-encoder.** *Pass:* with λ ramped + VICReg active, MID VE rises
  vs the detached baseline **without** recon regressing past tolerance and effective rank holding.
- **G-AR (v1.5, the real frontier) — Rollout stability.** *Pass:* TF→AR gap closes below threshold with
  noised-conditioning + (multi-step) scheduled sampling; no softening/blob over the training horizon.
- **G-SO(3) (later) — Multimodal lift.** *Pass:* GMM/diffusion head + best-of-N eval beats single-sample
  MSE; coarse stage doesn't re-blob under axis/depth ambiguity.

---

## 9. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Predictive gradient into encoder collapses latent (`Var(z)→0`; α=0.3 already collapsed recon 1.7×) | **High** | Joint OFF by default; stop-grad always; asymmetric predictor; VICReg variance hinge at TOP/MID; λ ramp from 0; effective-rank gate. |
| Bare L2 head → mean-regression blob (worse as rotation moves content across coarse/mid cells) | **High** | Gate MSE on the §4 constant baseline before GPU; swap to GIVT-GMM before SO(3); best-of-N eval under any multimodality. |
| Cross-mask future leak (`isdisjoint` overlap rule) | **High** | `max(parent)<=max(child)` + golden test; re-run oracle (an accuracy drop = you were leaking). |
| Topology/horizon mismatch (TOP-T3/MID-T5 too few native frames; coarse stages can't supply many causal targets) | **High** | G0 across candidate strides; if a stride starves the coarse stages, reduce downsampling rather than tuning the conditioner. |
| Budget mis-spent re-solving MID-teacher-forced (already solved) while AR drift (+0.151, unclosed) goes unaddressed | **High** | Treat MID-TF as solved; bake noised-conditioning hook + TF-vs-1-step-rollout logging from day 0; plan Self-Forcing / Diffusion-Forcing as v1.5. |
| "Temporal downsampling may be fatal" vs keeping it — **reconciled** | **Medium** | Evidence supports keeping it (1:1 did not rescue predictability; "repo1 worked at 1:1" is folklore). Default it; pick stride by G0; don't force 1:1 on a belief. |
| AMP NaN (GroupNorm + fp16 + −inf SDPA mask at stride-2 overlaps) | **Medium** | bf16; LayerNorm/fp32 norm; finite −1e4 mask + no-all-masked-row assert; qk_norm ON; non-finite-loss step-skip. |
| Recon-only pins latent to pixel geometry; permits dimensional collapse at TOP/MID | **Medium** | VICReg hinge at TOP/MID; CI collapse probe (per-dim std, effective rank, L_pred-vs-constant gap); ramp λ only after rank healthy. |
| Scale-by-scale exposure bias (TF good, AR diverges; coarse error amplifies up the pyramid) | **Low (deferred)** | OK to defer for v1; bake hooks now (noised parent + rollout-gap logging) so it's visible, not discovered post-claim. |

---

## 10. Decisions for you (these change the build)

1. **Codebase strategy.** Build on `feat/continuous-joint-blt` (keep the working +0.348 predictor +
   eval harness + 75 green tests; add temporal downsampling + go fully VQ-free incrementally) — *or*
   fresh on `feat/vq-free-predictor`, porting the clean skeleton but rewriting tokenizer/predictor? (I
   recommend building on the working branch; rewriting risks regressing a hard-won result.)
2. **v1 training mode.** Detached-target one-shot (recommended, honors "1-shot, no pretraining" without
   the collapse) — or true joint predictive-gradient-into-encoder as a non-negotiable thesis claim in
   v1 (its only test collapsed recon 1.7×)?
3. **v1 scientific priority.** Prove the **hierarchy** (cross-attention + temporal downsampling genuinely
   help — the refactor you described) — or pivot v1 straight to **AR drift** (the actual open problem)?
   These compose; the question is which is the headline of v1.
4. **Head for v1.** Bare-MSE **behind the G0 gate** with GMM as the committed pre-SO(3) swap — or pay for
   GMM now?
5. **SO(3) scope.** Confirm **Z-axis-only** as the explicit v1 contract (couple the SO(3) lift to the
   GMM/diffusion head + best-of-N, not to this architecture pivot)? The dataset's Z-lock is a one-line
   change (`shape_video.py` rotation step) when you're ready.
6. **Bottleneck.** Plain no-KL AE acceptable — or do you want a light-KL VAE / latent-space
   regularization to a pretrained image VAE for warm-start?

---

## 11. Immediate next milestone (fresh-rebuild sequencing)

Because we're rebuilding fresh and the working branch's encoder has **no temporal downsampling**, the
VQ-free causal-temporal-downsample encoder must exist before G0 can answer the stride question. Order:

1. **Scaffold** `hvp/` (package tree §7, frozen-dataclass config tree, **port `conftest.py` verbatim**,
   `pyproject.toml`, Z-only `shape_video.py`). One smoke test green.
2. **VQ-free encoder/decoder** — the one genuinely new component. Transplant *only* CausalConv3D
   `time_pad=K-1` + `temporal_downsample: List[bool]` + the `1+s·t↔1+t` decode frame-drop from repo2
   `improved_video_model.py`; LayerNorm; no ch_mult; no quant. **Gate G-recon** (decode∘encode PSNR
   floor + frame-count round-trip test).
3. **Quick recon-train** the encoder at 2–3 temporal strides on Z-only data; cache continuous MID
   features (fp16).
4. **G0 probe** `hvp/probes/continuous_predictability.py` (re-implement `g0_premise` +
   `ContinuousStagePredictor` cleanly): cross-split MID lift over persistence on changed cells +
   effective-rank canary. **This settles the stride and validates/falsifies the whole VQ-free plan in
   <1 GPU-hr before the predictor build.**
5. Only on G0 pass → build the single-step cascade + cross-attention conditioning (with the §5 leak
   fix) → G-stable → G-cascade.
