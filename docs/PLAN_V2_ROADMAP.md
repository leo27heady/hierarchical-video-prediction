# PLAN V2 — Roadmap from single-step proof-of-concept to a dissertation-grade hierarchical video predictor

> **Status:** plan (2026-06-19). Repo: `hierarchical-video-prediction`, branch `feat/vq-free-predictor`.
> Supersedes the deferred-frontier sections of [PLAN_VQFREE_HIER_PREDICTOR.md](PLAN_VQFREE_HIER_PREDICTOR.md).
> Built from four independent reviews (current-state code/training, SOTA positioning, prediction-horizon/AR,
> gap-enumeration) cross-checked against the actual source (`hvp/predictor/cascade.py`, `objectives.py`,
> `continuous.py`, `metrics.py`, `train.py`, `config.py`, `tokenizer/decoder.py`, `data/shape_video.py`).
> This document is the single source of truth for what to build next and in what order.

---

## 1. Honest state of the union

**Is it good? Yes — as a v1 foundation. Is it a dissertation-grade hierarchical *video predictor*? Not yet, because it cannot predict a video.** v1 is a clean, well-engineered, honestly-gated single-step proof-of-concept. The architecture is sound and the load-bearing invariants are real and verified in code, not just claimed: the per-frame `FrameNorm3d` that keeps the causal CNN from leaking the future through normalization statistics; the strictly-causal leak-free RF cross-attention mask (`max(RF(parent)) ≤ child_frame` in `hvp/predictor/masks.py:54`, with a golden no-leak test); the detached-target one-shot objective + VICReg variance/covariance floor that keeps all stages near full effective rank (bot 15.9/16, mid 7.9/8, top 6.9/8) and avoids the ~1.7× reconstruction collapse the predecessor's joint training hit. Training is stable: bf16 + qk_norm removes the AMP NaN risk, grad clipping + a non-finite step-skip guard hold, recon is sharp (32.3 dB / SSIM 0.975), and the cross-attention cascade gives a real (if modest) hierarchy lift (bot +0.061, mid +0.022 VE). The clean negative result — causal temporal downsampling 9→5→3 does **not** destroy held-out MID predictability (VE 0.73–0.84) — is a genuine, decision-redirecting finding and the most defensible thing here.

**But on the axis that defines a video predictor — prediction horizon — v1 has effectively nothing, and on the axis that defines SOTA it is multiple orders of magnitude away.** Horizon is *literally 1*: a grep across `hvp/` finds zero rollout / autoregress / scheduled-sampling / noised-conditioning code; `HierarchicalCascade.forward` is a single teacher-forced pass over the full ground-truth detached latent stack (`cascade.py:125`), supervising position `t` against `t+1`, and the decoder only ever runs on one-step predictions. There is no AR loop, no TF→AR gap number, no per-horizon drift metric — `metrics.py` has only single-frame PSNR/SSIM. The head is a deterministic L2 mean-regressor (`continuous.py:54`; the `gaussian` NLL path exists but is **never sampled**, never used in `variance_explained`, never decoded), which provably regresses to the conditional mean and is L2-blurry on detail — tolerable only because Z-only rotation is near-unimodal. The headline "+5.10 dB over persistence" is a one-step number, and persistence is a trivially strong, easily-gamed baseline on a static background with one moving object; it becomes the *wrong* baseline past a few frames. Data is the easiest possible dynamics (rotating polycubes, Z-axis only, 1-D constant angular velocity, 64×64, T=9, never real video); eval has no LPIPS, no FVD, no best-of-N, no distributional or per-horizon metric; the hierarchy lift is single-seed; reversibility (dissertation goal #3) is entirely absent; and checkpoints save only model+cascade+topo (no optimizer/RNG/step → no resume). **Verdict: a strong, anti-drift-*ready* foundation and an intellectually honest one — not a finished chapter, not SOTA, no horizon. The gaps are additive (the causal masks, KV-cacheable SDPA, detached target, and per-frame structure are exactly what the AR recipes assume), so this is a fill-in, not a rebuild.**

---

## 2. The thesis through-line

**Defensible contribution:** *A hierarchical, temporally-downsampled, coarse→fine cross-attention cascade that predicts a continuous (VQ-free) video latent, where each coarse stage's hidden conditions the finer stage through a strictly-causal, leak-free receptive-field-overlap mask — and which, with a distributional head and noised-history rollout, sustains a longer prediction horizon than a flat (non-hierarchical) predictor by using the slow coarse "plan" to anchor fine-stage drift.*

The three specific, already-earned novelties to protect and build on:
1. **The leak-free `max(RF(parent)) ≤ child` causal cross-mask** — a real correctness contribution (the naive `isdisjoint`-overlap rule imports the future; this was caught and golden-tested).
2. **The detached-target one-shot recipe** (encoder shaped only by recon + VICReg; cascade predicts stop-grad latents) as a stable anti-collapse alternative to the joint training that collapsed recon ~1.7×.
3. **The empirical falsification** that causal temporal downsampling (9→5→3) does not destroy held-out coarse-stage predictability — a clean negative result that redirected the project.

**What must become true to defend it (the binding claims, in priority order):**
- **(C1 — Horizon, existential).** The predictor must actually roll out autoregressively and the TF→AR gap must be *measured and bounded*. A "video predictor" with horizon 1 cannot anchor a dissertation titled *hierarchical video prediction*. Target: free-run quality stays within a bounded margin of teacher-forced out to ≥2–3× the training horizon, and beats persistence on **changed-region** metrics at every horizon.
- **(C2 — Hierarchy pays for horizon, the differentiating claim).** The hierarchy must *earn its name on the horizon axis*: the cascade-on rollout must drift slower than cascade-off (flat) rollout. This is the single result that turns "a nice cross-attention trick" into "hierarchy buys long-horizon stability." Without it the contribution is just a modest single-step lift.
- **(C3 — Distributional honesty).** Under any multimodality (longer horizons, then SO(3)) the head must be distributional and evaluated best-of-N; single-sample MSE must be demoted to a sanity floor. A deterministic L2 head will blob the moment the future is multimodal, and single-sample PSNR will then mis-rank a stable sampler below a drifting mean.
- **(C4 — Rigor).** The headline lifts must survive multi-seed error bars and a proper eval suite (FVD/LPIPS/per-horizon), or they are within-noise folklore.

Everything in §4 is ordered to make C1→C2→C3→C4 true, in that dependency order.

---

## 3. Prioritized gap table

| # | Gap | Severity | Why it matters for the thesis | Fix (one line) |
|---|---|---|---|---|
| G1 | **No autoregressive rollout exists** (`hvp/` has zero rollout code; `cascade.forward` is one teacher-forced pass) | **Critical** | C1. A video predictor that cannot roll out has no horizon; this is the user's stated goal and both predecessors' documented failure mode | Add `hvp/predictor/rollout.py` with pixel-feedback **and** latent-feedback free-run; KV-cache the SDPA blocks |
| G2 | **No long-horizon metric** (`metrics.py` is single-frame only; persistence baseline is trivial past a few frames) | **Critical** | C1/C4. "Good horizon" is unmeasurable as built, so it is an opinion not a number | Per-horizon PSNR/SSIM/LPIPS curves, TF→AR gap, FVD over rolled clips, **changed-region** metrics, drift strip — from day 0 |
| G3 | **Exposure bias unaddressed** (encoder/predictor never see their own outputs; predecessor's +0.151 TF→AR gap unclosed) | **Critical** | C1/C2. The train/inference distribution gap is exactly what collapses long rollouts | Train *with* rollout: scale-level scheduled sampling (anneal p(GT parent/history)→0) + noised-history conditioning (Diffusion-Forcing/GameNGen) + a short Self-Forcing-style unroll loss |
| G4 | **Deterministic L2 head; `gaussian` path never sampled** (`continuous.py:54`, `predictive_loss` mean-regresses) | **High** | C3. Mean-regression blur compounds off-manifold under rollout and re-blobs under SO(3) | Wire a real distributional head end-to-end: GMM (K=2–4 NLL) first, MAR per-token diffusion-MLP if GMM underfits; add a `sample()` path used by VE, the pixel decode, and the AR loop |
| G5 | **Eval lacks perceptual/distributional/per-horizon metrics** (no LPIPS, FVD, best-of-N, Density/Coverage) | **High** | C3/C4. Single-sample PSNR/SSIM structurally rewards the blurry mean — it cannot detect the failure the project is most exposed to | Add `hvp/eval/` (LPIPS via `lpips`, FVD via I3D, best-of-N, Density/Coverage); wire into `validate()` and the AR eval |
| G6 | **Cascade error-amplification under rollout** (parent detached at `cascade.py:135`; coarse drift cascades to every fine stage with no correction) | **High** | C2. The hierarchy could *amplify* drift instead of anchoring it — directly threatens the differentiating claim | Stronger noised-conditioning on the coarse "plan"; scale-level scheduled sampling on the *parent hidden*; History-Guidance/DFoT on a flexible-length partially-noised history |
| G7 | **Hierarchy lift is single-seed and modest** (bot +0.061, mid +0.022, top +0.000) | **High** | C4. Within plausible noise → not yet a defensible scientific result | ≥3–5 seeds with mean±std + paired test on cascade-on vs -off; ablate `detach_parent`; wider/adaLN-Zero TOP for the 1×1 degeneracy |
| G8 | **Z-axis-only synthetic data** (1-D constant-velocity; never SO(3), never translation/occlusion/real video) | **Medium** | C3/external validity. A win here risks being "sophisticated copy-with-rotation"; multimodality is what justifies the distributional head | Stage difficulty: translation/multi-object/occlusion → SO(3) (paired with G4/G5) → a small real-video stress slice |
| G9 | **BOT attention is the compute wall** (2304 tokens, ~1.8 it/s; blocks resolution, horizon, rollout-training) | **Medium** | Enables C1 at scale — the rollout-training loop is too slow to iterate without it | Windowed/local BOT attention (+ optional wavelet/space-to-depth front-end) behind a config flag; verify VE parity vs dense |
| G10 | **Thin checkpoint/repro** (`_save` stores only model+cascade+topo; no resume; `max_checkpoints_keep` not enforced; only `torch.manual_seed`) | **Medium** | C4. Cannot resume long rollout-training runs; multi-seed needs full RNG seeding | Full resumable checkpoint (opt/scaler/RNG/step/config/data-hash/code-sha); `--resume`; top-k pruning; seed numpy+cuda+torch. (Cross-split is already real: separate `train_hash`/`val_hash` render caches — a stale docstring had claimed otherwise; fixed.) |
| G11 | **Reversibility absent** (dissertation goal #3) | **Low** | A stated goal with zero code; needs a hook before it can be a chapter | Design doc + `reversibility.py` behind a flag with an invertibility round-trip golden; integrate after C1/C3 are solid |
| G12 | **Joint-training escalation untested net of recon cost** (the "organize the latent for predictability" claim) | **Low** | The strongest version of the thesis, but it collapsed recon ~1.7× last time | Gated experiment: λ 0→target, VICReg on, hard gate = MID VE rises AND recon does not regress AND erank holds; run last |

---

## 4. Phased plan V2

Phases are ordered by **dependency and thesis value**. **Prediction horizon (P1) and the distributional head (P3) are front-loaded because they are the binding constraints** — and the metric/eval scaffolding (P0) comes first because you cannot gate horizon without it. Each phase lists: goal, work items, new metrics, and a single falsifiable **GATE**.

Throughout: every new behavior is a **config flag** (the v1 discipline), every new invariant gets a **golden test**, and metrics are logged to wandb from the first run of the phase, not after.

---

### P0 — Eval scaffolding + repro spine *(do first; cheap; unblocks every gate)*  — ✅ DONE
> **Status:** done ([RESULTS_P0_P1.md](RESULTS_P0_P1.md)). changed-region/LPIPS/FVD metrics + resumable
> checkpoints (`--resume`) + full seeding shipped; G0-eval green. (Cross-split was already real; FVD via
> torchvision r3d_18; FID-style proxy not needed.)

**Goal:** make horizon and distribution *measurable* before changing the model, and make long runs resumable/reproducible.
**Work items:**
- `hvp/eval/` module: `metrics_horizon.py` (per-horizon PSNR/SSIM, TF→AR gap, drift/softening curve), `lpips.py` (pip `lpips`), `fvd.py` (pretrained I3D features), `changed_region.py` (reuse the `change` tensor already computed in `variance_explained`, `continuous.py:75`, so persistence stops being a free win), `bestofN.py` + `density_coverage.py` (stubs, activated in P3).
- Full resumable checkpoint: extend `_save` (`train.py:189`) to store optimizer, scaler/None, torch+numpy+cuda RNG, step, full config-as-dict, data-cache hashes, code SHA; add `--resume`; enforce `max_checkpoints_keep` (top-k). Seed numpy + cuda + torch from `optim.seed`.
- Make the cross-split honest: wire `data.val_hash` as a genuinely independently-rendered cache (the loader is currently an index-style split despite the docs' "true cross-split" claim) — or downgrade the claim. Stamp data hashes into every run.
- `configs/` run YAMLs captured for the committed v1 (`v1_64.yaml`) so results are reproducible.
**New metrics:** LPIPS, FVD, changed-region PSNR/SSIM (wired but reported on existing single-step preds for now).
**GATE G0-eval:** on the existing v1 checkpoint, the suite reports LPIPS + FVD + changed-region PSNR/SSIM for single-step, and a run can be killed and `--resume`d to bit-identical loss. *Pass = the numbers print and resume reproduces; this is a tooling gate, not a science gate.*

---

### P1 — Autoregressive rollout + per-horizon drift instrumentation *(THE prediction-horizon gap; highest priority)*  — ✅ DONE
> **Status:** done ([RESULTS_P0_P1.md](RESULTS_P0_P1.md)). `rollout.py` (pixel + latent feedback) +
> `eval/horizon.py` + `scripts/eval_rollout.py` shipped. **Result:** stable rollout to H=24 (2.6× train
> horizon), AR-pixel plateaus ~13 dB (nearly flat drift), **beats persistence 24/24** (+1.14 dB), **TF→AR
> gap +1.61 dB and not growing**; pixel-feedback ≫ latent-feedback (carry pixel forward). Blurry (L2) →
> P2/P3. No KV-cache needed (fixed-window rollout is O(n)).

**Goal:** make the model roll out and make drift **visible** as a curve, before trying to fix it.
**Work items:**
- `hvp/predictor/rollout.py`: `free_run(context_frames, n_steps)` that (a) encodes the seed clip, (b) runs the cascade one step, (c) advances the window. Implement **both** modes behind a flag: **pixel-feedback** (decode → re-encode the prediction → predict; robust to the RF masks, adds tokenizer round-trip error each step) and **latent-feedback** (advance the latent window directly; cheaper, needs the causal masks to be sliding-window-correct under generated frames). Reuse `decoder.decode_depth` (`decoder.py:56`).
- KV-cache the existing `SelfAttention` so rollout is O(n) not O(n²) (prerequisite for affordable Self-Forcing later, and for BOT at 2304 tokens). This is additive — the masks and RoPE are already relative/causal.
- Day-0 logging: per-horizon PSNR/SSIM/LPIPS vs frame index h, FVD over rolled clips, the **TF→AR gap** at each h, and a full-horizon rollout strip to wandb.
**New metrics:** per-horizon drift curve, TF→AR gap, rollout-FVD, changed-region per-horizon SSIM.
**GATE G-AR-baseline:** free-run produces a coherent (non-NaN, non-blank) rollout to H = 2× training horizon AND the drift curve + TF→AR gap are logged for both feedback modes. *Pass = the curves exist and the rollout decodes; this exposes the drift, it does not yet have to be small.* (Decision sub-gate: pick the feedback mode with the flatter drift curve to carry forward.)

---

### P2 — Anti-drift training (exposure-bias fix) *(makes the horizon usable)*
**Goal:** close the TF→AR gap by training the model to consume its own imperfect outputs. This is the single highest-leverage change for horizon.
**Work items (cheapest first):**
- **Scheduled sampling / latent-space curriculum:** anneal `p(feed GT latent) → p(feed predicted latent)` over training (Bengio et al. 2015), added as a probability schedule into `compute_oneshot_loss` (`objectives.py:102`). Start here.
- **Noised-history conditioning:** Gaussian-noise the conditioning context during training (GameNGen — context noise is the DOOM-stability fix), generalized to **per-frame independent noise levels** à la Diffusion-Forcing (each latent-frame carries its own noise level → trains the model to denoise partially-corrupted history → buys rollout past the training horizon). Bake the noise level in as a config flag.
- **Short Self-Forcing-style unroll loss:** unroll a few steps during training with the KV-cache from P1 and supervise the whole generated sub-sequence, with stochastic gradient truncation to bound cost.
**New metrics:** TF→AR gap over training (should shrink); per-horizon curve before/after.
**GATE G-AR:** free-run PSNR stays within a fixed margin (target ≤ ~1–1.5 dB, to be calibrated against the v1 single-step lift) of teacher-forced out to H ≥ 2–3× training horizon, AND beats persistence on **changed-region SSIM at every h**, AND the drift curve is non-monotone-collapsing (no run-away softening). *Pass = that bounded-gap-to-3× condition holds on held-out.*

---

### P3 — Distributional head, wired end-to-end *(de-blur + multimodal honesty; horizon prerequisite, not a nicety)*
**Goal:** replace the deterministic L2 path with a sampled head so rolled steps stay on-manifold and multimodal futures stop averaging to mush.
**Work items:**
- `hvp/predictor/heads.py` with `l2 | gmm(K=2–4 NLL) | diffusion_mlp`. **Critically, wire `sample()` through everything** — `variance_explained`, the pixel/AR decode, and the rollout loop — not just the loss term (the v1 `gaussian` path's bug is that it only touches the loss).
- Best-of-N at eval: generate N samples, report Density/Coverage (or Precision/Recall) and a best-of-N PSNR so single-sample MSE stops rewarding the mean.
- Calibration logging: NLL + coverage for the GMM.
- Pair with P2: a *sampled* (not mean) prediction is what keeps rollout on-manifold across steps.
**New metrics:** best-of-N PSNR/SSIM, Density/Coverage, GMM NLL + coverage calibration.
**GATE G-head:** best-of-N beats single-sample MSE by a clear margin AND GMM calibration is sane (coverage near nominal) AND the rolled-out (P2) drift curve with the distributional head is no worse — ideally flatter — than the L2 head. *Pass before any SO(3).*

---

### P4 — Hierarchy-anchors-horizon + multi-seed rigor *(the differentiating claim; must come after P1–P3)*
**Goal:** prove C2 — that the hierarchy *earns its name on the horizon axis* — and put error bars on it.
**Work items:**
- Coarse-stage stabilization for rollout: stronger noised-conditioning / history-guidance on the coarse "plan" (it is the slow anchor); **scale-level scheduled sampling on the parent hidden** (anneal `p(use GT parent) → p(use predicted parent)`) so BOT/MID learn robustness to an imperfect parent — train-what-you-test. Optional History-Guidance/DFoT on a flexible-length partially-noised history.
- Headline A/B at horizon: **cascade-on vs cascade-off rollout drift**, ≥3–5 seeds, mean±std, paired test. Also re-run the single-step cascade-lift A/B multi-seed (resolves G7).
- Ablations: `detach_parent` on/off (quantify the cross-stage-gradient tradeoff under rollout); wider-TOP (≥2×2) or adaLN-Zero for the 1×1-TOP degeneracy.
**New metrics:** per-horizon drift, cascade-on minus cascade-off, with confidence intervals.
**GATE G-hier-horizon:** cascade-on rollout drifts **significantly slower** than cascade-off (non-overlapping CIs on the area-under-drift-curve, or a paired test p < 0.05) over ≥3 seeds. *This is the dissertation's money result — pass = hierarchy buys horizon, with error bars.*

---

### P5 — SO(3) + multimodal eval *(the multimodality that justifies the distributional head)*
**Goal:** stress the system on genuinely multimodal dynamics (self-occlusion / axis-and-depth ambiguity) with the now-distributional head + best-of-N.
**Work items:**
- Unlock the dataset's Z-axis lock to full SO(3) (a renderer change in the shape toolbox → a new cache hash; the loader already keys on hash so the cross-split stays honest).
- Run the full P1–P3 stack on SO(3); the coarse stage must **not** re-blob.
- Optionally interpose harder-but-still-synthetic regimes first (translation, multi-object, occlusion) as cheaper multimodality controls.
**New metrics:** best-of-N + Density/Coverage + FVD on SO(3); coarse-stage erank/std (re-blob canary).
**GATE G-SO(3):** with the distributional head + best-of-N, prediction beats single-sample MSE AND the coarse stage holds effective rank (no re-blob) AND the P4 hierarchy-anchors-horizon result still holds. *Must come after P3/P4 or it will just reproduce the blob.*

---

### P6 — Efficiency + scaling path *(do once P1–P3 define the needed horizon/resolution)*
**Goal:** make the rollout-training loop and higher resolution affordable.
**Work items:** windowed/local (+ dilated/strided) BOT attention behind a flag, verified VE-parity vs dense on the current data; combine with the P1 KV-cache; optional wavelet/space-to-depth front-end (WF-VAE/Cosmos). Throughput-vs-resolution benchmark.
**New metrics:** it/s, tokens/s, VE-parity delta vs dense, throughput-vs-resolution curve.
**GATE G-eff:** windowed BOT preserves cascade lift and recon PSNR within tolerance at a clear it/s win (target ≥1.5–2×). *Pass = same science, materially faster.*

---

### P7 — Reversibility integration *(dissertation goal #3; after the core narrative is solid)*
**Goal:** bring back the stated reversibility goal without destabilizing the horizon result.
**Work items:** design doc identifying the conditioning structure (reuse the cascade? bidirectional attention?); `hvp/reversibility.py` behind a config flag with an invertibility round-trip golden; integrate after C1/C2/C3 hold so it cannot mask the core result.
**GATE G-rev:** invertibility round-trip passes the golden AND enabling reversibility does not regress recon or the P4 horizon result past tolerance.

---

### P8 — Joint-training escalation *(gated; run last)*
**Goal:** test the strongest form of the thesis — predictive gradient organizing the latent for predictability — without re-triggering the recon collapse.
**Work items:** anneal λ 0→target over 5–10k steps with VICReg active; compare to the detached baseline.
**GATE G-joint (hard, three-way):** MID VE rises vs the detached baseline **AND** recon does not regress past tolerance **AND** effective rank holds. *Any one fails → keep detached; the result is reportable either way.*

---

**Phase dependency summary:** P0 → P1 → P2 → P3 → P4 → {P5, P6 in parallel} → P7 → P8. P1–P4 are the dissertation spine (horizon + the hierarchy-buys-horizon claim + the distributional head + rigor). P5 onward are validity/scale/optional-goal extensions.

---

## 5. Eval upgrade — the metric suite and when it lands

The v1 suite (recon PSNR/SSIM, per-stage effective rank + std, VICReg var/cov, predictive VE all/changed, cascade lift, single-step pixel-vs-persistence) is **kept** as the training-health and collapse-canary layer. Add, in phase order:

| Metric | What it catches | Lands in | Why then |
|---|---|---|---|
| **Changed-region PSNR/SSIM** | persistence gaming the static background | **P0** | cheap; immediately makes the existing single-step numbers honest |
| **LPIPS** (perceptual, `lpips` pkg) | L2 blur that PSNR misses | **P0** | needed the moment we discuss sharpness/blur |
| **FVD** (I3D, Fréchet Video Distance) | video realism / distribution-level quality | **P0 wired, P1 reported** | the field-standard video bar; meaningful over rolled clips |
| **Per-horizon PSNR/SSIM/LPIPS curves** | drift / softening over h | **P1** | the field-standard drift diagnostic; the core horizon evidence |
| **TF→AR gap** (teacher-forced minus free-run at each h) | exposure bias | **P1** | the single number the predecessors couldn't close |
| **Rollout-FVD per-horizon** | realism collapse over long rollout | **P1** | distinguishes "blurry but stable" from "drifting off-manifold" |
| **Best-of-N PSNR/SSIM** | mode-averaging; demotes single-sample MSE | **P3** | only meaningful once the head is distributional |
| **Density / Coverage (or Precision/Recall)** | sample diversity vs fidelity | **P3** | the distributional-quality pair under multimodality |
| **GMM NLL + coverage calibration** | head calibration | **P3** | sanity-checks the distributional head |
| **Per-stage pixel decode** | attribute pixel quality to stages | **P4** | supports the hierarchy-attribution claim |
| **Multi-seed mean±std + paired test** | within-noise folklore | **P4** | turns every headline lift into a defensible result |

**Eval discipline:** the moment the head is distributional (P3) or the data is multimodal (P5), single-sample MSE/PSNR is demoted to a *sanity floor only* — best-of-N + distributional metrics + FVD become the scorecard. Borrow the short/medium/long horizon bucketing convention (scaled to this toy: e.g. 1–9 / 10–18 / 19+) for reporting drift.

---

## 6. Risks and de-risking

| Risk | Likelihood | De-risk |
|---|---|---|
| **AR rollout diverges/blobs** (predecessors' failure; +0.151 gap unclosed by light noise) | High | Instrument drift **before** fixing (P1), then layer the *stack* of fixes (scheduled sampling → per-frame noised conditioning → short Self-Forcing unroll), each behind a flag, each measured against the P1 baseline curve. Don't bet on one lever. |
| **Cascade amplifies drift instead of anchoring it** (detached coarse→fine chain, no correction) | Medium-High | P4 directly targets it: stabilize the coarse "plan", scheduled-sample the parent hidden, ablate `detach_parent`. If cascade-on never beats cascade-off on drift, the honest fallback is "hierarchy helps single-step, is neutral on horizon" — still reportable, just a weaker claim. Decide via G-hier-horizon, not vibes. |
| **GMM underfits / diffusion head too slow** | Medium | Tier it: GMM (K=2–4) is the cheap first option; MAR per-token diffusion-MLP only if GMM's best-of-N lift is <~+0.5 dB. Calibration logging tells you which way before committing. |
| **Pixel-feedback rollout injects tokenizer round-trip error; latent-feedback collides with RF masks** | Medium | P1 runs both as a measured A/B and picks the flatter drift curve — the mode choice is data-driven, not assumed. |
| **BOT compute makes rollout-training infeasible to iterate** | Medium | KV-cache lands in P1 (not deferred to P6); windowed BOT (P6) is the second lever. Keep horizon-training experiments at 64×64 / short H until both land. |
| **SO(3) re-blobs the coarse stage** | Medium | Hard-ordered after P3/P4; G-SO(3) gates on coarse erank holding. Optionally ramp through translation/occlusion first. |
| **Cascade lift was noise** | Medium | P4 multi-seed with CIs/paired test resolves it directly; if it's noise, the falsification + horizon results still stand. |
| **Joint training re-collapses recon** | Known (happened) | Run last, behind the three-way G-joint gate; detached stays the default and the reportable result. |
| **Synthetic-only undercuts external validity** | High (inherent) | Frame the dissertation as a *controlled study of hierarchical predictability and horizon*, not a generation result; add one real-video stress slice (P8/stretch) as an external-validity check, not the headline. |

---

## 7. Stretch / optional

- **SO(3) → richer dynamics → real video.** Beyond P5's SO(3): translation + multi-object + occlusion as multimodality controls, then a small real-video stress slice (e.g. a DMLab/Kinetics/Minecraft subset behind the same `(C,T,H,W)` interface) as an external-validity check. Not the headline — a generalization probe.
- **Reversibility (goal #3) beyond P7.** Full invertible-coupling integration with a reversibility-specific eval (forward-then-inverse reconstruction over the horizon), if the core horizon narrative leaves room.
- **Scaling.** Higher resolution (128²+) once windowed BOT + wavelet front-end land; larger param count / longer schedules; multi-GPU. This is laptop-scale by design — scale is a "future work" paragraph, not a chapter, unless cluster time appears.
- **Joint-training escalation as a positive result.** If G-joint passes, "predictive gradient organizes the latent for predictability without collapsing recon" becomes a second contribution; if it fails, the detached recipe's stability is itself the reportable finding.
- **Flexible-history / interactive conditioning (DFoT/Genie-style).** Condition on a variable-length partially-noised history for very-long rollout; a natural extension of P2/P6 if horizon becomes the headline.
- **Distributional-head escalation.** If GMM and diffusion-MLP both plateau, a per-token flow-matching head is the next rung — only if the eval demands it.

---

### Standing notes carried into V2 (verified against source)
- **The `gaussian` head is a stub:** `predictive_loss` computes NLL but `variance_explained` and every decode path use `mu = pred[..., :C]` only — there is no sampling anywhere. P3 must wire `sample()` through, not just add a loss.
- **Cross-split IS real:** train/val use separate independently-rendered caches (`train_hash` `e075debc82fc` / `val_hash` `9dffa66fdc0e`). (An earlier review flagged it as an index split — that was a *stale docstring*, now fixed; the code was always using separate hashes.)
- **Checkpoints are not resumable:** `_save` stores only `model`, `cascade`, `topo`. No optimizer/RNG/step → multi-seed and long rollout-training need P0's full resumable checkpoint first.
- **The masks, RoPE, and SDPA are already rollout-ready:** relative 3D RoPE, boolean causal masks as cached buffers, qk_norm SDPA — KV-caching (P1) is additive, not a rewrite.
- **The masks, RoPE, and SDPA are already rollout-ready:** relative 3D RoPE, boolean causal masks as cached buffers, qk_norm SDPA — KV-caching (P1) is additive, not a rewrite.
