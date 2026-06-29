# Project handoff — hierarchical video prediction (DLH / CW-VAE / reversible renderer)

> Self-contained context transfer for a fresh session. Read this + `memory/MEMORY.md` first.
> Branch `reproduce/dlh`. Date: 2026-06-29.

---

## 0. TL;DR / current state
- Two backbones: **CW-VAE** (sibling repo `../cwvae`, `cwvae_torch.py`, branch `pytorch-port-mmnist`, ckpt
  `../cwvae/logs_torch/cwvae/last.pt`) and **DLH** (this repo `hvp_dlh/`, ckpt `logs_dlh/baseline_iter8000.pt`).
  New work lives in **`hvp_rev/`**. Data: `../cwvae/data_mmnist/{train,test}.npy` (train 3000×100, test 100×1000, 64×64).
- **Biggest result this session: a distribution-matching (adversarial) RENDERER works** — fixes the VAE blur that
  blocked everything. Validated on CW-VAE (3 seeds). The recipe and findings are below.
- **Decision (user): go back to DLH.** The renderer reopens the DLH-vs-CW-VAE question (sharpness is no longer a
  CW-VAE-only advantage; DLH's long-horizon edge remains). New roadmap (user's priority order) in §5.
- venv: `venv/Scripts/python.exe` (has torch+cuda, lpips, torchvision/InceptionV3 cached, scipy, matplotlib).
  GPU: RTX 4060 Ti 16 GB. **DISK: ~12 GB free, 97% full — keep checkpoints weights-only + atomic; abort <3 GB.**

---

## 1. The journey & key findings (so you don't relearn them)
1. **DLH vs CW-VAE (matched eval, `scripts/match_eval.py`, `logs_dlh/match_eval.json`):** CW-VAE renders ~2.4×
   **sharper** (Laplacian-var) and better LPIPS (0.265 vs 0.334) at **tied fg-MSE**. Proven the blur is the
   **Gaussian-L2 objective**, NOT the latents: a perceptual decoder on FROZEN codes reaches GT-level sharpness
   (C0); mean-mode + teacher-forced-recon tests rule out noise/min_std/dynamics.
2. **Long-horizon (`scripts/a0_longhorizon_ci.py`, `logs_dlh/compare_curves.json`):** **DLH already BEATS CW-VAE
   long-horizon** — its fg-MSE crosses below CW-VAE at frame ~17–24 and stays below through 150 (bounded-drift /
   identity, the adaptive clock working). CW-VAE wins short-horizon. Gap is small (~0.01–0.02 fg-MSE), clip-CI
   significant but single-training-seed.
3. **E1 (`hvp_rev/train_rollout.py`, `docs/RESULTS_E1.md`):** on-the-fly rollout training with an **MSE** objective
   **FADES to the mean** (lap-var 0.0073→0.0017) — distortion-optimal under uncertainty. Mechanism is fine; the
   objective must be **distribution-matching**, not pixel.
4. **Step 1 — adversarial renderer (`hvp_rev/train_renderer.py`, `docs/RESULTS_STEP1.md`): ✅ SUCCESS.**
   Validates **distribution-matching > perceptual > pixel**. See §3 for the recipe + results.
5. **Probe:** training the renderer on free-run codes did NOT improve open-loop → the open-loop limit is the
   **dynamics, not the renderer**. So the renderer is done; the dynamics (clock/rollout) are the next frontier.
6. **Methodology lessons (do not relearn):**
   - **MMNIST aggregate SSIM/PSNR are background-dominated** (black scores 0.76) → gate on **fg-MSE / fg-LPIPS /
     FID**, never SSIM. (`memory/mmnist-ssim-is-background-dominated`.)
   - **Per-frame FID (Inception, out-of-domain) is GAMED by sharpness** when output is incoherent → never trust
     FID alone; pair with the visual + a coherent recipe. **LPIPS-only games fg-LPIPS** (best fg-LPIPS, worst FID).
   - The **temporal hierarchy / adaptive clock is ~null on MMNIST** (it has no real multi-timescale events);
     entropy-based clocking is dead here (KL-surprise ρ≈0.16, weak). Validate dynamics claims on **KTH/events**.
   - **Process that worked:** every step = plan → multi-agent red-team → implement → smoke → train → gate, with
     metric+sample-strip tracking and disk/GPU guards. Keep it.

---

## 2. Code & doc map
- `hvp_dlh/` — DLH port (model.py, modules.py, config.py, train.py, eval.py, gate.py, structure.py). Decodes from
  a 200-dim top-down context `c` via `img_dec` (stochastic-only → the blur).
- `../cwvae/cwvae_torch.py` — CW-VAE (pristine; decodes from `[s,det]` 220-dim). `eval_torch.py`, `train_torch.py`.
- `hvp_rev/` — NEW: `train_renderer.py` (3-arm renderer trainer), `gate_renderer.py`, `modules_gan.py`
  (RendererDecoder, **WarmStartG**, PatchGAN, adaptive_lambda), `fid.py` (InceptionV3 FID, validated), `codes.py`
  (`teacher_forced_codes`, `free_run_codes`), `train_rollout.py` (E1 rollout trainer), `gate_rollout.py`,
  `reversible_temporal_proto.py` (reversible block, P1–P4 proven).
- `scripts/` — match_eval, a0_longhorizon_ci, c0_decoder_headroom, c0b_perceptual_sweep, e0_exposure_gap,
  e1_clock_premise, e2_rollout_drift, e1_smoke, run_*.sh.
- `docs/` — DLH_EXPLAINED, RESULTS_DLH_REPRODUCTION, PLAN_DLH_SURPASS_CWVAE, PLAN_CWVAE_IMPROVEMENTS,
  PLAN_SHARP_RENDERER, PLAN_STEP1_ADV_RENDERER, RESULTS_STEP1, PLAN_REVERSIBLE_ROLLOUT, PLAN_STEP3_DM_ROLLOUT.
  (`../cwvae/CWVAE_EXPLAINED.md` is the CW-VAE reference.)
- **TempVerseFormer** = the user's own paper ("Temporal Modeling with Reversible Transformers", RS rs-6293520 /
  Springer NCA 2025): RevNet additive coupling along TIME + time-agnostic BPTT. The reversibility core.

---

## 3. The validated renderer recipe (reuse for DLH)
**Problem it solves:** VAE decoders trained with Gaussian-L2 predict the blurry mean. A distribution-matching
(adversarial) term renders sharp, in-distribution frames.

**Recipe (`hvp_rev/train_renderer.py`):**
- `WarmStartG`: init the renderer from the **pretrained (coherent) decoder** — the GAN *sharpens*, doesn't build
  structure from scratch (fresh-init collapsed/textured).
- Loss = **foreground-weighted L1** (`--fg-weight 8`, pins the sparse digit) + **LPIPS** + **adaptive-λ PatchGAN**
  (VQGAN adaptive weight from the decoder's last conv; D warm-up 500; spectral-norm; hinge).
- lr 5e-5, ~6k steps, batch 64. VRAM ~5 GB, ~25 it/s.

**Result (CW-VAE, 3 training seeds), open-loop:** gan **FID 69.8±3.1** vs baseline-blur 101, lpips-only 125,
fresh-L2 114; **sharpest** (lap-var 0.068 ≈ GT 0.056), coherent digits; cost = fg-MSE (perception-distortion
Pareto). Strips: `logs_render/s1_compare_strip.png`, `s1wfg_strip.png`.

**Note:** C0 headroom was measured on **DLH** codes — so this recipe is expected to transfer to DLH directly
(decoder dim 200 not 220; warm-start `hvp_dlh` `img_dec`; codes from `hvp_dlh` teacher-forced `c`).

---

## 4. Key numbers (open-loop, held-out, fg-metrics)
| model | FID ↓ | fg-LPIPS ↓ | fg-MSE | lap-var ↑ | note |
|---|---|---|---|---|---|
| GT | 0 | — | — | 0.056 | ceiling |
| CW-VAE (blur) | 101 | 0.300 | 0.425 | 0.008 | sharp-poor, short-horizon-strong |
| DLH (blur) | ~similar | ~0.33 | ~0.43 | 0.003 | blurrier, **long-horizon-strong** |
| **CW-VAE + renderer (Step 1)** | **70** | 0.24 | 0.49 | **0.068** | sharp+coherent, fg-MSE cost |

---

## 5. DLH improvement roadmap (USER'S PRIORITY ORDER, elaborated)

### Phase 0 — Stabilize vanilla DLH to CW-VAE parity (DO FIRST)
**Goal:** a clean, stable, precise DLH base before any novelty — matching CW-VAE on training stability + precision
(fg-MSE/SSIM), and (via the renderer) on sharpness.
**Steps:**
1. **Matched-compute, stable retrain.** Train faithful DLH to CW-VAE's budget (~18k steps, batch 50, seq 100) with
   the committed fixes (KL-economy static-prior = previous posterior; `force_tick` dense warmup; per-level
   free-nats off). Confirm: no high-β collapse, top-KL stays >0, clock survives. Gate vs CW-VAE on **fg-MSE @
   short + long horizon** (DLH should match short, win long). Tag `dlh_baseline_matched`.
   - Watch: DLH auto-collapsed to 2 levels on MMNIST (legitimate — minimum-levels). 2 levels is acceptable.
2. **Sharpness parity via the renderer (the test the user agreed to).** Apply the §3 recipe to DLH's frozen codes
   (warm-start `hvp_dlh.img_dec`, 200-dim `c`, fg-weight + adaptive-λ PatchGAN). **Head-to-head: DLH+renderer vs
   CW-VAE+renderer** on sharpness (FID/fg-LPIPS) AND long-horizon (fg-MSE). *Hypothesis: DLH+renderer wins BOTH*
   (DLH's long-horizon edge + now-sharp). This is the cheapest, highest-value first experiment (~20–30 min;
   infra exists; C0 headroom already on DLH). **If it holds, DLH becomes the backbone.**
**Exit gate:** DLH (±renderer) is as stable + precise as CW-VAE and at least ties it on the combined
sharpness+long-horizon picture.

### Phase 1 — Per-pixel state-update decisions (spatial adaptive clock) — the user's main new idea
**Idea:** DLH currently decides update-vs-persist **per level, per frame** on **vector** latents. Make it
**per-pixel/per-location** (update the state only where something happens) — more sensible for video (motion is
local).
**Concrete design:**
1. **Spatial-latent refactor (uniform resolution, NOT a pyramid — the pyramid was a NULL):** make each level's
   stochastic state a map `s^n ∈ (B,C,h,w)` (e.g. 16×16); `ConvGRU` temporal path; 1×1-conv Gaussian heads;
   conv top-down/bottom-up. **Memory gate:** forward+backward one seq100 batch < ~14 GB before any run.
2. **Per-pixel `e`:** `e^n_t ∈ (B,1,h,w)` from the same hard-KL model-selection (per-location
   `KL(post‖change) < KL(post‖static)`) or a learned 1×1-conv gate; state update `s = e·s_change + (1−e)·s_prev`
   (Hadamard). Nested constraint = same-location across levels. L1/entropy reg on the e-map (sparsity).
3. **Cheap pre-check FIRST (like E1's clock-premise, but spatial):** offline, does the per-location KL map
   visibly **segment the moving digits**? If it's flat (like the dead per-frame entropy), per-pixel gating is a
   null on MMNIST → move to KTH before building the full gate.
**Honest caveats:** the adaptive clock is ~null on MMNIST (weak event signal); per-pixel may be a null here too —
**plan to validate on KTH / an events dataset.** Per-pixel's compute *savings* only materialize with a hard
sparse inference path. First prove it doesn't hurt quality (matches vanilla DLH), then show it helps where events
are spatially localized.

### Phase 2 — Reversibility integration (TempVerseFormer)
**Idea:** reversible temporal core → constant-memory long-rollout BPTT (the dissertation's reversible contribution).
**Concrete (from `docs/PLAN_REVERSIBLE_ROLLOUT.md` + `reversible_temporal_proto.py`, which proved P1–P4):**
- Distinguish **depth-reversibility** (RevViT, layers) from **time-reversibility** (the valuable axis here).
- **Scope a reversible block to ONE level's same-rate recurrence; keep the sum-merge/multi-rate tiling OUT** of the
  reversible path (the clockwork is NOT a single invertible chain → reversal would be silently wrong).
- Handle the **no-forgetting theorem** (MacKay): a perfectly reversible recurrence can't forget — resolve via
  **store-the-noise (ε)** + **reversal-with-forgetting (store the gate bits)** + **spectral/Lipschitz coupling**
  (float32 reverse-replay drift exploded to ~1e3 over T=64 in the proto → Lipschitz control is mandatory).
- **Require a gradient-equivalence check vs true BPTT on the real backbone** before trusting constant-memory grads;
  use **gradient checkpointing** (correctness-safe, sub-linear-in-M memory) as the fallback.
- **Frame reversibility as a MEMORY enabler, not a stability claim** (measure it: memory-vs-window curve). The
  P1–P4 no-forgetting figure + memory curve are bankable deliverables even with checkpointing.

### Phase 3 — Everything together (the ideal end state)
DLH (stable, spatial **per-pixel** adaptive clock, **reversible** temporal core) + the **sharp renderer**
(+ optionally a **distribution-matching rollout** objective, `docs/PLAN_STEP3_DM_ROLLOUT.md` — but note its
red-team: needs a temporal/context-conditioned discriminator, horizon-resolved gate, and KTH validation; high
null-prior). The grand integration: hierarchical + adaptive + reversible + sharp + long-horizon.

---

## 6. Immediate next action (for the new session)
Run **Phase 0 step 2**: apply the §3 renderer recipe to DLH (warm-start `hvp_dlh.img_dec`, 200-dim codes) and do
the **DLH+renderer vs CW-VAE+renderer** head-to-head (sharpness + long-horizon, gated). Then Phase 0 step 1
(matched stable retrain) in parallel/after. Keep the plan→red-team→implement→gate discipline; gate on
fg-MSE/fg-LPIPS/FID (never SSIM); watch disk (<3 GB abort) and GPU.

## 7. Risks / open questions carried forward
- DLH's long-horizon edge is small + single-seed → needs ≥3 training seeds for a CI.
- Per-pixel & reversibility are likely **null on MMNIST** (hierarchy/clock null there) → budget a **KTH** port.
- Reversibility on the clockwork is correctness-hazardous → checkpointing fallback; decouple the deliverable.
- FID is out-of-domain on MMNIST (relative only, gamed by incoherence) → consider FVD (I3D) for a defensible
  temporal-coherence claim; pair with fg-LPIPS + visual always.
