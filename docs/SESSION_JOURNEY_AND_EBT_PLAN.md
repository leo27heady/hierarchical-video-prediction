# Master reference — full journey, experiments, results, assets & plan

**Branch:** `research/ebt-hierarchical-video`  ·  **Span:** 2026-06-30 → 2026-07-01  ·  **Hardware:** 1× RTX 4060 Ti 16 GB
**Interpreter:** `venv/Scripts/python.exe`  ·  **wandb:** offline (`WANDB_MODE=offline`, runs under `wandb/`)
**Project root:** `C:/Users/leoni/Documents/projects/hierarchical-video-prediction`

This is the single, comprehensive, pick-up-cold reference for the whole session: every direction explored, every
experiment with its numbers, every file/path/link, and the open options. Companion verdict/spec docs are linked inline.

---

## 0. TL;DR (state at hand-off)

- We explored many architectures. **Central empirical lesson:** on MMNIST-class data (determined-or-unknowable, no
  *hard-but-computable* middle regime) **every added dynamics mechanism nulls** — adaptive clock, spatial gate,
  reversibility-for-stabilization, CW-VAE clock, EBT System-2. We built a purpose-made **looping off-screen-memory
  testbed** to break this, which partially worked.
- **Robust, in-hand results (shippable):**
  1. **Ch. D theory** — a provably-invertible **stochastic** recurrence with exact constant-memory gradients
     (grad-cos vs BPTT = 1.000000) + the *invertibility ≠ stability* characterization. Measured, non-nullable.
     [`docs/THEORY_CHAPTER_D.md`](THEORY_CHAPTER_D.md).
  2. **Hierarchy helps** — the coarse-to-fine latent decomposition is a real, mechanism-agnostic win on the looping
     testbed (D2 oracle probe: +75% fine-prediction; all hierarchical predictors beat persistence).
  3. **Mechanism comparison** — a controlled regression-vs-EBT-vs-diffusion study: on 16 GB-tractable data **all three
     denoisers blur** (none > 55% of GT sharpness); the blur is **task/data + scale-inherent, not EBT-specific**
     (diffusion, MilliVid's own mechanism, blurs too — only marginally sharper).
- **Open decision (last question, user dismissed → awaiting):** ship hierarchy+mechanism-study+Ch.D / change the task /
  push scale within 16 GB / ship Ch.D only. See §10.

---

## 1. The journey (chronological, every direction + verdict)

Each direction was investigated with multi-agent review + adversarial red-team, and gated honestly.

### 1.1 RHCP — Reversible Hierarchical Clockwork Predictor
CW-VAE-class spatial pyramid (16/8/4[/2], geometric clock k=2ˡ) + per-level RevNet additive coupling along time +
identity-HOLD on non-tick frames (multi-rate by lossless holding, not forgetting → sidesteps MacKay no-forgetting).
**Built + validated** but the model **faded** on MMNIST (see §2.1, §2.6). Spec: [`docs/RHCP_ARCHITECTURE.md`](RHCP_ARCHITECTURE.md).
Code: `hvp_rev/rhcp.py`, `hvp_rev/rollout_train.py`, `hvp_rev/rhcp_validate.py`.

### 1.2 RHCP experiment plan (OpenSTL table / long-horizon drift)
Full plan verified by a 7-agent asset+red-team pass: baselines (CW-VAE, DLH) and the fair harness already on disk;
canonical `mnist_test_seq.npy` present. 3 blocking fixes folded in (shared objective; route both drift arms through
reverse-replay; fp32 inverse). Honest expectation: RHCP below the OpenSTL table from scratch; hierarchy null on MMNIST;
memory-win Pareto-neutral vs checkpointing. Docs: [`docs/RHCP_EXPERIMENT_PLAN.md`](RHCP_EXPERIMENT_PLAN.md),
[`docs/RHCP_RESULTS.md`](RHCP_RESULTS.md).

### 1.3 MilliVid + CW-VAE clock → **REDUNDANT**
MilliVid's `num_frames_per_level=[3,12,48,192]` IS already a factor-4 temporal clock; the 2-axis space×time hierarchy is
**MSPred (BMVC 2022)**, not new; the real lever is MilliVid's decoder, not the clock. Verdict:
[`docs/MILLIVID_CWVAE_VERDICT.md`](MILLIVID_CWVAE_VERDICT.md).

### 1.4 MilliVid + reversibility (AR stabilization) → **CATEGORY ERROR**
Invertibility ≠ stability (additive coupling det-Jac=1 but unbounded singular values → the RHCP fade is EXPECTED).
MilliVid is stateless per-step diffusion → no recurrence to make reversible. Self-Forcing (NeurIPS 2025) owns
rollout-stabilization and does it *without* deep BPTT. Verdict: [`docs/MILLIVID_REVERSIBILITY_VERDICT.md`](MILLIVID_REVERSIBILITY_VERDICT.md).

### 1.5 Theory + hierarchy-science (fallback that became the backbone)
Ch. D theory **done + verified** (§2.2). Purpose-built **looping testbed** created (§3.1). A learned token-pyramid AE
had an overfit bug (parked; replaced by a plain conv-AE).

### 1.6 MilliVid + EBT (the current line) — energy-based denoiser
Replace MilliVid's diffusion denoiser with an **Energy-Based Transformer** (arXiv 2507.02092): energy-descent inference
instead of diffusion. Investigated (3 agents) → gates PASSED (§2.3–2.5) → hierarchical build + diffusion control (§2.6).
Direction memory: [`memory/ebt-direction.md`]. Verdict this session: hierarchy helps, but the denoiser (EBT *or*
diffusion) can't overcome the blur on 16 GB-tractable data.

---

## 2. Experiments & results (with exact numbers)

### 2.1 RHCP core validation — `python -m hvp_rev.rhcp_validate`
| property | result |
|---|---|
| Block invertibility (real conv stage), fp64 / fp32 | 1.3e-15 / 5.4e-4 |
| Assembled 3-/4-level clock-hold reconstruction (fp64) | 1.3e-13 / 2.0e-13 (exact), HOLD + co-evolving parent |
| Reverse-replay grads == BPTT @ K=100 | **grad-cos 1.000000** |
| OpenSTL metrics vs reference | bit-identical (mse/mae/ssim diff 0.0e0; psnr 6e-4) |

### 2.2 Ch. D theory — `python -m hvp_rev.theory`
| test | result |
|---|---|
| T1 block invertibility (unconstrained / Lipschitz) | 5.5e-15 / 3.1e-5 |
| T2 **stochastic** store-the-noise invertibility | exact iff eps stored (8.4e-13); wrong eps breaks (10.9) |
| T3 **invertibility ≠ stability**, K=100 perturbation growth | 333× (unconstrained) → 26,600× (Lip(F)≤1) → 1.5× (Lip≤0.1) → 1.0× (Lip≤0.02) |
| T4 reverse-replay == BPTT (Lipschitz+stochastic) | grad-cos 1.000000 |

**Panel A — memory frontier** (`python -m hvp_rev.mem_frontier --levels 3 --batch 32`, real train loop + Adam):

| K | BPTT (MB) | checkpoint | reversible | BPTT ms |
|---:|---:|---:|---:|---:|
| 40 | 12162 | 1471 | 1091 | 740 |
| 80 | 23305* | 1637 | 1125 | 43797* |
| 120 | **OOM** | 1801 | 1152 | — |
| 160 | **OOM** | 1967 | 1176 | — |

\*K=80 BPTT spills to shared RAM (WDDM) → 44 s/step (~60× slower); hard OOM at K=120. Reversible flat ~1.2 GB.
Honest claim: constant-memory *capability* vs vanilla BPTT (OOMs at K=120); **~Pareto with checkpointing** (not a win).

### 2.3 CW-VAE / DLH long-horizon baselines — `python -m hvp_rev.compare --mode long --ctx 36`
(cwvae test, H=20, foreground metrics — aggregate MSE/SSIM are background-dominated, not for ranking)

| model | fg-MSE | centroid-L2 (px) | IoU | fg-LPIPS |
|---|---|---|---|---|
| persistence | 0.554 | 23.13 | 0.108 | 0.276 |
| DLH | 0.346 | 17.02 | 0.314 | 0.278 |
| CW-VAE (mean) | 0.285 | 10.29 | 0.414 | 0.188 |
| CW-VAE (best-of-5) | 0.270 | 9.31 | 0.430 | 0.189 |

### 2.4 EBT feasibility — Stage 0 / 0b
- **Stage 0** (`python -m hvp_rev.ebt --gate`, toy): the unrolled 2nd-order energy objective **trains stably** on the
  4060 Ti; energy converges monotonically. Multimodality is EBT's weak spot (mean-seeks on a bimodal toy).
- **Stage 0b** (`python -m hvp_rev.ebt_video`, real conv-AE latents): **trains stably** with the faithful recipe
  (SmoothL1, S1 per-step loss, per-element α, dedicated high-LR α group). On easy next-frame: competitive with
  regression but mildly blurs.

### 2.5 EBT value-prop — Stage 1 / 1b (`python -m hvp_rev.ebt_stage1`)
- **Stage 1 (S1 recipe): System-2 NULL** — fg-MSE flat across 1→32 inference steps (converges in ~2 steps).
- **Stage 1b (S2 recipe = replay buffer + no-detach + final-loss): System-2 WORKS** — think-longer reduces hard-frame
  error **25%** (1→0.038, 2→0.013, 4→0.010, plateau). **Honest nuance:** benefit *saturates at convergence* (~4 steps;
  flat 4→32); sharpness edge over regression marginal (0.0148 vs 0.0122, GT 0.0294). → S1 vs S2 recipe is decisive.

### 2.6 Minimal MilliVid+EBT (hierarchical) — the main build + control
- **D2 oracle-coarse diagnostic** (`python -m hvp_rev.ebt_diag`) — **GREEN**: knowing the ground-truth coarse latent
  improves fine-prediction fg-MSE **75% (all) / 77% (hard)** (0.0092→0.0023); the coarse is **predictable** (normalized
  residual 0.306). → the hierarchy has a real, usable job on the looping testbed.
- **Hierarchical EBT** (`python -m hvp_rev.hier_ebt`, consistency-coupled, S2, 8k steps):
  System-2 sweep converges by ~4 steps (1→0.0294, 4→0.0102). Final: **fg-MSE 0.0101 (all) / 0.0146 (hard), sharpness
  0.0120.** Sample grid `logs_rhcp/hier_ebt_samples.png` — blurry digit-blobs.
- **Hierarchical regression baseline** (same run): **fg-MSE 0.0098, sharpness 0.0106**. Persistence 0.0151.
- **Hierarchical DIFFUSION control** (`python -m hvp_rev.hier_diffusion`, v-pred + DDIM, 10k steps) — the "is it EBT?"
  test: single-sample fg-MSE 0.0133 / sharp 0.0157; **best-of-8 oracle fg-MSE 0.0110 (hard 0.0155) / sharp 0.0155.**
  Sample grid `logs_rhcp/hier_diffusion_samples.png` — also blurry/fragmented.

**Decisive mechanism-comparison verdict** (fine next-frame, GT sharpness 0.0294):

| mechanism | fg-MSE | sharpness | % of GT sharp |
|---|---|---|---|
| hier-regression (mean) | **0.0098** | 0.0106 | 36% |
| hier-EBT (energy-min) | 0.0101 | 0.0120 | 41% |
| hier-diffusion (best-of-8) | 0.0110 | **0.0157** | 53% |

→ **All blur.** Diffusion marginally sharpest (sampling commits a bit more than deterministic descent), but the limiter
is **data + model scale**, not EBT. No small denoiser reaches GT sharpness on this task.

---

## 3. Data assets (paths)

### 3.1 Looping off-screen-memory testbed (built this session) — `scripts/gen_looping.py`
64px viewport random-walk-pans over a 128px **torus** world (fixed bg texture + sprites at fixed world positions →
content exits and must re-enter = off-screen memory; periodic → long-horizon consistency; actions {-1,0,1} recorded).
Corner-column temporal std **0.125** (vs MMNIST-CIFAR 0.0) — background pans, hierarchy has a job.
- `data/loop_train_3000x100.npy` (+ `_actions.npy`) — (N=3000, T=100, 64, 64) uint8.
- `data/loop_test_500x200.npy` (+ `_actions.npy`) — (N=500, T=200).
- `data/loop_smoke_8x60.npy` — tiny sanity set.

### 3.2 Other datasets
- `data/mmnist_long_test_3000x{17,25,33}_d2_s64.npy` — long MMNIST (T-first), + `pturn0.1/0.15` genuinely-multimodal variants.
- `data/moving_fmnist_long_1400x80.npy` — 2nd dataset (`scripts/gen_moving.py`, OpenSTL-matched).
- `C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/mnist_test_seq.npy` — canonical OpenSTL test (20,10000).
  Also `mnist_cifar_test_seq.npy` (20,10000,64,64,**3**, static-per-clip bg). `moving_fmnist/`, `kth/` also present.
- `C:/Users/leoni/Documents/projects/cwvae/data_mmnist/{train.npy (3000,100), test.npy (100,1000)}` — the CW-VAE/DLH split.

---

## 4. Code inventory (`hvp_rev/`, `scripts/`) — ⭐ = built/edited this session

| file | purpose |
|---|---|
| ⭐`rhcp.py` | reversible core: `RevConvStage` (fwd+inverse, +lipschitz/out_scale), clock-hold `RHCPCore`, motion-aware `PyramidEncoder`, `PixelDecoder`, `RHCP` |
| ⭐`rollout_train.py` | free-running rollout training; `bptt_loss_grads` / `reverse_replay_loss_grads`; gradient-equivalence `gate()`; objectives |
| ⭐`theory.py` | Ch. D verification (T1–T4): stochastic invertibility + invertibility≠stability |
| ⭐`rhcp_validate.py` | STEP 0/1/2: block+assembled invertibility, memory bench |
| ⭐`mem_frontier.py` | Panel A memory frontier (BPTT vs checkpoint vs reversible, real train loop) |
| ⭐`eval_adapters.py` | layout adapters + OpenSTL-identical metrics (bit-verified) |
| ⭐`long_eval.py` | fg-MSE, fg-LPIPS, **per-digit centroid-L2** (red-team-fixed), IoU, bootstrap CI |
| ⭐`compare.py` | RHCP vs CW-VAE(mean/best5)/DLH/persistence; short + long panels |
| ⭐`convae.py` | plain frozen conv-AE (val recon 0.0016) — model `logs_rhcp/convae_loop.pt` |
| ⭐`hier_convae.py` | 2-level hierarchical conv-AE (coarse 4×4 + fine 8×8, per-level decodable) — model `logs_rhcp/hier_ae.pt` |
| ⭐`ebt.py` | EBT core (energy MLP + unrolled 2nd-order + stability recipe) + Stage-0 gate |
| ⭐`ebt_video.py` | EBT on latents; `mcmc` (S1/S2 + replay + y_init); Stage-0b |
| ⭐`ebt_stage1.py` | System-2 sweep + sharpness gate; S1/S2 recipes; replay buffer |
| ⭐`ebt_diag.py` | D1/D2 diagnostics (oracle-coarse ceiling + coarse predictability) |
| ⭐`hier_ebt.py` | **consistency-coupled hierarchical EBT** + hier-regression baseline + sample-image monitor |
| ⭐`hier_diffusion.py` | hierarchical **diffusion** control (v-pred, DDIM, best-of-N) |
| ⭐`viz.py` | `save_samples()` — context\|pred\|GT PNG grids |
| `tokenizer.py`, `train_tokenizer.py` | learned token-pyramid AE (parked, overfit bug) |
| ⭐`scripts/gen_looping.py` | the looping-testbed generator |
| ⭐`scripts/gen_moving.py` | long moving-MNIST/FashionMNIST generator |
| `scripts/a0_longhorizon_ci.py` | (pre-existing) fair long-horizon fg-MSE/LPIPS + bootstrap harness for CW-VAE/DLH |

Prior-session assets reused: `hvp_dlh/` (DLH port), `logs_dlh/baseline_iter8000.pt` (DLH ckpt), `logs/e2e_convnext_m8/`
(flat gSTA base MSE 27.21), `scripts/reversible_temporal_proto.py`, `hvp_rev/memory_axis_demo.py`.

---

## 5. Docs inventory (`docs/`) — this session
- **This file** — master reference.
- [`THEORY_CHAPTER_D.md`](THEORY_CHAPTER_D.md) — the verified theory backbone (T1–T4 + stability characterization).
- [`RHCP_ARCHITECTURE.md`](RHCP_ARCHITECTURE.md) — RHCP spec + measured results.
- [`RHCP_EXPERIMENT_PLAN.md`](RHCP_EXPERIMENT_PLAN.md) — verified P0–P4 plan + control table.
- [`RHCP_RESULTS.md`](RHCP_RESULTS.md) — RHCP results (Panel A, baselines, gates).
- [`MILLIVID_CWVAE_VERDICT.md`](MILLIVID_CWVAE_VERDICT.md) — CW-VAE-clock = redundant.
- [`MILLIVID_REVERSIBILITY_VERDICT.md`](MILLIVID_REVERSIBILITY_VERDICT.md) — reversibility ≠ stabilization.
- Prior: `docs/SUMMARY.md`, `docs/RESEARCH_DIRECTIONS.md`, `docs/MILLIVID_DIRECTION.md`, `docs/HANDOFF.md`.
Results logs (gitignored, `logs_rhcp/`): `panelA_memfrontier.md`, `baselines_H20_ctx36.md`, `*.log`,
`hier_ebt_samples.png`, `hier_diffusion_samples.png`, `hier_ebt_step{2000,4000}.png`.

---

## 6. External resources (links + local copies)
- **EBT** — "Energy-Based Transformers are Scalable Learners and Thinkers", arXiv **2507.02092** ·
  repo https://github.com/alexiglad/EBT · local code `C:/Users/leoni/Documents/projects/EBT` · local paper text
  `C:/Users/leoni/Downloads/arXiv-2507.02092v1`. Follow-up (adaptive stopping): EBT-Policy arXiv 2510.27545.
- **MilliVid** — arXiv **2606.09056** · local code `C:/Users/leoni/Documents/projects/millivid` · paper
  `C:/Users/leoni/Downloads/arXiv-2606.09056v1`. (DiT-B/AE-B scale; ~576 H200-GPU-hours; no released checkpoints; tokenizer
  is per-frame, `num_frames_per_level=[3,12,48,192]` = factor-4 clock.)
- **CW-VAE** — arXiv **2102.09532** · local torch port `C:/Users/leoni/Documents/projects/cwvae/cwvae_torch.py` +
  ckpt `logs_torch/cwvae/last.pt`.
- **OpenSTL** — arXiv **2306.11249** · local repo `C:/Users/leoni/Documents/projects/OpenSTL` (MMNIST table, metrics, data).
- **DLH** (ICLR 2023, "CW-VAE + learned clock") — port in `hvp_dlh/`.
- Theory refs: **MacKay et al. 2018** (Reversible RNNs, 1810.10999); **Behrmann 2019/2021** (i-ResNet / Exploding Inverses,
  Lipschitz for INN stability); **CrevNet** (ICLR 2020, reversible deterministic video); **Self-Forcing** (NeurIPS 2025,
  2506.08009); **MSPred** (BMVC 2022, space×time hierarchy); **AlignProp** (reversibility ties/loses to checkpointing).

---

## 7. Durable insights (the knowledge worth keeping)
1. **Invertibility ≠ stability.** Additive coupling det-Jac ≡ 1 but singular values unbounded → drift/fade; stability
   needs *small* Lip(F) (§2.2 T3). This explains the RHCP fade and is a clean publishable characterization.
2. **Reversibility ≈ gradient checkpointing on memory** (Pareto/throughput, not a win). Honest claim = constant-memory
   *capability* vs vanilla BPTT (OOMs at K=120).
3. **MMNIST-class data is determined-or-unknowable** (no hard-but-computable middle) → dynamics mechanisms null. Use a
   testbed with genuine off-screen memory / resolvable structure.
4. **Only foreground metrics are trustworthy** on sparse video (aggregate MSE/SSIM background-dominated). Per-digit
   centroid, not a single global one. **Always review decoded sample images** — scalars hid the fade and the mean-blur.
5. **EBT recipe is decisive:** S1 (detach+per-step loss) = stable-first, does NOT shape thinking; **S2 (replay buffer +
   no-detach + final-step loss + randomized α/steps)** makes think-longer help. SmoothL1 (not MSE); per-element
   multiplicative α; α needs its own high-LR zero-wd param group; latent standardization.
6. **On 16 GB-tractable data, no denoiser (regression/EBT/diffusion) reaches GT sharpness** — the generative value of
   EBT/diffusion needs MilliVid's scale/complexity, unreachable here. Diffusion ≈ EBT (both blur); the choice isn't the
   bottleneck, the data+scale is.
7. **The hierarchy is the one mechanism-agnostic positive** on the looping testbed (D2 +75%; all hierarchical models beat
   persistence).

---

## 8. Reproduce commands
```
# Ch.D theory (T1–T4)                 venv/Scripts/python.exe -m hvp_rev.theory
# RHCP core validation                venv/Scripts/python.exe -m hvp_rev.rhcp_validate
# reverse-replay gradient gate         venv/Scripts/python.exe -m hvp_rev.rollout_train --gate --K 40
# Panel A memory frontier              venv/Scripts/python.exe -m hvp_rev.mem_frontier --levels 3 --batch 32 --K_grid 10,20,40,80,120,160
# CW-VAE/DLH long-horizon baselines    venv/Scripts/python.exe -m hvp_rev.compare --mode long --ctx 36 --H 20
# EBT stage-0 toy gate                 venv/Scripts/python.exe -m hvp_rev.ebt --gate
# EBT System-2 (S2 recipe)             venv/Scripts/python.exe -m hvp_rev.ebt_stage1 --recipe s2 --steps 5000 --k 4
# frozen conv-AEs                       venv/Scripts/python.exe -m hvp_rev.convae --train ...   |   -m hvp_rev.hier_convae --train ...
# D2 oracle-coarse diagnostic          venv/Scripts/python.exe -m hvp_rev.ebt_diag --ae logs_rhcp/hier_ae.pt --k 4
# hierarchical EBT (consistency-coupled) venv/Scripts/python.exe -m hvp_rev.hier_ebt --steps 8000
# hierarchical diffusion control       venv/Scripts/python.exe -m hvp_rev.hier_diffusion --steps 10000
# looping testbed generator            venv/Scripts/python.exe scripts/gen_looping.py --n 3000 --frames 100 --world 128 --digits 3 --out data/loop_train_3000x100.npy
```

---

## 9. Practical gotchas (Windows / this repo)
- Models load with `venv/Scripts/python.exe` (base Python lacks torch). CUDA = RTX 4060 Ti 16 GB.
- `logs_rhcp/` and `logs*` are **gitignored** (models/logs/images live there, not in git).
- Python block-buffers stdout to files → use `-u` for monitorable background runs.
- Bash-tool `run_in_background:true` already detaches — do NOT also add `&` (double-background orphans the tool's handle).
- `| grep` to a file buffers → interim output hidden; write raw to the log and grep after.
- numpy 2.x: `np.ptp(a)` not `a.ptp()`. nn.Embedding needs 0-based indices (looping actions {-1,0,1} → +1).
- spectral_norm on a zero-init weight = 0/0 NaN; converge power-iteration then `.eval()`-freeze before an exact inverse.
- MilliVid/EBT repos are pre-release (SLURM, no small config, no checkpoints; EBT video pipeline needs WSL/FFmpeg) → we
  built minimal equivalents in our own framework rather than wiring their code.

---

## 10. Open direction options (last decision — awaiting)
The diffusion control settled that the blur is task/data+scale-inherent, not EBT-specific. On 16 GB you're stuck between
*determined* (regression wins, no generative value) and *unresolvable-multimodal+small-model* (all blur). Options put to
the user (dismissed → pending):
- **A. Ship hierarchy + mechanism-study + Ch.D** — consolidate the robust wins (D2 hierarchy result + the controlled
  regression/EBT/diffusion comparison + the theory chapter). Defensible, no overclaim.
- **B. Change the task to fit the hardware** — a testbed where a small model can predict sharply / graded by generation
  validity (FVD) not match-GT. Risk: collapses to "regression wins" or still needs scale.
- **C. Push scale within 16 GB** — bigger denoiser + longer training + higher-res latents; quantify the scale gap.
- **D. Ship Ch. D theory only** — the guaranteed verified paper; lowest ceiling.

**Recommendation on the table:** A (combines the three robust results into one honest paper) with D as the guaranteed
floor. Whichever is chosen, Ch. D is the non-nullable backbone.

---

## 11. How to continue
1. Read this file + [`THEORY_CHAPTER_D.md`](THEORY_CHAPTER_D.md) + the two MilliVid verdict docs.
2. Everything is on branch `research/ebt-hierarchical-video`; run the §8 commands to reproduce any number.
3. Pick a §10 option (or a new one); the hierarchy AE + looping testbed + EBT/diffusion harnesses are all built and ready.
4. Memory index: `memory/MEMORY.md` → `memory/ebt-direction.md` (current), `memory/rhcp-built-and-validated.md` (prior).
