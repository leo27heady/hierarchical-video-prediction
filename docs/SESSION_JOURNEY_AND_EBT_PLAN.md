# Session journey, insights & the EBT plan — comprehensive reference

**Branch:** `research/ebt-hierarchical-video`. **Date:** 2026-06-30 → 2026-07-01.
**Current direction:** MilliVid-style hierarchical video prediction with the diffusion denoiser replaced by an
**Energy-Based Transformer (EBT)** — built minimally in our own framework, gated at each stage.
**Verified fallback:** the reversibility **theory chapter (Ch. D)** — done, non-nullable.

This document is the single source of truth for this session: the arc of directions explored, every honest verdict,
the assets built, and the forward plan. Companion docs (all committed): `THEORY_CHAPTER_D.md`,
`MILLIVID_CWVAE_VERDICT.md`, `MILLIVID_REVERSIBILITY_VERDICT.md`, `RHCP_ARCHITECTURE.md`, `RHCP_EXPERIMENT_PLAN.md`,
`RHCP_RESULTS.md`.

---

## 1. TL;DR

- We chased several architectures. On MMNIST-class data, **every added dynamics mechanism nulled** (adaptive clock,
  spatial gate, reversibility-for-stabilization, CW-VAE clock, EBT System-2 under the wrong recipe) — the data is too
  low-entropy to reward the mechanism. This is the session's central empirical lesson.
- **Two things survived scrutiny:**
  1. **Reversibility as a CORRECTNESS theorem** (Ch. D) — a provably-invertible *stochastic* recurrence with
     exact constant-memory gradients; measured, non-nullable. This is the safe dissertation backbone.
  2. **MilliVid + EBT** — the one *method* idea that passed its go/no-go gates: EBT trains stably on this GPU, and
     with the correct **S2 recipe** its energy-descent "think-longer" mechanism measurably works (25% hard-frame error
     reduction to convergence). The user's preferred, higher-ceiling direction.
- **Current build:** minimal MilliVid-*style* hierarchical (coarse-to-fine) EBT, to test whether the hierarchy unlocks
  a benefit past the single-scale saturation.

---

## 2. The journey (chronological, with verdicts)

Each direction was investigated with multi-agent review + adversarial red-team, and gated honestly.

| # | Direction | Verdict |
|---|---|---|
| 1 | **RHCP** (reversible hierarchical clockwork predictor; CW-VAE pyramid + RevNet coupling + identity-HOLD clock) | Built + validated: exact inverse, **grad-cos vs BPTT = 1.000000**, memory frontier (**BPTT OOMs at K=120**). BUT the reversible model **faded** on MMNIST (centroid 63px, worse than persistence) — see #6 for why. |
| 2 | **RHCP for OpenSTL table (P1/P3)** | Honest: from-scratch below the table; hierarchy is a null on MMNIST; memory-win is Pareto-neutral vs checkpointing. Pipeline built (eval harness, reverse-replay trainer, baselines). |
| 3 | **MilliVid + CW-VAE clock** | **Redundant.** MilliVid's `num_frames_per_level=[3,12,48,192]` IS already a factor-4 temporal clock. The 2-axis hierarchy = MSPred (2022). The real lever is MilliVid's decoder, not the clock. Verdict doc: `MILLIVID_CWVAE_VERDICT.md`. |
| 4 | **MilliVid + reversibility (AR stabilization)** | **Category error.** Invertibility ≠ stability (additive coupling det-Jac=1 but unbounded singular values → the RHCP fade is EXPECTED). MilliVid is stateless per-step diffusion → no recurrence to make reversible. Self-Forcing (2025) owns rollout-stabilization and does it *without* deep BPTT. Verdict doc: `MILLIVID_REVERSIBILITY_VERDICT.md`. |
| 5 | **Theory + hierarchy-science** (chosen fallback) | Ch. D (theory) **done + verified**. Looping off-screen-memory **testbed built** (the fix for the MMNIST-null problem). Learned token-pyramid AE had a bug (parked; superseded by a plain conv-AE). |
| 6 | **MilliVid + EBT** (current) | **PASS the gates.** Stage 0/0b/1/1b below. The one method that survived. |

---

## 3. Verified findings & honest insights (the durable knowledge)

1. **Invertibility ≠ stability** (measured, `hvp_rev/theory.py`): additive coupling `s+F(s)` has det-Jacobian ≡ 1 but
   *unbounded* singular values; over K=100 steps perturbation growth is 333× (unconstrained), 26,600× (even Lip(F)≤1),
   1.5× (Lip(F)≤0.1), 1.0× (Lip(F)≤0.02). Stability needs *small* Lip(F); reversibility contributes nothing to it.
   This is *why* the reversible RHCP faded. It's also a clean, publishable characterization.
2. **Reversibility ≈ gradient checkpointing on memory** (Pareto/throughput point, not a categorical win) — AlignProp +
   this repo's own `memory_axis_demo.py`. The honest memory claim is "constant-memory capability vs vanilla BPTT
   (which OOMs at K=120)", never "beats checkpointing".
3. **MMNIST-class data is too low-entropy** to reward dynamics mechanisms. A frame is either *determined* (2-frame
   velocity → next frame, nothing to think about) or *unknowable* (off-screen content not in context). There is no
   *hard-but-computable* middle regime that iterative mechanisms (clock, hierarchy, EBT thinking) can exploit. This
   single fact explains ~all the nulls this session. → The **purpose-built looping testbed** (off-screen memory) exists
   to break this, and is the correct testbed going forward.
4. **Aggregate MSE/SSIM are background-dominated** on MMNIST; only **foreground** metrics (fg-MSE, fg-LPIPS, per-digit
   **centroid-L2** — the metric a red-team fixed from a broken single-global-centroid) are trustworthy. `long_eval.py`.
5. **EBT training recipe matters decisively** (verified vs the real EBT repo, `EBT_verification`): the paper's **S1**
   recipe (detach + per-step loss) gets it stable but does NOT shape a "thinking" landscape; **S2** (replay buffer +
   no-detach + final-step loss + randomized α/steps) is what makes *think-longer* help. Training S1-only produced a
   System-2 null; S2 fixed it. Also: SmoothL1 (not MSE) for latent-video; per-element multiplicative randomized α
   (whole-batch scalar hurts); α needs its own high-LR, zero-wd param group; latent standardization.
6. **The decoder/renderer is the one repeatable quality lever** across the whole project (`step1-adversarial-renderer`:
   FID 66 vs 101); distribution-matching > perceptual > pixel-MSE (which fades to the mean, `e1-mse-rollout`).

---

## 4. Theory chapter D — the verified backbone (fallback)

`docs/THEORY_CHAPTER_D.md`, `hvp_rev/theory.py`, `hvp_rev/rhcp.py`, `hvp_rev/rollout_train.py`. All measured:

| Property | Result |
|---|---|
| Reversible block exact inverse (fp64 / Lipschitz fp32) | 5.5e-15 / 3.1e-5 |
| **Stochastic** store-the-noise invertibility | exact iff eps stored (8.4e-13); breaks otherwise (10.9) |
| Invertibility ≠ stability (K=100 growth vs Lip(F)) | 333× → 26,600× → 1.5× → 1.0× (see §3.1) |
| Reverse-replay gradients == BPTT (Lipschitz+stochastic) | **grad-cos 1.000000** |
| Constant-memory frontier | BPTT **OOMs at K=120**; reversible flat ~1.2 GB |

Contribution = a *correctness/enabler theorem* (extends TempVerseFormer to the stochastic case), NOT a benchmark win.
Non-nullable because it's a theorem + figure. This is the guaranteed-shippable paper if EBT stalls.

---

## 5. Current direction — MilliVid + EBT

### 5.1 Why (from the 3-agent investigation, `MILLIVID_REVERSIBILITY_VERDICT.md` era + EBT investigation)
- **Clean module swap** (agent-verified vs MilliVid code): MilliVid's rollout is per-step *iterative refinement* (T
  diffusion steps). EBT replaces that inner loop with *energy-descent* steps — same slot, more principled optimizer.
  Tokenizer + coarse-to-fine rollout + decoder reused untouched; the change is `train_step` + the inner loop of
  `generate` + a scalar energy head.
- **Real, distinctive value props:** (a) a *verifiable* objective (low energy = self-consistent prediction), (b)
  **adaptive/stoppable test-time compute** (think longer on hard frames — EBT's headline, *unclaimed for video*), (c)
  it **re-homes reversibility**: EBT training backprops through the K-step unrolled energy loop → reversibility can make
  that constant-memory (the recurrence reversibility was built for finally exists here).
- **Novelty:** EBT already does *flat* video (SSV2, frozen SD-XL latents). The genuine delta is **hierarchical /
  coarse-to-fine energy-based video** (untaken) + the per-level adaptive-compute claim.
- **Honest caveats:** EBT is heavily-regularized/finicky; deterministic descent risks mean-collapse (mitigate with
  Langevin + Best-of-N + the hierarchy's lower per-level entropy); no released checkpoints (train from scratch);
  matching MilliVid-scale quality is infeasible on 16 GB (that's fine — the claim is the mechanism, not SOTA).

### 5.2 The staged gate (all PASSED so far)
- **Stage 0** (`hvp_rev/ebt.py`) — toy: the unrolled second-order energy objective **trains stably** on the 4060 Ti,
  energy converges monotonically. (Multimodality is EBT's documented weak spot — mean-seeks on a bimodal toy.)
- **Stage 0b** (`hvp_rev/ebt_video.py`) — real frozen conv-AE latents: **trains stably** (faithful recipe: SmoothL1,
  S1 per-step loss, per-element α, dedicated high-LR α group); on easy/weakly-multimodal next-frame it's competitive
  with regression but mildly blurs like it (value prop not yet shown on an easy task).
- **Stage 1** (`hvp_rev/ebt_stage1.py`, S1) — System-2 **NULL** (error flat across inference steps). Root cause found:
  S1 never trains the thinking landscape.
- **Stage 1b** (S2 recipe: replay buffer + no-detach + final-loss) — **System-2 WORKS**: think-longer reduces
  hard-frame error **25%** (converges by ~4 steps). *Honest nuance:* the benefit **saturates at convergence**; thinking
  *beyond* convergence doesn't yet exceed baseline (low-entropy-data limit). Sharpness edge over regression is marginal
  (0.0148 vs 0.0122, GT 0.0294). Mechanism validated; the **hierarchy is the next lever**.

---

## 6. Assets built this session

**Code (`hvp_rev/`):** `rhcp.py` (reversible core + Lipschitz/out_scale), `rollout_train.py` (reverse-replay trainer +
gate), `theory.py` (Ch. D verification), `eval_adapters.py` (OpenSTL-identical metrics), `long_eval.py` (fg/centroid/
LPIPS + bootstrap), `mem_frontier.py` (Panel A), `compare.py` (RHCP vs CW-VAE/DLH), `convae.py` (frozen conv-AE, val
recon 0.0016), `ebt.py` (EBT core + Stage-0 gate), `ebt_video.py` (EBT on latents, S1/S2 mcmc + replay), `ebt_stage1.py`
(System-2 + sharpness gate), `tokenizer.py`+`train_tokenizer.py` (token-pyramid AE — parked, has an overfit bug).
**Data (`data/`):** `loop_train_3000x100.npy` + `loop_test_500x200.npy` (+actions) — the looping off-screen-memory
testbed; `moving_fmnist_long_1400x80.npy` (2nd dataset). **Models (`logs_rhcp/`, gitignored):** `convae_loop.pt`.
**Docs (`docs/`):** this file + the six companions listed at top.

---

## 7. Forward plan — minimal MilliVid+EBT (step by step)

**Testbed:** the looping off-screen-memory dataset (not MMNIST — low-entropy null). **Recipe:** S2 (the one that makes
think-longer work). **Metrics:** fg-MSE + centroid-L2 + sharpness + fg-LPIPS vs horizon, **plus saved predicted-sample
image grids** reviewed each checkpoint (context | prediction | GT). **Discipline:** red-team the plan before each step;
only escalate to the user on an unresolvable blocker.

- **Step A — hierarchical latents:** extend the conv-AE (or add a coarse tap) to a 2-level pyramid (coarse 4×4 + fine
  8×8), each independently decodable (MilliVid's per-level-decodable property, minimal version).
- **Step B — coarse-to-fine EBT rollout:** predict the coarse latent first (energy descent), then the fine latent
  conditioned on the coarse (energy descent) — a cascade of energy minimizations. S2 recipe throughout.
- **Step C — the money experiment:** does the hierarchy (i) improve prediction quality/sharpness over single-scale, and
  (ii) unlock a *beyond-convergence* think-longer benefit on hard/re-entry frames? Compare hierarchical-EBT vs
  single-scale-EBT vs regression, with sample images.
- **Step D — (only if C is positive):** longer runs + wandb; then, optionally, wire MilliVid's *actual* tokenizer.

**Go/no-go:** if the hierarchy + EBT shows a defensible benefit → this is the paper (novel: hierarchical adaptive-
compute energy video). If it nulls like everything else on this data class → ship Ch. D (theory) as the paper.

---

## 8. Honest risks & open questions
- **Low-entropy-data limit:** the looping testbed may still be too easy for beyond-convergence thinking; if so, need a
  genuinely harder task. This is the dominant risk (it sank every prior mechanism).
- **EBT mean-collapse / multimodality:** the hierarchy is the hypothesized mitigation; unproven.
- **Second-order unroll memory** at larger K — the honest home for reversibility (constant-memory energy unroll), a
  possible synthesis if we scale K.
- **Sample-quality vs metrics:** must review decoded images, not just scalars (scalars hid the fade and the mean-blur).

---

## 9. Key commands
```
# theory chapter (Ch. D) verification
venv/Scripts/python.exe -m hvp_rev.theory
# EBT stage-0 toy gate
venv/Scripts/python.exe -m hvp_rev.ebt --gate
# EBT on real latents (S1/S2), System-2 sweep
venv/Scripts/python.exe -m hvp_rev.ebt_stage1 --recipe s2 --steps 5000 --k 4
# frozen conv-AE
venv/Scripts/python.exe -m hvp_rev.convae --train --data data/loop_train_3000x100.npy --out logs_rhcp/convae_loop.pt
# reverse-replay gradient-equivalence gate
venv/Scripts/python.exe -m hvp_rev.rollout_train --gate --K 40
```
Interpreter: `venv/Scripts/python.exe`. Hardware: RTX 4060 Ti 16 GB. wandb: offline (`WANDB_MODE=offline`).
