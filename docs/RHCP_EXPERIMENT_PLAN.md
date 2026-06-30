# RHCP Experiment Plan — Moving-MNIST (verified)

**Reversible Hierarchical Clockwork Predictor: free-running rollout training reduces compounding drift, with
reversibility as the constant-memory enabler.**

Verified by a 7-agent asset-verification + adversarial red-team (2026-06-30). Three findings were **BLOCKING** (Panel-C
confounds, Panel-A no-real-OOM, the shared objective) and their fixes are folded into P0. Two user decisions
(reversible-vs-BPTT head-to-head; checkpointing out of the *headline*) are honored — with the one non-negotiable
amendment the Panel-A red-team forces: **checkpointing must still appear as a third line in the memory *figure***, even
though it is out of the contribution framing.

> Architecture is built+validated ([docs/RHCP_ARCHITECTURE.md](RHCP_ARCHITECTURE.md)). This doc is the experiment plan.

---

## 0. Locked decisions (read first)

| Decision | Locked value | Why |
|---|---|---|
| **Short-table data** | `C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/mnist_test_seq.npy` — verified on disk (20,10000,64,64). Numbers are **bit-comparable to OpenSTL**; drop the "our-protocol" caveat for the short table. | Canonical file IS present; no download. |
| **Long-horizon data** | Project clips `data/mmnist_long_test_3000x{17,25}` + `2000x33` (label "our-protocol"). **BLOCKER:** longest non-pturn set is T=33 → max horizon from ctx=10 is **23**. **P0 must generate a ≥50-frame rigid set** or re-scope the headline horizon to 10→23. | 10→40 is physically unreachable from disk. |
| **CW-VAE baseline** | **DO NOT TRAIN.** Reuse `../cwvae/logs_torch/cwvae/last.pt` (standalone CW-VAE) **and** `logs_dlh/baseline_iter8000.pt` (DLH = CW-VAE + learned clock). Both on disk, both have `open_loop()`. Inference-only, minutes. | Both checkpoints real and runnable. |
| **CW-VAE fairness** | Evaluate CW-VAE in **mean-mode** (`open_loop(sample=False)`) for an apples-to-apples match to deterministic RHCP, **and** report **best-of-5** as the stochastic ceiling. | Determinism mismatch is the fairness trap. |
| **Ranking metric** | **Foreground-masked fg-MSE + fg-LPIPS per horizon** are the ONLY ranking metrics; aggregate MSE/SSIM/PSNR → "background-dominated, not for ranking" appendix. For Panel C **also** report **foreground POSITION drift (centroid-L2 / digit-IoU)** — fg-pixel-MSE saturates at the decorrelation floor (~73) by h≈15. | SSIM-background memory + testbed red-team. |
| **Panel C framing** | Vary **K_train ONLY**. Reframe from "long vs short horizon reduces drift" to the defensible mechanism: **free-running rollout training closes the exposure-bias gap that compounds even under deterministic dynamics.** | Rigid MMNIST is near-deterministic ballistic; pure horizon may null, exposure-bias is real. |
| **Panel C backward** | **BOTH** arms route through **reverse-replay** (promoted into the trainer). Reversibility is a *fixed enabler*, not a varied factor. | Today S=autograd, L=replay → confound. |
| **Panel C objective** | **ONE shared distribution-matching objective for both arms.** Primary: context-conditioned temporal PatchGAN on free-run frames + deadband fg-L1, decoded through the frozen Step-1 renderer. **Fallback: L1+LPIPS** if the GAN won't stabilize. **Never plain MSE/L1** — E1 proved it fades to the mean and penalizes the larger-K arm *more*. | MSE-rollout objective is actively biased against L. |
| **Memory figure** | Reversible-vs-BPTT is the headline framing, but the **figure plots three lines incl. checkpointing**, with the mandated caption. OOM point must be **measured at a real ship config**. | Current bench fits BPTT to K=100; omitting checkpoint is misleading. |
| **AMP / re-anchor** | Forward MAY use AMP. **Inverse recompute MUST be fp32** (AMP inflates drift ~17×). **Re-anchor every floor(√K)** steps; drift monitor triggers an extra anchor if drift > 5% of mean \|state\|. | Numerics red-team. |
| **Testbed order** | **Rigid MMNIST first** (keeps the deterministic core honest). Fallback if rigid nulls is **NOT pturn** (deterministic-mean blur = wrong failure mode) — it is a deterministic-but-harder task: **3+ digit MMNIST, KTH, or the gSTA real-clip base**. pturn0.1 is a *secondary P4 ablation only*. | Testbed red-team. |
| **Seeds** | ≥3 training seeds per Panel-C arm. Any "beats CW-VAE" headline needs ≥3 baseline seeds too (baselines have 1 → CI is a lower bound). | Single-seed deltas are within noise. |

---

## 1. The Panel-C control table (HELD-CONSTANT vs VARIED)

The result is attributable to training horizon **only** if every row but the last is identical across the two arms.

| Factor | S-model (short) | L-model (long) | Held / Varied |
|---|---|---|---|
| Architecture + param count | RHCPConfig 16/8/4, ~0.46M | identical | **HELD** |
| Weight-init RNG seed | seed s | identical seed s | **HELD** |
| Encoder context n_in | 10 | 10 | **HELD** |
| Train regime | free-running AR | free-running AR | **HELD** |
| Backward path | reverse-replay @ K=10 | reverse-replay @ K=40 | **HELD semantics** |
| Objective | shared DM objective | identical | **HELD** |
| Optimizer / LR | Adam, OneCycle pct_start=0.1, clip 1.0 | identical | **HELD** |
| Data source + split | split A | identical split A | **HELD** |
| Eval harness | fg-MSE+fg-LPIPS+centroid-L2+IoU vs horizon | identical | **HELD** |
| Effective batch | B (grad-accum if needed) | identical effective B | **HELD** |
| Inverse precision / re-anchor | fp32, anchor every √10≈3 | fp32, anchor every √40≈6 | HELD *policy* |
| **K_train** | **10** | **40** | **VARIED — the only free factor** |

**Compute-match:** *Primary* = equal total predicted-frames-seen (L does N updates at K=40; S does 4N at K=10).
*Secondary* = equal gradient-updates (then L sees 4× frames; if L still wins, stronger claim). Pre-register primary.

**Read-out:** evaluate S *past its training horizon* (10→40) against L; the horizon where the curves diverge is the
figure. ≥3 seeds; overlay individual curves, report mean±sd.

---

## 2. Ordered phases

Interpreter `venv/Scripts/python.exe`; single RTX 4060 Ti 16 GB.

### P0 — Data + eval harness + Panel-C prerequisites (BLOCKS P1, P3)
**Write:** `hvp_rev/eval_adapters.py` (reshape canonical `(20,N,H,W)` and project **TIME-FIRST** `(T,N,H,W)` →
`(N,T,1,H,W)` float[0,1]); `hvp_rev/long_eval.py` (fg-mask `gt>0.1`, fg-MSE, fg-LPIPS, **centroid-L2 + digit-IoU**,
per-horizon clip-bootstrap 95% CI); `scripts/gen_long_mmnist.py` (generate `data/mmnist_long_test_3000x55_d2_s64.npy`
via the OpenSTL-matched generator, p_turn=0).
**Reuse:** `hvp/eval/openstl_metrics.py` (<1e-4 self-test), `scripts/a0_longhorizon_ci.py` (the fair harness — extend).
**BLOCKING fixes (each gates P3):**
1. **Promote `reverse_replay` into the trainer** (`rhcp_validate.py:185-209` → reusable training step on the real
   `RHCP`); route **both** S and L through it. Gate: grad-cos vs autograd > 0.9999 at K=40 on the training config.
2. **Replace the objective** (`F.l1_loss+F.mse_loss` → the single shared DM objective; reuse `hvp_rev/gate_renderer.py`,
   `train_renderer_dlh.py`). Used identically at K=10 and K=40.
3. **Numerics guards**: fp32-inverse + re-anchor-every-√K + drift monitor (~30 lines).
4. **Seed + frames-budget plumbing**: seed flag, equal-predicted-frames controller, grad-accum.
**GO/NO-GO:** openstl_metrics self-test <1e-4 AND grad-cos vs autograd >0.9999 at K=40. Else stop.

### P1 — RHCP short-horizon (OpenSTL table) — anchor 1
Train RHCP 10→10 to convergence; eval via `openstl_metrics.py` on `mnist_test_seq.npy`.
**Deliverable:** short table — RHCP vs {PredRNN++ 22.06, SimVP 32.15, CrevNet 30.15} + flat base (logs/e2e_convnext_m8,
27.21) + CW-VAE-mean + DLH + persistence.
**Honest expectation:** RHCP **below the table**, high-30s/low-40s. **Kill criterion: MSE > 42.**

### P2 — Panel A: memory frontier (reversible vs BPTT, with checkpoint line)
**BLOCKING fix:** current bench fits BPTT to K=100 — no real OOM. Sweep batch/res/K in the **actual training loop**
(encoder+decoder+optimizer state) at the **ship config** (3–4 levels, batch 16–32, 64×64) with an OOM catch.
**Deliverable:** Panel A — peak-mem-vs-K, **THREE lines** (BPTT → OOM with last-fit/first-OOM marked; checkpoint flat;
reversible flat ~1.05× checkpoint). **Mandatory caption:** *"Per-step gradient checkpointing achieves the same
K-independent core memory at one full-forward recompute; we use exact reverse-replay because it is the native,
recompute-free mechanism for a reversible network, and the scientific claim (Panel C) is mechanism-agnostic."*
**GO/NO-GO:** a real measured OOM at the ship config. Do NOT headline a K-multiple as "N×" — checkpoint gets the same N;
realistic advantage before checkpoint matches is only ~2–4×.

### P3 — Panel C: drift payoff (MAKE-OR-BREAK) — blocked on P0
Run the Section-1 control table: 2 arms × ≥3 seeds, both reverse-replay, shared DM objective, equal-frames primary.
Extend `a0_longhorizon_ci.py` to add an RHCP runner + the `sample=False` CW-VAE mean-mode line so RHCP-S, RHCP-L,
CW-VAE-mean, CW-VAE-best5, DLH, persist, black share one axis.
**Deliverable:** Panel C — **foreground position drift (centroid-L2 + IoU) and fg-LPIPS vs horizon 10→40**, RHCP-S vs
RHCP-L (mean±sd, 3 seeds) vs CW-VAE-mean, CW-VAE-best5, DLH, persistence, black. Read on bins +1–10 / +11–25 / +26–40,
gated on the **slope** of fg-LPIPS — never aggregate MSE/SSIM.
**GO/NO-GO — kill:** if RHCP-L is not flatter than RHCP-S on fg-position drift beyond seed variance, report the **clean
null** (do NOT pivot to pturn). One confirmatory run on a deterministic-but-harder task; if still null, ship the
**memory-capability paper** (P2 + numerical-feasibility) as the contribution.

### P4 — Hierarchy ablation (honest, may null)
reversible-hierarchical vs matched-capacity reversible-flat, same eval; rigid primary, pturn0.1 secondary. Pre-registered
as neutral-to-null on the table metric (measured A/B: hier 40.42 vs flat 36.23); the hierarchy story lives only on the
drift curve. If null, frame as a controlled study.

---

## 3. Honest expected outcomes + fallback
- **Short table:** below OpenSTL, high-30s/low-40s. Kill if >42.
- **Memory:** reversible ~1.05× *worse* than checkpoint; ~2–4× horizon advantage before checkpoint matches. Headline =
  the **capability** ("a constant-memory K-step rollout vanilla BPTT cannot fit"), three-line plot, mechanism-agnostic caveat.
- **Numerics:** fp32 reverse-replay gives gradients cosine-identical to BPTT up to ~1e-3 rel-error at K≤100; drift ~linear
  in K. A **numerical-feasibility result, not a stability claim.** AMP-inverse forbidden; re-anchor every √K.
- **Panel C (make-or-break):** rigid MMNIST may null a *pure horizon* effect; the defensible win is the **exposure-bias
  gap** measured on **position drift**.
- **Fallback paper if Panel C nulls:** "exact constant-memory reverse-replay enables free-running rollout training at
  horizons vanilla BPTT cannot fit, validated numerically (grad-cos=1.0, linear drift) and behaviorally on the short
  table," with the drift study reported as an honest controlled null. Survives even without the drift edge.

**Single-seed caveat:** disk baselines have one seed → all CIs are lower bounds; any "beats CW-VAE" needs ≥3 baseline seeds.

---

## 4. Critical path
P0 (fixes 1–4 + ≥50-frame data) → **gate grad-cos>0.9999** → P1 (**gate MSE<42**) → P2 (**gate real OOM**) ∥ P3 (**gate
L flatter than S**) → P4. P3 is BLOCKED until P0 fixes land — do not burn the compute-matched seed-pairs until the shared
objective and shared backward are in the trainer.

**Verified assets present:** `mnist_test_seq.npy`, `../cwvae/logs_torch/cwvae/last.pt`, `logs_dlh/baseline_iter8000.pt`,
`logs/e2e_convnext_m8/best.pt`, all `data/mmnist_long_test_*.npy`. **New files:** `hvp_rev/{eval_adapters,long_eval,mem_frontier}.py`,
`scripts/gen_long_mmnist.py`; **edits:** `hvp_rev/rhcp_train.py` (fixes 1,2,4), `scripts/a0_longhorizon_ci.py` (+RHCP, +CW-VAE mean-mode).
