# RHCP — Results report (in progress)

End-to-end results for the **Reversible Hierarchical Clockwork Predictor**. Architecture spec:
[RHCP_ARCHITECTURE.md](RHCP_ARCHITECTURE.md); experiment plan: [RHCP_EXPERIMENT_PLAN.md](RHCP_EXPERIMENT_PLAN.md).
All numbers measured on one RTX 4060 Ti 16 GB, torch 2.11. Every phase was gated and red-teamed before the next.

> **Honest one-line status:** the architecture + constant-memory training mechanism are **built, verified, and
> measured**; the long-horizon drift contribution (Panel C) is the make-or-break and is reported with its honest
> outcome below. Absolute quality vs CW-VAE is capacity/compute-limited (1 M-param model, short training budget).

---

## 1. Verified core (training-free gates — all PASS)

| Property | Result | Where |
|---|---|---|
| Reversible block exact inverse (fp64) | **1.3e-15** | `rhcp_validate.py` STEP 0 |
| Assembled 3/4-level clock-hold reconstruction (fp64) | **1–2e-13** (exact), HOLD + co-evolving parent | STEP 1 |
| Reverse-replay gradients == BPTT (real RHCP, K=40) | **grad-cos 1.000000**, rel-err 4.7e-5 | `rollout_train.py --gate` (l1mse + l1lpips) |
| OpenSTL metrics vs reference | **bit-identical** (mse/mae/ssim diff 0.0e0; psnr 6e-4) | `eval_adapters.py --selftest` |

The reverse-replay rollout backprop is **exact**, not approximate — the long-horizon training is correct.

---

## 2. Panel A — memory frontier (measured, real training loop)

`mem_frontier.py --levels 3 --batch 32` (encoder+core+decoder+Adam):

| K | BPTT | checkpoint | reversible |
|---:|---:|---:|---:|
| 40 | 12.2 GB | 1.47 GB | 1.09 GB |
| 80 | 23.3 GB* | 1.64 GB | 1.13 GB |
| 120 | **OOM** | 1.80 GB | 1.15 GB |
| 160 | **OOM** | 1.97 GB | 1.18 GB |

\*K=80 BPTT spills to shared RAM → 44 s/step (~60× slower), unusable before the hard OOM at K=120.

**Finding:** vanilla BPTT **OOMs at K=120**; reversible trains K≥160 free-running rollouts at flat ~1.2 GB. Reversible
is in the same sub-linear memory class as gradient checkpointing (√K anchors vs K states) and **modestly better than
the naive per-step variant** (1.18 vs 1.97 GB at K=160). **Mandatory caption:** *checkpointing achieves the same
K-independent core memory at one full-forward recompute; reverse-replay is the native recompute-free mechanism for a
reversible network, and the Panel-C drift claim is mechanism-agnostic.*

---

## 3. Baselines (long-horizon, cwvae test, ctx=36) — the target band

`compare.py --mode long --ctx 36` (H=20, 32 clips; foreground/position metrics — the only trustworthy ones, aggregate
MSE/SSIM are background-dominated):

| model | fg-MSE | centroid-L2 (px) | IoU | fg-LPIPS |
|---|---|---|---|---|
| persistence | 0.554 | 23.13 | 0.108 | 0.276 |
| DLH (CW-VAE + learned clock) | 0.346 | 17.02 | 0.314 | 0.278 |
| CW-VAE (mean-mode) | 0.285 | 10.29 | 0.414 | 0.188 |
| CW-VAE (best-of-5) | 0.270 | 9.31 | 0.430 | 0.189 |

**Target band for RHCP:** the CW-VAE row (centroid ≈ 10, fg-MSE ≈ 0.28, IoU ≈ 0.41). Per-digit centroid-L2 (the
metric the P0 red-team corrected from a broken global centroid) gives a clean monotone separation.

---

## 4. Panel C — drift payoff (the make-or-break)  *[PENDING — training]*

Control: **vary K_train only** (S=10 vs L=40), both n_in=36, both reverse-replay, shared l1lpips objective,
compute-matched (equal predicted-frames). Reframed honestly (P0 red-team) as **compounding latent-rollout drift**:
does supervising 40 steps of the open-loop latent rollout produce flatter long-horizon drift than supervising 10?

`compare.py --mode long --ctx 36 --H 40 --rhcp_s ... --rhcp_l ...`

*(table filled on completion: RHCP-S vs RHCP-L vs CW-VAE-mean/best5, DLH, persistence on fg-position-drift + fg-LPIPS
vs horizon; GO/NO-GO = L flatter than S beyond seed variance.)*

---

## 5. Second dataset — Moving-FashionMNIST (generalization)  *[PENDING]*

Long set generated (`data/moving_fmnist_long_1400x80.npy`, OpenSTL-matched protocol). Validates the Panel-C
contribution on harder textures (no CW-VAE checkpoint exists for fmnist → S-vs-L drift only).

---

## 6. Honest assessment (running)
- **Mechanism: proven.** Exact reversible long-horizon training; constant memory; real OOM frontier vs BPTT.
- **Memory headline: capability vs BPTT**, mechanism-agnostic for the science (checkpointing parity acknowledged).
- **Absolute quality: capacity-limited.** 1 M-param model on a short budget; reaching the CW-VAE band likely needs
  more capacity + steps (the scientific S-vs-L comparison is valid at any fixed capacity).
- **Single-seed caveat:** disk baselines have 1 training seed; any "beats CW-VAE" headline needs ≥3 seeds.
