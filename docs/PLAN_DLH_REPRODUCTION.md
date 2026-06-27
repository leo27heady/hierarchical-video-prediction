# DLH reproduction plan — Dynamic Latent Hierarchy (Zakharov, Guo, Fountas, ICLR 2023)

> Branch `reproduce/dlh`, forked clean from `main`. Goal: a faithful, from-scratch PyTorch reproduction of DLH
> (arXiv:2212.14376) — the SOTA hierarchical long-horizon video predictor that beats CW-VAE — as the backbone
> for the dissertation. **No official code exists** (verified: Huawei never released it), so this is a port from
> the paper, like the CW-VAE port was. The paper LaTeX is at `C:/Users/leoni/Downloads/arXiv-2212.14376v2/`.

---

## 0. Why DLH, and what "working" means here

DLH = **CW-VAE with a *learned, content-adaptive* clock**. Where CW-VAE ticks each level on a fixed schedule
(factor 6), DLH decides *per timestep, per level* whether a level should **update (change)** or **stay (static)**,
via a binary indicator `eⁿ_t`. The set of levels that update at step `t` defines an adaptive "depth of update"
`K_t` — the inspectable abstraction/clock signal. On simple data it **auto-collapses** unused levels (Moving Ball:
3-level KL = 34.7 / 7.8 / 0.3 — level 3 unused). This is the apex of the hierarchical-video line and the grounded
version of the entropy-clock idea that CW-VAE could not support (its σ doesn't track error; DLH's `eⁿ_t` is trained
to fire on real state changes).

**Success criteria (calibrated to a single 16 GB RTX 4060 Ti, not a V100 cluster):**
- **P1 (mechanism works):** ELBO decreases; the dynamic structure emerges — `K_t` varies over time, and the
  per-level KL shows the collapse pattern (bottom levels high-KL, top levels →0). This is the scientifically
  interesting result and the dissertation hook. *Primary target.*
- **P2 (numbers in the ballpark):** MMNIST open-loop holds a **flat long-horizon SSIM curve** and lands in the
  **0.70–0.78** band (paper 0.78; we expect a gap from the compute/data budget, as CW-VAE did 0.571 vs 0.68).
- **Non-goal:** exact 0.78 / 3-day-V100 parity. We reproduce the **behavior and the mechanism**, then extend.

> Carry forward the eval lesson: **aggregate SSIM/PSNR on MMNIST are background-dominated** (all-black ≈ 0.76,
> beats CW-VAE). Judge with foreground-MSE / LPIPS / FVD / trivial baselines alongside SSIM, or the numbers are
> noise. See `memory: mmnist-ssim-is-background-dominated`.

---

## 1. Architecture (faithful to the paper; Table 2 + §2.4)

Indexing: **n = 1 is BOTTOM** (always updates, `e¹_t = 1`), **n = N is TOP**. 3 levels for MMNIST/Mazes,
2 levels for Moving Ball.

**Dimensions (paper §A.1):** deterministic `|xⁿ| = |dⁿ| = |cⁿ| = 200`; stochastic `|sⁿ| = 20`. Total ≈ 7M params.

**Three deterministic information pathways** (NVAE/VPR-style, Fig 1c):
- **bottom-up** `xⁿ_t` — image features rising through levels.
- **top-down** `cⁿ_t` — context descending from higher levels.
- **temporal** `dⁿ_t` — per-level recurrent memory of the past.

**Component networks (paper Table 2):**

| Component | Form | Maps |
|---|---|---|
| Image encoder | 4× Conv2d (k4,s2), 1→32→64→128→256 → flatten → Linear→200 | image `o_t` → `x¹_t` |
| Image decoder | Linear→1024 → 4× ConvT (k5/6,s2) → tanh | `c⁰_t` → `o_t` |
| Encoder `f_enc(xⁿ)` | MLP [200] | `xⁿ → xⁿ⁺¹` (bottom-up) |
| Decoder `f_dec(sⁿ, cⁿ)` | MLP [200] | `(sⁿ, cⁿ) → cⁿ⁻¹` (top-down) |
| Temporal `f_tem(sⁿ, dⁿ)` | GRU [200] | `→ dⁿ_{t+1}` |
| Posterior `q(sⁿ | xⁿ, s^{>n})` | MLP [40,40,40] → (μ,σ) 20-d | inference state |
| Prior-state (**change**) `p(sⁿ | sⁿ_{<t}, s^{>n})` | MLP [40,40,40] → (μ,σ) 20-d | conditioned on `dⁿ` (past) + `cⁿ` (above) |
| Prior-factor `p(eⁿ | sⁿ_{<t})` | GRU [200] → Bernoulli logit | transition prediction |

**Reuse from the CW-VAE port (`../cwvae/cwvae_torch.py`):** the **image conv encoder/decoder are identical**
(4-conv 32/64/128/256, 4-deconv, tanh) — lift them verbatim. Also reuse the Gaussian-reparam helper, the
analytic Gaussian KL, and the data pipeline + eval harness.

**Static prior** `p(sⁿ_t | eⁿ_t=0) = p(sⁿ_{t-1})` — the *previous* state distribution, carried forward, **no network**.

---

## 2. The dynamic mechanism (the heart of DLH; §2.2–2.3 + Alg. 1)

**Two-component temporal MoG per (n,t):**
```
p(sⁿ_t | eⁿ_t) = p(sⁿ_{t-1})              if eⁿ_t = 0   (STATIC: keep previous)
              = p_θ(sⁿ_t | sⁿ_{<t}, s^{>n}) if eⁿ_t = 1   (CHANGE: learned transition)
```

**Non-parametric inference of `q(eⁿ_t)`** — *no Gumbel/REINFORCE/straight-through*. Hard model-selection by
KL (Eq. 8): pick the component the inferred posterior `q'(sⁿ_t)` is closer to:
```
eⁿ_t = 1  (change)  if  KL[q' ‖ p_change] <  KL[q' ‖ p_static]
eⁿ_t = 0  (static)  otherwise
```
Set `q(eⁿ_t = i) = 1` for the winning component (hard).

**Nested-timescales constraint:** `q(eⁿ_t = 1 | eⁿ⁻¹_t = 0) = 0` — a level can change only if the level below
changed. So per timestep there is a single cut `K_t` = highest level that changed; levels `> K_t` are **frozen**
(state copied from `t-1`).

**Per-timestep unroll (Alg. 1):**
1. **Bottom-up e-pass** `n = 1 … N`: compute both KLs, decide `eⁿ_t`. First level with `eⁿ_t = 0` sets `K_t = n`
   and stops the upward pass (levels above frozen, `q'(s^{>n}_t) ← q'(s^{>n}_{t-1})`).
2. **Top-down state-pass** `n = K_t … 1`: compute the Gaussian posteriors `q(sⁿ_t | xⁿ_t, s^{>n}_t)` for the
   updating levels (top-down context `cⁿ` from the frozen/just-sampled levels above).
3. Decode `c⁰_t → o_t`; advance temporal GRUs `dⁿ` for updated levels.

> **THE key engineering subtlety:** `K_t` differs **per batch element**, so you cannot Python-`break` per sample.
> Implement the unroll **densely with masks**: compute all levels every step, then gate each level's state/temporal
> update by its per-sample `eⁿ_t` mask (`s_new = e·s_change + (1-e)·s_prev`). The nested constraint becomes a
> cumulative-product mask down the levels (`e_eff^n = e^n · Π_{m<n} e^m`). This vectorizes cleanly and keeps it on-GPU.

---

## 3. Objective (ELBO, §2.5) + training

Three terms (Eq. 5a–c), summed over `t`, averaged over batch:
- **(a) reconstruction:** Gaussian NLL of `o_t` under decoder mean, **fixed std tuned per dataset** (as CW-VAE's
  `dec_stddev`). Summed over pixels.
- **(b) state KL:** `Σ_n KL[q(sⁿ_t) ‖ p_selected(sⁿ_t)]` — KL to the *selected* MoG component. Frozen levels
  (copied q = copied prior) contribute ≈ 0 automatically.
- **(c) transition KL:** `Σ_n KL[q(eⁿ_t | eⁿ⁻¹_t) ‖ p_θ(eⁿ_t | dⁿ)]` — Bernoulli KL; trains the prior-factor GRU
  to anticipate changes. (`q(eⁿ)` is the hard 0/1 from §2.)

**β warmup:** multiply (b)+(c) by `β`, linearly `0 → 1.0` over the first **10k iters** (learn reconstructions
before restricting latent complexity). Gradients flow through the *selected* component's KL and the recon term;
the hard `e`-selection is forward-only routing (fine — the prior-factor is supervised by term (c)).

**Optimizer / schedule (paper §A.1):** Adam, **lr 5e-4, ε 1e-4**, **seq len 100, batch 100**, 3 levels.

---

## 4. Datasets — synthetic only (per request; skip KTH real-world)

| Dataset | Have it? | Levels | Res | Train→eval | Role |
|---|---|---|---|---|---|
| **Moving MNIST** | ✅ `../cwvae/data_mmnist/{train,test}.npy` | 3 | 64² | 30 ctx → 300 | **primary**; flat-SSIM + dynamic structure |
| **Moving Ball** | ✖ generate (~50 LOC) | 2 | 64² | 30 → 100 | **cheapest**; stochasticity + level-collapse table |
| **GQN/DML Mazes** | ✅ `../cwvae/datasets/gqn_mazes` | 3 | 64² | 30 → 200 | optional; textured bg (SSIM more meaningful) |
| KTH | ✖ | — | — | — | **skip** — real-world, slow to train |

**Moving Ball** (Zakharov 2022, DLH's stochasticity probe) is the recommended *first* dataset: a single ball with
**stochastic** velocity/colour changes, tiny, fast, 2-level. It's where DLH showed CW-VAE fails on "sharp random
changes," so it directly exercises the stochastic mechanism and reproduces the per-level-KL collapse table (Table 4)
in hours, not days. Generate it as a small script (`data/moving_ball.py`): ball on black 64², random bounce +
random colour switch with prob `p` per step.

Protocol detail: paper gives **30 context frames** (vs CW-VAE's 36) and evaluates **best-of-100** samples
(SVG protocol). Use **best-of-10** for the dev loop, best-of-100 only for final figures.

---

## 5. Compute fit (16 GB, single GPU)

The paper's batch 100 × seq 100 × 64² over a sequential per-timestep unroll **will not fit / will be slow** here.
Mitigations, in order:
1. **batch 16–32, seq 100** first; watch memory. Gradient-accumulate to an effective batch if needed.
2. **Gradient checkpointing over time** (`torch.utils.checkpoint` around the per-timestep step) if activations blow up.
3. If still tight, **seq 50** for early dev (the flat-curve behavior shows by ~100 frames anyway).
4. Subsample MMNIST train (the CW-VAE repro used 3k clips) to keep epochs fast.

Expect the sequential dynamic unroll to be the bottleneck (like CW-VAE's 1.4 it/s). Budget **overnight runs**, not
minutes. Moving Ball first keeps the iteration loop cheap while the model is debugged.

---

## 6. Staged build & validation (each phase gated)

- **Phase 0 — scaffold (½ day).** New package `hvp_dlh/`: lift conv enc/dec + data loader + eval from `../cwvae`;
  config dataclass; Gaussian/KL helpers. Gate: forward pass runs, shapes match Table 2.
- **Phase 1 — 2-level DLH on Moving Ball (1–2 days).** Implement the MoG, the KL decision rule, the masked dynamic
  unroll, the 3-term ELBO, β-warmup. **Gate (the real milestone):** ELBO ↓; `K_t` varies; reproduce the per-level
  KL **collapse** pattern (L1≫L2≫L3→0, paper Table 4). This proves the mechanism. ← *"see it working" lands here.*
- **Phase 2 — 3-level DLH on MMNIST (2–4 days incl. training).** Scale up; tune decoder std. **Gate:** flat
  long-horizon SSIM curve over 300 frames; SSIM 0.70–0.78 (best-of-10).
- **Phase 3 — honest eval + structure figures (1 day).** foreground-MSE / LPIPS / FVD + trivial baselines (black,
  persist, mean-ctx); plot `K_t` over a rollout and the per-frame metric curves; per-level KL bars. These are the
  dissertation figures.
- **Phase 4 — optional Mazes** (textured bg, sanity that it's not MMNIST-specific).

After Phase 3 we **stop, review, and decide** the extension (small additions / rethink) — per your plan.

---

## 7. Risks & mitigations

- **No reference code → silent faithfulness bugs.** Mitigate with unit checks: single-level DLH should ≈ a 1-level
  VAE; forcing `eⁿ_t≡1` should ≈ CW-VAE-without-fixed-clock; per-level KL collapse must reproduce (Table 4) — these
  are concrete correctness oracles.
- **Hard `e`-selection instability** (prior-factor "becomes overconfident early," paper §2.3 footnote). Mitigate with
  the β-warmup and, if needed, label-smoothing on the transition KL.
- **Masked dense unroll correctness** (the per-sample `K_t`). Write a tiny numeric test: dense-masked vs an explicit
  per-sample Python-loop reference on a 2-batch toy must match.
- **Compute.** Moving Ball first; checkpointing; subsample; overnight budget. Accept a 0.70–0.78 landing, not 0.78.
- **Metric trap.** Never report MMNIST SSIM alone; always with foreground/perceptual + trivial baselines.

---

## 8. Proposed file layout (branch `reproduce/dlh`)
```
hvp_dlh/
  model.py        # DLH: conv enc/dec (from cwvae), per-level MLP/GRU components, MoG, masked dynamic unroll, ELBO
  config.py       # dataclass: levels, dims (200/20), per-dataset decoder std, β-warmup, lr, seq/batch
  data.py         # MMNIST + Moving-Ball generator + (opt) GQN mazes loaders  (reuse ../cwvae loaders)
  train.py        # Adam, β-warmup, checkpointing, periodic open-loop eval
  eval.py         # 30→300 open-loop, best-of-N, SSIM/PSNR + foreground/LPIPS/FVD + trivial baselines
  structure.py    # K_t-over-time + per-level-KL diagnostics (the dynamic-structure figures)
docs/PLAN_DLH_REPRODUCTION.md   # this file
```

**Open question for you:** repo home — keep DLH here in `hierarchical-video-prediction` (this branch), or co-locate
with the CW-VAE port in `../cwvae`? This plan assumes here, pointing at `../cwvae` for data. Easy to change.
