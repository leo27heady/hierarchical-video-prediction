# DLH, explained — how the Dynamic Latent Hierarchy actually works

> Companion to the `hvp_dlh/` reproduction. Plain-language walkthrough of every moving part of DLH
> (Zakharov, Guo, Fountas, *Long-horizon Video Prediction using a Dynamic Latent Hierarchy*, ICLR 2023,
> [arXiv:2212.14376](https://arxiv.org/abs/2212.14376)), tied to the paper's sections/equations/figures
> and to where each piece lives in our code. Paper LaTeX: `C:/Users/leoni/Downloads/arXiv-2212.14376v2/`.

---

## 1. The problem it solves, and the one idea

**The motivation (paper §1, Fig. 1 — the "panda" figure).** A video is not one process; it is many processes
running at *different speeds*. In the paper's example: a panda changes position every frame (fast), an airplane
appears for a few frames (medium), the background never changes (static). A model that re-predicts *everything*
every frame wastes capacity and **accumulates error** on the slow/static parts — which is why deterministic and
single-rate models drift on long horizons.

**The one idea.** Represent the video as a **hierarchy of latent states**, and let each level decide, *per
timestep*, whether it needs to **update** or can **stay the same**. Slow features (background) live high in the
hierarchy and rarely update; fast features (motion) live low and update often. Crucially the update schedule is
**learned and content-dependent** ("separate and fluid timescales", abstract), not fixed.

This is the difference from **CW-VAE** (Saxena/Ba/Hafner 2021), which DLH builds on and beats: CW-VAE has the same
"slow levels on top" idea but with a **fixed** clock (level *n* updates every 6ⁿ frames). DLH makes that clock
**dynamic**. (Paper §3.1 "Benchmarks" describes CW-VAE as fixed-temporal-schedule; Table 1 shows DLH > CW-VAE on
Moving MNIST 0.78/15.7 vs 0.68/13.11, KTH, and Mazes, at 7M vs 12M params.)

---

## 2. The generative model — what the model "imagines" (paper §2.1, Fig. 1a)

DLH is a **latent-variable generative model**: it explains each frame `o_t` by a stack of hidden states and
generates video by rolling those states forward. Two kinds of latent per level `n`, per time `t`:

- **`sⁿ_t`** — a continuous **state** (diagonal-Gaussian vector, our `stoch_dim=20`). "What is the world like at
  level *n* right now." Level 0 = fine/fast detail, higher levels = coarse/slow content.
- **`eⁿ_t`** — a binary **indicator** (`Bernoulli`). "Did level *n* **change** at this step (1) or **stay** (0)?"
  This is the dynamic clock. (In the paper's indexing level 1 is the bottom; in our code level 0 is the bottom.)

The joint distribution (paper §2.1, joint-distribution equation) factorises as, for each `t`:
```
p(o_t, s⃗_t, e⃗_t) = p(o_t | s⃗_t) · Π_n  p(sⁿ_t | eⁿ_t, sⁿ_<t, s^{>n}_t) · p(eⁿ_t | sⁿ_<t)
                      └ decoder ┘        └──── state transition ────┘   └ when-to-change ┘
```
In words: an image is rendered from all the states (`decoder`); each state depends on its indicator, its own past,
and the states **above** it (top-down conditioning); and a small model predicts the indicator from the level's past.

### The heart: a two-component temporal Mixture-of-Gaussians (paper §2.1, the `eq:MoG` equation)

Each state's prior is **not one Gaussian but a mixture of two**, switched by `eⁿ_t`:
```
p(sⁿ_t | eⁿ_t) = p(sⁿ_{t-1})              if eⁿ_t = 0   → STATIC: "keep the previous belief"
               = p_θ(sⁿ_t | sⁿ_<t, s^{>n}_t)  if eⁿ_t = 1   → CHANGE: "predict a new state"
```
So at every step each level holds **two hypotheses**: *nothing changed* (reuse the old state) vs *something changed*
(a freshly predicted state). Inference (next section) decides which hypothesis the data supports. This is the
mechanism that lets a level "cluster temporally-persistent states together" (abstract) and only spend a transition
when the world actually moves.

*Code:* `model.py` — the **change** prior is `self.prior[n]` (a `GaussianHead` on `[dⁿ, cⁿ]`); the **static** prior
is the carried belief `state["bmean"/"bstd"][n]` (the persisted previous change-prior, updated only on a tick).

---

## 3. The three deterministic pathways (paper §2.4, Fig. 1c, Table 2)

The stochastic states are wired together by three deterministic information flows (think of them as the "plumbing"
that carries context between levels and across time):

| pathway | symbol | direction | meaning | our code |
|---|---|---|---|---|
| **bottom-up** | `xⁿ_t` | image → up the levels | features extracted from the frame, rising to inform each level | `img_enc` then `enc[n]` MLPs |
| **top-down** | `cⁿ_t` | top → down the levels | context from the slow levels conditioning the fast ones | `dec[n]` MLPs |
| **temporal** | `dⁿ_t` | left → right in time | a per-level recurrent memory of that level's own past | `tem[n]` GRUCell |

Concretely (paper §2.4, model-component equations): `xⁿ⁺¹ = f_enc(xⁿ)` (bottom-up), `cⁿ⁻¹ = f_dec(sⁿ, cⁿ)`
(top-down), `dⁿ_{t+1} = f_tem(sⁿ_t, dⁿ_t)` (temporal). DLH says its architecture follows **NVAE**
([Vahdat & Kautz 2020](https://arxiv.org/abs/2007.03898)) and **VPR** (Zakharov 2022) — deep hierarchical VAEs that
use deterministic variables to pass information between stochastic layers. The image encoder/decoder are 4 conv /
4 transposed-conv layers (paper §A.1); we lift them from the CW-VAE port (same DeepMind stack).

**Sizes (paper §A.1, Table 2):** deterministic `|xⁿ|=|dⁿ|=|cⁿ|=200`, stochastic `|sⁿ|=20`; posterior and
change-prior are `[40,40,40]` MLPs; temporal and "prior-factor" are GRU[200]; total ≈ 7M params. All configurable
in `config.py`.

---

## 4. Inference — reading the latents from a video (paper §2.2)

Training a VAE needs an approximate **posterior** `q` (what the latents probably were, given the observed video).
DLH factorises it as `q(sⁿ_t) q(eⁿ_t)` and estimates the two parts very differently.

### 4a. `q(sⁿ_t)` — the state posterior (paper §2.2, "Estimating q(s)")
A diagonal Gaussian produced by a network from the bottom-up feature and the states above:
`q_ψ(sⁿ_t | xⁿ_t, s^{>n}_t)`. Trained with the standard **reparameterisation trick** (Kingma & Welling 2014):
sample `s = μ + σ·ε`, so gradients flow. *Code:* `self.post[n]` (a `GaussianHead`), sampled in `_step` sweep B.

### 4b. `q(eⁿ_t)` — when did it change? The clever, training-free part (paper §2.2, Eq. `eq:likelihoodratio` / `eq:decisioninequality`)
Discrete latents are painful to train (Gumbel-softmax / REINFORCE are unstable, "exacerbated in hierarchical
models" — paper cites Falck et al. 2021). DLH **sidesteps this entirely** with a **non-parametric** rule: it asks
*which of the two MoG components is the inferred state more consistent with?* Formally it compares the expected
log-likelihood of the static vs change component, which reduces to a **comparison of two KL divergences** (the
"decision inequality"):
```
e = 1 (change)  ⇔  KL[ q(sⁿ_t) ‖ change-prior ]  <  KL[ q(sⁿ_t) ‖ static-prior ]
e = 0 (stay)    otherwise
```
Read it plainly: *if the state I just inferred is closer to "a freshly predicted state" than to "the old state",
declare a change; otherwise declare it static.* The winning component is taken hard (`q(e=i)=1`). The paper notes
(§2.2) this relates to the **VaDE** trick (Jiang et al. 2017) for Gaussian-mixture VAEs and is biased but, in
practice, trains better than the soft VaDE estimate. **This is why DLH is tractable** — no Gumbel, no policy
gradient, just two KLs and a comparison. *Code:* `_step` sweep A, `e_raw = (kl_c < kl_s).float()`.

> ⚠️ Paper §2.2 footnote (a real, documented failure mode): the *parametrised* prior-factor `p(eⁿ_t|·)` can become
> "overly confident even before any video features have been learned", skewing things. We watch for exactly this
> (see the bad-outcome plan in [PLAN_DLH_REPRODUCTION.md]).

---

## 5. Nested timescales and the adaptive depth `K` (paper §2.3, Fig. 2, Algorithm 1)

A loose hierarchy could let level 5 change while level 2 stays — incoherent. DLH forbids that with a **nested
constraint** (paper §2.3, the `nestedassumption` equation):
```
if eⁿ⁻¹_t = 0  then  eⁿ_t = 0   (a level can change only if the level below it changed)
```
Consequence: at any timestep the changing levels form a **contiguous block from the bottom**, `1 … K_t`, and
everything above `K_t` is frozen. `K_t` is the **adaptive abstraction depth** — the single inspectable number that
says "how deep did the update go this step." Bounces / scene changes → `K_t` spikes; quiet stretches → `K_t = 1`.
The paper also **prunes** computation above a frozen level ("block any further inference above the level where
e=0"), which is where the efficiency comes from.

**The per-step algorithm (paper Algorithm 1):** (1) bottom-up, decide each `eⁿ` (assuming levels above are static),
stop at the first frozen level `K`; (2) top-down, compute the state posteriors for the active levels `K…1`; (3)
decode. *Code:* `_step` does this as two sweeps — Sweep A (static-context e-decision, bottom-up + nested cumulative
product) and Sweep B (top-down state sampling). Because `K_t` differs **per video in a batch**, we cannot
Python-`break` per sample; instead the unroll is **dense with masks** — every level is computed, then gated by its
`eⁿ` (`s = e·s_change + (1-e)·s_prev`), and the nested rule is a cumulative product down the levels. This is the
main engineering subtlety and is mathematically equivalent to the paper's break-at-K.

---

## 6. The objective — the ELBO, in three plain sentences (paper §2.5, Eq. `eq:ELBO` a/b/c)

Maximising data likelihood is intractable, so DLH maximises a variational lower bound (ELBO) with three terms:

- **(a) reconstruction** `E_q log p(o_t | s⃗_t)` — *"decode the inferred states back to the frame and match it."*
  Implemented as a Gaussian negative-log-likelihood with a **fixed**, per-dataset standard deviation (paper §A.1).
  *Code:* the `recon` term in `loss()`.
- **(b) state KL** `Σ_n KL[ q(sⁿ_t) ‖ p(sⁿ_t | eⁿ_t) ]` — *"keep each inferred state close to whichever prior
  component (static or change) was selected."* The paper expands this (Eq. `ELBO-b-breakdown`) as
  `q(e=0)·KL[q‖static] + q(e=1)·KL[q‖change]`; with the hard `e`, exactly one term is active per level.
  **This term is what gives a frozen level gradient** (via `KL[q‖static]`) — getting this right was the fix that
  un-stuck our top level. *Code:* `skl_level[n] = e·KL(q‖change) + (1−e)·KL(q‖static)`.
- **(c) transition KL** `Σ_n KL[ q(eⁿ_t) ‖ p_θ(eⁿ_t | dⁿ) ]` — *"train the little prior-factor model to predict the
  change decisions the KL rule made."* At rollout time (no observations) there is no KL rule, so the model samples
  `eⁿ` from this learned prior factor — that is how the dynamic clock runs **at generation**. *Code:* `trans_kl`,
  using `bernoulli_kl_hard`.

The three are summed over time, the KL terms scaled by a warmup weight `β`, and divided by sequence length.

---

## 7. The training recipe (paper §A.1)

- Sequences of length **100**, batch **100** (we use 32–50 to fit 16 GB), **Adam** lr **5e-4**, ε **1e-4**.
- **β-warmup:** the KL weight is annealed **0 → 1 over the first 10k iters** — "learn good reconstructions before
  being severely restricted in latent complexity" (paper §A.1). *Code:* `beta = beta_max·min(1, step/warmup)`.
- Decoder predicts Gaussian means with a **fixed std tuned per dataset** (`dec_stddev`).
- 3 levels for MMNIST/KTH/Mazes, **2 levels for Moving Ball**. ~3 days on a V100 (we target the *mechanism* +
  ballpark, not 3-day parity — see [PLAN_DLH_REPRODUCTION.md]).

### One addition that is OURS, not the paper's: the dense warmup
A from-scratch cold-start problem (not discussed in the paper): an untrained top posterior sits at ≈`N(0,1)`,
which is ≈ the static prior, so the KL rule never fires a first change, so the level never gets gradient — a
dead-unit trap. We add a configurable **`force_tick_iters` dense warmup**: force *all* levels to tick for the first
few thousand iters so every level learns a meaningful posterior/prior, then hand control to the learned KL-selection.
This is a standard bootstrap for adaptive-computation models; it is clearly flagged in the code as a deviation.

---

## 8. What "working" looks like (the targets to reproduce)

1. **The dynamic structure exists** — `K_t` varies over a rollout; per-level state-KL shows the **auto-collapse
   signature** (paper Appendix C, Table 4: on Moving Ball, KL ≈ L1 35 / L2 6.8, higher levels → ~0). DLH "naturally
   tends to use the minimum number of levels" — a low top-level KL is a *feature*, not a bug; a *zero/dead* top
   level is the bug.
2. **Long-horizon identity** — open-loop rollouts keep object identity/position over hundreds of frames (paper §3.2,
   Fig. 3). Judge with foreground-MSE / LPIPS / a classifier **and** SSIM/PSNR, never SSIM alone on mostly-black
   MMNIST (our hard-won lesson — a black frame scores ~0.76 SSIM there).
3. **Ballpark numbers** — Moving MNIST best-of-100 SSIM toward 0.78 / PSNR 15.7 (paper Table 1); we expect a gap
   from the compute/data budget, as the CW-VAE repro landed 0.571 vs 0.68.

---

## 9. Code map (one line each)
- `config.py` — every dimension/width/schedule, paper Table 2 + §A.1 defaults.
- `modules.py` — conv image enc/dec (from CW-VAE port), Gaussian/Bernoulli KL, `GaussianHead`.
- `model.py` — `DLH`: bottom-up features, the two-sweep masked dynamic unroll (`_step`), the 3-term ELBO (`loss`),
  open-loop rollout (`open_loop`).
- `data.py` — synthetic Moving Ball generator + MMNIST loader.
- `train.py` — Adam + β-warmup + dense-warmup, full wandb logging.
- `eval.py` — best-of-N open-loop, SSIM/PSNR + foreground-MSE + trivial baselines.
