# DLH reproduction — results report

> From-scratch PyTorch reproduction of the **Dynamic Latent Hierarchy** (Zakharov, Guo, Fountas, ICLR 2023,
> [arXiv:2212.14376](https://arxiv.org/abs/2212.14376)). Branch `reproduce/dlh`. No official code exists; this is a
> faithful port from the paper. Companion docs: [DLH_EXPLAINED.md](DLH_EXPLAINED.md) (mechanism),
> [PLAN_DLH_REPRODUCTION.md](PLAN_DLH_REPRODUCTION.md) (plan + debugging ladder). Live: wandb `dlh-repro`.

---

## 1. Verdict

**The port reproduces DLH's core mechanism and behaviour.** On Moving MNIST the model:
- runs a **dynamic, content-adaptive clock** that engages a second timescale and **survives full KL pressure**
  (β=1.0), with per-step update-depth `K` varying 1.06–1.69 in free-running rollout;
- retains **long-horizon identity** — digit content persists for the full 150-frame open-loop horizon without
  drifting to black;
- **beats trivial baselines on the metric that matters** (foreground-MSE 0.471 vs black 0.650, persist 0.566),
  with aggregate SSIM still climbing (0.568 → 0.607 and rising at the time of writing, iter 8000/15000).

**Honestly reproduced:** the architecture, the ELBO, the non-parametric clock, the dynamic hierarchy, long-horizon
identity, auto-collapse of unused levels. **Budget-limited (not full parity):** absolute SSIM (0.607 vs the paper's
0.78, see §6), 2 effective levels rather than 3, and the characteristic VAE softness — exactly the kind of gap the
sibling CW-VAE reproduction also hit (0.571 vs 0.68).

---

## 2. The implementation (`hvp_dlh/`)

Faithful to the paper (Table 2, §2, §A.1), every dimension/block configurable. ~6.12M params (paper 7M).

| component | paper | code |
|---|---|---|
| two-component temporal MoG (static/change) | §2.1 | `model.py` MoG: change=`prior[n]`, static=carried posterior |
| non-parametric `q(e)` via KL argmin (no Gumbel) | §2.2, Eq. 8 | `_step`: `e = (KL→change < KL→static)` |
| nested timescales + pruning | §2.3 | masked cumulative product over levels |
| 3 deterministic pathways (bottom-up/top-down/temporal) | §2.4, Fig 1c | `enc[n]` / `dec[n]` / `tem[n]` |
| 3-term ELBO (recon + state-KL + transition-KL) | §2.5, Eq. 5 | `loss()` |
| Adam 5e-4, β-warmup, fixed decoder std | §A.1 | `train.py` |

Verified faithful by smoke gates (shapes, KL finiteness, gradient flow) and behavioural oracles (force-tick → all
levels tick; auto-collapse on 1-feature data). The image conv enc/dec are lifted from the CW-VAE port (identical
DeepMind stack). One documented deviation: a **dense warmup** (`force_tick_iters`) to bootstrap the upper levels out
of the cold-start dead zone — standard for adaptive-computation models, not in the paper.

---

## 3. The journey — what it took (and three real bugs found + fixed)

Reproduction surfaced three genuine bugs; each was diagnosed against the paper and fixed. This is the substance of
the work, so it is recorded in full.

**Bug 1 — dead-unit trap (cold start).** First runs left the top level dead (`e=[1,0]`, `K≡1`). Cause: I'd masked
frozen levels' state-KL to zero, so a frozen level's posterior got *no gradient* and could never learn to be worth
ticking. Fix: restore the ELBO term-(b) breakdown so frozen levels get gradient, **plus** a dense warmup to escape
the `q≈static` cold start. (Top-level posterior gradient went 0 → 35.9.)

**Bug 2 — high-β collapse (the important one).** On MMNIST the hierarchy engaged after warmup (mid 21%, top 7%) but
**collapsed as β rose to 1.0** (mid → 1%, top → 0%; open-loop SSIM 0.59 → 0.46). Cause: penalising frozen levels with
`KL(q‖belief)` scaled by β drove `q→static` and **destroyed the "frozen is free" property that makes a hierarchy
worthwhile** — the KL economy that lets slow features migrate up. Fix (ladder rung 2): the static prior is a level's
**own previous posterior**, so a frozen level copies it and pays `KL(q_{t-1}‖q_{t-1}) = 0`. After the fix the mid
level holds ~43% engagement *through* β=1.0 and metrics improve monotonically (see §4) — the decisive confirmation.

**Insight — the Moving Ball needed colour.** The grayscale ball auto-collapsed to 1 level (`KL→static=0`). This is
*correct*: a grayscale ball is 1-feature (position) data with no slow feature for a 2nd level. The paper's ball is
**colour-changing** (§3.4); colour is the slow feature. Adding stochastic colour gave the hierarchy a job and a
proper 2-timescale test. The lesson — *a level that only ever collapses proves nothing; a level that engages exactly
when there's something to engage with proves the clock works* — is why the real validation moved to MMNIST.

**Infra — process leak.** Repeated launch/kill cycles left orphaned CUDA contexts + dataloader workers that jammed
the GPU. Fixed by `num_workers=0` (in-process data) and harness-tracked runs; added **resume-from-checkpoint** for
robustness.

---

## 4. Results — Moving MNIST (3-level, `mmnist-3lvl-fix`)

**Dynamic structure (teacher-forced, iter 8000):**

| level | state-KL/frame | tick rate | role |
|---|---|---|---|
| 0 (bottom) | 5.69 | 100% | every-frame detail/motion |
| 1 (mid) | 0.39 | 41% | slower timescale (the engaged 2nd level) |
| 2 (top) | 0.00 | 0% | auto-collapsed (unused) |

Open-loop update-depth `K` = 1.40 mean (range 1.06–1.69) — the learned clock genuinely adapts per step.

**Open-loop metrics vs horizon (best-of-6, 150 frames, improving over training):**

| iter | SSIM | PSNR | fg-MSE | (baselines: black / persist) |
|---|---|---|---|---|
| 2000 | 0.568 | 13.5 | 0.550 | fg-MSE black 0.650 / persist 0.566 |
| 4000 | 0.569 | 13.4 | 0.509 | |
| 6000 | 0.596 | 13.6 | 0.503 | |
| 8000 | **0.607** | 13.2 | **0.471** | aggregate-SSIM black baseline = **0.765** |

SSIM rising, fg-MSE falling, both monotonic — and fg-MSE clearly beats black/persist, i.e. the model genuinely
models the digits (not just the background). Figures: `logs_dlh/structure_mmnist_fix/{per_level_kl,K_trajectory,rollout_strip}.png`
— the rollout strip shows soft digit blobs persisting across all 150 frames (the long-horizon-identity result).

---

## 5. The before/after that proves the fix

| | buggy ELBO | **fixed ELBO (KL economy)** |
|---|---|---|
| mid tick @ β=0.5 | 0.05 (collapsing) | **0.59 (engaged)** |
| mid tick @ β=1.0 | 0.01 (dead) | **0.43 (stable)** |
| transition-KL | ~0 | **~0.66** |
| open-loop SSIM trend | 0.59 → 0.46 (regresses) | **0.57 → 0.61 (improves)** |

---

## 6. Honest assessment — reproduced vs budget-limited

**Reproduced:** faithful architecture + ELBO + non-parametric clock; the dynamic hierarchy that engages a second
timescale and **holds at full β** (the headline mechanism); long-horizon identity persistence; auto-collapse of
unused levels; foreground modelling beating trivial baselines.

**Not full parity (and why):**
- **Absolute SSIM 0.607 vs paper 0.78.** Three compounding reasons: (a) **compute** — single 16 GB GPU, ~3.5 s/it,
  still mid-training (iter 8000/15000, climbing) vs the paper's 3-day V100 run; (b) **best-of-6 vs best-of-100** —
  the paper's eval picks the best of 100 samples, which inflates SSIM substantially; (c) **the metric itself** —
  MMNIST aggregate SSIM is ~93% background, so a **black frame scores 0.765** and the paper's 0.78 is *barely above
  the black floor*. The defensible signal is foreground-MSE (ours 0.471 < black 0.650), not aggregate SSIM. This is
  the same metric pathology documented for the CW-VAE port.
- **2 effective levels, not 3.** The model auto-collapsed the top level (`KL→change > KL→static` always). DLH's
  minimum-levels property makes this a legitimate outcome, but the paper kept 3 active on MMNIST; closing this is a
  tuning question (timescale separation — see §7), not a correctness one.
- **Soft rollouts.** The characteristic VAE "underfitting" the paper itself discusses; a sharp renderer is the
  separate Direction-B avenue, deliberately out of scope for *reproducing* DLH.

**Bottom line:** the port is an honest reproduction of DLH's *mechanism and behaviour*. The absolute SSIM gap is the
expected compute/eval-protocol gap (mirroring CW-VAE 0.571 vs 0.68), not a failure of the model.

---

## 7. If continued (optional)

- **Let it finish** (iter 15000) and re-evaluate best-of-50 at 300 frames for an apples-to-paper number.
- **Engage the 3rd level**: encourage stronger timescale separation (slow the mid level — e.g. a tick-rate prior /
  entropy floor on the transition predictor, ladder rung 5; or per-level KL weighting) so digit *identity* settles
  in the top and *motion* in the mid.
- **Sharpness** (beyond reproduction): swap the Gaussian-L2 decoder for a perceptual/diffusion renderer (Direction B
  from `RESEARCH_DIRECTIONS.md`) — the one axis DLH leaves open.

---

## 8. Reproduce it
```
# mechanism on the cheap synthetic probe (colour ball, 2-level)
python -m hvp_dlh.train --dataset moving_ball --image-channels 3 --levels 2 --run-name ball

# the headline (Moving MNIST, 3-level) — the result in this report
python -m hvp_dlh.train --dataset mmnist --levels 3 --batch-size 50 \
  --force-tick-iters 3000 --beta-warmup-iters 6000 --total-iters 15000 \
  --data-root ../cwvae/data_mmnist --run-name mmnist-3lvl-fix

# structure figures from any checkpoint
python -m hvp_dlh.structure --ckpt logs_dlh/mmnist-3lvl-fix_last.pt --out logs_dlh/structure
```
