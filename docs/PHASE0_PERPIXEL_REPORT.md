# Phase-0 validation + per-pixel readiness — autonomous-session report

> Written during the 8-hour autonomous window (2026-06-29 → 06-30). Covers: the DLH blur/instability diagnosis
> (verified), the Phase-0 long-training validation (with an adversarial re-review that corrected the framing), the
> ported renderer, and the per-pixel spatial-DLH build + calibration. Decision rule the user set: *if everything is
> totally fine/sharp/as-expected → continue per-pixel without interaction; else → report with suggested fixes/steps.*
>
> **Outcome: "yes-with-caveat" — this is the REPORT branch.** Phase-0 is directionally sound but the headline was
> overstated and one confound is real; per-pixel is well-prepped and its premise holds. Details + recommended next
> steps below. Nothing here is a blocker; all are "do these before locking the baseline / going deep on per-pixel".

---

## 0. TL;DR

1. **Blur = inherent** (Gaussian-L2 objective; C0-proven on frozen codes); **instability = port-config** (sigmoid
   init), already solved by the `tanh` default + the already-committed KL-economy. The dropped `(1-e)KL[q‖static]`
   term is **not a bug**. (Verified by a 40-agent audit + ablations.)
2. **σ=1.0 is the best Phase-0 config** (best fg-MSE 0.470 *and* SSIM 0.595 — they agree), and forcing hierarchy
   engagement via low σ (0.5/0.3) **hurt** accuracy across the board. **BUT** the long-horizon edge is **weaker and
   later** than first framed: σ=1.0 beats CW-VAE on only **47% of frames**, with the durable edge confined to the
   **last third** (frames 101–150, −0.017 fg-MSE). The earlier "beats CW-VAE from frame 23" was a **first-touch
   artifact**. Plus a real **step confound** (σ=1.0 at 8k vs others 12k) that biases toward σ=1.0.
3. **Renderer ported to DLH and works** (FID 208→97 with the GAN), but delivers **sharp-but-identity-imperfect**
   open-loop frames (vs CW-VAE-renderer FID 66) — capped by DLH's 2-level code bottleneck. "Sharpness solved" is
   partly illusory; the residual is dynamics + the bottleneck.
4. **Per-pixel: mechanism works, quality null on MMNIST.** Premise STRONG; spatial model built + code-reviewed +
   fixed + memory-safe; the per-location gate **demonstrably segments the moving digits** (3× motion preference, e-map
   concentrates on digits — `logs_spatial/segmentation_strip.png`). BUT the upper spatial level engages unstably and
   adds NO quality (same capacity-dump-on-clock-null pattern as the temporal hierarchy) → the *quality* demo needs a
   textured/events dataset, not MMNIST. [§5/§8]
5. **Recommended baseline:** σ=1.0 + renderer as the *working* base, but **re-gate at matched steps + ≥3 seeds on a
   CI'd late-window margin** before locking it. Treat "hierarchy hurts / clock-null" as **MMNIST-local** — do not let
   it pre-decide per-pixel/KTH.

---

## 1. Diagnosis (verified earlier; unchanged)

| symptom | root cause | port bug? |
|---|---|---|
| Blur | fixed-σ Gaussian-L2 likelihood (MLE = smooth conditional mean) | **No — inherent.** C0: same frozen codes, swap only objective → lap-var 0.0026→0.042 ≈ GT. Paper concedes it. |
| Instability (sigmoid→black) | decoder init: `sigmoid(0)=0.5` ≫ data-mean → saturates | port-config, avoided by `tanh` default |
| Instability (high-β collapse) | former ELBO defect | already fixed (KL-economy) |
| 2-level collapse | hard-clock pruning + MMNIST clock-null | correct DLH behavior, not a bug |

Bugs fixed this session: `train.py:41` cast every tuple CLI flag to `int` (float-tuple flags crashed); stale
`DLH_EXPLAINED.md:61,153-154` (described a nonexistent static-KL term + `bmean/bstd`).

---

## 2. Phase-0 long-training validation (CORRECTED after adversarial re-review)

Two 12k-iter runs (σ=0.5, σ=0.3; tanh, economy, seq100, bs50) + the σ=1.0 8k baseline, gated on a matched protocol
(ctx36 / horizon150 / best-of-5, 24 clips). **An adversarial review recomputed the long-horizon metric from the raw
curves and corrected two overstatements.**

### Honest 3-way (fg-MSE windows = mean(DLH − CW-VAE); negative = DLH wins)

| σ | iters | fg-MSE-all | SSIM | lap-var | win% vs CW | frames 1–25 | 26–75 | **101–150** | levels |
|---|---|---|---|---|---|---|---|---|---|
| **1.0** | 8k | **0.470** | **0.595** | 0.0033 | **47%** | +0.024 | +0.015 | **−0.017** | 2 |
| 0.5 | 12k | 0.511 | 0.535 | 0.0072 | 14% | +0.102 | +0.059 | +0.004 | 3 |
| 0.3 | 12k | 0.533 | 0.462 | 0.0089 | 7% | +0.143 | +0.084 | +0.024 | 3 |

- **What's robust:** σ=1.0 is unambiguously the most accurate (fg-MSE *and* SSIM agree); it owns the genuine
  late-horizon window (101–150: −0.017, a real ~1.7× the per-frame jitter); engaging the 3rd level via low σ hurts
  every accuracy metric. Stability is solid (no collapse, no fade).
- **What was overstated:** "beats CW-VAE from frame 23" was a **first-touch** artifact — CW-VAE actually wins frames
  1–~75; σ=1.0's sustained crossover is frame **146**. The DLH long-horizon thesis is **real but late and modest**,
  not frame-23-onward.
- **The confound that matters:** σ=1.0 is at **8k iters vs the others' 12k**. Less training = less free-run drift, so
  the comparison is biased *toward* σ=1.0 (at matched 3k steps, σ=1.0 ≈ σ=0.5 on fg-MSE — 0.474 vs 0.475). An unknown
  fraction of σ=1.0's lead is "hasn't drifted as far yet," not "structurally better."
- **Note:** the gate's `beats_cwvae_from` metric is gameable (first-touch) — replace with sustained-crossover /
  win-fraction / late-window margin in future gates.

---

## 3. Renderer (H1) on DLH — works, but honest ceiling

Ported (`hvp_rev/codes_dlh.py`, `train_renderer_dlh.py`; warm-start `img_dec`, free-run codes). L1+LPIPS-only:
OL lap 0.048≈GT but FID 208 + visually **incoherent** (the documented "LPIPS games fg-LPIPS / worst-FID" trap).
Adaptive-λ GAN (4k steps): OL FID **208→97**, lap 0.068≈GT, dacc~0.7 — sharp, in-distribution MNIST strokes, **but
global digit identity imperfect** (vs CW-VAE renderer FID 66). The residual identity gap is capped by DLH's 2-level ×
20-dim code bottleneck — i.e. the same low-capacity regime σ=1.0 selects. So "σ=1.0 + renderer" = *sharp-but-
sometimes-wrong-digit*, not a clean win; the residual is dynamics, not rendering.

---

## 4. Per-pixel spatial-DLH — built, reviewed, memory-safe

- **Premise pre-check: STRONG.** MMNIST: 9.4% pixels change/frame (sparse), 36% static-forever, 60% of changes on
  digits → a per-location gate has a clear localized target. (The temporal-frame clock is null on MMNIST, but the
  *spatial* axis is alive — a distinction I had initially conflated; the user was right that MMNIST is a fine
  starting testbed for per-pixel.)
- **Design:** memory-safe hybrid (`hvp_dlh/model_spatial.py`), red-teamed: N=2, Cs=8, uniform 16×16, ConvGRU,
  1×1-conv Gaussian heads, **maxpool-3×3 parent dilation** on the nested constraint (plain multiply collapses the
  upper level to 0.8% tick), full deterministic-carry of frozen locations, one-sided λ_e target-tick-rate penalty.
- **Code-reviewed (2nd pair of eyes) → 2 blockers fixed:** `persist_margin=0.5` (the collapse cause; → 0.0, matching
  the vector DLH which uses no margin), and a mis-signed λ_e term (pushed toward the static belief; → one-sided
  `relu(soft_rate − target)`). Also fixed: dilated transition-KL mask; softened the "bit-static" claim to
  "latent-static" (the 3×3 decoder leaks ~1px at tick boundaries).
- **Memory:** lean decoder, 0.57M params. **B32 seq50 = 9.3 GB (fits); B32 seq100 = 18.5 GB (OOMs)** → the *final*
  per-pixel training will need gradient-checkpointing or seq50. (The code-review's "widen decoder for sharpness"
  advice was overridden: the renderer handles sharpness, so the per-pixel decoder stays lean.)

---

## 5. Per-pixel calibration — gate segments, but upper level collapsed (schedule artifact → re-running)

**Run 1** (seq50/B32, 4000 iters, force_tick 2000, β-warmup 3000, λ_e 1e-2, margin 0):

| step | phase | top-tick | seg-corr | tick_motion | tick_static | fg-all | top sKL |
|---|---|---|---|---|---|---|---|
| 1000 | force_tick | 1.0 (forced) | **0.38** | **0.52** | 0.12 | 0.611 | 0.01 |
| 2000 | force_tick | 1.0 (forced) | **0.28** | 0.20 | 0.03 | 0.575 | 0.01 |
| 3000 | gate live (β=1.0) | **0.0** | 0.00 | 0.00 | 0.00 | 0.598 | 0.0 |
| 4000 | gate live | **0.0** | 0.00 | 0.00 | 0.00 | 0.432 | 0.0 |

**(a) Segmentation: YES.** During warmup the per-location hard-KL decision (evaluated with the *real* `kl_c<kl_s`
rule, not the forced e) fires far more on moving pixels than static background (tick_motion 0.52 vs tick_static 0.12,
corr 0.38). **The spatial gate mechanism works and segments the digits — the premise is empirically confirmed.**

**(b) Tick calibration: FAILED in run 1.** The moment force_tick ended (step 2000), the upper level went to 0% tick
and stayed dead (the all-persist / dead-unit collapse). **Cause = compressed schedule:** force_tick 2000 with β
already 0.5–0.67 crushed the upper level's state-KL to ~0 *during* the warmup, so when the gate took over there was
no signal to tick on. This is a hyperparameter artifact, not necessarily a fundamental null — the vector DLH's
*working* recipe was force_tick **3000** + β-warmup **6000** (β only 0.5 when the gate engages). **Run 2 with that
recipe is in flight (done ~06:10) to settle whether the per-pixel hierarchy SUSTAINS on MMNIST.**

**(c) Long-horizon match:** N/A in run 1 (upper level dead → effectively a 1-level spatial VAE). Note the whole model
reconstructs fine (fg-all 0.432 at step 4000, comparable to the vector DLH ~0.43–0.47) — the *base* level works; it's
the per-pixel *hierarchy* that didn't sustain under the bad schedule.

**Run 2** (force_tick 3000, β-warmup 6000, 7000 iters — the proven vector recipe):

| phase | steps | top-tick | seg-corr | tick mot/stat | fg-all | sKL [bot, top] |
|---|---|---|---|---|---|---|
| force_tick | 1000–3000 | 1.0 (forced) | **0.53–0.57** | 0.6–0.7 / 0.07–0.10 | 0.63 | [16–43, 23–77] |
| gate live, dead | 3250–5000 | **0.0** | 0.00 | 0/0 | 0.55–0.59 | [27–33, **0**] |
| gate live, **revived** | 5500–7000 | 0.06→**0.22** | 0.29–0.56 | 0.50 / 0.03–0.17 | 0.57–0.60 | [**1.3**, **6.2**] |

Three findings, and they're the honest per-pixel verdict for MMNIST:

1. **Segmentation works (robust across both runs).** The per-location hard-KL gate fires ~3× more on moving pixels
   than static (final-checkpoint tick-on-motion 0.55 vs 0.19; cleaner 0.6–0.7 vs 0.03–0.10 mid-run), and the e-map
   **visually concentrates on the digit region** (`logs_spatial/segmentation_strip.png`). **The user's premise is
   confirmed: the spatial gate learns where the motion is.** This is a bankable qualitative result.
2. **Upper-level engagement is UNSTABLE.** Even with the good schedule, the top level dies right after force_tick
   (steps 3250–5000), then *spontaneously revives* at step 5500+ (tick→0.22, sKL 0→6) with the KL **redistributing
   bottom→top** (32→1.3 / 0→6). So the per-pixel hierarchy *can* engage, but fragilely, and over-ticks (0.22 > the
   0.12 target → background leakage at the end) — λ_e / schedule need real calibration to hold a clean sparse gate.
3. **No quality benefit on MMNIST.** When the top engages, fg-MSE does **not** improve (~0.57–0.60 engaged vs ~0.43
   when it was a dead 1-level) — the **same capacity-dumps-without-benefit pattern** as the σ-sweep and the temporal
   clock. MMNIST has no real multi-scale structure for *any* hierarchy (temporal OR spatial); the upper level engages
   but encodes nothing useful.

**Per-pixel verdict (MMNIST): mechanism ✓ (segments digits, built, reviewed, validated) — quality payoff ✗ (null,
exactly like the temporal hierarchy).** This is the predicted outcome and matches the agreed plan: the per-pixel
*quality* demonstration needs a dataset with real spatially-localized multi-scale events (textured background /
events), NOT full MMNIST. The segmentation figure + the working mechanism are deliverable now; the "beats vanilla on
quality" claim is not, on MMNIST.

---

## 8. BOTTOM LINE / go-no-go

**This was the "report" branch (yes-with-caveat on Phase-0; per-pixel mechanism-positive but quality-null on MMNIST) —
so I did NOT auto-proceed into a full per-pixel training campaign on MMNIST, and did NOT auto-launch the 7-hour σ=1.0
settling run.** What's solid and what to do:

- **Solid:** diagnosis (blur inherent / instability solved); σ=1.0 is the most accurate config; the renderer works
  (FID 97); the per-pixel mechanism is built, code-reviewed, memory-safe, and **demonstrably segments the digits**.
- **Do before locking the Phase-0 baseline:** the matched σ=1.0@12k + ≥3-seed settling run, re-gated on a CI'd
  late-window margin (command below). The σ=1.0 long-horizon edge is real but late/modest and step-confounded.
- **Do for per-pixel:** either (i) calibrate the gate on MMNIST (tune λ_e to a clean 8–15%, add a free-nats floor /
  longer force_tick to stabilize the upper level) to harden the *segmentation* result, and/or (ii) move the *quality*
  test to a textured/events dataset where the spatial hierarchy can actually pay off. Do **not** expect per-pixel to
  beat vanilla on MMNIST.
- **Settling-run command (for you to launch — 7h, won't auto-run):**
  `venv/Scripts/python.exe -m hvp_dlh.train --dataset mmnist --levels 3 --seq-len 100 --batch-size 50 --dec-stddev 1.0 --dec-out-activation tanh --state-kl-mode economy --force-tick-iters 2500 --beta-warmup-iters 6000 --total-iters 12000 --logdir logs_phase0 --wandb-mode disabled --run-name phase0_std10 --seed 0` (repeat seeds 1,2).

---

## 6. Recommended next steps (suggested fixes/steps)

**Before locking the Phase-0 baseline:**
1. **Settling run:** retrain σ=1.0 to **12k iters** (matched to σ=0.5/0.3), ideally **≥3 seeds** each, and re-gate on
   a **clip-bootstrap CI'd late-window (101–150) fg-MSE margin** vs CW-VAE — *not* the first-touch crossover frame.
   (Launched / queued this session; will not finish before you're back — partial trajectory in `logs_phase0`.)
2. **Honest renderer head-to-head:** report DLH+renderer (FID 97, identity-imperfect) vs CW-VAE+renderer (FID 66)
   explicitly; the "sharpness solved" claim must own the residual identity gap.

**Per-pixel (if the calibration passes its 3 questions):**
3. If the e-map segments + tick calibrates + long-horizon matches → proceed to a full per-pixel run with
   gradient-checkpointing (for seq100) and the long-horizon-match gate as the hard success criterion.
4. If the e-map is flat / tick won't calibrate / long-horizon regresses → the per-pixel gate is a null on MMNIST too
   → move the *quality* demonstration to a cheap textured-background set (per the agreed plan; not full KTH).

**Generalization guard:** "engaging the hierarchy hurts / MMNIST is clock-null" is **MMNIST-local**. Do not use it to
conclude σ=1.0 (2 levels) is the right base for KTH/events, where a 3rd slow level may pay off.

---

## 7. Honest caveats carried forward
- Single training seed per config; long-horizon differences are within single-seed noise → need seeds + CI.
- σ=1.0 8k-vs-12k step confound (above).
- Renderer fixes texture, not the dynamics drift / identity bottleneck.
- best-of-100 (paper) vs best-of-5 (here) inflates the paper's headline SSIM — not a port deficiency.
- Per-pixel seq100 needs gradient-checkpointing (memory).
