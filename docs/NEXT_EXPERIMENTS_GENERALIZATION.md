# Best-model generalization chapter — pre-registered design memo (2026-07-06)

**Status:** DESIGN ONLY — written before any run; awaiting author approval to launch. This is the
author-requested chapter: *take the best measured recipe and evaluate it on 5–6 datasets.* Every
protocol, build item, comparison target, cost estimate and decision rule below is fixed in
advance; the result tables are empty skeletons to be filled verbatim. Numbers quoted from our
campaign are sourced in `docs/HANDOFF_2026-07-06.md`; external leaderboard numbers are sourced to
the LOCAL OpenSTL docs (`C:/Users/leoni/Documents/projects/OpenSTL/docs/en/model_zoos/*.md`) —
anything not locally verifiable is marked **[VERIFY]** or **[AUTHOR: …]**.

---

## 1. The recipe under test — stated honestly from the measured data

The campaign's own tables (HANDOFF_2026-07-06 §4) do not support a single "best model"; they
support a **regime-conditional recipe**, and that is what this chapter transfers:

- **Best-of-1 autoregressive rollout → REGRESSION head.** reg-long 0.1682 beats hier-long-diff
  0.1848 at equal sharpness (direct cluster CI [+0.0110,+0.0227]); reg-flat 0.2015 beats
  flat-budget-diff 0.2038; coarse history is statistically FREE under the regression head
  (spat18-reg Δ +0.0008 [−0.0013,+0.0029] at 42% token savings).
- **Single-step / best-of-N coverage → DIFFUSION head.** T1: diffusion beats regression on every
  column (0.0512 vs 0.0664 ALL; HARD 0.1549 vs 0.1808) with the best-of-N coverage signature.
- **Memory representation → hierarchical coarse+fine at the largest affordable window.** Window
  extension at fixed representation is worth −0.027 (G-S3), dwarfing the abstraction cost;
  the matched-budget hierarchy win is two-seed and both-heads on testbed A.
- **Boundary honesty (pre-registered for every dataset below):** hierarchy pays only where a
  MEMORY HORIZON exists (content leaves and verifiably re-enters). T7 already demonstrated the
  predicted flat≈hier null on standard MMNIST (Δ 0.4 MSE ≈ 0), and the testbed-B pair bounded
  the mechanism at the other end. **None of the datasets below has a loopmom-style memory
  horizon**, so the pre-registered expectation everywhere is *hier ≈ flat*; the chapter's claim
  is **RECIPE TRANSFER** — the frozen-AE + WindowDiT latent-prediction stack trains stably and
  predicts competitively across data regimes on one 16 GB GPU — not hierarchy dominance. Every
  hier-vs-flat null below is a *replication of the article's boundary condition*, and must be
  written as such, never as a failure.

**Arms per dataset (uniform, unless noted):**

| arm | definition | why |
|---|---|---|
| gen-hier-reg | regression head, coarse+fine history at the dataset's largest sensible window | THE transfer arm (cheapest strong recipe measured) |
| gen-hier-diff | same layout, diffusion head, NFE 16 | coverage regime + LPIPS-friendly sampling |
| gen-flat | token-matched flat control | the hier≈flat boundary read |
| persistence | copy last context frame | floor |
| AE round-trip | encode→decode GT | ceiling row (decomposes decoder vs dynamics error — the T7 device) |

**Metrics:** OpenSTL conventions via the bit-verified adapters already built for T7
(`hvp_rev/mmnist_bridge.py`): MSE (pixel-sum), MAE, SSIM, PSNR, LPIPS; plus per-frame MSE-vs-h.
Paired cluster CIs wherever multi-step rollouts produce autocorrelated records.

---

## 2. Shared protocol and build rules

1. **Fresh AE per dataset** (the AE-reuse trap is a measured failure mode — HANDOFF §8.2):
   HierConvAE retrained on the train split only, ~7k steps, <1h each; standardization stats from
   train, test gets `stats=train.stats` (the [STD-FIX] invariant).
2. **WindowDiT 22M (d384/8L/6H) unchanged** at 64×64; two declared variants below (§2.3).
3. **Training budget:** B16×15k bf16 EMA 0.999 (the T7 bridge regime — proven sufficient for a
   protocol anchor) as default; B16×30k where a rollout read matters. Best-of-1 inference.
4. **What must be BUILT (shared infrastructure):**
   - **(B1) channel parametrization** of HierConvAE + decode/viz paths (`in_ch` 1→3 for RGB,
     1→2 for TaxiBJ): ~0.25 day.
   - **(B2) 32×32 resolution variant** (TaxiBJ only): AE maps 64→8/64→4 today; the 32-px
     variant maps 32→fine 16ch@4×4 and coarse 16ch@2×2, and WindowDiT token geometry (16
     fine-tokens + 4 coarse-tokens per frame) must be threaded through window_batch/rollout:
     **~0.5–1 day** — the single biggest build item in this memo.
   - **(B3) per-dataset dump/loader adapters** (shapekit→npy; OpenSTL npz→npy; KTH videos→npy):
     ~0.25 day each.
   - **(B4) conditioning swap:** loopmom arms are action-conditioned; datasets below have no
     actions. Rotation datasets condition on the time/temporal-pattern channel the generator
     exposes; TaxiBJ/KTH/MMNIST-CIFAR run unconditional (action tokens dropped) — the T7 bridge
     already runs this path, so the code exists; per-dataset wiring ~0.1 day.
5. **Seeds:** seed 0 everywhere; seed 1 ONLY on any contrast that ends up carrying a claim
   (two-seed rule, campaign standard).
6. **GPU discipline:** one training job at a time; <15 GB total; all runs logged to wandb
   project `hgrid` with dataset-prefixed arm names (collision lesson, HANDOFF §8.3).

---

## 3. Dataset (i) — 2D rotations (the [A2] TempVerseFormer testbed)

**Generator LOCATED in the author's repo** — no [AUTHOR: provide] needed for code:
- `C:/Users/leoni/Documents/projects/TempVerseFormer/simple-shape-dataset-toolbox` — the
  `shapekit` package (CLI `main.py`): procedural 2D shapes (equilateral/general triangles,
  squares), configurable rotation speed/angle, scene/lighting, PNG/GIF output, metadata.json
  for reproducibility.
- Streaming loader: `TempVerseFormer/tempverse/shape_data_stream/dataset.py`
  (`Random2DShapeCreator`, IterableDataset; returns images + angles + temporal patterns
  (acceleration, deceleration, oscillation period, interruption period)).
- Config (`tempverse/config.py`): render 64×64, **im_channels = 3 (RGB)**, context 12,
  time_to_pred 1–5 (eval configs extend to 12), gradual-complexity schedule
  [0.2,0.4,0.1,0.1,0.2].

**Protocol (two reads, both pre-registered):**
- **(A) [A2]-matched cross-article read:** context 12 frames → predict the frame at
  t+Δ, Δ = 1…12 (the time-agnostic single-shot task of [A2]) at 64×64×3. This is the read that
  lands next to the TempVerseFormer tables in the dissertation (Розділ 4 §4.1.2 carries [A2]
  receipts; this adds the NEW model on the SAME testbed).
- **(B) rollout read:** 12 GT warmup → 24 AR frames, best-of-1, per-horizon MSE/LPIPS.

**Build:** B3 dump script (shapekit → `data/rot2d_{train,test}_*.npy`, fixed generation seeds
RECORDED — the loopmom provenance lesson), B1 RGB AE (in_ch=3, retrain <1h), B4 conditioning:
time-to-predict embedding token replaces the action token for read (A). ~0.5 day total.

**Comparison targets:** the [A2] published results on this testbed (TempVerseFormer vs vanilla
transformer vs LSTM vs pipe-transformer). **[AUTHOR: provide the exact result tables/units from
the NCA 2025 paper — they are not stored in any local repo doc; the trained baselines are on
HuggingFace `LKyluk/TempVerseFormer` and the wandb report is linked in that repo's README.]**
Optional strong move (+2–3h GPU): rerun the author's own `eval.py`
(`configs/eval/rev-transformer.yaml`) on OUR dump so the cross-article comparison shares data
exactly; this removes any generator-version doubt.

**Pre-registered expectations:** deterministic dynamics → the REGRESSION head should dominate
and diffusion measures pure stochastic-head overhead; shapes never leave the frame → **no memory
horizon → hier ≈ flat expected** (boundary replication #2). Failure mode to watch: RGB AE
capacity on thin rotating edges (AE ceiling row will show it).

**Cost:** generation CPU 1–2h; AE <1h; 3 arms × ~1.5–2h (640-token seqs, B16×15k) + evals ≈
**7–9 h GPU + 0.5 build-day**.

**Result skeleton (Table G-1a: [A2]-matched, Δ-sweep; fill per Δ or aggregate):**

| arm | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| gen-hier-reg | | | | | |
| gen-hier-diff | | | | | |
| gen-flat | | | | | |
| TempVerseFormer [A2] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] | [AUTHOR] |
| persistence | | | | | |
| AE round-trip | | | | | |

**Table G-1b (rollout, 24 AR frames):** same rows, columns MSE@h6 / MSE@h12 / MSE@h24 / LPIPS@h24.
**[FIG gen-1: per-Δ error curve, ours vs [A2] baselines. FIG gen-2: qualitative strip at Δ=1/6/12.]**

---

## 4. Dataset (ii) — 3D polycube rotations (the [A2] 3D testbed)

**Generator: same toolbox** — shapekit 3D connected-cube shapes ("configurable number of blocks
and connection complexity", occlusion, rotation about X/Y/Z, optional depth maps; PyVista
rendering — a render-speed notebook exists in the repo). 05_ROZDIL4 §4.1.2 describes the [A2]
usage: "procedurally generated volumetric objects with occlusion and non-uniform rotation".
**[AUTHOR: confirm the exact [A2] 3D generation config — cube count, complexity, rotation
schedule — so the dump matches the published testbed.]**

**Protocol:** identical to dataset (i), reads (A) and (B), 64×64×3.

**Build:** shares every item with (i) (same dump script, same RGB AE class — retrain on 3D
frames, <1h). 3D rendering is slower: budget 2–4h CPU generation **[VERIFY with a 100-clip
timing probe before the full dump]**.

**Pre-registered expectations:** occlusion + non-uniform rotation = the stochasticity axis the
2D set lacks; this is the ONE generalization dataset where the diffusion twin could win a
best-of-N read — pre-register a best-of-8 coverage column (T1 methodology) alongside best-of-1.
Still no memory horizon → hier ≈ flat expected.

**Comparison targets:** [A2] 3D tables **[AUTHOR: provide]**; optional same-data
TempVerseFormer eval as in (i).

**Cost:** ≈ **8–10 h GPU + shared build** (marginal build ~0.1 day given (i) done first).

**Result skeleton (Table G-2):** rows gen-hier-reg / gen-hier-diff (best-of-1) /
gen-hier-diff (best-of-8) / gen-flat / TempVerseFormer [A2] [AUTHOR] / persistence /
AE round-trip; columns MSE / MAE / SSIM / PSNR / LPIPS.
**[FIG gen-3: best-of-1 vs best-of-8 coverage read on occluded rotations — the regime-dependence
claim on the author's own 3D testbed.]**

---

## 5. Dataset (iii) — Moving-MNIST standard (OpenSTL 10→10) — **ALREADY MEASURED (T7), DONE**

Carried over verbatim from `docs/draft/T7_BRIDGE.md` / `logs_rhcp/mmnist_bridge_chain.log`
(2026-07-06, n=10,000 canonical test, bit-verified adapters):

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mm-flat (10 fine, 640 tok) | 180.91 | 229.94 | 0.7625 | 13.55 | 0.1320 |
| mm-hier (10c+6f, 544 tok) | 181.29 | 230.45 | 0.7607 | 13.54 | 0.1389 |
| AE round-trip ceiling | 3.30 | 20.79 | 0.9917 | 30.93 | 0.0062 |
| persistence | 286.28 | 337.89 | 0.6528 | 11.56 | 0.2533 |

Leaderboard anchors (local `OpenSTL/docs/en/model_zoos/video_benchmarks.md`, 200-epoch):
ConvLSTM-S 29.80 / PredRNN 23.97 / PredRNN++ 22.06 / SimVP+gSTA-S 26.69 / TAU 24.60 MSE.
Framing fixed in T7_BRIDGE.md: protocol anchor + disclosed-modest absolutes + **flat≈hier as the
predicted negative control**. **No further spend.** (A 15k→60k training extension to chase the
leaderboard is explicitly NOT recommended — the chapter's claim is transfer, not ranking.)

---

## 6. Dataset (iv) — MMNIST-CIFAR (OpenSTL variant)

**What it is:** Moving MNIST digits composited over CIFAR-10 backgrounds, 64×64×**3 (RGB)**,
10→10, canonical test set. **Local data PRESENT:**
`C:/Users/leoni/Documents/projects/OpenSTL/data/moving_mnist/mnist_cifar_test_seq.npy` (canonical
test) + `OpenSTL/data/cifar10/` (train-side backgrounds) + MNIST train images — so train
generation is fully local.

**Protocol:** exactly the T7 bridge protocol (10→10, n=10,000 canonical, OpenSTL metric
conventions, AE-ceiling row), at 3 channels.

**Build:** extend the `mmnist_bridge.py` faithful generator port with the CIFAR background
compositing (OpenSTL's `mnist_cifar` convention — port faithfully, bit-check a few frames
against the canonical test file): ~0.25–0.5 day. B1 RGB AE retrain (<1h). No other changes —
this is deliberately the cheapest NEW dataset.

**Comparison targets** (local `video_benchmarks.md`, MMNIST-CIFAR section, 200-epoch):

| method | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| ConvLSTM-S | 73.31 | 338.56 | 0.9204 | 23.09 |
| PredRNN | 50.09 | 225.04 | 0.9499 | 24.90 |
| PredRNN++ | 44.19 | 198.27 | 0.9567 | 25.60 |
| MIM | 48.63 | 213.44 | 0.9521 | 25.08 |
| SimVP+gSTA-S | 51.13 | 185.13 | 0.9512 | 24.93 |
| TAU | 48.17 | 177.35 | 0.9539 | 25.21 |

**Pre-registered expectations:** textured moving background = the AE round-trip ceiling becomes
the interesting row (does a 1.85M-class AE carry CIFAR texture? if the ceiling is high, that is
a *decoder-capacity* disclosure, not a dynamics failure — the T7 decomposition device is built
for exactly this). LPIPS is our favorable metric again. No re-entry → hier ≈ flat expected
(boundary replication #3, now on textured video).

**Cost:** generation ~1h CPU; AE <1h; 3 arms × ~2h + 10k-seq evals ~1h/arm ≈
**9–10 h GPU + 0.25–0.5 build-day**.

**Result skeleton (Table G-4):** rows mmc-hier-reg / mmc-hier-diff / mmc-flat / persistence /
AE round-trip / (leaderboard block above quoted for reference); columns MSE / MAE / SSIM /
PSNR / LPIPS. **[FIG gen-4: decoded samples over CIFAR background, pred vs GT at h1/h5/h10 —
the fig8 analogue.]**

---

## 7. Dataset (v) — TaxiBJ (real urban data — the "real dataset" requirement, cheap)

**What it is:** Beijing taxi inflow/outflow heatmaps, **32×32×2 channels**, 4→4 protocol.
**Local data PRESENT:** `C:/Users/leoni/Documents/projects/OpenSTL/data/taxibj/dataset.npz` —
zero acquisition cost, and it satisfies the dissertation's "real dataset" checkbox in one shot.

**Protocol:** OpenSTL 4→4 (context 4 frames → predict 4), MSE/MAE/SSIM/PSNR on the OpenSTL
normalization convention **[VERIFY the de-normalization convention in
`OpenSTL/openstl/datasets/dataloader_taxibj.py` before quoting absolute MSE — the leaderboard
MSE ≈ 0.31–0.34 scale implies normalized units]**. LPIPS is NOT meaningful on 2-channel heatmaps
— drop it, say why (one sentence).

**Build: THE variant build (B2 + B1).** HierConvAE 32×32×2 → fine 16ch@4×4 + coarse 16ch@2×2;
WindowDiT token geometry 16+4 tokens/frame; unconditional (no actions); window = the full
4-frame context (there is no window axis to extend — the protocol is 4 frames total).
**~0.5–1 day build**, then AE retrain ~0.5h.

**Role and honest framing (pre-registered):** at 4→4 there is no memory-horizon question and
barely a hierarchy question (a 4-frame window cannot be "long") — the hier arm here is a
*token-efficiency* read only (20 vs 64 tokens context). The claim this dataset carries:
**the recipe transfers to real, non-visual, 2-channel spatiotemporal data and lands in the
leaderboard's neighborhood on a 16 GB consumer GPU.** Diffusion twin optional; regression arm
is primary (deterministic-ish short-horizon task).

**Comparison targets** (local `traffic_benchmarks.md`, 50-epoch):

| method | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| TAU | 0.3108 | 14.93 | 0.9848 | 39.74 |
| MIM | 0.3110 | 14.96 | 0.9847 | 39.65 |
| PredRNN | 0.3194 | 15.31 | 0.9838 | 39.51 |
| SimVP+gSTA-S | 0.3246 | 15.03 | 0.9844 | 39.71 |
| ConvLSTM-S | 0.3358 | 15.32 | 0.9836 | 39.45 |
| PhyDNet | 0.3622 | 15.53 | 0.9828 | 39.46 |

**Cost:** AE 0.5h; arms are TINY (≈80-token seqs): 2–3 arms × ~1h; eval minutes ≈
**4–5 h GPU + 0.5–1 build-day**.

**Result skeleton (Table G-5):** rows taxibj-hier-reg / taxibj-flat-reg / (optional -diff twins)
/ persistence / AE round-trip; columns MSE / MAE / SSIM / PSNR (+ per-frame MSE h1..h4).
**[FIG gen-5: inflow/outflow heatmap prediction vs GT, h1..h4, one representative frame.]**

---

## 8. Dataset (vi) — KTH-64 (OPTIONAL stretch, with disclosure — or skip)

**What it is:** KTH human actions, grayscale, OpenSTL long-term protocol 10→20 at 128×128
**[VERIFY resolution in `OpenSTL/configs/kth/`]**. **Local data INCOMPLETE:**
`OpenSTL/data/kth/` contains only the `running` class — the full corpus must be downloaded
**[VERIFY size/availability; budget hours, not minutes]**.

**Protocol decision (pre-registered choice, pick ONE):**
- **(a) KTH-64-resized:** resize to 64×64, keep 10→20, run the unchanged 64-px stack, and
  DISCLOSE in every quote that absolute metrics are NOT comparable to the 128×128 leaderboard
  (report persistence-relative and AE-ceiling-relative reads instead, plus LPIPS at 64 px).
  This is the cheap honest option and the one this memo recommends IF the stretch is taken.
- **(b) skip** and state why (resolution mismatch would make every comparison row an asterisk).
  Skipping costs nothing scientifically: real-data transfer is already carried by TaxiBJ.

**Comparison targets, direction-only under (a)** (local `video_benchmarks.md`, KTH 10→20,
100-epoch, at native resolution): PredRNN.V2 39.57 / PredRNN++ 39.84 (SSIM 0.9124, LPIPS
0.19871) / MIM 40.73 / SimVP+gSTA-S 45.02 / TAU 45.32 (LPIPS 0.22856) / ConvLSTM-S 47.65
(LPIPS 0.26686); best leaderboard LPIPS = DMVFN 0.12842.

**Build:** video→npy adapter + person-split protocol port (~0.5 day) + grayscale AE retrain.
**Cost:** data wrangling + ~8–10h GPU (20-frame rollouts) ≈ **1–1.5 calendar days total.**

**Result skeleton (Table G-6, IF run):** rows kth-hier-reg / kth-flat / persistence /
AE round-trip; columns MSE / MAE / SSIM / PSNR / LPIPS, all at 64×64 with the disclosure line
fixed under the table. **[FIG gen-6: 10→20 strip on a held-out person.]**

---

## 9. Total-cost summary

| # | dataset | build (days) | GPU (h) | data acquisition | new claim carried |
|---|---|---:|---:|---|---|
| iii | MMNIST standard | 0 (done) | 0 (done) | done | negative control on the field's benchmark (T7) |
| v | TaxiBJ | 0.5–1 (B2 variant) | 4–5 | LOCAL (dataset.npz) | real-data transfer, 2-channel, 32×32 |
| i | 2D rotations [A2] | 0.5 (shared) | 7–9 | generate (CPU 1–2h) | cross-article continuity with [A2] |
| ii | 3D polycubes [A2] | ~0.1 marginal | 8–10 | generate (CPU 2–4h [VERIFY]) | occlusion/stochastic read; best-of-N on author's testbed |
| iv | MMNIST-CIFAR | 0.25–0.5 | 9–10 | LOCAL (test canonical + CIFAR) | textured-background robustness of the null |
| vi | KTH-64 (OPTIONAL) | ~1–1.5 incl. data | 8–10 | DOWNLOAD required | natural-video stretch, disclosure-bound |
| | **TOTAL (i–v, without KTH)** | **~1.5–2 build-days** | **~28–34 h ≈ 1.2–1.4 GPU-days** | | |
| | **TOTAL (with KTH)** | ~3–3.5 days | ~38–44 h | | |

Seed-1 reserve: +30–50% GPU on whichever contrasts end up claim-bearing (expected: none beyond
recipe-quality rows; the hier≈flat nulls are expectation-confirmations and single-seed suffices
with the pre-registration on record — disclose if so).

---

## 10. Recommended execution order (and why)

1. **(v) TaxiBJ first.** It requires the only real build (B2); everything else reuses B1/B3/B4.
   It adds the highest-value NEW claim (real data) at the lowest data risk (file already on
   disk). If the 32-px variant surfaces design issues, better to hit them before the rotation
   datasets queue behind it.
2. **(i) 2D rotations.** Generator in-hand; the cross-article [A2] tie is the dissertation's
   continuity argument (Розділ 4 §4.1.2 ↔ new §4.x); needs the [AUTHOR] numbers request sent
   EARLY so it doesn't block writing.
3. **(ii) 3D polycubes.** Rides on (i)'s build; carries the best-of-N coverage read.
4. **(iv) MMNIST-CIFAR.** A robustness rerun of the already-measured T7 null on textured
   video — valuable, but confirmatory; schedule it in a gap (its 10k-seq evals overlap nicely
   with a writing day).
5. **(vi) KTH-64 last and only if time allows** — the only dataset with acquisition risk and
   an unavoidable comparability asterisk; the chapter is complete without it.

(iii) MMNIST is already done and anchors the chapter's framing.

---

## 11. Pre-registered readings and risks (fixed before any run)

1. **hier ≈ flat expected on ALL six datasets** (no memory horizon anywhere). Each null is
   reported as the article's boundary condition replicating out-of-distribution. If any dataset
   unexpectedly shows a CI-clean hier win, that is a FINDING (an unmodeled long-range
   dependency) and triggers the two-seed rule before any claim.
2. **Regression primary / diffusion twin** on every dataset; if diffusion beats regression at
   best-of-1 anywhere, report the reversal-of-the-reversal honestly (it would scope Claim 2's
   regime statement, not break it).
3. **AE ceiling owns the absolute-quality risk** (CIFAR texture, KTH detail). High ceilings are
   decoder-capacity disclosures; the dynamics comparison still stands via paired arms — say so
   with the T7 sentence pattern.
4. **Leaderboard gaps are expected and disclosed** with the T7 reasons (22M stochastic
   generalist, ~10× less training, metric structurally favors deterministic pixel regressors);
   the LPIPS and persistence-relative columns are our fair ground.
5. **Provenance:** record generation seeds for every dump (rotation sets, MMNIST-CIFAR train);
   dataset-prefixed arm names; fresh AE per dataset; one wandb project (`hgrid`).
6. **No metric cherry-picking:** all five OpenSTL metrics are reported wherever the convention
   defines them; TaxiBJ drops LPIPS with the stated reason.

## 12. Figure list for the chapter

| # | figure | source |
|---|---|---|
| [FIG gen-0] | recipe card: the regime-conditional best recipe (head × regime × representation), one diagram | drawn |
| [FIG gen-1] | 2D rotations: per-Δ error, ours vs [A2] baselines | Table G-1a |
| [FIG gen-2] | 2D rotations: qualitative strip Δ=1/6/12 | decode |
| [FIG gen-3] | 3D polycubes: best-of-1 vs best-of-8 coverage | Table G-2 |
| [FIG gen-4] | MMNIST-CIFAR: pred vs GT over textured bg | decode |
| [FIG gen-5] | TaxiBJ: inflow/outflow heatmaps h1..h4 | decode |
| [FIG gen-6] | KTH-64 strip (optional) | decode |
| [FIG gen-7] | summary: our arm vs leaderboard band per dataset (normalized to persistence=1, ceiling=0) | all tables |

The normalization of [FIG gen-7] (persistence=1, AE-ceiling=0 per dataset) is the honest
cross-dataset picture this chapter exists to draw: where the recipe lands inside each dataset's
feasible band, independent of metric scale.
