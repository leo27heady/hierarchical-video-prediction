# Chapter 5 — Generalization of the selected configuration across datasets

> **Status (2026-07-06):** §5.3 (Moving-MNIST) is MEASURED and FINAL. All other dataset sections
> are protocol + pre-registered expectation only: every unmeasured value is marked [EXPECTED],
> [PENDING] or [AUTHOR: …] — no bare expected numbers anywhere in this chapter. Placement
> RESOLVED (2026-07-07, per CH1_INTRO.md canonical order): this IS Chapter 5; reversibility is
> Chapter 6 (`CH6_REVERSIBILITY.md`), related work Chapter 7, conclusion Chapter 8. `docs/NEXT_EXPERIMENTS_GENERALIZATION.md` did not exist at drafting time; protocols
> and costs below are derived from `docs/draft/T7_BRIDGE.md`, the `hvp_rev/mmnist_bridge.py`
> protocol block, and the locally verified OpenSTL model-zoo docs.

Chapter 4 selected a configuration; this chapter asks whether the selection survives contact
with datasets it was not built on. The test set spans the dissertation's declared corpus: the
author's own 2D-rotation testbed ([A2]), the 3D polycube testbed, and the open benchmarks
Moving MNIST, MMNIST-CIFAR, TaxiBJ, with KTH-64 as an optional stretch. One dataset (Moving
MNIST) is already measured and serves as both the credibility anchor and the *negative control*
for the memory-horizon claim; the rest carry protocols, empty tables, and dual-branch claim
language written before any run.

## 5.1 The selected configuration, stated per regime

Chapter 4's verdict is deliberately conditional, so the "selected configuration" is a small
decision table, not a single model — every entry below is a measured result, not a preference:

- **Best-of-1 rollout / pixel-accuracy scoring → regression head.** reg-long beats the diffusion
  system arm at every horizon band and hard bin at equal sharpness (ALL fg-MSE 0.1682 vs 0.1848;
  direct paired cluster CI on the delta [+0.0110, +0.0227]; §4.2.2). Every OpenSTL-style
  benchmark in this chapter scores exactly this regime (best-of-1, pixel-sum MSE), so the
  regression head is the regime-matched arm for the scored rows.
- **Single-step / coverage / perceptual regime → diffusion head.** Diffusion wins every column
  single-step at matched capacity and NFE (0.0512 vs 0.0664 ALL, best-of-8; monotone in NFE
  0.0546 → 0.0503 while regression is flat at 0.0680; §4.2.1) and carries the sharpness/LPIPS
  behaviour. The diffusion arm is retained on every dataset as the perceptual/coverage row.
- **History packaging → hierarchical coarse + recent-fine at the largest affordable window.**
  The window law dominates: extending the window via coarse tokens buys −0.027 (G-S3), an order
  above the abstraction cost; coarse history is statistically free under the regression head
  (+0.0008 [−0.0013, +0.0029] at 42% token savings, factorial G-S1) and a small real cost under
  diffusion (−0.0082 [−0.0115, −0.0050]). The cheapest strong configuration measured is
  **regression head + coarse history + long window** (reg-long, 0.1682 at 1,152 tokens; §4.4).
- **Transfer rules carried from the boundary study (§4.5):** a *fresh autoencoder per dataset*
  (the AE-reuse trap destroyed positional landmarks on hgrid and is a standing rule); per-dataset
  persistence and AE-round-trip ceiling rows in every table (the honesty devices); the target
  benchmark's exact metric convention, bit-verified through `hvp_rev/eval_adapters` before any
  model number is quoted.

**The generalization prediction is itself a claim of the memory-horizon thesis:** hierarchy pays
where content verifiably leaves the observation and returns (testbed A: two-seed CI-clean win),
and is a null where no such horizon exists. Of the datasets below, *none* of the open benchmarks
has a strong memory horizon — so the pre-registered expectation for the hierarchy factor is a
null nearly everywhere, and §5.3 already confirms it on Moving MNIST. What generalizes is the
*recipe* (heads per regime, window packaging, floors/ceilings protocol), not a hierarchy win.

[TABLE ch5-0: dataset × memory-horizon presence × predicted hierarchy outcome — loopmom (present;
WIN, measured, two seeds); hgrid3 (present beyond self-feed trust; bounded, measured, §4.5 pair
outcome); Moving MNIST (absent — nothing exits the frame; null, MEASURED §5.3); 2D rotations
(absent — object always fully visible; null [EXPECTED]); 3D polycubes (weak — self-occlusion
cycles content out of view; small hard-bin edge possible [EXPECTED]); MMNIST-CIFAR (absent;
null [EXPECTED]); TaxiBJ (absent — global field, nothing leaves; null [EXPECTED]); KTH-64 (weak —
actors exit/re-enter the frame in walking/jogging/running clips; untested [EXPECTED null-to-weak]).]

## 5.2 Protocol template and measured costs

Every dataset section instantiates one template, the one already executed end-to-end for §5.3
(`hvp_rev/mmnist_bridge.py`): (1) freeze the dataset into fixed files with a recorded seed (or
import the canonical test file unmodified); write a zero-filled actions file where the dataset
has no actions (the action embedding takes the constant no-op); (2) train a fresh `HierConvAE`
on training frames only (7k steps for 64×64×1; ceiling row measured from it); (3) register the
arms into `rhvd.ARMS` without touching `rhvd.py` and train at B16×15k bf16 (B16×30k where the
budget allows); (4) evaluate under the benchmark's canonical protocol with the bit-verified
metric adapters, always carrying AE-ceiling and persistence rows, plus LPIPS (AlexNet) where the
cached weights permit.

**Measured costs (from the §5.3 chain, single RTX 4060 Ti 16 GB, `logs_rhcp/
mmnist_bridge_chain.log`):** per-arm training at ≤704 context tokens runs ≈0.2–0.25 s/step
(cf. the factorial's measured 0.25 s/step at 736 tokens), so a 15k-step arm is ≈1–2 h; the full
10,000-sequence canonical eval is ≈57–65 min per arm; the complete chain (generation, AE, two
arms, two full evals) fits in a single GPU-day. Reference for larger contexts: a 3,392-token
sequence measures ≈0.55 s/step (`docs/HANDOFF_2026-07-06.md` §6). Budget rule for this chapter:
one GPU-day per new 64×64 dataset (two arms), two GPU-days where reg+diff twins are both run.

**Standing honesty rule (applies to every section below).** No claim of the form "beats the
OpenSTL leaderboard on aggregate MSE" is drafted anywhere in this chapter: the measured bridge
already refutes it on Moving MNIST (ours 180.91 vs SimVP+gSTA-S 26.69), and the structural
reasons — a 22 M stochastic latent-diffusion *generalist* at ~15k steps, scored best-of-1 on a
pixel-sum metric that punishes sampling variance, against deterministic pixel-regressors trained
~10× longer and tuned per-benchmark — transfer to every OpenSTL benchmark. The claims this
chapter is allowed to make are: beats persistence on every metric; long-horizon persistency
(per-frame error growth); perceptual (LPIPS) competitiveness; and comparison against the [A2]
reversible-transformer baselines at a fraction of training compute.

## 5.3 Moving MNIST (OpenSTL 10→10) — MEASURED, FINAL

The community-standard protocol, reproduced exactly: canonical `mnist_test_seq.npy` test set
(n = 10,000, never trained on), 10 context → 10 predicted frames, OpenSTL metric convention
bit-verified against `openstl.core.metrics`; training on a fixed-seed faithful port of OpenSTL's
generator (GEN_SEED = 0); fresh AE (`hier_ae_mmnist.pt`); unmodified WindowDiT 22 M arms,
B16×15k bf16, best-of-1, 10 frames = 3 AR window steps (4+4+2).

**Table 5.1 — OpenSTL Moving-MNIST bridge, measured (source `docs/draft/T7_BRIDGE.md`,
`logs_rhcp/mmnist_bridge_chain.log`).** MSE/MAE pixel-sum, SSIM/PSNR per-frame mean, full-frame
LPIPS (AlexNet).

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mm-flat (10 fine, 640 tok) | 180.91 | 229.94 | 0.7625 | 13.55 | 0.1320 |
| mm-hier (10 coarse + 6 fine, 544 tok) | 181.29 | 230.45 | 0.7607 | 13.54 | 0.1389 |
| AE round-trip ceiling | 3.30 | 20.79 | 0.9917 | 30.93 | 0.0062 |
| persistence | 286.28 | 337.89 | 0.6528 | 11.56 | 0.2533 |
| ConvLSTM-S (OpenSTL, 200 ep) | 29.80 | 90.64 | 0.9288 | 22.10 | — |
| PredRNN (OpenSTL, 200 ep) | 23.97 | 72.82 | 0.9462 | 23.28 | — |
| PredRNN++ (OpenSTL, 200 ep) | 22.06 | 69.58 | 0.9509 | 23.65 | — |
| SimVP+gSTA-S (OpenSTL, 200 ep) | 26.69 | 77.19 | 0.9402 | (inconsistent in source doc) | — |
| TAU (OpenSTL, 200 ep) | 24.60 | 71.93 | 0.9454 | 23.19 | — |
| [A2] reversible transformer | [AUTHOR: insert from A2, with protocol-match check — 10→10 canonical test?] | | | | |

Leaderboard rows are OpenSTL's own unified 200-epoch retraining, verified locally in the sibling
repo (`OpenSTL/docs/en/model_zoos/video_benchmarks.md`); the SimVP PSNR cell is internally
inconsistent in that document (38.35 vs 22.78) and is therefore not quoted.

Three readings, carried verbatim from the bridge record:

1. **Protocol anchor achieved.** The harness reproduces the standard benchmark exactly, and the
   AE-ceiling row (MSE 3.30) shows the tokenizer is not the limiter — the gap to the leaderboard
   is a dynamics/objective gap, not a decoder floor.
2. **Absolute numbers are modest, with disclosed reasons.** Ours is a 22 M stochastic
   latent-diffusion generalist at ~15k steps, best-of-1; the leaderboard entries are
   deterministic pixel-regressors trained ~10× longer and optimized exactly for the pixel-sum
   metric, which structurally punishes sampling variance. Stated plainly: 180.91 vs SimVP's
   26.69 — we do not compete on aggregate MSE here and do not claim to. The model beats
   persistence on every metric and *halves* its LPIPS (0.1320 vs 0.2533); per-frame error growth
   is bounded (h1 76.3 → h10 243.9, still below the persistence aggregate at h10). Head
   disclosure: both measured rows are diffusion-head best-of-1; under the §4.2.2 reversal the
   regression twin is the regime-matched arm for this protocol — recorded as an observation
   about the recipe, not a promised run; this section is final as measured.
3. **The flat ≈ hier null (Δ 0.4 MSE ≈ 0) is the PREDICTED boundary condition, not a failure.**
   Moving MNIST has no off-screen re-entry — nothing ever leaves the frame — so there is no
   memory horizon for a coarse level to carry. The bridge thereby closes the article's argument
   *on the field's own benchmark*: where the memory horizon is absent, hierarchy is a null
   (consistent with this project's founding MMNIST observations); where it exists (testbed A),
   the matched-budget win is two-seed CI-clean. "When does hierarchy pay" gets its negative
   control for free.

[FIG ch5-1: qualitative 10→10 strips, mm-flat vs mm-hier vs GT at h1/h5/h10 — assets EXIST:
`logs_rhcp/mmnist_bridge_mm-{flat,hier}_s0_t15000.png`. Caption: sharp committed digits, both
arms; the null is visible — no memory horizon to separate them.]

## 5.4 Author's 2D rotations — the [A2] testbed

**Protocol.** Procedurally synthesized rotating polygonal shapes, 64×64, deterministic dynamics,
time-to-predict as the controlled variable — the base test of the reversible-transformer method
[A2]. [AUTHOR: carry the exact generator parameters, train/test sizes, and prediction horizons
from the [A2] publication; the bridge must match them before any comparison row is quoted.]
Import per the §5.2 template: fixed-seed frozen files; rotation rate mapped to the {−1, 0, +1}
action channel where [A2]'s protocol conditions on it, zero-actions otherwise; fresh AE; arms
rot2d-reg (regression, coarse+fine at the largest window the clip length affords — the selected
best-of-1 recipe) and rot2d-diff (perceptual/coverage row); flat token-matched controls to keep
the null interpretable. Scored at the [A2] horizons plus our per-frame persistency curve.
Expected memory horizon: none (the object never leaves the frame) — the hierarchy factor is
pre-registered as a null; the section tests the *recipe* on the dissertation's own base testbed.

**Table 5.2 — 2D rotations (all model cells [EXPECTED]/[PENDING]; no measured values exist).**

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| rot2d-reg (selected recipe) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| rot2d-diff | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| AE round-trip ceiling | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| persistence | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| [A2] reversible transformer (published) | [AUTHOR: insert from A2] | | | | |

**Claims (dual-branch).** [EXPECTED] The selected configuration outperforms persistence and the
[A2] reversible-transformer baseline in accuracy and long-horizon persistency on [A2]'s own
testbed — the deterministic, fully-observed regime is exactly where the regression head's
best-of-1 advantage (§4.2.2) applies, and the stochastic machinery costs nothing when the
posterior is a point. [FALLBACK: if the [A2] baseline retains the accuracy edge on its home
testbed, the claim reduces to: the generalist recipe *matches* the specialized reversible
transformer within CI while adding the stochastic multi-hypothesis capability and the hierarchy
interface — a compatibility result, and the dissertation arc ("article 1 proposed the machinery;
this article measures where it pays") absorbs it without contradiction.] [EXPECTED] Competitive
on SSIM/LPIPS with [A2] at a fraction of training compute (≈1–2 GPU-h per arm, §5.2).
[FALLBACK: report the measured perceptual gap and attribute it via the AE-ceiling row —
decoder floor vs dynamics — before any interpretation.]

## 5.5 3D polycube rotations

**Protocol.** Procedurally generated volumetric objects (PyVista renders) with occlusion and
non-uniform rotation — the second [A2] receipt dataset. [AUTHOR: carry exact generation
parameters and result tables from A2.] Template as §5.2; arms poly-reg / poly-diff plus the
flat-vs-hier pair. This is the one synthetic corpus with a *weak but real* memory horizon:
self-occlusion cycles surface content out of view for a fraction of the rotation period, so a
coarse history spanning a full period can in principle retrieve what the fine window has lost.
The hardness label is approximable (faces hidden for ≥ k frames before re-appearance), weaker
than testbed A's exact label — disclosed as such.

**Table 5.3 — 3D polycubes (all model cells [EXPECTED]/[PENDING]).**

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| poly-reg (selected recipe) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| poly-diff | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| poly-flat (token-matched control) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| AE round-trip ceiling | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| persistence | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| [A2] reversible transformer (published) | [AUTHOR: insert from A2] | | | | |

**Claims (dual-branch).** [EXPECTED] Outperforms persistence and the [A2] baseline in accuracy
and long-horizon persistency; a small CI-clean hierarchy edge on occlusion-re-appearance frames
if the occlusion period exceeds the fine window — the only plausible off-testbed hierarchy win
in this chapter, and it would directly support the memory-horizon thesis on an [A2] dataset.
[FALLBACK: a hierarchy null here is the pre-registered outcome whenever the occlusion period
sits inside the fine window (the §5.3 boundary condition again) — report the measured occlusion
period alongside the null so the prediction is checkable either way.] [FALLBACK for the accuracy
claim: as §5.4.]

## 5.6 MMNIST-CIFAR

**Protocol.** OpenSTL's harder Moving-MNIST variant: two digits over CIFAR natural-image
backgrounds, 10→10, 64×64 RGB. Template as §5.2 with a 3-channel AE variant (fresh
`hier_ae_mmnist_cifar`); arms mmc-reg / mmc-hier / mmc-flat; canonical OpenSTL test protocol and
metric convention. No memory horizon (nothing exits the frame; the background is static per
clip) → hierarchy pre-registered null, extending the §5.3 negative control to textured
backgrounds. The textured background makes this the most LPIPS-relevant OpenSTL cell — and also
removes the 93%-black-background degeneracy that makes aggregate MMNIST metrics reward mean
prediction, so the leaderboard gap is expected to *compress* relative to §5.3 without vanishing.

**Table 5.4 — MMNIST-CIFAR (model cells [EXPECTED]/[PENDING]; leaderboard rows locally
verified).**

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mmc-reg (selected recipe) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| mmc-hier (diff) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| AE round-trip ceiling | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| persistence | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| ConvLSTM-S (OpenSTL, 200 ep) | 73.31 | 338.56 | 0.9204 | 23.09 | — |
| PredRNN (OpenSTL, 200 ep) | 50.09 | 225.04 | 0.9499 | 24.90 | — |
| PredRNN++ (OpenSTL, 200 ep) | 44.19 | 198.27 | 0.9567 | 25.60 | — |
| SimVP+gSTA-S (OpenSTL, 200 ep) | 51.13 | 185.13 | 0.9512 | 24.93 | — |
| TAU (OpenSTL, 200 ep) | 48.17 | 177.35 | 0.9539 | 25.21 | — |
| [A2] (published) | [AUTHOR: insert from A2] | | | | |

**Claims (dual-branch).** [EXPECTED] Beats persistence on every metric and the [A2] baseline on
accuracy and persistency; competitive on LPIPS with the deterministic leaderboard regressors —
the diffusion head's committed textures against regressors' known blur on natural background is
the realistic best case — at a fraction of training compute. [FALLBACK: if LPIPS parity fails,
the retained claim is the persistence-relative LPIPS halving (measured precedent §5.3) plus the
AE-ceiling decomposition; aggregate MSE/SSIM are expected mid-to-back of the leaderboard and are
reported without competitiveness claims per the §5.2 standing rule.] [EXPECTED] mmc-hier ≈
mmc-flat (null) — the negative control replicates on textured backgrounds. [FALLBACK: any
CI-clean hier-vs-flat separation here would *contradict* the memory-horizon account and must be
reported as an anomaly with a mechanism hunt, not quietly celebrated.]

## 5.7 TaxiBJ — real data, 32×32×2

**Protocol.** Real Beijing taxi-flow fields, 32×32, 2 channels (in/out flow), 4→4 prediction,
OpenSTL convention (metric normalization [VERIFY: TaxiBJ MSE normalization against
`openstl.core.metrics` before quoting any model number]). **Build note (the declared variant):**
this cell requires an AE/model variant, not just a re-run — `HierConvAE` with 2 input channels
at 32×32 (fine latent 4×4, coarse 2×2 per frame at the same patching ratios, i.e. 16 fine +
4 coarse tokens/frame — a 4× token reduction per frame) and the WindowDiT token layout re-derived
for the 4-frame protocol; window steps 4→4 = one prediction window. No memory horizon and no
long context to package (4 context frames) — the hierarchy factor is out of scope by
construction; the cell tests whether the *head recipe and floors/ceilings protocol* transfer to
real, low-resolution, multi-channel data at all. The regression head is the regime-matched arm
(dense aggregate dynamics, best-of-1 MSE scoring); the diffusion twin is the stochasticity probe
on real data.

**Table 5.5 — TaxiBJ 4→4 (model cells [EXPECTED]/[PENDING]; leaderboard rows locally verified,
50-epoch OpenSTL retraining, `traffic_benchmarks.md`).**

| row | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| taxi-reg (variant build) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| taxi-diff (variant build) | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| AE round-trip ceiling | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| persistence | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| ConvLSTM-S (OpenSTL, 50 ep) | 0.3358 | 15.32 | 0.9836 | 39.45 |
| PredRNN (OpenSTL, 50 ep) | 0.3194 | 15.31 | 0.9838 | 39.51 |
| MIM (OpenSTL, 50 ep) | 0.3110 | 14.96 | 0.9847 | 39.65 |
| SimVP+gSTA-S (OpenSTL, 50 ep) | 0.3246 | 15.03 | 0.9844 | 39.71 |
| TAU (OpenSTL, 50 ep) | 0.3108 | 14.93 | 0.9848 | 39.74 |
| [A2] (published) | [AUTHOR: insert from A2 — TaxiBJ appears in the dissertation exclusively as an [A2] receipt per author decision Q4] | | | |

**Claims (dual-branch).** [EXPECTED] Beats persistence clearly and lands within reach of the
leaderboard band on this short-horizon, low-resolution task — the 4→4 protocol removes the
compounding-rollout disadvantage that dominates the 10→10 gap, and the regression head is
exactly the scored regime. [FALLBACK: if the latent-AE detour costs too much on 32×32 fields
(the AE-ceiling row will show it directly), the honest conclusion is that the frozen-latent
generalist recipe does not pay at this resolution and TaxiBJ remains an [A2]-receipt benchmark
only — a scoping result, pre-authorized by decision Q4.] No leaderboard-MSE-superiority claim
under any branch (§5.2 rule); the leaderboard band is 0.31–0.34, tightly packed specialist
regressors.

## 5.8 KTH-64 — optional stretch

**Protocol.** KTH Action, grayscale real video, 10→20 prediction. Disclosure up front: the
OpenSTL leaderboard runs this benchmark at 128×128 (`dataloader_kth.py`, in_shape
[10, 1, 128, 128]; verified locally), while our stack is built at 64×64 — the KTH-64 variant is
therefore **not directly comparable** to the leaderboard rows on MSE/MAE (pixel-sum scales with
resolution) and only loosely on SSIM/LPIPS; the rows below are quoted for orientation only and
so marked. 20 predicted frames = 5 AR window steps — the deepest rollout in this chapter, and
the only open benchmark where the §4.3 persistency machinery (per-horizon curves) has room to
show. Weak memory horizon exists (actors exit and re-enter the frame in walking/jogging/running
clips) but is uncontrolled and unlabeled — no hierarchy claim is staked on it.

**Table 5.6 — KTH (model cells [EXPECTED]/[PENDING]; leaderboard rows at 128×128, orientation
only, locally verified 100-epoch OpenSTL retraining).**

| row | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| kth64-reg | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| kth64-diff | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] | [EXPECTED] |
| AE round-trip ceiling (64×64) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| persistence (64×64) | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| ConvLSTM-S (OpenSTL 128², 100 ep — not comparable) | 47.65 | 445.5 | 0.8977 | 26.99 | 0.26686 |
| PredRNN++ (OpenSTL 128², 100 ep — not comparable) | 39.84 | 370.4 | 0.9124 | 28.13 | 0.19871 |
| MIM (OpenSTL 128², 100 ep — not comparable) | 40.73 | 380.8 | 0.9025 | 27.78 | 0.18808 |
| SimVP+gSTA-S (OpenSTL 128², 100 ep — not comparable) | 45.02 | 417.8 | 0.9049 | 27.04 | 0.25240 |
| [A2] (published) | [AUTHOR: insert from A2 — KTH is an [A2] receipt per decision Q4; note A2's resolution/protocol] | | | | |

**Claims (dual-branch).** [EXPECTED] Beats persistence on every metric with bounded per-horizon
error growth over 20 frames; competitive on LPIPS in *relative* terms — the leaderboard's
deterministic regressors sit at LPIPS 0.19–0.27 at 128² and blur is their known long-horizon
failure mode, while the diffusion head commits (the §5.3 persistence-halving precedent) — at a
fraction of training compute. [FALLBACK: report measured LPIPS with the resolution caveat and
no cross-resolution comparison; if the 5-window rollout degrades (real-video drift), the
per-horizon curve is reported as the finding — this is the stretch cell and is allowed to fail
informatively.] Optional: this cell is explicitly droppable without weakening the chapter
(priority order: 5.4 → 5.7 → 5.5 → 5.6 → 5.8).

## 5.9 The cross-dataset claim, in dual-branch form

Main text, written for the realistic best case: **[EXPECTED] Across the corpus, the selected
configuration outperforms persistence and the [A2] reversible-transformer baselines in accuracy
and long-horizon persistency, and is competitive on perceptual metrics (SSIM/LPIPS) with
specialized deterministic regressors at a fraction of their training compute** (≈1 GPU-day per
dataset, §5.2, vs 200-epoch per-benchmark specialist training) — while the hierarchy factor
itself behaves exactly as the memory-horizon thesis predicts: a two-seed CI-clean win where the
horizon exists (testbed A), a measured null where it does not (Moving MNIST, §5.3), and
pre-registered nulls on every other horizon-free benchmark. [FALLBACK, per clause: if an [A2]
baseline wins its home testbed, the accuracy claim narrows to "matches within CI while adding
stochastic prediction and the hierarchy interface"; if perceptual parity fails on any cell, the
retained claim is the persistence-relative LPIPS improvement plus the AE-ceiling decomposition
of where the loss lives; if a horizon-free benchmark shows a hier-vs-flat separation, it is
reported as an anomaly against the thesis, not evidence for it.] Aggregate leaderboard MSE is
conceded everywhere by design (§5.2 standing rule; measured: 180.91 vs 26.69 on MMNIST) — the
generalist-vs-specialist trade is stated as the price of a single recipe that transfers, and the
one place the recipe is *predicted* to beat any flat specialist arrangement — a genuine memory
horizon at matched budget — is exactly the regime the purpose-built testbed measures (§4.3).

[FIG ch5-2: one row per dataset — ours vs persistence vs leaderboard band (where comparable) on
MSE-normalized-to-persistence and LPIPS; MMNIST row measured, others [PENDING]. Caption: the
recipe's generalization profile — persistence beaten everywhere, perceptual competitiveness,
aggregate-MSE concession disclosed.]

---

*Draft-only provenance notes (remove before submission).* §5.1 recipe numbers: CH4 §4.2.2
(reg-long 0.1682, hier-long 0.1848, delta CI [+0.0110,+0.0227]), Table 4.1 (0.0512/0.0664; NFE
0.0546→0.0503, reg 0.0680 flat), `docs/SPATIAL_FACTORIAL.md` MEASURED RESULTS (G-S1 diff −0.0082
[−0.0115,−0.0050], reg +0.0008 [−0.0013,+0.0029]; G-S3 −0.027; combined-optimum quote); fresh-AE
rule: CH4 §4.5.1. §5.2 protocol: `hvp_rev/mmnist_bridge.py` docstring (canonical test, GEN_SEED 0,
zero-actions, AE 7k, B16×15k bf16, DDIM-16, 4+4+2 windows, bit-verified adapters); costs:
`logs_rhcp/mmnist_bridge_chain.log` (eval mm-flat ≈3,900 s / mm-hier ≈3,430 s extrapolated from
the 8,000-seq marks), `docs/SPATIAL_FACTORIAL.md` cost block (0.25 s/step @736 tok),
`docs/HANDOFF_2026-07-06.md` §6 (0.55 s/step @3,392 tok). §5.3 table: `docs/draft/T7_BRIDGE.md`
verbatim + per-frame MSE h1–h10 from the chain log; leaderboard rows re-read 2026-07-06 from
`C:/Users/leoni/Documents/projects/OpenSTL/docs/en/model_zoos/video_benchmarks.md` (200-epoch
block; SimVP PSNR inconsistency noted there and in the bridge docstring). §5.4–5.5 dataset
identities: `docs/dissertation/DISS_DIGEST.md` §4.1 + `docs/dissertation/rewrite/05_ROZDIL4.md`
§4.1.2 (2D rotations = [A2] base test; polycubes PyVista/occlusion; TaxiBJ/KTH = [A2] receipts
only per decision Q4 — new runs here are ADDITIONS and need author sign-off against Q4 scope).
§5.6 rows: video_benchmarks.md MMNIST-CIFAR 200-epoch block. §5.7 rows: `traffic_benchmarks.md`
50-epoch block; 4→4 protocol stated there. §5.8: KTH 10→20 @128×128 verified in
`OpenSTL/openstl/datasets/dataloader_kth.py` (in_shape [10,1,128,128]); 100-epoch rows from
video_benchmarks.md. LPIPS absent from OpenSTL MMNIST/MMNIST-CIFAR/TaxiBJ tables (KTH only) —
hence "—" cells. All [EXPECTED]/[PENDING]/[AUTHOR] markers are load-bearing: no unmeasured value
may be replaced by a bare number without a named log/npz source.
