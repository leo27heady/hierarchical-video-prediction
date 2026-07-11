# Chapter 5 — Generalization of the selected configuration across datasets

> **Historical chapter source.** Moving-MNIST and TaxiBJ are descriptive boundary/competence
> anchors. Legacy Phase C claims are blocked; only the persistent-object v2 protocol may support
> identity, trajectory, coverage or equivalence language.

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
- **History packaging → cover the relevant span with the least lossy measured family.**
  The window law dominates: extending the window via coarse tokens buys −0.027 (G-S3), an order
  above the abstraction cost; coarse history is statistically free under the regression head
  (+0.0008 [−0.0013, +0.0029] at 42% token savings, factorial G-S1) and a small real cost under
  diffusion (−0.0082 [−0.0115, −0.0050]). Under the diffusion head, temporally subsampled
  full-resolution history beats the coarse package at seed 0 while using fewer tokens; replication
  and the regression-head twin are pending. Accordingly, coarse history is a measured economical
  option with per-level decodability, not the universal transfer default (§4.4).
- **Transfer rules carried from the boundary study (§4.5):** a *fresh autoencoder per dataset*
  (the AE-reuse trap destroyed positional landmarks on hgrid and is a standing rule); per-dataset
  persistence and AE-round-trip ceiling rows in every table (the honesty devices); the target
  benchmark's exact metric convention, bit-verified through `hvp_rev/eval_adapters` before any
  model number is quoted.

**The generalization question tests the scope of the memory-horizon thesis.** Testbed A supplies
the measured re-entry case; the short 10→10 Moving-MNIST bridge and TaxiBJ supply measured
null/economy-only boundary observations. Those short protocols do not prove that an open dataset
lacks longer-lived state: Phase C separately tests whether persistent digit identity and velocity
become memory-relevant over 300-frame rollout. What is already shown to generalize is the
measurement recipe (heads per regime, window packaging, floors/ceilings protocol), not a positive
hierarchy effect.

[TABLE ch5-0: dataset × memory-horizon presence × predicted hierarchy outcome — loopmom (present;
WIN, measured, two seeds); hgrid3 (present beyond self-feed trust; bounded, measured, §4.5 pair
outcome); Moving MNIST (no re-entry horizon at 10→10; null MEASURED §5.3; persistent-state
demand at 300 frames TESTED by Phase C); 2D rotations
(absent — object always fully visible; null [EXPECTED]); 3D polycubes (weak — self-occlusion
cycles content out of view; small hard-bin edge possible [EXPECTED]); MMNIST-CIFAR (absent;
null [EXPECTED]); TaxiBJ (absent — global field, nothing leaves; MEASURED §5.7: flat-reg wins at
2× tokens, hier = economy-only, recipe lands in the leaderboard band on MAE/PSNR); KTH-64 (weak —
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
3. **The flat ≈ hier null (Δ 0.4 MSE ≈ 0) is the measured short-protocol boundary, not a
   failure.** Moving MNIST has no off-screen re-entry in 10→10, so this protocol does not expose
   the re-entry-defined demand isolated by testbed A. It does not rule out a different long-lived
   state demand: persistent identity and velocity are tested separately by Phase C. Thus the
   short bridge is a boundary observation, while the 300-frame experiment determines whether
   that boundary persists at long horizon.

[FIG ch5-1: qualitative 10→10 strips, mm-flat vs mm-hier vs GT at h1/h5/h10 — assets EXIST:
`logs_rhcp/mmnist_bridge_mm-{flat,hier}_s0_t15000.png`. Caption: sharp committed digits, both
arms; the measured short-protocol null is visible — no re-entry-defined demand separates them.]

### 5.3.1 Long-horizon control: ~300-frame rollouts on standard Moving MNIST

The 10→10 protocol above tests the short-horizon regime; the hierarchical literature's canonical
claim on this dataset, however, is *long-horizon* retention — CW-VAE reports 1,000-frame
open-loop rollouts. This subsection extends the negative control *toward* that regime —
300 frames under our own protocol, flat versus the two-level spatial package only: a boundary
control, not a CW-VAE reproduction (their 36-context/1,000-frame protocol, and the strided/depth
shapes, are not tested here). The flat and
two-stage configurations at the headline 1,152-token budget, retrained on standard bouncing-digit
clips (OpenSTL-matched generator; zero-action conditioning, disclosed — the action interface is
inert on actionless data), rolled out ~300 frames on long held-out clips under the full §2
protocol. Two protocol notes are inherited and disclosed: the re-entry hardness axis is inert
here (nothing leaves the frame), and the SSIM display panel follows the §2.8 carve-out
(calibration with persistence and all-black floors, not evidence).

[EXPECTED-PhaseC (primary — extending the measured 10→10 null): replicated practical
equivalence over three training seeds. For fg-MSE, each seed's paired clip-cluster 95% CI must
lie wholly within ±0.005 on the primary ALL endpoint; horizon bins are secondary diagnostics.
Factor-metric margins fixed before launch: identity ±0.02 accuracy, trajectory ±1.0 pixel, count
deviation ±0.05 digit/frame. Text on confirmation: "persistent identity and velocity remained
recoverable from recent context under this protocol, so the spatial long-span stream bought no
measurable advantage and the two configurations degraded together." ALTERNATIVE (a hier win):
triggers a data/protocol examination first; if the run is sound,
it is reported as a genuine finding — the short 10→10 bridge did not expose a memory-relevant
regime (Moving MNIST carries persistent identity and velocity state), while the long-horizon
factor metrics did — extending the boundary account rather than contradicting it; all three
training seeds are already unconditional (pre-registered amendments, 2026-07-10).]

[FIG ch5-2: fg-MSE vs horizon, flat vs hier overlapping [EXPECTED], persistence + all-black
floors; optional SSIM calibration panel. Asset: `docs/figures/fig15_mmnist_control.png`
(QUEUED — Phase C).]

## 5.4 Author's 2D rotations — the [A2] testbed

> **Removed from the article core (approved triage 2026-07-10):** this suite is
> carried by the dissertation programme (Розділ 4 §4.6.x — strike-or-fill before the
> defense); per the shell end-state rule, no projected content ships in the article.
> Section number retained to keep cross-references stable; merge/renumber at venue
> formatting.

## 5.5 3D polycube rotations

> **Removed from the article core (approved triage 2026-07-10):** this suite is
> carried by the dissertation programme (Розділ 4 §4.6.x — strike-or-fill before the
> defense); per the shell end-state rule, no projected content ships in the article.
> Section number retained to keep cross-references stable; merge/renumber at venue
> formatting.

## 5.6 MMNIST-CIFAR

> **Removed from the article core (approved triage 2026-07-10):** this suite is
> carried by the dissertation programme (Розділ 4 §4.6.x — strike-or-fill before the
> defense); per the shell end-state rule, no projected content ships in the article.
> Section number retained to keep cross-references stable; merge/renumber at venue
> formatting.

## 5.7 TaxiBJ — real data, 32×32×2 — MEASURED (2026-07-07)

**Protocol (as executed; harness `hvp_rev/taxibj_bridge.py`, chain log
`logs_rhcp/taxibj_bridge_chain.log`).** Real Beijing taxi-flow fields, 32×32, 2 channels
(in/out flow), 4→4 prediction. Metric normalization RESOLVED (the pre-registered [VERIFY]):
OpenSTL's dataloader rescales TaxiBJ to $[0,1]$ via $(X{+}1)/2$ with mean 0 / std 1, and the
whole leaderboard convention lives there; scoring calls `openstl.core.metrics.metric` verbatim
(spatial_norm=False). **Build amendment (disclosed, pre-registered before the runs):** instead
of a native-32-px variant, frames are ×2-nearest-upsampled through the UNCHANGED 64-px
HierConvAE(2-ch) + WindowDiT 22M stack and predictions are 2×2-mean-downsampled back before
scoring — nearest-×2 is exactly inverted by 2×2-mean, so the task is information-identical and
"the unmodified WindowDiT" stays literally true. Regime: fresh AE (12k steps), arms B64 × 15k
(≈48 epochs — comparable to the leaderboard's 50-epoch protocol), batch chosen by VRAM probe
(B=64 → 9.8 GB peak; B=96 overshot the 14 GB target), lr 3e-4 not batch-scaled (disclosed),
seed 0, best-of-1, unconditional (constant no-op action). No memory horizon and no long context
to package (4 context frames) — the hierarchy factor is out of scope by construction; the hier
arms are a token-efficiency read (4 coarse-frames + 1 fine = 128 ctx tokens vs flat's 256).

**Table 5.5 — TaxiBJ 4→4, MEASURED (n = 500 canonical test pairs; leaderboard rows locally
verified, 50-epoch OpenSTL retraining, `traffic_benchmarks.md`).**

| row | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| **txb-flat-reg (256 tok)** | **0.3642** | **14.96** | 0.9781 | **39.83** |
| txb-hier-reg (128 tok) | 0.4170 | 15.44 | 0.9772 | 39.64 |
| txb-hier-diff (128 tok) | 0.4986 | 18.37 | 0.9692 | 37.87 |
| txb-flat-diff (256 tok) | 0.5152 | 18.44 | 0.9686 | 37.85 |
| AE round-trip ceiling | 0.0192 | 4.48 | 0.9987 | 50.92 |
| persistence | 3.8268 | 48.63 | 0.7853 | 30.20 |
| ConvLSTM-S (OpenSTL, 50 ep) | 0.3358 | 15.32 | 0.9836 | 39.45 |
| PredRNN (OpenSTL, 50 ep) | 0.3194 | 15.31 | 0.9838 | 39.51 |
| MIM (OpenSTL, 50 ep) | 0.3110 | 14.96 | 0.9847 | 39.65 |
| SimVP+gSTA-S (OpenSTL, 50 ep) | 0.3246 | 15.03 | 0.9844 | 39.71 |
| TAU (OpenSTL, 50 ep) | 0.3108 | 14.93 | 0.9848 | 39.74 |
| PhyDNet (OpenSTL, 50 ep) | 0.3622 | 15.53 | 0.9828 | 39.46 |
| [A2] (published) | [AUTHOR: insert from A2 — TaxiBJ appears in the dissertation also as an [A2] receipt per decision Q4] | | | |

**Readings (measured; the pre-registered primary branch confirmed, stronger than expected).**
(1) **The recipe transfers to real data — and lands inside the leaderboard band on two of four
metrics.** The regression arm's MAE 14.96 ties MIM for second-best of the quoted rows (TAU
14.93); its PSNR 39.83 exceeds every quoted leaderboard row; its MSE 0.3642 sits at the band's
lower edge (0.55% above PhyDNet's 0.3622, 17% above TAU); SSIM 0.9781 is below the band
(0.9828–0.9848). Stated per-metric and without a leaderboard-superiority headline (§5.2 rule);
still, a 22 M *generalist* latent stack, architecturally unchanged from the synthetic testbeds,
matching specialist per-benchmark regressors on MAE/PSNR at comparable epochs on one 16 GB GPU
is the strongest transfer read this chapter hoped for. (2) **Persistence is beaten ~10.5×** on
MSE (3.827 → 0.364) — the floor row confirms the task is non-trivial. (3) **The AE-ceiling risk
dissolved:** round-trip MSE 0.0192 is 16× below the best leaderboard row — the frozen tokenizer
is not the limiter on these fields (the [FALLBACK] scoping branch is measured-moot and deleted).
(4) **Regression ≥ diffusion at best-of-1, confirmed again** (0.364/0.417 vs 0.515/0.499) —
Claim 2's regime statement replicates on real data. (5) **Boundary replication #3 (hier ≈/≤
flat where no horizon exists):** at half the context tokens the hier-reg arm pays −0.053 MSE
(−14%) with a per-horizon curve that degrades faster (h1 0.303 → h4 0.533 vs flat's
0.324 → 0.385) — at a 4-frame protocol coarse history is pure economy, exactly as
pre-registered; notably the *diffusion* twins invert (hier-diff 0.499 ≤ flat-diff 0.515 at half
tokens). Disclosures: single seed (transfer row, not a claim-bearing contrast — campaign rule);
upsample-equivalent build; SSIM deficit stated.

## 5.8 KTH-64 — optional stretch

> **Removed from the article core (approved triage 2026-07-10):** this suite is
> carried by the dissertation programme (Розділ 4 §4.6.x — strike-or-fill before the
> defense); per the shell end-state rule, no projected content ships in the article.
> Section number retained to keep cross-references stable; merge/renumber at venue
> formatting.

## 5.9 The cross-dataset claim, in dual-branch form

> Retitled at the triage: **the article's external sections are measured boundary controls and a
> competence anchor — never positive transfer.**

Three measured statements close the article's external scope, each already stated in its own
section and gathered here without projection. First, the **short-protocol boundary control**:
on standard Moving MNIST with no off-screen re-entry, flat ≈ hierarchical at 10→10 (§5.3,
measured); the long-horizon extension separately tests persistent-state demand (§5.3.1,
[EXPECTED-PhaseC] with identity/trajectory
metrics per the pre-registered mandate). Second, the **competence anchor**: the unchanged 22 M
generalist stack lands inside the OpenSTL TaxiBJ leaderboard band on MAE/PSNR (§5.7, measured),
answering "is this system a toy?" without claiming leaderboard superiority anywhere (§5.2
standing rule; measured concession: 180.91 vs 26.69 aggregate MSE on MMNIST). Third, the
**boundary observation**: in the measured short/no-re-entry regimes the hierarchy factor is a
null (§5.3) or economy-only (§5.7), while the two-seed
CI-clean win exists precisely where the horizon exists (testbed A, §4.3). **No positive-transfer
claim is made**: demonstrating that the win band transfers to natural video requires a
natural-video re-entry protocol with estimated hardness labels, which is future work (§8.5),
not a pending table. The suites formerly projected here (2D rotations, polycubes, MMNIST-CIFAR,
KTH) are carried by the dissertation programme (§§5.4–5.6, 5.8 stubs) under the strike-or-fill
rule.

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
