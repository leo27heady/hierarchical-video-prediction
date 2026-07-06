# 05 — РОЗДІЛ 4 (rewrite, phase 1)

> **Status:** English working text of the full РОЗДІЛ 4 replacement (the experimental chapter —
> the old chapter's §§4.2–4.4 were literal "NAME"/"TEXT" stubs and are written new here); the
> Ukrainian pass translates it mechanically. Genre spots marked **[UA-PASS: …]**, author-input
> spots **[AUTHOR: …]**, citation spots **[REF: …]**.
> **Sources of truth:** `docs/draft/CH4_RESULTS.md` (the audited article results chapter,
> AUTHORITATIVE for every number, **including the §4.5 PAIR OUTCOME block of 2026-07-06** — the
> H-grid capability study closed RED; the PASS-branch capability headline is superseded and this
> chapter reports the FAIL/boundary reading ONLY, with the single-step threshold result kept as a
> characterized positive *within* the boundary study), `docs/SPATIAL_FACTORIAL.md` (MEASURED
> RESULTS), `docs/draft/VALIDITY_AUDIT.md` (corrections + disclosures), `docs/draft/CH3_MECHANISM.md`
> (the two-sided EBT study, condensed per author decision Q6(a) = full subsection),
> `docs/HGRID_DESIGN.md` (pre-registered boundary-study design), `docs/dissertation/DISS_DIGEST.md`
> (§6 «НОВИЙ РОЗДІЛ 4», §8 decisions Q4/Q6/Q7, item 35 §4.1 disposition, item 38 acts),
> `docs/dissertation/rewrite/00_FRONT_MATTER.md` and `04_ROZDIL3.md` (vocabulary, cross-references).
> **Disposition of the old Ch.4 (per DISS_DIGEST §3):** old §4.1 is **REWRITTEN/EXTENDED** (the
> [A2] TempVerseFormer receipt datasets are kept as-is per decision Q4; the looping memory-horizon
> testbed becomes the primary instrument; TaxiBJ/KTH survive only as [A2] receipts); §§4.2–4.4
> stubs are **WRITTEN NEW** from the [A4] campaign; old §4.5 (впровадження) is **KEPT** with the
> existing acts wording (TempVerseFormer + entropy tokenization only — no new claims, digest item
> 38); §4.6 becomes §4.7 with real content. Old Ch.3 figures Рис. 3.1/3.2/3.4 land here (§4.2) per
> the disposition table at the end of `04_ROZDIL3.md`.
> **Cross-reference convention:** the article's "Chapter 2" (testbed) = dissertation §3.4; the
> article's "Chapter 3" (mechanism study) = §4.2 of this chapter; the article's reversibility
> chapter = dissertation §§2.3–2.4.

---

# РОЗДІЛ 4 — ЕКСПЕРИМЕНТАЛЬНЕ ДОСЛІДЖЕННЯ ТА ПРАКТИЧНЕ ЗАСТОСУВАННЯ

*(EN: Experimental research and practical application.)*

This chapter performs the comprehensive experimental verification required by **Task 7** of the
dissertation research: the developed methods and the intelligent information technology are
evaluated on the testbed family formalized in §3.4, under the pre-registered protocol of
Table 3.1/Table 3.2, with every quoted comparison paired, stratified by the exact re-entry
hardness label, and interval-estimated by the clip-level cluster bootstrap. The order of the
sections is deliberate. The generation-head study (§4.2) is settled first, because every
subsequent table depends on knowing how much of a system arm's score is owed to its head rather
than to its memory design. The matched-token-budget memory study (§4.3) is the headline result of
the dissertation. The spatial-abstraction factorial (§4.4) isolates the factor that every
hierarchical arm otherwise confounds. The temporal-capability boundary study at the second
operating point (§4.5) closes the experimental program with an honestly bounded negative and the
methodology lessons it produced. §4.6 documents the implementation of the results, and §4.7
concludes the chapter.

Two standing conventions of the chapter, fixed by the validity audit of the experimental campaign
and stated once here: (i) the long-horizon evaluation protocol is a **48-frame ground-truth
warm-up followed by 152 autoregressive frames** (38 scheduler steps × 4 frames) — all text says
"152 AR frames", never "200-frame rollout"; (ii) all confidence intervals quoted in this chapter
are 95% paired **clip-level cluster bootstrap** intervals unless a disclosure states otherwise
(§4.1.5). [UA-PASS: «авторегресійне розгортання на 152 кадри після 48-кадрового прогріву
реальними кадрами»; «кластерний бутстреп на рівні кліпів».]

## 4.1 Інформаційна база, апаратне середовище та протокол вимірювань

*(Information base, hardware environment, and measurement protocol.)*

### 4.1.1 Родина тестових стендів: дві робочі точки

*(The testbed family: two operating points.)*

All experiments of §§4.2–4.5 run on the memory-horizon testbed family of §3.4 at two operating
points.

**Operating point A (primary)** is the momentum looping world *loopmom* formalized in §3.4.1: a
$192 \times 64$ toroidal world observed through a panning $64 \times 64$ viewport at 4 px/frame
($x^{\mathrm{vp}}_t = (x^{\mathrm{vp}}_{t-1} + 4a_t) \bmod 192$; measured
$p_{\mathrm{hold}} \approx 0.10$, $p_{\mathrm{cont}} \approx 0.90$), so ~67% of the world is
off-screen at any instant and off-screen content re-enters with a loop period of ≈48 frames.
Datasets: 3,000 training clips × 100 frames and 500 test clips × 200 frames, clip-disjoint
(§3.4.1); the frozen two-level autoencoder interface, standardization, hardness labels, metrics,
and floors are exactly those of §§3.1.1 and 3.4.2–3.4.4.

**Operating point B (capability regime)** is *hgrid3*: the same generator scaled to a 768-px
world (12 digits, speed 4 px/frame, near-ballistic circulation), whose **effective loop** is
$L_{\mathrm{eff}} \approx 213$ frames (eq. 4.1, §4.5.2) — deliberately beyond any affordable flat
context window, which is what makes it the *capability* rather than *efficiency* operating point.
Operating point B carries its own frozen autoencoder (fine reconstruction MSE 0.0024, coarse
0.0059) and its own re-measured references (Table 4.1); its regime properties are verified before
any model run: median re-entry gap 192 frames, 204 beyond-48 re-entry frames per clip, warm-up
world-coverage 88.5%, and viewport-crop uniqueness 0.004–0.010 (the positional-landmark property
whose destruction is diagnosed in §4.5.1).

**Table 4.1 — Reference floors and ceilings, measured under the identical pipeline per operating
point (never mixed across operating points).**

| Reference | Op. point A (loopmom) | Op. point B (hgrid3) |
|---|---:|---:|
| Persistence floor, fg-MSE (single-step, k = 4) | 0.2367 | 0.2381 |
| Persistence floor, fg-MSE (rollout protocol) | 0.2552 | — |
| Persistence fg-sharp (k = 4) | 0.0515 | — |
| AE round-trip fg-MSE ceiling | — | 0.0108 |
| AE round-trip sharpness ceiling | 0.2412 | 0.2084 |
| Raw ground-truth sharpness | 0.4785 | 0.4218 |
| Collapse threshold (no digit structure below) | ≈0.08 | ≈0.08 |
| Frozen AE reconstruction (fine / coarse MSE) | 0.0020 / — | 0.0024 / 0.0059 |

### 4.1.2 Набори даних методу оборотних перетворень ([A2])

*(The datasets of the reversible-transformation method — [A2] receipts, kept as published.)*

The experimental validation of the reversible-transformer method and time-agnostic
backpropagation (§§2.3–2.4) was published in [A2] and is reported here as its published receipts,
without new commitments: procedurally synthesized **2D rotations** (rotating polygonal shapes,
$64 \times 64$ — the base test of the method, where the time-to-predict is a controlled
variable), **3D polycubes** (procedurally generated volumetric objects with occlusion and
non-uniform rotation), and the open benchmarks **Moving MNIST**, **MMNIST-CIFAR**, **TaxiBJ**,
and **KTH**. **[AUTHOR: carry the exact dataset parameters and result tables from the [A2]
publication text; the old draft's §4.1 description (AdamW, lr $10^{-4}$ cosine, batch 8–32,
100–300 epochs) must be verified against the published paper before it is retained. The old
draft's "A100 80GB cloud" hardware claim is kept only if receipts exist — otherwise delete.]**
Per author decision Q4, no *new* TaxiBJ or KTH experiments are claimed for the hierarchical model
of Chapter 3; those benchmarks appear in this dissertation exclusively as [A2] receipts.

### 4.1.3 Міст до стандартного стенда OpenSTL/MMNIST

*(The OpenSTL Moving-MNIST bridge.)*

**[PENDING — runs launched; fill T7 numbers or soften to "planned" before defense.]** For
continuity with the literature, the developed stack is additionally evaluated on the standard
OpenSTL Moving-MNIST suite [REF: OpenSTL]. The pre-registered expectation, recorded before the
runs: mid-pack aggregate MSE/SSIM against the OpenSTL leaderboard — aggregate metrics on a
≈93%-black-background benchmark reward mean prediction (this project's own measurement: a
constant all-black prediction outperforms a trained CW-VAE [REF: CW-VAE] on aggregate SSIM/PSNR,
§3.4.3) — together with competitive perceptual (LPIPS) behavior and an autoencoder round-trip
ceiling row that decomposes decoder error from dynamics error. The purpose of the bridge is
connectivity to the standard suite, not a leaderboard claim; the memory-horizon claims of this
chapter cannot be tested on Moving MNIST at all, for the degeneracy reasons established in §3.4.

### 4.1.4 Апаратне та програмне середовище

*(Hardware and software environment.)*

All experiments of the [A4] campaign (§§4.2–4.5) were performed on a single consumer-grade GPU
(NVIDIA GeForce RTX 4060 Ti, 16 GB) — a deliberate constraint of the research design: the
resource-efficiency claims of Chapter 2 are made *for* this class of hardware, and the complete
benchmark (data generation, autoencoder training, all system arms, all evaluations) reproduces on
one such device. The software complex is implemented in Python/PyTorch (§4.6; the module map is
given in the appendix **[AUTHOR: appendix letter]**). The memory-consumption measurements quoted
in Chapter 2 (Table 2.3: BPTT 12.2 GB at rollout length $K = 40$ and OOM at $K = 120$ versus the
developed method's ≈1.1–1.2 GB through $K = 160$; the inner-loop panel behind the ≈0.2% no-go
result) were performed in this same environment, on the implemented stochastic hierarchical
recurrent core, and constitute part of the experimental program of this chapter even though their
tables live with the theory they verify.

Training protocol of the rollout arms (operating point A): 30,000 optimizer steps, batch 16,
bf16, EMA decay 0.999, evaluation on EMA weights (§3.2.3); the flat-long arm is the disclosed
exception (§4.3.4). The DiT backbone of every arm is the §3.2.2 configuration (d = 384, depth 8,
6 heads; 22.19 M parameters).

### 4.1.5 Статистичний протокол та загрози валідності

*(Statistical protocol and threats to validity.)*

The evaluation protocol is the §3.4.4 instrument, hardened by a six-surface adversarial validity
audit of the campaign (data/splits, rollout protocol, gate protocol, energy-model protocol,
autoencoder provenance, metrics/statistics; 24 findings, 2 confirmed major, both fixed and
recomputed the same day). The audit's material corrections and standing disclosures are part of
the protocol definition and are stated here once:

1. **Cluster-robust intervals.** Rollout records are 100 clips × 152 consecutive AR frames with
   lag-1 autocorrelation +0.86; frame-iid bootstrap intervals are 2–7× too narrow. All quoted CIs
   are therefore paired clip-level cluster bootstrap intervals. Under the corrected intervals,
   three secondary bin verdicts of the campaign flipped to null; **every headline verdict
   survived**. Records whose ground-truth frame contains no foreground (3,263 of 15,200 = 21.5%,
   where fg-MSE would silently fall back to full-frame MSE) are excluded from all stratified bins
   and reported as their own background-only row.

2. **Corrected hardness anchoring.** An audit-confirmed bug in the mechanism-study hardness
   labels (the target-anchored coverage window counted the k − 1 = 3 bridge frames as covered,
   mislabeling 11–14% of hard samples as easy; hard fraction 0.19 buggy → 0.27 correct) was fixed
   and Table 4.2 recomputed from the saved checkpoints with a single raw-ground-truth metric
   reference for all rows. The memory-gate and rollout evaluations always used the correct
   anchor. All hardness labels are computed from ground-truth geometry, independently of any
   model and identically across arms.

3. **Metric scope disclosures.** fg-MSE scores only ground-truth-sprite pixels; hallucinated
   off-sprite content is invisible to it — it is therefore never interpreted alone but jointly
   with foreground sharpness, decoded-sample review (§3.4.3), and a full-frame error column in
   the final tables **[AUTHOR: add the full-frame column when the final tables are typeset]**.
   Per-horizon *sharpness* series are chunk-batch means duplicated per record — valid for overall
   means and trend curves, never for per-bin or per-clip readings; every sharpness-versus-horizon
   statement in this chapter is so scoped.

4. **Pairing and replication conventions.** All arm-versus-arm comparisons share evaluation
   clips, horizons, and sampling noise (common random numbers); "2 seeds" for the single-step
   memory gate means two *training* seeds sharing one CRN evaluation set (by design, for exact
   pairing). At operating point B the evaluation adds the pre-registered
   **dt-since-last-visible** stratifier (per-record time since the target region was last in
   view, computed from the recorded trajectory; bins dt ∈ (48, 96], (96, 192], > 192, crossed
   with hardness) and a warm-up-SEEN restriction, so that gates are read only on re-entries of
   content that was answerable by construction. [UA-PASS: «стратифікатор за часом від останньої
   видимості (dt)».]

5. **External validity.** The testbed is purpose-built and synthetic; digit assets are shared
   across splits (texture familiarity only — re-entry memory requires clip-specific trajectory
   content, which does not transfer across clips), while trajectories and world layouts are
   split-disjoint. All claims are scoped to the memory-horizon regime the testbed
   operationalizes; the OpenSTL bridge (§4.1.3) anchors short-horizon behavior on a standard
   benchmark. Generation seeds of the shipped dataset files were not recorded (measured
   properties are stated as measured; a frame-hash duplication check found no cross-file clip
   duplication).

## 4.2 Механізмне дослідження за зрівняних обчислень: вибір генеративного ядра

*(The matched-compute mechanism study: selecting the generation core.)*

The claim defended in this section is deliberately conditional: **which prediction head wins
depends on the inference regime; there is no one best head.** Three results establish this — the
single-step coverage study (§4.2.2), its rollout reversal (§4.2.3), and the two-sided study of
the energy-based head (§4.2.4). The section thereby supplies the empirical selection evidence
promised in §3.2 (the diffusion core) *and* the honest boundary of that selection.

**[FIG 4-1 — PLACEHOLDER]**
*Content:* the three prediction objectives compared on one diagram: direct regression (predict
the target latent, penalize distance to the single truth), denoising diffusion (predict the
velocity of a noised target across a schedule), and energy-based descent (learn a scalar
consistency criterion, generate by gradient descent on the candidate); annotate what each
objective does with predictive multimodality (average / commit per sample / descend the
landscape).
*Source:* **ADAPT** (old Рис. 3.2 — the diffusion-vs-EBM objective comparison — retitled to
three objectives including regression, per the `04_ROZDIL3.md` disposition table).

### 4.2.1 Спільний випробувальний стенд механізмів

*(The shared mechanism harness.)*

All Part-A predictors are transformers of one shape (width 256, 6 blocks, 8 heads) over one token
layout — 16 coarse plus 64 fine tokens per frame with joint attention over both levels' tokens
(the §3.1.2 conditioning mechanism) — with parameter counts 7.27 M (regression), 7.53 M
(diffusion), and 7.28 M (energy-based): the differences are the mechanism-specific heads and
embeddings only. Context is 6 frames of both levels; the target is the frame k = 4 steps after
the last context frame; the four bridging actions are embedded through the same conditioning path
in every arm. Training is single-step and leakage-free (ground-truth context and target; no
prediction fed back); batch 48, fp32; the regressor trains 6,000 steps (its loss plateaus at
0.096), the diffusion arm 8,000 steps. Latents are standardized per channel and per level with
training-set statistics — the §3.1.1 protocol requirement, and a finding in its own right: with
raw latents (per-channel scale ≈0.12 coarse / ≈0.16 fine) the diffusion arm *underperformed* the
regressor, because unit-scale noise buried the signal mid-schedule and sampling initialized 4–7×
off-scale; the one-line standardization fix reversed the mechanism ranking. The failure mode is
silent — losses fall, samples look plausible, nothing errors — which is why standardization is
elevated to a protocol requirement rather than an implementation detail.

Inference budgets are matched in network function evaluations (NFE), not step counts: one DDIM
step = 1 forward pass; one energy-descent step = forward + gradient ≈ 2 forward passes; one
regression sample = 1 forward pass (evaluation keeps dropout 0.1 active, so N stochastic passes
yield N samples — MC-dropout, making best-of-N comparable across arms). Best-of-N selection is
performed in decoded pixel space on foreground MSE against ground truth — an **oracle coverage
diagnostic** (§3.4.4), always reported alongside the mean over N. [UA-PASS: «зрівняний бюджет
обчислень (NFE)»; «оракульне покриття».]

### 4.2.2 Одновкроковий режим оракульного покриття: дифузія виграє кожну колонку

*(The single-step oracle-coverage regime: diffusion wins every column.)*

Table 4.2 reports the corrected mechanism comparison (single-step, k = 4, best-of-8, NFE 16;
six evaluation batches of 48; hard split = novel-strip > 0, covering 27% of samples; references
per Table 4.1).

**Table 4.2 — Matched-capacity head comparison, single-step (corrected hardness anchor, single
raw-GT reference).**

| Head | fg-MSE (all) | easy | HARD | mean-over-8 | fg-sharp |
|---|---:|---:|---:|---:|---:|
| Regression (MC-dropout) | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| Diffusion (v-pred, DDIM) | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** |
| Persistence floor | 0.2367 | — | — | — | 0.0515 |

**Table 4.3 — Matched-NFE control (best-of-8 fg-MSE).**

| NFE per sample | Regression | Diffusion |
|---:|---:|---:|
| 4 | 0.0680 | 0.0546 |
| 8 | 0.0680 | 0.0514 |
| 16 | 0.0680 | 0.0503 |

Three observations. First, both heads are far below the persistence floor on every column: the
action-conditioned warp is learned by everything, as expected where the easy region is
deterministic given the action. Second, the separation concentrates exactly where the hardness
stratification says it should: on the easy bin diffusion's advantage is 0.011 absolute (0.0232 →
0.0120), on the genuinely generative HARD bin it is 0.026 (0.1808 → 0.1549), and diffusion is
sharper (0.1886 vs 0.1591 against the 0.2412 ceiling). Third, the advantage is not an
inference-budget artefact (Table 4.3): the regressor is budget-flat at 0.0680 while diffusion
already leads at NFE 4 and improves monotonically. The regressor's mean-over-8 ≈ best-of-8 gap is
near zero — it has *averaged over* the possible futures rather than distributing probability mass
across them; diffusion's best-of-N gap (0.0640 → 0.0512) is the coverage signature of a head that
commits to distinct hypotheses per sample. This is a controlled replication of the
"regression averages, sampling-based generation commits" finding of the stochastic
video-prediction literature [REF: SVG — Denton & Fergus 2018; REF: SAVP — Lee et al. 2018; REF:
DIAMOND — Alonso et al. 2024] under matched capacity, matched conditioning, matched inference
budget, and exact hardness stratification — and its protocol scope is the point: best-of-8
selection is an oracle coverage regime. [UA-PASS: «регресія усереднює — дифузія фіксує
гіпотезу».]

**[FIG 4-2 — PLACEHOLDER]**
*Content:* decoded panels (last context | prediction | ground truth) for the regression and
diffusion arms on identical evaluation inputs; the regression arm hedges the re-entering digit
into a blur, the diffusion arm commits to a sharp, correctly located digit.
*Source:* **ADAPT** (article assets `logs_rhcp/step1_mech_clean_{reg,diff}.png`, compose).

### 4.2.3 Розворот у розгортанні: регресія виграє режим best-of-1

*(The rollout reversal: regression wins the best-of-1 regime.)*

The pre-registered expectation for the rollout campaign was that the regressor, rolled out
autoregressively, would fade toward the mean while diffusion held sharpness. The measured outcome
**refutes** this expectation, and the reversal is reported as such, not reframed: under the
deterministic best-of-1 rollout protocol (§4.1), **the regression arm beats the hierarchical
diffusion arm at every horizon band and every hard bin at equal sharpness** — reg-long vs
hier-long: fg-sharp 0.0741 vs 0.0729; ALL fg-MSE 0.1682 vs 0.1848; direct paired cluster CI on
the ALL delta [+0.0110, +0.0227]. The same holds at the matched flat context: reg-flat 0.2015 vs
flat-budget (diffusion) 0.2038.

The two results reconcile without contradiction once the protocols are stated side by side: the
Table 4.2 diffusion win is single-step **best-of-8** — oracle coverage switched on; the rollout
result is **best-of-1** — coverage off, and every sampled hypothesis is fed back and compounded
over 152 frames. Diffusion wins the coverage regime; regression wins the deterministic best-of-1
rollout regime. Any claim of the form "head X is better" that omits the regime is, on this
evidence, wrong. [UA-PASS: «розворот» — keep the plain word for "reversal"; the genre must not
soften a refuted pre-registered expectation.]

### 4.2.4 Двобічне дослідження енергетичного ядра: охарактеризований негативний результат

*(The two-sided study of the energy-based head: a characterized negative result. Full subsection
per author decision Q6(a).)*

The third candidate head is energy-based descent: Energy-Based Transformers [REF: EBT —
Gladstone et al. 2025] train a scalar energy $E(\text{context}, \text{candidate})$ — reviewed in
§1.3.3 as a global consistency criterion — and generate by gradient descent on the candidate. The
mechanism is attractive for this dissertation's problem precisely because descent steps are a
natural "think-longer" knob for hard re-entry frames. The study below is *two-sided* in that it
decomposes the observed failure into two separable capabilities: whether the trained energy
landscape **orders** candidates correctly (target below mean below noise), and whether descent
from the prescribed initialization can **reach** a structured output. The headline result is that
these dissociate. [UA-PASS: «охарактеризований негативний результат»; «стабільно неправильно
впорядкований енергетичний ландшафт»; «недосяжність шляхом градієнтного спуску».]

**Setup: the official implementation, imported unchanged.** This dissertation's own
reimplementation of the mechanism, trained in the §4.2.1 harness, collapsed to a blank near-mean
output; a reimplementation failure is uninformative, so the study removes the author's code from
the loop entirely. The control module of the software complex imports the reference repository's
model classes unmodified (RMSNorm blocks, rotary positions, causal autoregressive attention, the
prepended descent-step time-embedding token, the bias-free linear energy head, the causal replay
buffer), and transcribes the reference MCMC descent control flow line-for-line (random-noise
initialization, per-step step-size randomization, Langevin noise, gradients with respect to the
pre-projection frame embeddings, single-graph and truncated variants per recipe). Two adaptations
were unavoidable and are flagged: (i) *frame-as-token* — the reference flattens each frame's
latent into one token via a learned down-projection; the identical scheme is applied to the
concatenated standardized latents of §3.1.1 ($16 \cdot 4 \cdot 4 + 16 \cdot 8 \cdot 8 = 1{,}280$
dimensions), with descent performed in the original 1,280-d space exactly as in the reference;
(ii) an additive learned *action embedding* on the candidate tokens (the reference video task is
action-free; this testbed is action-conditioned). One recipe number is re-tuned: the descent step
size $\alpha$, whose published values are calibrated to the reference latent space's gradient
scale, is recalibrated to this latent scale by the reference's own movement principle
($\alpha \cdot |\nabla E| \cdot n_{\text{steps}} \approx$ the initialization-to-target distance).

Both published training recipes are run at the paper's "xxs" video shape — width 384, 6 layers, 6
heads, FFN multiplier 4, context 16 frames; 14.66 M parameters in the frame-token instantiation —
with the paper's scheduler shape, peak learning rate $4 \times 10^{-4}$, weight decay 0.01,
gradient clip 1.0, effective batch 128 (micro-batch 32 × 4 accumulation), 20,000 optimizer steps,
fp32. The **S2 recipe** uses a single training landscape with randomized extra final-landscape
steps, truncated second-order graph, Langevin noise, a replay buffer, and fixed recalibrated
$\alpha = 15.97$; the **S1 recipe** (the configuration behind the paper's published video
generations) uses two descent steps with detach between them, a second-order graph on every step,
no replay, no training-time Langevin noise, and a learnable $\alpha$ (recalibrated initialization
23.95). Every 1,000 steps, held-out probes record: the energies $E(\text{target})$,
$E(\text{mean latent})$, $E(\text{random noise})$ on the final landscape; the cold-descent
output's latent distance to target $d_{\mathrm{tgt}}$ (the mean-to-target distance is 0.964 in
the same units); and the decoded output's foreground sharpness (collapse threshold ≈0.08; ceiling
≈0.24). For S1, the paper's own inference recipe is additionally run at each probe (best-of-4 by
minimum energy, 4 descent steps, inference $\alpha/4$, Langevin noise, probed at both the
current-$\alpha$ and initial-$\alpha$ readings).

**[FIG 4-3 — PLACEHOLDER]**
*Content:* the energy-based arm as tested: reference transformer backbone with the per-frame
scalar energy head and the K-step descent generation loop (random-noise init → energy gradient
steps → candidate), with the two flagged adaptations (1,280-d frame token; action embedding)
marked.
*Source:* **ADAPT** (old Рис. 3.4, condensed per digest item 31; the backbone half of the old
figure is superseded by FIG 3-1/3-2).

**The S2 recipe: a stably mis-ordered landscape.** The S2 run converges (training loss settles at
≈0.597 from roughly step 7,000 to the end) — and its landscape is stably wrong. At **every one of
the 21 probes** from step 0 to step 20,000, $E(\text{mean}) < E(\text{target})$: the batch-mean
latent has strictly lower energy than the true target, with a stable gap of ≈0.6–1.2 energy units
from step 3,000 onward. Cold descent behaves accordingly: $d_{\mathrm{tgt}}$ stays flat at
1.26–1.29 (against the mean's own 0.964 — the output is no closer to the target than the mean
is), and decoded foreground sharpness never leaves the collapse regime (0.008–0.014 across all
probes, versus the 0.08 structure threshold). Twenty thousand converged steps at the paper's
shape, batch semantics, and scheduler do not repair the ordering; the objection that the earlier
negative was an under-training artefact is thereby closed for this recipe. One disclosure: the
batch-mean latent has roughly eight times lower norm than an individual target, so a norm
preference could in principle carry part of the ordering; two probe features rebut a pure norm
artefact — $E(\text{random noise})$ sits far above both throughout, and the S1 recipe flips the
*same* probe to the correct ordering, which a norm effect alone cannot do.

**The S1 recipe: ordering repaired, generation still out of reach.** The S1 run is the
informative one. Its loss falls further (0.361 at 20,000 steps, still improving), the learnable
$\alpha$ drifts upward (23.95 → 33.27), and — unlike S2 — the landscape *becomes correctly
ordered*: $E(\text{target})$ is strictly lowest at **all fifteen consecutive probes from step
6,000 to 20,000**, with a target-below-mean gap ranging from +0.22 to +3.09 energy units across
those probes (the exact flip-onset step rests on one fixed 64-sample probe batch; the
fifteen-consecutive-probe persistence is the claim, not the onset). By the ordering criterion
that indicted S2, S1's energy is *right*. Generation still never materializes: cold descent from
the prescribed random-noise initialization stays at $d_{\mathrm{tgt}} \approx 1.12$–$1.17$, with
foreground sharpness creeping from 0.011 to a maximum of 0.035 — a real but small trend that
remains a factor of ≈2.3 below the structure threshold and ≈7 below the decoder ceiling. The
paper's own inference recipe does no better: $d_{\mathrm{tgt}} \approx 1.19$–$1.24$ and fg-sharp
0.010–0.015 at every probe and both $\alpha$ readings (final probe: 1.207 / 0.0127 at the
current-$\alpha$ reading; 1.233 / 0.0120 at the init-$\alpha$ reading). A correctly ordered
landscape, probed under the publication's own sampler, produces no structured output at this
scale.

**Decomposition: ordering × reachability.** Combining the two runs with the positive-control
probes gives a clean two-factor decomposition. *(a) Landscape ordering is recipe-dependent:* S2
never orders the landscape correctly (21/21 mean-lowest probes at converged loss); S1 orders it
correctly from step 6,000 onward (15/15 target-lowest). The choice between the two published
training recipes decides whether the energy function even prefers the truth. *(b) Descent
reachability from noise fails under both, and is not bought back by inference-time compute.* In
positive-control experiments on the same official model (copy and shift tasks), the trained
landscape is correct by every probe — on copy, $E(\text{target}) = -65.6$ vs mean $-57.2$ vs
noise $-14.8$, and an energy-conditioning matrix identifies the matching context for 8/8 targets
with an off-diagonal gap of +26.4 — yet cold descent still lands blank (fg-sharp 0.030 on copy,
0.016 on shift). Sweeping cold-descent steps 2 → 128 leaves $d_{\mathrm{tgt}}$ flat
(1.131–1.149); warm-starting *at the target* and descending does not hold it (at the operating
step size the output drifts to $d_{\mathrm{tgt}} 0.41 \to \approx 0.55$, settling in a shallow
basin between target and mean — the target is at best a weak attractor along the descent
dynamics); the minimum-energy sample among 16 cold descents remains collapsed (fg-sharp 0.0145);
and a persistence-initialized descent also fails to move toward the target ($d_{\mathrm{tgt}}$
0.835, fg-sharp 0.046, versus 0.348 and 0.088 for a plain regressor on the same task). The two
factors are separable because S1 fixes (a) while leaving (b) intact: what is observed at
16 GB-reachable scale is not "the energy model learns the wrong criterion" simpliciter — with the
generation recipe it learns a usefully ordered criterion whose minima are nonetheless unreachable
by the prescribed sampler from the prescribed initialization. This is consistent with theoretical
accounts of entropy collapse in gradient-descent sampling [REF: mean-field entropy-collapse
analysis, arXiv 2403.08362], cited as context being instantiated, not as a discovered mechanism.

**Scope and non-claims.** This study tests a capability the EBT publication does not claim at a
scale it did not run: the paper's video-section claims concern validation-loss scaling and
energy-as-uncertainty, not sharp generation, and its published video generations come from a
larger model class trained on natural video through a large pretrained autoencoder on multi-GPU
hardware. Nothing here is a reproduction failure, and nothing here contradicts the published
claims; published small-model positives in other domains [REF: EBT-Policy, arXiv 2510.27545] are
likewise untouched. What the study establishes is a **capability boundary with a mechanism
attached**: at the scales reachable in this work (Table 4.4), the energy-descent head does not
convert even a correctly ordered landscape into committed video predictions — while the diffusion
control at roughly half the parameter count (7.53 M vs 14.66 M), on the same frozen latents and
the same standardization, shows recognizable digit structure by 1,500 optimizer steps and at
convergence commits to sharp, correctly located digits (Table 4.2). Whatever resource the
energy sampler is missing at this scale, diffusion does not miss it. Residual limitations are
stated: the scaled runs remain a factor of 3–5 below the reference step budgets and 2 below its
batch size; the S1 sharpness trend (0.011 → 0.035), while far below structure, is non-zero, and a
longer run could in principle continue it — the claim is scoped accordingly.

**Table 4.4 — Configuration gap between this study's fair-scale runs and the reference training
configurations (the extrapolation-risk record).**

| | This study (S2 / S1 arms) | Reference S2 | Reference S1 |
|---|---|---|---|
| Model shape | xxs: d 384, 6L, 6H, FFN 4× (14.66 M with 1,280-d frame tokens) | xxs | xxs |
| Context | 16 frames | 16 frames | 16 frames |
| Steps (warm-up) | 20,000 (5,000) | 60,000 (10,000) | 100,000 (10,000) |
| Effective batch | 128 fresh + 64 replay / 128 fresh | 256 fresh + replay | 256 fresh |
| Step size $\alpha$ | fixed 15.97 / learnable, init 23.95 (recalibrated to latent scale) | fixed 3,000 | learnable 30,000 |
| Data | synthetic 64×64 grayscale; 2,700 clips × 100 frames | natural video, 224×224, pretrained-AE latents | same |
| Hardware | 1× RTX 4060 Ti 16 GB, fp32 | 4 GPUs/node | 4 GPUs/node |

**[FIG 4-4 — PLACEHOLDER]**
*Content:* $E(\text{target})$, $E(\text{mean})$, $E(\text{randn})$ versus training step, S1 and
S2 side by side; S2 stays mis-ordered (mean below target) through 20k converged steps; S1 flips
to the correct ordering at ≈6k and the gap grows — yet neither generates.
*Source:* **NEW** (plot from the probe data; the *conceptual* energy-landscape drawing of old
Рис. 3.1 may serve as an inset explaining what "ordering" means, per the `04_ROZDIL3.md`
disposition).

**[FIG 4-5 — PLACEHOLDER]**
*Content:* cold-descent foreground sharpness versus training step for S1 and S2, with the 0.08
structure threshold and the ≈0.24 decoder ceiling marked; descent output stays in the collapse
regime under both recipes; S1's small upward trend remains ≈2.3× below structure.
*Source:* **NEW** (plot from the probe data).

### 4.2.5 Висновок щодо генеративного ядра

*(Verdict on the generation core.)*

The head ranking is **regime-dependent**: diffusion for the single-step best-of-N coverage
regime, regression for the deterministic best-of-1 rollout regime, energy-based descent for
neither at 16 GB-reachable scale (a two-sided characterized negative: ordering recipe-dependent,
reachability failing under both recipes and under the paper's own sampler). The memory studies
that follow are therefore run with **both** viable heads wherever a conclusion could depend on
the head (§4.3.5, §4.4), so that no memory claim of this dissertation rides on a head choice.

## 4.3 Одновкроковий гейт пам'яті та розгортальне дослідження зі зрівняним бюджетом

*(The single-step memory gate and the matched-budget rollout study.)*

This section verifies the central pre-registered prediction of Table 3.1: at an equal
context-token budget, the hierarchical allocation (long coarse memory + recent fine window) beats
the flat allocation exactly where the memory horizon demands it.

### 4.3.1 Попередній гейт: здатність пригадування виникає пізно у навчанні

*(The pre-test: retrieval emerges late in training.)*

Before any rollout machinery, the single-step memory gate asked whether a long *coarse* context
helps at all: coarse context 48 versus 6 at fixed fine context 6 (1,232 vs 560 tokens including
the target slots), best-of-8, stratified by the exact hardness label. The result is an
emergence-with-training pattern: at 6,000 steps the HARD-bin delta is null on both training seeds
(+0.0014 [−0.0004, +0.0033] and +0.0008 [−0.0009, +0.0025]); at 20,000 steps it is a decisive win
on both (+0.0135 [+0.0080, +0.0196] and +0.0143 [+0.0083, +0.0208]; hard-bin absolute error
0.1053 → 0.0918 on seed 0 — a ≈13% relative reduction), with sharpness tied (≈0.188 both arms):
the win is retrieval, not sharpness. Two disclosures attach: "2 seeds" means two training seeds
sharing one CRN evaluation set (§4.1.5); and these evaluation windows are near-independent rather
than fully clustered, so the audit's worst-case design-effect correction widens the 20k hard-bin
CIs to ≈[+0.005, +0.022] (seed 0) and ≈[+0.006, +0.023] (seed 1) — the win survives. The
methodological moral carries through the whole chapter: **retrieval-from-memory is a late
capability, and short-budget ablations (6,000 steps here) would have spuriously killed it.**
[UA-PASS: «здатність пригадування з пам'яті виникає пізно» — keep "retrieval" as «пригадування/
відновлення з пам'яті» consistently.]

**[FIG 4-6 — PLACEHOLDER]**
*Content:* hard-bin Δ with confidence intervals at 6k vs 20k steps × 2 seeds: null at 6k on both
seeds, CI-clean win at 20k on both — the retrieval capability emerges late in training.
*Source:* **NEW** (data measured; plot from the four memory-gate logs).

### 4.3.2 Матриця розгортання

*(The rollout matrix.)*

Table 4.5 gives the full arm ordering under the rollout protocol (ALL-row fg-MSE over 152 AR
frames after the 48-frame ground-truth warm-up, seed 0; arms per Table 3.1).

**Table 4.5 — Rollout arm ordering, operating point A.**

| Arm | Context | Ctx tokens | ALL fg-MSE | fg-sharp | Note |
|---|---|---:|---:|---:|---|
| flat-long (diff) | 48 fine frames | 3072 (2.7×) | **0.1564** | **0.0996** | B = 8 × 60k sample-matched (caveat, §4.3.4) |
| reg-long | 48c + 6f, regression | 1152 | 0.1682 | 0.0741 | |
| hier-long (diff) | 48c + 6f | 1152 | 0.1848 | 0.0729 | THE system arm |
| reg-flat | 18 fine frames, regression | 1152 | 0.2015 | 0.0512 | |
| flat-budget (diff) | 18 fine frames | 1152 | 0.2038 | 0.0582 | token-matched flat control |
| hier-short (diff) | 6c + 6f | 480 | 0.2260 | 0.0392 | memory ablation |
| persistence | — | — | 0.2552 | — | floor |

Three regularities are visible in the ordering, each defended separately below: (i) **within each
head, the hierarchical long-memory arm beats its token-matched flat control** (hier-long <
flat-budget; reg-long < reg-flat); (ii) **across heads, regression wins the best-of-1 rollout**
(§4.2.3); (iii) **the full-resolution long window wins outright at 2.7× the tokens** (§4.3.4).
Both long-memory arms also beat the hier-short ablation in every bin — long context helps — but
the claim defended here is the *hierarchical packaging* of the budget, which is exactly what the
flat-budget control isolates.

**[FIG 4-7 — PLACEHOLDER]**
*Content:* per-horizon rollout curves, three panels over rollout horizon h: (a) fg-MSE vs h on
all frames and (b) on HARD re-entry frames only, seven series each (hier-long, flat-budget,
flat-long, reg-long, reg-flat, hier-short, persistence floor) with cluster-CI bands; (c)
foreground-sharpness retention vs h (batch-mean series; AE round-trip ceiling 0.2412 marked).
*Source:* **ADAPT** (article asset `logs_rhcp/fig_horizon_curves_loopmom.png` — 3 panels, 7
series, CI bands, verified; relabel axes to the dissertation's notation).

### 4.3.3 Головне парне порівняння за двома посівами: виграш, що вмикається за межею горизонту

*(The headline pairing at both seeds: the win turns on at the horizon boundary.)*

Table 4.6 is the headline of the dissertation: hier-long versus flat-budget at equal 1,152
context tokens, the full horizon × hardness matrix (Δ = flat − hier; positive = hierarchy
better).

**Table 4.6 — hier-long vs flat-budget (seed 0), 152 AR frames after the 48-frame GT warm-up,
paired per-frame fg-MSE, clip-level cluster bootstrap; empty-GT-foreground records (21.5%)
excluded from stratified bins.** Sharpness 0.0729 vs 0.0582; persistence floor 0.2552.

| Horizon | Hardness | n | flat-budget | hier-long | Δ | 95% CI | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| h 1–8 | easy | 434 | 0.0440 | 0.0419 | +0.0022 | [−0.0040, +0.0089] | null |
| h 1–8 | HARD | 150 | 0.1435 | 0.1359 | +0.0075 | [−0.0164, +0.0327] | null |
| h 9–24 | easy | 919 | 0.1216 | 0.1189 | +0.0028 | [−0.0075, +0.0133] | null |
| h 9–24 | mid | 45 | 0.1661 | 0.1624 | +0.0038 | [−0.0121, +0.0182] | null |
| h 9–24 | HARD | 282 | 0.1800 | 0.1728 | +0.0071 | [−0.0110, +0.0260] | null |
| h 25–60 | easy | 2286 | 0.2210 | 0.1989 | +0.0221 | [+0.0080, +0.0368] | **WIN** |
| h 25–60 | mid | 79 | 0.2218 | 0.2036 | +0.0182 | [−0.0049, +0.0437] | null |
| h 25–60 | HARD | 550 | 0.2554 | 0.2209 | **+0.0344** | [+0.0162, +0.0531] | **WIN** |
| h 61+ | easy | 5610 | 0.3060 | 0.2788 | +0.0272 | [+0.0166, +0.0382] | **WIN** |
| h 61+ | mid | 209 | 0.3224 | 0.2842 | +0.0382 | [+0.0178, +0.0616] | **WIN** |
| h 61+ | HARD | 1353 | 0.3073 | 0.2759 | **+0.0314** | [+0.0202, +0.0431] | **WIN** |
| bg-only | | 3263 | 0.0067 | 0.0047 | +0.0020 | [+0.0013, +0.0026] | WIN |
| **ALL** | | 15200 | 0.2038 | 0.1848 | **+0.0189** | [+0.0121, +0.0264] | **WIN** |

On seed 0 the matrix reads as the claim in one table: **null in every bin below horizon 25; a
CI-excluding-zero win in every easy/HARD bin from h 25–60 onward** (the sparse mid cell, n = 79,
is null at h 25–60 and a win at h 61+). The advantage appears exactly where the memory horizon
demands it — h ≈ 24 is half a loop period, past which the required horizon exceeds what 18 flat
frames carry — and nowhere else. [UA-PASS: «виграш вмикається на межі горизонту плоского вікна» —
the §5-table idiom.]

**Seed-1 replication (pre-registered before running).** The same pairing retrained at seed 1 wins
**in every cell**: ALL +0.0354 [+0.0267, +0.0447]; HARD h 25–60 +0.0622; sharpness 0.088 vs
0.057. Two-seed status of the headline: **CONFIRMED** (seed 0: ALL +0.0189 [+0.0121, +0.0264]).
The honest nuance is the onset boundary: it is seed-variable — seed 0 is null below h 25 while
seed 1 wins already from h 1–8, and the hier-long arm itself moves 0.1848 → 0.1690 across seeds
over a stable flat baseline. The robust two-seed claim is therefore **the win and its mid-horizon
concentration, not the exact onset frame**; the "no advantage before h 25" phrasing is a seed-0
observation, retained only as such.

**Per-horizon behavior and the deep-horizon disclosure.** hier-long stays below the persistence
floor at *every* horizon out to 152 frames (more than three loop periods); its error saturates at
0.221–0.222 over the last 70 frames rather than diverging — bounded drift. Sharpness, however,
fades monotonically for **all** arms (hier-long 0.2023 at h 1–4 → 0.0323 at h 105–152; batch-mean
series, §4.1.5): the pre-registered sharpness-retention clause of gate G2b (> 0.5× single-step
sharpness at depth) is **not met** at h ≥ 81, and this is stated plainly as the deep-horizon
limitation rather than tuned away. The blur ceiling is owned by the frozen decoder (AE round-trip
0.2412), not by the dynamics.

**[FIG 4-8 — PLACEHOLDER]**
*Content:* horizon × hardness win map — Δ(flat − hier) with CI-based verdict shading, seeds 0 and
1 side by side; the matched-budget win is horizon-gated on seed 0 and everywhere-positive on
seed 1; the shared feature is the mid-horizon peak on hard re-entry bins.
*Source:* **NEW** (data measured; plot).

**[FIG 4-9 — PLACEHOLDER]**
*Content:* qualitative rollout strips at re-entry events (last context | rollout frames | ground
truth): hier-long re-places the returning digit at the correct world position; flat-budget
hallucinates content at the re-entry position.
*Source:* **NEW** (from the saved rollout checkpoints; the decoded-sample review panels of the
§3.4.3 protocol).

### 4.3.4 flat-long: Парето-фронт за 2.7× токенів

*(flat-long: the Pareto frontier at 2.7× the tokens.)*

The flat-long arm (48 full-resolution frames, 3,072 tokens) beats every other arm on both error
and sharpness: ALL 0.1564, fg-sharp 0.0996; against flat-budget its ALL delta is +0.0473
[+0.0362, +0.0592]. This is reported with the same prominence as the headline, because it is what
keeps the claim honest: **at operating point A, hierarchy is an efficiency point, not a
capability necessity.** The hierarchical arm recovers most of the long-memory benefit at ≈37% of
the context tokens; a party willing to pay 2.7× buys the frontier outright. (Protocol caveat,
disclosed wherever flat-long is quoted: it trained at B = 8 × 60k steps — sample-matched to
B = 16 × 30k, but with a different EMA horizon and gradient-noise scale than the other arms.) The
regime where no affordable flat window can cover the loop at all is exactly what operating
point B exists to test (§4.5).

### 4.3.5 Механізм-агностичність: виграш пам'яті не є артефактом дифузійного ядра

*(Mechanism-agnosticism: the memory win is not a diffusion artefact.)*

The pre-registered mechanism-agnosticism gate required the regression pairing to reproduce the
memory result: reg-long versus the token-matched reg-flat. Outcome: **passed — reg-long beats
reg-flat in every bin** (ALL +0.0334 [+0.0248, +0.0423]; hard-bin wins at every horizon,
including h 1–8 HARD +0.0447; sharper, 0.0741 vs 0.0512). Notably, the regression pairing shows
no null prelude — the win starts at h 1–8 — so the horizon-gating observed on the diffusion
pairing at seed 0 is a property of that head/seed combination, not of the memory mechanism.
Combined with Table 4.6, the matched-budget hierarchy win holds under **both** viable heads:
whatever the head, packaging the same token budget as long-coarse-plus-recent-fine beats spending
it on a shorter full-resolution window. This is the precise sense in which the memory claim of
this dissertation is mechanism-agnostic. [UA-PASS: «механізм-агностичність» — keep as a term with
a defining gloss at first use.]

## 4.4 Факторне дослідження просторової абстракції: економічний, залежний від ядра вердикт

*(The spatial-abstraction factorial: an economic, head-conditional verdict.)*

Every hierarchy arm of §4.3 confounds two factors: the coarse spatial resolution of the history,
and the longer window its cheaper tokens buy (an accepted external-review finding). The factorial
of this section isolates the resolution factor: a window × history-resolution grid at operating
point A (shared frozen AE, seed 0, protocol identical to the rollout campaign so all existing
records pair exactly), holding recency equal — every coarse-history arm keeps the 6 most recent
fine frames, exactly the hier-long design. The hypothesis under test, in its original form:
coarse history is a *more abstract, less distracted* memory representation.

**Table 4.7 — Window × history-resolution factorial (fg-MSE ALL over 152 AR frames, cluster CIs;
context tokens in parentheses).**

| Window | Full-res history | Coarse history | Paired Δ (full − coarse) |
|---|---|---|---|
| 18 f, diffusion | flat-budget 0.2038 (1152) | spat18 0.2120 (672) | **−0.0082 [−0.0115, −0.0050] LOSS** (short-horizon-concentrated, null by h > 120; sharp 0.058 vs 0.050) |
| 18 f, regression | reg-flat 0.2015 (1152) | spat18-reg 0.2008 (672) | **+0.0008 [−0.0013, +0.0029] null** (h 61–120 HARD +0.0047 WIN; tiny h 1–8 loss) |
| 48 f, diffusion | flat-long 0.1564 (3072) | hier-long 0.1848 (1152) | −0.0284 (from Tables 4.5/4.6; 2.7× token gap) |

**Abstraction sufficiency at the fixed 18-frame window (read G-S1) — head-dependent.** For the
diffusion head, coarse history is a small, real, short-horizon-concentrated cost: −0.0082 ALL at
42% token savings. For the regression head — the better rollout head (§4.2.3) — coarse history is
**statistically free**: +0.0008 [−0.0013, +0.0029], null ALL at the same 42% savings, with the
long-horizon HARD bin slightly *better* (h 61–120 HARD +0.0047). The "intrinsically better
abstraction" hypothesis is therefore **not supported**; the supported form of the claim is
**economic and head-conditional**: spatial abstraction is lossy compression whose token savings
buy the memory window, and under the regression head the loss term is ≈0. [UA-PASS: «просторова
абстракція — це стискання з втратами, чия економія токенів купує вікно пам'яті».]

**Interaction (read G-S2).** The abstraction cost per saved token is similar at both windows for
the diffusion head (≈0.007–0.008 per 480 tokens); there is no evidence the cost shrinks as the
window grows.

**Window at fixed representation (read G-S3).** hier-long 0.1848 (W48, 1,152 tokens) versus
spat18 0.2120 (W18, 672 tokens): extending the window via coarse tokens buys −0.027 — an order
that dwarfs the abstraction cost, the same window-ordering law as Table 4.5.

**Combined-optimum readout (measured arm ordering, not a factorial claim).** spat18-reg (672
tokens) at 0.2008 matches or beats the diffusion flat control flat-budget (1,152 tokens) at
0.2038 — regression + coarse history at 58% of the tokens; and with the long window (reg-long,
0.1682 at 1,152 tokens) the recipe **regression head + coarse history + long window** is the
cheapest strong configuration measured in this campaign.

**Honest link to the prior spatial-gate null.** This project's earlier per-pixel spatial-gate
study on Moving MNIST found that an upper spatial level engaged unstably and added no quality — a
clean null for spatial abstraction as an *intrinsic* representational win. The factorial is
consistent with that history rather than overturning it: isolated from the window confound,
coarse history is not intrinsically better (it costs under diffusion and is at best free under
regression); its value is what its cheapness buys. One declared omission: the temporal-only
corner (strided full-resolution history) was not re-imported into operating point A, because the
strided mechanism is separately characterized at operating point B (§4.5) and the
window × resolution grid answers the resolution claim without it.

**[FIG 4-10 — PLACEHOLDER]**
*Content:* the 2×2 factorial as a diagram: axes window (18/48) × history resolution (full/
coarse), each cell annotated with arm name, token count, ALL fg-MSE; paired deltas drawn on the
edges (resolution edges: −0.0082 diff / +0.0008 reg; window edge: −0.027 at fixed coarse
representation), with the combined-optimum path highlighted.
*Source:* **NEW** (drawing from Table 4.7).

## 4.5 Дослідження межі часової здатності (робоча точка Б)

*(The temporal-capability boundary study at operating point B.)*

Operating point A proves *efficiency at matched budget*: the loop (≈48 frames) sits inside the
affordable flat window (48 frames), so flat-long can always buy the result with tokens. The
boundary study asks the capability question on the pre-registered design frozen before any
launch **[AUTHOR: attach the pre-registration document (`docs/HGRID_DESIGN.md` + addenda) as a
dissertation appendix — it is the pre-registration trail for this section]**: in a regime where
**no affordable flat window covers the memory horizon** (hgrid3, effective loop ≈196–213 frames),
does temporal abstraction — a strided, slow-rate coarse level — deliver what flat context cannot?
The study closed with a **negative rollout outcome under a binding pre-committed decision rule**,
and this section reports it as what it is: a boundary characterization, not a capability claim —
with a genuine single-step positive characterized inside it, three diagnosed methodology lessons,
and the trustworthy-memory boundary statement as the closing result. The §4.3 matched-budget
result is unaffected under every branch of this study, and remains the headline of the
dissertation. [UA-PASS: «межа часової здатності»; «характеризація граничних умов».]

### 4.5.1 Епізод аліасингу: діагностована відмова, а не відкинутий запуск

*(The aliasing episode: a diagnosed failure, not a discarded run.)*

The first dataset iteration at the wide operating point (*hgrid*: world 768 px, with background
wavenumbers scaled ×4 so that the frozen loopmom autoencoder could be reused) produced a
dead-flat null at the 20,000-step single-step pre-check (HARD −0.0008 [−0.0027, +0.0012]; all dt
bands null) and a CI-clean **LOSS** at the budget-matched 40,000-step extension (HARD −0.0025
[−0.0042, −0.0008]; ALL −0.0012; dt > 192 −0.0026): more optimization made the unaddressable
memory slots a pure attention cost. The cause is established, not speculative: the ×4 wavenumber
scaling made the background **exactly 192-px periodic**, so the four aliased viewport positions
are pixel-identical — the crop test measures pairwise MSE **0.000000** between them. Retrieval
lost the absolute-position landmark it plausibly used at operating point A (where unit-wavenumber
components made positions unique). The autoencoder-reuse constraint *caused* the failure and was
dropped (rule adopted: a fresh autoencoder per testbed variant; within-grid comparability needs
only one shared autoencoder per campaign). The second iteration (hgrid2, a DC brightness cue) was
aborted mid-run in favor of the mechanistically stronger variant; two position-probe attempts
from that period failed their own positive controls and are **retracted on the record**. The
third iteration, **hgrid3** (unscaled background with unit wavenumber forced), restores position
uniqueness (crop-uniqueness 0.004–0.010, versus hgrid2's 0.0006–0.003) and passes the full
regime verification of §4.1.1. [UA-PASS: «аліасинг»; «позиційні орієнтири» for "positional
landmarks".]

### 4.5.2 Геометрія: проліт пам'яті проти ефективного періоду циклу

*(Geometry: memory span versus the effective loop.)*

The pre-check arm carries 48 coarse slots at stride 4 — a nominal memory span of 192 frames. The
relevant horizon, however, is not $\mathrm{world}/\mathrm{speed} = 192$ but the **effective
loop**

$$
L_{\mathrm{eff}}
\;=\;
\frac{\mathrm{world}}{\mathrm{speed} \cdot (1 - p_{\mathrm{hold}})}
\;\approx\;
\frac{768}{4 \cdot 0.9}
\;\approx\; 213 \ \text{frames},
\tag{4.1}
$$

because hold frames advance time without advancing the pan. The 48-slot arm's 192-frame span
therefore falls *short* of its own target loop (≈196–213 measured) — a geometry error inherited
from the operating-point-A intuition, and the motivation for the span-224 arm of §4.5.4.
[UA-PASS: «ефективний період циклу».]

### 4.5.3 Пригадування існує: розширене оцінювання та внутрішній контроль

*(Retrieval exists: the widened evaluation and its internal control.)*

The 4×-widened paired re-evaluation of the same hgrid3 models (evaluation batches 24 → 96;
pre-registered before running; both reads disclosed) flips the underpowered nb = 24 nulls to
CI-clean wins for the strided arm — **48-slot stride-4 (reach 195 frames): HARD +0.0016
[+0.0007, +0.0025] WIN; REACHABLE-HARD +0.0021 [+0.0008, +0.0036] WIN; dt 49–96 +0.0032 WIN; ALL
+0.0005 WIN** — while the contiguous-48 positive-control arm (operating point A's exact winning
design, but with reach only 51 frames here) is **null in every cell**. The effect appears exactly
where the memory reach covers the re-entry interval and vanishes where it cannot: a reach
dose-response, serving as the internal control. Two readings attach. First, beyond-flat-window
retrieval on the position-unique wide world **exists** but is ≈7× smaller than at operating
point A (+0.002 vs +0.0135 absolute; ≈2% vs ≈13% relative) — consistent with the span shortfall
of §4.5.2 diluting it and with the harder addressing problem at this scale. Second, the nb = 24
pre-checks were underpowered for effects of this magnitude, not wrong about the aliased hgrid
(whose 40k read was a CI-clean LOSS; the aliasing verdict stands). [UA-PASS: «дозозалежність від
досяжності пам'яті» for "reach dose-response".]

### 4.5.4 Порогова гіпотеза прольоту та її одновкрокова перевірка за попередньо зафіксованим правилом

*(The span-threshold hypothesis and its single-step test under the pre-committed rule.)*

The pre-approved span-fix shot — 56 slots at stride 4, span 224 > loop, 40,000 steps, evaluation
width 96 — returned, on a single seed: **HARD +0.0273; REACHABLE-HARD +0.0363; beyond-227 null**,
with the effect confined to the hard-reachable cells (easy, mid, and beyond-reach all null).
Taken at face value this is the threshold signature — an order-of-magnitude jump once the span
exceeds the effective loop, localized exactly where reach predicts — but it was explicitly
recorded as *"threshold HYPOTHESIS — systematic-or-lucky, undetermined"* (single seed, no
same-pool 48-vs-56 comparison, measured against the weakest baseline). The resolution rule was
written before the disambiguation runs, converged with two external reviewers, and was binding;
it is quoted verbatim in the pre-registration appendix: R1 (replication on an independent seed,
CI > 0), R2 (reach causality: the strided arm must beat a contiguous arm with *identical* token
and slot counts, reach 227 vs 59, CI > 0), R3 (informative, non-gating: the 48-slot arm at
seed 1), with PASS = R1 AND R2 gating the decisive rollout pair, any uncovered outcome defaulting
to FAIL, and FAIL closing the study with no further capability spend.

**Single-step outcomes (the characterized positive).** R1 passed: the span-224 arm beats its
baseline on an independent seed — REACHABLE-HARD +0.0134 [+0.0107, +0.0161], HARD +0.0089
[+0.0069, +0.0109] — at 2.7× smaller magnitude than seed 0 (+0.0363 / +0.0273; seed mean
≈ +0.025 reachable): replicable but seed-noisy. R2 passed: the strided arm beats the contiguous
same-token control at both seeds (differential-reach band +0.0374 [+0.0334, +0.0416] seed 0;
+0.0144 [+0.0118, +0.0171] seed 1). R3: the span-192 arm stays ≈flat at both seeds (+0.0004;
+0.0011), and the contiguous arms never help at either seed. The mechanism *attribution* is
seed-dependent and is not over-read: at seed 1 the win concentrates in the reach-differential
bands exactly as a per-sample reach-causal account predicts (shared +0.0053 vs differential
+0.0181; versus the contiguous control, shared null −0.0005 vs differential +0.0144), but at
seed 0 it is uniform across shared and differential bands (+0.0343 vs +0.0366) — a circuit-level
rather than per-sample-reach signature. The reliable two-seed single-step statement is therefore:
**a memory span exceeding the effective loop reliably enables retrieval learning (2/2 span > loop
runs CI-positive versus 0/4 span ≤ loop runs at ≈0), while the magnitude and per-sample usage
pattern of the learned circuit vary by seed.** Disclosure: the single-step gate CIs in this
subsection are sample-iid, not clip-clustered; the load-bearing cells survive a conservative 2–3×
design-effect widening, and a clip-clustered recomputation is reported alongside. The
teacher-forced caveat is stated where it belongs: the single-step study conditions on
ground-truth context, and the rollout pair of §4.5.5 is the real test.

### 4.5.5 Вирішальна розгортальна пара: негативний результат і межа довіри до вмісту пам'яті

*(The decisive rollout pair: the negative outcome and the trustworthy-memory boundary.)*

The pre-registered decisive pair (the span-224 strided arm versus the long-context diffusion
baseline on hgrid3; warm-up 224 frames; uniform B = 8 × 40k training) returned **RED**: over 256
self-fed rollout frames the pre-registered gate cells are null — dt 49–96 HARD +0.0020
[−0.0071, +0.0111]; dt 97–192 −0.0028 [−0.0088, +0.0035]; dt > 192 −0.0018 [−0.0065, +0.0027] —
so **the single-step capability does not convert to a rollout capability at this scale**, and, by
the binding rule, no further capability experiments were run.

The horizon profile is the finding. The strided arm wins every h 1–24 bin CI-clean (HARD +0.0211
and +0.0224 in the two bands) while its slot memory still carries near-ground-truth content;
fades to null by h 25–60; and inverts to small systematic losses beyond h 61 as the buffer fills
with self-generated, degraded content. Read together with §4.3, this bounds the mechanism
honestly: **hierarchical slot memory pays where its contents remain trustworthy — continuously
refreshed within the recent window (operating point A: the matched-budget win over 152 frames) or
near a ground-truth anchor (operating point B: h ≤ 24) — and cannot yet be sustained by pure
self-feeding across a 213-frame loop.** This statement completes the boundary-condition study;
the §4.3 matched-budget result is unaffected. [UA-PASS: «межа довіри до вмісту пам'яті» — the
trustworthy-memory boundary; keep the full sentence as the section's quotable verdict.]

**[FIG 4-11 — PLACEHOLDER]**
*Content:* operating-point-B rollout horizon profile: paired Δ (strided − baseline) vs rollout
horizon with cluster-CI bands, annotated in three zones — CI-clean wins at h 1–24 (memory
contents near ground truth), null at h 25–60, small systematic losses at h 61+ (buffer filled
with self-generated content); the pre-registered dt-gate cells marked as null.
*Source:* **NEW** (plot from the decisive-pair rollout records).

### 4.5.6 Методологічні уроки дослідження межі

*(Methodology lessons of the boundary study — threats-style.)*

1. **The effective-loop formula.** A memory mechanism's span must be sized against
   $L_{\mathrm{eff}} = \mathrm{world}/(\mathrm{speed} \cdot (1 - p_{\mathrm{hold}}))$ (eq. 4.1),
   not $\mathrm{world}/\mathrm{speed}$: hold frames consume slots without advancing the loop. The
   48-slot arm's 192-frame span silently fell ≈2–10% short of its own target (≈196–213),
   plausibly diluting the effect it was built to detect.
2. **The autoencoder-reuse trap.** Reusing a frozen autoencoder across testbed variants forced a
   background transformation (×4 wavenumber scaling) that made the world exactly
   viewport-periodic and destroyed the positional landmarks retrieval depends on — a dataset-side
   confound that manifested as a *mechanism* failure (memory slots scoring as pure attention
   cost). Rule adopted: a fresh autoencoder per testbed variant; share one autoencoder only
   within a campaign.
3. **Evaluation power.** The nb = 24 pre-checks were adequately powered for operating-point-A
   magnitudes (+0.0135) but not for the ≈+0.002 effects of the harder regime; the pre-registered
   4× widening flipped nulls to CI-clean wins on the *same* checkpoints. Reading a null requires
   a power argument against the plausible effect size, not just a confidence interval spanning
   zero — and both the underpowered and the widened reads are disclosed.

## 4.6 Впровадження результатів дослідження

*(Implementation of the research results. — Kept as per the existing implementation acts, digest
item 38: the acts cover the reversible-transformer method and entropy tokenization; no claims
about the hierarchical model are added.)*

The results of the dissertation research have been implemented in:

- the **practical activities of ORNETIKA LLC** (Kyiv): the developed information technology was
  used in designing intelligent systems for the analysis of spatio-temporal video data and
  anomaly detection in complex technical environments; the resource-efficient methods (the
  TempVerseFormer reversible-transformer method) and adaptive entropy-based tokenization
  substantially reduced the computational cost of modeling long temporal sequences
  (implementation act of October 14, 2025);
- the **educational process of VNTU** in the disciplines «Інформаційні технології моніторингу та
  аналізу стану складних систем» and «Інтернет речей та інтелектуальний аналіз даних» for PhD
  students of specialty 124 (implementation act of September 19, 2025).

Copies of the implementation acts are given in the appendices **[AUTHOR: appendix letters Г/Д;
insert the act copies — the current draft has headings only]**.

## 4.7 Висновки до розділу

*(Conclusions to the chapter.)*

1. In accordance with **Task 7** of the dissertation research, comprehensive experimental studies
   of the developed methods and the intelligent information technology were conducted on the
   two-operating-point testbed family (§4.1) under an audit-hardened measurement protocol:
   paired clip-level cluster bootstrap confidence intervals (frame-iid intervals were shown to be
   2–7× too narrow at lag-1 autocorrelation +0.86), exact analytic hardness stratification with
   an audit-corrected anchoring, common random numbers, per-operating-point reference floors and
   ceilings (Table 4.1), and full disclosure of every correction and caveat (§4.1.5). The [A2]
   receipt datasets (2D/3D rotations, Moving MNIST, MMNIST-CIFAR, TaxiBJ, KTH) are retained as
   the published validations of the reversibility method; the OpenSTL bridge is pending
   **[AUTHOR: T7]**.

2. The matched-compute mechanism study (§4.2) established that **the generation-head ranking is
   regime-dependent**: the diffusion core wins the single-step oracle-coverage regime on every
   column at matched capacity and NFE (fg-MSE 0.0512 vs 0.0664 overall; 0.1549 vs 0.1808 on the
   hard, genuinely generative stratum), while the regression core wins the deterministic
   best-of-1 rollout at equal sharpness (ALL delta CI [+0.0110, +0.0227]) — a refutation of the
   pre-registered fade expectation, reported as such. No statement of the form "one best head"
   survives the chapter; the memory claims were therefore verified under both viable heads.

3. The energy-based head was characterized by the two-sided capability study (§4.2.4) with the
   official implementation and both published recipes at 20,000 converged steps: landscape
   ordering is recipe-dependent (S2 stably mis-ordered, 21/21 probes; S1 correctly ordered,
   15/15 probes from step 6,000), while descent reachability fails under both recipes and under
   the paper's own inference procedure ($d_{\mathrm{tgt}} \approx 1.12$–$1.29$ against the mean's
   0.964; foreground sharpness ≤ 0.035 against the 0.08 structure threshold). The result is a
   **characterized negative** — a capability boundary at 16 GB-reachable scale with the
   configuration gap stated (Table 4.4), not a reproduction failure — and it empirically
   justifies the diffusion selection of Chapter 3 jointly with conclusion 2.

4. The headline claim of the dissertation was confirmed (§4.3): **at an equal context-token
   budget (1,152 tokens), the hierarchical long-coarse-memory model beats the flat control on
   152-frame autoregressive rollouts on two seeds and under both heads** — seed 0 ALL +0.0189
   [+0.0121, +0.0264] with the win concentrated from h ≥ 25 on hard re-entry bins (h 25–60 HARD
   +0.0344 [+0.0162, +0.0531]); seed 1 ALL +0.0354 [+0.0267, +0.0447], winning every cell;
   regression pairing ALL +0.0334 [+0.0248, +0.0423]. The single-step memory gate showed the
   underlying retrieval capability emerging late in training (null at 6,000 steps, ≈13% hard-bin
   win at 20,000 on both seeds). The claim is honestly bounded: the onset horizon is
   seed-variable (the robust claim is the win and its mid-horizon concentration); sharpness fades
   for all arms at depth (the G2b retention clause is not met at h ≥ 81; the ceiling is owned by
   the frozen decoder); and flat-long at 2.7× the tokens is the outright Pareto frontier
   (0.1564) — at operating point A, hierarchy is an efficiency point, not a capability necessity.

5. The spatial-abstraction factorial (§4.4) isolated the resolution factor from the window
   confound and returned an **economic, head-conditional** verdict: coarse history costs the
   diffusion head −0.0082 [−0.0115, −0.0050] at 42% token savings and is statistically free for
   the regression head (+0.0008 [−0.0013, +0.0029]); its real value is what the saved tokens
   buy — window extension at fixed representation is worth −0.027, dwarfing the abstraction
   cost. The cheapest strong configuration measured is regression head + coarse history + long
   window (0.1682 at 1,152 tokens) — a measured arm ordering, not a factorial interaction claim.

6. The temporal-capability boundary study at operating point B (§4.5) closed under its binding
   pre-committed rule with a **negative rollout outcome and a characterized single-step
   positive**: retrieval beyond the flat window exists on the position-unique wide world and is
   reach-causal (dose-response with the internal contiguous control), a memory span exceeding the
   effective loop $L_{\mathrm{eff}}$ (eq. 4.1) reliably enables single-step retrieval learning
   (2/2 span > loop runs CI-positive vs 0/4 span ≤ loop), but the capability does not convert to
   a self-fed rollout capability at this scale (all pre-registered dt-gate cells null over 256
   frames; wins confined to h ≤ 24 where the slot contents are near ground truth). The resulting
   boundary statement — hierarchical slot memory pays where its contents remain trustworthy, and
   cannot yet be sustained by pure self-feeding across a 213-frame loop — together with the three
   methodology lessons (effective-loop sizing; the autoencoder-reuse trap; evaluation power)
   constitutes the honest delimitation of the dissertation's central mechanism.

7. The results of the research have been implemented in the practical activities of ORNETIKA LLC
   and in the educational process of VNTU (§4.6, per the implementation acts).

The results of this chapter have been published in [A4] (the mechanism, memory, factorial, and
boundary studies) and [A2] (the receipt validations of §4.1.2) **[AUTHOR: bind to the final
numbered publication list; add conference theses covering the experimental campaign if any]**.

---

## Working notes — figure register of this chapter (not part of the chapter text)

| Fig. | Content | Source |
|---|---|---|
| FIG 4-1 | three prediction objectives | ADAPT old Рис. 3.2 (per 04_ROZDIL3 disposition) |
| FIG 4-2 | mechanism decoded panels | ADAPT article assets `logs_rhcp/step1_mech_clean_{reg,diff}.png` |
| FIG 4-3 | EBT arm structure | ADAPT old Рис. 3.4 (condensed) |
| FIG 4-4 | energy ordering vs step (S1/S2) | NEW (probe data; old Рис. 3.1 optionally as inset) |
| FIG 4-5 | cold-descent sharpness vs step | NEW (probe data) |
| FIG 4-6 | memory-gate emergence 6k/20k × 2 seeds | NEW (gate logs) |
| FIG 4-7 | per-horizon rollout curves | ADAPT `logs_rhcp/fig_horizon_curves_loopmom.png` |
| FIG 4-8 | horizon × hardness win map, 2 seeds | NEW |
| FIG 4-9 | qualitative re-entry rollout strips | NEW (saved checkpoints) |
| FIG 4-10 | spatial factorial diagram | NEW (Table 4.7) |
| FIG 4-11 | op-B rollout horizon profile (boundary) | NEW (decisive-pair records) |

Numbers deliberately NOT carried into this chapter: the superseded pre-audit mechanism table
(0.0489/0.0363 — corrected by the hardness-anchor fix); the article's PASS-branch capability
headline for §4.5 (superseded by the 2026-07-06 pair outcome); any "200-frame rollout" phrasing
(protocol is 48 + 152); the old draft's TaxiBJ-as-key-benchmark framing and unverified A100
claims (kept only as [A2] receipts / author-verify).
