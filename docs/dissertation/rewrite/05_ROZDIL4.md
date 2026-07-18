# Chapter 4 — Experimental Evaluation and Practical Application

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
operating point (§4.5) closes the capability line of the experimental program with an honestly
bounded negative and the methodology lessons it produced. The generalization study (§4.6) then
carries the configuration selected by §§4.2–4.4 onto the standard and synthetic benchmark
suites — its measured components include the standard Moving-MNIST negative-control bridge and
the corrected persistent-object long Moving-MNIST external boundary. §4.7 documents the
implementation of the results,
and §4.8 concludes the chapter.

Two standing conventions of the chapter, fixed by the validity audit of the experimental campaign
and stated once here: (i) the long-horizon evaluation protocol is a **48-frame ground-truth
warm-up followed by 152 autoregressive frames** (38 scheduler steps × 4 frames) — all text says
"152 AR frames", never "200-frame rollout"; (ii) all confidence intervals quoted in this chapter
are 95% paired **clip-level cluster bootstrap** intervals unless a disclosure states otherwise
(§4.1.5).

## 4.1 Data, Hardware and Measurement Protocol

### 4.1.1 Testbed Family: Two Operating Points

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

**Figure 4.A — LoopMom geometry and the registered reach band.**
*Content:* toroidal world, moving viewport, persistent digits and the pair-derived
`14 < dt <= 48` population. The privileged pose-indexed raw-memory reference is drawn
separately from token-budget-matched neural packages.
*Source:* `docs/figures/fig15_loopmom_observability_schematic.png` and vector PDF; generated
deterministically from the frozen E2 geometry by `scripts/make_rft_core_schematics.py`.

### 4.1.2 [A2]

The experimental validation of the reversible-transformer method and time-agnostic
backpropagation (§§2.3–2.4) was published in [A2] on procedurally synthesized **2-D rotating
shapes** at $64 \times 64$ resolution. The suite controls the temporal law—constant, changing,
oscillating, interrupted, and combined rotation—and therefore isolates the effect of prediction
depth and temporal discontinuity. The published regime uses a 12-frame context, grows the
prediction depth from one to five frames during 30,000 training steps, evaluates 2,000 independently
generated sequences, and trains all compared models with Adam at learning rate $10^{-5}$ on one
NVIDIA RTX 4060 Ti 16 GB. The memory-scaling measurements use batch 32. These values replace the
unsupported old-draft claims of AdamW with cosine decay, 100–300 epochs, and A100 80 GB hardware.

The **3-D polycube** and entropy-tokenization material belongs to [A1], not [A2]. Moving MNIST,
MMNIST-CIFAR, TaxiBJ, and KTH are likewise not reported as [A2] experiments: Moving MNIST and
TaxiBJ are separate dissertation bridges (§§4.6.2 and 4.6.5), whereas MMNIST-CIFAR and KTH remain
explicitly unexecuted extensions. This separation preserves the content and scope of the existing
publications instead of retroactively enlarging them.

### 4.1.3 Bridge to the OpenSTL/Moving-MNIST Benchmark

**[MEASURED 2026-07-06 — full results, table, and reading in §4.6.2 (Table 4.8).]** For
continuity with the literature, the developed stack is additionally evaluated on the standard
OpenSTL Moving-MNIST suite [REF: OpenSTL]. The pre-registered expectation, recorded before the
runs: mid-pack aggregate MSE/SSIM against the OpenSTL leaderboard — aggregate metrics on a
≈93%-black-background benchmark reward mean prediction (this project's own measurement: a
constant all-black prediction outperforms a trained CW-VAE [REF: CW-VAE] on aggregate SSIM/PSNR,
§3.4.3) — together with competitive perceptual (LPIPS) behavior and an autoencoder round-trip
ceiling row that decomposes decoder error from dynamics error. The purpose of the bridge is
connectivity to the standard suite, not a leaderboard claim; the memory-horizon claims of this
chapter cannot be tested on Moving MNIST at all, for the degeneracy reasons established in §3.4.
The measured outcome (§4.6.2) confirmed the connectivity, ceiling-row, and
perceptual-versus-persistence clauses, **refuted the "mid-pack aggregate MSE" clause** (180.91
against the deterministic specialists' 22.06–29.80 — reported as a refutation, per the standing
convention of §4.2.3), and returned the predicted flat ≈ hierarchical null as the negative
control of the memory-horizon claim.

### 4.1.4 Hardware and Software Environment

All experiments of the [A4] campaign (§§4.2–4.5) were performed on a single consumer-grade GPU
(NVIDIA GeForce RTX 4060 Ti, 16 GB) — a deliberate constraint of the research design: the
resource-efficiency claims of Chapter 2 are made *for* this class of hardware, and the complete
benchmark (data generation, autoencoder training, all system arms, all evaluations) reproduces on
one such device. The software complex is implemented in Python/PyTorch (§4.7; the module map is
given in the appendix **[AUTHOR: appendix letter]**). The memory-consumption measurements quoted
in Chapter 2 (Table 2.3: BPTT 12.2 GB at rollout length $K = 40$ and OOM at $K = 120$ versus the
developed method's ≈1.1–1.2 GB through $K = 160$; the inner-loop panel behind the 0.26%-measured no-go
result) were performed in this same environment, on the implemented stochastic hierarchical
recurrent core, and constitute part of the experimental program of this chapter even though their
tables live with the theory they verify.

Training protocol of the rollout arms (operating point A): 30,000 optimizer steps, batch 16,
bf16, EMA decay 0.999, evaluation on EMA weights (§3.2.3); the flat-long arm is the disclosed
exception (§4.3.4). The DiT backbone of every arm is the §3.2.2 configuration (d = 384, depth 8,
6 heads; 22.19 M parameters).

### 4.1.5 Statistical Protocol and Threats to Validity

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
   with foreground sharpness, decoded-sample review (§3.4.3), and the complete-metric evidence
   artifacts, which include full-frame, union-mask and false-positive-sensitive endpoints.
   Per-horizon *sharpness* series are chunk-batch means duplicated per record — valid for overall
   means and trend curves, never for per-bin or per-clip readings; every sharpness-versus-horizon
   statement in this chapter is so scoped.

4. **Pairing and replication conventions.** All arm-versus-arm comparisons share evaluation
   clips and horizons; sampling noise is common only when target tensor shapes match. Cross-shape
   cells retain record pairing but not elementwise CRN (unbiased, with less variance reduction).
   "2 seeds" for the single-step
   memory gate means two *training* seeds sharing one CRN evaluation set (by design, for exact
   pairing). At operating point B the evaluation adds the pre-registered
   **dt-since-last-visible** stratifier (per-record time since the target region was last in
   view, computed from the recorded trajectory; bins dt ∈ (48, 96], (96, 192], > 192, crossed
   with hardness) and a warm-up-SEEN restriction, so that gates are read only on re-entries of
   content that was answerable by construction.

5. **External validity.** The testbed is purpose-built and synthetic; digit assets are shared
   across splits (texture familiarity only — re-entry memory requires clip-specific trajectory
   content, which does not transfer across clips), while trajectories and world layouts are
   split-disjoint. All claims are scoped to the memory-horizon regime the testbed
   operationalizes; the OpenSTL bridge (§4.1.3, measured in §4.6.2) anchors short-horizon
   behavior on a standard benchmark. Generation seeds of the shipped dataset files were not recorded (measured
   properties are stated as measured; a frame-hash duplication check found no cross-file clip
   duplication).

## 4.2 Matched-Compute Mechanism Study: Selecting the Generative Core

The claim defended in this section is deliberately conditional: **which prediction head wins
depends on the inference regime; there is no one best head.** Three results establish this — the
single-step coverage study (§4.2.2), its rollout reversal (§4.2.3), and the two-sided study of
the energy-based head (§4.2.4). The section thereby supplies the empirical selection evidence
promised in §3.2 (the diffusion core) *and* the honest boundary of that selection.

**Figure 4.1 — Three prediction heads in the shared context testbed.**
Direct regression predicts one target latent, diffusion predicts a denoising velocity and commits
through DDIM sampling, while the energy head learns a scalar landscape traversed by candidate
descent. The diagram separates objective and inference from the shared context question.
*Source:* `docs/figures/fig24_generation_objectives.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`.

### 4.2.1 Shared Mechanism Testbed

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
diagnostic** (§3.4.4), always reported alongside the mean over N.

### 4.2.2 Single-Step Oracle-Coverage Regime: Diffusion Wins Every Column

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
selection is an oracle coverage regime.

**Figure 4.2 — Single-step regression and diffusion predictions.**
Each row shows last context, prediction and ground truth on identical inputs. The panels are
descriptive examples of the registered single-step regime; the aggregate and paired results,
not a selected row, determine the comparison.
*Source:* `docs/figures/fig29_mechanism_single_step_samples.png`, composed without pixel editing
from `logs_rhcp/step1_mech_clean_{reg,diff}.png` by
`scripts/make_dissertation_missing_figures.py`.

### 4.2.3 Rollout Reversal: Regression Wins Best-of-1

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
evidence, wrong.

### 4.2.4 Two-Sided Energy-Head Study: A Characterized Negative Result

The third candidate head is energy-based descent: Energy-Based Transformers [REF: EBT —
Gladstone et al. 2025] train a scalar energy $E(\text{context}, \text{candidate})$ — reviewed in
§1.3.3 as a global consistency criterion — and generate by gradient descent on the candidate. The
mechanism is attractive for this dissertation's problem precisely because descent steps are a
natural "think-longer" knob for hard re-entry frames. The study below is *two-sided* in that it
decomposes the observed failure into two separable capabilities: whether the trained energy
landscape **orders** candidates correctly (target below mean below noise), and whether descent
from the prescribed initialization can **reach** a structured output. The headline result is that
these dissociate.

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

**Figure 4.3 — Energy head and gradient-descent generation loop.**
The tested head scores a 1,280-dimensional per-frame candidate together with context and actions.
Generation repeatedly updates the candidate using its energy gradient and the registered noise
rule. This makes landscape ordering and reachability by descent separately testable.
*Source:* `docs/figures/fig25_ebt_generation_loop.png`; deterministic source
`scripts/make_dissertation_missing_figures.py`.

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

**Figure 4.4 — Energy-landscape ordering under two recipes.**
S1 changes to the desired target-below-mean ordering after approximately 6,000 updates; S2 keeps
the batch mean below the target through 20,000 updates. Both series include the random-candidate
reference.
*Source:* `docs/figures/fig26_ebt_landscape_ordering.png`, parsed deterministically from
`logs_rhcp/ebt_scaled_{s1,s2}.log` by `scripts/make_dissertation_missing_figures.py`.

**Figure 4.5 — Foreground sharpness after cold energy descent.**
Both recipes remain below the preregistered 0.08 structure threshold and far below the
approximately 0.24 decoder reference, even after S1 learns the desired energy ordering.
*Source:* `docs/figures/fig27_ebt_descent_sharpness.png`, parsed deterministically from
`logs_rhcp/ebt_scaled_{s1,s2}.log` by `scripts/make_dissertation_missing_figures.py`.

### 4.2.5 Generative-Core Conclusion

The head ranking is **regime-dependent**: diffusion for the single-step best-of-N coverage
regime, regression for the deterministic best-of-1 rollout regime, energy-based descent for
neither at 16 GB-reachable scale (a two-sided characterized negative: ordering recipe-dependent,
reachability failing under both recipes and under the paper's own sampler). The memory studies
that follow therefore use **both** viable heads for the central matched-budget memory comparison
(§4.3.5), while §4.4 estimates and reports package preference separately by head and training
seed. The central reach result is not tied to one head; the broader ordering of temporal and
spatial packages is explicitly head- and seed-conditional.

## 4.3 Single-Step Memory Gate and Matched-Budget Rollout Study

This section tests the central pre-registered prediction of Table 3.1: at an equal
context-token budget, does a hierarchical allocation (long coarse memory + recent fine window)
beat a flat allocation when only the former reaches memory-relevant history? The measured studies
then bracket the tested budgets where that prediction holds and where full-resolution fidelity
instead wins; they do not assume a continuous or unique transition.

### 4.3.1 Preliminary Gate: Recall Emerges Only after Sufficient Optimization

Before any rollout machinery, the single-step memory gate asked whether a long *coarse* context
helps at all: coarse context 48 versus 6 at fixed fine context 6 (1,232 vs 560 tokens including
the target slots), best-of-8, stratified by the exact hardness label. The result is an
emergence-with-budget pattern: at 6,000 steps the HARD-bin delta is null on both training seeds
(+0.0014 [−0.0004, +0.0033] and +0.0008 [−0.0009, +0.0025]); at 20,000 steps it is a decisive win
on both (+0.0135 [+0.0080, +0.0196] and +0.0143 [+0.0083, +0.0208]; hard-bin absolute error
0.1053 → 0.0918 on seed 0 — a ≈13% relative reduction), with sharpness tied (≈0.188 both arms):
the win is retrieval, not sharpness. Four disclosures attach: "2 seeds" means two training seeds
sharing one CRN evaluation set (§4.1.5); these evaluation windows are near-independent rather
than fully clustered, so the audit's worst-case design-effect correction widens the 20k hard-bin
CIs to ≈[+0.005, +0.022] (seed 0) and ≈[+0.006, +0.023] (seed 1) — the win survives; **each
budget is its own complete, fully annealed training schedule** (a 6,000-step run and a
20,000-step run, budget-matched within each contrast), so the compared quantity is the
optimization budget with schedule shape a bundled covariate — and the 6,000-step arms were the
better-converged endpoints of their own schedules, which works against a mid-anneal objection;
and the 20,000-step budget was a **disclosed post-null follow-up** — the originally registered
read at 6,000 steps returned its null, which is on the record, and the extension was then run
and replicated on both seeds, so the claim rests on the twice-replicated 20k result, not on a
selected read. The methodological moral carries through the whole chapter:
**retrieval-from-memory requires a large optimization budget, and short-budget ablations (6,000
steps here) would have spuriously killed it.**

**Figure 4.6 — Emergence of single-step memory benefit with optimization budget.**
The HARD-bin effect is unresolved at 6,000 updates for both seeds and positive with intervals
excluding zero at 20,000 updates for both seeds. This is an optimization-budget result, not a
claim about asymptotic convergence.
*Source:* `docs/figures/fig28_memory_gate_optimization.png`, parsed deterministically from the
four `logs_rhcp/mem_gate_full_s{0,1}{,_t20k}.log` files by
`scripts/make_dissertation_missing_figures.py`.

### 4.3.2 Rollout Matrix

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

Three seed-0 regularities are visible in the ordering, each defended separately below: (i) the
48-frame coarse-plus-fine arm beats the shorter contiguous-fine control under both tested heads
at this seed; (ii) regression wins the deterministic best-of-1 rollout (§4.2.3); and (iii) the
full-resolution 48-frame window wins outright at 2.7× the tokens (§4.3.4). The first observation
replicates at seed 1 for this particular long-versus-short comparison, but it must not be promoted
to a head-independent law. The complete five-seed temporal-versus-spatial factorial (§4.4.3)
instead finds stable package direction under diffusion and sign-changing direction under
regression. The defensible claim is therefore configuration- and head-scoped: reach helps where
the shorter control cannot address the relevant past, while the preferred way to package that
reach is not universally mechanism-agnostic.

**Figure 4.7 — Error and sharpness by rollout horizon.**
*Content:* foreground MSE on all frames, foreground MSE on HARD re-entry frames, and
foreground-sharpness retention, with persistence and frozen-decoder references.
*Source:* `docs/figures/fig1_claim1_horizon_curves.png`; deterministic source
`scripts/make_figures.py`.

### 4.3.3 Main Two-Seed Pair: Advantage beyond the Flat Memory Horizon

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
frames carry — and nowhere else.

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

**Figure 4.8 — Effect map by horizon and difficulty.**
*Content:* horizon × hardness map of Δ(flat − hierarchy), seeds 0 and 1 side by side. The
horizon-gated onset is seed-0-specific; the common feature is the positive effect and
mid-horizon peak rather than one exact onset.
*Source:* `docs/figures/fig2_claim1_winmap.png`; deterministic source
`scripts/make_figures.py`.

**Figure 4.9 — Qualitative slices of autoregressive rollout.**
*Content:* identical ground truth and seed-1 `hier-long`/`flat-budget` predictions at registered
horizons. Both models become blurred at long horizon; the figure illustrates a modest average
advantage, not a universal per-frame recovery or a claim that one named arm uniquely
hallucinates.
*Source:* `docs/figures/fig7_samples_claim1.png`, from the saved rollout checkpoints.

### 4.3.4 flat-long: the 2.7×-Token Pareto Reference

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

### 4.3.5 Prediction-Head Robustness and Its Boundary

The preregistered two-seed gate asked whether the specific `reg-long` versus token-matched
`reg-flat` comparison reproduced the diffusion reach result. It did: `reg-long` beats `reg-flat`
in every bin at both tested seeds (ALL +0.0334 [+0.0248, +0.0423] at seed 0 and +0.0501
[+0.0396, +0.0612] at seed 1; hard-bin wins at every horizon, including h 1–8 HARD +0.0447;
sharpness 0.0741 versus 0.0512 at seed 0). The regression pairing has no null prelude, so the
seed-0 diffusion onset near h = 25 is not a general property of the memory geometry.

This is evidence that one long-coarse-versus-short-fine reach advantage is not solely an artifact
of the diffusion head. It is **not** evidence that package preference is mechanism-agnostic. The
complete five-seed temporal-versus-spatial factorial in §4.4.3 gives two temporal and three
spatial regression winners and a sign-changing head × package interaction. Accordingly, the
dissertation reports the two-seed replication as a scoped robustness result and rejects the
broader head-independent wording.

### 4.3.6 Completing the Token-Budget Frontier with Matched Upper and Lower Partners

The arm matrix of Table 4.5 leaves the two extreme token budgets without same-budget partners:
flat-long (3,072 tokens) has no hierarchical arm at its budget, and hier-short (480 tokens) has
no flat control at its. Two frontier-completion arms close the grid,
pre-registered before launch (the FRONTIER COMPLETION block of the factorial pre-registration):
**hier-rich** (48 coarse + 36 fine frames = 3,072 tokens, trained B = 8 × 60,000 steps —
sample-matched to flat-long's disclosed regime, so the pairing inherits flat-long's §4.3.4
protocol caveat *symmetrically*) and **flat-short** (8 fine frames = 512 tokens ≈ hier-short's
480, standard B = 16 × 30,000). Diffusion heads first; the regression twins are optional.

The reading was fixed in advance and is quoted verbatim from the pre-registration: "at 3072
tokens the window is SATURATED at 48 frames by the T=100 train clips — hierarchy cannot buy more
window, so per the factorial's abstraction cost the honest expectation is hier-rich ~<= flat-long
(CONVERGENCE at the frontier top). The efficiency claim's correct form is 'the hierarchy curve
reaches a given quality at fewer tokens and converges with flat where the window saturates' — a
frontier-from-below, not dominance at every budget. At the low end, hier-short vs flat-short
tests the frontier's bottom."

**Table 4.6a — Token-budget frontier of operating point A (letter-suffixed to avoid renumbering
the audited table register; final numbering at typeset). Measured anchor rows from Tables
4.5/4.6; frontier-completion cells measured 2026-07-07 (`logs_rhcp/frontier_top.log`,
`logs_rhcp/frontier_bottom.log`); lower-edge pair measured 2026-07-07
(`logs_rhcp/frontier_low.log`, pre-registered the same morning before the runs).**

| Token budget | Flat arm | ALL fg-MSE | Hierarchical arm | ALL fg-MSE | Paired Δ (flat − hier), cluster CI |
|---|---|---:|---|---:|---|
| ≈3,072 | flat-long | 0.1564 | hier-rich | 0.1695 | **−0.0131 [−0.0201, −0.0069]** (flat wins; seed 1 REPLICATED: −0.0107 [−0.0163, −0.0058]) |
| 1,152 | flat-budget | 0.2038 | hier-long | 0.1848 | +0.0189 [+0.0121, +0.0264] (Table 4.6) |
| **896** | flat-min (14 f) | 0.2106 | hier-min (48c+2f) | **0.1706** | **+0.0399 [+0.0299, +0.0501]** (hier wins EVERY cell; seed 1 REPLICATED: +0.0358 [+0.0267, +0.0454], every cell) |
| ≈480–512 | flat-short | 0.2224 | hier-short | 0.2260 | **−0.0036 [−0.0063, −0.0010]** (flat wins) |

**Figure 4.10 — Confirmed token-budget frontier.**
*Content:* ALL fg-MSE versus context-token budget at operating point A, one 6-point curve per
family (flat: flat-short → flat-min → flat-budget → flat-c1 → flat-c2 → flat-long; hierarchical:
hier-short → hier-min → hier-long → hier-c1 → hier-c2 → hier-rich). Mark the tested
family-wise-confirmed hierarchy points at 896/1,152, the unresolved 1,792 point, and the confirmed
flat points at 2,432/3,072. Draw the geometric span threshold as an explanatory annotation and
the persistence floor (0.2552); do not shade an interpolated win band.
*Source:* `docs/figures/article_confirmed_frontier.png`, generated from the untouched E2
simultaneous family in `artifacts/confirmation_e2/results.json`. The historical development
`fig9_frontier.png` is retained for provenance but is not used for confirmation.

The measurement resolved the pre-registered dual branch to a **third outcome, stricter than the
pre-registered convergence expectation, and reported as measured under the no-soft-reads rule**
(the default-to-FAIL discipline of §4.5 applied to wording). At the top budget hier-rich does
not converge with flat-long — it loses CI-clean at equal tokens and equal training regime
(−0.0131 [−0.0201, −0.0069], a small uniform loss across essentially every horizon × hardness
bin); at the bottom budget flat-short beats hier-short (−0.0036 [−0.0063, −0.0010], concentrated
at h ≤ 60, null beyond). The frontier therefore *inverts at both ends*: this lands on the first
fallback branch at the top (the abstraction cost does not vanish at the top budget — the −0.0131
loss is of the same kind as the diffusion-head abstraction cost of §4.4) and refutes the
matched-budget law at the bottom, where the 6–8-frame window is below the memory-relevance scale
and coarse slots are redundancy rather than memory. The mechanism reading: hierarchy pays where
its coarse tokens buy a memory window that (a) exceeds what fine tokens afford and (b) matters;
at window saturation (48 frames, clip-bounded) coarse slots merely displace full-resolution
frames, and below the memory-relevance threshold they duplicate it. Sharpness follows the window,
not the hierarchy: hier-rich 0.0898 versus flat-long 0.0996, hier-short 0.0392 versus flat-short
0.0432 — at matched saturation the hierarchy does **not** retain sharpness better.

This mechanism reading makes a quantitative prediction, which was pre-registered and then
measured the same day (the lower-edge row of Table 4.6a): the win must switch on at the cheapest
budget whose coarse span covers the ≈48-frame loop — 48 × 16 coarse tokens plus a minimal fine
tail, ≈896 tokens, a budget at which a flat model affords only a 14-frame window. The measured
pair confirms the prediction, and more strongly than expected: hier-min beats flat-min in
**every horizon × hardness cell** (overall +0.0399 [+0.0299, +0.0501] — the largest hierarchical
advantage measured in the campaign), with the visibility-stratified gates showing the reach
signature exactly where the mechanism operates (dt ≤ 48 HARD +0.0750; dt 49–96 HARD +0.0282;
dt 97–192, beyond the 48-frame span, null). Two structural conclusions follow. First, the win
smallest tested dense-family hierarchy win coincides with the geometric span-threshold arithmetic
written before the run: the next smaller dense-family point favors flat, while hierarchy wins at
896. This is a tested ordering rather than a continuous switch. Second, the **largest tested
dense-family advantage occurs at 896 tokens**: reducing the
budget removes the flat model's window from the memory horizon faster than the hierarchical
abstraction cost grows — the efficiency argument is strongest exactly where budgets are
tightest, and hier-min at 896 tokens (0.1706) reaches a quality the flat curve attains only at
substantially larger budgets in the development curve. The untouched family leaves 1,792
unresolved and confirms flat at 2,432, so no equal-quality multiplier or interpolated crossing is
claimed.
The pre-registered seed-1 replication confirmed the point: every cell WIN again (overall
+0.0358 [+0.0267, +0.0454]) with the same visibility-stratified reach signature — the 896-token
896-token point is therefore a two-seed result (seed-mean Δ ≈ +0.0379, the largest two-seed effect
of the campaign), alongside the two-seed 1,152-token pairing of §4.3.3. The §4.3
efficiency claim takes its final, tested-bracket form: **the simultaneous confirmation family
supports hierarchy at 896/1,152, leaves 1,792 unresolved, and supports flat fidelity at
2,432/3,072; no interval, exact edge or unique crossing is inferred.**

## 4.4 Forms of Hierarchical Abstraction: Spatial, Temporal and Depth

A hierarchy in a latent video predictor is not one construction but a **family of forms**. A
context-token budget can be packaged along four axes:
**spatial abstraction** (every frame, at lower resolution), **temporal abstraction** (full
resolution, at a coarser tick), **depth** (how many nested levels), and **allocation** (the split
between long-span and recent-detail streams). Each shape buys memory span at a different token
price and pays a different information cost. Throughout, hierarchy means multi-rate and/or
multi-RESOLUTION context — independent axes: a configuration can be hierarchical in tick rate
while single-resolution (the §4.4.3 strided control — a two-rate temporal hierarchy in the
clockwork/CW-VAE sense, whose levels are likewise temporal) or hierarchical in resolution while
single-rate (dense coarse arms); where the two depth counts differ, both are reported
(resolution depth / temporal-rate depth). The systems-analysis question is therefore not whether
hierarchy helps universally, but **which form pays, where, and why**. Three cells of this grid
are measured in earlier sections (allocation — §4.3.6; the spatial factorial — §4.4.1 below; the
slow-clock boundary — §4.5); the remaining cells (temporal family §4.4.2, span control §4.4.3,
depth §4.4.5) follow the pre-registered program of `SPATIAL_FACTORIAL.md` (final block), and
§4.4.5 assembles the grid into one comparison table in the manner of the Clockwork VAE benchmark
[REF: CW-VAE] — the format adopted, the controls tightened (Chapter 1, §1.3; fourth article):
matched token budgets, backbone-capacity-neutral depth (levels add tokens to a shared
backbone; per-level I/O heads +0.30 M / +1.4% on the 22.19 M predictor + the frozen 1.25 M AE
tap, disclosed), foreground-masked metrics.

**Table 4.7-A — What each packaging axis buys and costs.**

| Axis | Form | What it buys | Information or causal cost | Evidence |
|---|---|---|---|---|
| spatial | dense 4×4 coarse stream, stride 1 | 4× reach per token; per-level decodability; dense temporal coverage | −0.008 under diffusion and approximately null under regression at fixed window | §4.4.1, measured |
| temporal | full-resolution anchors, stride 4 | 4× reach per token while retaining detail at anchors | blind intervals between anchors; no cheap dense decodable level | §4.4.3: temporal package wins under diffusion on all five seeds; regression is heterogeneous |
| spatial × temporal | coarse stream, stride 2 | 8× nominal reach per token; cheapest tested loop coverage | couples both information costs and the legacy target-interface change | §4.4.2: +0.0330/+0.0338 at 512 tokens under diffusion |
| depth | additional 2×2 stream | a third representational scale | target-loss allocation, optimization load and top-stream floor are inseparable | §4.4.5: −0.0196 versus two levels at fixed budget, seed 0; appendix-grade |
| allocation | spend on reach before recency/detail | confirmed lower/interior hierarchy operating points | sacrifices full-resolution fidelity after reach is saturated | E2: hierarchy at 896/1,152; 1,792 unresolved; flat at 2,432/3,072 |

### 4.4.1 Spatial Factorial: an Economical, Head-Dependent Verdict

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
buy the memory window, and under the regression head the loss term is ≈0.

**Interaction (read G-S2).** The abstraction cost per saved token is similar at both windows for
the diffusion head (≈0.007–0.008 per 480 tokens); there is no evidence the cost shrinks as the
window grows.

**Window at fixed representation (read G-S3).** hier-long 0.1848 (W48, 1,152 tokens) versus
spat18 0.2120 (W18, 672 tokens): extending the window via coarse tokens buys −0.027 — an order
that dwarfs the abstraction cost, the same window-ordering law as Table 4.5.

**Within-family readout (measured arm ordering, not a factorial claim).** spat18-reg (672
tokens) at 0.2008 matches or beats the diffusion flat control flat-budget (1,152 tokens) at
0.2038 — regression + coarse history at 58% of the tokens. At the long window, the complete
five-seed factorial below separates a stable diffusion package effect from regression and
cross-head training-seed heterogeneity.

**Honest link to the prior spatial-gate null.** This project's earlier per-pixel spatial-gate
study on Moving MNIST found that an upper spatial level engaged unstably and added no quality — a
clean null for spatial abstraction as an *intrinsic* representational win. The factorial is
consistent with that history rather than overturning it: isolated from the window confound,
coarse history is not intrinsically better (it costs under diffusion and is at best free under
regression); its value is what its cheapness buys. The temporal-only corner (strided
full-resolution history), declared an omission in the factorial's pre-registration, is filled at
operating point A by the span control of §4.4.3.

**Figure 4.11 — Spatial factorial by prediction head.**
*Content:* the 2×2 factorial as a diagram: axes window (18/48) × history resolution (full/
coarse), each cell annotated with arm name, token count, ALL fg-MSE; paired deltas drawn on the
edges (resolution edges: −0.0082 diff / +0.0008 reg; window edge: −0.027 at fixed coarse
representation), with the combined-optimum path highlighted.
*Source:* `docs/figures/fig4_claim3_factorial.png`; deterministic source
`scripts/make_figures.py`. The figure does not promote the combined arm ordering to a
training-seed-robust head interaction.

### 4.4.2 Temporal Family and the Moving Tested Boundary

Span-per-token provides an explanatory calculation, not an exact edge law. The dense coarse
configuration first favors hierarchy at the tested 896-token point. The stride-2 coarse
configuration (`hier-s2-min`, 24 slots + 2 fine) purchases the same 48-frame reach at **512
tokens** and wins on both development seeds (+0.0330/+0.0338) and on the untouched E2 holdout
(+0.0340/+0.0381). Its reach-band effects are also positive on both E2 seeds. This is strong
evidence for the complete stride-2 configuration and is consistent with the span calculation.

It is not a pure context-stride intervention: the legacy arm also changes target slots,
target-token count, auxiliary supervision and total sequence length. Therefore the result is not
worded as an exactly moved lower edge. A future stride dose-response, including any 320-token
point, would require a fixed target interface and a new preregistration; it is not part of the
current closure portfolio. The measured regression seed-0 B1 result (+0.0639
[+0.0516,+0.0763]) supports the configuration but does not remove this confound.
**Figure 4.13:** `docs/figures/fig13_family_edges.png`; the caption treats B1 as a configuration
comparison and does not draw a continuous edge or unmeasured point.

### 4.4.3 Coarse Level or Reach Alone? Sparse Full-Resolution Control

The control `strided-flat` keeps NO coarse level: 11 full-resolution slots at stride 4 (span 48)
+ 6 recent fine = 1,088 tokens — 64 short of the 1,152 pairing, a deficit that *favors the
control*. The complete five-seed factorial returns the following package effects (positive means
the temporal package wins; negative interaction means regression shifts toward spatial
coarsening):

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | **−0.0089 [−0.0151, −0.0025]** |
| 1 | +0.0046 [+0.0007, +0.0084] | −0.0104 [−0.0146, −0.0062] | **−0.0150 [−0.0199, −0.0104]** |
| 2 | +0.0111 [+0.0070, +0.0151] | −0.0208 [−0.0270, −0.0150] | **−0.0319 [−0.0401, −0.0245]** |
| 3 | +0.0075 [+0.0037, +0.0114] | +0.0248 [+0.0181, +0.0323] | **+0.0173 [+0.0106, +0.0245]** |
| 4 | +0.0060 [+0.0027, +0.0092] | −0.0166 [−0.0229, −0.0108] | **−0.0226 [−0.0297, −0.0160]** |

Diffusion favors temporal subsampling on all five seeds (mean +0.0098, SD 0.0060, t95 [+0.0023,
+0.0173]). Regression has two temporal and three spatial material winners (mean −0.0024, SD
0.0195, t95 [−0.0266, +0.0217]). The head interaction changes sign and has mean −0.0122 with t95
[−0.0353, +0.0109], so the preregistered robust negative-shift rule fails. The clip-cluster
intervals condition on each trained model; the t intervals use training seeds as units. The
measured statement is that, at this operating point, the diffusion head consistently uses
temporally sparse full-resolution packaging, whereas the regression head is training-seed
dependent—not that one package is universally preferred or that a directional head law holds.

All cells in this factorial use the existing adaptively used evaluation pool; E4a/E4b is
seed-resolved development evidence, not a second untouched confirmation.

**Figure 4.12 — Head × package heterogeneity across training seeds.**
*Content:* five seed-resolved diffusion, regression and interaction effects. Clip-bootstrap
intervals condition on each trained model; the shaded t intervals use the five training seeds as
the inferential units.
*Source:* `docs/figures/fig17_head_package_seed_heterogeneity.png`, generated directly from
`artifacts/e4b_head_package/results.json`.

Three binding scope qualifiers (mirror of
the article §4.4.3): (1) **static world** — nothing changes between stride-4 anchors, so the
testbed is maximally kind to temporal subsampling; on data with inter-anchor dynamics the stride
aliases what a dense stream tracks—this is an operating-point result, not a general law;
(2) the abstraction cost is a property of this autoencoder's 4×4
layout code (frozen-AE discipline fixes the choice); (3) the winning control is itself a
hierarchy in the RATE sense (two tick rates, one resolution — the clockwork quadrant), so the
measured statement is that the multi-rate form beats the multi-resolution form at this operating
point under diffusion, not that hierarchy is unnecessary.

### 4.4.4 What Each Level Preserves: Blinding Probes

Rerun the trained arm's full 152-frame rollout with one context stream **blinded** (zeroed in
standardized latent space) at every step. **MEASURED (2026-07-10): asymmetric essentiality.** Blind-coarse = the predicted
retrieval signature (hard re-entry +0.073 vs h1–8 +0.022; ALL 0.2236 past flat-budget) ✓;
blind-fine = renderer collapse, expectation partially wrong (h1–8 0.2427; sharp 0.0086 below
the structural threshold; ALL below persistence — brittleness-dominated per the binding OOD
caveat); blind-top (hier3) = +0.0035 ALL only — the top stream carries almost nothing usable,
independent convergent evidence for the §4.4.5 mechanism reading. Report-only. Binding
interpretation caveat (per the
pre-registered amendment): a zeroed stream is an out-of-distribution input, so the collapse
magnitude bounds the stream's contribution from below confounded with OOD brittleness from
above — the probe licenses “stream removed → capability lost,” not a precise information
accounting. **Figure 4.14:** `docs/figures/fig14_level_blinding.png`; measured
eval-only blinding curves with the out-of-distribution qualification retained.

### 4.4.5 Consolidated Comparison: Depth, Heads and the Assembled Grid

**Table 4.7-B — Consolidated context-shape grid (152-step autoregressive rollout; `ALL`
foreground MSE, lower is better).**

| Context budget | Resolution / temporal depth | Temporal factor | Spatial factor | Head | Configuration | Reach | ALL fg-MSE |
|---:|---|---:|---:|---|---|---:|---:|
| 512 | 1 / 1 | 1 | 1 | diffusion | `flat-short` | 8 | 0.2215 (2 seeds) |
| 512 | 2 / 2 | 2 | 2 | diffusion | `hier-s2-min` | 48 | 0.1881 (2 seeds) |
| 512 | 2 / 2 | 2 | 2 | regression | `hier-s2-min-reg` | 48 | 0.1566 (1 seed) |
| 896 | 1 / 1 | 1 | 1 | diffusion | `flat-min` | 14 | 0.2104 (2 seeds) |
| 896 | 2 / 1 | 1 | 2 | diffusion | `hier-min` | 48 | 0.1725 (2 seeds) |
| 1,088 | 1 / 2 | 4 | 1 | diffusion | `strided-flat` | 48 | 0.1644 (2 seeds) |
| 1,088 | 1 / 2 | 4 | 1 | regression | `strided-flat-reg` | 48 | 0.1590 (5 seeds) |
| 1,152 | 1 / 1 | 1 | 1 | diffusion | `flat-budget` | 18 | 0.2042 (3 seeds) |
| 1,152 | 2 / 1 | 1 | 2 | diffusion | `hier-long` | 48 | 0.1762 (3 seeds) |
| 1,152 | 1 / 1 | 1 | 1 | regression | `reg-flat` | 18 | 0.2009 (5 seeds) |
| 1,152 | 2 / 1 | 1 | 2 | regression | `reg-long` | 48 | 0.1592 (5 seeds) |
| 1,152 | 3 / 1 | 1 | 2×2 | diffusion | `hier3` | 48 | 0.2044 (1 seed) |
| 1,792 | 1 / 1 | 1 | 1 | diffusion | `flat-c1` | 28 | 0.1876 (1 development seed) |
| 1,792 | 2 / 1 | 1 | 2 | diffusion | `hier-c1` | 48 | 0.1799 (1 development seed; E2 simultaneous result unresolved) |
| 2,432 | 1 / 1 | 1 | 1 | diffusion | `flat-c2` | 38 | 0.1703 (B16×30k) |
| 2,432 | 2 / 1 | 1 | 2 | diffusion | `hier-c2` | 48 | 0.1820 (B16×30k) |
| 3,072 | 1 / 1 | 1 | 1 | diffusion | `flat-long` | 48 | 0.1572 (B8×60k; 2 seeds) |
| 3,072 | 2 / 1 | 1 | 2 | diffusion | `hier-rich` | 48 | 0.1691 (B8×60k; 2 seeds) |

*Notes.* Multi-seed values are means; exact seed effects and intervals appear in the corresponding
subsections. `hier-s2-min` also changes the target interface and is therefore a whole-configuration
effect. `strided-flat` has 1,088 context tokens and `hier-long` 1,152: the comparison is near
matched, not exact. The machine-generated grid with sequence length, throughput, VRAM and
sharpness is stored in `docs/CWSTYLE_TABLE.md`.

The depth row `hier3` (2×2/4×4/8×8 factor-2 pyramid,
48 top + 24 coarse + 9 fine = the same 1,152 tokens) tests a task with a single memory scale and a
protocol-capped 48-frame history. **MEASURED (2026-07-10, s0) — the pre-registered ALTERNATIVE (loss) branch:** hier3 ALL
**0.2044** / sharp 0.0564 (class-lowest): CI-clean loss to hier-long (**−0.0196
[−0.0248, −0.0146]**), null vs flat-budget (−0.0007) — the third stream erases the two-stream
win at fixed budget. Three inseparable disclosed mechanisms: loss-weight dilution (per the
amendment; the class-lowest sharpness is its predicted signature), fixed-step optimization
load, and the top-stream information floor (2×2 = 32-px cells vs ~20-px digits, with the coarse
stream cut 48→24 to pay for the top). It is reported as a depth-plus-loss-allocation probe that
costs approximately 0.02 at fixed budget, not as evidence that depth is intrinsically harmful;
the depth hypothesis
stands (single-scale data cannot reward a third scale — consistent with the per-pixel-gate
null, the §4.5 boundary, CW-VAE's two-levels-sufficed observation). Disposition:
appendix-grade; a weight-matched follow-up would be required for any causal depth statement.
**Figure 4.15:** `docs/figures/fig12_cwstyle_curves.png`, degradation curves over the
152-frame rollout for the measured configuration grid.

## 4.5 Operating Point B

Operating point A proves *efficiency at matched budget*: the loop (≈48 frames) sits inside the
affordable flat window (48 frames), so flat-long can always buy the result with tokens. The
boundary study asks the capability question on the pre-registered design frozen before any
launch (the `docs/HGRID_DESIGN.md` preregistration trail is included in the reproducibility
appendix): in a regime where
**no affordable flat window covers the memory horizon** (hgrid3, effective loop ≈196–213 frames),
does temporal abstraction — a strided, slow-rate coarse level — deliver what flat context cannot?
The seed-0 study closed with a **negative rollout outcome under a binding pre-committed decision
rule**. The registered E5a repetition then retained the genuine single-step positive and the early
self-fed benefit, but did not reproduce the late inversion at either new training seed. This
section therefore reports a negative replication and three diagnosed methodology lessons, not a
trust boundary or capability law. The §4.3 matched-budget
result is unaffected under every branch of this study, and remains the headline of the
dissertation.

### 4.5.1 Aliasing Episode: Diagnosed Failure, Not a Discarded Run

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
regime verification of §4.1.1.

### 4.5.2 Geometry: Memory Reach versus Effective Loop Period

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

### 4.5.3 Recall Exists: Extended Evaluation and Internal Control

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
(whose 40k read was a CI-clean LOSS; the aliasing verdict stands).

### 4.5.4 Reach-Threshold Hypothesis and Preregistered Single-Step Test

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

### 4.5.5 Decisive Rollout Pair and Registered Negative Replication

The pre-registered decisive pair (the span-224 strided arm versus the long-context diffusion
baseline on hgrid3; warm-up 224 frames; uniform B = 8 × 40k training) returned **RED**: over 256
self-fed rollout frames the pre-registered gate cells are null — dt 49–96 HARD +0.0020
[−0.0071, +0.0111]; dt 97–192 −0.0028 [−0.0088, +0.0035]; dt > 192 −0.0018 [−0.0065, +0.0027] —
so the seed-0 programme did not establish rollout capability on its registered gate. Its horizon
profile nevertheless suggested an early hierarchy benefit followed by a late inversion.

E5a preregistered this early-positive/late-negative pattern and repeated the same pair at training
seeds 1 and 2. Positive effects favor the hierarchy: seed 0 is +0.0204
[+0.0133,+0.0279] at h 1–24 and −0.0064 [−0.0095,−0.0032] at h 61–256; seed 1 is +0.0082
[+0.0035,+0.0133] and +0.0074 [+0.0038,+0.0112]; seed 2 is +0.0230
[+0.0150,+0.0315] and +0.0087 [+0.0035,+0.0139]. Thus the early benefit is CI-clean at all
three seeds, but neither new seed reproduces the late-negative direction. The registered verdict
is **NOT_REPLICATED**, and the gated dose-response is not authorized. Addressability and self-fed
reliability remain distinct constructs, but the present study does not establish a stable
approximately-24-frame trust boundary. The §4.3 matched-budget result is unaffected.

**Figure 4.16 — Registered negative replication of the self-fed boundary.**
*Content:* seed-resolved paired Δ (strided − baseline) versus rollout horizon with cluster-CI
bands: early hierarchy benefit at seeds 0–2, late inversion at seed 0 only, and positive late
effects at seeds 1–2; the registered primary windows are highlighted.
*Source:* `docs/figures/fig18_trust_negative_replication.png`, generated from
`artifacts/e5a_trust_replication/results.json` and independently validated E5a manifests.

### 4.5.6 Methodological Lessons from the Boundary Study

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

## 4.6 Generalization across Standard and Synthetic Datasets

The studies of §§4.2–4.5 select a configuration and delimit it on the purpose-built testbed
family; this section carries that configuration outward — to the standard benchmark of the field
and to the synthetic suites on which the reversible-transformation method was published ([A2],
§4.1.2) — so that the dissertation's two experimental programs meet on shared ground. Scope
discipline, stated once: **§§4.6.2 and 4.6.5 are measured**; the 2D-rotation, 3D-polycube,
MMNIST-CIFAR and KTH extensions in §§4.6.3, 4.6.4 and 4.6.6 are future protocols and contribute
no projected number or performance claim.

### 4.6.1 Selected Configuration by Inference Regime

The recipe transferred below is not a preference but the measured selection of §§4.2–4.4, and it
is regime-dependent, exactly as §4.2.5 concluded. For the **deterministic best-of-1
autoregressive rollout** — the deployment regime of every long-horizon claim — the selected head
is **regression** (it wins the rollout reversal, §4.2.3, and the memory win replicates under it,
§4.3.5), combined with a context package selected to cover the relevant span with the least
measured information loss. The coarse-plus-fine package (long coarse memory + recent fine
window) is the replicated spatial option, whose coarse history is statistically free under this head
(+0.0008 [−0.0013, +0.0029], Table 4.7), at the **largest affordable window** — window extension
at fixed representation dwarfs the abstraction cost (−0.027 versus −0.008, read G-S3). The
strongest replicated spatial recipe is reg-long (seed mean 0.1593 at 1,152 tokens, §4.4), while
the cross-family temporal option remains seed/head-gated. For the **single-step best-of-N coverage regime**, the selected head is **diffusion** (it
wins every column at matched capacity and NFE, Table 4.2), with the same hierarchical context
rule. Transfers below use the rollout recipe as primary and report the diffusion head where
sampling diversity is part of the question; where a measured transfer predates this formulation,
its actual arm configuration is disclosed in place (§4.6.2 used the diffusion head, best-of-1).

### 4.6.2 OpenSTL Moving-MNIST Bridge: Measured Negative Control

The bridge announced in §4.1.3 is measured. Protocol: the canonical OpenSTL 10→10 suite on the
canonical test set (n = 10,000 sequences), with bit-verified reimplementations of the OpenSTL
metric conventions (pixel-sum MSE/MAE) and a faithful port of the standard sequence generator; a
fresh two-level autoencoder trained for this benchmark (per the §4.5.6 rule: a fresh autoencoder
per testbed variant); the unmodified hierarchical DiT backbone of §3.2.2 (22.19 M parameters),
trained B = 16 × 15,000 steps, bf16, evaluated best-of-1 with three autoregressive window steps
(4 + 4 + 2 frames). Two arms: **mm-flat** (10 fine frames of context, 640 tokens) and
**mm-hier** (10 coarse + 6 fine, 544 tokens).

**Table 4.8 — OpenSTL Moving-MNIST bridge (10→10, canonical test set, n = 10,000; pixel-sum
MSE/MAE per the OpenSTL convention).**

| Arm | MSE | MAE | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|
| mm-flat (10 fine frames, 640 tokens) | 180.91 | 229.94 | 0.7625 | 13.55 | **0.1320** |
| mm-hier (10 coarse + 6 fine frames, 544 tokens) | 181.29 | 230.45 | 0.7607 | 13.54 | 0.1389 |
| AE round-trip ceiling | 3.30 | 20.79 | 0.9917 | 30.93 | 0.0062 |
| Persistence | 286.28 | 337.89 | 0.6528 | 11.56 | 0.2533 |

Leaderboard reference (deterministic specialist regressors at ≈200 epochs, OpenSTL
documentation): ConvLSTM-S 29.80, PredRNN 23.97, PredRNN++ 22.06, SimVP+gSTA-S 26.69, TAU 24.60 aggregate MSE [REF: OpenSTL] (range parity with article Table 5.1; the strongest specialist, PredRNN++ 22.06, sets the quoted floor).

Three readings, in the order of their epistemic weight.

1. **The protocol anchor is achieved.** The harness reproduces the standard benchmark protocol
   exactly (canonical test sequences, bit-verified metric conventions), and the autoencoder
   ceiling row (3.30 against the arms' ≈181) shows the tokenizer is not the limiter: the gap to
   the specialists is owned by the dynamics model and objective, not the representation.

2. **The absolute aggregate numbers are modest, with disclosed reasons — and the "mid-pack
   aggregate MSE" clause of the §4.1.3 pre-registered expectation is refuted and reported as
   such.** The measured 180.91 stands against the specialists' 22.06–29.80: the evaluated arm is
   a 22 M stochastic latent-diffusion *generalist* at ≈15,000 optimizer steps, evaluated
   best-of-1, against deterministic pixel-regressors trained roughly an order of magnitude longer
   and optimized exactly for the pixel-sum-MSE aggregate — an aggregate that structurally
   punishes sampling variance (§3.4.3: on this ≈93%-black benchmark, aggregate metrics reward
   mean prediction). Accordingly, **no statement of the form "competitive with, or beating, the
   OpenSTL leaderboard on aggregate MSE" appears anywhere in this dissertation** — the
   measurement refutes it. What does hold: the model halves persistence's LPIPS (0.1320 vs
   0.2533) and beats persistence on every metric of the table.

3. **The flat ≈ hier null (Δ ≈ 0.4 aggregate MSE ≈ 0) is the measured short-protocol boundary,
   not a failure.** Moving MNIST has no off-screen re-entry at 10→10, so this protocol does not
   expose the re-entry-defined demand isolated at operating point A. It does not prove that
   persistent identity/velocity is irrelevant over long rollout; a valid persistent-object study
   would test that separately.
   Thus the bridge supplies a short/no-re-entry boundary observation, while the matched-budget
   hierarchy win is two-seed CI-clean (§4.3.3). The question "when does hierarchy pay" receives
   its negative control from the standard benchmark itself.

**Historical Phase C preregistration (2026-07-10; before any result).** The original
long-horizon design registered `mmL-flat` and `mmL-hier` at seeds 0/1/2, a 48-frame warm-up and
approximately 300 self-fed frames. Its continuity primary was paired foreground-MSE: replicated
practical equivalence required every seed's complete clip-cluster interval to lie inside ±0.005,
with horizon bins secondary. The design also proposed digit identity, trajectory and count
endpoints with preliminary margins ±0.02 accuracy, ±1 pixel and ±0.05 digit/frame. A sound
positive hierarchical effect was to be examined and reported. **The study did not launch.**

**Validity amendment (2026-07-11; before launch).** The registered foreground-MSE question is
retained narrowly. The legacy factor evaluator inferred identity from horizontal component order,
merged overlaps, used many-to-one trajectory matching and intersected frames scorable in both
arms; those choices cannot support identity, trajectory, coverage or equivalence conclusions. The
corrected hybrid protocol stores persistent object/glyph IDs, official train/test glyph splits,
boxes and per-object trajectories; uses one-to-one Hungarian assignment; penalizes misses and
extras; reports coverage; validates the digit classifier on rendered, overlapping and
autoencoder-reconstructed crops; and justifies native-unit margins. The original registration and
this amendment are both retained in the scientific record. The conditional protocol is documented
in `docs/PHASE_C_V2_PROTOCOL_2026-07-11.md`; no result or figure enters the dissertation until its
CPU, classifier and protocol-freeze gates pass.

### 4.6.2a Persistent-object long Moving-MNIST: a measured external boundary

The corrected E6 protocol passed every CPU, codec, classifier, production-crop, data-integrity and
protocol-freeze gate before protected evaluation. The frozen comparison used `mmL-flat` and
`mmL-hier` at training seeds 0/1/2, a 48-frame ground-truth warm-up, 300 self-fed target frames and
100 protected test clips. Both arms used the same 1,152 context-token budget, predictor regime,
codec, sampler and evaluation geometry. The generator saved persistent object and source-glyph
identity, boxes and trajectories; the evaluator used one-to-one Hungarian assignment, explicit
missed/extra-object penalties and intention-to-score coverage. The official train/test glyph split
and classifier-domain floors were verified before the holdout was opened.

**Table 4.8a — E6 persistent-object long Moving-MNIST.** Positive effects favor hierarchy; all
intervals belong to one simultaneous 95% max-statistic clip-cluster family over the 15
co-primary effects.

| Seed | Foreground MSE | Union MSE | Trajectory error, px | Identity accuracy | Object coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.0036 [−0.0074,+0.0146] | +0.0102 [+0.0019,+0.0185] | +2.4786 [+0.2130,+4.7443] | −0.0762 [−0.1106,−0.0418] | +0.0304 [+0.0016,+0.0591] |
| 1 | −0.0031 [−0.0127,+0.0065] | +0.0122 [+0.0052,+0.0192] | +1.4223 [−1.0646,+3.9092] | −0.0561 [−0.0934,−0.0188] | +0.0159 [−0.0163,+0.0482] |
| 2 | −0.0024 [−0.0131,+0.0083] | +0.0169 [+0.0093,+0.0245] | +4.1325 [+1.5271,+6.7380] | −0.0647 [−0.0955,−0.0338] | +0.0498 [+0.0161,+0.0834] |

The preregistered verdict is **MIXED_BOUNDARY**, not equivalence and not a null result. Union-mask
error favors hierarchy on all three seeds. Trajectory and coverage favor hierarchy cleanly on
seeds 0 and 2 and are directionally positive but unresolved on seed 1. Missed-object rate,
full-frame error, background error and foreground F1 also improve for hierarchy at all seeds as
secondary descriptive endpoints. Digit identity, however, favors flat context with simultaneous
intervals wholly below zero at all three seeds; foreground-only MSE is unresolved and changes
sign.

This division is consistent with the dissertation's systems interpretation. The coarse stream
retains spatial support economically—where an object is, whether it persists and how its motion
continues—but discards fine glyph information needed to preserve what the object is. Flat
full-resolution storage retains that identity fidelity better. Thus E6 extends, but does not
universally generalize, the LoopMom reach–fidelity frontier. It is a controlled synthetic external
boundary, not evidence of natural-video transfer.

![E6 persistent-object boundary](../../figures/fig22_phasec_v2_boundary.png)

**Figure 4.22 —** Persistent-object long Moving-MNIST simultaneous effects. Positive favors
hierarchy; support/continuity and fine identity fidelity separate.

### 4.6.3 Synthetic [A1]/[A2] Testbeds: 2D Rotations and 3D Polycubes

Two distinct published sources motivate this future bridge. [A2] supplies the $64 \times 64$
procedural 2-D rotation suite and its controlled temporal laws; [A1] supplies the V-BLT
2-D-polygon and 3-D-polycube sequences with entropy maps. Neither publication evaluates the
Chapter 3 reach–fidelity model. A future transfer study would therefore ask whether the selected
hierarchical configuration retains accuracy and consistency on these source-faithful suites,
using each publication's exact split, metrics, horizon and baselines. No such transfer has been
executed, so no result table or projected advantage is inserted here. The published examples
remain documented in Chapter 2, Figures 2.3–2.4.

### 4.6.4 Moving-MNIST-CIFAR

The MMNIST-CIFAR variant replaces the black background with natural-image texture, removing the
background degeneracy that makes aggregate metrics mean-seeking on standard Moving MNIST
(§3.4.3) while keeping the same no-re-entry kinematics — so the pre-registered expectation of
the §4.6.2 negative control (flat ≈ hier) carries over, now with more informative aggregate
metrics. This extension is not required to establish the present reach–fidelity result and has
not been executed; no numerical outcome is implied.

### 4.6.5 TaxiBJ: Measured Real Spatio-Temporal Forecasting Data

TaxiBJ is the one *real-data* suite of the transfer set: Beijing taxi in/out-flow grids,
$32 \times 32 \times 2$ (two flow channels), 4 → 4 frame prediction under the standard OpenSTL
protocol, scored by the benchmark's own metric implementation invoked verbatim (the
normalization convention — data rescaled to $[0,1]$ by the benchmark's loader — was verified in
the OpenSTL source before any model number was read). Instead of the anticipated dedicated
32-px model variant, the transfer used the **exact-upsample equivalent** (disclosed, registered
before the runs): frames nearest-upsampled ×2 through the *unchanged* 64-px two-level
autoencoder (2-channel input) and WindowDiT stack of Chapter 3, predictions downsampled back by
the exactly inverse 2×2 mean before scoring — an information-identical formulation that keeps
the architectural constants of §3.2 literally unchanged across every dataset of the study.
Regime: fresh autoencoder (12,000 steps), four arms at batch 64 × 15,000 steps (≈48 epochs,
comparable to the leaderboard's 50-epoch protocol), best-of-1, unconditional, seed 0.

**Table 4.9 — TaxiBJ 4→4, MEASURED (n = 500 canonical test pairs; OpenSTL convention;
leaderboard rows = OpenSTL's own 50-epoch retraining, locally verified).**

| Arm | MSE | MAE | SSIM | PSNR |
|---|---:|---:|---:|---:|
| **Flat, regression (256 ctx tokens)** | **0.3642** | **14.96** | 0.9781 | **39.83** |
| Hierarchical, regression (128 ctx tokens) | 0.4170 | 15.44 | 0.9772 | 39.64 |
| Hierarchical, diffusion (128 ctx tokens) | 0.4986 | 18.37 | 0.9692 | 37.87 |
| Flat, diffusion (256 ctx tokens) | 0.5152 | 18.44 | 0.9686 | 37.85 |
| AE round-trip ceiling (2-ch) | 0.0192 | 4.48 | 0.9987 | 50.92 |
| Persistence | 3.8268 | 48.63 | 0.7853 | 30.20 |
| TAU (OpenSTL, 50 ep) | 0.3108 | 14.93 | 0.9848 | 39.74 |
| MIM (OpenSTL, 50 ep) | 0.3110 | 14.96 | 0.9847 | 39.65 |
| PredRNN (OpenSTL, 50 ep) | 0.3194 | 15.31 | 0.9838 | 39.51 |
| SimVP+gSTA-S (OpenSTL, 50 ep) | 0.3246 | 15.03 | 0.9844 | 39.71 |
| PhyDNet (OpenSTL, 50 ep) | 0.3622 | 15.53 | 0.9828 | 39.46 |

The measured readings, stated per metric and without a leaderboard-superiority headline (the
§4.2.3 rule): the regression arm of the *unchanged generalist stack* lands **inside the
leaderboard band on MAE (14.96 — tying MIM for second-best of the quoted rows) and on PSNR
(39.83 — above every quoted row)**, at the band's lower edge on MSE (0.3642, 0.55% above
PhyDNet), and below the band on SSIM (0.9781 vs 0.9828–0.9848). Persistence is beaten ≈10.5×;
the autoencoder ceiling (0.0192, 16× below the best specialist row) shows the frozen tokenizer
is not the limiter — the pre-registered decoder-capacity fallback branch is measured-moot. The
regression-over-diffusion best-of-1 ordering replicates on real data (0.364/0.417 vs
0.515/0.499), and the no-memory-horizon boundary replicates as pre-registered: at half the
context tokens the hierarchical regression arm pays −0.053 MSE with faster per-horizon
degradation (4-frame protocol → coarse history is pure token economy). Disclosure: single seed
(a transfer row, not a claim-bearing contrast, per the campaign's two-seed rule).

### 4.6.6 KTH-64 — Future Extension

KTH (human actions, grayscale, evaluated at 64 × 64) is a natural-video extension named in the
research prospects; it is not an [A2] result. It is retained as a future external-validity test,
not as an optional empty result shell.

### 4.6.7 Boundaries of the Measured Generalization Claim

The measured generalization claim is deliberately narrow. On OpenSTL Moving MNIST the package
comparison is a no-re-entry negative control and does not match specialist aggregate MSE. On
TaxiBJ the unchanged generalist stack is competitive on some reported metrics, but the
hierarchical package loses to the larger flat regression context and only one training seed was
tested. The remaining synthetic and natural-video suites are future work. Accordingly, **a
claim of beating (or matching) the OpenSTL leaderboard on aggregate MSE is never made** — Table
4.8 refutes it (180.91 versus SimVP's 26.69), and the honest statement of that comparison is the
disclosed-generalist reading of §4.6.2.

### 4.6.8 Independent Confirmation on an Untouched LoopMom Holdout

The original test population informed qualitative monitoring and configuration selection.
Accordingly, the central reach claims were evaluated once on the newly generated, hash-pinned
holdout `loopmom-confirmation_holdout-20260712-e033abee6e04`. Before first access, the protocol
froze 26 checkpoint evaluations, two primary pairs, their geometric reach bands, the complete
metric suite and one studentized max-statistic clip-cluster family for 12 correlated frontier
contrasts. No model was trained or selected on this holdout.

For `hier-min` versus `flat-min`, the registered `14 < dt <= 48` effects are +0.0980
[0.0788, 0.1179] at seed 0 and +0.0914 [0.0736, 0.1098] at seed 1. For the B1
configuration versus `flat-short`, the `8 < dt <= 48` effects are +0.0692
[0.0543, 0.0845] and +0.0743 [0.0588, 0.0899]. Whole-population, union-mask,
full-frame, recall/F1, missed-component and coverage endpoints corroborate both directions.
B1 still changes its target interface and is therefore a whole-configuration result.

The simultaneous frontier confirms hierarchy at 896 and 1,152 context tokens, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. Thus 1,152 is the
largest tested family-wise-confirmed hierarchy point and 2,432 the smallest tested confirmed flat
point. No monotonic curve, unique crossing or interpolation is inferred. In exact memory
arithmetic, 48 fine frames cost 3,072 context tokens, while 48 coarse frames plus two or six recent
fine frames cost 896 or 1,152. The hierarchy therefore provides the same nominal 48-frame reach
with 2.7–3.4 times fewer context tokens. This is a reach-efficiency statement, not a claim of
equal retained information or equal performance at every budget.

![Untouched frontier and exact context-token arithmetic](../../figures/article_confirmed_frontier.png)

**Figure 4.18 — Untouched reach–fidelity frontier and memory arithmetic.** Positive effects favor
hierarchy. All intervals in the left panel belong to one simultaneous family; the right panel
shows exact token counts for nominal 48-frame reach.
*Source:* `artifacts/confirmation_e2/results.json`; generated by
`scripts/make_rft_result_figures.py`.

### 4.6.9 Robustness to Sampling Budget and DDIM Randomness

E7c separates a property of the trained checkpoints from dependence on one default diffusion
allocation. Four frozen seed-0 contrasts—896-token reach, the 1,152-token interior point, the
near-matched 1,088-token temporal versus 1,152-token spatial package shape, and the 2,432-token
flat-fidelity inversion—were crossed with NFE 8/16/32 at sampler seed 0 and NFE 16 at sampler
seeds 1/2. All 20 effects were covered by one preregistered simultaneous family.

All 20 point estimates retain the registered direction and 18 simultaneous lower bounds exceed
zero. The reach, interior and package-shape contrasts are family-wise clean for every variant.
The upper inversion remains positive but is unresolved at sampler seed 1
(+0.0098 [−0.0001,+0.0196]) and seed 2 (+0.0077 [−0.0013,+0.0167]). The registered verdict is
therefore `DIRECTIONALLY_ROBUST`, not universal sampler invariance. Non-empty foreground,
full-frame, background and union-mask error, recall and F1 agree in all 20 variants; component
counts are not uniformly favorable.

![Sampler and NFE robustness](../../figures/fig19_sampler_nfe_robustness.png)

**Figure 4.19 — Robustness to NFE and sampler randomness.** The two open upper-inversion points
have simultaneous intervals that include zero.
*Source:* `artifacts/e7c_sampler_nfe/results.json`.

### 4.6.10 Robustness to an Independently Trained Latent Representation

E7a is a whole-codec intervention: a second codec seed is trained independently, normalization is
recomputed, and the same seven seed-0 diffusion predictors are retrained in the new latent space.
It does not isolate codec weights, reconstruction error, latent scale or normalization. Four
original-codec and four alternate-codec effects share one new purpose-locked holdout and one
simultaneous family.

**Table 4.10 — Robustness of the four registered directions to an independent codec.**

| Codec | 896 reach | 1,152 interior | 1,088 vs 1,152 shape | 2,432 inversion |
|---|---:|---:|---:|---:|
| Original | +0.0494 [0.0339, 0.0649] | +0.0283 [0.0169, 0.0397] | +0.0253 [0.0163, 0.0343] | +0.0150 [0.0065, 0.0235] |
| Alternate | +0.0521 [0.0374, 0.0669] | +0.0439 [0.0310, 0.0569] | +0.0112 [0.0039, 0.0184] | +0.0079 [−0.0006, 0.0165] |

All eight point estimates retain direction and seven simultaneous lower bounds exceed zero. The
sole unresolved interval is the alternate-codec 2,432-token inversion. The verdict is
`CODEC_DIRECTIONALLY_ROBUST`. Codec seed is not an inferential replicate, so descriptive effect
changes do not identify a component-level latent-space cause.

![Alternate-codec robustness](../../figures/fig20_alternate_codec_robustness.png)

**Figure 4.20 — Whole-codec robustness.** The open alternate-codec upper-inversion interval
includes zero.
*Source:* `artifacts/e7a_codec_robustness/results.json`.

### 4.6.11 Privileged Pose-Indexed Explicit-Access Reference

E8a separates having relevant history from addressing and reusing it. A fixed world-coordinate
mosaic is built from raw warm-up pixels and queried at the exact saved future viewport positions.
Unobserved columns use a row prior computed only from training data. The mosaic receives raw
pixels and future poses, has no token-matched budget and is not learned; it is therefore a
privileged access reference, not an architecture or efficiency competitor.

**Table 4.11 — Simultaneous E8a effects for pose-indexed raw memory.**

| Registered contrast | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| Raw K14 minus raw K48 | +0.0723 | [+0.0548, +0.0898] |
| Raw K38 minus raw K48 | +0.0210 | [+0.0096, +0.0325] |
| `hier-min` minus raw K48 | +0.1219 | [+0.1031, +0.1407] |
| `hier-long` minus raw K48 | +0.1368 | [+0.1190, +0.1545] |
| `strided-flat` minus raw K48 | +0.1134 | [+0.0944, +0.1324] |
| `flat-c2` minus raw K48 | +0.1310 | [+0.1122, +0.1498] |

K48 reaches foreground MSE 0.0494, full-frame MSE 0.0027 and foreground recall 0.8438, versus
foreground MSE 0.1628–0.1862 for the four neural references. All neural-minus-K48 lower bounds
are positive, yielding `RAW_ACCESS_CEILING_CI_CLEAN`. Median stored world columns rise from 80 at
K6 to 154 at K48, and mean observed future-crop coverage from 0.454 to 0.809. The result shows
that reach alone does not guarantee reliable access; it does not rank the privileged mosaic
against deployable neural models.

![Privileged pose-indexed raw memory](../../figures/fig21_pose_indexed_raw_memory.png)

**Figure 4.21 — Pose-indexed raw-memory reference.** Coverage grows with the stored span and all
six registered K48 effects have positive simultaneous lower bounds.
*Source:* `artifacts/e8a_pose_mosaic/results.json`.

## 4.7 Implementation and Use of the Research Results

The results of the dissertation research have been implemented in:

- the **practical activities of ORNETIKA LLC** (Kyiv): the developed information technology was
  used in designing intelligent systems for the analysis of spatio-temporal video data and
  anomaly detection in complex technical environments; the resource-efficient methods (the
  TempVerseFormer reversible-transformer method) and adaptive entropy-based tokenization
  substantially reduced the computational cost of modeling long temporal sequences
  (implementation act of October 14, 2025);
- the **educational process of VNTU** in the disciplines *Information Technologies for Monitoring
  and Analysing the State of Complex Systems* and *Internet of Things and Intelligent Data
  Analysis* for PhD
  students of specialty 124 (implementation act of September 19, 2025).

Copies of the implementation acts are given in the appendices **[AUTHOR: assign the final
appendix letters and insert the signed act copies]**.

## 4.8 Chapter Conclusions

1. In accordance with **Task 7** of the dissertation research, comprehensive experimental studies
   of the developed methods and the intelligent information technology were conducted on the
   two-operating-point testbed family (§4.1) under an audit-hardened measurement protocol:
   paired clip-level cluster bootstrap confidence intervals (frame-iid intervals were shown to be
   2–7× too narrow at lag-1 autocorrelation +0.86), exact analytic hardness stratification with
   an audit-corrected anchoring, common random numbers, per-operating-point reference floors and
   ceilings (Table 4.1), and full disclosure of every correction and caveat (§4.1.5). The [A2]
   receipt datasets (2D/3D rotations, Moving MNIST, MMNIST-CIFAR, TaxiBJ, KTH) are retained as
   the published validations of the reversibility method; the OpenSTL bridge is **measured**
   (§4.6.2, Table 4.8): the protocol anchor is achieved, and the pre-registered "mid-pack
   aggregate MSE" clause is refuted and reported as such.

2. The matched-capacity and matched-NFE mechanism study (§4.2) established that **the generation-head ranking is
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
   regression pairing ALL +0.0334 seed 0 and +0.0501 seed 1 — two-seed. The single-step memory
   gate showed the
   underlying retrieval capability emerging late in training (null at 6,000 steps, ≈13% hard-bin
   win at 20,000 on both seeds). The claim is honestly bounded: the onset horizon is
   seed-variable (the robust claim is the win and its mid-horizon concentration); sharpness fades
   for all arms at depth (the G2b retention clause is not met at h ≥ 81; the ceiling is owned by
   the frozen decoder); and flat-long at 2.7× the tokens is the outright Pareto frontier
   (0.1564) — at operating point A, hierarchy is an efficiency point, not a capability necessity.

5. The spatial-abstraction factorial (§4.4) returned an **economic, head-conditional** verdict.
   At the fixed 18-frame window, coarse history costs diffusion −0.0082 [−0.0115, −0.0050] at
   42% token savings and is statistically free for regression (+0.0008 [−0.0013, +0.0029]); its
   value is what the savings buy. At 48-frame reach, temporal full-resolution anchors beat spatial
   coarsening under diffusion on all five seeds, while regression has two temporal and three
   spatial material winners. The head × package interaction changes sign and its five-seed t95
   crosses zero. The robust conclusion is a stable diffusion package result plus regression and
   cross-head training-seed heterogeneity, not a directional head law or one universally optimal
   package. The separate E7c inference study (§4.6.9) preserved the expected point-estimate
   direction in all 20 registered NFE/sampler variants, with 18 simultaneous lower bounds above
   zero. The 2,432-token inversion is family-wise unresolved at alternate sampler seeds 1/2;
   therefore the result is directionally robust rather than universally sampler-invariant.
   The independent-codec study E7a (§4.6.10) likewise preserves all four point-estimate
   directions: the 896 reach, 1,152 interior and package-shape contrasts remain
   simultaneous-CI clean, while the alternate-codec 2,432 inversion is +0.0079
   [−0.0006, 0.0165]. Thus the reach–fidelity account is directionally robust to both inference
   allocation/randomness and one whole-codec intervention, while its upper tested edge remains
   the precision-limited boundary. The privileged explicit-access study E8a (§4.6.11) further
   gives a clean simultaneous reference: raw K48 foreground MSE is 0.0494, and all four
   registered neural-minus-K48 lower bounds are positive. This is evidence that geometric reach
   and reliable addressing are distinct, not a token-matched model comparison.

6. The temporal-capability study at operating point B (§4.5) closed with a characterized
   single-step positive and a **registered negative replication of the proposed late trust
   boundary**: retrieval beyond the flat window exists on the position-unique wide world and is
   reach-causal (dose-response with the internal contiguous control), a memory span exceeding the
   effective loop $L_{\mathrm{eff}}$ (eq. 4.1) reliably enables single-step retrieval learning
   (2/2 span > loop runs CI-positive vs 0/4 span ≤ loop). The early rollout benefit is positive
   at all three training seeds, whereas the late inversion appears only at seed 0 and reverses at
   seeds 1–2. No stable trust boundary is therefore claimed. The negative replication, together
   with the three methodology lessons (effective-loop sizing; the autoencoder-reuse trap;
   evaluation power), constitutes the honest delimitation of this mechanism.

7. The generalization study (§4.6) carries the measured per-regime configuration selection —
   regression head for the deterministic best-of-1 rollout, diffusion for the single-step
   coverage regime, hierarchical coarse-plus-fine context at the largest affordable window —
   onto the standard and synthetic suites. Its measured component, the OpenSTL Moving-MNIST
   bridge (Table 4.8), achieves the protocol anchor (bit-verified standard metrics; the
   autoencoder ceiling 3.30 shows the tokenizer is not the limiter) and returns the **predicted
   negative control**: flat ≈ hierarchical (180.91 vs 181.29 aggregate MSE) on a benchmark with
   no off-screen re-entry — hierarchy pays where the memory horizon exists (§4.3) and is a null
   where it does not, while the aggregate-MSE gap to the deterministic specialists (22.06–29.80)
   is disclosed as the refutation of the pre-registered "mid-pack" clause, and no leaderboard
   claim is made. The valid persistent-object long Moving-MNIST study (§4.6.2a, Table 4.8a)
   returns a separate **mixed boundary**: hierarchy improves union error at all three seeds and
   trajectory/coverage cleanly at two, while flat context preserves digit identity at all three
   and foreground-only error is unresolved. It therefore supports a support/continuity-versus-
   identity-fidelity interpretation rather than universal superiority. TaxiBJ is a measured
   real-data transfer (§4.6.5); the unexecuted 2D-rotation,
   3D-polycube, MMNIST-CIFAR and optional KTH-64 protocols are retained only as future-work
   designs and contribute no projected result to this conclusion. The token-budget frontier
   completion (§4.3.6; hier-rich at 3,072 tokens, flat-short at 512) is **measured**: the
   frontier inverts at both ends (hier-rich loses to flat-long −0.0131 [−0.0201, −0.0069] at
   equal budget and regime; flat-short beats hier-short −0.0036 [−0.0063, −0.0010]) — an outcome
   stricter than the pre-registered convergence expectation, reported as measured. The final
   bounded efficiency claim is therefore discrete and empirical: hierarchy is family-wise
   confirmed at the tested 896- and 1,152-token points, unresolved at 1,792, and full-resolution
   flat context is confirmed at 2,432 and 3,072; no untested interval or unique crossing is
   inferred.

8. The results of the research have been implemented in the practical activities of ORNETIKA LLC
   and in the educational process of VNTU (§4.7, per the implementation acts).

The results of this chapter have been published in [A4] (the mechanism, memory, factorial, and
boundary studies) and are connected to [A2] through the published receipt summarized in §4.1.2.
The stable labels are bound mechanically to the final numbered publication list.

9. The chapter's findings consolidate into **scoped empirical regularities and explicit
   boundaries** (single source of truth: `docs/CLAIM_LEDGER.md`; every statement names its
   testbed, head, representation and seed level): **(L1) reach** — long-span packaging wins at
   tested lower/interior budgets when it contains the relevant re-entry past and the comparator
   does not; **(L2) fidelity inversion** — the simultaneous family confirms hierarchy at
   896/1,152, leaves 1,792 unresolved and confirms flat at 2,432/3,072, without interpolation;
   **(L3) stride-family configuration** — the 512-token stride-2 arm wins strongly but changes
   its target interface; **(L4) package shape** — sparse full-resolution history wins under
   diffusion on five seeds, while regression and the cross-head interaction are
   training-seed-heterogeneous; **(L5) depth — hypothesis, not law** — a
   depth-plus-objective-allocation probe on single-scale data; **(L6) self-fed reliability —
   negative replication, not law** — the seed-0 late inversion is not reproduced at seeds 1–2;
   and **(F4) inference robustness** — E7c retains all 20 NFE/sampler effect directions, with
   18 simultaneous lower bounds positive and the upper inversion unresolved at alternate sampler
   seeds 1/2; **(F5) representation robustness** — E7a retains all four alternate-codec point
   directions, with three family-wise clean contrasts and the upper inversion again
   precision-limited; **(F6) explicit-access reference** — E8a gives a clean privileged
   raw-memory ceiling while remaining outside token-matched model ranking; and **(X3) external
   persistent-object boundary** — E6 separates hierarchy-favoring spatial support/continuity from
    flat-favoring digit identity under one valid protected simultaneous family. Together, these
    confirmed regularities and explicit boundaries form a scoped, falsifiable theory of context
    allocation—the main scientific conclusion of the chapter.

---
