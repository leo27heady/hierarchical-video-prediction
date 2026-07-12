# Chapter 3 — Prediction mechanisms at matched capacity: regression, diffusion and energy-based descent

> **Canonical fourth-article source.** Mechanism material belongs to the authoritative CH1–CH8
> record, although the venue-length assembly may move detail to a supplement. Preserve measured
> negatives; do not imply matched total compute where only capacity or per-step settings are
> matched.

## 3.1 Question and design

Before building a hierarchical system around any one generative mechanism, this chapter asks the
prior question: *at the model scale this study can afford, which prediction mechanism actually
commits to a correct, sharp hypothesis on the genuinely uncertain part of the next frame?* On the
testbed of Chapter 2, most of a predicted frame is a deterministic, action-conditioned warp of the
context; only the newly revealed leading strip of the pan is generative. A mechanism comparison is
therefore only meaningful if (i) all mechanisms share one backbone, one action-conditioning path
and one training protocol, and (ii) results are stratified by the exact hardness labels of §2.5,
so that the generative region is scored separately from the warp.

Part A (§§3.2–3.4) compares direct regression against latent diffusion under this matched
protocol. The qualitative outcome — regression averages over futures, sampling-based generation
commits to one — replicates a finding that is well established across the stochastic
video-prediction lineage [SVG18, SAVP18] and in latent world models [DIAMOND24]; we present it as
a *controlled replication* whose new content is the matched-capacity, matched-inference-budget,
hardness-stratified protocol, together with a methodological corrective (latent standardisation,
§3.3) that changed the ranking of the arms. Part B (§3.5) is the substantive novelty of the
chapter: a two-sided capability study of Energy-Based Transformers [EBT25] as the generative
mechanism, run with the official implementation, which decomposes the observed small-scale failure
into a recipe-dependent *landscape-ordering* component and a recipe-independent
*descent-reachability* component.

## 3.2 Shared harness

**Backbone.** All Part-A predictors are transformers of the same shape (width 256, 6 blocks, 8
heads) over the same token layout: each context frame contributes 16 coarse tokens (4×4) and 64
fine tokens (8×8), the target contributes both levels' token slots, and attention is *joint over
both levels' tokens* — the conditioning mechanism of hierarchical latent video diffusion
[MilliVid26], not upsample-and-concatenate. Parameter counts are 7.27 M (regressor), 7.53 M
(diffusion) and 7.28 M (energy-based), the differences being the mechanism-specific heads and
embeddings only.

**Conditioning and protocol.** Context is 6 frames of both levels; the target is the frame k = 4
steps after the last context frame; the four bridging actions are embedded and injected through
the same conditioning path in every arm. Training is single-step and leakage-free: context and
target are ground-truth latents, and no prediction is ever fed back (multi-step rollout is
deferred to Chapter 4). Latents are standardised per channel and per level (§3.3). Training uses
batch 48 in fp32; the regressor trains 6,000 steps (its loss plateaus, 0.096 at stop), the
diffusion arm 8,000 steps.

**Mechanisms.**
*Regression*: the backbone predicts the target latents directly (SmoothL1); at evaluation,
dropout (0.1) is kept active so that N stochastic forward passes yield N samples (MC-dropout),
making best-of-N comparable across arms.
*Diffusion*: v-prediction with a 1,000-step cosine schedule; sampling by DDIM. (Standard-technique
citations to be inserted in the LaTeX pass.)
*Energy-based descent* is studied in §3.5 with the official EBT implementation rather than our
reimplementation, for reasons given there.

**Matched inference budget.** Comparisons are made at equal function-evaluation budgets (NFE), not
equal step counts: one DDIM step = 1 forward pass; one energy-descent step = forward + gradient
≈ 2 forward passes; one regression sample = 1 forward pass. Best-of-N selection is performed in
decoded pixel space on foreground MSE (an oracle coverage measure, §2.6), never in latent space.

## 3.3 A methodological corrective: latent standardisation

The autoencoder's tanh latents have per-channel standard deviations of ≈0.12 (coarse) and ≈0.16
(fine). A diffusion model that noises such latents at unit noise scale is not training on the
schedule it nominally uses: at a nominal mid-schedule signal-to-noise ratio the signal is already
buried, the terminal distribution is far from the latent distribution, and random-normal
initialisations at sampling time are four to seven times off-scale. The same mis-scaling affects
any mechanism that initialises from unit Gaussian noise, including energy descent. Standardising
each latent channel to unit variance (training-set statistics; test sets standardised with the
*same* constants) is a one-line fix, and it materially changed this chapter's conclusion: with raw
latents the diffusion arm underperformed the regressor, and only after standardisation did
diffusion lead every column of Table 3.1. We flag this as a finding in its own right because the
failure mode is silent — losses fall, samples look plausible, and nothing errors — and because
mechanism comparisons in small latent spaces are easy to invalidate this way.

## 3.4 Part A results: regression averages, diffusion commits

Table 3.1 reports the clean standardised run on the held-out test clips (best-of-8, NFE 16, six
evaluation batches of 48; hard split = novel-strip > 0, which covers 27% of samples).

**Table 3.1 — Matched-capacity mechanism comparison (single-step, k = 4, best-of-8, NFE 16).
fg-MSE: lower is better; fg-sharp: reference ceiling 0.2412 (AE round-trip), GT 0.4785;
persistence baseline: fg-MSE 0.2367, fg-sharp 0.0515.**

| Mechanism | fg-MSE (all) | fg-MSE (easy) | fg-MSE (hard) | mean-over-8 | fg-sharp |
|---|---|---|---|---|---|
| Regression (MC-dropout) | 0.0664 | 0.0232 | 0.1808 | 0.0714 | 0.1591 |
| Diffusion (v-pred, DDIM) | **0.0512** | **0.0120** | **0.1549** | **0.0640** | **0.1886** |

Three observations. First, both mechanisms are far below persistence on every column: the
action-conditioned warp is learnt by everything, as expected on a testbed whose easy region is
deterministic given the action. Second, the separation concentrates where the theory says it
should: on the easy bin diffusion's advantage is 0.011 absolute (0.0232 → 0.0120), whereas on the
hard, genuinely generative bin it is 0.026 (0.1808 → 0.1549), and diffusion is sharper (0.1886 vs
0.1591 against the 0.2412 ceiling). Third, the regressor's mean-over-N (0.0714) is close to its
best-of-N (0.0664): MC-dropout produces near-identical samples, so extra inference-time samples
buy the regressor little coverage — it has averaged over the futures rather than distributing
mass across them. Diffusion's larger best-of-N gap (0.0640 → 0.0512) is the coverage signature of
a mechanism that commits to distinct hypotheses per sample.

Table 3.2 varies the per-sample inference budget. Regression is budget-invariant by construction
(one forward pass per sample). Diffusion already beats it at 4 DDIM steps and improves
monotonically; the comparison is not an artefact of granting diffusion a large budget.

**Table 3.2 — Matched-NFE table (best-of-8 fg-MSE, two evaluation batches).**

| NFE per sample | Regression | Diffusion |
|---|---|---|
| 4 | 0.0680 | 0.0546 |
| 8 | 0.0680 | 0.0514 |
| 16 | 0.0680 | 0.0503 |

[FIG ch3-1: decoded panels (last context | prediction | ground truth) for the regression and
diffusion arms on identical evaluation inputs. Caption: regression hedges the re-entering digit
into a blur; diffusion commits to a sharp, correctly located digit.]

We emphasise the epistemic status: this is a controlled replication of "regression averages,
diffusion commits" [SVG18, SAVP18, DIAMOND24] under matched capacity, matched conditioning and a
matched inference budget, with exact hardness stratification. It is also protocol-scoped:
best-of-8 selection is an oracle coverage regime, and under the deterministic best-of-1 rollout
protocol of Chapter 4 the ranking reverses — the regression arm beats the hierarchical diffusion
arm at equal sharpness; the two results are reconciled explicitly there. The new cells in the
comparison are the protocol itself and the third mechanism, to which we now turn.

## 3.5 Part B: a two-sided capability study of Energy-Based Transformers

> **Triage (2026-07-10, approved):** NON-CORE for the article's central claim; retained for the
> dissertation's mechanism-study mapping (decision Q6(a)). Appendix-vs-main at venue constraints.

Energy-Based Transformers [EBT25] train a scalar energy E(context, candidate) and generate by
gradient descent on the candidate. The published system-level results include autoregressive video
modelling (their §5.2), and the mechanism is attractive here precisely because descent steps are a
natural "think-longer" knob for hard re-entry frames. Our own reimplementation, trained in the
Part-A harness, collapsed to a blank, near-mean output. A reimplementation failure is
uninformative, so the study below removes our code from the loop entirely, then removes the scale
and recipe objections in turn. It is *two-sided* in the sense that we probe two separable
capabilities: whether the trained energy landscape *orders* candidates correctly (target below
mean below noise), and whether descent from the prescribed initialisation can *reach* a structured
output. The headline result is that these dissociate.

### 3.5.1 Setup: the official model and recipes, imported unchanged

The control (`hvp_rev/ebt_reference_control.py`) imports the reference repository's model classes
unmodified: `EBTTimeConcat` (RMSNorm blocks, rotary positions, causal autoregressive attention, a
prepended descent-step time-embedding token, a bias-free linear energy head yielding one scalar
per predicted frame) and `CausalReplayBuffer`. The MCMC descent control flow of the reference
`EBT_VID.forward` is transcribed line-for-line (random-noise initialisation, per-step step-size
randomisation, Langevin noise, gradients taken with respect to the pre-projection frame
embeddings, single-graph and truncated variants as per recipe). Two adaptations were unavoidable
and are flagged as such: (i) *frame-as-token*: the reference flattens each frame's SD-VAE latent
(3,136-d) into one token via a learned down-projection; we apply the identical scheme to our
frames' concatenated standardised latents (16·4·4 + 16·8·8 = 1,280-d → model width), with descent
performed in the original 1,280-d space exactly as theirs is; (ii) an additive learned *action
embedding* on the candidate tokens, since the reference video task is action-free and ours is
action-conditioned. One recipe number is re-tuned: the descent step size α, whose published values
(3,000/30,000) are calibrated to SD-VAE gradient scale; we recalibrate to our latent scale using
the reference's own movement principle (α·|∇E|·n_steps ≈ the initialisation-to-target distance in
unit-standardised space).

Both published training recipes are then run at the paper's "xxs" video shape — width 384, 6
layers, 6 heads, FFN multiplier 4, context 16 frames — 14.66 M parameters in our frame-token
instantiation, with the paper's scheduler shape (linear warm-up 5,000 steps, then the cosine with
min-lr scale 10 over a nominal 10⁶ scheduling steps), peak learning rate 4×10⁻⁴, weight decay
0.01, gradient clip 1.0, effective fresh batch 128 (micro-batch 32 × 4 accumulation), 20,000
optimiser steps, fp32 (`hvp_rev/ebt_ref_scaled.py`). The **S2 recipe** (their `ebt_s2_vae.sh`)
uses a single training landscape with randomised extra final-landscape steps, no detach through
the descent, truncated second-order graph, Langevin noise 1, replay buffer (here 16 replayed rows
per micro-batch, 64 per optimiser step), fixed α (recalibrated: 15.97). The **S1 recipe** (their
`ebt_s1_vae.sh` — the configuration behind the paper's published video generations) uses two
descent steps with detach between steps, second-order graph on every step, the loss averaged over
per-step reconstructions, no replay, no training-time Langevin noise, and a *learnable* α
(recalibrated initialisation 23.95; its own parameter group at the paper's 90,000× learning-rate
ratio, here 2.87×10⁻²). Every 1,000 steps we probe, on held-out data: the energies E(target),
E(mean latent) and E(random noise) on the final landscape; the cold-descent output's latent
distance to target (d_tgt; the mean-to-target distance is 0.964 in the same units); and the
decoded output's foreground sharpness (collapse threshold ≈0.08; AE ceiling ≈0.24). For S1 we
additionally run the paper's own inference recipe at each probe: best-of-4 by minimum energy, 4
descent steps on the final landscape, inference α = α/4 (their 7,500/30,000 ratio, probed at both
the current-α and initial-α readings), Langevin noise 1.

### 3.5.2 The S2 recipe: a stably mis-ordered landscape

The S2 run converges — training SmoothL1 settles at ≈0.597 from roughly 7,000 steps to the end —
and its landscape is *stably wrong*. At every one of the 21 probes from step 0 to step 20,000,
E(mean) < E(target): the batch-mean latent has strictly lower energy than the true target, with a
stable gap of ≈0.6–1.2 energy units from step 3,000 onward (3.7 at the step-1,000 probe, inside
warm-up). Cold descent behaves accordingly: d_tgt stays flat at 1.26–1.29 (against a mean-to-target
distance of 0.964 — the output is no closer to the target than the mean is), and decoded
foreground sharpness never leaves the collapse regime (0.008–0.014 across all probes, versus the
0.08 structure threshold and ≈0.24 ceiling). Twenty thousand converged steps at the paper's shape,
batch semantics and scheduler do not repair the ordering; the earlier objection that our negative
was an under-training artefact is thereby closed for this recipe.

One disclosure attaches to this probe: the batch-mean latent has roughly eight times lower norm
than an individual target, so a smoothness/norm preference could in principle carry part of the
E(mean) < E(target) ordering. Two features of the probe rebut a pure norm artefact: E(random
noise) sits far above both throughout, and the S1 recipe (§3.5.3) flips the *same* probe to the
correct ordering — a norm effect alone cannot produce opposite orderings on identical probe inputs.

### 3.5.3 The S1 recipe: ordering repaired, generation still out of reach

The S1 run is the informative one. Its loss falls further (0.361 at 20,000 steps, still slowly
improving), the learnable α drifts upward (23.95 → 33.27), and — unlike S2 — the landscape
*becomes correctly ordered*: after mean-lowest probes through step 5,000 (with one marginal
target-lowest reading at 2,000), E(target) is strictly lowest at all fifteen consecutive probes
from step 6,000 to 20,000, with a target-below-mean gap ranging from +0.22 to +3.09 energy units
across those probes. By the ordering criterion that indicted S2, S1's energy is *right*.

Generation still never materialises. Cold descent from the prescribed random-noise initialisation
is flat throughout: d_tgt ≈ 1.12–1.17 (again versus 0.964 for the mean itself), foreground
sharpness creeping from 0.011 to a maximum of 0.035 — a real but small trend that remains a factor
of ≈2.3 below the structure threshold and ≈7 below the ceiling. The paper's own inference recipe
does no better: best-of-4 by minimum energy with four descent steps and Langevin exploration
yields d_tgt ≈ 1.19–1.24 and fg-sharp 0.010–0.015 at every probe, at both α readings. A correctly
ordered landscape, probed under the publication's own sampler, still produces no structured output
at this scale.

[FIG ch3-2: E(target), E(mean), E(randn) versus training step, S1 vs S2 side by side. Caption: the
S2 landscape stays mis-ordered (mean below target) through 20k converged steps; S1 flips to the
correct ordering at ~6k and the gap grows — yet neither generates.]

[FIG ch3-3: cold-descent foreground sharpness versus training step for S1 and S2, with the 0.08
structure threshold and the 0.24 decoder ceiling marked. Caption: descent output stays in the
collapse regime under both recipes; S1 shows a small upward trend that remains ~2.3× below
structure.]

### 3.5.4 Decomposition: ordering × reachability

Combining the two runs with earlier diagnostic probes gives a clean two-factor decomposition of
the small-scale failure:

**(a) Landscape ordering is recipe-dependent.** S2 never orders the landscape correctly (21/21
mean-lowest probes at converged loss); S1 orders it correctly from 6,000 steps onward (15/15
target-lowest). The choice between the two published training recipes decides whether the energy
function even prefers the truth.

**(b) Descent reachability from noise fails under both, and is not bought back by inference-time
compute.** The reachability failure is visible even when the ordering is provably correct. In
positive-control experiments on the same official model (copy and shift tasks, where the target is
the last context frame, or a shifted copy of it), the trained landscape is *correct by every
probe* — target energy strictly lowest (copy: −65.6 vs mean −57.2 vs noise −14.8), and an
energy-conditioning matrix that identifies the matching context for 8/8 targets with an off-diagonal
gap of +26.4 — yet cold descent still lands blank (fg-sharp 0.030 on copy, 0.016 on shift).
Inference-time scaling does not rescue it: sweeping cold-descent steps 2 → 128 leaves d_tgt flat
(1.131–1.149); warm-starting *at the target* and descending does not hold it — at the operating
step size (α ≈ 12.6) the output drifts off the target and settles in a shallow basin between
target and mean (d_tgt 0.41 → ≈0.55), so the target is at best a weak attractor of the learnt
landscape along the descent dynamics (runaway d_tgt readings up to 85 occur only at step sizes
two orders of magnitude above the operating α, i.e. numerical divergence, not landscape shape);
and picking the minimum-energy sample among 16 cold
descents remains collapsed (fg-sharp 0.0145). A persistence-initialised variant (descent started
from the previous frame rather than noise) also fails to move towards the target (d_tgt 0.835,
fg-sharp 0.046, versus 0.348 and 0.088 for a plain regressor on the same task). A provenance note
attaches to these positive-control probes: the diagnostic log that records them drew the opposite
interim conclusion at the time (a suspected harness wiring/scale bug that would have made the
whole negative suspect); the scaled S1 run — which repairs the ordering in the same harness while
descent still fails — resolved that hypothesis, and the probes are cited here under the resolved
reading.

The two factors are separable because S1 fixes (a) while leaving (b) intact. What we observe at
16 GB-reachable scale is therefore not "EBT learns the wrong energy" simpliciter: with the
generation recipe, it learns a usefully ordered energy whose minima are nonetheless unreachable by
the prescribed sampler from the prescribed initialisation. This is consistent with theoretical
accounts of entropy collapse in mean-field microcanonical gradient-descent sampling
[MFEC24, arXiv 2403.08362], which we cite as context being instantiated, not as a mechanism we
discovered.

### 3.5.5 Scope and non-claims

This study tests a capability the EBT publication does not claim at a scale it did not run.
The paper's video-section claims (§5.2) concern validation-loss scaling rates and
energy-as-uncertainty, not sharp generation; its actual video generations come from an S1-recipe
model of the "medium" size class, trained on Something-Something-v2 through an SD-VAE latent space
on multi-GPU hardware — well beyond the single 16 GB card this study is confined to. Nothing here
is a reproduction failure, and nothing here contradicts the published claims; published small-EBT
positives in other domains (EBT-Policy in robotic control [EBTPol25, arXiv 2510.27545]) are
likewise untouched by this negative. What the study does establish is a *capability boundary with
a mechanism attached*: at the scales reachable in this work (Table 3.3), the energy-descent
mechanism does not convert even a correctly ordered landscape into committed video predictions,
while diffusion at a fraction of the parameter count does (§3.5.6). Table 3.3 makes the
configuration gap explicit so the reader can judge the extrapolation risk.

**Table 3.3 — Configuration gap between this study's fair-scale runs and the reference training
configurations.**

| | This study (S2 / S1 arms) | Reference S2 (`ebt_s2_vae.sh`) | Reference S1 (`ebt_s1_vae.sh`) |
|---|---|---|---|
| Model shape | xxs: d384, 6L, 6H, FFN 4× (14.66 M with 1,280-d frame tokens) | xxs | xxs |
| Context | 16 frames | 16 frames | 16 frames |
| Steps (warm-up) | 20,000 (5,000) | 60,000 (10,000) | 100,000 (10,000) |
| Effective batch | 128 fresh + 64 replay / 128 fresh | 256 fresh (32×4 GPU×2 acc.) + replay | 256 fresh (16×4 GPU×4 acc.) |
| Step size α | fixed 15.97 / learnable, init 23.95 (both recalibrated to latent scale) | fixed 3,000 | learnable 30,000, 90,000× lr |
| Data | synthetic 64×64 grayscale; 2,700 clips × 100 frames (≈2.2×10⁵ distinct ctx-16 windows) | Something-Something-v2, 224×224, SD-VAE latents | same |
| Hardware | 1× RTX 4060 Ti 16 GB, fp32 | 4 GPUs/node | 4 GPUs/node |
| Inference probe | best-of-4 min-energy, 4 steps, α/4, Langevin 1 (both α readings) | — | paper inference on "medium" model: 4 samples, 4 steps, α override 7,500, Langevin 1 |

### 3.5.6 Diffusion control at the same scale

The negative is comparative, not absolute: the task is generable at this scale. The Part-A
diffusion arm — a 7.53 M-parameter model, roughly half the parameter count of the scaled EBT runs,
on the same frozen latents and the same standardisation — shows recognisable digit structure in
its decoded training samples by 1,500 optimiser steps, and at convergence commits to sharp,
correctly located digits (best-of-8 fg-MSE 0.0512; fg-sharp 0.1886 against the 0.2412 ceiling;
Table 3.1, Fig. ch3-1). Whatever resource EBT's sampler is missing at this scale, diffusion does
not miss it.

## 3.6 Limitations

All Part-A results are single-seed at a single model scale (≈7.3–7.5 M) on one testbed; the
regressor and diffusion arms trained for 6,000 and 8,000 steps respectively (each past its
useful-progress point, but not step-matched), and best-of-N selection is an oracle that bounds
coverage rather than deployable accuracy. The EBT study inherits three unavoidable adaptations
(frame-as-token projection to 1,280-d latents, an added action embedding, and step-size
recalibration), any of which could in principle interact with the mechanism, although the
recalibration follows the reference's own calibration principle and the positive controls show
the harness can learn correct, context-conditioned landscapes. The scaled runs close the
under-training and recipe objections at 20,000 steps and effective batch 128, but remain a factor
of 3–5 below the reference step budgets and a factor of 2 below its batch size (Table 3.3);
we therefore claim a capability boundary at 16 GB-reachable scale, not a property of the method in
the large. The S1 sharpness trend (0.011 → 0.035 over 20,000 steps), while far below structure, is
non-zero; a longer run could in principle continue it, and our claim is scoped accordingly.

---

*Draft-only provenance notes (remove before submission).* Table 3.1/3.2, params, latent stds,
references and persistence: `logs_rhcp/step1_mech_clean2.log` (corrected hardness anchor + single
raw-GT reference; supersedes `step1_mech_clean.log` — do NOT cite the old 0.0489/0.0363 values;
headline eval: 6 batches × 48, best-of-8, NFE 16; NFE table: 2 batches). Protocol/red-team definitions: `hvp_rev/step1_mech.py`
docstring ([RT-shared], [RT-mcdrop], [RT-nfe], [RT-pxbest], [RT-sharp], [STD-FIX]). Standardisation
changed the ranking: `docs/STAGE2_DESIGN.md` (G0b) + `docs/FINAL_ARTICLE_REVIEW_2026-07-02.md` §Q2(1).
Faithfulness inventory + flagged adaptations: `hvp_rev/ebt_reference_control.py` header. Scaled-run
config: `hvp_rev/ebt_ref_scaled.py` + `logs_rhcp/ebt_scaled_s2.log` line 8 / `ebt_scaled_s1.log`
lines 8–9. S2 numbers: `logs_rhcp/ebt_scaled_s2.log` (21 probes, all LOWEST=mean; gap computed from
probe lines; loss ≈0.597 from steps 7k–20k; d_tgt 1.264–1.292; fg-sharp 0.0081–0.0144; α 15.97).
S1 numbers: `logs_rhcp/ebt_scaled_s1.log` (GOOD at 2k then 6k–20k = 15 consecutive; gap +0.22 to +3.09;
loss 0.3614; α 23.95→33.27, α-lr 0.02874; cold d_tgt 1.123–1.166, fg-sharp max 0.0345 @19k;
paper-inference lines d_tgt 1.186–1.260, fg-sharp 0.0102–0.0154). Copy/shift/reachability probes:
`logs_rhcp/ebt_verify_full.log` (copy E −65.586/−57.193/−14.840, 8/8, gap +26.374, sharp 0.0303;
shift +13.187, sharp 0.0160; steps 2→128 d_tgt 1.131–1.149; warm-start α sweep 0.406→85.141;
argmin-of-16 sharp 0.0145). Persistence-init: `logs_rhcp/ebt_ref_persist.log` (EBT 0.835/0.046 vs
reg 0.348/0.0878). Reference recipes: `C:/Users/leoni/Documents/projects/EBT/job_scripts/vid/pretrain/
ebt_s2_vae.sh`, `ebt_s1_vae.sh`, `inference/ebt_ssv2_vae.sh` (medium, samples 4, steps 4, α 7500,
Langevin 1). Diffusion-commits-by-1.5k: grid `logs_rhcp/step1_mech_clean_diff_train1500.png` +
`docs/STAGE2_DESIGN.md` G2a note. NOTE: the paper's medium-model parameter count (~400 M in project
memory) is NOT verifiable from the job scripts and is not stated in the text.
