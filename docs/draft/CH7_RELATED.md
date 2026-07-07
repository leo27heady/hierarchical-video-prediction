# Chapter 7 — Related work and honest scoping

> **Status (2026-07-07):** journal-register prose, content-complete. STANDING DEBT carried at the
> end: every citation-mechanism claim below must be verified against the paper PDFs before
> submission, plus a fresh prior-art sweep (<60-day collisions). [REF: …] tags resolve with the
> bibliography build (COMPLETION_PLAN E6).

## 7.1 Hierarchical latent video prediction

The coarse-carries-slow-state idea runs from Clockwork VAE [REF: CW-VAE] (upper levels tick more
slowly, retaining longer temporal context) through DLH [REF] — which is already a
*learned-clock* CW-VAE, the pre-emption that closed our own adaptive-clock direction — to
TECO [REF], MSPred [REF], NUWA-XL [REF], and the hierarchical latent video diffusion family
[REF: MilliVid] that our system miniaturizes, with FDM [REF: Harvey et al. 2022] as the
flexible-conditioning relative. Our own first step in this line is the two-stage V-BLT
architecture [REF: A1] — a local byte-level encoder/decoder around a global latent transformer
with entropy-adaptive tokenization: hierarchy as *adaptive decomposition*, before the present
article's hierarchy as *memory packaging*. What none of these works supply — and what this
article adds — is the **matched-token-budget frontier with an exact memory-demand label**: prior
comparisons credit "hierarchy" for what may be "more context," and prior benchmarks rarely
require memory at all (our Ch. 5 negative control makes that point on Moving MNIST directly).
The coarse-carries-memory *architecture* is therefore adopted here, not claimed; the measured
*conditions under which it pays* (the win band and its geometric edges) are the contribution.

## 7.2 Memory in world models

Purpose-built memory benchmarks exist at scale: Memory Maze [REF], TECO-Minecraft [REF], and the
recent long-context world-model evaluations [REF: LSSVWM 2505.20171; WorldPack 2512.02473;
VidMem 2506.05284]. Our testbed is deliberately the *minimal fully-analytic member* of this
family: a one-dimensional pan over a procedural torus, where hardness ("recoverable only from
memory") is an exact label computed from recorded actions rather than an estimate, floors and
ceilings are measured per operating point, and the whole protocol trains and evaluates on one
16 GB GPU. We trade visual richness for exactness and reproducibility — the right trade for a
mechanism study, and the wrong one for a capability demo (stated as a limitation in §2.9).

## 7.3 Prediction-mechanism comparisons

"Regression averages, diffusion commits" descends from the stochastic-video line
[REF: SVG; SAVP] and the diffusion world-model line [REF: DIAMOND]; our Chapter 3 contributes
the *controlled* version (one frozen latent interface, matched capacity and NFE) plus the
regime-dependent reversal at rollout (best-of-N coverage vs best-of-1) that reconciles the two
regimes. For energy-based prediction, our two-sided study engages the Energy-Based Transformer
[REF: EBT 2507.02092] on its official implementation: one recipe yields a stably mis-ordered
landscape, the other repairs the ordering while descent still fails to reach structured outputs
at our scale. We explicitly do NOT contradict published small-EBT positives on other tasks
[REF: EBT-Policy 2510.27545]; and the collapse mechanism we measure instantiates — does not
discover — the mean-field gradient-descent entropy-collapse analysis [REF: 2403.08362]. The
configuration gap between our runs and the paper's (steps, batch, α, data, hardware) is
tabulated in Ch. 3 Table 3.3 precisely so the reader can judge the extrapolation risk.

## 7.4 Rollout training and reversibility

Rollout-training objectives that stabilize self-feeding are an active line [REF: Self-Forcing,
NeurIPS 2025]; our reversibility theorem is the *memory enabler* such objectives would need at
long horizons, not a competitor to them. The reversible-computation lineage: RevNet [REF: Gomez
et al. 2017] (depth axis), reversible RNNs [REF: MacKay et al. 2018] (time axis; the
no-forgetting accounting we extend), CrevNet [REF], invertible-network stability analysis
[REF: Behrmann et al.] (whose lesson our P3 instantiates), and on the inner-optimization axis:
reversible learning [REF: Maclaurin et al. 2015] (the storage arithmetic we apply), reversible
diffusion samplers [REF: EDICT; DOODL; MALI] (pre-empting that variant), and the large-scale
practice of AlignProp/DRaFT [REF] — which choose checkpointing precisely because state ≪ graph,
the regime our measured 0.26% panel confirms.

## 7.5 Explicit non-claims

For the record, this article does **not** claim:

1. **No beats-checkpointing claim.** Reversibility is a constant-memory capability vs vanilla
   BPTT and ~Pareto with checkpointing (measured); the inner-loop axis is closed at 0.26%.
2. **No universal EBT refutation.** The negative is scoped to 16 GB-reachable scale, two
   published recipes, our data regime; the config-gap table quantifies the distance to the
   paper's setting.
3. **No hierarchy-dominance claim.** Hierarchy wins a bounded budget band; flat wins outside it
   (below the span threshold and at window saturation) — measured, with CIs, at four budgets.
4. **No rollout-stabilization claim for reversibility.** Invertibility ≠ stability is one of our
   own results (P3).
5. **No leaderboard claim on OpenSTL suites.** Aggregate pixel-sum MSE on mostly-static
   benchmarks structurally favors deterministic per-benchmark regressors trained ~10× longer;
   our bridge row (Ch. 5) is a protocol anchor and a negative control, with the gap disclosed
   (180.91 vs specialists' 22.06–29.80 on Moving MNIST).
6. **No FVD headline** (I3D/Kinetics mismatch at 64×64 grayscale) and no aggregate-SSIM/PSNR
   claims on background-dominated benchmarks (measured: all-black beats trained models there).

## 7.6 Citation-verification debt (pre-submission checklist)

Every mechanism attribution above ("DLH = learned-clock CW-VAE", "AlignProp chooses
checkpointing because state ≪ graph", "EBT-Policy is a small-EBT positive", the three memory
benchmarks' protocols, EDICT/DOODL/MALI scope) is to be re-verified against the PDFs at
bibliography build; prior-art re-sweep at submission for <60-day collisions on: matched-budget
hierarchy analyses, analytic memory-horizon benchmarks, invertible stochastic recurrences.
