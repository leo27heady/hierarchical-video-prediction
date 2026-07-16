# Chapter 7 — Related work and honest scoping

> **Detailed fourth-article source chapter; not the binding submission narrative.** Final
> positioning acknowledges recent memory benchmarks, compression/retrieval systems and Self-Forcing-class
> work. Novelty is the controlled instrument and package analysis, not the existence of
> memory-focused evaluation.

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

**The CW-VAE benchmark as a presentation ancestor — adopted format, tightened controls.** Our
shape study (§4.4) deliberately adopts the *comparison format* of the Clockwork VAE benchmark:
one configuration table, quality-retention curves over long open-loop rollouts, and level-content
probes (their §5.1 prior-following experiment; our stream-blinding probes, sharpened by per-level
decodability). We adopt the format while tightening three controls it lacks. First, capacity:
CW-VAE's compared models differ in parameter count (RSSM 5M vs CW-VAE 12M in their own parameter
table; only their NoTmpAbs ablation is matched), whereas our depth axis is backbone-capacity
neutral — levels add tokens to a shared backbone, with only per-level I/O heads added (+1.4%
parameters, disclosed in §2.4). Second, context cost: a recurrent
state has no token-budget axis, so hierarchy-vs-flat comparisons in the RNN family cannot hold
context cost fixed; our grid is budget-matched by construction, which is what licenses the
frontier reading. Third, metrics: their evaluation aggregates SSIM/PSNR/LPIPS/FVD over full
frames — and on sparse-foreground video we measured a constant all-black frame outperforming a
trained CW-VAE on aggregate SSIM/PSNR (§2.8) — so our cells are foreground-masked and
re-entry-stratified. It is also worth stating what CW-VAE's paper does and does not show about
depth: its depth evidence is an information-allocation (KL-per-level) analysis, not a
prediction-quality dose–response, and the authors themselves observe that a two-level model
sufficed for their maze dataset; our table carries the quality-per-depth cells directly,
including the measured single-scale depth probe (§4.4.5: a cost at fixed budget, with the
objective-allocation confound disclosed — agreeing with their observation from the quality
side). VTA [REF] (a two-level model whose lower
level resets when the upper level ticks — the reset artifacts CW-VAE documents) and the clockwork
lineage [REF: Koutník et al. 2014, Clockwork RNN] are the temporal-abstraction relatives of our
strided arms; NoTmpAbs is the conceptual ancestor of our "same budget, no abstraction" flat
controls.

## 7.2 Memory in world models

Purpose-built memory benchmarks and retrieval/compression systems already exist at scale: Memory
Maze [REF], TECO-Minecraft [REF], WorldPack/LoopNav [REF: WorldPack2025], MBench
[REF: Zhang2026mbench], CaR/SceneFly [REF: Peng2026car], GIM-World [REF: Wei2026gimworld],
MemLearner [REF: Yu2026memlearner], and the dynamic disappear/reappear MemoBench
[REF: Chen2026memobench]. Our testbed is deliberately a minimal analytic member of this family: a
one-dimensional pan over a procedural torus, where geometric observability is computed from
recorded actions and the whole protocol fits one 16 GB GPU. Shared digit priors still permit
statistical guesses, so observability is not called information-theoretic necessity. We trade
visual richness for a controlled mechanism study and do not present it as a capability benchmark.

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

Temporally sparse or keyframe-conditioned history is standard practice in video modeling; this
article does not claim to invent it. The contribution is the matched-context-budget causal
comparison against spatial coarsening, on a benchmark with exact re-entry labels, together with
the measured interaction with prediction head and the resulting boundary conditions.

For the record, this article does **not** claim:

1. **No beats-checkpointing claim.** Reversibility is a constant-memory capability vs vanilla
   BPTT and ~Pareto with checkpointing (measured); the inner-loop axis is closed at 0.26%.
2. **No universal EBT refutation.** The negative is scoped to 16 GB-reachable scale, two
   published recipes, our data regime; the config-gap table quantifies the distance to the
   paper's setting.
3. **No hierarchy-dominance or exact-edge claim.** The simultaneous family confirms hierarchy at
   the tested 896/1,152-token points, leaves 1,792 unresolved, and confirms flat at 2,432/3,072.
   The B8×60k c2 result is a matched-example optimizer-regime control. E7c retains all four
   central directions across its NFE/sampler grid, but the upper inversion is family-wise
   unresolved at alternate sampler seeds 1/2. None of this establishes monotonicity, a unique
   crossing or universal sampler invariance.
3a. **No depth claim on single-scale data.** Measured (§4.4.5, s0): the three-stream probe
   COSTS at matched budget on this testbed, with the objective-allocation confound disclosed;
   depth is claimed only as bounded by the data's memory-scale count — never as generally
   beneficial, and never as generally harmful either.
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
