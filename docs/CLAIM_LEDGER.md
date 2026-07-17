# Claim ledger — scientific source of truth

**Revision:** 2026-07-17 E7a terminal alternate-codec robustness integration
**Rule:** a headline claim must name its testbed, intervention, comparator, metric population,
representation, head, seeds, training regime, and evidence artifact.  Opposite signs are not
equivalence; tested brackets are not continuous crossings; configuration changes are not pure
factor interventions.

Context packaging is the manipulated design space.  **Reach, fidelity, and trust are explanatory
axes to be tested**, not assumed laws:

- reach: the relevant past is included in the package;
- fidelity: relevant information survives the stored representation;
- trust: stored information remains usable after self-fed generation.

Status vocabulary: `CONFIRMED`, `MEASURED`, `NOT_REPLICATED`, `CONFIRMATION_REQUIRED`,
`EXPLORATORY_POST_HOC`, `BLOCKED_VALIDITY`, `CONDITIONAL`, and `FUTURE`.

## Stable L1–L10 compatibility map

The public article/dissertation identifiers L1–L10 remain stable. The R/F/T/M/D/X/V rows below
are finer-grained evidence views, not replacements that invalidate existing citations.

| Stable ID | Current evidence rows | Stable meaning / status |
|---|---|---|
| L1 | R1, R2 | Reach threshold and comparator-specific observability mechanism; confirmed on the frozen untouched LoopMom holdout. |
| L2 | F1 | Upper-budget saturation/inversion regularity; tested configurations and optimizer regimes, never an exact crossing law. |
| L3 | F2 | Stride-family 512-token configuration effect; causal stride isolation remains conditional on the fixed-target control. |
| L4 | F3 | Spatial-versus-temporal package shape; diffusion replicated, regression head/seed-sensitive. |
| L5 | D1 | Depth-plus-objective/allocation probe on a single-scale testbed; appendix evidence, not a general depth law. |
| L6 | T1, P8 | Addressability and self-fed reliability are distinct questions, but the seed-0 late inversion did not replicate at seeds 1–2; no stable trust boundary is established. |
| L7 | M1, M2 | Prediction-mechanism regime dependence and the EBT characterized negative. |
| L8 | X1, P1 | External boundary programme; short controls measured, legacy Phase C factor claims invalid, hybrid Phase C conditional. |
| L9 | V1 | Reversible temporal optimization as a dissertation enabler, not the cause of package gains. |
| L10 | future counterfactual/multiscale study | A genuine multiscale/depth or counterfactual hidden-state benefit remains a future claim, not required for present closure. |

Old prose may cite L1–L10 through this map. New evidence discussions may append sublabels such as
L1-R, L2-F or L6-T, but no source should cite a missing ledger row.

## Central article claims

| ID | Defensible claim | Evidence now | Status | Required limitation / next gate |
|---|---|---|---|---|
| R1 | On LoopMom under the diffusion head, packages whose reach includes the ~48-frame re-entry past outperform shorter token-matched packages at the tested lower/interior budgets. | Untouched E2 whole-population effects: `hier-min` vs `flat-min` +0.0474 [0.0354, 0.0595] and +0.0436 [0.0326, 0.0549] across seeds 0/1. The simultaneous frontier also confirms `hier-long` over `flat-budget` at 1,152 tokens on seeds 0/1/2. | `CONFIRMED; LOOPMOM_SCOPE` | This is controlled synthetic evidence under one frozen representation/head, not a universal hierarchy advantage or natural-video generalization. |
| R2 | The effect is largest in records visible to the longer package but outside the comparator's reach. | Untouched E2 registered bands: `hier-min` vs `flat-min`, `14 < dt <= 48`, +0.0980 [0.0788, 0.1179] and +0.0914 [0.0736, 0.1098]; B1 vs `flat-short`, `8 < dt <= 48`, +0.0692 [0.0543, 0.0845] and +0.0743 [0.0588, 0.0899]. | `CONFIRMED_MECHANISTIC_INSTRUMENT` | Geometric observability is exact for the generator; it is not information-theoretic necessity. B1 remains a whole-configuration comparison. |
| F1 | The tested reach–fidelity frontier has a lower/interior hierarchy-favouring region and an upper flat-fidelity inversion. | E2 simultaneous max-statistic family: hierarchy wins at 896 and 1,152 on every tested seed; 1,792 is unresolved; flat wins at 2,432 under B16x30k and B8x60k and at 3,072 on two seeds. | `CONFIRMED_TESTED_BRACKET; REGIME_QUALIFIED` | The family-wise-confirmed bracket is 1,152 (largest confirmed hierarchy win) to 2,432 (smallest confirmed flat win). No monotonicity, uniqueness, continuous crossing or asymptotic edge is established. |
| F2 | The stride-2 B1 configuration strongly outperforms its compared short-flat configuration. | Development diffusion +0.0330/+0.0338 and regression seed 0 +0.0639; untouched E2 diffusion +0.0340/+0.0381 overall and +0.0692/+0.0743 in the registered reach band. | `CONFIRMED_CONFIGURATION_EFFECT` | Context stride, target slots, auxiliary loss, sequence length and optimization difficulty change together. Fixed-target control is required only for causal stride wording. |
| F3 | At the tested static-world operating point, temporal subsampling and spatial coarsening are not interchangeable under diffusion; cross-head generalization is training-seed-sensitive. | Diffusion temporal advantages are +0.0197/+0.0046/+0.0111/+0.0075/+0.0060 at seeds 0–4: all five are positive, with across-seed mean +0.0098, SD 0.0060 and t95 [+0.0023, +0.0173]. Regression effects are +0.0107/−0.0104/−0.0208/+0.0248/−0.0166; interactions are −0.0089/−0.0150/−0.0319/+0.0173/−0.0226. | `MEASURED; FIVE_SEED_DIFFUSION_STABLE; REGRESSION_AND_INTERACTION_HETEROGENEITY` | Development-pool evidence. Regression has two temporal and three spatial material winners; its mean is −0.0024 with t95 [−0.0266, +0.0217]. The interaction mean is −0.0122 with t95 [−0.0353, +0.0109], so the preregistered robust negative head-shift rule fails. Claim the stable diffusion package result and the cross-head scope boundary separately; do not claim a directional head law. |
| T1 | Addressable memory and self-fed reliability are experimentally separable, but the tested long-loop pair does not exhibit a training-seed-stable late trust boundary. | E5a registered primary effects (positive favors hierarchy): early h1–24 is +0.0204 [+0.0133,+0.0279], +0.0082 [+0.0035,+0.0133], +0.0230 [+0.0150,+0.0315] at seeds 0/1/2. Late h61–256 is −0.0064 [−0.0095,−0.0032], +0.0074 [+0.0038,+0.0112], +0.0087 [+0.0035,+0.0139]. Neither new seed reproduces the registered early-positive/late-negative direction. | `NOT_REPLICATED; EARLY_BENEFIT_THREE_SEED; SEED0_LATE_FAILURE_ONLY` | Do not claim or title a general trustworthy-memory boundary. E5b is not authorized. If trust is studied again, preregister a new multi-seed/task design rather than localizing the unreplicated seed-0 inversion. |
| F4 | The four central diffusion directions retain their point-estimate signs across the registered DDIM sampling seeds and NFE allocations. | On the independent E7c robustness set, all 20 effects are positive over NFE 8/16/32 at sampler seed 0 and NFE 16 at sampler seeds 1/2. Eighteen simultaneous 95% lower bounds exceed zero. The 2,432-token flat-fidelity inversion remains positive but is family-wise unresolved at sampler seeds 1 (+0.0098 [−0.0001,+0.0196]) and 2 (+0.0077 [−0.0013,+0.0167]). | `MEASURED; DIRECTIONALLY_ROBUST; UPPER_INVERSION_SAMPLER_UNCERTAINTY` | E7c is checkpoint-only on seed-0 predictors and one frozen codec/testbed. It supports directional sampler/NFE robustness, not universal sampler invariance, training-seed robustness or codec robustness. No adaptive E7c extension is allowed. |
| F5 | The four central diffusion directions survive an independently trained latent codec at the point-estimate level. | E7a's joint original/alternate-codec family is `CODEC_DIRECTIONALLY_ROBUST`. Alternate-codec effects are: 896 reach +0.0521 [0.0374,0.0669], 1,152 interior +0.0439 [0.0310,0.0569], 1,088-vs-1,152 package shape +0.0112 [0.0039,0.0184], and 2,432 inversion +0.0079 [−0.0006,0.0165]. Original controls are all simultaneous-CI clean. | `MEASURED; CODEC_DIRECTIONALLY_ROBUST; UPPER_INVERSION_CODEC_UNCERTAINTY` | Whole-codec intervention conditional on predictor seed 0 and one new holdout. Codec weights, latent geometry and normalization change together. The upper inversion is directionally retained but family-wise unresolved; do not claim component-level codec causality or universal representation invariance. |

## Supporting and boundary claims

| ID | Defensible claim | Evidence now | Status | Scope |
|---|---|---|---|---|
| M1 | Generation-head preference depends on evaluation regime, while package utilization under regression is training-seed-sensitive. | Regression wins deterministic best-of-1 rollout; diffusion has different coverage behaviour; the five-seed package factorial gives stable diffusion signs but heterogeneous regression and interaction signs. | `MEASURED_BOUNDARY` | Fixed backbone/representation and reachable scale; protocol allocation and seed-resolved effects must be reported. No directional cross-head law is established. |
| M2 | EBT is a characterized negative at the tested scale: ordering and descent reachability dissociate. | Official-code controls and two recipes. | `MEASURED_NONCORE` | Does not establish impossibility at paper scale. Move from main fourth article. |
| D1 | Adding a third level at fixed budget costs performance on the single-scale loopmom probe. | B3 seed 0 −0.0196 [−0.0248, −0.0146] versus two-level arm; disclosed objective/allocation confounds. | `MEASURED_APPENDIX` | A depth-plus-objective-allocation probe, never a general statement that depth hurts. |
| X1 | Short-horizon Moving-MNIST and TaxiBJ are competence/boundary anchors, not strong equivalence or transfer evidence. | T7 descriptive near-null; TaxiBJ in-band competence/economy. | `MEASURED_DESCRIPTIVE` | Moving-MNIST uses one seed/unequal budgets/no paired equivalence test; TaxiBJ is not a memory-re-entry test. |
| X2 | Full-frame SSIM/PSNR are background-dominated on the archived sparse Moving-MNIST protocol. | On the exact 100-clip, 36-context, 964-target CW-VAE population, deterministic all-black scores 0.7681 SSIM / 13.941 dB versus the trained best-of-5 CW-VAE mean curve at 0.5708 / 12.918. | `MEASURED_METRIC_SANITY_DIAGNOSTIC` | Same clips/targets/metric code, but not architecture- or budget-matched; only CW-VAE mean curves survive, so no paired inference. Use this only to motivate foreground/union/component endpoints, not as a model-quality ranking. |
| V1 | The stochastic reversible recurrence supports exact reverse replay and constant activation memory relative to BPTT. | Reconstruction/gradient/memory panels. | `MEASURED_DISSERTATION_STRAND` | Enabler, not cause of Chapter 4 gains; novelty scoped against reversible RNNs. Move out of core fourth article. |

## Invalid, blocked, or retired claims

| ID | Retired wording or analysis | Status | Remedy |
|---|---|---|---|
| P1 | Legacy Phase C identity, trajectory, coverage and replicated equivalence. | `BLOCKED_VALIDITY` | Persistent IDs/classes/trajectories; official train/test glyph split; Hungarian assignment; miss/extra penalties; coverage; classifier validation; justified margins. |
| P2 | “Exact arithmetic edge,” “switch occurs at 896,” or a unique crossing in (1792,2432). | `RETIRED_OVERCLAIM` | Use largest/smallest tested winning configurations and tested brackets only. |
| P3 | B1 as a pure context-stride intervention. | `RETIRED_CAUSAL_OVERCLAIM` | Call it a whole-configuration result until the fixed-target-interface control lands. |
| P4 | B8x60k as twice the data or closer to asymptotic training. | `RETIRED_REGIME_OVERCLAIM` | Same examples; changed batch, updates, noise, grouping and effective EMA horizon. |
| P5 | Averaged regression “tie”/equivalence or a stable regression-induced shift toward spatial packaging. | `RETIRED_EQUIVALENCE_AND_HEAD_LAW` | Report all five regression effects (+0.0107/−0.0104/−0.0208/+0.0248/−0.0166), the two-temporal/three-spatial sign counts and the interaction t95 [−0.0353, +0.0109]. |
| P6 | Geometric labels as information-theoretic necessity. | `RETIRED_TERMINOLOGY` | Use exact geometric observability. Reserve necessity for paired counterfactual worlds. |
| P7 | OOD blinding loss as a lower bound on stream contribution. | `RETIRED_INTERPRETATION` | OOD corruption can inflate loss. Prefer in-distribution shuffle/freeze/cross-clip or trained missing-stream controls. |
| P8 | A stable self-fed trust boundary located after approximately 24 frames, or a general late inversion of the hierarchy advantage. | `NOT_REPLICATED` | Report the complete E5a seed table: early hierarchy benefit is positive at all three seeds, while the late inversion occurs only at seed 0 and reverses at seeds 1–2. |

## Cost and comparison vocabulary

Every table must report separately:

- context tokens;
- target tokens;
- total sequence length;
- parameters;
- training examples;
- optimizer updates;
- batch size and EMA;
- per-step and total NFE;
- runtime and peak VRAM when measured.

Use “matched context-token budget,” “parameter-matched,” “matched examples,” or “matched total NFE”
only when literally true.  Do not use “matched compute” as a substitute for these distinctions.

## Confirmation and expansion order

1. Finish the already-running seed-2 trio and c2; stop before Phase C.
2. Freeze current claims and analysis/table shells.
3. Validate the complete metric suite and immutable provenance on CPU.
4. Evaluate the untouched confirmation set once. **COMPLETE: E2 PASS.**
5. Complete the regression seed-2 factorial because F3 remains central, after H3b parity. **COMPLETE: E4a and E4b PASS; diffusion is five-seed-stable, while regression and the head interaction are heterogeneous. E4b is terminal and authorizes no seeds 5+.**
6. Replicate/isolate trust. **COMPLETE: E5a NOT_REPLICATED; E5b/E5c not authorized; remove trust from the title-level empirical claim.**
7. Run E7c sampler/NFE robustness on its separate locked set. **COMPLETE:
   DIRECTIONALLY_ROBUST; 20/20 effects retain sign and 18/20 simultaneous lower bounds are
   positive; the upper inversion is family-wise unresolved at sampler seeds 1/2.**
8. Prepare and separately gate E7a alternate-AE robustness, then E8a classical pose-indexed
   retrieval. **E7a COMPLETE / CODEC_DIRECTIONALLY_ROBUST; proceed only through E8a's separate
   prelaunch gates.**
9. Assemble and internally freeze the article/dissertation sources; run the fixed-target B1
   control only if causal stride wording returns.
10. Run redesigned Phase C only if its validity/classifier and assembled-article utility gates pass.
11. Treat counterfactual/dynamic-content and natural re-entry studies as post-freeze/future work.

Pre-C seed-2 and c2 results must be integrated as new evidence rows without silently upgrading the
scope of these claims.
