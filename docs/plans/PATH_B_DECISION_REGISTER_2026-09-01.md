# Path B decision register

**Purpose:** authoritative compact record of decisions made through 1 September 2026. Read this
file, all other files in `docs/plans/`, and the original critical review before continuing after
context compaction. If a later signed decision conflicts with this register, update this file and
record the superseded rule; do not silently rely on chat history.

## 1. Repository and preservation

1. The existing dissertation, code, and evidence chain remain a frozen fallback on
   `research/ebt-hierarchical-video`; they are not rewritten in place.
2. New work proceeds in
   `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-path-b` on
   `codex/hierarchy-allocation-v2`.
3. Use one repository and a sibling worktree, not a disconnected new repository.
4. Historical artifact names and hashes remain available for provenance. New code/configs use the
   `hvp_hca`, `experiments/v2`, `scripts/v2`, `artifacts/v2`, and English-v2 namespaces.
5. New experiment code is configuration-driven. Do not add new scientific arms to the historical
   hard-coded `hvp_rev.rhvd.ARMS` registry.
6. Every stable implementation increment is tested and committed. Do not delete or rewrite legacy
   evidence or Git history.

## 2. Dissertation language, narrative, and migration

1. The new manuscript is written directly in English. The current Ukrainian manuscript remains
   the fallback and content/provenance source.
2. Central question: how spatial and temporal hierarchy allocate limited memory in transformer
   video prediction, and how allocation, reversible computation, and information-density-aware
   routing affect quality and practical memory.
3. “Memory” is never a single undifferentiated value. Report retained inference state, training
   activations, parameters/optimizer, and temporary workspace separately.
4. Spatial hierarchy, temporal hierarchy, and computational/reversible hierarchy are distinct
   mechanisms. Their ratios are not multiplied unless an end-to-end measurement directly supports
   the combined ratio.
5. A 3--4x hierarchy gain and 2--3x reversibility gain are stretch hypotheses, not promised results.
6. Preserve the current semantic order of Chapters 1 and 2 during experiments. Chapter 1 receives
   surgical hierarchy/memory framing. Chapter 2 retains the published TempVerseFormer and V-BLT
   material from Articles 2 and 3.
7. Rebuild Chapter 3 around Article 4: allocation design, architecture, TRB protocol, experiments,
   results, operating region, and limitations.
8. Article 5 becomes a distinct integrated-model Chapter 4 only if correctness, real-state routing,
   reversible parity/memory, component ablations, and two-dataset OpenSTL competitiveness gates
   pass. Otherwise Chapter 4 is synthesis, limits, and bounded integration feasibility.
9. Transfer useful text, figures, and tables from Articles 1--3 through the content-transfer ledger.
   Preserve published pixels and attribution where appropriate; rewrite prose into one dissertation
   voice.
10. Figures have one analytical role. Main-text tables contain results, not mixed configuration and
    interpretation categories. Configuration registries and secondary diagnostics go to appendices.
11. Number a formula only if it defines an implemented operation, a measured estimand, or a relation
    used later. Move standard derivations and unused equations to appendices or remove them.

## 3. Article 4 scientific scope

1. Article 4 is a controlled hierarchy/allocation investigation, not a general natural-video
   superiority study.
2. Use the name **Toroidal Re-entry Benchmark (TRB)** in prose. `LoopMom` remains only in legacy
   artifact paths.
3. Article 4 uses TRB only. Long Moving MNIST is abandoned for this article.
4. The prediction mechanism remains the shared latent diffusion-transformer with v-prediction and
   deterministic DDIM evaluation, aligning the redesigned study with the current Article 4 model.
   Regression may appear only as a bounded robustness/control analysis, not as an unfrozen per-arm
   choice.
5. The four-frame target, target stride, target resolutions, loss, width, depth, optimizer, schedule,
   batch protocol, precision, and evaluation sampler are invariant within allocation comparisons.
6. Context allocation is the manipulated variable. A level-count experiment that changes capacity
   must disclose the parameter difference and is not called a pure allocation effect.

## 4. Article 4 data and statistical protocol

1. Development data are the existing 3,000 x 100 TRB training clips plus aligned actions.
2. Use the hash-pinned seeded 2,700/300 train/validation clip split in
   `experiments/v2/manifests/trb_v2_development.manifest.json`.
3. The old file named `loopmom_test_500x200.npy` is excluded from all new selection and confirmation
   claims because it was already used during historical development.
4. Generate a fresh 200-frame TRB confirmation set only after configurations, selection rules, and
   code commit are locked. Record actions, absolute positions, object metadata, generator revision,
   seed, and hashes. Read it once for locked confirmation configurations.
5. The development artifact's original generation seed is unknown and remains disclosed. Exact
   clip/action hashes identify the bytes used.
6. Every final Article 4 cell uses training seeds `0` and `1`. There is no one-seed primary result.
7. With two seeds, show both points, their mean, and paired seed-wise effects. Do not report a
   seed-level 95% confidence interval. Clip bootstrap intervals are conditional diagnostics only.
8. A third seed is a post-week robustness option for 6--8 locked primary configurations.

## 5. Article 4 approved experiment matrix

1. **A4-COR:** data, split, indexing, target invariance, token accounting, causal-boundary leakage, tiny overfit,
   deterministic replay, performance, convergence, and ranking-stability gates.
2. **A4-FRT:** budgets `320, 768, 1152, 1792, 2432, 3072`; policies Flat recent, Balanced recent,
   Linear age, and Geometric age.
3. **A4-ALC:** budget 1,792 only; seven policies Flat recent, Fine-heavy, Balanced recent,
   Coarse-heavy, Linear age, Geometric age, and Reach-oriented.
4. **A4-CEX:** six complete recent fine positions held fixed; coarse positions
   `0, 4, 8, 16, 32, 64`; exclusive older coarse support.
5. **A4-LVL:** one, two, and three levels at 1,792 tokens; adapt the two-seed A4-ALC winner.
6. **A4-OVR:** nested, exclusive, and residual support at 1,792 tokens.
7. **A4-STS:** budgets 1,152 and 1,792; fine quotas `64, 32, 16, 8`; recent, uniform, and geometric
   temporal placement. Reserve exactly 256 fine tokens; coarse bases contain 56 and 96 positions.
8. **A4-MEC:** mechanism diagnostics only on locked best flat, best hierarchy, and one failure; no
   new training.
9. Exact deterministic schedules are implemented in `hvp_hca.allocation`; Article 4 selection is
   pixel-independent. Learned/content-dependent routing belongs to Article 5.
10. Machine IDs use `A4_<family>_B<budget>_L<levels>_<policy>_<overlap>_S<seed>`. Dissertation prose
    uses `F`, `H-B`, `H-L`, `H-G`, `H-F`, `H-C`, and `H-R`, with exact IDs in captions/method tables.

## 6. Article 4 model and runtime protocol

1. Candidate controlled instrument: width 256, six blocks, eight heads, approximately 7.54 M
   parameters, bf16 autocast with fp32 master/optimizer weights, batch 16, four-frame target.
2. Use fused scaled-dot-product attention and the tested mixed-precision RMSNorm adapter.
3. The historical 384x8 instrument is retained for 4--6 bounded capacity-robustness repeats after
   the primary matrix, not for all 120+ one-week runs.
4. The packed builder gathers selected fine latent cells on CPU and transfers only `(B,K,C)` plus
   physical cell/time identities. It does not stage the complete dense historical fine tensor on
   CUDA.
5. Keep one rolling full optimizer checkpoint for an active run and compact immutable EMA/model
   checkpoints for selected milestones. Do not retain 120 large optimizer checkpoints.
6. Record analytic tokens, serialized state including metadata, allocated/reserved CUDA peaks,
   inclusions/exclusions, update time, total runtime, parameters, and evaluation time separately.
7. All Article 4 arms use one learned temporal-embedding table with 100 positions, covering the
   complete TRB clip. Its parameter count is therefore invariant to context reach; an arm cannot
   gain parameters merely by retaining older context.
8. Every Article 4 arm uses frame 96 as the common first-target anchor and frames 96--99 as the
   four-frame target. This is required because TRB has 100 frames and the largest registered
   context reaches 96 positions. It makes training and validation target windows identical across
   allocations: age 1 always maps to physical temporal position 95, not to an arm-relative row.
   The limitation is that the study varies clips and context allocations but not target time.

## 7. Training-length calibration authorized on 1 September 2026

1. Before freezing 8k or 12k, run two 30k calibrations at equal 1,792-token budget and seed 0:
   - `A4_COR_B1792_L1_FLAT_RECENT_EXCL_S00_T30K`;
   - `A4_COR_B1792_L2_BAL_RECENT_NEST_S00_T30K`.
2. Use the same data split, diffusion head, model capacity, optimizer, batch, target, precision, and
   validation records for both.
3. Save compact EMA/model milestones and fixed-validation metrics every 2,000 updates through 30k.
   Keep only a rolling optimizer checkpoint plus the final resumable state.
4. Evaluate the same fixed validation windows and sampler stream at every milestone. At minimum
   record diffusion validation loss, decoded full-frame MSE, foreground MSE at horizon four,
   standardized latent MSE, runtime, and CUDA peaks.
5. Compare candidate stopping points `8k, 12k, 16k, 20k, 24k, 30k`. Earlier 2k/4k/6k records are
   learning-curve diagnostics, not presumed final candidates.
6. Freeze the earliest candidate satisfying all of:
   - each arm's primary validation metric is within 2% relative of its best later value, or later
     changes are smaller than a predeclared practical tolerance;
   - the flat-versus-hierarchy direction is unchanged at all later candidate checkpoints;
   - neither validation loss nor decoded metric shows a sustained later improvement inconsistent
     with a plateau;
   - there is no divergence or systematic overfitting after the candidate.
7. If no checkpoint by 12k passes, the final matrix schedule may be 16k/20k or a revised staged
   design. Do not force 8k/12k simply to preserve the original one-week estimate.
8. The calibration procedure, curves, and selection rule are legitimate methodology evidence and
   should be described in the dissertation protocol section. They are not primary hierarchy-result
   seeds and are not reused as seed-confirmed final cells unless the final config/protocol is
   byte-identical and reuse was predeclared.

## 8. Article 5 architecture and evidence decisions

1. Article 5 integrates three resource-specific mechanisms: hierarchical retained state,
   reversible internal computation, and information-density-aware spatial routing.
2. Routing/top-k/token dropping is outside the reversible boundary. Reversible blocks reconstruct
   only their internal activations and must preserve the same external predictor transition.
3. The current second-order reversible rollout is not an Article 5 memory-only comparator because
   it changes forward dynamics.
4. Dynamic tokens use physical `(x,y)`, time/age, scale, and level metadata with coordinate Fourier
   features or coordinate-conditioned RoPE plus level embeddings. Static packed-slot embeddings are
   prohibited for moving regions.
5. Fixed-width quotas are required for dense kernels. Padding to a fixed maximum is not claimed as
   a real memory saving.
6. Dynamic routing must change stored state, not merely apply soft weights to the same fixed cells.
7. The router is trained or exposed to route variation before inference. “Turn on only at inference”
   is not accepted without a route-robustness experiment because it creates distribution shift.
8. Measure routing churn/Jaccard, coverage, entropy, seams, and flicker. Use hysteresis/EMA or fall
   back to deterministic routing if learned routing is unstable.
9. Internal ablations are Flat-Dense, Hier-Static, Flat-Rev, Hier-Static-Rev, Hier-Dynamic, and Full.
10. External comparisons rerun matched OpenSTL baselines such as ConvLSTM, SimVP, and TAU. Published
    numbers are context only.
11. Article 5 superiority requires at least two OpenSTL datasets, matched protocols, two-seed effect
    reproduction, quality within the declared 5% memory-focused tolerance (2% for parity language),
    and a material measured memory advantage. TRB evidence remains Article 4 evidence.
12. Component memory ratios are reported at their own endpoints. Full-model superiority depends on
    measured end-to-end Pareto position, not multiplication of component ratios.

## 9. Current implementation findings that remain binding

1. TRB loading is not the dominant cost; full attention is. Batch assembly was less than 0.5% of a
   measured optimized update.
2. The old q/k RMSNorm mixed bf16 inputs and fp32 affine weights, preventing fused dispatch. The
   Path B adapter reduced the measured 1,792-token update by about 15% and peak allocation by about
   17% while retaining fp32 master weights.
3. The prepared selected-fine path conflated local gather indices and absolute temporal positions.
   They are now separate and regression-tested.
4. A packed fine path exists and must be used by the new runner so dense candidate history is not
   counted or transferred as retained state.
5. Existing static grid embeddings are acceptable for deterministic Article 4 8x8 cell identities.
   They are not the final positional design for Article 5 dynamic/variable-scale regions.

## 10. Continuation checklist

Before doing new work after compaction, read:

1. every file in `docs/plans/`, beginning with this register and the master plan;
2. `C:\Users\leoni\Documents\projects\hierarchical-video-prediction\docs\reviews\DISSERTATION_CODE_EXPERIMENT_AND_RESTRUCTURE_REVIEW_2026-08-31.md`;
3. the current branch status, recent commits, run registry, and active GPU process list;
4. any A4-COR status/analysis document and immutable run manifests.

The current A4-COR gate record is `docs/plans/A4_COR_STATUS_2026-09-01.md`.

Never infer that a pilot or 30k calibration is a publishable confirmed result. Check config hashes,
seeds, split purpose, checkpoint selection, and gate status first.

## 11. Causality contract for the Article 4 diffusion predictor

1. The primary predictor jointly denoises the fixed four-frame future window. Full attention among
   the four noisy target groups is intentional and is not a target leak.
2. The causal boundary is between observations and the target window: context indices must be
   strictly earlier than the first target index, target latents must never be included in context,
   and only the known action sequence for the forecast window may condition the target groups.
3. A future-token perturbation invariance test would contradict the selected joint-window diffusion
   semantics and is therefore not an A4-COR requirement. A4-COR instead tests disjoint source
   indices, exact positional identities, fixed target horizons, and the absence of validation/test
   examples from training.
