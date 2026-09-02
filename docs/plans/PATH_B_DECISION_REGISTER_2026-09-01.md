# Path B decision register

**Purpose:** authoritative compact record of decisions made through 2 September 2026. Read this
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

1. **Superseded on 2 September 2026 by §§13--15.** The original Path B decision was to write a new
   English manuscript and retain Ukrainian only as fallback. The current decision is to refactor
   the Ukrainian manuscript as the foundation while drafting new/materially revised blocks in
   English.
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
   material from Articles 2 and 1 respectively; Article 3's static consistency synthesis remains in
   Chapter 1.
7. **Superseded on 2 September 2026 by §§13--15.** Chapter 3 is not rebuilt as a standalone Article
   4 chapter. It retains its current architecture/method role while receiving the registered
   allocation programme and the new integrated architecture.
8. **Superseded on 2 September 2026 by §§13--15.** There is no distinct promised Article 5.
   Hierarchy and integrated OpenSTL evidence form one new article across the retained Chapter 3
   method and Chapter 4 results roles; integration claims remain conditional on completed gates.
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

The fully expanded, deduplicated matrix and description of every family is authoritative in
`docs/plans/ARTICLE4_COMPLETE_EXPERIMENT_MATRIX_2026-09-01.md`. It also records the pending
16k-versus-30k interpretation and the pre-launch A4-LVL/A4-OVR definition gaps.

1. **A4-COR:** data, split, indexing, target invariance, token accounting, causal-boundary leakage, tiny overfit,
   deterministic replay, performance, convergence, and ranking-stability gates.
2. **A4-FRT:** budgets `320, 768, 1152, 1792, 2432, 3072`; policies Flat recent, Linear age, and
   Geometric age. Balanced recent remains in A4-ALC but is excluded from A4-FRT.
3. **A4-ALC:** budget 1,792 only; seven policies Flat recent, Fine-heavy, Balanced recent,
   Coarse-heavy, Linear age, Geometric age, and Reach-oriented.
4. **A4-CEX:** six complete recent fine positions held fixed; coarse positions
   `0, 4, 8, 16, 32, 64`; exclusive older coarse support.
5. **A4-LVL:** one, two, and three levels at 1,792 tokens; adapt the two-seed A4-ALC winner.
6. **A4-OVR:** nested, exclusive, and residual support at 1,792 tokens.
7. **A4-STS:** budget 1,792 only; fine quotas `64, 32, 16, 8`; recent, uniform, and geometric
   temporal placement. Reserve exactly 256 fine tokens; the coarse base contains 96 positions.
8. **A4-MEC:** mechanism diagnostics only on locked best flat, best hierarchy, and one failure; no
   new training.
9. Exact deterministic schedules are implemented in `hvp_hca.allocation`; Article 4 selection is
   pixel-independent. Learned/content-dependent routing belongs to Article 5.
10. Machine IDs use `A4_<family>_B<budget>_L<levels>_<policy>_<overlap>_S<seed>`. Dissertation prose
    uses `F`, `H-B`, `H-L`, `H-G`, `H-F`, `H-C`, and `H-R`, with exact IDs in captions/method tables.

11. The deduplicated primary target is 43 unique configurations and 86 two-seed training runs:
    18 A4-FRT, four A4-ALC additions, six A4-CEX, one A4-LVL addition, two A4-OVR additions, and
    12 A4-STS. A4-COR calibrations and A4-MEC evaluations are tracked separately.
12. The dependency rule is locked: Flat remains the scientific winner if it wins A4-ALC, while
    A4-LVL uses the best non-flat hierarchy. A4-OVR uses the best overlap-feasible hierarchy among
    Fine-heavy, Balanced, and Coarse-heavy. Exact three-level and overlap semantics are implemented
    and tested in `hvp_hca.allocation` and the packed model/data path.
13. A4-STS uses nested coarse/fine support in the authoritative generator. The older `EXCL`
    seed-0 scaffold is non-authoritative and must be regenerated before launch.

## 6. Article 4 model and runtime protocol

1. Candidate controlled instrument: width 256, six blocks, eight heads, approximately 7.54 M
   parameters, bf16 autocast with fp32 master/optimizer weights, batch 16, four-frame target.
2. Use fused scaled-dot-product attention and the tested mixed-precision RMSNorm adapter.
3. The historical 384x8 instrument is retained for 4--6 bounded capacity-robustness repeats after
   the primary matrix, not for all 86 primary runs.
4. The packed builder gathers selected fine latent cells on CPU and transfers only `(B,K,C)` plus
   physical cell/time identities. It does not stage the complete dense historical fine tensor on
   CUDA.
5. Keep one rolling full optimizer checkpoint for an active run and compact immutable EMA/model
   checkpoints for selected milestones. Do not retain 86 large optimizer checkpoints.
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
9. The paired calibration completed on 1 September 2026. Every recorded metric rejects 8k--24k
   under the registered 2%/ranking rule and recommends 30k. Leonid confirmed 30,000 updates for
   every primary Article 4 run on 1 September 2026.
10. The result invalidated the original one-GPU/one-week full-matrix estimate. After removing
    Balanced from A4-FRT and the 1,152-token A4-STS surface, the 86-run matrix is estimated at
    179--193 training GPU-hours at 30k, or 95--104 training GPU-hours at 16k. Including evaluation
    and a recovery margin, reserve 205--234 GPU-hours at 30k or 111--130 at 16k.
11. “Ranking reversal” denotes the observed seed-0 validation ordering crossing between 12k and
    16k (hierarchy better early, Flat better from 16k onward). It is not a code operation or model
    feature. At 16k the later ordering is stable, but neither arm satisfies the registered 2%
    convergence rule.
12. The 16k contingency is superseded. Every primary cell trains to 30k, and the final 30k
    checkpoint is the primary endpoint. Intermediate validation points are diagnostics, not
    per-arm early-stopping choices.

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

## 12. Full-campaign execution requirements requested on 1 September 2026

Leonid confirmed the schedule. The following requirements are implementation-authorized:

1. Provide one command/script that consumes an immutable ordered list of every agreed Article 4
   experiment and runs it to completion.
2. The campaign is idempotent and resumable: completed config hashes are skipped, interrupted runs
   resume from their latest rolling checkpoint, duplicate active runs are rejected, and campaign
   state is durable across process/host restarts.
3. Local structured artifacts remain authoritative: run/campaign manifests, JSONL train and
   validation metrics, status transitions, stdout/stderr, checkpoints, hashes, Git/environment/data
   provenance, timing, throughput, CUDA peaks, retained-state accounting, and failure traces.
4. Create a new W&B project after confirmation (proposed name
   `hierarchical-video-prediction-article4-v2`). Use deterministic run IDs and resume the same W&B
   run after interruption. Verify authentication/connectivity before campaign launch.
5. Mirror comprehensive train/validation metrics, optimizer state summaries, memory/timing metrics,
   allocation metadata, fixed validation prediction panels, ground truth, error maps, and useful
   tables/artifacts to W&B. A transient W&B failure must not destroy a scientific run; unsynced
   local evidence remains recoverable and explicitly marked.
6. Supply a dry-run/preflight command that validates config uniqueness, hashes, split isolation,
   target invariants, token accounting, disk/GPU capacity, W&B access, and estimated queue duration.
7. Supply an English architecture/data-flow/configuration description, operating guide, exact
   launch/resume/status commands, and troubleshooting/recovery instructions.
8. Do not generate configs, create the W&B project, or launch post-calibration experiments before
   Leonid explicitly confirms the update count and execution path. This condition was satisfied on
   1 September 2026; the 80 concrete wave-1 specs and the six-run dependency resolver are prepared.
9. Authoritative campaign ledger:
   `experiments/v2/campaigns/article4_primary_30k_2026-09-01.json`; single controller:
   `scripts/v2/run_article4_campaign.py`; worker: `scripts/v2/train_article4_campaign.py`.
10. W&B project: `hierarchical-video-prediction-article4-v2`; group:
    `article4-30k-primary-2026-09-01`. Local evidence is canonical and can be replayed after an
    outage with `scripts/v2/sync_article4_wandb.py`.
11. The single controller continues beyond the 86 training runs. It aggregates final 30k
    development metrics, writes a hash-locked selection ledger, generates and audits a fresh TRB
    confirmation set, evaluates the locked shortlist, and records bounded A4-MEC attention and
    re-entry diagnostics through `scripts/v2/finalize_article4_campaign.py`.
12. The post-training selection roles are fixed before launch: best Flat at 1,792; best non-flat
    A4-ALC hierarchy at 1,792; the worst A4-ALC hierarchy as the mechanism failure case; and the
    smallest hierarchical state within 5% of matched Flat quality as the memory-Pareto candidate.
    All values are two-seed arithmetic means from final 30k development validation only.
13. Fresh TRB confirmation is fixed at 500 clips x 200 frames, seed `26090141`, world 192,
    viewport 64, three digits, speed 4, `p_cont=0.9`, and `p_hold=0.1`. The finalizer writes the
    selection ledger before generating or reading the holdout, then checks hashes, action/position
    consistency, uniqueness, and exact collision absence against development data.
14. The complete implementation/operations record is
    `docs/plans/ARTICLE4_30K_CAMPAIGN_IMPLEMENTATION_AND_OPERATIONS_2026-09-01.md`.

## 13. Single-article and integrated-model decisions confirmed on 2 September 2026

These decisions supersede the earlier assumption that the redesigned hierarchy study and the
integrated model require two additional articles.

1. Prepare one new article. Its first evidence block is the controlled TRB hierarchy/allocation
   campaign formerly labelled Article 4; its second evidence block is the integrated OpenSTL model
   formerly labelled Article 5. Legacy `A4_*` and `A5_*` identifiers remain stable provenance IDs,
   not promises of separate publications.
2. The dissertation remains in the existing Ukrainian manuscript tree. Preserve the present
   structure and most existing prose, especially Chapters 1 and 2. New or materially rewritten
   passages are drafted in English. Chapters 3 and 4 retain their current semantic roles unless
   the final evidence makes a bounded reorganization necessary.
3. Do not start manuscript rewriting while the integrated architecture is unresolved. Later edits
   add the spatial--temporal allocation framing to Chapter 1, replace obsolete Chapter 3/4
   experiments with the registered studies, remove redundant formulae and composite figures, and
   transfer useful Articles 1--3 material through the provenance ledger.
4. Final integrated quality confirmation uses only OpenSTL Moving MNIST and TaxiBJ, with seeds 0
   and 1: four long full-model runs in total. Moving MNIST is the synthetic benchmark; TaxiBJ is
   the real-data benchmark. Long Moving MNIST is excluded.
5. Four full-model runs do not replace component evidence. Component memory effects are measured
   with paired, same-checkpoint profiles and bounded development ablations. Article 4 already
   supplies the broad hierarchy quality--memory study; the OpenSTL block tests whether the selected
   mechanisms compose.
6. A separate HVP worktree `hierarchical-video-prediction-integration` on branch
   `codex/integrated-openstl-v2` holds scientific contracts, manifests, profilers, and plans. A
   clean OpenSTL worktree `OpenSTL-integrated` on branch `codex/hvp-integrated-v2` holds the native
   OpenSTL method. The live Article 4 worktree and Leonid's modified OpenSTL checkout are read-only.
7. The legacy fixed-region/packed-SimVP/second-order prototype is feasibility evidence only. It is
   not the final integrated architecture because fixed learned region slots cannot identify moving
   routes and second-order rollout changes multistep forward dynamics.
8. The final reversible mechanism is confined to deterministic internal transformer blocks. The
   ordinary and reversible modes execute the exact same forward equations; numerical output,
   input-gradient, and parameter-gradient parity is a release gate.
9. Routing is a non-reversible boundary before the core. Dynamic detail tokens are positioned by
   normalized physical `(x,y,time,scale,level)` metadata. Packed tensor slot is never treated as
   spatial identity.
10. The final hierarchy keeps coarse coverage for every observed time and selects a real fixed
    quota from fine candidates across both space and time. The selection tensor is gathered before
    the transformer; dense candidates are not padded into the retained state.
11. Dynamic routing at the same quota is a quality/allocation mechanism at equal retained memory,
    not an independent memory multiplier. It may demonstrate a memory win only by matching quality
    at a smaller measured quota. Hierarchy reduces retained/attention state; reversibility reduces
    training activation storage. Ratios are never multiplied into a claimed end-to-end ratio.
12. A model trained only with static membership is not assumed to support dynamic routing at
    inference. Inference switching is admissible only after route-aware or mixed-route training and
    explicit static/dynamic switch, stability, and quality tests.
13. During the live Article 4 GPU campaign, integrated-model work is limited to static inspection,
    lightweight CPU tests, documentation, and configuration generation. It must not allocate CUDA,
    duplicate datasets/checkpoints, or dirty the pinned Article 4 worktree.
14. The detailed architecture, gates, experiment matrix, and operating sequence are recorded in
   `docs/plans/INTEGRATED_OPENSTL_MODEL_AND_SINGLE_ARTICLE_PLAN_2026-09-02.md`.

## 14. Dissertation refactoring decisions confirmed for planning on 2 September 2026

1. Refactor the current Ukrainian dissertation as the foundation; do not create a replacement
   manuscript from zero. Preserve the current four-chapter semantic order and most Chapters 1--2
   prose. New or materially rewritten passages are drafted in English as complete blocks.
2. Do not edit the live Path B worktree while the Article 4 controller owns it. Planning is stored
   on the isolated integration branch; manuscript editing begins from a clean post-campaign
   worktree and a hash-preserved fallback.
3. Chapters 3 and 4 retain their method/results roles. Chapter 3 receives the complete registered
   FRT/ALC/CEX/LVL/OVR/STS design and the new OpenSTL-native integrated architecture. Chapter 4
   makes the new hierarchy campaign primary and replaces the old fixed-region integration headline
   with measured component and dataset results.
4. Every main figure has one question. Both seed values are displayed for two-seed results; a
   training-seed confidence interval is not inferred from two points. Clip-level intervals are
   explicitly conditional on a trained model.
5. Main-text numbered tables contain empirical results. Configuration registries, tensor shapes,
   symbol dictionaries, hashes, and detailed uncertainty tables move to appendices.
6. Number only original/reused model definitions, implemented interfaces, primary estimands, and
   repeatedly referenced decision rules. Standard/tutorial derivations move inline or to the
   appendix. The target main-text range is roughly 46--60 numbered equations, not a fixed quota.
7. The E6 coverage--identity boundary and E5a non-replication remain visible. EBT, E8a, full legacy
   robustness, and the old integrated prototype become secondary/appendix evidence unless final
   results establish a specific need.
8. Unfinished results use traceable placeholders and conditional positive/neutral/negative prose.
   No invented numeric result, memory multiplier, or Pareto claim is permitted.
9. Articles 1--3 material removed from the main narrative is preserved in a publication-support
   appendix and provenance ledger rather than discarded.
10. The authoritative detailed plan is
    `docs/plans/DISSERTATION_DETAILED_REFACTORING_PLAN_2026-09-02.md`.

## 15. Independent-review corrections accepted on 2 September 2026

1. The v2 Article 4 matrix is a long-context/distant-evidence allocation study with a jointly
   predicted four-frame endpoint. It does not establish long autoregressive rollout stability.
2. The current 86-run campaign remains valid and is not restarted. Legacy E5/E6 remain substantive
   long-rollout evidence. A v2 selected-checkpoint rollout, if authorized, is a separately
   registered inference-only evaluation using existing trained checkpoints and a new sealed long
   TRB set; it does not retrain the matrix or reselect models.
3. The complete 30k matrix is registered development evidence. Sealed-holdout confirmation applies
   only to the exact selected roles/populations opened by the finalizer. Use “registered development
   frontier with sealed confirmation of selected contrasts,” not “confirmed frontier.”
4. Front matter, both abstracts, dissertation-level tasks, conclusions, current appendices,
   administrative acts, contribution records, licences, and permissions are explicit migration and
   claim-audit targets.
5. Preserve the conceptual substance of Articles 1--3. Static diffusion-consistency/control content
   belongs to Article 3/[A3], not Article 1.
6. Integrated evidence is claim-gated by actual cells. Four final full-model runs do not imply a
   completed six-arm component study or matched baseline campaign. Unmatched published values do
   not support Pareto claims.
7. Activation checkpointing is added as the practical same-model reversible comparator when
   feasible; otherwise the reversible claim is narrowed to ordinary autograd storage.
8. Static/density/hybrid are inference routing policies; mixed is a route-aware training regime.
   Same-checkpoint switching and separately trained regime comparisons answer different questions.
9. CEX, LVL, and OVR are described as registered compound interventions rather than pure isolated
   effects of reach, level count, or overlap.
10. Positional normalization, formula dependencies, result-bearing claim placeholders, provisional
    snapshot provenance, appendix migration, and bilingual/institutional cleanup are mandatory
    parts of the revised plan.
