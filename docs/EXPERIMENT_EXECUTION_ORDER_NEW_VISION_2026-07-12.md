# Comprehensive experiment execution order for the context-packaging vision

**Frozen:** 2026-07-12

**Authority:** the author retains scientific and GPU-launch authority. After the current-run
handoff, the author may grant Codex time-bounded execution autonomy within an explicitly approved
tranche. Delegation expires at the tranche stop condition or when the author withdraws it.

**Scientific identity:** context packaging is the manipulated design space. Reach, fidelity and
trust are explanatory axes. The approved article remains about when hierarchical/multirate context
pays under bounded budgets; reach–fidelity–trust structures the mechanism and discussion rather
than forcing a new title.

**Dataset roles:** loopmom is the primary causal memory instrument. Moving-MNIST is retained as a
standard boundary/generalization dataset, not replaced by loopmom. A future counterfactual Dynamic
LoopWorld tests genuine hidden-state necessity and is additional to both.

## 1. Operating rules

1. One launch owner at a time; the author remains the ultimate owner.
2. No launch before its design, estimand, cost definition, seeds and stop rule are recorded.
3. CPU/schema/oracle gates precede GPU training.
4. Existing evidence is frozen and hashed before trainer/evaluator refactoring.
5. Pre-registered evidence, post-hoc analysis and untouched confirmation remain distinct.
6. Dense frontier and B2 comparisons are target-interface matched. The target-interface confound
   applies specifically to strided-coarse arms such as legacy B1/E-320.
7. Phase C foreground-MSE and factor claims are separate: the registered foreground endpoint may
   remain primary; identity/trajectory claims require the v2 evaluator.
8. No experiment is interpreted through a result-dependent branch invented after launch.
9. A failed/null study is integrated into the claim ledger and manuscript; it is not silently
   replaced by another run.
10. Generated tables and figures come from artifact manifests, never hand-entered manuscript
    numbers.
11. GPU time is charged mechanically as wall-clock time while any project process owns the GPU,
    including pilots, evaluation, failed runs and retries. Every run records its authorized maximum
    and actual consumption; c2's planning estimate is replaced by measured time at handoff.
12. The programme has three envelopes: at most 181 GPU hours for closure, at most 150 for optional
    extension, and 50 protected reserve hours. The reserve cannot fund discretionary experiments.
13. Stop when the claims are resolved. Unused budget is not a reason to launch another study.

### 1.1 Canonical execution and status ledger

This document is also the single canonical live ledger for the programme. Do not create a second
execution-order document. Each four-hour audit appends a dated entry here containing expected
versus actual progress, metric/throughput/health trends, projected completion, cumulative GPU
hours, deviations and the resulting gate decision. Raw logs, W&B and immutable artifacts remain
the evidence; this section is the human-readable index. Status vocabulary is `NOT STARTED`,
`GATED`, `RUNNING`, `COMPLETE`, `DEFERRED`, `BLOCKED` or `CANCELLED`.

#### Programme status at 2026-07-12 23:58 Europe/Kyiv

| ID | Workstream | Expected before execution | Actual/status | Decision |
|---|---|---|---|---|
| E0 | c2-only optimizer-regime control | Finish without entering Phase C; replace stale runtime estimate | **COMPLETE**; 14.223 measured project GPU-hours; flat beats hierarchy by 0.00773, 95% CI [0.00304, 0.01233] in flat's direction | Retain as optimizer-regime robustness, not more-data/asymptotic evidence |
| E1 | Evidence freeze and governance repair | 8–16 CPU hours; no GPU | **COMPLETE**; evidence manifest/independent backup, result integration, canonical text/claim governance and Phase-C amendment restored | E2 preparation permitted |
| H3a/H4 | Confirmation evaluator, provenance, metrics and CPU gates | Exact checkpoint readers, immutable artifacts, complete metrics, clean CPU/hash gates | **COMPLETE**; 35 tests plus design/schema/consistency/compile gates; frozen dataset and checkpoint hashes pass | E2 authorized on commit `b9d91da` |
| E2a | Untouched LoopMom holdout | 1–2 CPU hours | **COMPLETE**; 500×200 set, seed 20260712, dataset ID `loopmom-confirmation_holdout-20260712-e033abee6e04`; no train/test glyph collisions | Dataset is now accessed and permanently frozen |
| E2b | Two primary confirmation pairs | 4–8 GPU-hours | **COMPLETE**; all 8 evaluations succeeded in 0.687 GPU-hours | Both registered reach-band effects confirm on both seeds with clip-bootstrap intervals excluding zero |
| E2c | Frozen frontier confirmation | 6–10 additional GPU-hours | **COMPLETE**; all 18 evaluations succeeded; 26/26 E2 runs complete | Simultaneous frontier confirms lower/interior hierarchy wins and upper flat inversion; no interpolation |
| E3a | Fixed-target diffusion B1 | 14–18 GPU-hours only if causal stride claim remains central | **GATED / NOT STARTED** | Reassess after E2 and manuscript claim freeze |
| E4a | Seed-2 head × package factorial | 7–10 GPU-hours after H3b numerical parity | **GATED / NOT STARTED** | Next planned training study unless E2 changes priority |
| E5a/b | Trust replication and dose response | 28–42 GPU-hours while trust remains central | **GATED / NOT STARTED** | Requires E2 final verdict and H3b |
| E7–E9 | Robustness, baseline, Phase C and optional extensions | Claim-triggered only | **NOT STARTED / DEFERRED as specified below** | No launch authority yet |

#### E2 run ledger

Foreground values below are provisional raw all-record means, not the registered inferential
analysis. `Δdev` is confirmation minus the matching development-pool mean. It is a drift/health
diagnostic, not a hypothesis test. The 3,072-token legacy files use their historical untagged names
and are reconciled in the final analysis.

| # | Arm | Seed | Updates | Status | Actual GPU h | Confirmation FG | Development FG | Δdev |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | hier-min | 0 | 30k | COMPLETE | 0.112 | 0.1625 | 0.1706 | -0.0082 |
| 2 | flat-min | 0 | 30k | COMPLETE | 0.102 | 0.2098 | 0.2106 | -0.0007 |
| 3 | hier-min | 1 | 30k | COMPLETE | 0.107 | 0.1666 | 0.1744 | -0.0078 |
| 4 | flat-min | 1 | 30k | COMPLETE | 0.102 | 0.2102 | 0.2102 | -0.0001 |
| 5 | hier-s2-min | 0 | 30k | COMPLETE | 0.065 | 0.1872 | 0.1893 | -0.0021 |
| 6 | flat-short | 0 | 30k | COMPLETE | 0.066 | 0.2212 | 0.2224 | -0.0011 |
| 7 | hier-s2-min | 1 | 30k | COMPLETE | 0.065 | 0.1821 | 0.1869 | -0.0048 |
| 8 | flat-short | 1 | 30k | COMPLETE | 0.066 | 0.2202 | 0.2206 | -0.0004 |
| 9 | hier-short | 0 | 30k | COMPLETE | 0.064 | 0.2250 | 0.2260 | -0.0010 |
| 10 | hier-short | 1 | 30k | COMPLETE | 0.063 | 0.2262 | 0.2257 | +0.0005 |
| 11 | hier-long | 0 | 30k | COMPLETE | 0.130 | 0.1825 | 0.1848 | -0.0023 |
| 12 | flat-budget | 0 | 30k | COMPLETE | 0.125 | 0.2020 | 0.2038 | -0.0018 |
| 13 | hier-long | 1 | 30k | COMPLETE | 0.130 | 0.1640 | 0.1690 | -0.0050 |
| 14 | flat-budget | 1 | 30k | COMPLETE | 0.125 | 0.2029 | 0.2044 | -0.0016 |
| 15 | hier-long | 2 | 30k | COMPLETE | 0.130 | 0.1725 | 0.1747 | -0.0022 |
| 16 | flat-budget | 2 | 30k | COMPLETE | 0.125 | 0.2032 | 0.2046 | -0.0014 |
| 17 | hier-c1 | 0 | 30k | COMPLETE | 0.190 | 0.1729 | 0.1799 | -0.0070 |
| 18 | flat-c1 | 0 | 30k | COMPLETE | 0.183 | 0.1813 | 0.1876 | -0.0063 |
| 19 | hier-c2 | 0 | 30k | COMPLETE | 0.250 | 0.1789 | 0.1820 | -0.0031 |
| 20 | flat-c2 | 0 | 30k | COMPLETE | 0.245 | 0.1639 | 0.1703 | -0.0065 |
| 21 | hier-c2 | 0 | 60k | COMPLETE | 0.252 | 0.1743 | 0.1785 | -0.0042 |
| 22 | flat-c2 | 0 | 60k | COMPLETE | 0.246 | 0.1620 | 0.1708 | -0.0087 |
| 23 | hier-rich | 0 | 60k | COMPLETE | 0.318 | 0.1627 | 0.1695 | -0.0068 |
| 24 | flat-long | 0 | 60k | COMPLETE | 0.313 | 0.1498 | 0.1564 | -0.0066 |
| 25 | hier-rich | 1 | 60k | COMPLETE | 0.330 | 0.1638 | 0.1687 | -0.0049 |
| 26 | flat-long | 1 | 60k | COMPLETE | 0.312 | 0.1504 | 0.1580 | -0.0076 |

#### Intermediate audit E2-I1 — 2026-07-12 22:03 Europe/Kyiv

- **Progress:** 24/26 runs complete; 3.575 charged GPU-hours. The active run reached 64/100 clips
  in 0.20 hours. Including its elapsed time, project GPU ownership is approximately 3.78 hours.
- **Throughput:** actual evaluation is substantially faster than the conservative 10–18-hour E2
  estimate. Projected E2 evaluation total is approximately 4.2–4.5 GPU-hours, followed by CPU
  aggregation/bootstrap and artifact upload. This projection is plausible if the final two arms
  retain observed throughput.
- **Health:** every completed run exited zero and remained below its per-run authorization. The
  active evaluator and controller are alive; W&B is online. At the audit instant the GPU was 100%
  utilized, 4,080/16,380 MiB allocated, 68°C and approximately 129 W.
- **Metric trends:** confirmation-to-development shifts are small (mostly modest improvements),
  finite and coherent. Provisional raw pair differences preserve the two primary hierarchy signs:
  `flat-min - hier-min` = +0.0473/+0.0436 and `flat-short - hier-s2-min` =
  +0.0340/+0.0381 for seeds 0/1. The provisional frontier also retains the expected inversion:
  hierarchy wins at 896, 1,152 and 1,792 tested-token regimes, while flat wins at 2,432 and the
  completed 3,072 seed-0 cell. These are health observations only until simultaneous registered
  intervals are computed.
- **GPU-memory decision:** do **not** change E2. It is checkpoint-only evaluation, not training;
  there is no learning rate. The GPU is compute-saturated despite low VRAM occupancy. Increasing
  rollout `chunk` would change the preregistered batching and the CRN seed schedule
  (`100000 + 1000 * chunk_index + rollout_step`) after 24 completed runs, breaking exact
  comparability. Future training tranches may benchmark larger batches/microbatches, but batch,
  update count, LR and EMA horizon must be treated as an optimizer regime and frozen before the
  first compared arm.
- **Decision:** continue E2 unchanged. No scientific or operational trigger currently justifies
  intervention.

At E2 completion, replace the two remaining row statuses, append the registered max-statistic
results and complete-metric summary, record exact total GPU hours, and issue the E3/E4/E5 gate
decision here before any new training launch.

#### Final audit E2-F — 2026-07-12 23:58 Europe/Kyiv

- **Operational verdict:** PASS. All 26 frozen evaluations completed with exit code zero. The
  successful-run ledger charges **4.2169 GPU-hours**, far below the 18-hour tranche maximum.
  Twenty-six W&B evaluation runs and the analysis run are cloud-finished with complete summaries.
  The result JSON and Markdown reproduced byte-for-byte in an independent CPU rerun.
- **Artifact verdict:** PASS. `artifacts/confirmation_e2/completion_validation.json` verifies 26
  unique content-addressed RunSpecs, checkpoint/config/data bindings, COMPLETE markers, every
  artifact hash, 15,200 aligned records per run, required finite metrics on their defined
  populations, exactly one successful access per RunSpec, 48 selected-pair rows and 12 frontier
  contrasts. Analysis SHA-256 is
  `85d01085e9ac074cd099e7b611101b30b7b7112b1f0230ea9163d9825cf2166e`.
- **Access/accounting caveat:** the access log contains one additional entry at 18:11:57 Kyiv from
  the aborted Windows/W&B launcher attempt (`3d64d75e2f780e3f`). It generated no metric artifact
  and informed no decision, so it does not contaminate the frozen confirmation. The earlier claim
  that the failed attempt never opened the holdout was too strong. Conservatively charge roughly
  0.01 additional process/GPU-hour, giving **approximately 4.23 total E2 project GPU-hours**.
- **Primary reach result:** `hier-min` versus `flat-min` in the frozen `14 < dt <= 48` band is
  +0.0980 [0.0788, 0.1179] at seed 0 and +0.0914 [0.0736, 0.1098] at seed 1. The whole-population
  continuity effects are +0.0474 [0.0354, 0.0595] and +0.0436 [0.0326, 0.0549].
- **Primary B1-configuration result:** `hier-s2-min` versus `flat-short` in the frozen
  `8 < dt <= 48` band is +0.0692 [0.0543, 0.0845] and +0.0743 [0.0588, 0.0899]. Whole-population
  effects are +0.0340 [0.0249, 0.0438] and +0.0381 [0.0285, 0.0482]. This confirms the
  configuration effect, not a pure context-stride intervention.
- **False-positive-sensitive corroboration:** both reach comparisons improve union/full-frame
  error, foreground recall/F1, missed-component count and component coverage on both seeds. The
  48-versus-14 comparison also improves precision and extra-component count on both seeds. For
  B1, precision and extra-component differences remain inconclusive, preventing an over-broad
  “every metric improves” claim.
- **Frontier family:** the 95% simultaneous max-statistic family (critical value 2.741) confirms
  hierarchy at 896 and 1,152 tokens on every tested seed; the 1,792 interval includes zero. Flat
  wins at 2,432 in both optimizer regimes and at 3,072 on both seeds. Near-bottom is flat-favouring
  at seed 1 and unresolved at seed 0. Therefore the defensible bracket is now: **1,152 is the
  largest tested budget with a family-wise-confirmed hierarchy win, while 2,432 is the smallest
  tested budget with a family-wise-confirmed flat win**; 1,792 is directionally hierarchical but
  not family-wise resolved. No continuous crossing is inferred.
- **Scientific decision:** R1/R2 no longer require untouched confirmation. The core reach effect,
  comparator-specific observability mechanism and reach–fidelity inversion survive confirmation.
  F2 remains configuration-level, so E3a stays conditional on retaining causal stride language.
  F3 remains central and unresolved under regression; **E4a fires next**, but only after the H3b
  numerical-parity/provenance gate. T1 remains title-relevant and E5a follows E4a. No E4b, E3a,
  Phase C or optional extension is authorized by E2 alone.

## 2. Current handoff boundary — E0

### E0 — Finish c2-only optimizer-regime control

**Final state at 2026-07-12 15:15 Kyiv:**

- seed-2 diffusion trio: complete (`S2TRIO_DONE`, 2026-07-11 23:22);
- original master queue: terminated by the watcher before c2/Phase C;
- `hier-c2` B8x60k: trained and evaluated;
- `flat-c2` B8x60k: trained and evaluated;
- Phase C: not launched.

**Question:** does the 2,432-token result change when the number of examples is approximately held
constant but batch size, optimizer updates, gradient noise and effective EMA horizon change?

**Interpretation:** optimizer-regime robustness, not “twice the data” and not asymptotic proof.

**Measured time:** **14.223 project GPU-hours**, including both arms, rollout and comparison. This
replaces the stale 4.5-hour planning allowance and reduces later conditional Pool-A capacity.

**Handoff gate:** c2 completion marker; old controller absent; no project-scoped Phase C or other
project GPU process; project launch permission inactive; queue lock absent; logs/checkpoints/NPZs
enumerated; measured GPU time and result-generation worktree recorded. Unrelated GPU activity does
not block the handoff.

## 3. Evidence and implementation freeze — E1

### E1 — Result integration and governance repair

This is not a GPU experiment, but no later study should begin before it is complete.

Tasks:

1. Add measured S1-a, S1-b, regression-twin, seed-2 and c2 blocks to
   `docs/SPATIAL_FACTORIAL.md`.
2. Replace stale `EXPECTED-B1/B2/REG/S2/C2` markers in CH1–CH8, ROZDIL4 and VYSNOVKY.
3. Restore L1–L10 ledger continuity or add an explicit alias table to the refined claim rows.
4. Restore CH1–CH8 as canonical; use reach/fidelity/trust as direct discussion surgery.
5. Restore the Phase C preregistration trail as a dated amendment, not a deletion.
6. Hash and back up all raw evidence before changing the legacy trainer/evaluator.
7. Correct the confirmation launcher: central confirmation must use a fresh loopmom set, not
   Moving-MNIST.

**Time:** **8–16 CPU hours**.

**Status:** **COMPLETE.** The evidence manifest and independent backup, result integration,
canonical text/claim governance, Phase-C amendment, frozen confirmation design and fail-closed
launcher are recorded in the post-c2 commits and handoff dossier.

## 4. Untouched central confirmation — E2

### E2a — Generate the untouched loopmom confirmation set

**Design:** identical generator family and operating point to the primary loopmom campaign; new
recorded seed; new clips/actions/positions; training latent statistics retained; immutable SHA-256
manifest; dataset rejected by training/monitoring loaders.

**Question:** do the registered package effects and post-hoc reach-band mechanism survive a set
that did not influence arm design, qualitative monitoring or theory development?

**Time:** **1–2 CPU hours** after the generator/manifest path is corrected.

### E2b — Primary confirmation pairs

Evaluate without retraining:

1. `hier-min` versus `flat-min`, seeds 0 and 1;
2. `hier-s2-min` versus `flat-short`, seeds 0 and 1;
3. legacy registered non-empty foreground error as continuity primary;
4. geometry-derived reach bands (`14 < dt <= 48`, `8 < dt <= 48`) as frozen mechanistic
   primaries;
5. full-frame, background, union-mask, foreground precision/recall, component misses/extras and
   coverage as complete corroborating endpoints.

**GPU time:** **4–8 hours**.

### E2c — Full-frontier confirmation sweep

Evaluate all available matched-budget partners/checkpoints on the same holdout:

- bottom/near-bottom package pair;
- 896-token lower regime;
- 1,152-token interior;
- 1,792-token c1;
- 2,432-token c2 under both available optimizer regimes;
- 3,072-token saturated-window regime.

No interpolation is fitted unless a separately preregistered model is justified. Report the largest
tested spatial-package win and smallest tested flat-package win by regime.

The selected pairs and their reach bands are the confirmatory primary hypotheses. The frozen
frontier is a family-level secondary confirmation using a simultaneous max-statistic clip-cluster
bootstrap because its contrasts share clips. A predeclared Holm procedure is an acceptable simpler
fallback. Complete false-positive-sensitive metrics are corroborating endpoints. c2 optimizer
regimes are reported separately, and no post-result cell may be promoted to a new primary.

**Additional GPU time:** **6–10 hours**.

**E2 total:** **10–18 GPU hours + 1–2 CPU hours**.

## 5. B1 causal isolation — E3

### E3a — Fixed-target-interface B1 under diffusion

**Arms:**

- `hier-s2-fixed-target`: 24 coarse context slots at stride 2 + 2 fine; 512 context tokens; four
  per-frame coarse targets + four fine targets;
- `flat-short`: 8 fine context frames; 512 context tokens; the same four-coarse/four-fine target
  interface.

Existing `flat-short` seed-0/1 checkpoints can be reused if the compatibility audit proves exact
identity. Train the fixed-target hierarchical arm at seeds 0/1/2 and the missing flat seed-2
partner.

**Question:** how much of the legacy B1 effect remains when context stride changes but target slots,
target loss, target tokens and rollout target interface do not?

**GPU time:** **14–18 hours**.

### E3b — Fixed-target B1 under regression

Repeat the same controlled pair under the regression head for seeds 0/1/2, reusing compatible
existing flat checkpoints and training missing partners.

**Question:** is the controlled context-reach effect head-stable, or does the target/prediction head
change the value of the spatial package?

**GPU time:** **14–20 hours**.

**Portfolio status:** deferred outside the current ≤400-hour programme. E3a is launched only if the
assembled article retains a causal stride/temporal-rate statement; otherwise legacy B1 remains a
whole-configuration result. E3b belongs to a later head-stability study.

## 6. Head × package interaction — E4

### E4a — Complete the seed-2 four-cell factorial

Factorial:

- package: spatially coarse dense history versus temporally sparse full-resolution history;
- head: diffusion versus regression;
- seed: 0, 1, 2.

Seed-2 diffusion cells already exist in the completed trio. Train/evaluate the two missing seed-2
regression cells.

**Question:** does regression consistently shift the package contrast toward spatial packaging,
despite seed-sensitive signs within regression?

**GPU time:** **7–10 hours**.

### E4b — Heterogeneity expansion to seeds 3–4

Run all four factorial cells at seeds 3 and 4 only under a prewritten **training-seed** rule. The
trigger must use seed-level heterogeneity or a defined across-seed precision target. Clip-cluster
bootstrap intervals quantify evaluation-set uncertainty conditional on trained models and cannot
trigger additional training seeds. Desired sign or an "interesting" result is never a trigger.

**GPU time if triggered:** **28–36 hours**.

**Budget rule:** E4b has no standing allocation. If triggered, its authorized maximum must replace
or reduce E6, E8c or E9; it cannot consume protected reserve.

## 7. Addressable versus trustworthy memory — E5

### E5a — Replicate the long-loop decisive pair

Train/evaluate the long-loop spatial-memory arm and its matched comparison at seeds 1 and 2 using
the frozen H-grid3 protocol.

**Question:** does the single-step retrieval success followed by self-fed inversion replicate
beyond seed 0?

**GPU time:** **20–28 hours**.

### E5b — Horizon dose-response and diagnostic interventions

Using the replicated checkpoints, evaluate horizons approximately 24/48/96 and longer available
horizons under:

1. teacher-forced/addressable context;
2. pure self-fed rollout;
3. periodic ground-truth coarse-memory refresh;
4. controlled coarse-latent corruption;
5. multiple sampler seeds;
6. matched per-step and matched total-NFE views.

**Question:** where does stored information cease to be reliable, and is failure caused primarily
by memory-state corruption, decoding or stochastic sampling?

**GPU time:** **8–14 hours**.

### E5c — Rollout-aware/Self-Forcing-class intervention

Train a scoped rollout-aware version only after E5a/E5b identify a reproducible trust boundary.
Compare against the same checkpoints and horizons.

**Question:** can training on self-generated context move the trust boundary without sacrificing
short-horizon fidelity?

**GPU time:** **24–45 hours**.

**Portfolio status:** E5c is deferred. The current programme measures and diagnoses the trust
boundary; it does not expand into a new rollout-aware training paper. E5a runs before E5b. If the
boundary does not replicate, shorten the diagnostic programme and demote the empirical trust claim.

**Current E5 allocation:** **28–42 GPU hours** for E5a/E5b.

## 8. Long Moving-MNIST boundary study — E6

### E6 — Phase C hybrid, not legacy factor interpretation

Moving-MNIST is retained as a standard long-rollout boundary dataset. It is not replaced by
loopmom and does not establish the same off-screen geometric mechanism.

Required preparation:

- official MNIST training glyphs for training clips and official test glyphs for evaluation;
- persistent object IDs, glyph classes, boxes, trajectories and overlap metadata;
- deterministic distinct split seeds and collision audit;
- validated classifier on clean, translated, rendered-component, overlap, AE-reconstruction and
  generated/degraded domains;
- Hungarian assignment wired into rollout records;
- dated `SPATIAL_FACTORIAL` amendment.

Primary:

- preregistered paired foreground-MSE effect/equivalence decision at the recorded ±0.005 margin,
  described narrowly as protocol-defined foreground equivalence rather than broad memory
  equivalence.

Co-primary/secondary:

- full/background/union-mask errors;
- foreground precision/recall;
- missed/extra objects and coverage;
- intention-to-score trajectory error;
- identity accuracy with persistent labels.

**Question:** on a familiar long-rollout benchmark without loopmom's explicit re-entry geometry,
do flat and hierarchical packages differ in foreground fidelity, identity, trajectory or survival?

**CPU preparation:** **1–3 days**.

**GPU time:** six arm×seed training runs plus long evaluation, approximately **30–40 hours**.

**Launch gate:** prepare and CPU-validate E6, but authorize training only if the evidence-frozen
article shows that this recognizable cross-dataset boundary materially improves the argument.

## 9. Representation and sampler robustness — E7

### E7a — Alternate autoencoder seed

Train a second AE seed and rerun the minimal central package set:

- 896-token lower regime;
- 1,152-token interior;
- one high-budget fidelity regime;
- spatial versus temporal package comparison.

Use two predictor seeds where feasible.

**Question:** is the frontier specific to one learned codec/latent geometry?

**GPU time:** **24–36 hours**.

### E7b — Simple codec controls

Add deterministic spatial pooling/downsampling and, if feasible, an alternate latent dimensionality.
Report reconstruction floors before predictor comparisons.

**Question:** does the effect require the learned AE, or follow the rate–resolution information
trade-off under simpler codecs?

**GPU time:** **12–24 hours**.

**Portfolio status:** deferred in favor of the stronger alternate-AE and cheaper sampler/NFE
controls.

### E7c — Sampler/NFE robustness

Evaluate central diffusion arms across sampler seeds and a small NFE grid, reporting both per-step
and total inference allocation.

**GPU time:** **4–8 hours**.

**Current E7 allocation:** **28–44 GPU hours** for E7a/E7c.

## 10. Retrieval and external architecture baselines — E8

### E8a — Classical pose-indexed retrieval/panorama baseline

Use known actions/positions to maintain a world-coordinate mosaic or retrieve the most relevant
past crop. Evaluate with the same re-entry records and complete metrics.

**Question:** how much of the gain requires learned hierarchical generation rather than explicit
access to the relevant past?

**Time:** **2–5 CPU days**, **0–4 GPU hours** for decoder/evaluation if needed.

### E8b — Learned retrieval/compression baseline

Implement a compact query/retrieval or memory-compression baseline with the same stored-token and
inference budget where technically possible.

**Question:** is multistream packaging competitive with modern learned retrieval at the same memory
budget?

**Time:** **3–7 development days + 20–40 GPU hours**.

**Portfolio status:** deferred. Classical retrieval and a tightly audited CW-VAE bridge cover the
present baseline question without opening a separate learned-retrieval research programme.

### E8c — CW-VAE-style baseline/protocol bridge

Mandatory output regardless of training: an architecture/protocol comparison table with CW-VAE,
DLH, TECO, WorldPack, CaR/SceneFly, MemLearner, GIM-World, Self Forcing and this work.

Direct numerical comparison is allowed only after matching dataset, resolution, context, horizon
and metric. Attempt either:

1. reproduce a CW-VAE-style model on the long Moving-MNIST protocol; or
2. adapt a clockwork latent baseline to loopmom and disclose every architecture difference.

**Question:** does the controlled context-package result persist against a recognized clockwork
hierarchy rather than only the project's flat controls?

**Time:** **3–7 development days + 25–50 GPU hours**.

**Launch gate:** first complete a protocol-equivalence audit covering data, decoder, context and
target budgets, training horizon, rollout, NFE and metrics. If a defensible bridge cannot be built
within the 50-hour maximum, retain the transparent comparison table and do not train an ambiguous
baseline.

**Current E8 allocation:** **25–54 GPU hours** for E8a/E8c.

## 11. Counterfactual Dynamic LoopWorld — E9

### E9a — Generator and observability instrument

Create paired worlds that are identical over the short visible context but differ in hidden state
that determines re-entry. Cross:

- static versus dynamically evolving off-screen state;
- low versus high spatial-detail identity;
- short versus long occlusion/re-entry delay.

Use instance textures, barcodes or state variables that shared MNIST priors cannot guess. Save
persistent metadata and exact counterfactual pair IDs.

**Question:** is memory genuinely necessary, and does the preferred package follow the temporal and
spatial structure of the hidden information?

**CPU development:** **4–10 days**.

### E9b — Minimal package factorial

Compare flat contiguous, spatially coarse and temporally sparse full-resolution packages across the
static/dynamic × low/high-detail cells, initially under diffusion and at two seeds. Add regression
only after the primary factorial is complete.

**GPU time:** **60–100 hours** for the primary factorial. The head extension is deferred.

### E9c — Retrieval baselines on counterfactual worlds

Evaluate classical and learned retrieval baselines from E8 on the same paired worlds.

**GPU time:** **10–25 hours** depending on reuse.

**Portfolio status:** E9c is deferred. E9 is an extension phase after the current article is
internally frozen, not a dissertation-closure prerequisite. It requires a validated CPU
counterfactual oracle, one timing/variance pilot and a frozen minimal factorial. The absolute study
cap is **100 GPU hours**. If the causal question cannot be answered cleanly inside that cap, defer
the study rather than shrink it into an ambiguous experiment.

## 12. Natural/3-D re-entry extension — E10

Use a rendered or natural environment with camera pose, occlusion and object re-entry. Replace
exact pixel observability with geometry-derived or annotated visibility while retaining paired
package budgets and self-fed evaluation.

**Question:** do the synthetic reach/fidelity/trust relationships transfer when geometry,
appearance and hidden dynamics are realistic?

This is a separate flagship study, not required for the existing dissertation/article claim and
deferred outside the current ≤400-hour programme.

**Estimated time:** **2–5 development weeks + 80–160 GPU hours** for a first controlled study.

## 13. Execution order summary

This is a decision tree, not a promise to exhaust the portfolio. H3a precedes evaluation-only E2;
the H3b trainer/provenance integration and its numerical-parity gate precede every new training run.

| Order | ID | Experiment/workstream | Portfolio status | GPU maximum | CPU/development estimate |
|---:|---|---|---|---:|---:|
| 0 | E0 | Finish c2-only control | already running; replace estimate with actual | 4.5 h planning allowance | — |
| 1 | E1 | Evidence freeze, swap pass, governance repair | required | 0 | 8–16 h |
| 2 | E2 | Untouched loopmom confirmation, selected + full frontier | required | 10–18 h | 1–2 h plus evaluator integration |
| 3 | E4a | Complete seed-2 head × package factorial | closure; run after H3b | 7–10 h | 2–4 h |
| 4 | E3a | Fixed-target diffusion B1 | claim-gated causal control | 14–18 h | 1–2 d integration/QA |
| 5 | E5a/b | Trust replication, then dose response | closure while trust is central | 28–42 h | 2–5 d |
| 6 | E7a/c | Alternate-AE and sampler/NFE robustness | closure robustness | 28–44 h | 2–5 d |
| 7 | E8a | Classical pose-indexed retrieval | closure baseline | 0–4 h | 2–5 d |
| 8 | E6 | Phase C hybrid on long Moving-MNIST | article-utility gated | 30–40 h | 1–3 d |
| 9 | E8c | CW-VAE protocol bridge | equivalence-audit gated extension | 25–50 h | 3–7 d |
| 10 | E9 | Counterfactual Dynamic LoopWorld primary factorial | post-freeze extension | 60–100 h | 1–2 wk |
| — | E4b | Seeds 3–4 factorial expansion | replacement-only seed trigger | 28–36 h | 2–4 h |
| — | E3b/E5c/E7b/E8b/E9c/E10 | Additional methods/extensions | deferred outside portfolio | — | future work |

## 14. Totals

All totals assume one GPU and sequential project ownership. CPU/document work can overlap GPU
execution. GPU-hour consumption is wall-clock project ownership, not successful-training time.

### Pool A — committed/claim-gated closure

The closure envelope contains E0, E2, E4a, claim-gated E3a, E5a/E5b while trust remains central,
E7a/E7c, E8a and article-utility-gated E6.

- **maximum allocation:** approximately **181 GPU hours**;
- **without E6:** approximately **141 GPU hours**;
- E6 preparation does not imply launch;
- actual c2 consumption replaces its 4.5-hour planning allowance at handoff; any excess reduces a
  later conditional Pool-A allocation rather than increasing the 181-hour envelope.

### Pool B — optional extension

The extension envelope contains only a protocol-audited E8c and the post-freeze E9 primary
factorial.

- **maximum allocation:** **150 GPU hours**;
- E8c: at most 50 hours;
- E9: at most 100 hours;
- unused closure budget does not automatically authorize extension work.

### Pool C — protected reserve

- **allocation:** **50 GPU hours**;
- covers pilots, failed runs, retries and estimation error;
- cannot fund discretionary experiments or broaden a completed study.

### Hard ceiling and replacement rule

The expected maximum is approximately **381 GPU hours**, leaving about 19 hours below the absolute
400-hour ceiling. Plan toward roughly **330 hours of scientific runs**, not 400 hours of spending.
E4b has no additive allocation: if its preregistered training-seed trigger fires, it replaces or
reduces E6, E8c or E9. E3b, E5c, E7b, E8b, E9c and E10 are outside this portfolio.

Before each tranche, replace estimates with measured probes and a run-count ledger. A tranche is
not authorized unless its maximum fits the remaining non-reserve envelope. Stop when the claims are
resolved; unused GPU hours indicate successful governance, not incomplete work.

## 15. Expected article/dissertation evidence products

1. **Configuration/cost table:** levels, grids, strides, context slots/reach, context/target/total
   tokens, parameters, examples, updates, NFE, runtime and VRAM.
2. **Reach–fidelity frontier table:** budget, reaches, per-seed errors, effects/CIs and tested
   winner by optimizer regime.
3. **Shape/head factorial table:** flat/spatial/temporal/depth packages × diffusion/regression ×
   seeds.
4. **Reach-differential table:** comparator reaches, exact band, per-seed effect and confirmation
   status.
5. **Trust table/figure:** teacher-forced versus self-fed error by horizon, refresh and corruption.
6. **Complete-metric table:** foreground, background, union, full-frame, precision/recall,
   misses/extras and coverage.
7. **CW-VAE/prior-art protocol table:** architecture, levels, clocks, memory mechanism, context,
   horizon, self-feeding, dataset, metrics, budget matching and reproduction status.
8. **Qualitative evidence:** frozen-selection re-entry strips, failure cases and overlap/object
   examples.
9. **Artifact map:** every manuscript number linked to immutable RunSpec/data/AE/result hashes.

## 16. Handoff instruction

When E0 finishes, the author should provide:

- final c2 log and comparison;
- new/modified artifact paths;
- current branch/commit/worktree;
- project process/GPU/lock state and measured c2 GPU-hour consumption;
- resume/retry history and W&B caveats;
- explicit time-bounded authorization for Codex to execute the next approved tranche under this
  plan while the author retains scientific and launch authority.

Codex should then begin at E1, not by launching E2 immediately. The evidence swap and immutable
backup come first.
