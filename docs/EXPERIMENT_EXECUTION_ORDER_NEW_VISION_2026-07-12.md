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
| H3b | Safe new-training baseline and numerical parity | Same scientific kernel plus RunSpec, RNG-complete explicit resume, immutable artifacts and cloud identity | **COMPLETE**; 38 tests, exact first-update/batch/model/optimizer/EMA parity and uninterrupted-versus-resumed trajectory; report bound to `c0ab78c` | New training may use only the provenance-v2 wrapper |
| E2a | Untouched LoopMom holdout | 1–2 CPU hours | **COMPLETE**; 500×200 set, seed 20260712, dataset ID `loopmom-confirmation_holdout-20260712-e033abee6e04`; no train/test glyph collisions | Dataset is now accessed and permanently frozen |
| E2b | Two primary confirmation pairs | 4–8 GPU-hours | **COMPLETE**; all 8 evaluations succeeded in 0.687 GPU-hours | Both registered reach-band effects confirm on both seeds with clip-bootstrap intervals excluding zero |
| E2c | Frozen frontier confirmation | 6–10 additional GPU-hours | **COMPLETE**; all 18 evaluations succeeded; 26/26 E2 runs complete | Simultaneous frontier confirms lower/interior hierarchy wins and upper flat inversion; no interpolation |
| E3a | Fixed-target diffusion B1 | 14–18 GPU-hours only if causal stride claim remains central | **GATED / NOT STARTED** | Reassess after E2 and manuscript claim freeze |
| E4a | Seed-2 head × package factorial | 7–10 GPU-hours after H3b numerical parity | **COMPLETE / PASS**; two cells and analysis consumed 6.1228 project GPU-hours; completion validator PASS | Three-seed head shift is consistently negative; regression package effects remain heterogeneous |
| E4b | Heterogeneity expansion, seeds 3–4 | 28–36 GPU-hours, replacement-only | **RUNNING** from source commit `fa4906f`; first cell `hier-long` seed 3, RunSpec `e6ff5f95708c15ce` | Eight cells sequential; four-hour deep audits; no regime changes or automatic retry |
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

#### Intermediate audit E4-I0 — 2026-07-13 00:20 Europe/Kyiv

- **Launch gates:** PASS on clean commit `ab88a4f`; H3b parity, frozen E4 protocol, frozen
  data/AE hashes, two-cell design, 10 allocated/11 authorized GPU-hours and online W&B write all
  passed before launch.
- **Active cell:** spatially coarse dense regression package, `reg-long`, seed 2, B16×30k,
  RunSpec `c1eefc99257e1a3a`; W&B run
  `e4-head-package-seed2-c1eefc99257e1a3a`.
- **First trend:** updates 1–500 average loss 0.7273 at 0.40 s/update and 6.46 GB measured peak.
  This is bracketed by the legacy seed-0/1 first intervals (0.7263/0.7280), while throughput and
  peak memory exactly match the legacy regime (0.38–0.40 s/update; 6.46 GB). No parity drift is
  visible.
- **GPU:** 99–100% compute utilization, approximately 8.19/16.38 GB board memory, 64°C and 113 W.
  Compute is saturated; forcing 14 GB through a larger batch would change the frozen factorial's
  optimizer regime and is prohibited.
- **Projection:** approximately 3.3–3.5 hours for this training plus evaluation/upload, within the
  5-hour cell maximum. The temporal regression cell historically runs at about 0.35 s/update and
  has the same 5-hour cap. Continue unchanged unless loss departs materially from the legacy
  envelope, the log stalls, W&B fails, or a per-cell/budget limit is reached.

#### Intermediate audit E4-I1 — 2026-07-13 02:05 Europe/Kyiv

- **Progress/throughput:** `reg-long` seed 2 reached update 17,000/30,000 (56.7%) in approximately
  1.82 wall-clock hours including data setup. Update time is stable at 0.37–0.40 s with no WDDM
  spill. At the observed 0.378 s/update, the remaining training is approximately 1.37 hours;
  evaluation and cloud artifact upload should keep the cell near 3.3–3.5 total hours and below
  its 5-hour maximum.
- **Optimization trend:** the warm-up/convergence shape matches the legacy regime: loss
  0.7273/0.7001/0.5371/0.1804 at 500/1,000/1,500/2,000, bracketed closely by legacy seeds at the
  decisive warm-up transition. It falls to 0.0317 at 6,000 and 0.0176 at 17,000. Across the 34
  matched 500-step intervals it averages 0.0023 below the legacy seed-0/1 mean; the maximum
  absolute deviation is 0.0111 and is consistent with training-seed variation rather than a
  regime shift. There are no non-finite values or loss explosions.
- **Validation:** fixed-seed h4 foreground MSE progresses 0.2627 → 0.0534 → 0.0421 → 0.0400 →
  0.0393 → 0.0349 → 0.0313 → 0.0299 at updates 2k…16k. The legacy seed-1 values at 14k/16k are
  0.0340/0.0344, so seed 2 is converging slightly better, not degrading. Training loss cannot by
  itself predict the eventual 152-frame package contrast; the registered rollout remains
  necessary.
- **Checkpoint/provenance:** the atomic v2 checkpoint is present at update 17,000 (355,264,107
  bytes), format 2, RunSpec-bound, with model/optimizer/EMA, Python/NumPy/Torch CPU+CUDA RNG,
  latent statistics and sampler state (`examples_seen=272000`, `next_update=17001`). Eight
  immutable diagnostic grids exist through 16k.
- **Cloud health:** W&B state is `running`; 33 history rows, eight validation images and current
  summaries are uploaded. The initial empty `scan_history` response was API query lag/selection,
  not a logging failure; local filestream records continuous successful history uploads and no
  errors.
- **GPU/process health:** controller and trainer trees are alive. GPU utilization is 100%, board
  memory approximately 8.27/16.38 GB, temperature 63°C and draw 113 W. Low unused VRAM is not a
  throughput fault because compute is saturated, and the frozen B16×30k factorial forbids a batch
  change.
- **Decision:** continue unchanged. No stop, resume, redesign or budget trigger fires. The likely
  operational outcome is successful completion of cell 1 followed by cell 2; the scientific
  interaction sign remains unknowable until both seed-2 rollout artifacts are paired.

#### Intermediate audit E4-I2 — 2026-07-13 06:05 Europe/Kyiv

- **Cell 1 complete:** `reg-long` seed 2 finished training, evaluation and cloud artifact upload
  with exit code zero in **3.173 GPU-hours**, below its 5-hour maximum. Final loss is 0.0130,
  fixed-grid h4 foreground MSE 0.0268, and 152-frame rollout foreground MSE 0.1528. The rollout
  lies between legacy seed 0 (0.1682) and seed 1 (0.1503), so it is plausible training-seed
  variation rather than a new regime.
- **Cell 1 integrity:** RunSpec `c1eefc99257e1a3a` is COMPLETE; its manifest contains 19
  artifacts and every byte count/hash verifies. The rollout has 15,200 finite foreground records.
  W&B is `finished` with 60 training intervals, 15 validation grids, rollout summary/media and
  the checkpoint/result artifact.
- **Cell 2 progress:** `strided-flat-reg` seed 2 reached 27,000/30,000 at 0.35 s/update, 6.26 GB
  measured peak and RunSpec `cf50f37223afbf33`. Its atomic checkpoint is at 27,000 with complete
  CPU/CUDA RNG and sampler state (`examples_seen=432000`, `next_update=27001`); 13 grids exist
  through 26k. Controller, trainer and W&B are alive, with no spill, stall or non-finite loss.
- **Cell 2 trend versus legacy:** the early warm-up follows the same shape, and loss declines
  smoothly to 0.0190 at 27k. It is about 0.004 above the two legacy seeds late in training.
  More importantly, fixed-grid h4 error is 0.0377 at 26k versus 0.0212/0.0210 for legacy seeds.
  The qualitative grid remains recognizable and aligned, but the quantitative plateau is a real
  seed-2 warning, not something to average away. H3b exact parity and the healthy v2 spatial cell
  make a wrapper/regime fault unlikely; material training-seed heterogeneity is the leading
  explanation. The last 3k updates may still improve it, and only the registered rollout can
  determine the package effect.
- **Cloud/GPU/budget:** both W&B streams are continuous; the active run has 53 uploaded training
  intervals and 13 validation images. GPU is 99–100% utilized, approximately 8.08/16.38 GB board
  memory, 63°C and 115 W. Cell 2 has used approximately 2.65 hours and should finish training in
  about 0.29 hour plus evaluation/upload. Projected tranche consumption is approximately
  6.1–6.4 hours, safely below 11.
- **Scientific implication:** the current trend raises the probability that regression seed 2
  will favor the spatial package, but no result is inferred from training/short-grid metrics.
  If the final rollout is materially worse for the temporal cell, it will strengthen the
  preregistered across-training-seed heterogeneity trigger rather than invalidate the run.
- **Decision:** continue unchanged to the frozen endpoint. Do not tune, extend, restart or select a
  checkpoint. Final acceptance requires the second immutable artifact, paired seed-2 analysis and
  the preregistered E4b trigger calculation.

#### Final audit E4-F — 2026-07-13 09:28 Europe/Kyiv

- **Training and budget:** both frozen B16×30k cells completed without a retry, checkpoint
  selection, extension or optimizer change. `reg-long` RunSpec `c1eefc99257e1a3a` used 3.1733
  project GPU-hours and produced ALL foreground MSE 0.1528. `strided-flat-reg` RunSpec
  `cf50f37223afbf33` used 2.9495 hours and produced 0.1736. The E4a total is **6.1228 GPU-hours**
  against the 11-hour authorized maximum. Project GPU ownership ended after the second upload;
  no subsequent experiment launched.
- **Training health:** the second cell's late fixed-grid warning correctly anticipated a weaker
  seed-2 temporal-regression model, but its training remained finite, smooth and operationally
  healthy. Final loss was 0.0173, fixed-grid h4 foreground MSE 0.0401, stable throughput 0.35
  s/update and 6.26 GB measured peak. This is scientific training-seed variation, not a spill,
  stall or wrapper failure: H3b parity was exact, the spatial v2 cell matched the legacy envelope,
  and both qualitative rollouts remained coherent.
- **Artifact and cloud integrity:** each v2 run has a COMPLETE marker, 19-item immutable manifest,
  15,200 finite paired records and verified byte counts/SHA-256 hashes. Both W&B training runs and
  analysis run `e4-analysis-284bda366cf7` are cloud-state `finished`, with final summaries and
  artifacts present.
- **Analysis compatibility incident:** the launcher's first post-training analysis stopped before
  writing results because the legacy hierarchical NPZs predate arm-independent `dt/seen` fields.
  This did not affect training or metric artifacts. The adapter now requires and verifies the
  actual paired estimand geometry (`clip`, `h`, `hard`, `fg`) and cross-checks `dt/seen` wherever
  present. Three focused regression tests and the full CPU suite (**41 passed**) validate the
  repair. The analyzer was then rerun once; no model evaluation or holdout access was repeated.
- **Independent completion gate:** `scripts/validate_e4_completion.py` independently verifies both
  RunSpec identities, frozen hyperparameters, COMPLETE markers, every manifest artifact, GPU
  ledger limits, paired geometry, all observed effects/bootstrap intervals and the E4b rule.
  Result: `E4_COMPLETION_PASS` in
  `artifacts/e4_head_package/completion_validation.json`.
- **Registered results:** positive package effects favor temporally sparse full-resolution
  history; negative interactions mean regression shifts toward spatial coarsening.

  | Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
  |---:|---:|---:|---:|
  | 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | −0.0089 [−0.0151, −0.0025] |
  | 1 | +0.0046 [+0.0007, +0.0084] | −0.0104 [−0.0146, −0.0062] | −0.0150 [−0.0199, −0.0104] |
  | 2 | +0.0111 [+0.0070, +0.0151] | −0.0208 [−0.0270, −0.0150] | −0.0319 [−0.0401, −0.0245] |

- **Scientific verdict:** E4a resolves the three tested seeds in favor of a consistent
  head-induced shift: every interaction is negative and every within-seed conditional interval
  excludes zero. It does **not** establish that regression always favors spatial packaging;
  regression's package sign varies by seed. Report the two statements together. All factorial
  cells use the existing adaptively used evaluation pool, so E4a remains development evidence and
  is not a second untouched confirmation.
- **Preregistered next gate:** E4b fires for both registered reasons—regression effects include a
  value ≥ +0.005 and values ≤ −0.005, and their sample SD is 0.0161 > 0.010. E4b remains
  replacement-only and is **not automatically authorized**. Before another launch, choose which
  of E6/E8c/E9 it replaces, freeze its eight-cell RunSpec ledger and confirm the revised
  sub-400-hour envelope. T1 remains article/title-relevant, so E5a remains the next independent
  closure study if E4b is not taken immediately.

Evidence: `docs/confirmation/E4_RESULTS.md`, `artifacts/e4_head_package/results.json`,
`artifacts/e4_head_package/completion_validation.json`, and W&B group
`e4-head-package-seed2`.

#### E4b preparation decision — 2026-07-13 09:45 Europe/Kyiv

- **Authority and replacement:** the author approved the preregistered recommendation to proceed.
  E4b is replacement-only: its **36-hour maximum replaces E8c's 50-hour GPU allocation**. The
  architecture/protocol comparison table remains mandatory CPU-only work. Protected reserve is
  unavailable to E4b.
- **Frozen design:** all four head × package cells run at seeds 3 and 4 under the unchanged
  B16×30k, LR 3e-4, warm-up 2k, EMA 0.999, bf16, NFE-16, 100-clip/152-frame development-pool
  regime. There are eight sequential cells with 4.5 hours maximum each. Batch enlargement to fill
  VRAM remains prohibited because it would change the factorial's optimizer regime.
- **Training-seed analysis:** report every conditional clip-bootstrap interval, then treat the five
  training seeds as units for mean, sample SD and a two-sided Student-t interval (df=4). A robust
  head shift requires all five interaction estimates negative and the seed-level interval below
  zero. Regression package stability requires the same material sign on all five seeds; otherwise
  report sign counts and heterogeneity.
- **Terminal rule:** E4b ends after seeds 3–4 for every result. It cannot authorize seeds 5+.
- **Authorization gate:** preparation commit `f8de000` passes the full CPU suite (45 tests), Python
  compilation, PowerShell parsing, exact H3b scientific-source hashes, design/data/AE/trigger
  validation and the 36-hour budget/replacement check. The frozen config hash is
  `91546ce4499c580dffe7340dd8a1401c89254c3cb88afd324d44a3faeb72610d`; hash-bound protocol and
  design attestations are `gates/e4b_protocol_frozen.json` and
  `gates/e4b_design_validated.json`. E4b is authorized. A clean worktree, no competing project
  trainer, a repeated design check and a live W&B write remain mandatory immediately before GPU
  launch.

#### E4b launch gate E4b-I0 — 2026-07-13 09:52 Europe/Kyiv

- Authorization ledger: `head-package-seeds3-4 = AUTHORIZED`; all three required hash-bound
  attestations pass.
- Repeated design/data/parity/budget validation: PASS with zero existing E4b RunSpecs and exact
  36/36 replacement-hour allocation.
- W&B online write: PASS at
  `https://wandb.ai/leo27heady/hgrid/runs/e4b-gate-54814fdf-20260713095143`, using `_netrc`
  after removing only the inherited process-level API key.
- CPU/software state: 45 tests pass; Python compilation and PowerShell parser pass; worktree is
  otherwise clean. No project Python trainer is active and no competing project controller was
  detected.
- Frozen launch order: complete all four seed-3 cells, then all four seed-4 cells; each cell has a
  4.5-hour cap and the tranche has a 36-hour cap. No automatic retry, tuning, extension or seed-5
  branch is allowed.

#### E4b launch audit E4b-I1 — 2026-07-13 09:58 Europe/Kyiv

- **Launch:** hidden sequential controller PID 22476 started from clean source commit `fa4906f` at
  09:53:09. The launch-time authorization, repeated design/data/parity check and second live W&B
  write all passed. No other project trainer was active.
- **Active cell:** diffusion spatial package `hier-long`, seed 3, RunSpec `e6ff5f95708c15ce`;
  active trainer PID 24632. Its RunSpec is bound to the frozen train/test/action/AE hashes,
  B16×30k, LR 3e-4, warm-up 2k, EMA 0.999, bf16, NFE 16 and source commit `fa4906f`.
- **First interval:** updates 1–500 average loss **1.5590** at 0.40 s/update and 6.46 GB measured
  peak. This is essentially identical to the seed-2 legacy-kernel interval (1.5590 at 0.38
  s/update) and bracketed by seeds 0/1 (1.5641/1.5620). No numerical drift or instability signal
  is present.
- **GPU/throughput:** 100% compute utilization, approximately 8.05/16.38 GB board memory, 63°C and
  113.5 W. Compute is saturated. Increasing batch to consume 14 GB remains scientifically
  prohibited because it would change the frozen optimizer/noise/EMA regime.
- **Cloud/provenance:** W&B run
  `https://wandb.ai/leo27heady/hgrid/runs/e4b-head-package-seeds3-4-e6ff5f95708c15ce`
  is online; the active registry records the same RunSpec, PID, source commit and 4.5-hour cap.
- **Projection:** at 0.40 s/update, training is approximately 3.33 hours plus rollout/upload. The
  cell should remain below 4.5 hours; the initial eight-cell projection is approximately 27–31
  project GPU-hours, within the frozen 36-hour replacement maximum. Continue unchanged.

#### Intermediate audit E4b-I2 — 2026-07-13 10:08 Europe/Kyiv

- **Progress and liveness:** `hier-long` seed 3 reached update 2,000/30,000 (6.7%) approximately
  13.5 minutes after cell start. Controller PID 22476, venv shim PID 10880 and trainer PID 24632
  remain alive in the correct parent chain. The experiment registry, process arguments and
  RunSpec all agree on study/cell/seed/source commit and the 4.5-hour cell cap. No spill, stall,
  non-finite value or unexpected process is present.
- **Expected versus actual convergence:** every observed interval follows the legacy warm-up
  envelope. The small 2k excess is 0.9% above the largest prior seed and is not a divergence.

  | Update | Seed 3 actual loss | Legacy seeds 0/1/2 | Assessment |
  |---:|---:|---:|---|
  | 500 | 1.5590 | 1.5641 / 1.5620 / 1.5590 | inside envelope |
  | 1,000 | 1.1988 | 1.1813 / 1.2038 / 1.1941 | inside envelope |
  | 1,500 | 0.8495 | 0.7900 / 0.8450 / 0.8639 | inside envelope |
  | 2,000 | 0.3995 | 0.3895 / 0.3960 / 0.3749 | +0.0035 above prior maximum; immaterial |

- **Validation/qualitative state:** the fixed-seed 2k grid exists and shows the expected noisy,
  partially localized warm-up-boundary predictions; it is comparable to the legacy 2k grids and
  is too early for a rollout/package inference. The local trainer logged the 2k grid while the
  W&B public API was visible through update 1,500 at the audit cutoff. This short queue lag is not
  a cloud outage; the run remains state `running`, with the exact 500/1,000/1,500 losses, LR
  warm-up, throughput and 6.461 GB peak present online.
- **Checkpoint/provenance:** the 2,000-update atomic checkpoint is 355,264,107 bytes, format 2,
  bound to RunSpec `e6ff5f95708c15ce`. It contains model, AdamW, EMA, Python/NumPy/Torch
  CPU+CUDA RNG, four latent-stat tensors and sampler state `examples_seen=32000`,
  `next_update=2001`. The run is explicitly recoverable but no resume is currently needed.
- **GPU and throughput:** current utilization is 95–100%, board memory approximately
  8.15/16.38 GB, 63°C and 113.5 W; observed intervals are 0.38–0.40 s/update, matching the modern
  legacy seeds. The GPU is compute-saturated, so unused VRAM is not evidence of an unexploited
  speed path. Frozen B16 remains mandatory.
- **Budget/projection:** no completed-cell ledger row exists yet, as expected. At the observed
  rate, remaining training is approximately 2.96 hours plus diffusion rollout/upload, keeping
  the cell near 3.3–3.6 hours and below 4.5. The tranche remains projected at 27–31 hours below
  its 36-hour replacement cap.
- **Scientific validity/decision:** the manipulated factors, target interface, optimizer, EMA,
  evaluation geometry and adaptive-development-pool disclosure remain exactly frozen. No early
  metric is used for selection. Continue unchanged; no stop, restart, tuning or design trigger
  fires.

## 2. Historical handoff boundary — E0

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

### E8c — CW-VAE-style baseline/protocol bridge — GPU allocation replaced by E4b

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

**Current E8 allocation:** E8a retains 0–4 GPU hours. E8c training is removed from the active
portfolio and replaced by E4b; its architecture/protocol comparison table remains CPU-only.

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
| 9 | E4b | Seeds 3–4 factorial expansion | triggered; replaces E8c | 28–36 h | 2–4 h |
| 10 | E9 | Counterfactual Dynamic LoopWorld primary factorial | post-freeze extension | 60–100 h | 1–2 wk |
| — | E8c | CW-VAE protocol bridge training | removed/replaced by E4b; table remains CPU-only | 0 | 3–7 d table/audit |
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

The extension envelope contains triggered E4b in place of E8c training, plus the post-freeze E9
primary factorial.

- **maximum allocation:** **136 GPU hours**;
- E4b: at most 36 hours, replacing E8c;
- E9: at most 100 hours;
- unused closure budget does not automatically authorize extension work.

### Pool C — protected reserve

- **allocation:** **50 GPU hours**;
- covers pilots, failed runs, retries and estimation error;
- cannot fund discretionary experiments or broaden a completed study.

### Hard ceiling and replacement rule

The revised expected maximum is approximately **367 GPU hours**, leaving about 33 hours below the
absolute 400-hour ceiling. Plan toward roughly **316 hours of scientific runs**, not 400 hours of
spending. E4b has replaced E8c rather than adding to the portfolio. E3b, E5c, E7b, E8b, E9c and
E10 remain outside this portfolio.

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
