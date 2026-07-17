# Comprehensive experiment execution order for the context-packaging vision

**Frozen:** 2026-07-12

**Authority:** the author retains scientific and GPU-launch authority. After the current-run
handoff, the author may grant Codex time-bounded execution autonomy within an explicitly approved
tranche. Delegation expires at the tranche stop condition or when the author withdraws it.

**Scientific identity:** context packaging is the manipulated design space. Reach and fidelity are
the confirmed explanatory axes; self-fed trust remains a falsifiable robustness question. The
article is about when hierarchical/multirate context pays under bounded budgets. E5a showed that
the seed-0 late trust inversion is not a stable third title-level result.

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
12. Measured consumption through terminal E8a is 104.452 project GPU-hours. The forward portfolio
    reserves at most 40 hours for article-gated E6, 100 for post-freeze E9 and 50
    protected reserve hours. The reserve cannot fund discretionary experiments.
13. Stop when the claims are resolved. Unused budget is not a reason to launch another study.

### 1.1 Canonical execution and status ledger

This document is also the single canonical live ledger for the programme. Do not create a second
execution-order document. Each four-hour audit appends a dated entry here containing expected
versus actual progress, metric/throughput/health trends, projected completion, cumulative GPU
hours, deviations and the resulting gate decision. Raw logs, W&B and immutable artifacts remain
the evidence; this section is the human-readable index. Status vocabulary is `NOT STARTED`,
`GATED`, `RUNNING`, `COMPLETE`, `DEFERRED`, `BLOCKED` or `CANCELLED`.

#### Programme status at 2026-07-17 12:28 Europe/Kyiv

| ID | Workstream | Expected before execution | Actual/status | Decision |
|---|---|---|---|---|
| E0 | c2-only optimizer-regime control | Finish without entering Phase C; replace stale runtime estimate | **COMPLETE**; 14.223 measured project GPU-hours; flat beats hierarchy by 0.00773, 95% CI [0.00304, 0.01233] in flat's direction | Retain as optimizer-regime robustness, not more-data/asymptotic evidence |
| E1 | Evidence freeze and governance repair | 8–16 CPU hours; no GPU | **COMPLETE**; evidence manifest/independent backup, result integration, canonical text/claim governance and Phase-C amendment restored | E2 preparation permitted |
| H3a/H4 | Confirmation evaluator, provenance, metrics and CPU gates | Exact checkpoint readers, immutable artifacts, complete metrics, clean CPU/hash gates | **COMPLETE**; 35 tests plus design/schema/consistency/compile gates; frozen dataset and checkpoint hashes pass | E2 authorized on commit `b9d91da` |
| H3b | Safe new-training baseline and numerical parity | Same scientific kernel plus RunSpec, RNG-complete explicit resume, immutable artifacts and cloud identity | **COMPLETE**; 38 tests, exact first-update/batch/model/optimizer/EMA parity and uninterrupted-versus-resumed trajectory; report bound to `c0ab78c` | New training may use only the provenance-v2 wrapper |
| E2a | Untouched LoopMom holdout | 1–2 CPU hours | **COMPLETE**; 500×200 set, seed 20260712, dataset ID `loopmom-confirmation_holdout-20260712-e033abee6e04`; no train/test glyph collisions | Dataset is now accessed and permanently frozen |
| E2b | Two primary confirmation pairs | 4–8 GPU-hours | **COMPLETE**; all 8 evaluations succeeded in 0.687 GPU-hours | Both registered reach-band effects confirm on both seeds with clip-bootstrap intervals excluding zero |
| E2c | Frozen frontier confirmation | 6–10 additional GPU-hours | **COMPLETE**; all 18 evaluations succeeded; 26/26 E2 runs complete | Simultaneous frontier confirms lower/interior hierarchy wins and upper flat inversion; no interpolation |
| E3a | Fixed-target diffusion B1 | 14–18 GPU-hours only if causal stride claim remains central | **DEFERRED / NOT STARTED** | Current article uses whole-configuration wording, so E3a is not required for present scope; reconsider only if causal stride language returns |
| E4a | Seed-2 head × package factorial | 7–10 GPU-hours after H3b numerical parity | **COMPLETE / PASS**; two cells and analysis consumed 6.1228 project GPU-hours; completion validator PASS | Historical three-seed trigger is superseded by terminal E4b evidence below |
| E4b | Heterogeneity expansion, seeds 3–4 | 28–36 GPU-hours, replacement-only | **COMPLETE / PASS**; 8/8 cells, registered analysis and independent validator passed in 24.9922 project GPU-hours | Retain five-seed-stable diffusion package result; demote directional head shift; regression and interaction are training-seed-heterogeneous; no seeds 5+ |
| E5a/b | Trust replication and dose response | E5a ≤28 GPU-hours; E5b only if the registered boundary replicated | **E5a COMPLETE / NOT_REPLICATED** in 17.7852 project GPU-hours; both new seeds reproduce the early hierarchy benefit but neither reproduces the late inversion | **E5b NOT AUTHORIZED**; retain the seed-0 inversion as a negative-replication case, not a stable trust boundary |
| E7c | Sampler/NFE robustness | 35 checkpoint-only evaluations; ≤8 GPU-hours | **COMPLETE / DIRECTIONALLY_ROBUST**; 35/35 cells and 20/20 positive registered effects in 5.9976 project GPU-hours; 18/20 simultaneous lower bounds exceed zero | Freeze the directional result; disclose that the 2,432-token inversion is family-wise unresolved at sampler seeds 1/2; no adaptive extension |
| E7a | Alternate-codec robustness | Frozen seven-arm seed-0 matrix, new locked holdout, joint eight-effect family; ≤36 GPU-hours | **COMPLETE / CODEC_DIRECTIONALLY_ROBUST**; 7/7 trainings, 14/14 evaluations, eight-effect simultaneous analysis, independent validator and W&B audit pass in 30.0238 project GPU-hours | Retain all four directions across the independent codec at point-estimate level; disclose that the alternate-codec 2,432-token inversion is family-wise unresolved |
| E8a/E6/E9 | Retrieval, Phase C and optional extension | Decision-tree gated | **E8a COMPLETE / RAW_ACCESS_CEILING_CI_CLEAN** in 1.0897 GPU-hours; E6/E9 remain gated/deferred | Freeze E8a; complete M1 article assembly before applying the E6 utility gate |

#### Intermediate audit E7a-I0 — 2026-07-16 04:27 Europe/Kyiv

- **Frozen estimand:** whole alternate-codec-training-seed robustness, conditional on predictor
  seed 0. The intervention includes codec weights, latent geometry, codec-specific normalization
  statistics and retrained predictors; it does not isolate one representation component.
- **Matrix:** exactly seven alternate-codec B16x30k diffusion predictors and four registered
  contrasts: 896 reach, 1,152 interior, 1,152 package shape and 2,432 inversion. The same new
  holdout evaluates seven frozen original-codec controls and seven alternate-codec rows. No
  second predictor seed is included because the high-budget pair makes that incompatible with
  the 36-hour tranche; training-seed robustness is supplied by the earlier programme.
- **Decision:** the eight codec-by-contrast effects form one studentized max-statistic
  clip-cluster family. Outcome categories are holdout-control failure, codec-sensitive,
  directionally robust or CI-clean. No codec retry, third codec, extra seed, arm or threshold may
  be added after results are visible.
- **Validity floor:** before any predictor or locked-data access, seed-1 reconstruction and latent
  non-collapse must pass fixed 4,000-frame training-tail limits relative to the original codec.
  Failure closes E7a without retraining.
- **Code/CPU gate:** 56 tests pass; article/dissertation consistency and scientific-config gates
  pass; PowerShell and Python compile gates pass. The explicit codec seed-0 path is bit-exact to
  two legacy CPU updates, while seed 1 is distinct and checkpoint metadata is complete.
- **Budget:** expected 31.2 GPU-hours. Frozen components are 0.6 shared by codec training/floor,
  31.4 across predictor cells and 3.3 across fourteen evaluations: **35.3 allocated within a
  36.0-hour hard maximum**. Failed attempts and resumes remain charged.
- **Final prelaunch gate:** the independently generated 500×200 set is
  `loopmom-codec_robustness_holdout-20260716-77a4ec5aee94`. Shape/dtype, action-position
  consistency, persistent-object metadata, empirical hold/continuation rates and all artifact
  hashes pass. There are zero exact clip collisions internally or against development train/test,
  E2 and E7c. Hash-bound protocol/config/CPU/design/data attestations pass.
- **Launch decision:** E7a is authorized under the user's standing tranche autonomy. Start only
  from the final clean commit after the controller's live W&B write gate; the codec floor remains
  an outcome-independent stop before predictors or semantic holdout access.

#### Intermediate audit E7a-I1 — 2026-07-16 04:40 Europe/Kyiv

- **Launch:** PASS from clean commit `d75d12f` after the controller repeated the authorization,
  14-cell design, data/hash and live W&B write gates. The pinned execution commit is
  `d75d12f0322a657f727a6be7a1beda18f4731b53`; later documentation commits cannot alter resume
  identity. Controller PID 10464 owns the sole scientific child.
- **Active phase:** alternate `HierConvAE` seed 1, 7,000 updates, B64, Adam 0.0003 and unchanged
  random-level objective; RunSpec `460d2f4a301438d9`. W&B run:
  `https://wandb.ai/leo27heady/hgrid/runs/e7a-alternate-codec-seed1-460d2f4a301438d9`.
- **Initial numerical trend:** at 1k/2k/3k updates, loss is 0.2631/0.1135/0.0737, coarse validation
  MSE 0.0178/0.0090/0.0063 and fine validation MSE 0.0083/0.0040/0.0027. The legacy seed-0
  reference is loss 0.2786/0.1143/0.0774, coarse 0.0192/0.0094/0.0072 and fine
  0.0069/0.0038/0.0030. The seed-1 trajectory is finite, convergent and within the expected
  representation-seed envelope; the small early fine difference has already narrowed/reversed.
- **Hardware/runtime:** 94–96% GPU compute, approximately 2.66/16.38 GB board memory, 64°C and
  117 W. Low codec VRAM is expected for this small convolutional model and cannot be converted
  into a batch change because B64 is the frozen intervention. Three thousand updates completed
  in roughly 80 seconds, projecting codec training plus floor comfortably inside their shared
  0.6-hour allocation.
- **Isolation:** no E7a GPU-ledger row exists until the codec process exits, no predictor has
  started, and the purpose-locked holdout has not been semantically opened. Continue unchanged;
  the next automatic stop is the registered reconstruction/latent non-collapse floor.

#### Intermediate audit E7a-I2 — 2026-07-16 04:43 Europe/Kyiv

- **Codec completion:** seed-1 training completed normally at 7,000/7,000 updates in 0.05385
  project GPU-hours. Final loss was 0.0452, with coarse/fine validation MSE 0.00468/0.00197;
  all values remained finite and the checkpoint/artifact upload completed.
- **Registered floor:** `CODEC_FLOOR_PASS`. Alternate versus original full-frame MSE was
  0.001202 versus 0.001265 at the coarse level and 0.000509 versus 0.000516 at the fine level;
  foreground MSE was 0.02907 versus 0.03009 and 0.01253 versus 0.01254. Every latent-channel
  standard deviation exceeded 0.1075, far above the frozen 0.005 non-collapse floor. These are
  gate diagnostics, not evidence for selecting the alternate codec.
- **Advance:** the unchanged controller correctly entered predictor 1/7, `hier-min` seed 0,
  RunSpec `993ee4b6535259e3`, at 04:41:57. Training geometry remains B16×30k, learning rate
  0.0003, warmup 2,000, bf16 and predictor seed 0. W&B run:
  `https://wandb.ai/leo27heady/hgrid/runs/e7a-alternate-codec-predictors-993ee4b6535259e3`.
- **Health/budget:** training initialization, latent encoding and standardization completed
  without exception. GPU telemetry reached 100% compute, 6.72/16.38 GB board memory, 63°C and
  112 W. Measured codec plus floor consumption is 0.05769 hours versus their 0.6-hour shared
  maximum; no budget or validity deviation exists. Continue unchanged to the first four-hour
  audit. The locked E7a outcome evaluation remains downstream of all seven predictor trainings.

#### Four-hour audit E7a-I3 — 2026-07-16 08:44 Europe/Kyiv

- **Expected versus actual:** expected the codec/floor to be closed and the first predictor to be
  complete or nearly complete. Actual progress is slightly ahead: the codec floor remains PASS,
  `hier-min` completed 30,000/30,000, and `flat-min` reached 17,500/30,000 (58.3%). The sole
  controller and sole scientific child remain alive with the frozen source commit, codec hash,
  seed, batch, learning rate, warmup, precision and update count.
- **Completed `hier-min` trajectory:** all 60 logged loss/throughput intervals are finite. Relative
  to the original-codec seed-0 run, training loss averaged 4.1% lower (interval range −15.5% to
  +4.6%) and ended 1.3% lower, 0.04859 versus 0.04923. Validation foreground MSE averaged 1.3%
  lower over the fifteen matched grids. Its final 0.03364 is above the original's final 0.02670
  but remains inside the original late-training 0.0267–0.0423 fluctuation band and follows several
  alternate-codec grids that were lower than the original; this is grid noise, not sustained
  numerical drift. Visual grids show matched glyph structure without collapse.
- **Active `flat-min` trajectory:** loss is finite, decreasing from 1.5809 at 500 to 0.0840 at
  17,500, and is approximately 9% above the original-codec trajectory on average. That difference
  is compatible with the registered whole-codec intervention: after per-channel standardization,
  alternate latent geometry can still change the diffusion objective scale. The directly useful
  grid diagnostic does not deteriorate: after the warmup grids, alternate validation foreground
  MSE is lower at every matched 6k–16k checkpoint (12–26% below the original), reaching 0.00724
  versus 0.00870 at 16k. Do not treat the loss-scale shift as an outcome or alter training.
- **Throughput/hardware:** `hier-min` used 2.5624 GPU-hours versus 2.752 expected and 3.2
  authorized; it averaged 3.4% faster than legacy. `flat-min` is averaging 0.291–0.299 seconds per
  update, approximately 2.2% faster than legacy, and projects to about 2.45–2.55 hours. Current
  telemetry is 100% compute, 89% memory-controller utilization, 6.75/16.38 GB board memory,
  66°C, 118 W and P2. Allocated model peak is the frozen 5.43 GB. There is no stall, thermal
  throttling, allocation growth or WDDM spill signal.
- **Artifacts/cloud:** the completed `hier-min` RunSpec is `993ee4b6535259e3`; its 18/18 manifest
  entries, including the 355,264,107-byte checkpoint, rehash exactly and its COMPLETE marker
  matches the RunSpec. Codec, codec-floor and `hier-min` W&B runs are finished with COMMITTED
  artifacts; active `flat-min` RunSpec `019fd82cf6963442` is syncing normally. The recurring
  PowerShell `NativeCommandError` text is W&B's stderr stream classification; all completed native
  children exited zero and it has no operational effect.
- **Isolation/budget/projection:** predictor training continues to use the frozen legacy
  development validation hash; no E7a evaluation/access-log artifact exists and the locked outcome
  evaluation has not started. Completed ledger consumption is 2.6201 hours; including the active
  cell, approximately 4.09 project GPU-hours have elapsed. Observed throughput projects the full
  tranche near 30–32 hours, still below 36, with a provisional terminal window around 2026-07-17
  11:00–13:00 Kyiv. The whole-codec estimand remains valid. Continue unchanged; no user action or
  adaptive branch is warranted.

#### Four-hour audit E7a-I4 — 2026-07-16 12:43 Europe/Kyiv

- **Expected versus actual:** expected `flat-min` to finish and `hier-long` to be well underway.
  Actual progress is ahead: `flat-min` completed 30,000/30,000 and `hier-long` reached
  28,000/30,000 (93.3%). Controller PID 10464 and the one active scientific child remain healthy;
  the pinned source commit, alternate-codec checkpoint, predictor seed 0 and all frozen training
  geometry remain unchanged.
- **Completed `flat-min`:** final training loss is 0.06939 versus the original-codec 0.05987
  (+15.9%; +11.5% averaged over all 60 intervals). This persistent objective-scale shift is paired
  with better, not worse, fixed-grid reconstruction: alternate validation foreground MSE averaged
  12.1% lower over the fifteen grids and ended 23.9% lower, 0.00709 versus 0.00932. Every grid from
  6k through 30k favored the alternate codec, and visual inspection shows stable, aligned glyph
  predictions. The result is consistent with codec-dependent standardized latent geometry rather
  than optimizer failure.
- **Active `hier-long`:** all values remain finite and the loss decreased from 1.5757 at 500 to
  0.0459 at 28k. Against the actual local legacy B16 log, loss differs by only +1.0% on average and
  is 2.8% lower at 28k; model peak memory is identically 6.46 GB. Against the immutable
  provenance-v2 seed-3/4 runs, mean loss differs by only +0.08%/+0.25% and throughput by
  +0.34%/+0.27%, respectively. The 26k validation grid is higher at 0.04231 than the seed-3/4
  references (0.02618/0.03180), and the matched-grid mean is approximately 15% higher. This is a
  mild representation-specific validation shift to monitor, not a numerical-validity failure:
  training convergence is matched, the saved grid remains structurally coherent, and the locked
  rollout—not this noisy eight-row grid—is the registered endpoint.
- **Reference correction:** the W&B ID `rhvd-hier-long-s0` contains telemetry from a later reused
  regime and therefore falsely suggested 3.49 GB and 0.22 seconds/update. The immutable local
  original B16 log records 6.46 GB and 0.73–0.76 seconds/update, while the provenance-v2 references
  record approximately 0.376–0.377 seconds/update at 6.46 GB. The active 0.376–0.381 seconds/update
  is the expected v2 regime. This is a legacy cloud-identity caveat only; E7a's content-addressed
  RunSpecs and scientific estimand are unaffected.
- **Artifacts/cloud:** `flat-min` RunSpec `019fd82cf6963442` has a matching COMPLETE marker and
  18/18 manifest hashes, including its 355,055,211-byte checkpoint. Its W&B run is finished with a
  COMMITTED training artifact. `hier-long` RunSpec `c483500021b83d2e` is active and syncing; no
  failed or duplicate scientific child is present.
- **Hardware/budget/projection:** current telemetry is 99–100% compute, 87% memory-controller
  utilization, 7.88/16.38 GB board memory, 63–66°C, 114–117 W and P2. There is no allocation
  growth, thermal throttle, stall or WDDM spill signal. `flat-min` used 2.4689 hours versus 2.752
  expected and 3.2 authorized. Completed consumption is 5.0890 hours; including active
  `hier-long`, approximately 8.07 hours have elapsed. The active cell projects to approximately
  3.20 hours, below its 3.5-hour maximum. The tranche remains projected at roughly 30–32 hours,
  below 36, with the provisional terminal window still around 2026-07-17 11:00–13:00 Kyiv.
- **Isolation/decision:** predictor RunSpecs still reference only the frozen legacy development
  validation array. No E7a evaluation or access-log artifact exists, so the locked outcome set has
  not been evaluated early. Continue unchanged. There is no blocker, estimand change or
  justification for an adaptive cell.

#### Four-hour audit E7a-I5 — 2026-07-16 16:42 Europe/Kyiv

- **Expected versus actual:** expected `hier-long` to complete and `flat-budget` to be well
  underway. Actual progress is ahead: both completed 30,000/30,000 and predictor 5/7,
  `strided-flat`, reached 6,000/30,000. The controller remains the sole launch owner and the
  active RunSpec retains the frozen alternate codec, predictor seed 0, B16×30k geometry, learning
  rate, warmup, bf16 precision and legacy development-validation hashes.
- **Completed `hier-long`:** final loss is 0.04446. It differs by +4.2% from the original local
  B16 seed-0 final loss (0.0427), by +1.5% from provenance-v2 seed 3 (0.04378), and by −0.6% from
  provenance-v2 seed 4 (0.04470); the all-interval mean is only +1.4% versus the local original.
  Final validation foreground MSE is 0.04646, higher than the available original/provenance-v2
  seed-1–4 finals (0.02670–0.03605). This is an honest codec-dependent package response, not a
  numerical failure: loss convergence, finite values, model memory and saved glyph structure all
  remain normal. It may weaken the hierarchical arm on the locked rollout, which is precisely the
  registered robustness question; no intervention is permitted.
- **Completed `flat-budget`:** final loss is 0.06854 versus the local original 0.0698 (−1.9%;
  +3.5% averaged over all intervals), at identical 6.47 GB model peak. Its final validation
  foreground MSE is 0.07546, inside the observed original seed envelope represented by seed 1
  (0.07330) and seed 2 (0.07933). Visual inspection shows aligned glyph predictions and expected
  background uncertainty, with no collapse or missing-output pattern.
- **Active `strided-flat`:** loss decreases normally from 1.5865 at 500 to 0.1353 at 6k. At 5.5k
  it is −1.0% versus original seed 0, −2.3% versus provenance-v2 seed 3 and +4.2% versus seed 4.
  Throughput is 0.350–0.352 seconds/update and 6.26 GB, matching the v2 references
  (0.350–0.352 seconds/update, 6.26 GB). The 4k validation grid is 0.05801, 14–27% above the three
  matching references, but only two early grids exist; the 6k qualitative grid is coherent and
  this early difference is not a stop criterion.
- **Artifacts/cloud:** `hier-long` RunSpec `c483500021b83d2e` and `flat-budget` RunSpec
  `e7edc69a480412a7` each have matching COMPLETE markers and 18/18 verified manifest hashes.
  Their 355,264,107-byte and 355,079,723-byte checkpoints are intact. Both W&B runs are finished
  with COMMITTED training artifacts; active `strided-flat` RunSpec `fb603c2f0565f70c` is syncing
  normally.
- **Hardware/budget:** telemetry is 97–100% compute, 81–83% memory-controller utilization,
  7.70/16.38 GB board memory, 68°C, 119 W and P2, with no stall, thermal throttle, allocation
  growth or WDDM spill signal. `hier-long` used 3.1822 hours versus 3.01 expected/3.5 maximum;
  `flat-budget` used 3.1126 versus 2.924 expected/3.4 maximum. Both exceeded the planning estimate
  by approximately 6% but remained safely inside their hard cell limits. Completed tranche
  consumption is 11.3838 hours; including the active cell, approximately 12.02 hours have elapsed.
  `strided-flat` projects near 2.9–3.0 hours versus its 3.3 maximum. The full tranche remains
  projected near 30–32 hours, below 36, with a provisional terminal window around
  2026-07-17 11:00–13:00 Kyiv.
- **Isolation/decision:** no E7a evaluation/result/access-log artifact exists and the purpose-
  locked outcome set has not been evaluated. The higher hierarchical validation grid is a
  possible scientific outcome, not a validity or health blocker. Continue the frozen queue
  unchanged; no adaptive checkpoint, seed, codec or package is justified.

#### Four-hour audit E7a-I6 — 2026-07-16 20:42 Europe/Kyiv

- **Expected versus actual:** expected `strided-flat` to complete and the first high-budget arm to
  begin. Actual progress matches or slightly exceeds that expectation: `strided-flat` completed
  30,000/30,000 and `hier-c2` reached 7,500/30,000 (25%). The controller and one active child
  retain the frozen source commit, seed, alternate-codec hash, B16×30k optimizer geometry and
  development-validation hashes.
- **Completed `strided-flat`:** final loss is 0.05764, which is 5.3% below original seed 0,
  8.3% below provenance-v2 seed 3 and 6.4% below provenance-v2 seed 4. Final validation
  foreground MSE is 0.02209 versus 0.02155/0.02163/0.02102 for those references, a small
  +2.1–5.1% difference. Although early validation grids produced a higher all-grid mean, the gap
  closed by 30k and the final qualitative grid is structurally matched. This is stable completion,
  not representation collapse.
- **Active `hier-c2`:** loss is finite and decreases from 1.5925 at 500 to 0.0823 at 7.5k.
  Across the fourteen intervals available through 7k, loss differs from original seed 0 by only
  +0.46% on average (range −6.1% to +6.9%) and is 2.9% lower at the latest matched interval.
  Validation foreground MSE averages 6.9% above the original across the first three grids and is
  0.04816 versus 0.04599 at 6k. The saved grid remains coherent. Throughput averages 2.35% faster
  than the original B16 run, at approximately 0.80–0.81 seconds/update, with identical 11.68 GB
  allocated model peak.
- **Artifacts/cloud:** completed `strided-flat` RunSpec `fb603c2f0565f70c` has a matching COMPLETE
  marker and 18/18 verified manifest hashes; its 355,264,043-byte checkpoint is intact. Its W&B
  run is finished with a COMMITTED training artifact. Active `hier-c2` RunSpec
  `7e3621066377f579` is syncing normally and has no duplicate or failed scientific child.
- **Hardware/high-budget headroom:** telemetry is 97–100% compute, 68–72% memory-controller
  utilization, 13.34/16.38 GB board memory, 67°C, 120 W and P2. Model peak is the expected
  11.68 GB, leaving approximately 3.0 GB of board headroom after desktop allocations. There is no
  monotonic memory growth, thermal throttle, utilization collapse or WDDM spill indicator.
- **Budget/projection:** `strided-flat` used 2.9546 hours versus 2.838 expected and 3.3 maximum.
  Completed tranche consumption is 14.3384 hours; including active `hier-c2`, approximately
  16.04 hours have elapsed. At the observed 0.80–0.81 seconds/update, `hier-c2` projects to
  approximately 6.7 hours, below its 7.4-hour maximum. Allowing the same pace for `flat-c2` plus
  all fourteen evaluations leaves the full tranche near 31–32 hours and below the 36-hour cap,
  with the terminal window still approximately 2026-07-17 11:00–13:00 Kyiv.
- **Isolation/decision:** no E7a evaluation/result/access-log artifact exists and the locked
  outcome set remains unevaluated. No numerical, operational, budget or scientific-validity
  blocker is present. Continue unchanged into the high-budget pair; no adaptive cell is warranted.

#### Four-hour audit E7a-I7 — 2026-07-17 00:43 Europe/Kyiv

- **Expected versus actual:** expected `hier-c2` to be in late training with adequate margin to
  its 7.4-hour cap. It reached 25,500/30,000 (85%) after approximately 5.71 active hours, matching
  the projection. The controller and single scientific child are alive; source commit,
  alternate-codec hash, predictor seed 0, B16×30k optimizer geometry, EMA and development
  validation remain frozen.
- **Loss and validation:** all fifty W&B intervals through 25k are finite. Alternate-codec
  training loss averages 8.0% above original seed 0 and is 21.9% higher at 25k
  (0.04657 versus 0.03822). This is not accompanied by predictive deterioration: across twelve
  matched validation grids, foreground MSE averages 2.9% lower, and the alternate codec is
  19.1% lower at 24k (0.03690 versus 0.04561). The latest six validation comparisons all favor
  the alternate codec. The late 24k qualitative grid remains aligned and structurally coherent.
  The higher diffusion objective is therefore a codec-dependent latent-scale response, not a
  stall, divergence or collapse signal.
- **Throughput/memory:** throughput averages 2.58% faster than the original B16 run and remains
  0.798–0.804 seconds/update in the late interval. Allocated model peak is exactly stable at
  11.6785 GB. Board telemetry is 100% compute, 76% memory-controller utilization,
  13.29/16.38 GB, 68°C, 120 W and P2. There is approximately 3.1 GB board headroom with no
  monotonic allocation growth, thermal throttle, utilization collapse or WDDM spill evidence.
- **Cloud/artifact state:** active RunSpec `7e3621066377f579` is continuously updating W&B with
  the expected training summary and media; no training artifact is committed before normal
  completion, as designed. No duplicate or failed scientific process exists.
- **Budget/projection:** completed consumption remains 14.3384 hours; including the active cell,
  approximately 20.05 project GPU-hours have elapsed. At current pace, `hier-c2` should finish
  near 6.70–6.75 hours, retaining 0.65–0.70 hours of cell headroom. Allowing the same conservative
  duration for `flat-c2` plus the registered evaluations projects 31–32 total tranche hours,
  below 36, with terminal completion still near 2026-07-17 11:00–13:00 Kyiv.
- **Isolation/decision:** no E7a evaluation/result/access-log artifact exists; the locked outcome
  set remains unevaluated. The whole-codec estimand is intact and there is no numerical,
  operational or budget blocker. Continue unchanged to `hier-c2` completion and the frozen
  `flat-c2` cell.

#### Four-hour audit E7a-I8 — 2026-07-17 04:43 Europe/Kyiv

- **Expected versus actual:** expected `hier-c2` to finish near 01:40 Kyiv and `flat-c2` to be
  approximately three hours into the final predictor cell. Actual progress matches that forecast:
  `hier-c2` completed 30,000/30,000 at 01:41, and `flat-c2` reached 13,500/30,000 (45%) by the
  audit. Controller PID 10464 and the sole active scientific child remain alive. The pinned
  source commit, alternate-codec hash, predictor seed 0, B16×30k optimizer/EMA geometry and
  development-validation hashes are unchanged.
- **Completed `hier-c2`:** all sixty loss/throughput intervals and fifteen validation grids are
  finite. Training loss averages 9.7% above the original-codec seed-0 B16 trajectory, is 18.7%
  higher over the final 5,000 updates and ends 22.5% higher (0.04462 versus 0.03641). The directly
  useful fixed-grid diagnostic moves in the opposite direction: alternate-codec foreground MSE
  averages 4.3% lower over all matched grids, 12.7% lower over the last five and ends 15.1% lower
  (0.03709 versus 0.04370). The final qualitative grid retains aligned glyph structure. This
  confirms a codec-dependent diffusion-objective scale without predictive collapse, divergence
  or an optimization stall.
- **Active `flat-c2`:** loss decreases normally from 1.5962 at 500 to 0.0880 at 13.5k. Across the
  27 matched intervals it averages 7.6% above the uninterrupted/resumed original B16 trajectory
  and is 17.8% higher at 13.5k. Its six completed validation grids average 1.7% *lower* than the
  original, ending at 0.05445 versus 0.05520 at 12k; the 12k saved grid is structurally coherent.
  Throughput averages 1.1% faster than the matching original interval at approximately
  0.79–0.80 seconds/update. The same loss/validation dissociation seen in `hier-c2` is therefore a
  legitimate whole-codec response, not an operational health failure.
- **Artifacts/cloud:** completed `hier-c2` RunSpec `7e3621066377f579` has the matching COMPLETE
  marker and all 18/18 manifest entries rehash exactly, including the 355,264,107-byte checkpoint.
  Its W&B run is finished and its training artifact is COMMITTED. All six completed predictor
  runs, the codec and codec-floor runs are finished with committed artifacts; active `flat-c2`
  RunSpec `afe335a270f4a293` is online and syncing normally. The reused legacy c2 W&B IDs still
  expose the later B8×60k telemetry, so all B16 trajectory comparisons above use the immutable
  local original logs, including `rhvd_flat-c2_resume.log` after its historical 16k interruption.
- **Hardware/budget/projection:** `hier-c2` used 6.7041 GPU-hours versus 6.364 expected and 7.4
  authorized. Completed E7a consumption is now 21.0426 hours. Board telemetry during `flat-c2`
  is 100% compute, 57% memory-controller utilization, 13.21/16.38 GB, 63°C, 120 W and P2; model
  peak is stable at 11.6820 GB. There is approximately 3.2 GB board headroom and no allocation
  growth, thermal throttle, stall or WDDM spill signal. Current pace projects `flat-c2` near
  6.65–6.75 hours, below its 7.4-hour cell cap, and the complete E7a tranche near 31–32 hours,
  below 36. Terminal completion remains approximately 2026-07-17 11:00–13:00 Kyiv.
- **Isolation/decision:** no E7a evaluation, result or access-log artifact exists; the locked
  outcome set remains unopened until all seven predictors complete. The whole-codec estimand and
  registered joint eight-effect family remain intact. Continue the frozen final predictor and
  subsequent fourteen evaluations unchanged; there is no blocker or basis for an adaptive cell.

#### Four-hour audit E7a-I9 — 2026-07-17 08:43 Europe/Kyiv

- **Expected versus actual:** expected `flat-c2` to finish and the fourteen-cell locked evaluation
  matrix to begin. All seven predictor trainings did finish, but the first evaluation
  (`original` codec, `hier-min`) failed at command-line parsing with exit code 2 before rollout.
  The controller exited fail-closed and the GPU is idle. E7a is therefore operationally blocked
  at evaluation dispatch, not scientifically failed and not terminal.
- **Completed `flat-c2`:** the final cell completed 30,000/30,000 in 6.6424 GPU-hours, below its
  7.4-hour maximum. All sixty loss/throughput intervals and fifteen validation grids are finite.
  Relative to the original B16 trajectory reconstructed from the immutable initial and resumed
  logs, alternate-codec loss averages 10.3% higher, 12.8% higher over the final 5,000 updates and
  ends 23.5% higher (0.06716 versus 0.05440). Validation foreground MSE is effectively unchanged:
  −0.08% averaged over all grids and −0.64% at 30k (0.05663 versus 0.05700). Throughput averages
  3.0% faster, model peak stays at 11.6822 GB, and the final qualitative grid remains coherent.
  This closes training with the already observed codec-dependent objective-scale shift but no
  predictive collapse, divergence or stall.
- **Immutable training evidence:** the alternate codec COMPLETE marker and all 3/3 manifest
  entries verify. Every one of the seven predictor COMPLETE markers matches its RunSpec and all
  126/126 predictor-manifest entries rehash exactly. The final `flat-c2` checkpoint is
  355,202,539 bytes with SHA-256
  `2f947c8354b9ee0d99922c770d9b635d28b77eb61674af4c2b1af52022a7ada6`.
  Its W&B run is finished with a COMMITTED training artifact, as are the other six predictors,
  codec and codec-floor runs.
- **Failure localization and holdout integrity:** no E7a access log, evaluation RunSpec,
  evaluation artifact, metric file, result file or W&B evaluation run exists. `argparse` rejected
  the command because `--ae_path` had no following value, before application code could open the
  500×200 array. The locked outcome therefore remains analytically unopened and no point estimate
  is visible. The failed dispatch consumed 0.00225 project GPU-hours and produced no scientific
  record.
- **Root cause:** the controller iterates with an outer object variable named `$codec`, then calls
  `Invoke-E7aGpuCell`, whose string parameter is named `$Codec`. PowerShell variable names are
  case-insensitive and the invoked script block resolves through dynamic scope; inside it,
  `$codec.ae` therefore refers to the function's string parameter and evaluates to null. The same
  deterministic wrapper defect would affect both codec rows. It is not a checkpoint, model,
  dataset, evaluator or W&B failure.
- **Budget:** valid completed codec/floor/training consumption is 27.6850 hours; including the
  failed pre-evaluation dispatch it is 27.6872 hours. The complete fourteen-cell evaluation
  allocation still fits comfortably below the frozen 36-hour tranche maximum. Hardware after the
  fail-closed exit is idle at 0.84/16.38 GB board memory and 43°C; no orphan controller,
  evaluator or trainer remains.
- **Recovery gate:** do not launch an evaluation manually and do not change frozen scientific
  geometry. Make a wrapper-only repair that captures the outer AE path in a non-shadowed scalar
  before entering `Invoke-E7aGpuCell`; add a CPU argv-spy test proving that all fourteen frozen
  rows receive non-empty, hash-matched AE/checkpoint paths and distinct codec identities; rerun
  the existing design/config/compile tests; refresh only the implementation attestation affected
  by the wrapper; record the recovery commit and failed-attempt provenance; then invoke the
  controller with explicit `-Resume`. The original execution source commit, config hash, dataset,
  checkpoints, NFE, sampler seed, rollout geometry, metrics and joint family remain frozen.

#### Recovery gate E7a-I10 — 2026-07-17 09:43 Europe/Kyiv

- **Repair scope:** the defect was confined to PowerShell argument resolution. No model/evaluator
  kernel, checkpoint, codec, dataset, metric, NFE, sampler seed, rollout geometry, contrast,
  bootstrap family or decision rule changed. The controller no longer interpolates an outer
  `$codec` object inside `Invoke-E7aGpuCell`.
- **Exact dispatch artifact:** `scripts/resolve_e7a_evaluation_dispatch.py` now resolves the two
  codecs × seven arms into fourteen complete command vectors before locked-data access. It
  verifies the original and alternate codec hashes, every original and alternate predictor
  checkpoint hash/COMPLETE marker, the pinned execution/config/dataset identity and every
  non-empty valued option. The controller executes the validated `argv` array directly rather
  than rebuilding it through dynamically scoped variables.
- **Real-artifact gate:** `gates/e7a_evaluation_dispatch.json` is
  `E7A_EVALUATION_DISPATCH_PASS` for all 14/14 rows. Every row contains the frozen NFE 16,
  sampler seed 0, 100 clips, 48 warmup and 152 rollout frames; all AE and checkpoint paths are
  non-empty and SHA-256 matched. The report is deterministic for the pinned execution, and
  `gates/e7a_dispatch_validated.json` hash-attests it. Resolving the matrix reads only metadata,
  codecs and checkpoints; it does not open the locked clip array.
- **Regression/completion gates:** a fixture-backed test constructs the full fourteen-cell matrix,
  verifies exact codec/arm coverage and checks that the actual argument following `--ae_path` and
  `--checkpoint_path` is non-empty and hash-bound. The terminal validator now requires the
  dispatch report and proves each completed evaluation used its dispatched codec/checkpoint.
  Full CPU validation passes: 58 tests, scientific-config/authorization validation,
  reach–fidelity–trust text consistency, existing-artifact-aware E7a design validation,
  bit-exact seed-0 AE parity, Python compilation, PowerShell parsing and clean diff checks.
- **Authorization/provenance:** the refreshed CPU report and dispatch report have hash-bound
  attestations, and `check_experiment_gate.py alternate-autoencoder-robustness` returns
  `AUTHORIZED / GATE PASS`. The original execution remains pinned to source commit `d75d12f`,
  dataset `loopmom-codec_robustness_holdout-20260716-77a4ec5aee94` and the unchanged frozen
  config hash. The failed code-2 dispatch remains in the GPU-hours ledger and will not be erased.
- **Resume decision:** commit the wrapper-only recovery and terminal codec-floor evidence from a
  clean worktree, then invoke the unchanged controller with explicit `-Resume`. It must skip the
  completed codec/floor and all seven completed trainings, regenerate the same dispatch report,
  and begin `original/hier-min`. The remaining registered maximum leaves more than five hours of
  tranche headroom; no adaptive cell or manual evaluator launch is authorized.

#### Terminal audit E7a-I11 — 2026-07-17 12:10 Europe/Kyiv

- **Terminal integrity:** all seven alternate-codec predictor trainings and all fourteen locked
  evaluations are complete. Every content-addressed RunSpec, COMPLETE marker, metric NPZ and
  artifact manifest rehashes; the codec floor remains `CODEC_FLOOR_PASS`. The independent
  completion validator reports 7 trainings, 14 evaluations, 8 primary effects and
  **30.0238/36.0 project GPU-hours**. The failed pre-access command-line attempt remains charged
  in that total.
- **Registered simultaneous family:** all four original-codec controls are positive and
  simultaneous-CI clean. Under the independently trained codec, the 896 reach effect is
  **+0.0521 [0.0374, 0.0669]**, the 1,152 interior effect is
  **+0.0439 [0.0310, 0.0569]**, and the near-matched package-shape effect is
  **+0.0112 [0.0039, 0.0184]**. The 2,432-token flat-fidelity inversion remains positive at
  **+0.0079**, but its simultaneous interval **[−0.0006, 0.0165]** includes zero. The frozen
  verdict is therefore **`CODEC_DIRECTIONALLY_ROBUST`**, not CI-clean and not codec-sensitive.
- **Corroborating endpoints:** the alternate-codec 2,432 comparison also favors flat on
  non-empty GT-foreground MSE (+0.0097 [0.0019, 0.0177]) and recall
  (+0.0266 [0.0088, 0.0452]); full-frame, background, union, precision/F1 and component-count
  intervals are mixed. These prespecified diagnostics qualify mechanism and metric sensitivity;
  they do not override the simultaneous primary category.
- **Codec response:** the alternate codec strengthens the 896 and 1,152 interior effects,
  while shrinking the package-shape and upper-inversion effects. Those effect changes are
  descriptive because codec was not replicated as an inferential unit. E7a supports
  whole-codec directional robustness conditional on predictor seed 0; it does not isolate latent
  scale, reconstruction fidelity, normalization or another representation component.
- **Operational recovery:** after the final evaluation, PowerShell again treated W&B's ordinary
  stderr status line as a terminating native error in the unwrapped analysis block. The
  deterministic result JSON had already been written, no scientific input changed, and no W&B
  analysis run existed. The same error-handling-only wrapper repair was applied; the immutable
  analysis reran to the identical SHA-256
  `3f113628babb58cd9adf5223f04a28cc7bb54b59a451b6f5f9aa4ac8d2dd7173`,
  uploaded once, and the independent validator passed. Canonical dispatch paths were normalized
  to repository-relative form without changing any codec/checkpoint hash.
- **Cloud:** the final W&B audit passes for all 27 group runs: three connectivity gates, one
  codec training, one codec-validity gate, seven predictor trainings, fourteen evaluations and
  one analysis. Every scientific job is finished and has its required cloud artifact.
- **Gate:** E7a is closed with no adaptive codec, seed, arm or endpoint. Proceed to the separately
  frozen, privileged E8a pose-indexed raw-memory reference after its own code/access/CPU/data
  gates; E7a supplies no authority to reopen E7c or add a third codec.

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

#### Intermediate audit E4b-I3 — 2026-07-13 14:14 Europe/Kyiv

- **Completed-cell acceptance:** `hier-long` seed 3 completed training, the frozen 100-clip ×
  152-frame rollout and cloud upload with exit code zero in **3.2963 project GPU-hours**, below
  its 4.5-hour cell cap. Final loss was 0.0438, fixed-grid h4 foreground MSE 0.0267 and rollout
  foreground MSE 0.1713 (h1–48 0.0992; h49–192 0.2046). The rollout lies inside the prior
  seed-0/1/2 range 0.1690–0.1848 and is close to seeds 1–2, so it is plausible training-seed
  variation rather than a regime shift.
- **Artifact and pairing gate:** PASS. RunSpec `e6ff5f95708c15ce` is COMPLETE and source-bound to
  `fa4906f`. All **19** manifest entries reproduce their byte counts and SHA-256 hashes. The NPZ
  contains exactly 15,200 unique `(clip,horizon)` records over 100 clips × 152 horizons; pairing
  fields and the registered foreground metric are complete and finite. NaNs occur only in
  auxiliary metrics whose populations are semantically undefined (for example non-empty-mask or
  matched-component endpoints). The checkpoint is 355,264,107 bytes. Its final diagnostic and
  long-rollout strip are coherent and show expected self-fed horizon degradation, not corruption.
- **Completed-cell convergence:** all 60 logged loss intervals are finite and smooth. Relative to
  the three legacy seeds, the average interval difference is +0.0027 and the maximum absolute
  difference is 0.0165 at the warm-up boundary. Final loss 0.0438 is inside the legacy
  0.0414–0.0444 range. The first-cell outcome therefore passes the frozen-regime plausibility
  check without using its result for selection.
- **Cloud acceptance:** W&B state is `finished` with 60 history rows, 15 validation grids, the
  final evaluation summary/media and a committed `training-run` artifact. W&B rejected only the
  optional per-chunk rollout-progress rows because their clip counters were lower than the 30k
  training step; the immutable NPZ, final summary, strip and artifact uploaded successfully, so
  this is a telemetry nuisance rather than missing scientific evidence.
- **Active cell and expected-versus-actual trend:** `strided-flat` seed 3 reached 10,000/30,000 at
  0.35–0.38 s/update and 6.26 GB measured peak. The 2k–6k training losses were above the legacy
  envelope after the full-LR transition, but the excess contracted from +0.0439 at 2k to +0.0041
  at 6k and +0.0068 relative to the legacy mean at 10k. More importantly, the fixed-grid h4
  trajectory is healthy: 0.2591 at 2k is within the legacy range, 0.0510 at 4k is essentially at
  the prior upper edge, and 0.0331/0.0287 at 6k/8k is better than all three prior seeds. This is a
  slower training-loss transition with good validation convergence, not a divergence or stall.
- **Active provenance/cloud state:** the 10k atomic checkpoint is present, format 2 and bound to
  RunSpec `7251715870f7fc9a`; it includes model/optimizer/EMA, Python/NumPy/Torch CPU+CUDA RNG,
  latent statistics and sampler state `examples_seen=160000`, `next_update=10001`. W&B is online
  and state `running`; its short local-to-cloud lag is below one 500-step interval. Controller PID
  22476 and trainer PID 20132 are alive and responsive in the expected parent chain.
- **GPU, budget and completion projection:** the audit sample showed 100% utilization,
  7,949/16,380 MiB board memory, 64°C and approximately 115.6 W with no WDDM spill. At current
  throughput the active cell should finish near 3.1–3.3 hours. Extrapolating measured diffusion
  and E4a regression rates places the eight-cell tranche near **25–28 GPU-hours**, approximately
  2026-07-14 11:00–14:00 Europe/Kyiv, with substantial margin below the 36-hour cap.
- **Scientific validity/decision:** no target, batch, LR, update, EMA, sampler, evaluation or data
  geometry changed, no retry occurred and no early result selected a branch. Seed 3's diffusion
  package effect remains unknown until the active rollout is paired with the accepted spatial
  cell. Continue unchanged; no stop, relaunch, tuning or budget trigger fires.

#### Intermediate audit E4b-I4 — 2026-07-13 18:11 Europe/Kyiv

- **Second-cell acceptance:** `strided-flat` seed 3 completed the frozen training/evaluation/upload
  path with exit code zero in **3.0668 project GPU-hours**, below its 4.5-hour cap. Final loss was
  0.0628, fixed-grid h4 foreground MSE 0.0216 and rollout foreground MSE 0.1638 (h1–48 0.0897;
  h49–192 0.1980). The rollout is inside the narrow legacy seed-0/1/2 range 0.1636–0.1652 and the
  final fixed-grid value is likewise inside the 0.0207–0.0219 range.
- **Second-cell artifact/cloud gate:** PASS. RunSpec `7251715870f7fc9a` reproduces its
  content-addressed directory identity and frozen source/configuration. All 19 manifest entries
  reproduce their byte counts and SHA-256 hashes. The rollout contains exactly 15,200 paired
  records, all registered foreground values are finite, and `clip`, `h`, `hard`, `dt` and `seen`
  align exactly with the accepted `hier-long` cell. W&B is `finished` with 60 history rows, 15
  validation grids, final summary/media and a committed training-run artifact.
- **Second-cell convergence assessment:** seed 3's training loss remained approximately 0.0075
  above the legacy-seed mean on average and ended at 0.0628, slightly above the prior 0.0558–0.0608
  range. This does not indicate regime drift: throughput/VRAM stayed at 0.35 s/update and 6.26 GB,
  the fixed-grid endpoint matched the legacy envelope, the immutable long rollout matched the
  legacy outcome envelope, and no non-finite value, stall or qualitative corruption occurred.
  The evidence supports a slower seed-specific training-loss path with normal generalization.
- **Provisional seed-3 diffusion pair:** applying the registered clip bootstrap to the aligned
  immutable pair gives `hier-long − strided-flat` = **+0.0075 [0.0037, 0.0114]**, so temporally
  sparse full-resolution packaging wins this seed. This continues the positive diffusion signs at
  seeds 0–2 (+0.0197, +0.0046, +0.0111). Full-frame and union error, precision, recall/F1,
  misses/extras and matched-component IoU also favor `strided-flat`. This is a within-tranche
  health result on the disclosed adaptive development pool, not a queue-selection event or final
  five-seed conclusion.
- **Active regression cell:** `reg-long` seed 3 reached 18,000/30,000 at 0.37–0.40 s/update and
  6.46 GB measured peak. Across 36 intervals its loss averages +0.0039 above the three legacy
  seeds; the largest deviation is +0.0293 at 1.5k, before the warm-up transition. After 2k the
  absolute excess is at most 0.0060 and the curve continues downward to 0.0218 without explosion,
  oscillatory growth or a plateau in training loss.
- **Active validation warning, bounded:** the 2k/4k fixed-grid values 0.2565/0.0510 match legacy
  seeds 1–2. Values at 6k–16k are modestly worse, peaking at +0.0048 above the legacy maximum at
  12k and narrowing to +0.0007 by 16k (0.0352 versus 0.0344). Legacy seed 0 did not preserve this
  fixed-grid series. The 16k qualitative grid remains aligned and recognizable. This is a real
  slower-validation signal worth tracking, but it neither predicts the 152-frame package effect
  nor meets a failure/intervention criterion.
- **Active provenance/cloud/process state:** the 18k checkpoint is format 2, RunSpec-bound and
  complete with model/optimizer/EMA, Python/NumPy/Torch CPU+CUDA RNG, latent statistics and
  sampler state `examples_seen=288000`, `next_update=18001`. W&B is online and `running` with 35
  history rows; the public summary lags the newest local validation grid by one interval. The
  controller and trainer chain is alive and responsive.
- **GPU/budget/projection:** the two accepted cells charge **6.3630 GPU-hours**. Including the
  active cell's elapsed ownership gives approximately 8.3 hours at the audit boundary. Six GPU
  samples ranged from 86–100% utilization, 8.21–8.25/16.38 GB board memory, 62–65°C and
  approximately 112–114 W; stable 0.38 s/update and fixed measured allocation show no WDDM spill.
  The active cell should finish near 19:25 Kyiv. The tranche remains projected at **25–28 total
  GPU-hours**, completing around 2026-07-14 11:00–14:00 Kyiv and below the 36-hour cap.
- **Registered decision state:** seed 3's head × package interaction is not estimable until both
  regression cells finish, and the robust-shift rule is not evaluated until all five training
  seeds are available. The frozen terminal rule still ends after seed 4 for every outcome.
  Continue unchanged; no stop, retry, tuning, budget or validity trigger fires.

#### Intermediate audit E4b-I5 — 2026-07-13 22:10 Europe/Kyiv

- **Third-cell acceptance:** `reg-long` seed 3 completed the frozen training, deterministic
  rollout and cloud-upload path with exit code zero in **3.1784 project GPU-hours**, below its
  4.5-hour cap. Final loss was 0.0168, fixed-grid h4 foreground MSE 0.0291 and rollout foreground
  MSE 0.1870 (h1–48 0.1133; h49–192 0.2210). The run is operationally valid, but its rollout is
  materially worse than the earlier seed-0/1/2 range 0.1503–0.1682 and its decoded sharpness
  0.0600 is below their 0.0741–0.0865 range.
- **Third-cell heterogeneity interpretation:** the warning seen during training was informative.
  Across all 60 intervals, seed-3 loss averaged +0.0035 above the three legacy seeds and ended
  above their 0.0130–0.0152 range. Its final fixed-grid value was also worse than preserved seeds
  1–2 (0.0242/0.0268). The immutable rollout strip shows coherent early predictions followed by
  excess fading/missed content at long horizons, not file corruption or model explosion. Treat
  this as genuine training-seed variation in spatial regression—the uncertainty E4b was designed
  to measure—not as a failed or selectable run.
- **Third-cell artifact/cloud gate:** PASS. RunSpec `64efc369b0e6c931` reproduces its directory
  identity and frozen source/configuration. All 19 manifest entries reproduce their byte counts
  and SHA-256 hashes. The NPZ has 15,200 finite registered foreground records and aligns exactly
  with the accepted diffusion cells on `clip`, `h`, `hard`, `dt` and `seen`. W&B is `finished`
  with 60 history rows, 15 validation grids, final summary/media and a committed training-run
  artifact.
- **Active temporal-regression trend:** `strided-flat-reg` seed 3 reached 27,500/30,000 locally
  (27,000 in the W&B history snapshot) at 0.35–0.36 s/update and 6.26 GB measured peak. Across 54
  cloud-visible intervals, its loss averages −0.0008 relative to the three legacy seeds and the
  maximum deviation is 0.0143 at 1.5k. At 27k, loss 0.0151 is within the legacy 0.0149–0.0190
  range. There is no non-finite value, stall, late growth or numerical-regime warning.
- **Active validation trajectory:** the 4k grid was temporarily worse than every legacy seed
  (0.0595 versus 0.0520–0.0532). It re-entered or improved on the legacy envelope thereafter:
  0.0411 at 8k, 0.0401 at 10k, 0.0277/0.0252 at 20k/22k and 0.0228/0.0218 at 24k/26k. Unlike
  seed 2's late 0.037–0.040 plateau, seed 3 has reached the seed-0/1 neighborhood. The 26k grid is
  aligned and visually coherent. Only the frozen 152-frame rollout can determine its package
  effect.
- **Active provenance/cloud/process state:** the active RunSpec hash reproduces
  `cde73e5f05f5acfc` and is source-bound to `fa4906f`. The 27k atomic checkpoint is format 2 with
  model/optimizer/EMA, all Python/NumPy/Torch CPU+CUDA RNG state, latent statistics and sampler
  state `examples_seen=432000`, `next_update=27001`. W&B is online and `running` with 54 history
  rows and validations through 26k. The controller and trainer chain remains alive and responsive.
- **GPU/budget/projection:** three accepted cells charge **9.5414 GPU-hours**. Including active
  ownership gives approximately 12.3 hours at the audit boundary. Six samples were 99–100%
  utilized at 8,175/16,380 MiB board memory, 65–66°C and approximately 119 W; throughput and
  measured allocation are stable with no WDDM spill. Cell 4 should complete around 22:25 Kyiv,
  placing the full seed-3 block near 12.5 hours. The tranche remains projected at **approximately
  25–27 total GPU-hours**, completing around 2026-07-14 11:00–13:00 Kyiv below the 36-hour cap.
- **Registered decision state:** the poor `reg-long` seed-3 outcome is retained without
  checkpoint selection or extension. Seed 3's regression package effect and head interaction
  remain unknown until the active immutable rollout exists; the five-seed t interval and robust
  all-negative rule remain unavailable until seed 4 completes. Continue unchanged; no stop,
  retry, tuning, budget or validity trigger fires.

#### Intermediate audit E4b-I6 — 2026-07-14 02:12 Europe/Kyiv

- **Fourth-cell acceptance:** `strided-flat-reg` seed 3 completed with exit code zero in **2.9710
  project GPU-hours**, below its 4.5-hour cap. Final loss was 0.0148, fixed-grid h4 foreground MSE
  0.0219 and rollout foreground MSE 0.1622 (h1–48 0.0882; h49–192 0.1963). The rollout is within
  the earlier temporal-regression seed range and sharply better than seed 3's accepted
  `reg-long` result. Its W&B run is `finished` with 60 history rows, 15 grids, final summary/media
  and a committed training-run artifact.
- **Seed-3 four-cell integrity:** PASS. Every one of the four RunSpecs reproduces its
  content-addressed identity and COMPLETE marker; all 76 manifest entries reproduce byte counts
  and SHA-256 hashes. All four NPZs contain 15,200 finite registered foreground records and align
  exactly on `clip`, `h`, `hard`, `dt` and `seen`. There was no retry, checkpoint selection,
  extension or data/evaluation geometry change.
- **Seed-3 registered effects:** the registered clip bootstrap gives:

  | Effect (`spatial − temporal`) | Estimate | 95% conditional CI | Interpretation |
  |---|---:|---:|---|
  | Diffusion package | +0.0075 | [+0.0037, +0.0114] | temporal package better |
  | Regression package | +0.0248 | [+0.0181, +0.0323] | temporal package better |
  | Head × package interaction | **+0.0173** | **[+0.0106, +0.0245]** | regression shifts toward temporal, opposite to seeds 0–2 |

- **Preregistered robustness verdict already determined:** the robust negative head-shift rule
  requires all five interactions to be negative *and* the across-seed t interval below zero.
  Seed 3 is conclusively positive, so this conjunctive criterion cannot pass regardless of seed
  4. The seeds-0–2 finding remains accurate but must be described as three-seed-local, not a
  stable head-induced shift. Regression package signs through seed 3 are `+ / − / − / +`, so a
  stable regression package preference is also ruled out. Seed 4 remains mandatory to estimate
  the five-seed mean, SD and t interval and to complete the frozen terminal design.
- **Fifth-cell acceptance:** `hier-long` seed 4 completed with exit code zero in **3.2961
  GPU-hours**. Final loss was 0.0447, fixed-grid h4 foreground MSE 0.0361 and rollout foreground
  MSE 0.1682 (h1–48 0.0966; h49–192 0.2012). Training loss averaged +0.0038 relative to seeds
  0–3 and the final fixed-grid value is modestly worse, but the long-rollout result is inside the
  prior outcome envelope and the strip is coherent. All 19 manifest artifacts verify, pairing
  fields are complete, and W&B is `finished` with a committed artifact. This is accepted seed
  variation, not operational drift.
- **Active sixth cell:** `strided-flat` seed 4 reached 4,500/30,000 at 0.35 s/update and 6.26 GB
  measured peak. Loss was above all prior seeds at 1.5k (1.0217 versus 0.9642–0.9893), then crossed
  below the prior envelope at 3k/3.5k and returned inside it at 4k. Fixed-grid h4 MSE is healthy:
  0.2404 at 2k lies inside the prior range and 0.0458 at 4k is better than every prior seed. This
  is a transient warm-up-shape difference, not divergence.
- **Active provenance/cloud/process state:** RunSpec `b9e170d46d96db84` reproduces its identity
  and frozen source commit. The 4k checkpoint is format 2 with model/optimizer/EMA, complete
  Python/NumPy/Torch CPU+CUDA RNG, latent statistics and sampler state `examples_seen=64000`,
  `next_update=4001`. W&B is online and `running` with eight history rows and two validation
  grids. The controller/trainer chain is alive and responsive.
- **GPU/budget/projection:** five accepted cells charge **15.8085 GPU-hours**; including active
  ownership gives approximately 16.3 hours. Six samples were 95–100% utilized at about
  8.05–8.06/16.38 GB board memory, 63–65°C and 114–115 W. Stable throughput/allocation show no
  WDDM spill. The current cell should finish near 04:45 Kyiv; the full tranche remains projected
  at **approximately 24.5–26 GPU-hours**, completing around 2026-07-14 10:45–12:00 Kyiv and below
  the 36-hour cap.
- **Decision:** continue the three remaining seed-4 cells unchanged. They can refine the
  five-seed estimand but cannot rescue the failed all-negative robustness criterion. No stop,
  retry, tuning, budget or validity trigger fires, and no subsequent study is authorized here.

#### Intermediate audit E4b-I7 — 2026-07-14 06:13 Europe/Kyiv

- **Sixth-cell acceptance:** `strided-flat` seed 4 completed with exit code zero in **3.0591
  project GPU-hours**, below its 4.5-hour cap. Final loss was 0.0616, fixed-grid h4 foreground MSE
  0.0210 and rollout foreground MSE 0.1622. Its content-addressed RunSpec
  `b9e170d46d96db84` and all 19 manifest entries independently reproduce their byte counts and
  SHA-256 hashes; the COMPLETE marker is valid. W&B is `finished` with 60 history rows, 15
  validation grids, final media/summary and a committed run artifact. There was no retry,
  extension or checkpoint selection.
- **Seed-4 diffusion pair:** the `hier-long` and `strided-flat` rollouts align exactly on all
  15,200 finite records (`clip`, `h`, `hard`, `dt` and `seen`). The registered spatial-minus-
  temporal package effect is **+0.0060 [0.0027, 0.0092]** by paired clip bootstrap, again favoring
  the temporal package.
- **Five-seed diffusion result:** seed-level effects for seeds 0–4 are `+0.0197`, `+0.0046`,
  `+0.0111`, `+0.0075` and `+0.0060`. All five favor temporal packaging. Their mean is
  **+0.0098**, sample SD **0.0060**, with a five-seed t interval **[+0.0023, +0.0173]**. This is
  now a stable diffusion package result across the five tested training seeds. It remains
  development-pool evidence and is kept distinct from the failed cross-head robustness claim;
  it does not restore untouched confirmatory status to this adaptive pool.
- **Active seventh-cell training:** `reg-long` seed 4 reached 13,500/30,000 at 0.37–0.38
  s/update with a 6.46 GB measured peak. Its unusually fast 1.5k transition (loss 0.4558 versus
  0.5023–0.5590 previously) is seed-specific warm-up shape, not sustained numerical drift. Across
  the 27 logged 500-update intervals its mean loss difference from seeds 0–3 is -0.0043; after
  warm-up it is mostly inside the prior envelope, finite and without late growth or a stall.
- **Active validation and qualitative health:** fixed-grid h4 foreground MSE is 0.2525, 0.0546,
  0.0446, 0.0424, 0.0404 and 0.0393 at 2k–12k. The 4k/6k values are mildly above the prior
  maximum, but the trajectory re-enters the seed-3 envelope by 8k–12k. The frozen 12k grid is
  aligned and visually coherent, including foreground placement and legitimate empty rows. This
  bounded early delay cannot determine the long-rollout package effect and does not justify
  intervention.
- **Active provenance/cloud/process state:** RunSpec `a50e6c1b15304ce6` reproduces its canonical
  identity and frozen source commit `fa4906f`. The 13k atomic checkpoint is format 2 with
  model/optimizer/EMA, Python/NumPy/Torch CPU+CUDA RNG state, latent statistics and sampler state
  `examples_seen=208000`, `next_update=13001`. W&B is online and `running` with history through
  13k and validation through 12k. Controller PID 22476, environment shim PID 18136 and trainer PID
  6980 are alive and responsive.
- **GPU/budget/projection:** six accepted cells charge **18.8676 GPU-hours**; including active
  ownership gives approximately **20.3 hours** at the audit boundary. The GPU is at 100%
  utilization with 8,324/16,380 MiB board memory, 63°C and 113 W; throughput and allocation remain
  stable with no WDDM spill. The active cell should finish near 08:00 Kyiv and the final
  `strided-flat-reg` cell near 11:00 Kyiv. The tranche is projected at approximately **25.0
  GPU-hours**, safely below its 36-hour cap.
- **Decision:** continue the final two frozen cells unchanged. The active and final regression
  rollouts are still required for the seed-4 and five-seed interaction estimates, even though the
  preregistered all-negative robustness rule already cannot pass. No stop, retry, tuning, budget
  or estimand-validity trigger fires, and no later study is authorized by this audit.

#### Intermediate audit E4b-I8 — 2026-07-14 10:15 Europe/Kyiv

- **Seventh-cell acceptance:** `reg-long` seed 4 completed with exit code zero in **3.1713 project
  GPU-hours**, below its 4.5-hour cap. Final loss was 0.0141, fixed-grid h4 foreground MSE 0.0248
  and rollout foreground MSE **0.1579** (h1–48 0.0887; h49–192 0.1899). The rollout is inside the
  prior `reg-long` seed range of 0.1503–0.1870 and its qualitative strip shows the expected
  mixture of correct persistence/re-entry and long-horizon failures rather than collapse.
- **Seventh-cell integrity/cloud:** all 15,200 foreground records are finite and its pairing
  fields align exactly with the preceding factorial cells. RunSpec `a50e6c1b15304ce6`, COMPLETE
  marker and all 19 manifest artifacts independently reproduce their identities, byte counts and
  SHA-256 hashes. W&B is `finished` with 60 history rows, 15 validation grids, final summary/media,
  a committed 355.8 MB training-run artifact and committed history artifact. There was no retry,
  resume, extension or checkpoint selection.
- **Active eighth-cell loss trajectory:** the final `strided-flat-reg` seed 4 cell reached
  23,500/30,000. Across the first 45 complete 500-update intervals, its mean loss difference from
  seeds 0–3 is -0.0038; 15 intervals lie outside the prior min–max envelope, driven principally by
  a much faster warm-up transition at 1.5k (0.3732 versus 0.4554–0.5408). After 3.5k, 34/39
  intervals are inside the prior envelope. The isolated highs at 6k, 9.5k, 19k and 22.5k are
  followed by recovery; loss is 0.0198 at 23.5k. There is no non-finite value, monotonic growth,
  optimizer stall or numerical-regime break.
- **Active validation/qualitative trajectory:** fixed-grid h4 foreground MSE is 0.2550, 0.0535,
  0.0436, 0.0427, 0.0424, 0.0412 and 0.0418 at 2k–14k, all within the four-seed envelope. It rises
  above that envelope at 16k/18k (0.0426/0.0406), then improves to 0.0382/0.0355 at 20k/22k and
  re-enters the prior range. The 22k grid is aligned and visually coherent. The temporary
  validation delay is real but resolving; it does not predict the immutable long-rollout result
  closely enough to infer the seed-4 package sign before evaluation.
- **Active provenance/checkpoint/cloud:** the file, checkpoint and registry identities all equal
  `88a6f1f4707eb85c` and bind to source commit `fa4906f`. The audited 23k format-2 checkpoint has
  model/optimizer/EMA, Python/NumPy/Torch CPU+CUDA RNG state and sampler state
  `examples_seen=368000`, `next_update=23001`. W&B is online and `running` with 45 history rows and
  validation through 22k; no remote artifact is expected before completion. The controller and
  trainer process chain is alive and responsive.
- **GPU/budget/projection:** seven accepted cells charge **22.0388 GPU-hours**; including active
  ownership gives approximately **24.3 hours**. Six telemetry samples were 94–100% utilized at
  8.06–8.15/16.38 GB board memory, 63–65°C and 115–116 W. Throughput is predominantly 0.35
  s/update, with 0.37 s/update in the two latest intervals but no WDDM spill or sustained
  slowdown. Training, evaluation and upload should finish around **10:55–11:05 Kyiv**, with the
  tranche near **25.0 GPU-hours**, safely below the 36-hour cap.
- **Decision:** continue the final cell unchanged. Its rollout is required to determine the
  seed-4 regression package effect, the terminal five-seed interaction interval and regression
  sign counts. The failed all-negative rule is a scoped negative/boundary result, not a blocker
  for the primary diffusion claim. No stop, retry, tuning, budget or estimand-validity trigger
  fires, and no subsequent experiment is authorized before final E4b validation and integration.

#### Final audit E4b-I9 — 2026-07-14 14:12 Europe/Kyiv

- **Final-cell acceptance:** `strided-flat-reg` seed 4 completed with exit code zero in **2.9534
  project GPU-hours**, below its 4.5-hour cap. Final loss was 0.0177, fixed-grid h4 foreground MSE
  0.0234 and rollout foreground MSE **0.1745** (h1–48 0.1038; h49–192 0.2071). Its rollout strip
  is coherent and exhibits ordinary long-horizon errors rather than collapse.
- **Final-cell training health:** all 60 loss intervals are finite. Relative to seeds 0–3, mean
  loss difference is −0.0025; 18/60 intervals fall outside the prior min–max envelope, principally
  because seed 4 converged much faster during warm-up. After 3.5k, 42/54 intervals are inside the
  envelope; isolated late highs recover without growth or stall. Validation was briefly above
  the prior range at 16k/18k, then improved monotonically enough to finish at 0.0234 inside the
  prior range. Mean throughput was 0.351 s/update (range 0.349–0.372) at 6.26 GB measured peak;
  no numerical drift or WDDM spill occurred.
- **Seed-4 paired result:** all four seed-4 cells align exactly on 15,200 finite records. The
  regression spatial-minus-temporal effect is **−0.0166 [−0.0229, −0.0108]**, favoring spatial
  coarsening for this seed. After subtracting the diffusion effect, the head × package interaction
  is **−0.0226 [−0.0297, −0.0160]**.
- **Registered five-seed result:**

  | Estimand | Seed effects | Mean | SD | Across-seed t95 |
  |---|---|---:|---:|---:|
  | Diffusion temporal advantage | `+0.0197 / +0.0046 / +0.0111 / +0.0075 / +0.0060` | +0.0098 | 0.0060 | **[+0.0023, +0.0173]** |
  | Regression temporal advantage | `+0.0107 / −0.0104 / −0.0208 / +0.0248 / −0.0166` | −0.0024 | 0.0195 | [−0.0266, +0.0217] |
  | Head × package interaction | `−0.0089 / −0.0150 / −0.0319 / +0.0173 / −0.0226` | −0.0122 | 0.0186 | [−0.0353, +0.0109] |

  Diffusion favors temporal packaging on all five tested training seeds. Regression has two
  temporal and three spatial material winners. Because the interaction changes sign and its
  training-seed interval crosses zero, the preregistered robust negative head-shift rule is
  **false**. The E4a three-seed pattern remains historically accurate but is not a robust
  directional head law.
- **Immutable/cloud completion:** independent validation status is PASS. The frozen config SHA-256
  is `91546ce4499c580dffe7340dd8a1401c89254c3cb88afd324d44a3faeb72610d`; result SHA-256 is
  `eb917d948d852d022b872532b826bdb20fc1f17ecfddb55a76446f1821bc9e89`. All eight new RunSpec
  identities, COMPLETE markers and **152/152 manifest artifacts** reproduce; all 20 factorial
  inputs align on the registered geometry. W&B shows eight finished scientific-training runs and
  one finished factorial-analysis run, each with a committed primary artifact. Rollout chunk
  progress produced non-monotonic-step warnings and was omitted from W&B history, but final
  metrics, media, summaries and immutable artifacts are present; this is a telemetry-maintenance
  issue with no effect on the estimand.
- **Budget/terminal state:** E4b consumed **24.9922 project GPU-hours** against 36 authorized, with
  no failed attempt, resume or retry. The controller exited after `E4B_FACTORIAL_DONE`; the active
  registry is terminal `complete`, no project experiment process remains, and the GPU is idle for
  this project.
- **Claim integration:** `docs/CLAIM_LEDGER.md`, the canonical article sources, dissertation
  Chapter 4 and `docs/SPATIAL_FACTORIAL.md` now separate the five-seed-stable diffusion result
  from regression/cross-head heterogeneity. The directional head-shift claim is demoted, not
  hidden; E4b is terminal and authorizes no seeds 5+.
- **Next decision-tree gate:** E3a remains deferred because current prose consistently calls B1 a
  whole-configuration effect. The next title-level scientific question is E5a trust replication.
  Only CPU preparation is now in scope: freeze the decisive H-grid3 pair, seeds 1–2, estimand,
  artifact reuse, runtime probe and stop rule; then run design/provenance/oracle gates. This audit
  grants no new GPU launch authority, and E5b remains outcome-contingent on E5a replication.

#### Intermediate audit E5-I0 — 2026-07-14 15:30 Europe/Kyiv

- **Authorization received:** the author explicitly authorized continuation under the agreed plan
  and allowed GPU use. E5a is the next decision-tree tranche; E5b remains separately gated.
- **Frozen cells/order:** `d1-56-diff` then `d0-diff-long` at seed 1, followed by the same pair at
  seed 2. Every cell preserves the H-grid3 B8×40k, LR 3e-4, 2k warmup, EMA 0.999, bf16 and NFE-16
  regime. The configuration comparison intentionally differs in reach, spatial fidelity, token
  count and the legacy coupled coarse target interface; it is not described as a pure stride or
  equal-token intervention.
- **Registered estimand:** effect is comparator foreground error minus long-reach-hierarchy error
  on non-empty GT-foreground records. The two primary windows are early h1–24 (required positive)
  and late h61–256 (required negative), with paired 10,000-replicate clip-cluster intervals. A new
  seed is CI-clean only when the early lower bound is above zero and the late upper bound is below
  zero. Both new seeds must pass for `REPLICATED`; secondary HARD/reach rows cannot rescue failure.
- **Seed-0 reconstruction:** the frozen legacy artifacts reproduce early **+0.02042
  [+0.01334,+0.02795]** on 1,746 records and late **−0.00636 [−0.00955,−0.00320]** on 13,781
  records. The separately measured GT-fed addressability result remains an external prerequisite,
  not an E5a endpoint. Thus E5a tests replication of self-fed benefit-to-inversion, not a null or
  equivalence claim for long-range retrieval.
- **Provenance/code preparation:** RunSpec identity now hashes optional position arrays because
  those arrays determine H-grid geometric labels. Dataset and provenance tags are explicit in
  W&B. The fail-closed design validator pins all six data arrays, the AE, single-step evidence,
  seed-0 checkpoints/metrics/oracles, interfaces, clip pool, budget and parity sources. The
  independent completion validator re-hashes manifests, checks the GPU ledger and recomputes the
  registered decision.
- **Runtime/budget:** measured seed-0 wall time was 2.491 h for the long-reach arm and 6.940 h for
  the contiguous fine arm, projecting approximately 18.9 GPU-hours for two new seeds. Authorized
  per-cell maxima are 4.5/9.5 hours, totaling **28.0 GPU-hours**. Batch or LR changes to fill VRAM
  are prohibited because they would change the replication regime; the measured 3.64/7.39 GiB
  allocations are accepted.
- **CPU status:** JSON/schema, compile, PowerShell parse, targeted provenance tests and the full
  suite pass (**48 tests**). The only expected pre-attestation failure is that H3b correctly detects
  the just-edited provenance wrapper/test hashes. Commit the implementation, regenerate H3b, bind
  protocol/design attestations, run W&B connectivity and `-CheckOnly`, then launch. No GPU process
  has started at this audit.

#### Intermediate audit E5-I1 — 2026-07-14 15:28 Europe/Kyiv

- **All gates passed:** H3b was regenerated from committed source (`7c8bf86`) and again passed exact
  batch/update/model/optimizer/EMA and resume parity plus the full 48-test suite. The E5a design
  validator passed with zero failures; protocol and report are hash-attested. A complete no-GPU
  launcher rehearsal passed authorization, data/evidence hashes, seed-0 reconstruction and W&B
  connectivity through `C:\Users\leoni\_netrc`.
- **Launch:** controller PID 2988 started from source commit `6a6332c`. Cell 1/4 is
  `d1-56-diff`, seed 1, RunSpec `c82816c7f0818744`, W&B run
  `https://wandb.ai/leo27heady/hgrid/runs/e5a-trust-replication-c82816c7f0818744`.
  E5b is not part of this controller and cannot start automatically.
- **Startup health:** H-grid3 train/test encoding completed in 23.8/6.0 seconds, matching the
  22.0/5.7-second seed-0 reference closely. Latent standard deviations reproduce exactly
  (coarse 0.1363, fine 0.1921), the split remains 900/300 train-validation and 150/150 test split,
  and the model reports 22.26M parameters. Initial board usage is approximately 4.9 GB at full GPU
  utilization; no competing trainer, numerical warning beyond the known bf16 LayerNorm fallback,
  or WDDM spill is present.
- **Monitoring:** heartbeat `e5a-four-hour-health-audit` is active every four hours in this same
  task. Each audit compares loss, validation, throughput and VRAM with the same-arm seed-0 trace,
  checks content-addressed checkpoints/cloud artifacts and GPU-hour headroom, and updates this
  ledger without altering the frozen regime.

#### Intermediate audit E5-I2 — 2026-07-14 19:31 Europe/Kyiv

- **Controller/current state:** controller PID 2988 is healthy. Cell 1/4, `d1-56-diff` seed 1,
  completed normally in **2.4362 project GPU-hours** versus 2.4914 hours at seed 0 and 4.5 hours
  authorized. Cell 2/4, `d0-diff-long` seed 1, is active at approximately 11k/40k under RunSpec
  `25bc12c6a4bd3b71`; its RNG-complete checkpoint is valid at update 11,000 with Python, NumPy,
  CPU and CUDA RNG state plus the correct next-update sampler state.
- **Completed hierarchy training versus seed 0:** all 80 logged 500-update intervals are finite.
  Mean observed throughput is **0.1969 s/update** versus 0.2011, maximum 0.2060; peak allocated
  VRAM is exactly 3.635 GiB. Seed-1 training loss is on average 1.128× seed 0 and ends at 0.09988
  versus 0.0847. This is not accompanied by validation degradation: across the twenty fixed-grid
  reads, mean h4 foreground MSE is **0.05129 versus 0.05108**, with overlapping ranges
  0.0330–0.1803 versus 0.0309–0.1752; the final values are 0.0380 versus 0.0343. The trace is
  decreasing and stable, with no NaN, stall, restart or spill.
- **Completed hierarchy rollout/artifacts:** evaluation contains exactly **25,600 records**, 100
  clips 150–249 and horizons 1–256, paired labels and every complete-v2 metric. Primary and
  false-positive-sensitive arrays are finite where defined; matched-component quantities are
  finite on their 8,384 scorable records rather than through survivorship intersection. Mean
  foreground MSE is 0.22414 versus 0.22771 for the seed-0 hierarchy; h1–48 is 0.19309, h49–192
  0.23348, h193+ 0.22640 and answerable HARD dt>48 0.31852. These single-arm values are health
  checks only—the registered E5a effect remains unreadable until its seed-matched comparator lands.
  The content-addressed COMPLETE marker and all **24/24 manifest artifacts** independently hash,
  and W&B is terminal `finished` with a 357.2 MB `training-run` artifact under aliases `latest`
  and `d1-56-diff-s1`.
- **Active comparator versus seed 0:** through 10.5k common updates, loss is only 1.013× the seed-0
  trajectory; at 10k it is 0.12651 versus 0.1330. The five fixed-grid values through 10k average
  0.04702 versus 0.04512 and end at **0.00852 versus 0.0085**. Mean throughput is 0.5277 s/update
  versus 0.5495 over the same early interval, peak allocated VRAM is 7.386 GiB and the latest
  observed interval remains normal. Board telemetry is 8.75/16.38 GB, 97–100% compute,
  approximately 67–69 °C and 120 W. No WDDM spill, competing compute process or numerical anomaly
  is present; W&B is live and current.
- **Projection/budget:** cell 2 should finish near **2026-07-15 00:00 Kyiv** if 0.528 s/update and
  the measured approximately 0.22-hour rollout persist. Reusing observed cell times projects seed-2
  hierarchy completion near 02:30 and terminal E5a analysis/validation near **08:30–09:00 Kyiv**.
  Expected total is approximately **17.1 GPU-hours** (conservatively still about 19 hours), leaving
  roughly 9–11 hours below the frozen 28-hour maximum. The estimand, source commit, batch/LR/update/
  EMA/NFE geometry and clip pool remain unchanged.
- **Decision:** continue unchanged. The UTF-16 controller log and W&B rollout-step warnings are
  operational formatting/telemetry details already seen in E4b; training histories, summaries,
  immutable artifacts and the scientific records are intact. No intervention or E5b authorization
  is justified.

#### Intermediate audit E5-I3 — 2026-07-14 23:31 Europe/Kyiv

- **Controller/current state:** controller PID 2988 and the sole project trainer PID 2972 are
  healthy. Cell 2/4, `d0-diff-long` seed 1, reached **38,000/40,000** updates under the unchanged
  RunSpec `25bc12c6a4bd3b71`. The update-38,000 checkpoint is readable and provenance-complete:
  format version 2, the expected RunSpec hash, all Python/NumPy/CPU/CUDA RNG states, four latent
  statistics tensors, 304,000 examples seen and sampler next update 38,001. There is no restart,
  resume, process conflict or geometry change.
- **Training versus the matching seed-0 arm:** all 76 available 500-update intervals are finite.
  Seed-1 mean loss is **1.021×** seed 0 over the common trace and **1.038×** over the last 10k
  updates; the current 37.5k cloud value is 0.08273 versus 0.0736. This small seed-level offset is
  not mirrored by validation deterioration. The eighteen fixed-grid reads average **0.01893 versus
  0.01842**, range 0.00680–0.16128 versus 0.0069–0.1640, and the latest value is 0.00790 versus
  0.0071. Loss and validation remain decreasing, bounded and consistent with ordinary training-seed
  variation.
- **Throughput/GPU health:** mean observed throughput is **0.5283 s/update** versus 0.5727 for the
  seed-0 run, with a tight 0.5239–0.5660-second interval and no analogue of the legacy 1.12-second
  WDDM stall. Peak allocated VRAM remains exactly 7.386 GiB. Board telemetry is 8.87/16.38 GB,
  100% utilization, approximately 68 °C and 120 W; the trainer is the only project compute
  process and there is no spill or numerical warning. W&B is live through `C:\Users\leoni\_netrc`,
  reports the correct run as `running`, and appropriately has no terminal training artifact before
  rollout/completion.
- **Expected versus actual:** the preceding audit projected the end of cell 2 near 00:00 Kyiv. At
  38k, approximately 17.6 minutes of training plus the previously measured roughly 0.22-hour
  rollout/upload remain, so the projection is holding. Reusing measured cell durations places
  seed-2 hierarchy completion near 02:30–02:45 and terminal E5a validation near **08:30–09:15 Kyiv**
  on 2026-07-15. Project GPU ownership consumed by this point is approximately **8.0 hours**;
  expected terminal consumption remains about **17.1 hours** (conservatively approximately 19),
  comfortably inside the registered 28-hour maximum.
- **Scientific gate:** the seed-1 hierarchy artifact is valid, but neither the registered early
  h1–24 effect nor late h61–256 effect is evaluated until this seed-matched comparator produces its
  complete 25,600-record rollout. Single-arm health values cannot trigger the two-window decision.
  The estimand, paired clip pool, batch/LR/update/EMA/NFE regime and source commit remain intact.
- **Decision:** continue unchanged. No intervention, frozen-regime modification or E5b
  authorization is justified. The next audit should first validate the completed seed-1 paired
  effects and content-addressed cloud artifacts, then examine the active seed-2 cell.

#### Restart recovery E5-I4 — 2026-07-15 08:00 Europe/Kyiv

- **Restart boundary:** Windows initiated a planned servicing restart (system event 1074 at
  01:44:11 Kyiv), shut down at 01:45:58 and restarted twice during servicing. The E5a controller
  and trainer are absent after boot, the GPU is idle, and no competing project process exists.
  This is an infrastructure interruption rather than a numerical or scientific failure.
- **Completed seed-1 comparator:** `d0-diff-long` seed 1 finished normally before the restart in
  **6.4141 GPU-hours**, versus 6.940 hours for seed 0 and 9.5 authorized. Its rollout contains the
  required 25,600 paired records and its 24-artifact manifest independently hashes under RunSpec
  `25bc12c6a4bd3b71` (manifest SHA-256
  `beaf29a619e832fb7cc900e949161ed55e7fd6914cabc4c92b70d6a8768a6b7d`). W&B is terminal
  `finished` with the content-addressed 356.1 MB `training-run` artifact.
- **Registered seed-1 result:** early h1–24 remains positive at **+0.00816
  [+0.00350,+0.01329]**, but late h61–256 is also positive at **+0.00745
  [+0.00377,+0.01123]** rather than negative. Seed 1 therefore fails both the directional and
  CI-clean early-benefit-to-late-inversion boundary. This already precludes the registered
  two-new-seed `REPLICATED` verdict, but it does not authorize stopping or reinterpretation:
  seed 2 remains required by the frozen terminal rule and will determine whether the outcome is
  `HETEROGENEOUS_PARTIAL` or `NOT_REPLICATED`.
- **Interrupted seed-2 hierarchy:** cell 3/4, `d1-56-diff` seed 2, reached update **26,000** under
  RunSpec `f8fc583e62c6ea0a`. The 356,345,643-byte checkpoint hashes to
  `013484c06f69b9d5e6831401d99929731214fc9977509537ac191e3ecddd1974` and passes native format-v2
  loading: model/optimizer/EMA state, four latent-statistics tensors, Python/NumPy/CPU/CUDA RNG,
  208,000 examples seen and next update 26,001 are present and consistent. W&B correctly marks the
  interrupted attempt `crashed`; there is no terminal artifact yet.
- **Gates and accounting:** the authorization gate and frozen design validator pass with zero
  failures, including all pinned data, AE, prior evidence, arm definitions and implementation
  hashes. The interrupted attempt is conservatively charged from controller start through the
  recorded shutdown: **1.476890 hours**, not merely the 1.414 cloud runtime. Completed plus
  interrupted ownership is now **10.3272 hours**. Cell 3 retains 3.0231 authorized hours, ample for
  updates 26,001–40,000 and rollout; expected terminal E5a consumption remains below 19 hours and
  below the frozen 28-hour maximum.
- **Recovery decision:** commit this immutable interruption row and audit, rerun the launcher's
  CPU/W&B `-Resume -CheckOnly` path from a clean worktree, then resume the same RunSpec. The resume
  may restore the exact RNG/sampler trajectory and continue the W&B run; it may not change batch,
  LR, updates, EMA, NFE, clip geometry or source. No E5b launch is authorized.

#### Attested resume E5-I5 — 2026-07-15 08:07 Europe/Kyiv

- **Fail-closed probe:** the first five-second resume attempt stopped before data loading or any GPU
  update because `ArtifactStore` compared the JSON-round-tripped `loss_weights` list with the
  identical in-memory tuple using Python container equality. The content identity remained
  `f8fc583e62c6ea0a`; this was a provenance-wrapper false collision, not a checkpoint or numerical
  mismatch. The failed attempt is explicitly charged **0.001482 GPU-hours** in the immutable
  ledger.
- **Minimal repair:** partial-store reopening now compares canonical JSON representations. Two new
  tests prove that tuple/list-equivalent specifications reopen while a real scientific-field
  change still fails closed. No model, data, batch, target, loss, optimizer, EMA, RNG, sampler,
  precision, NFE or evaluation code changed. The repair is commit `1e0aee0`; the original RunSpec
  continues to name source commit `6a6332c` so its content identity is preserved.
- **Numerical and governance gates:** the mandatory H3b suite passes all four parity tests and the
  full CPU suite passes **50/50 tests**, including uninterrupted-versus-resumed RNG trajectory.
  The recovery subject binds the exact checkpoint, new H3b report and every scientific file hash;
  its hash-bound marker is committed at `581d268`. The refreshed E5a design validator again passes
  all data, AE, prior-evidence, implementation, arm and budget checks and is re-attested for future
  recovery.
- **Live resume:** controller PID 19032 and trainer PID 17188 started at 08:05 Kyiv. W&B reopened
  the same run `e5a-trust-replication-f8fc583e62c6ea0a`; both datasets reproduce their latent
  standard deviations exactly, the native loader reports checkpoint update 26,000, and training
  begins at **26,001/40,000** with RunSpec `f8fc583e62c6ea0a`. The frozen batch 8, LR 3e-4, warmup
  2k, EMA 0.999, bf16, NFE-16 and rollout geometry are unchanged. The first post-resume interval
  reaches 26,500 with loss **0.1083**, 0.21 s/update and 3.63 GiB allocated, continuous with the
  pre-restart 26,000 loss 0.1071 and free of numerical or throughput discontinuity.
- **Budget/projection:** charged ownership before this successful resume is **10.3287 GPU-hours**;
  cell 3 retains 3.0216 authorized hours. At the observed approximately 0.20 s/update, its remaining
  training plus rollout should finish near 09:00–09:15 Kyiv. Reusing the measured seed-1 comparator
  duration projects terminal E5a analysis near **15:30–16:00 Kyiv**, at roughly 17.7–19 total
  GPU-hours versus 28 authorized.
- **Decision:** continue the exact resumed trajectory. Seed 1's failed late-inversion direction is
  retained as the registered result; seed 2 remains necessary for the terminal heterogeneous-partial
  versus not-replicated decision. No E5b action is authorized.

#### Intermediate audit E5-I6 — 2026-07-15 11:56 Europe/Kyiv

- **Expected versus actual recovery:** E5-I5 projected the resumed hierarchy cell to finish at
  09:00–09:15 Kyiv; it completed normally at **09:06:15**. The resumed attempt consumed 1.0151
  GPU-hours. Including the conservatively charged OS-interrupted attempt and the fail-closed probe,
  total cell-3 ownership is **2.4935 hours**, essentially identical to the seed-0 reference of
  2.4914 hours and well below its 4.5-hour authorization.
- **Completed seed-2 hierarchy health:** the post-resume trace remains continuous and finite through
  40k. Final loss is **0.09054**, between seed 0 (0.08470) and seed 1 (0.09988); final fixed-grid
  foreground MSE is **0.03451**, matched to seed 0 (0.03428) and better than seed 1 (0.03804).
  Across cloud-visible common intervals, mean loss is 1.040× seed 0 and 0.964× seed 1, while mean
  validation is only 1.042×/1.035×. Throughput is stable at 0.1976 s/update, with 3.635 GiB peak
  allocation and no resumed-run stall or numerical discontinuity.
- **Completed rollout/provenance:** rollout has exactly **25,600 records**, clips 150–249 and
  horizons 1–256, with mean foreground MSE 0.22051. The COMPLETE marker and all **24/24** artifacts
  independently hash under RunSpec `f8fc583e62c6ea0a`; manifest SHA-256 is
  `d74c691f9feaf007750bbef9bc12dde5055f8d8c7ac3bc11fb17063e0c676639`. W&B is terminal
  `finished` with the 357.2 MB content-addressed artifact and alias `d1-56-diff-s2`. The servicing
  interruption left two training and one validation telemetry rows unsynced in cloud history, but
  the local log and RNG-complete checkpoint cover that interval and the scientific rollout artifact
  is complete; this is a disclosed telemetry gap, not missing scientific data.
- **Final comparator state:** cell 4/4, `d0-diff-long` seed 2, is healthy at **19,000/40,000** under
  RunSpec `8df2dfe3e392b652`. Its format-v2 checkpoint hashes to
  `fcdf2540d6f689d4dd0a141f7e04ef8254874ce338c91f918e7f31ca3a676fa6` and contains the exact
  152,000-example sampler position, next update 19,001 and all Python/NumPy/CPU/CUDA RNG states.
  Controller PID 19032 and trainer PID 1588 are healthy and W&B is current through 18k.
- **Comparator trends versus prior seeds:** through 18k common cloud intervals, mean training loss
  is only **1.027× seed 0 and 1.020× seed 1**; its latest 0.09986 is below seed 0's 0.10575 and
  effectively seed-1 matched at 0.09942. Nine validation reads average 0.03331 versus 0.02932/0.03008;
  this 11–14% average offset is driven by the initial 2k value (range 0.00715–0.16958), while the
  latest 0.00833 is better than seed 0's 0.00976 and equal to seed 1's 0.00834. The trajectory is
  convergent rather than divergent.
- **Throughput/GPU health:** mean current throughput is **0.5303 s/update**, versus 0.5287 for seed 1
  and 0.5726 for the seed-0 run containing a legacy WDDM stall. The current cloud range is
  0.5263–0.5398 s/update; the local 19k interval is 0.56 seconds and remains below the spill guard.
  Allocated VRAM is exactly 7.386 GiB; board telemetry is 8.72/16.38 GB, 100% utilization, 67 °C
  and approximately 120 W. There is no competing trainer, NaN, restart or WDDM spill.
- **Projection/budget:** completed charged ownership is **11.3438 GPU-hours** and active cell
  ownership is approximately 2.8 hours at this audit. The remaining 21k updates plus rollout project
  cell-4 completion near **15:15–15:35 Kyiv**, followed by registered analysis and validation within
  minutes. Terminal E5a consumption remains approximately **17.7–19 hours**, leaving about 9–10
  hours below the frozen 28-hour maximum.
- **Decision:** continue unchanged. The paired seed-2 early/late effect remains unreadable until the
  comparator's complete 25,600-record rollout lands. The estimand, clip alignment, source RunSpec,
  batch/LR/update/EMA/NFE regime and decision rule remain valid. No E5b authorization is implied.

#### Terminal audit E5-I7 — 2026-07-15 15:55 Europe/Kyiv

- **Expected versus actual:** E5-I6 projected cell-4 completion at 15:15–15:35 Kyiv. Training and
  rollout completed normally at **15:32:44**, followed within minutes by the frozen registered
  analysis and the independent completion validator. All four E5a cells are terminal.
- **Artifact and cloud gate:** every new rollout contains exactly 25,600 aligned records and every
  content-addressed manifest contains **24/24** files. All four W&B training runs and the registered
  analysis run are `finished`; their cloud artifacts match the local RunSpecs. The independent
  validator reports `PASS` in `artifacts/e5a_trust_replication/completion_validation.json`. The
  already disclosed seed-2 hierarchy telemetry gap from the Windows interruption does not affect
  its RNG-complete checkpoint, rollout or manifest.
- **Primary registered results:** positive effects favor the 224-frame-reach hierarchy.

  | Training seed | Early h1–24 effect [95% CI] | Late h61–256 effect [95% CI] | Registered boundary |
  |---:|---:|---:|:---:|
  | 0 | +0.0204 [+0.0133, +0.0279] | −0.0064 [−0.0095, −0.0032] | yes |
  | 1 | +0.0082 [+0.0035, +0.0133] | +0.0074 [+0.0038, +0.0112] | no |
  | 2 | +0.0230 [+0.0150, +0.0315] | +0.0087 [+0.0035, +0.0139] | no |

- **Decision rule:** zero of the two new seeds reproduce the required early-positive/late-negative
  direction, and zero reproduce it CI-clean. The registered verdict is **NOT_REPLICATED** and the
  frozen gate returns **FULL_E5B_NOT_AUTHORIZED**. Secondary HARD and answerable-HARD rows cannot
  rescue the failed primary gate.
- **Scientific interpretation:** the early hierarchy advantage is positive and CI-clean in all
  three training seeds. The late seed-0 inversion is not robust: both new seeds instead have a
  positive, CI-clean late hierarchy advantage. Addressability and self-fed reliability remain
  distinct constructs, but this experiment does not locate a reproducible trust boundary or phase
  transition. The title-level empirical story is therefore **reach and fidelity**; trust is retained
  as a seed-sensitive robustness question and a transparent negative replication.
- **Accounting and next gate:** exact charged ownership is **17.785247 project GPU-hours** versus
  28 authorized. Unspent E5 capacity does not transfer automatically. E5b and E5c are closed under
  their frozen gates. The next portfolio item is E7a/c robustness, but it requires its own CPU
  freeze, estimand review and explicit launch authorization; no subsequent study is launched by
  this audit.

#### E7c preparation audit E7c-I0 — 2026-07-15 19:20 Europe/Kyiv

- **Order and scope:** governance is reconciled to the live forward order E7c → E7a → E8a →
  article/dissertation internal freeze → utility-gated E6 → optional post-freeze E9. E3a remains
  deferred under configuration-only B1 wording; E5b/E5c remain closed.
- **Frozen estimand:** four seed-0 diffusion contrasts (896 reach, 1,152 interior, 1,152 package
  shape, 2,432 inversion) × five variants (NFE 8/16/32 at sampler seed 0; NFE 16 at sampler seeds
  1/2) on a new purpose-locked robustness set. This is 35 checkpoint-only evaluations and does
  not reuse E2 or select models.
- **Parity and metrics:** sampler seed 0 retains the exact legacy CRN integer formula. The CPU
  numerical sample gate is designed to require bit-identical NFE-16 tensors. Every GPU cell writes
  complete-v2 false-positive-sensitive metrics, an immutable RunSpec/manifest, W&B run/artifact,
  access entry and GPU-ledger row.
- **Budget:** measured E2 runtimes project 6.0236 GPU-hours. E7c has an 8.0-hour hard maximum and
  a twice-reference per-cell stop. E7a remains separately gated inside the 44-hour total E7 cap.
- **Current gate:** implementation tests are 53/53 PASS; pre-data matrix/checkpoint/AE/training-
  data validation passes. No E7c dataset has yet been generated and no GPU process has launched.
  The next steps are an implementation commit, clean-commit CPU/parity attestations, dataset
  generation and collision validation, final authorization, W&B/clean-worktree check, then launch.

#### E7c data/design gate E7c-I1 — 2026-07-15 19:25 Europe/Kyiv

- **Independent set:** generated 500 × 200 clips with seed 20260715 from clean commit `9947ba4`;
  dataset ID `loopmom-robustness_holdout-20260715-477845150808`, clip SHA-256
  `477845150808b667...`. Its purpose is `robustness_holdout`; training and monitoring are forbidden.
- **Structural/distribution gate:** PASS. Shape/dtype/action-position consistency are exact;
  measured hold probability is 0.09991 and nonzero-direction continuation is 0.90115. All 500
  clips are unique with zero exact collisions against legacy training and test pools.
- **Design gate:** PASS for all 35 registered cells, seven frozen checkpoint hashes, AE/training-
  data hashes, independent-from-E2 dataset hash, 6.0236-hour projection and 8.0-hour maximum.
- **Authorization:** E7c is authorized only under its committed protocol/config/CPU/data/design
  attestations. The remaining immediate launch gates are a clean committed worktree, no competing
  project process, repeated design check and successful online W&B write. No GPU process has yet
  launched.

#### E7c final prelaunch gate E7c-I2 — 2026-07-15 19:26 Europe/Kyiv

- **Clean boundary:** commit `abac094`; clean worktree; no project controller/evaluator; no stale
  E7c RunSpec or GPU ledger; GPU has desktop-only occupancy.
- **Repeated gates:** authorization and all hash-bound markers PASS; 35-cell design/data/checkpoint
  validation PASS with dataset-manifest SHA-256 `135f999f13ce...`; CPU sampler parity remains
  bit-exact at NFE 16/sampler seed 0.
- **Cloud gate:** online W&B write PASS using `_netrc` after removing only the inherited process-
  level `WANDB_API_KEY`: `https://wandb.ai/leo27heady/hgrid/runs/e7c-gate-loopmom-
  robustness_holdout-20260715-477845150808-abac0945-20260715192535`.
- **Frozen launch:** one hidden controller, command
  `scripts/run_e7c_sampler_nfe.ps1 -Manifest data/robustness/loopmom_e7c_500x200.manifest.json`.
  It cannot train, alter the matrix, launch E7a, or exceed the 8-hour tranche without failing.

#### E7c launch health E7c-I3 — 2026-07-15 19:29 Europe/Kyiv

- **Live identity:** controller PID 23536; first cell `hier-min`/training-seed 0/NFE 8/sampler 0,
  RunSpec `d58c544280c45de3`, W&B `https://wandb.ai/leo27heady/hgrid/runs/e7c-
  d58c544280c45de3`. The RunSpec binds commit `07ae403`, dataset/AE/checkpoint hashes, complete-v2
  metrics, 48+152 geometry and the explicit CRN formula.
- **Expected versus actual:** after 48/100 clips, elapsed rollout time is 0.03 hours and running
  foreground MSE is 0.1731, converging toward the frozen seed-0 NFE-16 reference region rather
  than diverging. This projects about 0.06–0.07 hours plus encoding/upload, inside the 0.1123-hour
  per-cell stop and consistent with the 6.0236-hour tranche estimate.
- **Health:** 83–94% sampled GPU utilization, approximately 3.3/16.38 GB board memory, 64–67°C and
  about 116 W. Low VRAM occupancy is expected for checkpoint-only sequential evaluation; compute
  is saturated and changing batch/evaluation geometry would invalidate the frozen estimand.
  Latent statistics match the frozen training reference (coarse 0.1213, fine 0.1618); no nonfinite
  metric, stall, spill, pairing change or cloud failure is present.
- **Decision:** continue unchanged. The four-hour same-task heartbeat is active as
  `e7c-four-hour-health-audit`; no E7a authorization is implied.

#### E7c four-hour audit E7c-I4 — 2026-07-15 23:28 Europe/Kyiv

- **Progress and identity:** 27/35 cells are complete; the 28th is `hier-c2`/NFE 32/sampler 0,
  RunSpec `dc89c303df49a187`, at 80/100 clips. All 28 RunSpecs share one frozen source commit,
  dataset hash and AE hash. The controller and one evaluator tree are alive; no competing project
  process exists.
- **Expected versus actual runtime:** completed cells charge **3.632616 GPU-hours** versus
  **3.676550 expected** (−1.2%). Every cell exited zero and remains below its twice-reference stop.
  Eight cells including the active one remain with 2.347 expected hours. Current scaling projects
  approximately **5.95–6.2 total GPU-hours**, completion around **2026-07-16 02:00–02:30 Kyiv**,
  and at least 1.8 hours of headroom below the 8-hour maximum.
- **Compute/stability:** the active NFE-32 cell is at 100% sampled utilization, approximately
  3.37/16.38 GB board memory, 66°C and 127 W. Its running foreground MSE moves 0.2269 → 0.1874 as
  the clip population fills, consistent with completed `hier-c2` NFE-8/16 means 0.1866/0.1885.
  Encoding time and latent statistics remain invariant; no nonfinite value, timeout, WDDM spill,
  stall, traceback or disabled-cloud condition is present.
- **Immutable/cloud audit:** 27 COMPLETE directories contain every required metric/evaluation/
  manifest artifact; all **81 manifest-declared SHA-256 hashes** and COMPLETE identities verify.
  The purpose-locked access log has 28 unique rows for 28 unique RunSpecs and one dataset. W&B
  group state is 29 finished runs (including gates) and one running evaluation; completed cells
  uploaded their metric artifact and rollout strip.
- **Provisional paired health, not the terminal decision:** all 15 currently complete registered
  pair×variant effects are positive with ordinary clip-cluster 95% intervals excluding zero:
  896 reach +0.0465 to +0.0482, 1,152 interior +0.0140 to +0.0201, and temporal package shape
  +0.0221 to +0.0292. Full-frame, union-mask and recall directions corroborate them; every paired
  key and all 15,200 complete-metric records align. The 2,432 inversion is not readable until its
  flat partners land, and only the frozen 20-effect simultaneous family may issue the verdict.
- **Decision:** continue unchanged. Do not edit scientific files, alter the factorial, inspect a
  partial terminal family, or authorize E7a. The expected untracked live-output directory is the
  only worktree change; this audit commits documentation only.

#### E7c terminal audit E7c-I5 — 2026-07-16 03:28 Europe/Kyiv

- **Completion and budget:** the controller ended normally at 01:26 Kyiv after **35/35** registered
  checkpoint-only evaluations. All ledger rows exited zero and total mechanically charged GPU
  time is **5.9975507 hours**, 0.0324 below the 6.03-hour projection and 2.0024 below the 8-hour
  authorization. No retry, failed cell or competing project GPU process occurred.
- **Independent immutable validation:** all 35 unique RunSpecs are COMPLETE and share the frozen
  source, dataset, AE and checkpoint identities. Every one of the 105 manifest-declared files
  matches its byte count and SHA-256. Each NPZ contains 15,200 finite records for every required
  complete metric; all clip, horizon, hardness, observability and foreground-population pairing
  keys align. The access ledger has 35 unique RunSpecs and one purpose-locked dataset.
- **Registered family:** an independent recomputation exactly reproduces the stored 10,000-
  replicate studentized max-statistic clip-cluster family (critical value **2.7160037**, maximum
  numeric difference 0). All **20/20** expected-direction point estimates are positive and
  **18/20** simultaneous lower bounds exceed zero. The three lower/interior/package-shape
  contrasts are CI-clean for every NFE/sampler variant. The 2,432-token flat-fidelity inversion
  is CI-clean at NFE 8/16/32 with sampler seed 0, but its NFE-16 intervals at sampler seeds 1 and 2
  are +0.0098 [−0.0001,+0.0196] and +0.0077 [−0.0013,+0.0167]. The frozen terminal verdict is
  therefore **DIRECTIONALLY_ROBUST**, not `ROBUST_CI_CLEAN` and not sampler-sensitive by sign.
- **Metric interpretation:** non-empty foreground, full-frame, background and union-mask errors
  favor the registered candidate in all 20 variants; recall and F1 do so in all 20. Component-count
  diagnostics are not uniformly favorable, so no claim that every endpoint improves is made.
- **Cloud and evidence:** the W&B group contains 38/38 finished runs: 35 evaluation cells, two
  connectivity gates and one analysis. The cloud verdict, 35-cell inventory, 20-effect family and
  max-statistic critical value match local evidence. Terminal results SHA-256 is
  `93eef5069598b099f62013384e6d91978d7c9996f8fd4f67ab790486758054e6`.
- **Scientific decision and next gate:** E7c supports point-estimate robustness of the four frozen
  diffusion directions across a fourfold per-step NFE range and three sampler seeds on a separate
  LoopMom robustness set. It does not establish universal sampler invariance, training-seed or
  codec robustness, and it does not replace E7a. E7c is closed with no adaptive cells. E7a may now
  enter protocol/data/provenance/budget preparation, but remains separately unlaunched and
  unauthorized until those gates pass.

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

**Final result (2026-07-15):** **COMPLETE / NOT_REPLICATED** in **17.785247 GPU-hours**. The early
benefit is positive and CI-clean at seeds 0, 1 and 2; the late inversion occurs only at seed 0 and
reverses to positive late hierarchy effects at seeds 1 and 2.

**GPU time:** **17.785247 hours actual** against a 28-hour authorization.

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

**Terminal gate:** **NOT AUTHORIZED / CANCELLED.** E5a produced zero directional replications among
the two new seeds, so the frozen condition for a full dose-response is not met. Any future
self-fed-reliability study must be designed and preregistered as a new study rather than presented
as localization of a replicated boundary.

**GPU time:** **0 hours actual**.

### E5c — Rollout-aware/Self-Forcing-class intervention

This originally proposed a scoped rollout-aware version only after E5a/E5b identified a
reproducible trust boundary. That prerequisite failed.

**Historical question:** can training on self-generated context improve self-fed reliability
without sacrificing short-horizon fidelity?

**GPU time:** **24–45 hours**.

**Portfolio status:** E5c is deferred and not authorized. E5a did not replicate the boundary, so
the programme stops this branch and demotes the empirical trust claim as precommitted.

**Closed E5 accounting:** **17.785247 GPU-hours actual** for E5a; E5b/E5c consumed zero. The unused
allocation is released, not transferred to a discretionary study.

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

Co-primary registered family:

- foreground MSE and union-mask MSE;
- intention-to-score trajectory error;
- intention-to-score identity accuracy;
- persistent-object assignment coverage.

Positive effects always favor the hierarchy. One studentized max-statistic clip-cluster family
covers all five endpoints across all three training seeds. Full/background errors, foreground
precision/recall/F1, misses/extras, matched-only diagnostics and overlap strata are secondary.
Equivalence is disabled: the earlier ±0.005 foreground margin is withdrawn before protected
access because it did not have a defensible native-unit smallest effect of interest.

**Question:** on a familiar long-rollout benchmark without loopmom's explicit re-entry geometry,
do flat and hierarchical packages differ in foreground fidelity, identity, trajectory or survival?

**CPU/prelaunch preparation:** implementation complete; immutable public/development gates in
progress.

**GPU time:** six arm×seed training runs plus long evaluation, approximately **30–40 hours**.

**Launch gate:** the article-utility decision passed at M1. Predictor training remains blocked
until official-source hashes, persistent train/development data, classifier domains, evaluator
reliability, exact six-cell dispatch, protected-data isolation, W&B and the mechanical 40-hour
budget gate all pass.

## 9. Representation and sampler robustness — E7

### E7c — Sampler/NFE robustness (first)

Evaluate seven frozen seed-0 diffusion checkpoints on a new purpose-locked LoopMom robustness set.
For every arm run NFE 8/16/32 at sampler seed 0 and NFE 16 at sampler seeds 1/2. The four fixed
contrasts are the 896 reach pair, 1,152 interior pair, 1,152 package-shape pair and 2,432 flat
inversion. Sampler seed 0 must be numerically identical to the legacy CRN path; common random
numbers are retained across compared arms.

**Question:** are the article's four central diffusion directions stable to sampling randomness
and to a fourfold per-step compute range?

**GPU time:** expected 6.03 hours; **hard maximum 8.0 hours**.

**Protocol:** `docs/E7C_SAMPLER_NFE_ROBUSTNESS_PROTOCOL_2026-07-15.md`.

**Terminal result (2026-07-16):** **DIRECTIONALLY_ROBUST** in 5.9976 project GPU-hours. All 20
registered point estimates retained their expected direction over NFE 8/16/32 at sampler seed 0
and NFE 16 at sampler seeds 1/2. Eighteen simultaneous lower bounds were positive. The two
exceptions were the 2,432-token inversion at alternate sampler seeds 1/2; both point estimates
still favored flat fidelity, but their family-wise intervals included zero. This closes E7c with
no post-result extension and qualifies only the precision of the upper inversion.

### E7a — Alternate autoencoder seed (second)

Train codec seed 1 under an otherwise identical codec regime and retrain exactly seven seed-0
diffusion predictors:

- 896-token lower regime;
- 1,152-token interior;
- the 2,432-token high-budget fidelity inversion;
- spatial versus temporal package comparison.

Evaluate both the original-codec controls and alternate-codec predictors on the same new locked
holdout. Use one fixed predictor seed; this isolates the codec-seed pipeline cleanly inside the
budget, while prior studies carry training-seed robustness. Apply one simultaneous family across
all eight codec-by-contrast effects. The frozen reconstruction/latent non-collapse floor precedes
predictor training; no failed codec may be retrained or replaced.

**Question:** is the frontier specific to one learned codec/latent geometry?

**GPU time:** **24–36 hours**.

**Protocol:** `docs/E7A_ALTERNATE_CODEC_ROBUSTNESS_PROTOCOL_2026-07-16.md`.
**Frozen design:** `experiments/configs/alternate_codec_robustness.json`.

### E7b — Simple codec controls

Add deterministic spatial pooling/downsampling and, if feasible, an alternate latent dimensionality.
Report reconstruction floors before predictor comparisons.

**Question:** does the effect require the learned AE, or follow the rate–resolution information
trade-off under simpler codecs?

**GPU time:** **12–24 hours**.

**Portfolio status:** deferred in favor of the stronger alternate-AE and cheaper sampler/NFE
controls.

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

## 13. Article/dissertation assembly and internal freeze

After E7c, E7a and E8a are integrated, assemble one fourth-article source from the binding compact
reach–fidelity specification and the detailed CH1–CH8 evidence chapters. Update the dissertation
rewrite directly: preserve the dissertation topic and the three existing article strands, while
using the fourth article as the controlled reach–fidelity/context-packaging study. Generate all
tables from immutable artifacts, complete the claim-to-artifact index, verify primary citations,
run clean-environment reproduction checks, and perform article/dissertation consistency review.

This internal freeze precedes the E6 utility decision. E6 runs only if the assembled article makes
clear that a recognizable Moving-MNIST boundary materially resolves a reviewer-facing gap. E9 is
post-freeze extension work and cannot delay dissertation closure.

## 14. Execution order summary

This is a decision tree, not a promise to exhaust the portfolio. Completed studies remain listed
as history; the live forward order begins at E7a. H3b parity precedes every new predictor run.

| Order | ID | Experiment/workstream | Portfolio status | GPU maximum | CPU/development estimate |
|---:|---|---|---|---:|---:|
| 0 | E0/E1/E2/E4a/E4b/E5a/E7c/E7a | Completed evidence and governance programme | **complete/frozen** | 103.362 h actual | complete |
| 1 | E7c | Sampler/NFE robustness on separate locked set | **complete / DIRECTIONALLY_ROBUST** | 5.9976 h actual | complete |
| 2 | E7a | Alternate-AE seed and frozen seven-arm/14-evaluation matrix | **complete / CODEC_DIRECTIONALLY_ROBUST** | 30.0238 h actual | complete |
| 3 | E8a | Classical pose-indexed retrieval | **COMPLETE / RAW_ACCESS_CEILING_CI_CLEAN** privileged closure reference | 1.0897 h actual | complete |
| 4 | M1 | Article/dissertation provisional assembly and internal freeze | **COMPLETE**; E8a integrated and article identity frozen | 0 | complete 2026-07-17 |
| 5 | E6 | Phase C hybrid on long Moving-MNIST | **UTILITY PASS / PREPARATION FROZEN / PROTECTED RUN BLOCKED** | 30–40 h | run public/development gates, then freeze hashes before holdout access |
| 6 | M2 | Final tables, claim/artifact index, clean-environment QA and release | required | 0 | 2–5 d |
| 7 | E9 | Counterfactual Dynamic LoopWorld primary factorial | optional post-freeze extension | 60–100 h | 1–2 wk |
| — | E3a | Fixed-target diffusion B1 | deferred while article retains configuration-only wording | 0 now | future if causal wording returns |
| — | E8c | CW-VAE protocol bridge training | removed/replaced by E4b; table remains CPU-only | 0 | 3–7 d table/audit |
| — | E3b/E5c/E7b/E8b/E9c/E10 | Additional methods/extensions | deferred outside portfolio | — | future work |

## 15. Totals

All totals assume one GPU and sequential project ownership. CPU/document work can overlap GPU
execution. GPU-hour consumption is wall-clock project ownership, not successful-training time.

### Measured and forward allocations

- measured E0+E2+E4a+E4b+E5a+E7c+E7a+E8a consumption: **104.452 GPU-hours**;
- E8a: **1.0897 hours actual; terminal**;
- E6: **40 hours maximum, article-utility gated**;
- E9: **100 hours maximum, optional and post-freeze**;
- without E6, the scientific-run maximum is approximately **204.452 hours**;
- with E6, the scientific-run maximum is approximately **244.452 hours**.

### Pool C — protected reserve

- **allocation:** **50 GPU hours**;
- covers pilots, failed runs, retries and estimation error;
- cannot fund discretionary experiments or broaden a completed study.

### Hard ceiling and replacement rule

Including the protected reserve, the live maximum is approximately **294.452 GPU-hours** with E6,
or **254.452 hours** without it. This is below the 400-hour ceiling without treating unused budget
as a reason to run work. E4b is already included in measured consumption and replaced E8c rather
than adding to the portfolio. E3a/b, E5b/c, E7b, E8b/c training, E9c and E10 remain outside.

Before each tranche, replace estimates with measured probes and a run-count ledger. A tranche is
not authorized unless its maximum fits the remaining non-reserve envelope. Stop when the claims are
resolved; unused GPU hours indicate successful governance, not incomplete work.

## 16. Expected article/dissertation evidence products

1. **Configuration/cost table:** levels, grids, strides, context slots/reach, context/target/total
   tokens, parameters, examples, updates, NFE, runtime and VRAM.
2. **Reach–fidelity frontier table:** budget, reaches, per-seed errors, effects/CIs and tested
   winner by optimizer regime.
3. **Shape/head factorial table:** flat/spatial/temporal/depth packages × diffusion/regression ×
   seeds.
4. **Reach-differential table:** comparator reaches, exact band, per-seed effect and confirmation
   status.
5. **E5 negative-replication table:** all three early/late seed effects and the failed registered
   replication rule; no refresh/corruption localization table.
6. **Complete-metric table:** foreground, background, union, full-frame, precision/recall,
   misses/extras and coverage.
7. **E7c robustness table:** four contrasts × five NFE/sampler variants, simultaneous intervals,
   per-step NFE and total inference allocation.
8. **E7a codec table:** reconstruction floors/latent statistics plus the frozen alternate-AE
   predictor contrasts and provenance.
9. **E8a reference table:** privileged pose-indexed retrieval and any budget-matched variant kept
   explicitly separate.
10. **CW-VAE/prior-art protocol table:** architecture, levels, clocks, memory mechanism, context,
   horizon, self-feeding, dataset, metrics, budget matching and reproduction status.
11. **Qualitative evidence:** frozen-selection re-entry strips, failure cases and overlap/object
   examples.
12. **Dissertation synthesis map:** each fourth-article result linked to the unchanged dissertation
   programme and the three existing article strands.
13. **Machine-readable artifact map:** every manuscript number linked to immutable
   RunSpec/data/AE/result hashes.

## 17. Current execution instruction

E7c is terminal as `DIRECTIONALLY_ROBUST`; no sampler/NFE cells may be added. E7a is terminal as
`CODEC_DIRECTIONALLY_ROBUST` in 30.0238 project GPU-hours: all four alternate-codec point
estimates are positive, while the alternate 2,432-token inversion remains family-wise unresolved.
Do not add a codec, predictor seed, arm or endpoint. E8a is terminal as
`RAW_ACCESS_CEILING_CI_CLEAN`; its privileged reference must remain visually and verbally
separate from token-matched neural comparisons. The live workstream is M1: internally freeze the
assembled fourth article and dissertation synthesis, then apply the preregistered E6 utility and
validity gate. E9 remains optional post-freeze work and cannot delay dissertation closure.

### E8a launch audit — 2026-07-17 12:28 Europe/Kyiv

- **Issue audit and repair:** the initial live check-only gate correctly exposed that the E8a
  validator treated the committed byte-identical manifest copy as an earlier independent dataset.
  The origin was path-based exclusion rather than manifest-identity handling. The validator now
  skips only byte-identical manifest mirrors; an explicit regression test distinguishes a mirror
  from a changed manifest.
- **CPU/scientific gate:** 72 tests pass after the repair, including 11 E8a-specific tests.
  Scientific-config, authorization-ledger, article/dissertation consistency, label-parity,
  compilation and whitespace gates pass from clean source. Frozen protocol/config and CPU
  attestations are committed.
- **Data gate:** the new purpose-locked set is
  `loopmom-retrieval_reference_holdout-20260717-804b9f5d2c13`, with 500 unique
  `uint8` clips of 200 frames. Action-position consistency, persistent-object schema, motion
  statistics and artifact hashes pass. Exact-clip collisions are zero against legacy
  train/test, E2, E7c and E7a. The data-side and committed manifests have identical SHA-256
  `06a88f48676a7c0dc1bd96223c3ca09d6ca8e8c8ef9efb0dd731962155d1558c`.
- **Cloud/launch gate:** the purpose-matched access switch, fail-closed authorization check,
  post-data design validator and W&B online connectivity gate pass. E8a launched from commit
  `dedcf01` at 12:26 Europe/Kyiv in group
  `e8a-pose-mosaic-loopmom-retrieval_reference_holdout-20260717-804b9f5d2c13`.
- **Early actual versus expected:** the CPU pose-mosaic reference completed all 15,200 aligned
  records and immutable cloud/artifact output before the first GPU cell. `hier-min` seed 0 is
  the first active neural evaluation. GPU utilization is high and VRAM remains safely below the
  WDDM spill region; the frozen checkpoint-only NFE geometry is unchanged.
- **Decision:** no scientific blocker. Continue the seven registered neural evaluations,
  registered simultaneous analysis and independent completion/W&B validation. No adaptive span,
  arm, checkpoint, sampler, endpoint or storage rule is authorized.

### E8a terminal audit — 2026-07-17 16:35 Europe/Kyiv

- **Expected versus actual execution:** all four CPU pose-mosaic spans and all seven frozen neural
  evaluations completed. The seven GPU cells consumed **1.0897 GPU-hours** versus 1.0952
  expected and 4.0 authorized. Per-cell runtimes tracked their frozen references; no stall,
  numerical failure, WDDM spill or retry occurred.
- **Operational issue and origin:** the first analysis attempt stopped before computing effects
  because neural artifacts serialize horizon-major records while mosaic artifacts serialize
  clip-major records. Both contain the same 15,200 unique `(clip,horizon)` keys and identical
  `hard/dt/seen/fg_nonempty` labels. The analysis now canonicalizes by registered keys, rejects
  duplicates, and the independent validator repeats the alignment check. Seventy-four tests pass.
  No model, dataset, metric, endpoint or GPU artifact was changed or rerun.
- **Registered primary family:** raw K14→K48 is +0.0723 [0.0548,0.0898]; raw K38→K48 is
  +0.0210 [0.0096,0.0325]. Neural-minus-K48 effects are `hier-min` +0.1219
  [0.1031,0.1407], `hier-long` +0.1368 [0.1190,0.1545], `strided-flat` +0.1134
  [0.0944,0.1324], and `flat-c2` +0.1310 [0.1122,0.1498]. All intervals are simultaneous
  95% max-statistic clip-cluster intervals.
- **Reference performance and storage:** K48 foreground MSE is 0.0494, full-frame MSE 0.0027
  and recall 0.8438. Median stored columns increase 80→100→140→154 for K6/K14/K38/K48;
  mean future-crop observed coverage increases 0.454→0.550→0.750→0.809. K38 saturation is
  false.
- **Integrity/cloud gate:** completion validation passes for 4 mosaic variants, 7 neural
  evaluations, 6 effects, access logs, immutable manifests and budget. The W&B group has exactly
  12 finished runs: 3 connectivity gates, 1 pose-mosaic reference, 7 neural evaluations and
  1 analysis; every scientific run has its expected artifact.
- **Scientific verdict:** **RAW_ACCESS_CEILING_CI_CLEAN**. Exact pose-indexed raw access removes
  much of the error remaining in learned self-fed systems on this static world. This is not a
  token-matched model win, efficiency result, learned-retrieval result or natural-video claim.
- **Next gate:** freeze E8a evidence and complete M1 article/dissertation assembly. E6 may proceed
  only if the assembled article still has a material reviewer-facing need for the valid
  persistent-object long Moving-MNIST study. E9 remains optional post-freeze extension work.

### M1 internal freeze and E6 utility gate — 2026-07-17 16:45 Europe/Kyiv

- **M1 outcome:** the article, claim ledger, dissertation synthesis, generated tables, Figure 21
  and artifact map now include terminal E8a. The frozen article identity remains reach and
  fidelity in budgeted context packaging; E8a is explicitly a privileged access reference.
- **Utility decision:** **PASS**. The short Moving-MNIST 10→10 bridge has no re-entry and all
  load-bearing long-memory evidence otherwise remains on LoopMom. A valid persistent-object long
  Moving-MNIST study materially addresses the recognizable cross-dataset boundary while allowing
  positive, null, heterogeneous or unscorable outcomes.
- **Authorization state:** preparation only. The legacy Phase C path remains prohibited. E6
  training stays blocked until official glyph split isolation, persistent metadata, classifier
  domain validation, Hungarian/intention-to-score semantics, overwrite/collision tests, frozen
  six-cell RunSpecs, cloud connectivity and the 40-hour mechanical budget gate all pass.
- **Decision record:** `docs/RFT_INTERNAL_FREEZE_AND_E6_UTILITY_DECISION_2026-07-17.md`.

### E6 preparation audit — 2026-07-17 17:02 Europe/Kyiv

- **Expected versus actual inherited state:** the article-utility gate had passed, persistent
  generator metadata and a Hungarian scoring primitive existed, but the protected evaluator,
  classifier-domain fixture, evaluator reliability ceiling, immutable six-cell controller,
  simultaneous analysis and completion audit did not. The legacy Phase C path remains
  fail-closed and was not reused.
- **Frozen scientific repair:** E6 is now explicitly an external Moving-MNIST boundary, not a
  LoopMom mechanism replication. The exact 1,152-token `mmL-flat`/`mmL-hier` pair, seeds 0–2,
  30k/B16 training, 48-frame warm-up, 300-frame NFE-16 rollout, persistent assignment semantics
  and protected clips 250–349 are fixed. Equivalence is disabled before protected access; the
  unjustified ±0.005 margin has been withdrawn.
- **Metric validity:** the production path uses persistent IDs/classes/boxes, one-to-one Hungarian
  matching, frame-diagonal miss penalties, explicit extras, intention-to-score identity,
  false-positive-sensitive frame metrics and overlap strata. The registered family is five
  endpoints × three training seeds under one studentized max-statistic clip-cluster bootstrap.
- **Preparation implementation:** official train/test glyph roles, immutable zero-action dataset
  manifests, exact-clip collision validation, a train-only provenance-v2 predictor path,
  classifier training/domain validation, raw/AE evaluator ceilings, protected access logging,
  W&B-required cells, mechanical GPU ledgers and terminal inventory/cloud validation are wired.
- **CPU actual:** 79 repository tests pass; the focused E6 suite has 15 passing tests, including
  crossing/permutation, overlap, miss/extra, empty, wrong-location/class, threshold/tie,
  abstention and overwrite cases. Python compilation, JSON semantics and PowerShell parsing pass.
- **Budget:** the fixed maximum remains 40 GPU-hours. Cell maxima total 39.35 hours: codec 0.8,
  classifier 0.25, classifier/evaluator validation 0.5, six predictor trainings 28.2 and six
  protected evaluations 9.6. Failures and retries are charged; reserve use is forbidden.
- **Gate decision:** no scientific blocker. Commit the preparation implementation, generate a
  hash-bound CPU attestation, then run only the public/development preparation tranche. Protected
  holdout generation and predictor training remain blocked until the resulting codec,
  classifier, dataset and evaluator hashes are reviewed and frozen.

### E6 preparation launch audit — 2026-07-17 17:10 Europe/Kyiv

- **Prelaunch:** the focused 15-test gate, full 79-test suite, implementation hashes, 39.35-hour
  allocation arithmetic, clean-source check and online W&B check passed. The four official MNIST
  IDX sources were recovered from the frozen PyTorch mirror; counts and SHA-256 values are checked
  by the dataset validator.
- **Fail-closed operational repairs:** check-only initially consumed a deterministic W&B gate ID,
  so the first live attempt stopped before data generation when W&B refused `resume=never`.
  Connectivity IDs are now timestamp-unique and the controller verifies every implementation hash
  against the CPU report. A second attempt exposed a misplaced local variable in the duplicate-clip
  validator; it stopped before GPU work. The validator was corrected, independently exercised on
  both generated arrays, reattested and resumed without modifying either immutable dataset.
- **Data actual:** training is 3,000×100 from official train glyphs; development is 500×348 from
  official test glyphs. Each set has unique exact clips, persistent metadata and zero action
  conditioning; cross-role exact-clip collisions are zero. Development overlap-frame prevalence
  is 0.397 and is retained for stratified evaluation.
- **Active codec:** the fresh seed-0 two-level codec is healthy at 2,000/7,000 updates. Validation
  coarse/fine MSE improved from 0.0216/0.0092 at 1k to 0.0125/0.0055 at 2k. GPU utilization is
  about 93%, peak live allocation is approximately 2.5/16.4 GB, temperature 66°C and power 119 W.
  Low VRAM reflects the frozen small codec/batch rather than a spill or throughput problem; batch
  or optimizer geometry will not be changed mid-study.
- **Decision:** no scientific blocker. Continue codec, classifier and exact production-evaluator
  reliability gates. Protected holdout generation and all six predictor runs remain disabled.

### E6 classifier-gate audit — 2026-07-17 17:17 Europe/Kyiv

- **Codec/classifier actual:** the codec completed in 0.0532 charged hours with coarse/fine
  development MSE 0.0079/0.0033; the official-train-only classifier completed in 0.0071 hours.
  Both immutable W&B artifacts finished normally.
- **First domain result:** clean 0.9885, translated 0.9845, AE reconstruction 0.9830,
  deliberately degraded 0.9795 and diagnostic overlap 0.6815 behaved as expected. The nominal
  rendered-component domain failed its 0.94 floor at 0.926.
- **Origin:** generator `overlap=false` means the two digit stroke masks share no positive pixel;
  it does not guarantee that the other 28×28 glyph box is absent from the target classifier crop.
  The fixture therefore mixed isolated components with ambiguous two-digit crops, contrary to the
  frozen protocol's isolated rendered-component gate. Ambiguous crops already have their own
  overlap diagnostic.
- **Repair and independent check:** the isolated domain now requires pairwise source boxes to be
  separated by a two-pixel crop margin. No threshold, classifier weight, protected datum or
  outcome endpoint changed. A CPU rebuild using the same immutable codec/classifier yields
  rendered-component accuracy 0.977; all five thresholded domains pass, while the overlap
  diagnostic remains deliberately weak at 0.673.
- **Accounting/provenance:** the failed 8.9-second fixture attempt is charged in the GPU ledger and
  its failed report is retained. The controller now catches and charges failed scriptblocks rather
  than allowing an inner shell exit to bypass accounting. Completion permits charged failed cells
  but requires the exact successful phase inventory.
- **Decision:** no scope or estimand blocker. Reattest the repaired fixture/controller and resume
  only the classifier/evaluator validation cell. Protected holdout access remains disabled.
