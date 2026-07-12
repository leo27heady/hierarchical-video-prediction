# Comprehensive experiment execution order for the context-packaging vision

**Frozen:** 2026-07-12

**Authority:** the author owns scientific and GPU-launch authority. Codex may execute autonomously
only after the author supplies the current-run handoff and explicitly transfers execution control.

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

## 2. Current handoff boundary — E0

### E0 — Finish c2-only optimizer-regime control

**Current state at 2026-07-12 09:56 Kyiv:**

- seed-2 diffusion trio: complete (`S2TRIO_DONE`, 2026-07-11 23:22);
- original master queue: terminated by the watcher before c2/Phase C;
- `hier-c2` B8x60k: trained and evaluated;
- `flat-c2` B8x60k: approximately 30k/60k;
- Phase C: not launched.

**Question:** does the 2,432-token result change when the number of examples is approximately held
constant but batch size, optimizer updates, gradient noise and effective EMA horizon change?

**Interpretation:** optimizer-regime robustness, not “twice the data” and not asymptotic proof.

**Remaining time:** approximately **3.8–4.5 GPU hours**, including rollout and comparison.

**Handoff gate:** c2 completion marker; old controller absent; no Phase C process; GPU idle; queue
lock absent; logs/checkpoints/NPZs enumerated; result-generation worktree recorded.

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

**E3 total:** **28–38 GPU hours**.

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

Run all four factorial cells at seeds 3 and 4 only under the prewritten precision rule: expand if
the three-seed interaction interval remains too wide for the planned statement or seed variance is
scientifically material. This rule is based on uncertainty, not desired sign.

**GPU time if triggered:** **28–36 hours**.

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

**E5 total:** **52–87 GPU hours**.

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

### E7c — Sampler/NFE robustness

Evaluate central diffusion arms across sampler seeds and a small NFE grid, reporting both per-step
and total inference allocation.

**GPU time:** **4–8 hours**.

**E7 total:** **40–68 GPU hours**.

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

**E8 total:** **45–94 GPU hours + 1–3 development weeks**.

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

**GPU time:** **60–100 hours** for the primary factorial; **30–50 additional hours** for the head
extension.

### E9c — Retrieval baselines on counterfactual worlds

Evaluate classical and learned retrieval baselines from E8 on the same paired worlds.

**GPU time:** **10–25 hours** depending on reuse.

**E9 total:** **70–175 GPU hours + 1–2 development weeks**.

## 12. Natural/3-D re-entry extension — E10

Use a rendered or natural environment with camera pose, occlusion and object re-entry. Replace
exact pixel observability with geometry-derived or annotated visibility while retaining paired
package budgets and self-fed evaluation.

**Question:** do the synthetic reach/fidelity/trust relationships transfer when geometry,
appearance and hidden dynamics are realistic?

This is a separate flagship study, not required for the existing dissertation/article claim.

**Estimated time:** **2–5 development weeks + 80–160 GPU hours** for a first controlled study.

## 13. Execution order summary

| Order | ID | Experiment/workstream | Status after current handoff | GPU estimate | CPU/development estimate |
|---:|---|---|---|---:|---:|
| 0 | E0 | Finish c2-only control | already running | 3.8–4.5 h remaining | — |
| 1 | E1 | Evidence freeze, swap pass, governance repair | required | 0 | 8–16 h |
| 2 | E2 | Untouched loopmom confirmation, selected + full frontier | required | 10–18 h | 1–2 h plus evaluator integration |
| 3 | E3 | Fixed-target B1, diffusion + regression | core causal control | 28–38 h | 1–2 d integration/QA |
| 4 | E4a | Complete seed-2 head × package factorial | core interaction | 7–10 h | 2–4 h |
| 5 | E4b | Seeds 3–4 factorial expansion | precision-gated | 28–36 h | 2–4 h |
| 6 | E5 | Trust replication, dose response, rollout-aware intervention | core if trust is headline | 52–87 h | 2–5 d |
| 7 | E6 | Phase C hybrid on long Moving-MNIST | cross-dataset boundary | 30–40 h | 1–3 d |
| 8 | E7 | Representation, codec and sampler robustness | generality | 40–68 h | 2–5 d |
| 9 | E8 | Classical/learned retrieval + CW-VAE bridge | baseline defensibility | 45–94 h | 1–3 wk |
| 10 | E9 | Counterfactual Dynamic LoopWorld | strongest new synthetic study | 70–175 h | 1–2 wk |
| 11 | E10 | Natural/3-D re-entry | separate flagship | 80–160 h | 2–5 wk |

## 14. Totals

All totals assume one GPU and sequential runs. CPU/document work can overlap GPU execution.

### Current article/dissertation causal-completion set

Includes E0, E2, E3, E4a and E5a/E5b; excludes rollout-aware training, optional seed expansion,
Phase C and external/future studies.

- **GPU:** approximately **77–113 hours** including the remaining c2 work;
- **continuous single-GPU time:** approximately **3.2–4.7 days**;
- **CPU/integration:** approximately **4–8 working days**.

### Full dissertation robustness set

Adds E4b if triggered, E5c, E6 and E7.

- **GPU:** approximately **199–302 hours** from the current snapshot;
- **continuous single-GPU time:** approximately **8.3–12.6 days**;
- **CPU/development:** approximately **2–4 weeks**, overlapping GPU work.

### Complete extended new-vision programme

Adds E8, E9 and E10.

- **GPU:** approximately **394–731 total hours** from the current snapshot;
- **continuous single-GPU time:** approximately **16–30.5 days** at perfect utilization;
- **realistic calendar time:** approximately **6–12 weeks**, dominated by implementation,
  validation, failures/recovery and text integration rather than raw training alone.

These are planning envelopes. Before each phase, replace them with measured probe timings and a
run-count ledger. The programme may be scientifically complete without E10 because E10 is a
separate realism/flagship paper.

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
- process/GPU/lock state;
- resume/retry history and W&B caveats;
- explicit authorization for Codex to become the sole executing agent under this plan.

Codex should then begin at E1, not by launching E2 immediately. The evidence swap and immutable
backup come first.
