# REVIEW OF THE RFT REMEDIATION (commits a2438cf + 2689500) — 2026-07-11

**What this is.** An independent review (2 focused reviewers + direct verification) of the Codex
"Reach, Fidelity, Trust" dossier/plan/remediation, at the author's request. Read-only; nothing
was changed. Companion to `docs/REVIEW_FIVE_LENS_2026-07-10.md`.

**Verdict in one line: the dossier's science criticisms are largely correct and the code is
solid, but the program built on them overshoots — it un-adopts author-gated decisions, roughly
doubles the remaining GPU work, demotes verified pre-registered results to "development
evidence," and replaces a content-complete article with a 265-line skeleton. Adopt the
corrections and the tooling; reject the identity swap; amend the Phase C block into a hybrid.**

---

## 1. VERIFIED GOOD (adopt / keep)

- **The 30-test CPU suite passes** (reproduced twice, 2.5–2.9 s). The pytest-collection repair is
  real and valuable: the old `tests/test_model_spatial.py` trained 30 steps on an external
  dataset AND probed CUDA *at collection time*.
- **No registry drift**: `arm_registry.from_legacy` was cross-checked programmatically against
  all 34 `rhvd.ARMS` entries — reach rule (incl. hybrid-history), token arithmetic, target-slot
  rule (`WIN//stride` only when strided AND coarse present), heads, loss weights all match; 18
  headline budgets confirmed.
- **Live campaign provably safe**: zero shared files with the running chain, one-way imports
  only; hier-long s2 healthy (~24.5k/30k at review time); the Phase C block cannot fire before
  c2 (QUEUE 3/5 precedes it) and survives even ledger tampering (unconditional exit 97).
- **All 7 new citations are real** (verified on arXiv): WorldPack 2512.02473, MBench 2606.00793,
  CaR/SceneFly 2606.23105, GIM-World 2606.02436, MemLearner 2606.31734 (ECCV 2026), MemoBench
  2606.27537, Self Forcing 2506.08009. MemoBench's disappear-and-reappear paradigm is a genuine
  near-collision with the loopmom premise — the narrowed novelty positioning is *necessary*, and
  the field visibly moved into this niche in 2026 (fresh-sweep before submission is mandatory).
- **The post-hoc reach-band numbers are real**: hier-min vs flat-min on 14<dt≤48 recomputed from
  raw npz = +0.0880/+0.0809 (CIs match); B1 band +0.0675/+0.0709. Correctly labeled
  EXPLORATORY_POST_HOC; freezing them as confirmation primaries is good design.
- **Legitimate scientific corrections** (keep regardless of framing):
  P2 "tested bracket, not a point/continuous crossing"; P4 B8×60k = matched-example
  *optimizer-regime* change (batch/updates/EMA), not "2× data/convergence"; P6 geometric
  observability ≠ information-theoretic necessity (MNIST priors permit guessing); P5 opposite
  seed signs ≠ equivalence; the B1 target-interface confound is REAL (verified: hier-s2-min
  trains 2 coarse target slots vs flat-short's 4; 288 vs 320 target tokens); the legacy Phase C
  evaluator critique is factually grounded (left-to-right identity matching, many-to-one
  trajectories, both-arm-scorable intersection).
- **v2 metrics code is correct** under adversarial edge-case probing (Hungarian assignment:
  more-preds-than-GT, both-empty, ties, crossings, cost cutoffs; empty-GT frames named stratum —
  the legacy full-frame-fallback blind spot genuinely fixed; union-mask/precision expose
  hallucinations).
- **v2 generator fixes the contamination class properly on its own path**: seed REQUIRED,
  train/test glyph-pool split, SHA-256 manifests, overwrite refusal, purpose enforcement at read
  time.
- **The untouched-confirmation idea** is the single most cost-effective plan item (~4–8 GPU-h,
  no retraining) — the two-seed replications cover training variance, not analyst adaptivity on
  the reused 500-clip pool. No conflict with CRN/paired design.
- **The synthesis doc's layered-programme table** (docs/dissertation/REACH_FIDELITY_TRUST_
  SYNTHESIS) is a good answer to the "pivot-history bundle" defense question; the reversibility
  "enabler, not cause" boundary matches the five-lens recommendation.
- Fresh queue numbers cited by the new ledger all verify from raw npz (recomputed):
  S1-a B1 s1 **+0.0338** (family edge law now two-seed); S1-b B2 s1 **+0.0046** (sign holds,
  magnitude collapses — the pre-committed softening amendment worked exactly as designed);
  512-edge under regression **+0.0639** (edge law now cross-head, the biggest new result);
  strided-flat-reg vs reg-long **+0.0107 s0 / −0.0104 s1** (true seed-level sign flip under
  regression; head→spatial shift consistent across seeds).

## 2. WHAT OVERSHOOTS (reject / amend)

### 2.1 Governance — unilateral supersession of author-gated decisions (CRITICAL)
- New article title + "canonical fourth article" declared by fiat; the record gates title
  changes on author/supervisor (a72c490) and the packaging title was author-approved (dc3aaba).
  CH1–CH8 — the *implementation of the author-directed reframe*, content-complete prose — are
  stamped "historical source." A 265-line bullet-grade skeleton replaced ~2,000 lines of
  finished chapters.
- CLAIM_LEDGER.md replaced wholesale: the verbatim binding rule's provenance is gone and
  **L1–L10 row IDs were renamed (R/F/T/M/D/X/V/P)** while CH8/ROZDIL4/VYSNOVKY/SPATIAL_FACTORIAL
  still cite L-numbers — the ledger's own contract ("texts cite rows") is broken project-wide.
- Ownership recorded as an ephemeral agent session, not the author.

### 2.2 Scope — an engineering program on the defense's critical path (CRITICAL)
Prior baseline: article ≈ 1 week text + ~3 GPU-days; author cluster = the true critical path.
The plan's own numbers: ~60–120 GPU-h of "required + conditional-central" work beyond the
running queue, plus 1–2+ weeks CPU engineering, plus an article rewrite. **The RFT title
manufactures its own obligations**: put "Trust" in the title → owe a trust replication +
dose-response; make stride the central mechanism → owe the fixed-target control. Keep the
approved packaging framing and ~20–45 GPU-h of obligations evaporate with no loss to current
claims. No plan item advances the author cluster or the UA pass.

### 2.3 The blanket "development evidence" demotion (MAJOR)
Reframing the ENTIRE measured campaign as development-only pending a one-time holdout hands a
reviewer the sentence "the authors say their results are development evidence." Pre-registered
primary cells + two seeds + clip-cluster CIs is already above venue norm. Right resolution: keep
pre-registered, replicated claims as primary; run the (simplified) untouched confirmation as an
added robustness layer; label only the genuinely post-hoc items (reach bands) as post-hoc.
Also: the confirmation protocol makes union-mask MSE primary #1 — conflating confirmation with
metric reform (a failed confirmation could reflect the metric change, not the claim). Legacy
fg-MSE estimand must stay primary; new metrics co-secondary. Decouple from Gates 2–3.

### 2.4 Phase C over-block (MAJOR)
The evaluator critique is right, but the block discards Phase C's still-valid pre-registered
PRIMARY (paired fg-MSE equivalence, ±0.005 margins fixed before launch, computed by the eval
path verified 30/30) along with the weak secondaries — and **v2 disables equivalence testing**,
so the redesigned study cannot deliver the boundary/null conclusion Phase C exists to produce.
v2 is also a gate without a pipeline: no classifier-training CLI, no fixture builder for the
6-domain validation npz, no evaluator wiring `score_objects` to rollouts — the demanded markers
cannot currently be produced. **Hybrid recommended**: legacy schedule + fg-MSE equivalence
primary (pre-registered margins intact) + v2 Hungarian/persistent-ID object metrics as upgraded
secondaries; log as a dated amendment.

### 2.5 Pre-registration trail breached by the remediation itself (MAJOR)
The [EXPECTED-PhaseC] pre-registered block (margins + 2026-07-10 interpretation amendments) was
deleted from ROZDIL4 with **no dated amendment in SPATIAL_FACTORIAL** (untouched by 2689500).
Changing a registered protocol before the run is legitimate; doing it outside the trail is
exactly what the project's moat forbids. Restore as amended-not-deleted.

### 2.6 Control-layer contradictions (MAJOR)
(a) EXPERIMENT_CONTROL says "no live-queue file may be semantically changed before handoff" —
the same commit changed three of them (safely, but the rule contradicts the practice);
(b) c2 is authorized "RUN ALONE / old queue stopped" while the live queue runs c2 as step 3/5;
(c) with pipefail, the queue now ALWAYS terminates in engineered failure at Phase C (exit 97):
`NEWVISION_QUEUE_ALL_DONE` never prints and the optional QUEUE 5/5 probes (hier3-reg, E-320)
silently never run; (d) the stop watcher is not checked in (only its UTF-16 log);
(e) "fail-closed" holds against absence/corruption but not tampering/staleness — authorization
is an editable JSON, gate markers are grepped substrings whose hashes are never re-verified at
launch, and `gates/*.txt` is gitignored (no audit trail). Mitigating: the one dangerous path is
blocked unconditionally, and the .ps1 launchers throw before any GPU code.

### 2.7 Contamination tripwire still missing on the legacy path (MAJOR)
Distinct seeds between generated datasets are not programmatically enforced anywhere live; the
frame-0 hash audit added to run_phaseC_mmnist.sh is dead code behind exit 97; legacy
`gen_moving.py` still defaults `--seed 1234` for both splits. Policy text is the only guard.

### 2.8 Framing omissions and process gaps (MEDIUM)
- The B1-confound story never states that the confound is **stride-arm-specific**: all dense
  frontier arms (896–3,072), hier-min, and B2's strided-flat share the identical 4+4 target
  interface — the frontier and B2 are NOT interface-confounded. As written, the campaign sounds
  more contaminated than it is.
- Evidence-flow fragmentation: the new ledger cites S1-a/S1-b/regtwins numbers that have **no
  measured blocks in SPATIAL_FACTORIAL**, while VYSNOVKY/CH tables still carry stale
  [EXPECTED-B1-s1: queued] markers for results already measured. The swap-pass discipline
  (evidence block → text swap) is currently nobody's job. Biggest immediate process gap.
- Estimator triplication (rhvd.paired_ci / phasec_summary.cluster_ci / analysis.cluster_
  bootstrap_mean — currently identical semantics, verified); trajectory_yx stores top-left while
  object_metrics expects centroids (+14,+14 px footgun; use bbox); zero-IoU association window
  (~27 px) needs a protocol line; CI installs the CUDA torch wheel on ubuntu (~900 MB, timeout
  risk) and ignores requirements-cpu.txt; fg_precision=1.0 on empty predictions — never report
  precision solo.

## 3. RECOMMENDED RESOLUTION (author decisions)

1. **Let the queue finish through c2** (matches both agents' plans; verified safe).
2. **Reject the canonical-article swap and RFT title.** Keep CH1–CH8 + the approved packaging
   framing; apply the §3 wording contracts (bracket-not-point, regime-qualified c2, observability
   terminology, B1 configuration wording, seed-resolved reporting) as wording surgery to the
   existing chapters; if the R/F/T trichotomy appeals, use it as the discussion-section rubric —
   the naming benefit at zero GPU cost.
3. **Restore ledger continuity**: keep the new rows' content, restore L-IDs or add an alias
   table, restore the binding-rule provenance line.
4. **Phase C hybrid** (per §2.4) logged as a dated SPATIAL_FACTORIAL amendment; restore the
   ROZDIL4 pre-registration as amended-not-deleted; run it only if the synthetic cross-dataset
   claim is retained (it is NOT needed for the core article — Codex is right there).
5. **Run the simplified untouched confirmation** (legacy estimand + frozen reach bands primary;
   new metrics co-secondary; decoupled from the engineering gates). ~1 GPU evening.
6. **Do the missing swap-pass NOW**: SPATIAL_FACTORIAL measured blocks for S1-a/S1-b/regtwins/
   S2-trio + text swaps (B1 two-seed; B2 softened per amendment — s1 +0.0046 means "matches or
   exceeds" is the exact right wording; 512-reg edge +0.0639; reg sign-flip reported per-seed).
7. **Defer**: fixed-target B1 control (mechanism refinement, not validity rescue), trust
   dose-response (next paper; = the five-lens loop-96 recommendation), representation
   sensitivity (already author-deferred), RunSpec migration/refactor/packaging (post-defense).
   **Keep now**: artifact manifest + 18 GB backup (pre-existing five-lens item), the pytest
   suite, the citations → CH7 with the MemoBench near-collision paragraph.
8. **Ownership**: single-owner principle adopted, owner = the author; agents execute under the
   author's sign-off. Check the stop-watcher script into the repo.

## 4. SCIENCE UPSHOT OF THE FRESH NUMBERS (verified from npz)

The queue results reshape the shape-law story exactly as the pre-committed amendments
anticipated: **B1/edge results strengthened** (two-seed diffusion + a huge +0.0639 regression
point at 512 — the edge law is now the campaign's strongest cross-head result), while **B2's
temporal-beats-spatial headline weakened** to "matches or exceeds" (s1 margin +0.0046) with a
genuine seed-level sign flip under regression (+0.0107/−0.0104) — heterogeneity, not
equivalence, with a consistent head→spatial shift. The discipline held: nothing needs
retracting, only the already-written softened wording. The system worked.
