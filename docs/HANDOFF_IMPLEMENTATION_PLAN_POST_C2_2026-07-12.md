# Post-c2 handoff implementation plan

**Prepared:** 2026-07-12

**Purpose:** exact implementation specification for the moment the current c2-only run finishes
and the author transfers execution authority to Codex.

**Review status:** proposed for independent agent review before execution.

**Project root:** `C:\Users\leoni\Documents\projects\hierarchical-video-prediction`

## 1. Binding scientific and governance decisions

The implementation must preserve these decisions unless the author explicitly changes them:

1. The dissertation topic, approved purpose, seven tasks and the contents/topics of publications
   [A1]–[A3] do not change.
2. The existing CH1–CH8 packaging article remains the canonical fourth-article source. The
   reach–fidelity–trust draft is an explanatory/discussion resource, not an automatic title or
   identity replacement.
3. Context packaging is the manipulated design space. Reach, fidelity and trust explain measured
   outcomes:
   - **reach:** whether the package includes the relevant past;
   - **fidelity:** how much relevant information survives its representation;
   - **trust:** whether stored information remains usable under self-fed rollout.
4. Loopmom is the primary causal memory instrument. Moving-MNIST is retained as a standard
   boundary/generalization dataset. Counterfactual Dynamic LoopWorld is an additional future
   benchmark; none replaces the others.
5. Pre-registered, replicated primary evidence is not blanket-labelled “development-only.” Its
   adaptive test-pool limitation is disclosed precisely. Genuinely post-hoc reach-band analyses
   remain post-hoc until untouched confirmation.
6. The legacy B1 target-interface confound is stride-arm-specific. Dense frontier arms, hier-min
   and B2's temporal package have the matched four-coarse/four-fine target interface and must not
   be described as contaminated by B1's target-slot change.
7. Phase C's registered foreground-MSE endpoint is separable from its invalid legacy identity and
   trajectory metrics. Any Phase C continuation uses a dated hybrid amendment.
8. The author is the scientific/GPU authority. Agents execute only within explicit author-approved
   scope. “Current task” or another ephemeral session is never recorded as the ultimate owner.
9. No new GPU study begins immediately at handoff. Evidence preservation, result integration,
   direct text repair and CPU validation come first.
10. Existing live-generated `docs/CWSTYLE_TABLE.md` and Figure 12 belong to the current result
    owner until the handoff explicitly includes them.

## 2. Authoritative resources and recommended review order

An independent reviewer should read these in order.

### Tier A — decisions and adversarial review

1. `docs/HANDOFF_IMPLEMENTATION_PLAN_POST_C2_2026-07-12.md` — this implementation plan.
2. `docs/REVIEW_RFT_REMEDIATION_2026-07-11.md` — independent review of commits `a2438cf` and
   `2689500`; especially governance, Phase C and scope findings.
3. `docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md` — complete experiment dependency
   order, costs and evidence products; commit `055d126`.
4. `docs/PROJECT_DISSERTATION_ARTICLE_REVIEW_DOSSIER_2026-07-11.md` — project-wide evidence and
   reproducibility map; commit `a2438cf`.
5. `docs/REVIEW_FIVE_LENS_2026-07-10.md` — earlier five-lens scientific review.

### Tier B — claims and preregistration

6. `docs/SPATIAL_FACTORIAL.md` — historical preregistration and measured-result trail.
7. `docs/CLAIM_LEDGER.md` — current refined content, but its changed IDs must be reconciled with
   existing L1–L10 references.
8. `docs/EXPERIMENT_CONTROL.md` and `experiments/authorization.json` — current control layer;
   both require post-handoff governance correction.
9. `docs/UNTOUCHED_CONFIRMATION_PROTOCOL_2026-07-11.md` and
   `docs/PHASE_C_V2_PROTOCOL_2026-07-11.md` — useful protocol drafts that require the changes
   specified below.

### Tier C — article and dissertation

10. `docs/draft/CH1_INTRO.md` through `docs/draft/CH8_CONCLUSION.md` — canonical, content-complete
    fourth-article sources.
11. `docs/draft/ARTICLE_SKELETON.md` — historical evidence/source map.
12. `docs/draft/ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md` — discussion/reframing resource, not
    canonical replacement.
13. `docs/dissertation/rewrite/00_FRONT_MATTER.md` through `07_REFERENCES.md` — dissertation source.
14. `docs/dissertation/REACH_FIDELITY_TRUST_SYNTHESIS_2026-07-11.md` — useful layered-programme
    synthesis to integrate selectively.
15. `docs/dissertation/DISS_DIGEST.md` — locked structural/author decisions.

### Tier D — code and live evidence

16. `hvp_rev/rhvd.py`, `hvp_rev/long_eval.py`, `hvp_rev/mmnist_metrics.py`,
    `hvp_rev/phasec_eval.py`, `hvp_rev/phasec_summary.py` — legacy scientific path.
17. `hvp_rev/evaluation/`, `hvp_rev/provenance.py`, `hvp_rev/arm_registry.py`,
    `hvp_rev/fixed_target_control.py`, `hvp_rev/rollout_artifacts.py` — prepared remediation code.
18. `tests/` and `scripts/run_cpu_gate.py` — current CPU validation.
19. `logs_rhcp/newvision_queue.log`, `logs_rhcp/frontier_c2_60k.log`,
    `logs_rhcp/stop_before_phasec.log` — current queue and watcher evidence.
20. `scripts/run_newvision_queue.sh`, `scripts/run_frontier_c2_60k.sh`,
    `scripts/run_phaseC_mmnist.sh` — queue semantics and legacy block.

## 3. Handoff prerequisites supplied by the current owner

Before Codex changes any scientific code or result text, the handoff must contain:

- final c2 log and comparison output;
- c2 completion marker and exit status;
- every new checkpoint, NPZ, table, figure and log path;
- exact commands and start/end times;
- retry/resume history, including W&B step warnings and legacy RNG limitations;
- current branch, HEAD and full worktree status;
- confirmation that the old master controller and all children are absent;
- confirmation that no Phase C process or launch marker exists;
- project-scoped GPU process/permission state; unrelated GPU activity need not be idle;
- queue-lock state;
- explicit treatment of the live CW table, Figure 12 and independent review file;
- explicit author message granting Codex time-bounded execution autonomy for the next approved
  tranche while the author retains scientific and GPU-launch authority.

If any of these is unavailable, Stage H0 records the gap. Missing metadata does not justify
reconstructing facts from memory.

## 4. Stage H0 — accept and verify the handoff

**Estimated time:** 30–90 minutes.

### Objectives

Establish a verified, immutable boundary between the old queue and all future work.

### Read-only checks

1. Record current time, branch, HEAD and `git status --short`.
2. Enumerate relevant process trees; verify no old queue, c2, Phase C or orphan trainer remains.
3. Verify no project process owns the GPU and project launch permission is inactive; record but do
   not disturb unrelated GPU activity.
4. Verify the queue lock is absent or explain why it remains.
5. Verify `S2TRIO_DONE`, c2 completion and watcher stop markers.
6. Confirm both `_60k` NPZs align on clip/horizon keys and contain the expected metadata fields.
7. Record c2 checkpoint step, size and modification time.
8. Compare the handoff artifact list against filesystem changes since commit `055d126`.

### Output

Create `docs/HANDOFF_POST_C2_<date>.md` containing:

- ownership statement;
- process/GPU/lock snapshot;
- git/worktree snapshot;
- artifact inventory;
- known caveats;
- authorization state;
- next allowed action: H1 only.

### Acceptance gate H0

- no GPU process in project scope;
- no Phase C launch;
- c2 result artifacts readable;
- live outputs assigned to an owner;
- time-bounded author delegation state recorded.

Failure means stop and ask the author; do not infer authority or start cleanup.

## 5. Stage H1 — preserve and integrate the completed evidence

**Estimated time:** 6–12 hours.

### H1.1 Immutable evidence backup

Use `scripts/build_artifact_manifest.py` or a corrected successor to create a manifest covering:

- all headline NPZ files;
- seed-2 trio artifacts;
- c2 30k and 60k artifacts;
- associated checkpoints;
- source logs;
- current generated tables/figures;
- preregistration documents and producing commits.

Each manifest entry records path, bytes, SHA-256, claim IDs, producing command/run identity and
whether the artifact is canonical, diagnostic or transient. Copy the evidence set to the
author-approved backup/archive location before refactoring output paths.

### H1.2 Integrate missing measured blocks

Edit `docs/SPATIAL_FACTORIAL.md` first, preserving chronological history. Add dated measured blocks
for:

- S1-a B1 seed 1 (+0.0338);
- S1-b B2 seed 1 (+0.0046 and the precommitted softened interpretation);
- regression B1 seed 0 (+0.0639 with interval);
- regression temporal/spatial signs (+0.0107/−0.0104 by seed);
- seed-2 diffusion trio, using its newly completed effects;
- c2 optimizer-regime outcome.

Each block must include artifact path, exact command/regime, effect convention, CI, sample/cluster
count, preregistered interpretation branch and honest confounds.

### H1.3 Perform the missing swap pass

After the evidence blocks exist, replace stale result markers in:

- `docs/draft/CH1_INTRO.md`;
- `docs/draft/CH4_RESULTS.md`;
- `docs/draft/CH8_CONCLUSION.md`;
- `docs/dissertation/rewrite/00_FRONT_MATTER.md` where relevant;
- `docs/dissertation/rewrite/05_ROZDIL4.md`;
- `docs/dissertation/rewrite/06_VYSNOVKY.md`;
- generated comparison tables and figures.

Never update a result sentence before its evidence block is committed.

### H1.4 c2 interpretation

Describe c2 as:

> matched-example, changed-batch/update/gradient-noise/EMA-regime robustness control.

Do not call it twice the data, convergence, asymptotic performance or an exact edge locator.

### Acceptance gate H1

- all fresh results appear in `SPATIAL_FACTORIAL` before derived text;
- no stale `EXPECTED-B1/B2/REG/S2/C2` marker survives in canonical result statements;
- every fresh number resolves to a hashed artifact;
- raw evidence is backed up;
- live output modifications are committed separately from structural code.

## 6. Stage H2 — repair governance and source-of-truth continuity

**Estimated time:** 4–8 hours.

### H2.1 Restore canonical article governance

Remove “historical/noncanonical” warning headers from CH1–CH8. Restore the approved packaging article
as canonical. Retain `ARTICLE_REACH_FIDELITY_TRUST_2026-07-11.md`, but relabel it explicitly as a
discussion/restructuring resource whose language may be merged through normal author review.

No title change is made without author/supervisor approval.

### H2.2 Restore claim-ledger compatibility

Preserve the refined content of the new claim ledger while restoring compatibility with L1–L10.
Preferred implementation:

- retain L1–L10 as stable public IDs;
- add subclaims such as L1-R (reach mechanism), L2-F (fidelity/saturation), L6-T (trust boundary);
- map retired P1–P7 wording in a separate “retired interpretations” table;
- restore the binding-rule provenance line;
- update all references atomically.

An alias table is acceptable if it is simpler, but no canonical text may point to a nonexistent
claim row.

### H2.3 Restore Phase C amendment history

Restore the original registered Phase C block in ROZDIL4 as historical preregistration, followed by
a dated pre-launch amendment explaining:

- foreground-MSE primary retained narrowly;
- legacy identity/trajectory interpretation invalidated;
- v2 complete/object metrics required for factor claims;
- Phase C remains optional to the core article.

Add the same dated amendment to `docs/SPATIAL_FACTORIAL.md`. Git history alone is not the scientific
amendment trail.

### H2.4 Correct ownership and plan hierarchy

Update `docs/EXPERIMENT_CONTROL.md`:

- owner = author;
- Codex = sole executing agent only during explicit delegated intervals;
- current execution order = `EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md`;
- older master/RFT plans remain rationale/history rather than conflicting launch authority.

### Acceptance gate H2

- CH1–CH8 canonical status restored;
- zero dangling claim IDs;
- Phase C amendment exists in the preregistration trail and dissertation;
- author ownership is explicit;
- plan/source hierarchy is unambiguous.

## 7. Stage H3 — establish a safe post-c2 code baseline

**Estimated time:** 1–2 working days.

This stage has two independently gated parts. **H3a** is the evaluation-critical path required
before checkpoint-only E2: legacy readers, correct record-level metrics, immutable evaluation
identity, dataset-purpose enforcement and comparison checks. **H3b** is required before any new
training: full RunSpec integration, explicit RNG/sampler resume, W&B/run identity, arm registry and
controller redesign. H3b does not block evaluation-only confirmation, but no E3/E4/E5/E7 training
may begin without it.

### H3.1 Freeze legacy readers

Document legacy checkpoint and NPZ schemas. Add readers/tests that can regenerate every existing
table after future output-path changes. Do not rename or rewrite old artifacts.

### H3.2 Correct run identity for new studies

Integrate `hvp_rev/provenance.py` incrementally:

1. Resolve a `RunSpec` before model/data construction.
2. Include dataset, actions/positions, AE, arm config, seed, examples, updates, batch, precision,
   NFE, warm-up, rollout horizon, evaluation population, metric version, commit and dirty hash.
   Resume compatibility must also compare optimizer, scheduler, EMA and sampler mappings rather
   than checking identity fields while accepting changed training dynamics.
3. Write new runs under content-addressed directories.
4. Refuse completed-run overwrite.
5. Keep a compatibility flag for explicitly reading legacy checkpoints; never silently resume
   them as v2 runs.

### H3.3 Explicit, trajectory-reproducible resume

For new runs save/restore:

- model, optimizer, scheduler/scaler and EMA;
- update/example counters;
- Python, NumPy and Torch CPU/CUDA RNG states;
- data-sampler/order state;
- RunSpec and latent normalization statistics.

Require `--resume`; remove implicit checkpoint discovery for v2 runs. Reject scientific-field
mismatches.

### H3.4 Central arm registry

Make `hvp_rev/arm_registry.py` the new-run source for context/target interfaces and generated
configuration tables. Preserve a tested legacy adapter until old scripts are retired.

The registry must separate:

- context temporal stride;
- target temporal stride;
- context and target slots/tokens;
- fine-history stride;
- spatial level;
- loss weights;
- context reach;
- head and sampler.

### H3.5 Complete rollout artifact path

Integrate `hvp_rev/evaluation/` and `hvp_rev/rollout_artifacts.py` so each record saves:

- clip/horizon and observability labels;
- full, non-empty foreground, background and union MSE;
- foreground precision/recall/F1;
- component counts, misses/extras, IoU and centroid distance;
- true per-record sharpness;
- scoring/identity coverage where available;
- metric implementation version.

Retain the registered legacy foreground metric as a separate field for continuity.

### H3.6 Correct known implementation footguns

- Fix W&B identity for changed step regimes so 60k runs do not reuse a 30k monotonic step stream.
- Make legacy `gen_moving.py` require an explicit seed or add a split-seed/collision tripwire.
- Resolve the `trajectory_yx` top-left-versus-centroid mismatch: store unambiguous names and derive
  centroids from boxes in the evaluator.
- Document/freeze the zero-IoU distance association threshold.
- Consolidate the three cluster-bootstrap implementations behind one tested function while keeping
  output compatibility.
- Report empty-frame precision together with recall/counts; never present the `precision=1` empty
  convention alone.
- Make gate markers content/hash verified rather than substring-only; preserve an audit record.
- Configure CI to install a CPU Torch wheel and use the locked CPU requirements.

### H3.7 Queue cleanup after current completion

Replace engineered exit-97 flow with an explicit clean stop state:

- old master queue is archival;
- new queue ends at `PRE_PHASE_C_QUEUE_DONE` unless Phase C is separately authorized;
- optional probes are separate explicit jobs;
- check in the stop/watcher logic or replace it with a deterministic guarded controller;
- never rely on an untracked watcher for scientific safety.

### CPU tests required

- legacy artifact reading/golden table reproduction;
- RunSpec identity and mismatch rejection;
- uninterrupted versus resumed CPU trajectory;
- smoke outputs cannot collide with canonical output;
- arm registry parity for all legacy arms;
- fixed-target B1 batch/model/buffer phasing;
- complete metric edge cases;
- comparison metadata mismatch rejection;
- queue/gate dry-run tests;
- dataset split/collision checks.

### H3b numerical-parity gate

E4a completes a factorial begun on the legacy trainer, so provenance improvements must not silently
create a new scientific regime. On CPU and fixed seeds, compare the legacy and H3b paths for:

- sampled batch indices and tensors;
- model construction and initial parameters;
- loss inputs, weights and first fixed-seed losses;
- optimizer, warm-up, clipping and EMA updates;
- sampler/NFE behavior and saved evaluation metadata.

Require exact equality where deterministic and a predeclared tight tolerance where kernels differ.
If parity fails, classify H3b outputs as a new regime and either rerun the necessary factorial
partners under H3b or retain the legacy trainer inside a guarded compatibility path. Never mix the
two regimes as if they were exchangeable.

### Acceptance gate H3

- default CPU suite passes with CUDA hidden;
- legacy result regeneration remains unchanged;
- new runs cannot overwrite or silently resume;
- no canonical trainer change lacks a golden test;
- H3b numerical parity is demonstrated or the new regime/rerun consequence is recorded;
- queue can stop cleanly before Phase C.

## 8. Stage H4 — implement the correct untouched loopmom confirmation path

**Estimated preparation:** 6–12 CPU hours before GPU authorization.

### H4.1 Correct dataset generator/launcher

The current `scripts/generate_confirmation_set.ps1` incorrectly generates Moving-MNIST for a
protocol whose primary claims concern loopmom. Replace or split it:

- `generate_loopmom_confirmation.ps1` calls the loopmom generator with the exact primary operating
  point and a new required seed;
- saves clips, actions and positions;
- writes generator config/commit, source glyph pool, shapes and hashes;
- sets `purpose=confirmation_holdout`;
- refuses overwrite;
- is rejected by training and monitoring loaders.

Moving-MNIST confirmation generation remains a separate Phase C utility.

### H4.2 Freeze confirmation analyses

Primary continuity endpoints:

- registered non-empty foreground error for the selected central comparisons;
- exact package effect convention and clip-bootstrap implementation.

Frozen mechanistic primary endpoints:

- `hier-min` versus `flat-min`, `14 < dt <= 48`;
- legacy B1 configuration versus `flat-short`, `8 < dt <= 48`.

Complete corroborating endpoints:

- full/background/union errors;
- foreground precision/recall;
- component misses/extras/IoU;
- coverage;
- empty-frame stratum.

The registered metric and metric reform are reported side by side so a difference cannot be
misrepresented as failed replication.

### H4.3 Full-frontier evaluation configuration

Create explicit no-training RunSpecs for every available frontier checkpoint and seed. Confirm
compatibility of dataset/AE/stats/warm-up/NFE before evaluation. Freeze table and figure shells.

Selected central pairs and their pair-specific reach bands are confirmatory primary hypotheses.
The complete frozen frontier is a family-level secondary confirmation. Use a simultaneous
max-statistic clip-cluster bootstrap for the correlated frontier contrasts; a predeclared Holm
procedure is an acceptable simpler fallback. c2 regimes remain separate, complete metrics remain
corroborating, and no post-result frontier cell may be promoted to a new primary.

### H4.4 Holdout access controls

- explicit `--confirmation-eval` switch;
- append-only access log with manifest and RunSpec hashes;
- no training-time or qualitative grid path accepts holdout purpose;
- official report command is deterministic and single-use in interpretation, though technical
  reruns caused by documented process failure remain auditable.

### Acceptance gate H4

- CPU synthetic/dev dry-run passes end to end;
- fresh loopmom manifest validated;
- analysis/table shells frozen;
- holdout has not been opened for model/threshold selection;
- author explicitly authorizes E2 GPU evaluation.

## 9. Stage H5 — direct fourth-article integration

**Estimated time:** 1–3 working days for the first complete pass, followed by result updates.

The implementation edits CH1–CH8 directly. The RFT draft is mined for language but does not replace
them.

H5 has two deliverables: (1) corrected, canonical CH1–CH8 source chapters preserving the complete
scientific record, and (2) a focused venue-length manuscript assembled from them. The assembled
paper should use approximately six main sections: problem/instrument; package definitions and cost;
frontier; spatial/temporal/head factorial; trust and external boundaries; related work,
limitations and conclusion. Detailed EBT and reversibility material moves to appendices,
supplement or dissertation unless the selected venue requires it in the main text.

### CH1 — Introduction

- Retain the approved packaging title candidates and author gate.
- State the bounded-context allocation question.
- Introduce reach/fidelity/trust as explanatory questions, not assumed laws.
- Update the abstract with measured B1/B2/regression/seed-2/c2 results only after H1.
- Replace exact crossing/edge language with tested-configurations wording.
- Scope B1 confound to strided-coarse target coupling.
- Narrow novelty against 2026 memory/retrieval benchmarks.

### CH2 — Testbed and evaluation

- Use “exact geometric observability,” not information-theoretic necessity.
- Add the pair-specific reach-band derivation.
- Separate registered foreground, complete frame and object metrics.
- Explain empty-frame and false-positive handling.
- Disclose adaptive test-pool use and untouched confirmation without blanket-demoting registered
  replicated evidence.
- Add the future counterfactual-world condition required for true necessity claims.

### CH3 — Mechanism study

- Replace generic “matched compute” with exact matched dimensions.
- Separate diffusion coverage, regression best-of-1 and EBT negative evidence.
- Report total NFE/sampling allocation.
- Compress EBT to the amount needed for the article; preserve full detail in dissertation/supplement.

### CH4 — Main results

- Integrate all seed-1, regression, seed-2 and c2 results.
- Show per-seed effects rather than pooled cancellation.
- Present 1,792 and 2,432 as tested bracket endpoints by regime.
- Treat legacy B1 as a replicated configuration result; reserve context-stride causality for E3.
- State explicitly that dense frontier and B2 target interfaces are matched.
- Separate spatial, temporal and depth package results.
- Add reach-differential effects as post-hoc/current-pool and confirmation-frozen.
- Treat trust as a measured seed-scoped boundary until E5.

### CH5 — Generalization/boundaries

- Keep short Moving-MNIST as a measured negative/boundary control.
- Keep TaxiBJ as competence/economy evidence, not memory re-entry proof.
- Present Phase C as optional hybrid and remove projected outcomes.
- Explain the distinct roles of loopmom, Moving-MNIST and Dynamic LoopWorld.

### CH6 — Reversibility

- Preserve mathematical and empirical content.
- Position reversibility as a resource/correctness enabler rather than the cause of Chapter 4
  package gains.
- Scope novelty against reversible RNNs and distinguish time/depth/inner-computation axes.
- Consider moving detailed proofs/panels to supplement in the journal version.

### CH7 — Related work

- Integrate verified WorldPack, MBench, CaR/SceneFly, GIM-World, MemLearner, MemoBench and Self
  Forcing references.
- Add a specific MemoBench near-collision paragraph: dynamic disappear/reappear is close to the
  proposed extension; this work's distinction is controlled budget packaging and exact synthetic
  observability/counterfactual design.
- Include a protocol table rather than claiming no prior memory-focused evaluation.

### CH8 — Conclusion

- Restore valid L-claim references.
- Use empirical regularity/boundary language rather than universal laws.
- Summarize reach, fidelity and trust as the explanation of the package phase diagram.
- Distinguish current evidence, confirmation, dissertation-complete controls and next-paper work.

### Article tables/figures to implement

1. configuration and total-cost table;
2. reach–fidelity frontier by seed/regime;
3. package × head × seed factorial;
4. reach-differential table;
5. trust horizon/refresh figure;
6. complete-metric decomposition;
7. CW-VAE/recent-memory protocol comparison;
8. frozen qualitative re-entry/failure strips.

### Acceptance gate H5

- CH1–CH8 are internally consistent and canonical;
- every result sentence maps to ledger/evidence;
- no unresolved result projection remains in main prose;
- abstract/conclusion scopes match tables;
- article can be assembled without the separate RFT draft;
- a venue-length six-section manuscript exists in addition to the canonical source chapters.

## 10. Stage H6 — direct dissertation integration

**Estimated time:** 3–7 working days for technical integration; Ukrainian/author inputs may take
longer and can overlap later experiments.

### Front matter and introduction

Files: `00_FRONT_MATTER.md`, `01_VSTUP.md`.

- Preserve the approved purpose, object, subject and seven tasks.
- Preserve [A1]–[A3] publication content and map [A4] accurately.
- Integrate the layered-programme synthesis without implying one already-complete monolith.
- Synchronize novelty propositions between front matter and introduction.
- Scope “for the first time” to the controlled methodology/instrument, not hierarchical memory as
  a general idea.
- Retain author-dependent NDR/publication/admin fields as explicit blockers.

### Chapter 1 — systems analysis and criteria

File: `02_ROZDIL1.md`.

- Update 2026 long-memory/retrieval/benchmark literature.
- Present flat context, hierarchy, retrieval and rollout training as alternatives in a systems
  design space.
- Make trajectory consistency operationally consistent with complete metrics.
- Remove any necessity/novelty statement contradicted by MemoBench/CaR/WorldPack.
- Preserve the systems-analysis taxonomy and [A3] criteria contribution.

### Chapter 2 — entropy/RoPE/reversibility methods

File: `03_ROZDIL2.md`.

- Preserve [A1]/[A2] mathematical content.
- Clarify V-BLT/RoPE versus the learned embeddings used by the final predictor.
- Scope reversible-method novelty against prior reversible RNNs.
- Keep exact inverse/gradient/memory claims only where mathematically or numerically established.
- Separate reversible resource optimization from predictive package causality.

### Chapter 3 — hierarchical model and instrument

File: `04_ROZDIL3.md`.

- Update configuration definitions from the typed registry.
- Separate context and target stride in equations/tables.
- Define context tokens, target tokens and total sequence explicitly.
- Add complete metric equations/populations and pair-specific reach bands.
- Use geometric observability language.
- Document RunSpec/artifact provenance as part of the intelligent technology/software complex.
- Retain the model, ring buffer and testbed as the implemented integration point.

### Chapter 4 — experiments and application

File: `05_ROZDIL4.md`.

- Integrate H1 results chronologically from `SPATIAL_FACTORIAL`.
- Update mechanism, frontier, shape, seed-2 and c2 sections.
- Distinguish B1 configuration effect from matched-interface frontier/B2 evidence.
- Add untouched-confirmation methods/results when E2 lands.
- Update trust section only to the evidence level achieved by E5.
- Restore Phase C original+amended trail; integrate results only if E6 runs.
- Separate measured OpenSTL/TaxiBJ evidence from future shells.
- Generate the configuration, frontier, factorial, metric and prior-art tables from artifacts.
- Preserve implementation acts wording; do not invent deployment evidence.

### General conclusions

File: `06_VYSNOVKY.md`.

- Map conclusions one-to-one to the seven tasks.
- Separate mathematical proofs, implemented methods and empirical regularities.
- Update seed/result wording after H1/E2/etc.
- Avoid claiming full executable integration unless a demonstrator exists; otherwise state layered
  integration and identify the model/testbed/software complex that is executable.
- Keep natural/counterfactual extensions as future work unless measured.

### References

File: `07_REFERENCES.md`.

- Verify every `[VERIFY]` entry against primary PDF/publisher records.
- Replace all `[AUTHOR]` publication entries using author-supplied bibliographic facts.
- Add recent works with consistent transliteration/format.
- Rebuild binding table and numbered references at assembly.

### Ukrainian and administrative pass

- Translate/edit final English working text into formal Ukrainian consistently.
- Resolve NDR numbers, publication metadata, defense council requirements, page/figure/table counts,
  implementation-act appendix letters and author contribution statements.
- This is a core completion workstream, not cosmetic cleanup.

### Acceptance gate H6

- topic/purpose/tasks unchanged and synchronized;
- [A1]–[A3] content preserved;
- all Chapter 4 claims map to artifacts;
- conclusions map to tasks;
- no false integration or universal hierarchy claim;
- Ukrainian/admin blocker inventory assigned to the author.

## 11. Stage H7 — prepare the next experiment tranche

**Estimated time:** 4–10 CPU hours per tranche, after H3/H4.

The governing sequence is `docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md`.

The experiments form a claim-driven decision tree, not a mandatory E2–E10 sequence. For each
authorized tranche:

1. Write/finalize the JSON/Markdown design.
2. Declare required and reused checkpoints/data.
3. Declare arm/run count, seeds, examples, updates, target interface and NFE.
4. Declare primary/secondary/diagnostic endpoints and populations.
5. Declare result-independent interpretation branches and stop rule.
6. Produce RunSpec hashes and output destinations.
7. Run CPU data/model/evaluator/oracle/resume smoke tests.
8. Run a short timing/VRAM probe where architecture changes.
9. Update authorization status only after author approval.
10. Launch one guarded queue; record PID/process tree and command.
11. Record the authorized maximum and actual GPU-hour consumption. GPU hours are wall-clock time
    while any project process owns the GPU, including pilots, evaluation, failures and retries.

### Specific preparation corrections

- E2: use loopmom confirmation, not Moving-MNIST.
- E3a: run only if the article retains a causal stride/temporal-rate claim; validate exact reuse of
  flat-short checkpoints and separate context from target stride. Defer E3b.
- E4a: run after H3b. E4b can trigger only from preregistered training-seed heterogeneity or an
  across-seed precision target—not clip-bootstrap uncertainty—and must replace another allocation.
- E5a/E5b: reproduce the trust boundary before diagnostics; defer rollout-aware E5c.
- E6: build the actual classifier-domain fixture/evaluator, but launch only if the assembled article
  needs the cross-dataset boundary.
- E7: retain alternate-AE and sampler/NFE robustness; defer simple-codec proliferation.
- E8: prioritize classical retrieval. Run a CW-VAE bridge only after protocol-equivalence audit;
  defer learned retrieval.
- E9: begin only after the current article is internally frozen, with a CPU oracle, pilot, frozen
  minimal factorial and 100-hour cap. Defer extensions if the question cannot be answered cleanly.
- E10: separate flagship future work outside the current portfolio.

### GPU budget pools and stop rules

- closure envelope: at most **181 GPU hours**;
- optional E8c/E9 extension envelope: at most **150 GPU hours**;
- protected failure/retry reserve: **50 GPU hours**;
- expected maximum: approximately **381 GPU hours**, below the absolute 400-hour ceiling;
- reserve cannot fund discretionary experiments;
- E4b replaces/reduces E6, E8c or E9 rather than adding to the ceiling;
- stop when claims are resolved, even if budget remains.

At H0, measured c2 consumption replaces its planning allowance. Any excess is absorbed by reducing
a later conditional allocation; it does not enlarge the closure envelope.

### Acceptance gate H7

- gate can be satisfied through real generated artifacts, not impossible placeholders;
- all launch configs resolve and dry-run;
- author approves cost and scope;
- the tranche maximum fits the remaining non-reserve envelope;
- exactly one controller is prepared.

## 12. Stage H8 — table, figure and artifact generation system

**Estimated time:** 1–3 working days, overlapping text integration.

### Required generated tables

1. Configuration/cost: levels, spatial grids, temporal strides, context reach, context/target/total
   tokens, parameters, examples, updates, NFE, runtime, VRAM.
2. Claim/evidence: contrast, seed, population, metric, effect/CI, evidence status, artifact hash.
3. Frontier: tested budget, reaches, regime and winner; no interpolated exact crossing.
4. Shape/head factorial: flat/spatial/temporal/depth × diffusion/regression × seed.
5. Reach-differential: pair geometry, causal band, effect/CI, post-hoc/confirmation status.
6. Trust: teacher/self-fed/refresh/corruption by horizon and seed.
7. Complete metrics: foreground/background/union/full, detection/count/coverage.
8. Prior art: CW-VAE, DLH, TECO, WorldPack, CaR, MemLearner, GIM-World, MemoBench, Self Forcing and
   this study.

### Figure rules

- all figures generated from manifest-addressed artifacts;
- seeds shown, not hidden in pooled curves;
- optimizer regimes separated;
- confidence/coverage displayed where relevant;
- no unique crossing line through two sign-opposed points;
- qualitative selections use a frozen deterministic rule;
- filenames include study/run identity; no canonical strip overwrite.

### Artifact map

Generate a machine-readable and Markdown claim-to-artifact index. Every article/dissertation table
cell and headline number must be traceable to:

`claim → table/figure generator → result NPZ → checkpoint/RunSpec → data/AE manifest → commit`.

### Acceptance gate H8

- regenerating tables/figures produces no unreviewed numeric diff;
- manuscript contains no hand-entered result absent from the snapshot source;
- artifact hashes resolve;
- old figures remain archived.

## 13. Stage H9 — quality assurance and reviewer package

**Estimated time:** 1–2 working days per major release.

### Code gates

- CUDA-hidden pytest suite;
- compile/static checks;
- clean-environment CPU install/smoke;
- legacy result regeneration;
- gate dry runs;
- artifact manifest verification;
- no hard-coded developer path in new scientific commands.

### Scientific gates

- claim ledger cross-reference audit;
- preregistration-to-result trace;
- post-hoc versus confirmatory labels;
- comparison-cost vocabulary audit;
- empty-frame/coverage audit;
- seed and cluster-count audit;
- reviewer-attack checklist from the dossier/reviews.

### Text gates

- canonical article/dissertation source check;
- no stale `EXPECTED/PENDING` in final assembled prose;
- no unresolved author/reference marker except explicitly assigned author blockers;
- abstract/conclusion/table consistency;
- dissertation purpose/tasks/novelty/conclusions synchronization;
- Ukrainian terminology and numbering pass.

### Reviewer package

Provide the next reviewer:

1. handoff snapshot;
2. this plan with stage statuses;
3. commits grouped by stage;
4. CPU validation output;
5. changed-file and claim-impact map;
6. artifact manifest and generated table diff;
7. unresolved decisions/blockers;
8. next experiment config and authorization state.

## 14. Implementation critical paths

### Before any new GPU run

For checkpoint-only E2, required: H0 → H1 → H2 → H3a → H4 → E2 authorization. Before any new
training, H3b and its numerical-parity gate are additionally mandatory.

**Estimated elapsed work:** approximately **2–5 focused working days**, depending on legacy-output
integration and article swap scope. Article/dissertation work can continue while E2 evaluates.

### Before the canonical article can be frozen

Required: H1, H2, H5, H8, H9 plus E2 results and any E3/E4/E5 evidence retained as headline.

### Before the dissertation can be frozen

Required: H1, H2, H6, H8, H9; author-supplied administrative data; Ukrainian pass; all experiments
whose claims remain in Chapter 4.

### Post-defense/research-extension path

Broad refactoring, learned retrieval, counterfactual Dynamic LoopWorld and natural/3-D transfer may
continue after the current article/dissertation are internally complete. Their code can be prepared
now, but unmeasured claims do not enter final conclusions.

## 15. Commit and change-isolation strategy

Use small commits in this order:

1. `handoff: record post-c2 state and artifact inventory`
2. `evidence: manifest and integrate seed2/c2 measured blocks`
3. `governance: restore canonical article ledger and prereg trail`
4. `eval: add legacy-compatible complete rollout metrics`
5. `provenance: enable RunSpec for new runs`
6. `confirmation: implement loopmom holdout pipeline`
7. `article: integrate measured packaging and RFT discussion`
8. `dissertation: integrate evidence and scoped synthesis`
9. `reporting: generate tables figures and artifact map`
10. per-experiment design/code/result/text commits thereafter.

Never combine raw result integration, trainer refactoring and manuscript reinterpretation in one
commit. Never stage unrelated live outputs by broad `git add`.

## 16. Known risks the reviewer should challenge

1. Does the plan preserve the author-approved article identity rather than repeating the canonical
   RFT overreach?
2. Are pre-registered replicated results described precisely despite adaptive evaluation-pool use?
3. Does confirmation test loopmom rather than accidentally generating Moving-MNIST?
4. Are registered and new metrics separated clearly?
5. Is the B1 confound scoped only to strided-coarse target coupling?
6. Does the Phase C amendment preserve the historical preregistration and narrow foreground
   endpoint while blocking invalid factor claims?
7. Can every gate marker be produced and cryptographically validated by implemented code?
8. Does RunSpec integration preserve legacy artifact readability?
9. Are W&B step/regime identities corrected?
10. Does the dataset schema distinguish top-left coordinates, centers, boxes and visibility?
11. Are bootstrap/cluster semantics unified without changing past intervals?
12. Are article/dissertation numbers generated rather than hand-copied?
13. Are [A1]–[A3] and dissertation tasks genuinely unchanged?
14. Is the Ukrainian/admin path treated as first-class completion work?
15. Does each future experiment add a distinct estimand rather than merely more configurations?
16. Does GPU-hour accounting include pilots, evaluation, failures and retries, and does every
    tranche fit the 181/150/50 envelope without converting reserve into discretionary work?
17. Did H3b preserve numerical parity with legacy partners, or were affected results correctly
    classified and rerun as a new regime?

## 17. Requested independent-review response format

The reviewing agent should return:

1. **Overall verdict:** approve / approve with amendments / reject.
2. **Blocking findings:** issues that must be fixed before H0/H1 execution.
3. **Stage-by-stage findings:** H0 through H9, classified critical/major/minor.
4. **Scientific alignment:** whether context packaging and reach/fidelity/trust are separated
   correctly.
5. **Governance check:** article identity, ledger IDs, preregistration and ownership.
6. **Code feasibility:** whether proposed integrations can be implemented without invalidating
   legacy evidence.
7. **Text feasibility:** whether article/dissertation edits preserve approved content/tasks.
8. **Missing resources/files:** anything the reviewer could not verify.
9. **Recommended execution changes:** reorder, merge, split or remove stages.
10. **Explicit authorization recommendation:** whether Codex should begin H0/H1 after c2 handoff.

## 18. Definition of successful handoff implementation

The handoff implementation is successful when:

- the current experiment evidence is immutable and fully integrated;
- the approved article is canonical and directly corrected;
- the dissertation retains its topic/tasks/publications and receives direct scoped integration;
- claim IDs and preregistration history are continuous;
- new scientific runs use immutable provenance, explicit resume and complete metrics;
- central confirmation uses a fresh loopmom holdout;
- Phase C can only run under a valid hybrid protocol;
- new training cannot begin until H3b numerical parity is demonstrated or its rerun consequence is
  accepted;
- the experiment portfolio remains below 400 project GPU hours with 50 hours protected for
  failures and retries;
- all tables/figures are artifact-generated;
- the author owns launch authority and Codex executes one reviewed queue at a time;
- the next reviewer can trace every decision, number and code path without relying on chat memory.
