# Independent dissertation, live A4, and integrated implementation review

Review date: 5 September 2026. Operational snapshot: approximately 07:53 Europe/Kyiv.

**Verdict: continue A4; fix the issues below; do not yet treat the expanded dissertation or integrated experiment infrastructure as submission-ready. No observed finding requires restarting the current A4 training.**

The dissertation has a coherent research problem and substantial scientific substance. Its strongest contribution is a controlled study of what retained video context can recover under a constrained budget, including negative evidence and explicit limits. It is less convincing as evidence of a generally superior, integrated, long-horizon video predictor. The new experiments may strengthen that case, but they have not yet supplied that evidence.

The four passing manuscript gates are useful inventory and consistency checks. They do not establish scientific correctness, adequate experiments, effective gradient tests, or final page layout. I independently reproduced all four passes and found substantive issues anyway.

## Scope and evidence

Reviewed revisions:

| Component | Worktree | Revision |
|---|---|---|
| Current Ukrainian dissertation | `hierarchical-video-prediction-dissertation` | `1454a0125f77c1df6c1b2f6ea26acb7560c2889c` |
| Live A4 | `hierarchical-video-prediction-path-b` | `7a5979615f5354564a793083b673cf1ed9ec77be` |
| Integrated controller/profiler | `hierarchical-video-prediction-integration` | `344f7a79a4905bbd0a7cd7809ebd3cb30de50e98` |
| Integrated OpenSTL model | `OpenSTL-integrated` | `ca335ba206ef6f51515a4f8eefcad1eed8c36c10` |

I examined the handoff, manuscript structure and selected passages across its chapters, new method contracts, key equations and result tables, evidence/provenance records, A4 allocations and training/evaluation code, integrated model/test/profiler code, and current local logs. All 85 image insertions were inspected in contact sheets, with seven important assets also inspected individually. This is a substantive technical review, not a claim to have line-edited every paragraph, rederived all 119 equations, checked every bibliographic assertion, or reproduced historical training. The current typeset Ukrainian manuscript was not available in the inspected output directory.

Only this report was added to the dissertation worktree. No manuscript, model, campaign, manifest, checkpoint, authorization, or result artifact was changed. CPU probes hid CUDA and restricted math threads to one. I did not launch training, finalization, or confirmation evaluation.

## What is already good

- The introduction connects partial observability, history retention, identity/support errors, and separate resource costs. The four-chapter structure can work without another wholesale rewrite.
- The distinction between context reach and forecast fidelity is meaningful and experimentally testable. The legacy discrete frontier is presented more honestly than a fitted universal crossover would be.
- Negative E5a replication and mixed E4b/E6 findings are retained. Sample uncertainty is usually distinguished from training-seed uncertainty.
- The manuscript generally avoids multiplying token and activation savings. Chapter 4 now correctly limits the old integrated prototype's memory comparison.
- The new integrated model physically gathers fine tokens, uses physical coordinates rather than packed-slot identities, and places routing outside the reversible core. These are sound architectural decisions.
- The A4 implementation pins the prediction target and training duration, controls most architecture variables, logs useful evidence, and preserves optimizer and RNG state for resumption.

## Findings and change proposals

P0 means observed corruption or an immediate reason to stop. P1 means important before accepting the affected claim or proceeding with the affected experimental gate. P2 means a material methodological, reproducibility, or presentation correction. P3 means editorial polish. **No P0 was established.**

### R1 — P1: Chapter 3 contradicts Chapter 4 about the old reversible prototype

At [Chapter 3, line 887](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/04_CHAPTER_3.md:887), the text says only the activation-storage strategy changes. At line 893 it describes the first memory profile as a causal comparison of the same model. But [Chapter 4, line 380](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/05_CHAPTER_4.md:380) correctly says the external dynamics differ after the first transition.

The equations themselves expose the difference: with P0=0, Q1=F(Q0), but Q2=Q0+F(Q1), whereas ordinary repeated prediction produces F(Q1). Shared modules and similar diagnostic losses do not make these the same forward function.

**Change:** retain equations 3.49–3.53 as historical prototype equations, but replace the two incorrect causal statements locally. State that the 2.58 ratio is an engineering comparison of different recurrences. Reserve identical-forward activation-storage claims for the new internal reversible core. Search the abstract, introduction, captions, and provenance guide for the same implication. This requires no experiment restart.

### R2 — P1: The 30k calibration acceptance rule cannot establish convergence at its final endpoint

[The calibration analyzer, lines 94–106](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/scripts/v2/analyze_article4_schedule_calibration.py:94) takes the best value at or after the candidate step. At the final recorded step, 30k, that set contains only the current value, so the relative gap is necessarily zero. The remaining ranking sequence also contains only one direction and necessarily passes stability. Consequently, the final endpoint passes even for a still-improving trajectory.

[Chapter 4, line 110](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/05_CHAPTER_4.md:110) and the conclusions give this rule too much evidential weight. Agreement across six metrics does not remove the endpoint dependence. The separately reported 1.07% distance from a recent best is useful descriptive information, but does not prove a plateau or convergence beyond 30k.

**Change:** describe 30k as the common fixed optimization budget selected after observing that shorter runs could misrank configurations. Explicitly acknowledge the terminal acceptance property. Preserve the registered 30k campaign. If convergence becomes central to a claim, preregister a limited continuation of frozen decisive pairs beyond 30k and use an actual plateau/continuation criterion. Do not retrospectively change stopping rules across all models.

### R3 — P1: The integrated profiler does not yet deliver the promised memory experiment

[The profiler](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-integration/scripts/v2/profile_integrated_openstl_memory.py:88) records peak allocation minus the pre-forward baseline. It does not retain the absolute allocated/reserved peaks or the parameter/gradient/optimizer breakdown required to establish total training memory. There is no inference-only case or activation-checkpointing case in `CASES`.

Further issues:

- Warmup and measurement call `optimizer.step()`. Cases begin at the same checkpoint but then change their weights, including learning on random targets. This is not a fixed-checkpoint measurement throughout the comparison; routing can change during the measurement.
- CUDA parity is evaluated outside the requested autocast context. It checks output and input-gradient errors but neither parameter-gradient parity nor a fail-closed tolerance.
- Core-only profiling silently uses `inputs[:1]`, even if the report's top-level batch is larger.
- The optional fp16 path uses no GradScaler. It should not silently stand in for the actual mixed-precision training recipe.
- The analytic byte estimate assumes all floating payload uses the requested precision. Actual packed tokens and metadata need dtype inspection, especially where positional embeddings can promote intermediate tensors.

**Change before CUDA gates:** add absolute and incremental endpoints, inference profiles, a checkpointed comparator, explicit core batch/precision fields, actual tensor/storage accounting, and parameter-gradient parity at the actual training precision. Warm optimizer state without allowing uncontrolled checkpoint drift; separate fixed-weight profiling from any training-throughput experiment. Declare random-input profiles as capacity measurements and include representative real inputs for routing/workspace behavior. The existing script is a useful starting point, not a complete publication profiler.

### R4 — P1: Several integrated tests are degenerate at initialization

[The decoder output convolution is zero-initialized](C:/Users/leoni/Documents/projects/OpenSTL-integrated/openstl/models/hvp_integrated_model.py:623). The whole-model parity, learned-router-gradient, and decoder-permutation tests at [lines 164–215](C:/Users/leoni/Documents/projects/OpenSTL-integrated/tests/test_models/test_hvp_integrated.py:164) exercise this initial state.

I independently measured exact last-frame persistence at initialization and zero gradient magnitude in the encoder, core blocks, and learned router. Checking that a gradient is non-None is insufficient: it can be identically zero. A permutation test also passes if predictions ignore packed memory entirely. The whole-model test does not compare all parameter-gradient values.

Encouragingly, after setting a small nonzero output head in an isolated CPU probe, the new core still matched: output maximum error 0, input-gradient maximum error about 4.66e-10, and parameter-gradient maximum error about 1.51e-9. The prediction also responded to zeroing packed memory. This is positive evidence for the implementation, although only on a tiny fp32 case.

**Change:** strengthen the existing tests with a nonzero head or a few deterministic optimization steps. Require nonzero upstream gradients, compare all active parameter gradients, and include a negative control that corrupts memory/positions. Repeat on the trained checkpoint and actual CUDA precision before accepting the production gate. Zero-initializing a residual head is not itself a model bug.

### R5 — P1 operational: Current disk state would block the controller's restart preflight

C: had about 16.6 GiB free; [preflight requires at least 25 GiB](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/scripts/v2/run_article4_campaign.py:164). The running process can continue, but a later controller restart currently cannot pass that check.

The 38 completed run directories occupy about 7.09 GiB, averaging 0.187 GiB each. Another 48 scheduled runs at that observed average would add about 8.96 GiB, excluding finalization, additional caches, W&B growth, and temporary writes. D: had approximately 74.9 GiB free.

**Change:** arrange verified storage headroom before an interruption occurs. Prefer freeing unrelated recoverable data or an explicitly planned, verified storage migration over weakening the guard. Do not move live checkpoints or alter the active manifest casually. This is a recoverability concern, not evidence that the current run is corrupt; no immediate interruption is recommended.

### R6 — P2: The claimed deduplicated A4 matrix still contains a duplicate

I resolved all 80 wave-one specs into their ordered physical token plans. All token budgets were exact, and all specs used anchor 96, 100 temporal positions, and 30,000 updates. However, there were only **78 unique physical plans including seed**.

For each seed, `A4_ALC_B1792_L2_REACH_OPT_NEST` and `A4_STS_B1792_L2_K64_RECENT_NEST` have identical coarse ages, fine age/cell ordering, top ages, overlap, and fine representation: 96 coarse positions plus four complete recent fine positions. See [ALC construction](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/hvp_hca/allocation.py:198) and [STS construction](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/hvp_hca/allocation.py:346).

Policy/family labels make their config hashes differ even though the manipulated allocation is the same. The nominal 43-configuration total is therefore at most 42 distinct allocation designs under the stated wave-two plan; the scheduled run count remains 86 unless the controller contract is deliberately revised.

**Change:** add a semantic allocation fingerprint that excludes descriptive labels and an explicit alias/reuse ledger. Do not count these as independent allocation evidence. The conservative operational choice is to leave the live queue intact and report the duplicate as a repeated run, checking reproducibility when it finishes. Avoid patching the pinned controller mid-campaign merely to save two runs; its finalizer currently requires 86 entries.

### R7 — P2: A4 selection uses far less validation data than the split description suggests

[Validation uses four fixed batches of 16](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/scripts/v2/train_article4_campaign.py:191), sampled with replacement by [the batcher](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/hvp_hca/data.py:235). Replaying those sample-index draws yields **59 distinct validation clips**, not 300. The same small subset repeatedly influences calibration and allocation selection.

This is not train/test leakage, and a common deterministic subset is useful for learning curves. But it is a weak selection base for many policies and narrow differences. Fresh confirmation on 500 clips mitigates selection optimism; it does not make the selected configuration reliably best over all 300 development clips.

**Change:** disclose the effective selection sample in the methods and table captions. Preserve the already registered selection rule. Add full-300-clip evaluation of the frozen relevant checkpoints as a clearly labelled development robustness check, without using it to silently reselect the confirmation shortlist. Prefer all validation clips for selection in future campaigns.

### R8 — P2: Development and confirmation change the foreground-MSE estimand

The [training evaluator](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/scripts/v2/train_article4_campaign.py:213) computes a foreground-pixel-weighted ratio within each batch, then averages batch values. The [confirmation evaluator](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-path-b/scripts/v2/finalize_article4_campaign.py:505) first computes each clip/horizon's ratio, then averages records. These differ when foreground sizes differ. Selection uses horizon 4, while the headline confirmation summary averages all four horizons.

The implementation saves per-clip/horizon arrays, which makes a transparent final analysis possible. But the current finalizer produces descriptive summaries rather than the complete paired uncertainty and decision analysis needed for a confirmation claim.

**Change:** define weighting, zero-foreground handling, horizon, effect sign, paired clip resampling, multiplicity family, and any quality tolerance before opening confirmation. Report horizon-4 and all-horizon endpoints separately. Either use a matched estimand or explicitly label the difference; do not compare the two numbers as an unchanged endpoint. Preserve existing training metrics and add a versioned analyzer instead of overwriting history.

### R9 — P2: Static versus dynamic routing also changes token features

At [model lines 348–352](C:/Users/leoni/Documents/projects/OpenSTL-integrated/openstl/models/hvp_integrated_model.py:348), density/hybrid paths add a learned score embedding to selected features; static does not. Thus the planned static/dynamic contrast changes both membership and feature transformation. In a CPU probe with the entire fine quota selected, static and hybrid returned identical indices but their token values differed by up to 1.18.

The score embedding is a reasonable differentiable training path. It is not direct differentiation through the discrete selection and does not by itself prove that the learned score optimizes membership.

**Change:** either describe the result as a full routing-policy-package comparison, or add controlled membership × score-feature ablations. For a clean allocation comparison, keep feature treatment common while changing indices. A learned-score benefit needs an ablation against density/change-only routing and some evidence that selection tracks predictive utility, not merely that gradients exist.

### R10 — P2: The routing constraint and margin diagnostic need precision

[Equation 3.55](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/04_CHAPTER_3.md:937) promises at least m tokens per frame. [The implementation](C:/Users/leoni/Documents/projects/OpenSTL-integrated/openstl/models/hvp_integrated_model.py:283) silently reduces m when K/T is too small. Current default quotas are feasible, so this is a contract edge case, not a default-run failure.

Also, the reported margin is the global unconstrained Kth versus K+1th score. With mandatory per-frame picks, it need not be the last-selected versus first-unselected margin described in the manuscript. Static selection is another case where that interpretation does not apply.

**Change:** reject infeasible requested minima or define m_effective in the equation and diagnostics; document the standardization epsilon for constant scores. Name the existing margin as an unconstrained ranking statistic or compute a constraint-aware feasible-exchange margin. Check ties and near-ties as a robustness diagnostic.

### R11 — P2: Packed payload is not the complete live state

[PackedContext.payload_bytes](C:/Users/leoni/Documents/projects/OpenSTL-integrated/openstl/models/hvp_integrated_model.py:97) excludes its `coarse_grid`, while the decoder uses the last coarse map through a separate path. During training, the encoder and router also retain autograd intermediates from dense fine candidates. Real gather reduces the sequence entering attention; it does not establish that the full fine encoder graph disappears.

The manuscript largely warns about this distinction correctly. The measurement infrastructure must enforce it, rather than making a stronger practical claim from the 3.2 token ratio.

**Change:** distinguish logical packed-interface payload, live unique tensor storage including bypasses, saved activations, workspace, and total GPU peak. Avoid double-counting shared storage. Profile static/dynamic routes at equal K and the full encoder/core/decoder pipeline. A smaller core sequence can coexist with modest end-to-end savings.

### R12 — P1 for final release: The available PDF is not the current dissertation

`output/pdf/DISSERTATION_REVIEW_MANUSCRIPT.pdf` has 152 pages, identifies itself as an English review edition dated 3 August 2026, and contains no Figure 3.20. It is not a typeset rendering of the current Ukrainian source.

**Change:** build a current Ukrainian review artifact from an explicitly listed source set and commit, then inspect every page for equation rendering, overflow, caption attachment, page breaks, running headings, table continuation, reference wrapping, and final contents/pagination. Do not present the stale English PDF as evidence that current formatting passes. I cannot honestly sign off on current pagination from Markdown and individual assets.

### R13 — P2: Visual coverage is extensive, but visual language and emphasis are uneven

All 85 image insertions resolved and could be rendered. The collection mixes Ukrainian diagrams, English explanatory figures, and preserved handwritten-style publication figures. For example, Figure 4.3 combines an English historical optimization plot with a Ukrainian A4 calibration figure; Figure 4.7 and Figure 2.18 have continuous English annotations. Preservation is an explicit author constraint, so this is a release-design issue, not authorization to replace legacy assets.

The new [Figure 3.20](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/figures/restructured/ch3_fig3_20_packed_integrated_architecture.svg) has crowded labels; the future-query title extends beyond its box. Its router lacks an explicit incoming coarse-feature arrow despite using the coarse/fine residual, and the decoder's last-frame residual bypass is described in a label rather than drawn. Its small text will be difficult at ordinary portrait-page width.

Figure 2.18 uses truncated vertical axes for SSIM and PSNR bars, visually exaggerating differences. Figure 4.12 is readable but depends on its adjacent caption to disclose the different external dynamics. Figure 4.3's zoom is useful, but its endpoint annotation inherits R2's limitation.

**Change:** keep preserved figures, add concise Ukrainian keys where needed, and obtain a specific decision before any frozen-asset modification. Fix the new architecture diagram with wider boxes or two readable panels and explicit data paths. Use dot plots for future metric comparisons with restricted axes. Keep each historical limitation attached to the figure during layout, rather than separated across pages.

### R14 — P2: The main results need a compact empirical failure/success example

The main text has many helpful conceptual schematics and statistical plots, but the direct model-output montages are concentrated in appendices. The reader should see what a reach failure and an identity failure actually look like before accepting a long sequence of aggregate effects.

**Change:** add a compact main-text montage from frozen evaluation records: observed context with age markers, ground truth, matched flat/hierarchical outputs, identical-scale error maps, and relevant object/support scores. Include a hierarchy success, a flat success, and a failure of both. Declare case selection so these are explanatory examples rather than cherry-picked evidence. Preserve existing figures and appendix galleries.

### R15 — P2: Historical tables need stronger measurement provenance

[Table 2.6](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/03_CHAPTER_2.md:472) reports precise memory/time values, including an allocation beyond physical GPU memory. Its device and WDDM caveat help, but a frozen receipt with exact code/config/precision/measurement endpoint and repeat timings is still needed to make that table independently auditable. The adjacent missing T1–T4 record concerns a related diagnostic and should not be mistaken for a complete source of the table.

[Table 2.10](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/03_CHAPTER_2.md:747) reports V-BLT MSE 0.039 and PSNR 32.3 dB. Those are not related by -10 log10(MSE) under a single shared unit-range pooled-pixel convention, which would give about 14.09 dB. This is not proof the published values are wrong: differing scaling or averaging can explain it. The problem is that the convention is not sufficiently explicit for reproduction. The values are indeed inherited from the [published V-BLT article](https://visnyk.vntu.edu.ua/index.php/visnyk/article/download/3346/2996/3902); reproducing them faithfully does not resolve the ambiguity.

**Change:** retain published numbers and add metric range, reduction order, channel treatment, and unit definitions where recoverable. Mark unresolved conventions as reported-source limitations. Supply immutable historical receipts where they exist, or label the affected local profile as supporting evidence and avoid strong new conclusions from it. Do not fabricate missing raw values. Publication metadata for [TempVerseFormer](https://link.springer.com/article/10.1007/s00521-025-11707-3) was also spot-checked; this review does not certify every cited claim.

### R16 — P2: The experiments are broad inside TRB but not sufficient for broad long-horizon integration claims

A4's six allocation families are substantial and mostly complementary. Their limitation is inferential breadth: one procedural benchmark, one new main predictor recipe, two training seeds per cell, and one jointly predicted four-frame future. That is a strong controlled allocation study; it is not a comprehensive test of long self-fed stability or general natural-video prediction.

Moving MNIST 10→10 and TaxiBJ 4→4 add useful external interfaces, but TaxiBJ is a traffic-flow field and the proposed horizons are short. Four full-model runs alone cannot establish the causal contribution of hierarchy, reversibility, and routing. The plans recognize much of this; the evidence still needs to be produced.

**Change:** preserve the approved dataset scope. Complete A4 and its frozen confirmation, then add a bounded long-rollout evaluation of selected existing A4 checkpoints if a new long-horizon claim is intended. For integration, make component comparisons and matched local baselines mandatory for the claims they support. Two seeds should be shown separately, with uncertainty conditional on each trained model; a third seed on decisive contrasts is more valuable than another large grid of similar configurations. Natural-video claims must remain limited without a corresponding study.

### R17 — P2: The link from V-BLT to the final router remains conceptual

V-BLT supplies categorical-byte entropy maps; the integrated router uses residual feature energy, temporal feature change, and a learned scalar correction. These are different signals. The integrated architecture is a synthesis of principles, not a literal combination of the published V-BLT and TempVerseFormer implementations. The current method text is mostly honest about this, but the connection is easy to overread from the overall story.

**Change:** add a compact component-lineage table: inherited principle, actual implementation, changed assumption, and experiment verifying it. Do not describe the final router as entropy-driven without measuring that signal. A density-score versus measured prediction-error diagnostic would strengthen the rationale; it should include where the proxy fails.

### R18 — P3: The prose still contains editorial/process residue and repetitive qualifications

The scientific narrative is generally understandable, but it sometimes reads like a record of manuscript maintenance. Examples include Figure 1.1's source note about verification “for this edition,” [Table 2.8's caption](C:/Users/leoni/Documents/projects/hierarchical-video-prediction-dissertation/docs/dissertation/condensed_translated_humanized_ukrainian/03_CHAPTER_2.md:581) explaining why the table was retained rather than replaced, and repeated statements about what was preserved or moved. Several subsections repeat the same token-versus-GPU and feasibility-versus-superiority warning.

**Change:** retain necessary scientific scope statements, but remove editorial-history sentences from the submitted prose. Establish one stable vocabulary for physical support, reach, fidelity, packed state, training seed, and legacy/final integration. Use short local reminders only when a result could otherwise be misunderstood. Add a compact early map linking each research question to its method, evidence, and limitation. Preserve the agreed structure and existing mathematical/visual material.

## Live campaign assessment

At the snapshot, A4 had **38 complete, one running, zero failed, 41 pending in wave one**, followed by six dependent wave-two runs. The active Balanced seed-0 run had reached step 2,200 in the log; the rolling checkpoint had been written at step 2,000. The GPU was active at 100% utilization with roughly 6.3 GiB reported device memory in use. Status step zero observed earlier was simply a coarse checkpoint-status snapshot; subsequent logs demonstrated progress.

Independent aggregate checks across all 38 completed runs found:

- 38/38 clean launches at the pinned source commit;
- 38/38 with exactly the expected 15 ordered validation points, 2k through 30k;
- finite numeric validation records throughout;
- 76/76 matching training/validation file hashes against terminal status;
- approximately 85 completed training hours.

Mean completed run duration increases plausibly with token budget: 0.50 h at 320, 0.93 h at 768, 1.43 h at 1,152, 2.36 h at 1,792, 3.47 h at 2,432, and 4.69 h at 3,072. Recent losses, gradient norms, and validation trajectories are numerically plausible; one completed run worsened by more than 2% from its 28k to 30k primary validation value, which alone is not a fault. These are health checks, not final family-level scientific conclusions. I did not load and hash every historical checkpoint or establish bitwise restart equivalence across all runs.

The immediate priority is disk recoverability, not interrupting training. No new ETA is asserted from a single active run. No partial A4 family result should be inserted into the manuscript.

The integrated controller reports all four dataset/seed runs pending with no resumable checkpoint. Its model has useful CPU correctness evidence, but predictive competitiveness, useful routing, and CUDA-memory savings remain unmeasured. The original dirty OpenSTL and visionBLT development checkouts should not be mistaken for immutable historical reproduction packages; they were left untouched.

## Experiments that would most improve the dissertation

| Priority | Work | What it resolves | Relaunch current A4? |
|---|---|---|---|
| Required | Finish registered A4, dependency resolution, locked shortlist, fresh 500-clip evaluation, and paired final analysis | Whether the allocation findings replicate outside selection data | No |
| Required before integrated confirmation | Nondegenerate CPU/CUDA parity, tiny overfit on both datasets, 200-update timing/capacity screens, repaired profiler | Whether the implementation and intended memory measurements are valid | No |
| Required for integrated efficiency claims | Matched local baseline protocol, ordinary/checkpointed/reversible execution, flat/hierarchical training comparison, static/dynamic routing controls | Which component explains a quality/resource change | No; separate bounded integrated work |
| High value | Frozen-shortlist self-fed evaluation at several longer horizons, with registered buffer/action/metric rules | Whether increased context reach survives recursive prediction | No; existing A4 checkpoints can be evaluated |
| High value | Full development-validation robustness and an additional seed on a few decisive contrasts | Sensitivity to the small selection subset and initialization | No; label added work separately |
| High value | Context ablation/shuffling or age-band removal on frozen checkpoints | Whether old information is actually used rather than merely present | No; interpret as a diagnostic intervention |
| Conditional | Smaller-K routing frontier against a predeclared quality tolerance | Whether learned allocation buys actual retained-state savings | Separate integrated ablation |
| Optional for broader claims | A natural-video benchmark | External validity beyond synthetic scenes and flow fields | Outside the approved core scope |

For the integrated block, freeze the local baseline, validation criterion, acceptable quality tolerance, data processing, update count, batch/precision, and memory endpoint before the four long runs. Parameter-count differences must remain visible. A much smaller model using less total memory is not by itself evidence that reversible execution caused the saving.

## Readiness answers

| Question | Assessment |
|---|---|
| Is the topic properly described? | Largely yes. The problem is coherent; the general title needs a clear operational scope centered on history-dependent prediction under resource limits. |
| Is the current evidence enough? | Substantial for a bounded systems/methodology dissertation; incomplete for the expanded integrated claims. Submission adequacy ultimately also depends on the institution's standards. |
| Are experiments comprehensive? | Broad within TRB allocation, incomplete across long-rollout behavior, integration ablations, and external visual domains. More runs alone are not the missing ingredient. |
| Are formulas and explanations good? | Mostly coherent in the reviewed portions; R1, R2, and R10 require correction. Inventory checks are not proof that all equations match all implementations. |
| Are images and tables good? | Useful and extensive, but language, measurement definitions, visual emphasis, and direct empirical examples need work. |
| Is formatting ready? | Not established. A current typeset Ukrainian artifact must be rendered and inspected. |
| Is the integrated implementation okay? | Promising and structurally sensible. A nondegenerate tiny CPU parity probe passed; tests and profiling still need strengthening before its major claims are evaluated. |
| Should A4 be stopped or restarted? | No observed scientific or numerical corruption warrants that. Continue, address storage recoverability, and retain the duplicate/estimand disclosures. |

The recommended disposition is **continue the current A4 campaign, make targeted manuscript corrections, repair integrated verification/profiling before CUDA gates, then complete the evidence and final Ukrainian layout**. The eleven administrative fields are not the only remaining work.
