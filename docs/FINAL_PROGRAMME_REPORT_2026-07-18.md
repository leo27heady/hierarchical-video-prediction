# Final scientific programme report

**Date:** 2026-07-18

**Experimental state:** terminal through E6

**Live GPU work:** none

**Authoritative language:** English
**Canonical status ledger:** `docs/EXPERIMENT_EXECUTION_ORDER_NEW_VISION_2026-07-12.md`

## 1. Final scientific identity

The fourth article is a controlled context-packaging study. It does not claim that a hierarchical
architecture always wins.

The manipulated design space is how a bounded context-token budget is distributed across temporal
rate, spatial resolution and package depth. The confirmed explanatory axes are:

- **reach:** whether the supplied package contains the memory-relevant past;
- **fidelity:** whether task-relevant detail survives its representation;
- **self-fed reliability:** a measured robustness question whose proposed late boundary did not
  replicate across training seeds.

The compact result is:

> **Hierarchy buys reach; flat context buys fidelity.**

The dissertation-level systems synthesis adds the orthogonal activation-memory result:

> **Compress what must be remembered; reconstruct what need not be stored.**

Hierarchical/multirate packaging reduces the context-state cost of distant history.
TempVerseFormer/reversible temporal optimization reduces the activation tape required for long
temporal computation. They are complementary resource axes, not evidence that one causes the
other or that their savings automatically multiply.

## 2. Terminal experimental evidence

### E0 — matched-example optimizer-regime control

The B8×60k control is a changed-batch/update/noise/grouping/EMA regime, not twice the data. Flat
beats hierarchy by 0.00773, 95% CI [0.00304, 0.01233] in flat's direction. This qualifies the
upper frontier without being interpreted as asymptotic convergence.

### E2 — untouched LoopMom confirmation

The registered pair-derived reach bands confirm the mechanism: the largest effects occur on
records inside the 48-frame package reach but outside the shorter flat comparator's reach. The
simultaneous frontier confirms hierarchy at the tested 896- and 1,152-token points, leaves 1,792
unresolved, and confirms flat full-resolution context at 2,432 and 3,072. Thus 1,152 is the
largest tested family-wise-confirmed hierarchy win and 2,432 the smallest tested confirmed flat
win. No monotonic curve, unique crossing or interpolation is claimed.

The memory arithmetic is material: in the frozen latent interface, a fine frame costs 64 context
tokens and a coarse frame 16. Forty-eight fine frames cost 3,072 tokens; 48 coarse plus two or six
recent fine frames cost 896 or 1,152. The hierarchy therefore obtains the same nominal 48-frame
reach with approximately 2.7–3.4× fewer context tokens, and the untouched effects establish that
the added reach matters in the reach-limited regime.

### E4a/E4b — five-seed head × package factorial

Temporally sparse full-resolution history beats spatially coarse packaging under diffusion on all
five training seeds. Regression has two temporal and three spatial material winners, and the
cross-head interaction is training-seed-heterogeneous. This is not equivalence and not a stable
directional head law.

### E5a — self-fed reliability replication

The early hierarchy benefit is positive at all three seeds. The late inversion exists only at
seed 0 and reverses to a hierarchy advantage at seeds 1–2. The registered verdict is
`NOT_REPLICATED`; trust is removed from the title-level claim and no adaptive E5b diagnostic was
run.

### E7c — sampler/NFE robustness

All 20 central diffusion effects retain their point-estimate direction across NFE 8/16/32 and
sampler seeds 0/1/2. Eighteen simultaneous lower bounds are positive. The 2,432-token inversion
is family-wise unresolved at alternate sampler seeds 1/2. Verdict: `DIRECTIONALLY_ROBUST`.

### E7a — independent-codec robustness

All four registered point-estimate directions survive a separately trained codec. The lower reach,
interior and package-shape effects are simultaneous-CI clean; the upper inversion remains positive
but its interval crosses zero. Verdict: `CODEC_DIRECTIONALLY_ROBUST`, conditional on predictor
seed 0 and a whole-codec intervention.

### E8a — explicit-access reference

The privileged exact-pose raw K48 mosaic reaches foreground MSE 0.0494. All four registered
neural-minus-K48 effects have positive simultaneous lower bounds. This demonstrates the remaining
value of lossless storage and reliable addressing after geometric reach is available. It is not
token matched, learned retrieval, a deployable neural baseline or an efficiency ranking.

### E6 — valid persistent-object long Moving-MNIST boundary

E6 completed six predictor trainings and six protected evaluations using persistent generator
identity, official glyph separation, Hungarian assignment, explicit misses/extras, coverage and
one simultaneous 15-effect family.

The registered verdict is `MIXED_BOUNDARY`:

- union-mask error favors hierarchy at all three seeds;
- trajectory and coverage favor hierarchy cleanly at seeds 0 and 2 and directionally at seed 1;
- hierarchy misses fewer objects and has lower full-frame/background error at every seed as
  secondary evidence;
- digit identity favors flat context cleanly at all three seeds;
- foreground-only MSE is unresolved and changes sign.

This is the external form of the reach–fidelity tension. Coarse long memory better preserves where
an object is, whether it survives and how it moves; flat full-resolution context better preserves
what digit it is. It is neither a failed experiment nor a universal hierarchy victory.

## 3. Robustness, provenance and cost

- E6 independent completion status: `PHASEC_V2_COMPLETION_PASS`.
- Six training cells and six protected evaluations are hash-valid.
- Each E6 evaluation contains 30,000 paired records.
- Protected access contains exactly seven permitted entries.
- E6 W&B audit contains 15 finished runs with the required immutable artifacts.
- Independent CPU recomputation matches all 15 effects exactly.
- E6 mechanical project ownership: 20.5127031 GPU-hours, including three charged failed
  preparation cells.
- Measured project consumption through E6: approximately 124.965 GPU-hours.
- Full CPU repository suite: 83 tests passed.
- Packaging/RFT consistency, text consistency and scientific-config/authorization validation pass.

## 4. Content completed

The result is integrated directly—not only through planning documents—into:

- the binding fourth-article manuscript;
- the detailed article Introduction, Results, Generalization and Conclusion sources;
- the dissertation front matter, Introduction, Chapter 4 and General Conclusions;
- the claim ledger and resource-efficiency synthesis;
- the generated evidence tables and machine-readable claim/artifact index;
- Figure 22 and the figure registry;
- the E6 result record, protocol, execution ledger and experiment authorization;
- the reproducibility appendix and repository README.

The binding article is 6,127 words before venue formatting. The assembled authoritative English
dissertation is 54,178 words, with 57 figure captions, 21 table captions and 80 bound external
references.

## 5. What is intentionally not claimed

- universal hierarchical superiority;
- a continuous or unique token-budget crossing;
- a pure causal context-stride effect for B1;
- regression equivalence or a stable directional cross-head interaction;
- a replicated late trust boundary;
- natural-video transfer;
- an efficiency ranking against the privileged E8a mosaic;
- equivalence or one average winner on E6;
- validity of the legacy Phase C evaluator.

These exclusions strengthen the work by converting architecture promotion into a falsifiable
systems result with measured operating regions.

## 6. Release completion and remaining author-owned work

No further GPU experiment is authorized or required for the present article/dissertation scope.

The research and repository release work is complete:

- the modular English dissertation sources assemble into one validated manuscript;
- the binding fourth article has a release Markdown source;
- both manuscripts have visually audited DOCX review editions and PDF review renders;
- 12 remaining `[AUTHOR: ...]` fields are isolated in
  `docs/dissertation/AUTHOR_INPUT_CHECKLIST.md`;
- a documented public `hvp` API/CLI wraps the frozen research implementation;
- scientific-config, claim/text consistency, assembly, compilation, CPU test and wheel-build
  gates pass.

The remaining items require author or institutional facts rather than scientific work:

1. Select the target venue and apply its length, style and bibliography format.
2. Bind [A1]–[A4] to the final numbered publication list and resolve institutional front-matter
   fields marked `[AUTHOR: …]`.
3. Apply the university/venue template to the authoritative Markdown content and update final
   pagination; the supplied DOCX/PDF files are review editions rather than a claim of institutional
   template compliance.
4. Publish or archive the large datasets/checkpoints/NPZs named by the content-addressed manifests
   if external reproduction access is required.
5. Perform a Ukrainian localization only if explicitly required later; it is not part of the
   current scientific workflow.

E9 Counterfactual Dynamic LoopWorld is terminally
`SKIPPED_POST_FREEZE_NO_INCREMENTAL_UTILITY` for this programme. It remains a high-value future
study for information necessity and hidden-state structure, but it requires a new generator,
oracle, factorial, scope and explicit authorization. E6 has already closed the current
reviewer-facing external boundary, so E9 must not delay dissertation closure. E3a remains
conditional only if the article later restores a pure causal stride claim.
