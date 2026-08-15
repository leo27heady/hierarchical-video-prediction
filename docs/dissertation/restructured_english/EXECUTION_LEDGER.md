# Restructured English Dissertation — Execution Ledger

**Binding blueprint:** `docs/dissertation/DISSERTATION_RESTRUCTURE_AND_WRITING_BLUEPRINT_2026-08-10.md`  
**Final assembled source:** `docs/dissertation/final/DISSERTATION_RESTRUCTURED_ENGLISH.md`  
**Working section directory:** `docs/dissertation/restructured_english/`  
**Rule:** the pre-existing review dissertation and published-figure pixels remain untouched.

## Scientific spine

The dissertation answers one systems question: how should a long-horizon video predictor allocate spatial detail, retained history, and training memory so that task-relevant consistency can be maintained under bounded resources?

The argument proceeds through four questions:

1. What must remain consistent in generated visual sequences?
2. What prevents temporal training through long processes?
3. How can visual information be represented without equal capacity everywhere?
4. When does hierarchical retained context help under a fixed budget?

The Chapter 3 hierarchical-context predictor and Chapter 4 operating-region study are the empirical core. TempVerseFormer and V-BLT are separately validated sibling methods at different system interfaces. They are not components of the final predictor and are not presented as one jointly evaluated architecture.

## Non-negotiable scientific boundaries

- V-BLT used fixed `4 x 4 x 4` spatio-temporal patches over uncompressed raster pixel-channel bytes. Entropy maps were measured; dynamic entropy-driven regioning remains future work.
- Predictive entropy is a dispersion/complexity diagnostic, not calibrated uncertainty or a validated allocation controller.
- Reversibility applies only to the verified persistent temporal state and replay procedure. The VAE, projections, fusion, loss interface, and whole end-to-end pipeline are not silently included in an exact-invertibility claim.
- TempVerseFormer shows an activation-memory/computation exchange and mixed predictive quality, not a universal quality gain or end-to-end integration with the final predictor.
- The procedural 2-D rotation sources are described as a shared family unless every generator and split detail is verified as identical.
- LoopMom is a synthetic mechanism-identification instrument. Persistent long Moving-MNIST is a separate synthetic external boundary. Neither is natural-video evidence.
- Hierarchy is favored at tested 896- and 1,152-token operating points; 1,792 is unresolved; flat context is favored at 2,432 and 3,072. No continuous crossing is fitted.
- At nominal 48-frame reach, 48 coarse plus 6 fine uses 1,152 tokens (2.67 times fewer than 3,072 dense fine tokens); 48 coarse plus 2 fine uses 896 tokens (3.43 times fewer).
- E6 supports a support/continuity versus identity boundary: hierarchy improves spatial support and continuity, while flat fine context better preserves identity. Absolute task performance remains difficult.
- No new GPU experiment is required for the approved scope.

## Section ownership and status

| File | Scope | Owner | Status |
|---|---|---|---|
| `00_FRONT_MATTER.md` | Title placeholders, English annotation, publication list, abbreviations | root | drafted; audit pending |
| `01_INTRODUCTION.md` | Plain-text Introduction with six tasks and calibrated novelty | root | drafted; audit pending |
| `02_CHAPTER_1.md` | System formulation, consistency, bottlenecks, literature, requirements | root | drafted at target length; visual audit pending |
| `03_CHAPTER_2.md` | Reversible training and fixed-patch V-BLT/entropy | draft_chapter2 | drafting |
| `04_CHAPTER_3.md` | Central hierarchical-context model and evaluation method | draft_chapter3 | drafting |
| `05_CHAPTER_4.md` | Experiments, frontier, robustness, boundaries, decision rule | draft_chapter4 | drafted at target length; four new figures pending |
| `06_CONCLUSIONS.md` | Six task-aligned conclusions | root | drafted; audit pending |
| `07_REFERENCES.md` | Claim-bound bibliography | root | pending |
| `08_APPENDICES.md` | Registries, supplementary evidence, provenance and author inputs | root | pending |
| `DISSERTATION_RESTRUCTURED_ENGLISH.md` | Complete assembled review source | root | pending |

## Artifact gates

- Annotation and Introduction: no figures, tables, or displayed equations.
- Main scientific text: 73 core purposeful figures; up to 7 conditional figures only if useful and permissible.
- New schematics: white background, consistent font/palette, legible arrows, vector PDF plus high-resolution PNG where practical.
- Published figures: unchanged pixels, original figure/page/source attribution, permissions status, extraction receipt, and hash.
- Derived charts: frozen data or exact published point estimates only; never invented uncertainty.
- Chapter 4 must retain the executed configuration matrix, five-seed head-by-package factorial, context-form evidence map, unresolved 1,792 point, and complete boundary evidence.

## Implementation-contract audit (2026-08-10)

### TempVerseFormer

The available implementation at `C:/Users/leoni/Documents/projects/TempVerseFormer` resolves the reversible boundary. `InputProjectionModule` maps 256-dimensional VAE latents to 768-dimensional tokens and duplicates the projected tensor into a doubled 1,536-dimensional state. `OutputProjectionModule` normalizes and fuses that doubled state back to 256 dimensions. Both classes are explicitly marked `NotReversibleModule`. Under the default `REVERSE_CALCULATION` mode, ordinary autograd handles these two interfaces; `EfficientRevBackProp` is applied only to the repeated stack of additive-coupling `ReversibleBlock` modules. Consequently:

- the exactly reconstructable state is the doubled transformer state inside the repeated additive-coupling stack;
- the split is a channel split of that doubled state, while the initial operation is duplication after a learned projection;
- VAE encoding/decoding, learned input projection, output fusion, and the final loss are outside the exact reversible state;
- the custom backward stores the terminal reversible state, reconstructs earlier states, rebuilds local graphs, and accumulates gradients;
- the dissertation may describe activation replay through the repeated transformer chain, but not an exactly invertible end-to-end VAE-to-VAE pipeline.

### V-BLT

The published 2025 paper explicitly identifies its proof-of-concept package as fixed `4 x 4 x 4` spatio-temporal byte cubes and places dynamic entropy-based packaging in future work. The available `visionBLT`/`minVBLT` source confirms a local/global/local transformer, 256-way categorical raster-byte prediction, frame-causal masks, multidimensional rotary coordinates, and entropy computed from categorical logits. It also exposes a provenance discrepancy: the tracked default configuration uses spatial tiles `4 x 4` and `model.patch_size: 5.0`; no tracked experiment configuration establishing temporal length 4 has yet been found. Therefore:

- `4 x 4 x 4` is attributed to the published experimental specification, not called code-reproduced verification;
- the default-config mismatch is disclosed in the reproducibility/limitations material;
- published table values remain published point estimates, not independently regenerated measurements;
- entropy is defined per predicted pixel-channel byte; no calibration or quantitative entropy-to-dynamics association is inferred from the qualitative maps;
- the current source contains infrastructure suggestive of flexible patch identifiers, but it does not convert the published fixed-patch experiment into evidence for a validated adaptive controller.

## Required verification sequence

1. Resolve the reversible-state contract and V-BLT factual contract before accepting Chapter 2 prose.
2. Verify each section against the blueprint and published-source facts.
3. Bind technical claims to primary references; aim for coverage, not a reference quota.
4. Create every missing core visual and verify figure links, captions, provenance, and readability.
5. Assemble the manuscript without modifying the previous review source.
6. Run structural, numerical, citation, prose-repetition, and Markdown checks.
7. Commission an independent subagent audit against the complete blueprint and all three published papers.
8. Correct all P0/P1 and justified P2 findings, then record remaining author-owned or institutional inputs.

## Author-owned inputs that may remain

- final UDC and approved title wording;
- two state-budget research-project numbers, titles, dates, and applicant role;
- venue and final status for publication A4;
- complete conference/appraisal records;
- final aggregate publication statement;
- appendix letters and signed implementation-act scans;
- rights/permissions decisions for reused source figures;
- final institutional pagination and template-specific metadata;
- Ukrainian translation of the approved English annotation and manuscript.

## Completion definition

“Finalized” here means scientifically coherent, internally consistent, fully drafted in English, claim-bounded, reference-bound, visually specified or rendered, and independently audited. It does not mean institutionally deposit-ready while the author-owned fields, permissions, translation, signatures, or final pagination remain open.
