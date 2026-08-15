# Compact dissertation content-parity report

**Date:** 12 August 2026  
**Edition reviewed:** `docs/dissertation/condensed_english`  
**Assembled manuscript:** `docs/dissertation/final/DISSERTATION_CONDENSED_ENGLISH.md`

## Verdict

The compact English dissertation is scientifically comprehensive and content-final for author and
supervisor review. It is no longer a summary edition that omits evidence-bearing material. The
shortening was achieved by removing repeated explanation, campaign chronology, repeated caveats and
section-level restatement while retaining the scientific structure, mathematical definitions,
experimental contracts, results and limitations.

No additional experiment is required for the claims made in this edition. Final institutional
submission still depends on author-owned administrative data, translation and typesetting.

## Final inventory

| Item | Compact final edition | Long reference edition | Parity result |
|---|---:|---:|---|
| Scientific words, audit definition | 33,958 | approximately 61,000 | about 44% shorter |
| Numbered main figures | 86 | 82 | all retained; four additional main-text placements |
| Numbered tables | 30 | 30 | complete |
| Numbered displayed equations | 102 | 102 | complete and text-identical after normalization |
| Bibliography entries | 174 | 174 | all cited; no undefined citations |
| Scientific headings and subheadings | complete | complete | no heading removed |
| Broken image links | 0 | 0 | pass |
| Unresolved editorial markers | 0 | 0 | pass |

The apparent increase from 82 to 86 main figures has two causes. The published reverse-diffusion
diagram was moved from an appendix into the relevant Chapter 1 explanation, and three additional
figures from the 2024 consistency review were restored. All image paths used by the long edition are
also present in the compact source set.

## Material restored during the final parity pass

### Chapter 1

- Restored the temporal-consistency failure montage, cross-level consistency diagram, consistency-
  control interface diagram and support-versus-identity bridge.
- Restored the omitted comparison table that separates consistency criteria, observable failures and
  suitable measurements.
- Brought all nine figures from the 2024 consistency review into the dissertation where they have a
  clear pedagogical role. Previously retained figures remain unchanged. Original review Figures 4,
  6 and 8 were extracted without redesign or recompression; the author has confirmed ownership of the
  artwork and locally generated examples.
- Added concise context around forward and reverse diffusion, the image-generation pipeline,
  image-to-image conditioning, ControlNet, inversion and StyleAligned. These illustrations support
  the transition from static consistency controls to the later video criteria; they are not presented
  as experimental evidence for the hierarchical predictor.

### Chapter 2

- Restored the complete TempVerseFormer visual sequence, including constant, accelerated,
  decelerated, oscillating and interrupted motion, OpenSTL examples, memory scaling and training-time
  scaling.
- Restored the tables that distinguish the published architecture, the inspected implementation and
  the exact reversible-state boundary.
- Restored the V-BLT architecture, causal mask, entropy pipeline, qualitative entropy maps,
  point-estimate comparison and implemented-versus-proposed boundary.
- Added the missing high-horizon qualification: the published plot extends to 1,024 prediction steps,
  but exact raw coordinates are unavailable, so numerical comparisons remain anchored to the
  one-, four- and sixteen-step table.
- Preserved the distinction between the measured fixed-patch V-BLT and entropy-guided adaptive
  regioning as a proposed extension. No executed adaptive allocator is implied.

### Chapter 3

- Restored the gradient-path and persistent-assignment figures and the missing protocol table.
- Restored the complete six-step Algorithm 3.1, including its inputs, outputs, package construction,
  denoising, decoding and ring-buffer update.
- Retained all definitions for package geometry, context/target/total tokens, reach, objectives,
  paired clip-cluster estimands and simultaneous inference.

### Chapter 4

- Restored the generation-head comparison figure and complete execution/configuration table.
- Retained the complete tested frontier: hierarchy confirmed at 896 and 1,152 context tokens,
  1,792 unresolved, and flat context confirmed at 2,432 and 3,072.
- Retained equal-reach arithmetic, pair-specific reach effects, seed/sampler/codec robustness,
  failed self-fed-boundary replication, explicit-access reference and the E6 support-versus-identity
  boundary.
- Retained the resource-axis synthesis and final binding-constraint decision rule.

### Front matter and conclusions

- Added a concise V-BLT result and limitation to the English abstract.
- Restored entropy-based spatial-complexity estimation to General Conclusion 3 while explicitly
  stating that dynamic region allocation was not implemented or evaluated.
- Removed repetitive `The`/`This`/`These` opening runs from the General Conclusions.

## Coverage against the four dissertation publications

| Publication strand | Material checked | Coverage in the compact dissertation | Boundary retained |
|---|---|---|---|
| 2024 consistency review | taxonomy, diffusion stages, image-to-image, DreamBooth/LoRA, ControlNet, inversion, StyleAligned | Chapter 1 contains the conceptual progression and all nine published illustrations | static-image review is extended to video criteria, not treated as temporal evidence |
| TempVerseFormer | reversible block, temporal chain, controlled motion, OpenSTL, loss, memory and runtime | Chapter 2 includes the architecture, implementation boundary, equations, tables and qualitative/quantitative evidence | exact inversion belongs to the repeated doubled transformer state; the VAE and projection interfaces are outside it; replay costs computation |
| V-BLT/RoPE-ND | raster-byte representation, local-global-local processing, RoPE-ND, categorical head, fixed patches, entropy maps and reported comparisons | Chapter 2 includes the complete method contract, principal equations, architecture and result figures/tables | entropy is a diagnostic under fixed patches; online adaptive regioning remains future work |
| Hierarchical context article | LoopMom, package intervention, reach populations, untouched frontier, robustness, negative replication and E6 | Chapters 3 and 4 retain the complete method and evidence chain | hierarchy buys reach only in the reach-limited region; flat context recovers fidelity after reach saturation or when identity detail dominates |

The final hierarchy article was also checked as the binding source for the Chapter 3/4 claims. The
compact dissertation preserves its decisive numbers and does not promote the unresolved 1,792-token
comparison into a hierarchy win.

## Compactness and writing-pattern check

The final prose contains 33,958 scientific words. Paragraph medians range from 58 to 85 words across
the front matter, introduction, chapters and conclusions. No ordinary prose paragraph exceeds 170
words; the only longer detected block is a deliberately formatted numbered criteria list. No run of
three prose paragraphs begins with the same opening word.

The revision varies paragraph and sentence length, removes filler transitions, avoids repeated
mini-conclusions and keeps evaluative statements where they clarify evidence boundaries. It does not
use synonym substitution as a substitute for restructuring. Formula density remains higher than in
the colleague's reference dissertation, as requested, but equations now appear only where they define
the method, resource accounting, objective or estimand.

## Page-budget implication

The earlier compact Word trial used about 31,300 scientific words, 69 figures and 24 tables and
occupied roughly 174 English pages. The final edition adds about 2,700 scientific words, 17 figures
and six tables. With grouped small illustrations, compact captions and landscape treatment for wide
tables, the likely English typeset length is around 190–205 pages. Translation expands prose but not
the full visual area, so the 220-page Ukrainian ceiling remains plausible but must be checked in the
actual institutional template. This is now a layout-management question, not a reason to delete
scientific content.

## Remaining author-owned inputs

These are not scientific content gaps:

- Ukrainian abstract and full translation;
- UDC and final title-page wording;
- official research-project titles, numbers and the author's role;
- final conference/approbation records;
- implementation-act and appendix identifiers;
- final authorship, venue and publication status for Article A4;
- final page, figure and table counts after institutional typesetting.

## Reproducibility pointers

The dissertation now points readers to the public repositories for the historical implementations:

- `https://github.com/leo27heady/simple-shape-dataset-toolbox`
- `https://github.com/leo27heady/TempVerseFormer`
- `https://github.com/leo27heady/visionBLT`

These repositories improve reproducibility, but they do not manufacture missing archived seed panels
or exact numeric coordinates from older published plots. The dissertation states that distinction
directly and limits historical claims to the available evidence.

## Final acceptance checks

- Compact dissertation audit: **PASS**.
- Long reference dissertation audit: **PASS** after recording both long and compact figure identifiers
  in the shared provenance ledger.
- Citation closure: **174 cited / 174 listed**.
- Figure, table and equation parity: **PASS**.
- Article-content coverage: **PASS** for all four strands and their stated limitations.
- New experiments needed: **none for the current claims**.

