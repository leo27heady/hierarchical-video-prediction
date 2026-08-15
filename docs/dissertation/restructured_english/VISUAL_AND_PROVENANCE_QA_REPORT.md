# Visual and provenance QA report

> **Historical snapshot.** The author subsequently confirmed ownership of the reused artwork and local generation of the displayed samples. Current ledgers record the applicable CC BY 4.0 terms and author declaration, and the earlier open-permission warning below is superseded. New review-article figures were added after this QA pass without commissioning another audit.

**Review date:** 11 August 2026  
**Binding specification:** `docs/dissertation/DISSERTATION_RESTRUCTURE_AND_WRITING_BLUEPRINT_2026-08-10.md`  
**Scope:** all numbered main-text figures in Chapters 1--4, their live Markdown links, source captions, three provenance ledgers, and deterministic figure builders.

## Decision

The visual set is ready for scientific and editorial review. All 79 numbered figures were inspected at original raster resolution; the 79 captions resolve to 83 linked raster files because Figures 2.17 and 2.25 contain separately preserved published panels. No linked image is missing, clipped at its file boundary, or absent from provenance. All author-generated linked rasters have white corners, and every newly generated raster has a same-stem vector PDF where the generator promises one.

The figure package is not yet institutionally deposit-ready for one administrative reason: reuse permission remains open for 26 published assets, and upstream data terms remain open for two author-generated MNIST-derived qualitative composites. This is not a scientific defect and does not block manuscript review. It must be resolved before deposit, or the affected assets must be removed.

No published TempVerseFormer, V-BLT, or 2024-review pixels were redrawn, cropped, recoloured, relabelled, or otherwise modified. The automated audit compares each reused file with the supplied author-source file or exact PDF image object and passes byte parity.

## Inventory and outcome

| Chapter | Numbered figures | Linked raster files | Original-resolution decision |
|---|---:|---:|---|
| 1 | 18 | 18 | Pass after replacing three semantically mismatched schematics and repairing Figure 1.3 |
| 2 | 27 | 31 | Pass; published details preserved; several dense panels require full-width or landscape placement |
| 3 | 17 | 17 | Pass; white backgrounds, legible labels, and clear causal arrows |
| 4 | 17 | 17 | Pass after repairing the EBT recursion labels and clarifying the Figure 4.17 `no` branch |
| **Total** | **79** | **83** | **Pass for content review** |

The 83 linked files comprise 26 unchanged published assets and 57 author-generated assets: 25 conceptual, 20 artifact-derived, 11 implementation-informed, and one evidence-synthesis matrix. The ledgers contain 119 records in total because 36 new author-generated raster figures also have registered vector-PDF counterparts.

## Changes made during this pass

| Figure | Finding | Final action |
|---|---|---|
| 1.1 | The earlier image did not explicitly show the latent-world/viewport boundary described by its caption. | Replaced with a dissertation-native system-boundary schematic separating world transition, observation, context construction, prediction, control, and generated feedback. |
| 1.3 | Row labels collided with the first state boxes; right-hand annotations collided with the last states; a dashed curve ended in empty space. | Re-laid out both sequences, typeset generated observations consistently, moved the row annotations outside the boxes, and removed the ambiguous unfinished curve. |
| 1.8 | The linked asset showed context reach rather than the promised smoothness/trajectory/re-entry distinction. | Replaced with a three-panel conceptual diagnostic matching the caption. |
| 1.9 | The linked asset showed only macro/meso/micro levels, not compatible versus contradictory predictions. | Replaced with an explicit cross-level agreement/contradiction comparison. |
| 2.4 | `recurrent state / metadata` exceeded the storage box at source resolution. | Broke the label across two lines and rebalanced the box. |
| 4.4 | Candidate labels were crowded against their boxes and the update equation was unnecessarily small. | Split labels, typeset the update mathematically, and clarified the return arrow. |
| 4.17 | A grey `if no` note sat under the preceding red branch, making its parent question ambiguous. | Added a distinct grey arrow and terminal box from the fine-detail question's `no` branch. |
| Chapter 4 captions | Most captions named results but did not explicitly classify the image as conceptual, implementation-informed, or artifact-derived. | Added source-and-status notes to all 17 caption blocks. |

All replacements are new author-generated assets. They do not alter or overwrite any published figure.

## Published-figure preservation and the 2024 review

| Published source | Assets used | Preservation check | Permission state |
|---|---:|---|---|
| Kulyk, *Temporal Modeling with Reversible Transformers* [135] | 18 | Three byte-identical author-source copies and 15 exact PDF image-object extractions | Open: confirm final publisher reuse terms |
| Kulyk and Mokin, *Scaling Video Prediction with Spatio-Temporal Patches* [134] | 6 | Six exact PDF image-object extractions | Open: confirm journal licence/permission |
| Kulyk and Mokin, *Methods for Ensuring Consistent Generation in Diffusion Models* [133] | 2 | Exact embedded-image extraction with `pypdf`; encoded bytes unchanged | Open: confirm journal permission and the terms of the underlying ControlNet/StyleAligned examples |

Only original Figures 7 and 9 from the 2024 consistency review are retained. Figure 7 provides the structural-conditioning example used to distinguish geometry control from identity control; Figure 9 provides the shared-attention/style example used to distinguish appearance consistency from object identity. Every other figure in that review is omitted. Those omitted review graphics would either repeat the dissertation-native taxonomy and interface diagrams, add no temporal or hierarchical-memory evidence, or increase third-party reuse obligations without strengthening the argument. The two retained figures are explicitly labelled as literature-review context, not evidence for the hierarchical predictor.

## Provenance contract

All three ledgers now use the same schema version and record:

- figure and file identity, asset class, SHA-256, byte size, and raster or PDF dimensions;
- modification, rights, and permission status;
- scientific role and caption attribution;
- source paper, original figure/page, and extraction method for every reused published asset;
- generator, generator SHA-256, and numerical, implementation, or conceptual source for every author-generated asset.

| Ledger | Records | SHA-256 |
|---|---:|---|
| `figure_provenance.json` | 26 | `47641c2b7ad904b2defd529eb31dcb427ba7e1cfe3036d14e743bbab58bf3f8c` |
| `ch2_ch4_figure_provenance.json` | 72 | `1822fa5edcf8a5ad2464c768f556777fcf0b6b9c06dac92dd294e2263ea84f77` |
| `existing_figure_provenance.json` | 21 | `3d1fbe324c9fe1e5b289b719422409c769fe5a16a91f066006b2aaaf287b0241` |

The audit fails closed on incomplete metadata, stale generator hashes, file/hash/byte/dimension drift, duplicate ledger ownership, missing source notes, caption-to-ledger number mismatches, non-white corners in author-generated rasters, and loss of byte parity with supplied published sources.

## Open rights items

| Priority | Assets | Required action |
|---|---|---|
| Deposit blocker | 18 TempVerseFormer figures | Confirm the publisher's dissertation/thesis reuse licence or obtain written permission. |
| Deposit blocker | 6 V-BLT figures | Confirm the journal's dissertation/thesis reuse licence or obtain written permission. |
| Deposit blocker | 2 figures from the 2024 review | Confirm journal reuse and determine whether the embedded ControlNet/StyleAligned examples require permission or replacement with a citation-only author schematic. |
| Deposit blocker for public distribution | Figures 4.3 and 4.9 | Record the applicable MNIST-derived dataset terms for the displayed glyphs/ground truth and project outputs. |

No other linked figure was identified as a pixel-copy of third-party or publisher artwork.

## Page-layout recommendations

These are typesetting requirements, not requests to redraw the figures.

| Placement | Figures | Reason |
|---|---|---|
| Landscape or dedicated full-width page | 2.3, 2.17, 2.26, 4.1, 4.4, 4.16, 4.17 | Multi-column structure or dense labels become too small at ordinary portrait text width. Figure 4.1's matrix should not be relied on as the only readable registry; Table 4.2 remains authoritative. |
| Maximum-width/full-height portrait page | 2.6, 2.8, 4.9 | Tall architecture/procedure/gallery. Preserve the published TempVerseFormer detail; do not simplify or redraw it. |
| Full text width, avoid side-by-side placement | 2.5, 2.11--2.16, 2.21, 2.22, 2.25, 3.3, 3.9, 3.14, 4.3, 4.7, 4.10--4.15 | Small labels, multiple panels, or long interval annotations need the available width. |

Figure 2.17 should be assembled as a landscape 2-by-2 group or as two consecutive full-width rows. Its four published rasters must remain separate and unmodified. Figure 2.6 is dense by design and remains legible at original resolution; its detail is appropriate for a dissertation when it receives a dedicated page.

## Automated verification

The final command was:

```text
python scripts/audit_restructured_dissertation.py
```

Result: exit code 0; 79 numbered figures; Chapter counts 18/27/17/17; 170 references defined and cited; no missing figure, provenance, hash, byte-size, dimension, generator, source-note, caption-number, or source-parity error. The only audit warning is the intentionally open set of 28 permission/data-licence records described above.

`git diff --check` also passes. A final page-by-page PDF proof remains necessary after institutional typesetting to check actual-size labels, greyscale behaviour, cross-references, page breaks, and caption separation. No scientific visual refactor remains open before that stage.
