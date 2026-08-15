# Dissertation and Fourth-Article Restructure and Writing Blueprint

**Date:** 2026-08-10  
**Status:** binding plan for the next content rewrite; current manuscript sources are not modified by this document  
**Primary dissertation source after implementation begins:** `docs/dissertation/rewrite/`  
**Current assembled review source:** `docs/dissertation/final/DISSERTATION_MANUSCRIPT.md`  
**Fourth-article source:** `docs/article/final/ARTICLE_REACH_FIDELITY_TRUST.md`

## 1. Purpose and authority

This document records the agreed narrative, detailed table of contents, content allocation,
visual plan, mathematical plan, size targets, writing rules, and later execution sequence. It is
intended to survive context summarization and prevent a future rewrite from returning to the
publication-strand structure.

### 1.1 Reference basis

The structural and presentation reference supplied by the author is:

- `C:/Users/leoni/Downloads/Ісаєнков_дисертація_final 2.pdf`;
- `C:/Users/leoni/Downloads/Ісаєнков_дисертація_final 2.docx`.

The scientific source papers inspected for this revision are:

- `C:/Users/leoni/Downloads/Article_LKulyk_OMokin_124_2024.pdf` -- the qualitative consistency
  taxonomy and control-method review;
- `C:/Users/leoni/Downloads/Temporal_Modeling_with_Reversible_Transformers__NCAv2_.pdf` -- the
  TempVerseFormer method, controlled temporal-pattern experiments, memory/time evidence, and
  OpenSTL comparisons;
- `C:/Users/leoni/Downloads/Article_LKulyk_OMokin_124_2025_mt.pdf` -- the fixed-patch V-BLT proof
  of concept, RoPE-ND, categorical byte prediction, and entropy-map evidence;
- `C:/Users/leoni/.codex/attachments/7a4e8da1-9ee6-4421-b46b-80c84d7774cb/pasted-text.txt` -- the
  earlier proposed dissertation structure used as a requirements checklist;
- `C:/Users/leoni/.codex/attachments/4337ed60-dfdc-4ce6-bc5d-fe30eceb5c2d/pasted-text.txt` -- the
  supplied prose-editing recommendations, interpreted as clarity and anti-uniformity guidance.

The reference is used for its continuous thesis-like exposition, restrained front matter,
paragraph-to-paragraph movement from problem to method and evidence, integration of visuals with
the surrounding argument, and limited use of publication labels in the scientific body. Its
scientific content is not copied, and its speciality-126 conventions do not override the
requirements applicable to this speciality-124 dissertation. The present dissertation may also
use a higher mathematical density where equations genuinely specify a model, cost, or estimand.

For the next content refactor, this blueprint supersedes earlier narrative plans and the old
resource-synthesis framing where they conflict with it. It does **not** supersede frozen scientific
artifacts, experiment verdicts, evidence manifests, published-paper boundaries, the author-input
checklist, or the published-figure freeze.

No new experiment is required for the structure defined here. A new experiment becomes necessary
only if the dissertation later claims that adaptive entropy-driven regioning or reversible temporal
training was implemented and jointly evaluated inside the final hierarchical predictor.

### 1.2 Source-assimilation rule

Text from the author's papers may supply facts, equations, experiment settings, numerical results,
and the historical progression of an idea. It must not be pasted into the dissertation as a block.
The dissertation rewrite reconstructs each passage around its role in the thesis argument,
rechecks terminology and citations, removes promotional language, and preserves the narrower
evidence boundary established by the corresponding experiment.

Published figures selected for reuse remain pixel-identical. They receive a dissertation caption,
an accurate source statement, and enough page width to remain legible. If a published figure is
ambiguous, the original stays unchanged and a separately numbered dissertation-native companion
diagram explains the missing detail. Figures inherited from a third-party source inside an author
paper require the original citation and a permissions check; author reuse alone is not sufficient.

Charts derived from published tables must say that they are derived point estimates. They must not
invent error bars, seed variation, sample sizes, calibration, or uncertainty that the source paper
did not report.

## 2. Agreed scientific position

### 2.1 Narrative spine

The research develops through four questions:

1. **What must remain consistent in generated visual sequences?**
2. **What prevents a temporal model from being trained through long processes?**
3. **How can visual information be represented without assigning the same capacity everywhere?**
4. **When does hierarchical retained context improve long-horizon prediction under a fixed budget?**

The dissertation is therefore about one systems problem:

> How should a long-horizon video predictor allocate spatial detail, retained history, and
> training memory so that task-relevant consistency can be maintained under bounded resources?

The fourth article and its hierarchical predictor are the empirical core. The earlier works supply
the evaluation language and two separately validated interventions:

- the consistency study defines the requirements and failure vocabulary;
- TempVerseFormer addresses activation growth during long temporal differentiation;
- V-BLT and RoPE-ND address hierarchical visual representation, positional structure, and
  model-internal spatial-complexity estimation;
- the final hierarchical-context study addresses retained history, reach, fidelity, and reliable
  reuse.

These components form a coherent systems methodology. They are **not** presented as four stages of
one jointly trained network.

### 2.2 Non-negotiable claim boundaries

1. The final hierarchical predictor does not contain V-BLT, RoPE-ND, or TempVerseFormer.
2. The V-BLT publication evaluated fixed `4 x 4 x 4` spatio-temporal patches. It produced
   per-byte categorical entropy visualized as pixel maps, but fully dynamic entropy-driven
   regioning remains a proposed extension unless a separate frozen experiment is supplied.
3. Reversible temporal computation was validated separately. It may be described as an available
   training method when activation storage binds, not as the cause of the hierarchical-context
   results.
4. The two papers use a shared procedural 2-D rotation family. Call it the identical dataset only
   after generator code, rendering, split, conditioning, and target settings are verified. The 3-D
   polycube testbed belongs to V-BLT. OpenSTL is direct TempVerseFormer evidence and a later
   dissertation bridge, not evidence from the V-BLT paper.
5. LoopMom is a purpose-built mechanism-identification instrument. It is not natural-video
   evidence.
6. Persistent-object long Moving-MNIST is an external synthetic boundary, not a replication of
   LoopMom geometry.
7. The central hierarchy claim remains conditional: hierarchy pays in the tested reach-limited
   region; flat context recovers when fine fidelity becomes limiting.
8. No continuous Pareto crossing or universal hierarchy ranking is claimed.
9. The 2024 consistency paper is a qualitative static-image review. Its style, identity, shape,
   and conceptual criteria are extended to video; they are not temporal or hierarchical evidence
   by themselves.
10. TempVerseFormer supports reduced activation-memory growth with a measured computation premium
   and competitive, not universally superior, predictive quality. It does not demonstrate
   continual adaptation, resolution of temporal decay, edge deployment, or end-to-end integration
   with the Chapter 3 predictor.
11. The reversible state boundary must be verified against the implementation before final prose.
   The paper's clone, projection, fusion, and VAE interfaces are not automatically invertible. Do
   not call the full VAE-to-VAE pipeline exactly reversible until the persistent state, split versus
   duplication, feedback tensor, loss placement, and retained boundary values are reconciled.
12. V-BLT works on uncompressed raster pixel-channel bytes, not a compressed video bytestream.
   Byte targets avoid a learned output codebook but do not prove a lossless hidden representation.
13. Predictive entropy is a complexity/dispersion diagnostic. The published paper does not show
   calibrated uncertainty, a quantitative entropy-to-dynamics correlation, or improved
   compute-quality trade-offs from an adaptive controller.

### 2.3 Publication references in the prose

Publication labels such as `[A1]` through `[A4]` are allowed only in:

- the front-matter publication list;
- the personal-contribution subsection of the Introduction;
- one attribution sentence at the end of a relevant chapter if required by VNTU;
- an appendix that maps publications to dissertation results.

The scientific body must read as a continuous dissertation. It should not repeatedly say “the
first article,” “the second article,” “the fourth article,” or “this strand.”

### 2.4 Source-paper facts that must survive the rewrite

| Source | Facts to retain | Qualification to retain |
|---|---|---|
| 2024 consistency review | Style/appearance, identity, geometry/shape, and conceptual criteria; control through reference input, parameter adaptation, structure, inversion/noise, or shared attention | Qualitative static-image review; no dataset, experiment, formal operationalization, or systematic-review protocol |
| TempVerseFormer | 64 x 64 RGB triangle generator; constant, acceleration, deceleration, oscillation, interruption, and combined regimes; context 12; training curriculum of 1-5 predicted steps; evaluation to 12; 30,000 updates; batch 32; 2,000 evaluated sequences; FID/MSE/SSIM | One fixed generator seed and no multi-seed inference; results are mixed across regimes; temporal chaining and reversibility cannot be causally separated by the published comparisons |
| TempVerseFormer memory/time | Temporal-model memory remains near 0.89 GB at the reported 1-, 4-, and 16-step points while matched non-reversible transformer variants rise to about 2.60-2.61 GB at 16; the long-horizon plot remains near 1 GB through the reported 1,024-step range | Training time is higher; temporal-model memory, total-pipeline memory, replay metadata, and extended/extrapolated points must be distinguished |
| TempVerseFormer OpenSTL | Moving MNIST, Moving MNIST-CIFAR, KTH Action, and TaxiBJ comparisons; best reported TaxiBJ MSE and otherwise competitive results | Not best on every dataset/metric; no uncertainty or declared qualitative selection rule |
| V-BLT setup | 12-frame 64 x 64 rotating 2-D triangles and 3-D polycubes; fixed 4 x 4 x 4 patches; local encoder, global transformer, local decoder; causal time mask; categorical 256-way byte objective; RoPE-ND | No real-video/OpenSTL evaluation, seed uncertainty, component ablation, scaling sweep, or calibrated uncertainty |
| V-BLT cost | Reported size/GFLOPs: V-BLT 6.5 MB/51.2, ViViT 6.5 MB/59.3, UNet-3D 24.0 MB/70.4 | Identical model-size numbers do not prove equal parameter count; GFLOPs are not measured latency, throughput, or activation memory |
| V-BLT quality | 2-D V-BLT 0.039 MSE, 0.98 SSIM, 32.3 PSNR; 3-D V-BLT 0.11 MSE, 0.91 SSIM, 23.4 PSNR | ViViT reports the best 3-D PSNR, 24.8; aggregation and normalization must be verified before derived plotting |

The numerical receipts above are planning anchors, not replacements for the final source table and
implementation audit.

## 3. Global document design

### 3.1 Size and artifact budget

The ranges below supersede every earlier local word target in this blueprint. They were recalibrated
from two actual Word/PDF references rather than from a generic words-per-page estimate: the
colleague dissertation occupies 208 A4 pages, and the author's earlier template occupies 129 pages
with 21,525 Word-counted words and 29 inline figures. The English working edition must remain at or
below 200 rendered Word pages. This leaves a practical translation margin for the Ukrainian edition,
which must not exceed 220 pages including front matter, references, and appendices.

These are maximum envelopes, not reasons to pad a short section. Page count measured in the final
Word layout prevails over word count. If a later section-level paragraph target conflicts with this
table, compress or remove the paragraph; do not preserve obsolete volume for its own sake.

| Part | Target English words | Rendered-page envelope | Purposeful figures | Main-text tables | Displayed equations |
|---|---:|---:|---:|---:|---:|
| English abstract | 1,100-1,500 | 4-5 | 0 | 0 | 0 |
| Ukrainian annotation, later translation | comparable content | 4-6 | 0 | 0 | 0 |
| Publications, contents, abbreviations | concise institutional form | 7-10 | 0 | 0-1 | 0 |
| Introduction | 1,800-2,200 | 6-7 | 0 | 0 | 0 |
| Chapter 1 | 4,800-5,800 | 21-26 | 13-16 | 2-4 | 7-10 |
| Chapter 2 | 6,000-7,200 | 27-33 | 16-21 | 5-7 | 30-42 |
| Chapter 3 | 6,500-7,600 | 28-34 | 12-15 | 4-6 | 30-42 |
| Chapter 4 | 7,500-8,800 | 31-38 | 14-17 | 6-8 | 6-9 |
| General Conclusions | 1,200-1,500 | 4-5 | 0 | 0 | 0 |
| References | 170-190 entries | 13-17 | 0 | 0 | 0 |
| Appendices | receipts essential to examination | 8-12 | 0-3 | compact registries | only when necessary |
| Complete English working edition | about 31,000-35,000 scientific words | **target 185-200 pages; hard ceiling 200** | purposeful inventory, currently 86 | currently 30 | normally 100-110 |

Appendices count toward the hard ceiling. Full machine-readable registries, detailed per-seed dumps,
hash ledgers, and software logs remain in the repository and are cited by path; they are not copied
into the dissertation merely for completeness. An appendix retains only material that an examiner
must see inside the deposited document.

The mathematical density may be higher than in the provided 126 dissertation. Equations are
welcome where they define the system, prove a property, specify an estimand, or make an
experimental comparison reproducible. They must not be added merely to make a section look more
technical.

There is no numerical figure quota. A figure is retained whenever it materially clarifies an
architecture, mechanism, dataset, experiment, or boundary and remains legible at dissertation
width. Related panels should be combined when they answer one question; redundant pedagogical
diagrams and repeated galleries should be removed. Figure count is never increased by splitting a
coherent chart into arbitrary panels or adding decorative samples without an argumentative role.
Each figure must perform one of four jobs: define a system, explain an algorithm, show a qualitative
behaviour, or support an empirical decision.

### 3.2 Front-matter artifact rule

The annotations and Introduction must be plain text. They must contain:

- no figures;
- no graphs;
- no tables;
- no displayed equations;
- no evidence crosswalk;
- no architecture schematic.

Quantitative results may be written in sentences. Essential notation may be named verbally, but
its mathematical definition begins in Chapter 1.

### 3.3 Visual style for new figures

All newly created schematics must use:

- a white background;
- one consistent sans-serif typeface;
- dark text with a restrained blue/teal/orange palette;
- arrows with unambiguous direction and no crossings where avoidable;
- consistent meanings for solid and dashed lines;
- captions that state what the reader should learn, not only what elements are shown;
- readable labels at dissertation page width;
- vector PDF plus high-resolution PNG where practical.

Published TempVerseFormer and V-BLT figures remain unchanged. Their detail is acceptable. They may
be placed on a full page or landscape page if required, but their contents must not be redrawn or
simplified.

The same preservation rule applies to selected figures from the 2024 consistency paper. A reused
source image is never silently recolored, relabelled, cropped into a different claim, or presented
as newly generated evidence. Several unchanged source figures may be placed in a dissertation
gallery when each remains intact and separately attributed.

For the page-bounded figure programme:

- do not place more than two dense figures on one portrait page;
- use half-width placement only when every label remains at least 8.5-9 pt at final size;
- give architecture, qualitative sequence, and attention-mask figures full width;
- use a landscape page for the complete TempVerseFormer architecture and any four-source OpenSTL
  gallery that is unreadable in portrait orientation;
- distinguish source-reused, artifact-derived, and conceptual figures in the caption/source note;
- place optional historical or repetitive galleries in appendices before cutting an explanatory
  model or decision figure;
- hash-lock every newly extracted published figure before manuscript assembly.

### 3.4 Visual candidate inventory: select by argumentative value and page cost

The following inventory is a candidate pool, not a binding minimum. During compaction, retain the
strongest visual for each explanatory or evidential job and combine compatible panels. Any item may
move to an appendix or be omitted if it repeats a nearby explanation, consumes disproportionate
page area, or becomes illegible at the final width. A compound gallery counts as one dissertation
figure only when every source image remains intact, legible, and separately attributed.

Source-status terms:

- **existing** -- a repository asset exists but still receives a final semantic and visual check;
- **new** -- a dissertation-native schematic or artifact-derived plot must be produced;
- **published unchanged** -- reuse without pixel modification, subject to rights and attribution;
- **derived** -- regenerate from frozen numeric evidence or exact published point estimates;
- **conditional** -- retained only if it survives the page-budget and non-duplication test.

#### Chapter 1 visual inventory: 21 main-text figures plus one appendix figure

| ID | Role | Source/status | Planned location |
|---|---|---|---|
| 1.1 | Partially observed video-prediction system boundary | existing `fig30_control_system`, verify routing | 1.1.1 |
| 1.2 | Hidden world state versus viewport observation storyboard | new | 1.1.1 |
| 1.3 | Teacher-conditioned versus self-fed recurrence | new | 1.1.2 |
| 1.4 | Parameter, optimizer, activation, context/state, and transient-memory locations | new | 1.1.3 |
| 1.5 | Conceptual multi-step consistency landscape | existing `fig31_consistency_landscape`; label conceptual | 1.2.1 |
| 1.6 | Illustrative SDXL outputs and the limit of single-image inspection | 2024 paper Fig. 1, published unchanged; non-evidentiary | 1.2.1 |
| 1.7 | Multidimensional static consistency taxonomy | new; informed by the 2024 paper | 1.2.2 |
| 1.8 | Local visual input versus integrated scene understanding | 2024 paper Fig. 5, published unchanged; heuristic only | 1.2.2 |
| 1.9 | Appearance, identity, geometry, trajectory, and hallucination failure montage | new | 1.2.3 |
| 1.10 | Temporal and trajectory consistency operationalization | dissertation-native | 1.2.3 |
| 1.11 | Consistent and contradictory macro/meso/micro predictions | dissertation-native | 1.2.4 |
| 1.12 | Extension from static criteria to temporal, trajectory, and cross-level video criteria | new | 1.2.5 |
| 1.13 | Forward diffusion as progressive corruption | 2024 paper Fig. 2, published unchanged; diffusion time separated from video time | 1.2.5 |
| 1.14 | Geometry/shape conditioning through ControlNet | 2024 paper Fig. 7 published unchanged; cite original ControlNet | 1.2.5/1.4.4 |
| 1.15 | Style consistency under changing object content | 2024 paper Fig. 9 published unchanged; cite StyleAligned | 1.2.5/1.4.4 |
| 1.16 | Five control interfaces: input, weights, structure, latent/noise, attention | new synthesis | 1.2.5/1.4.4 |
| 1.17 | Geometry/support versus identity/fidelity bridge | new; analogy to E6, not shared mechanism | 1.2.5 |
| 1.18 | Activation-tape growth across a temporal unroll | new | 1.3.1 |
| 1.19 | Fixed spatial grid under non-uniform information density | existing `fig35_fixed_grid_tokenization` | 1.3.2 |
| 1.20 | Relevant history outside a dense flat context window | existing `fig39_flat_long_context_barrier`, retain schematic scale | 1.3.3 |
| 1.21 | Three bottlenecks, separately validated methods, and final decision layer | existing `fig61_dissertation_systems_map`, moved from Introduction and revised only if needed | 1.3.4/1.5 |

Original Figure 3 from the 2024 paper is retained unchanged in Appendix H because its simplified
reverse-process notation is useful historically but is not the dissertation's authoritative
mathematical definition. A future author-supplied video-generation example has a visible,
unnumbered insertion point after Figure 1.13 and becomes a counted figure only when its generation
receipt and declared qualitative role are available.

#### Chapter 2 visual inventory: 23 core plus 4 conditional

| ID | Role | Source/status | Planned location |
|---|---|---|---|
| 2.1 | Dataset progression from controlled motion to long-memory instruments | new | 2.1.4 |
| 2.2 | Rotating 2-D triangle sequence | 2025 V-BLT paper Fig. 5, published unchanged | 2.1.2 |
| 2.3 | Rotating 3-D polycube sequence | 2025 V-BLT paper Fig. 6, published unchanged | 2.1.3 |
| 2.4 | Standard BPTT activation tape | new | 2.2.1 |
| 2.5 | Additive reversible transformer block | TempVerseFormer Fig. 2, published unchanged; cite RevViT origin | 2.2.2 |
| 2.6 | Complete TempVerseFormer architecture | TempVerseFormer Fig. 3, published unchanged, full/landscape page | 2.2.3 |
| 2.7 | Temporal chaining mechanism | TempVerseFormer Fig. 4, published unchanged | 2.2.3 |
| 2.8 | Time-agnostic backpropagation | TempVerseFormer Fig. 5, published unchanged | 2.2.4 |
| 2.9 | Five-step constant-rotation prediction comparison | TempVerseFormer Fig. 6, published unchanged | 2.2.5 |
| 2.10 | Constant-rotation FID versus prediction horizon | TempVerseFormer Fig. 7, published unchanged | 2.2.5 |
| 2.11 | Eleven-step prediction beyond the five-step training horizon | TempVerseFormer Fig. 8, published unchanged | 2.2.5 |
| 2.12 | Accelerated-rotation predictions | **conditional**, TempVerseFormer Fig. 9 unchanged | 2.2.5 or appendix |
| 2.13 | Decelerated-rotation predictions | **conditional**, TempVerseFormer Fig. 10 unchanged | 2.2.5 or appendix |
| 2.14 | Oscillating-rotation predictions | **conditional**, TempVerseFormer Fig. 11 unchanged | 2.2.5 or appendix |
| 2.15 | Interrupted-rotation predictions | **conditional**, TempVerseFormer Fig. 12 unchanged | 2.2.5 or appendix |
| 2.16 | Combined-pattern FID versus horizon | TempVerseFormer Fig. 13, published unchanged | 2.2.5 |
| 2.17 | Memory versus 1-1,024 prediction steps | TempVerseFormer Fig. 14, published unchanged; provenance qualifier retained | 2.2.7 |
| 2.18 | Training time versus prediction horizon | TempVerseFormer Fig. 15, published unchanged | 2.2.7 |
| 2.19 | Moving MNIST, Moving MNIST-CIFAR, KTH, and TaxiBJ qualitative gallery | TempVerseFormer Figs. 16-19, four intact source panels | 2.2.6 |
| 2.20 | Measured fixed-patch V-BLT architecture | 2025 V-BLT paper Fig. 3, published unchanged | 2.3.2 |
| 2.21 | Causal spatio-temporal attention mask | 2025 V-BLT paper Fig. 4, published unchanged | 2.3.3 |
| 2.22 | Fixed `4 x 4 x 4` patch tensor and local/global/local flow | new | 2.3.1-2.3.2 |
| 2.23 | RoPE-ND disjoint coordinate subspaces | new; verify against implementation | 2.3.4 |
| 2.24 | Categorical byte logits to entropy-map pipeline | new | 2.3.5 |
| 2.25 | 2-D and 3-D prediction/entropy comparison | 2025 V-BLT Figs. 7-8 as intact published panels | 2.3.5 |
| 2.26 | V-BLT cost and quality point estimates | derived from published Tables 1-2 after metric audit; no error bars | 2.3.6 |
| 2.27 | Implemented/evaluated versus proposed dynamic allocation | new | 2.3.7 |

The four conditional temporal-pattern galleries move together. Do not keep only the most visually
successful case. The biological replay figure, BLT source architecture, and ViViT architecture are
appendix/literature options because they are third-party-derived and do not define the developed
system.

#### Chapter 3 visual inventory: 16 core plus 1 conditional

| ID | Role | Source/status | Planned location |
|---|---|---|---|
| 3.1 | Re-entry after relevant appearance leaves the dense window | new | 3.1.1 |
| 3.2 | Flat, spatial-hierarchical, and temporal-subsampling packages | existing `fig40_context_package_comparison`, update only if exact arithmetic is missing | 3.1.2 |
| 3.3 | Reach and fidelity as separate axes without an inferred crossing | new | 3.1.3 |
| 3.4 | Frozen two-level codec with tensor dimensions | new | 3.2.1 |
| 3.5 | Exact token sequence and embeddings | existing `fig22_token_sequence_layout` | 3.2.2 |
| 3.6 | Joint-attention block and causal mask | new | 3.2.3 |
| 3.7 | Frozen, detached, and trainable gradient paths | new | 3.2.4 |
| 3.8 | Windowed latent-diffusion training objective | new, specific to the final predictor | 3.3.1 |
| 3.9 | Complete hierarchical-context predictor | existing `fig16_context_packaging_architecture` | 3.3.2 |
| 3.10 | DDIM inference timeline and NFE accounting | new | 3.3.3 |
| 3.11 | Coarse and fine rollout ring-buffer updates | existing `fig23_rollout_ring_buffer`, verify both return paths | 3.4.1 |
| 3.12 | Stream-anchor geometry | new | 3.4.2 |
| 3.13 | Context, target, payload, and total-token arithmetic | new/derived from exact configuration values | 3.4.3 |
| 3.14 | LoopMom world, viewport, and analytic observability | existing `fig15_loopmom_observability_schematic` | 3.5.2 |
| 3.15 | Multi-frame disappearance and re-entry storyboard | new | 3.5.2 |
| 3.16 | Pair-specific reach-differential population | new/derived from registered geometry | 3.5.3 |
| 3.17 | Persistent-object assignment and identity tracking | **conditional**, new | 3.5.5 |

#### Chapter 4 visual inventory: 16 core plus 1 conditional

| ID | Role | Source/status | Planned location |
|---|---|---|---|
| 4.1 | Experiment sequence and evidence levels | existing `fig63_executed_configuration_matrix` or simplified derived replacement | 4.1.1 |
| 4.2 | Regression, diffusion, and energy objectives/inference | existing `fig24_generation_objectives` | 4.2.1 |
| 4.3 | Matched single-step regression and diffusion samples | existing `fig29_mechanism_single_step_samples` | 4.2.1 |
| 4.4 | Corrected EBT update, energy ordering, and descent sharpness | regenerate from `fig25`-`fig27`; candidate update must be correct | 4.2.2 |
| 4.5 | Optimization-stage memory-gate effects | **conditional**, verify and regenerate from `fig28_memory_gate_optimization` | 4.2.2/4.3 |
| 4.6 | Per-seed pair-specific reach-band effects | new/derived from frozen effects | 4.3.1 |
| 4.7 | Untouched tested frontier and exact token arithmetic | existing `article_confirmed_frontier` | 4.4.1 |
| 4.8 | Equal-reach token and payload comparison | new/derived | 4.4.2 |
| 4.9 | Selection-rule-controlled qualitative rollouts | existing `fig7_samples_claim1`, captioned honestly | 4.3.2 |
| 4.10 | Five-seed head-by-package effects | existing `fig17_head_package_seed_heterogeneity` | 4.5.2 |
| 4.11 | Sampler and NFE robustness | existing `fig19_sampler_nfe_robustness` | 4.6.1 |
| 4.12 | Independently trained codec robustness | existing `fig20_alternate_codec_robustness` | 4.6.2 |
| 4.13 | Failed late self-fed-boundary replication | existing `fig18_trust_negative_replication`, verify no data occlusion | 4.7.1 |
| 4.14 | Privileged pose-indexed raw memory | existing `fig21_pose_indexed_raw_memory` | 4.7.2 |
| 4.15 | Persistent-object support/continuity versus identity | existing `fig22_phasec_v2_boundary` | 4.7.3 |
| 4.16 | Separately measured context-state and activation-memory axes | existing `fig62_two_resource_synthesis` | 4.8.1 |
| 4.17 | Final binding-constraint decision flow | new | 4.8.1 |

The historical development `fig9_frontier` and `fig10_flat_vs_hier` remain provenance-only and
must not re-enter the main chapter. Likewise, `fig4_claim3_factorial`, `fig12_cwstyle_curves`, and
`fig13_family_edges` require regeneration from current evidence if they are used in an appendix;
their old labels or interpretations are not accepted.

## 4. Paragraph and prose design system

### 4.1 Objective

The objective is natural, precise academic prose with clear authorial reasoning. It is not to
defeat an AI detector. Automated detector scores are not an evidentiary standard for authorship.
The final text must be reviewed by the author, retain correct citations, and express the author's
actual judgments.

An automatic “humanizer” must not be applied to binding scientific text. Such tools can alter
claim scope, numbers, citations, negation, and technical terminology. If a passage sounds
mechanical, revise its argument and paragraph structure manually.

### 4.2 Paragraph-length codes

Every detailed outline below uses the following codes.

| Code | Target length | Normal function |
|---|---:|---|
| **X** | 20-45 words | short emphasis, result, limitation, or transition; use sparingly |
| **S** | 45-80 words | direct claim, local transition, definition, or interpretive pivot |
| **M** | 80-120 words | standard evidence paragraph: claim, support, interpretation |
| **L** | 120-170 words | literature synthesis, method explanation, or multi-step reasoning |
| **F** | 60-150 words around an equation | motivation before the equation and interpretation after it |
| **V** | 60-160 words around a figure | setup, figure, then a concrete reading of the figure |
| **T** | 60-160 words around a table | comparison question, table, then interpretation and limitation |

These are ranges, not templates. A substantial subsection should mix paragraph lengths; a short
subsection need not manufacture three paragraph types. No prose paragraph should exceed 170 words
in the compact edition. Do not place more than two paragraphs of nearly equal length next to one
another. A longer technical paragraph may be followed by a short interpretive statement.

### 4.3 Sentence rhythm

1. Mix short conclusions with longer explanatory sentences.
2. Do not start three consecutive sentences with “The,” “This,” “These,” or the same noun phrase.
3. Use different logical openings when they are justified: a result, a condition, a contrast, a
   concrete example, a limitation, or a consequence.
4. Avoid a permanent claim-evidence-summary rhythm in every paragraph. Some paragraphs should
   introduce a difficulty; another may resolve it only later.
5. Do not end every subsection with “Thus,” “Therefore,” or a mini-summary.
6. Lists of three are allowed only when the subject genuinely has three components. Do not force
   every argument into a triplet.
7. Prefer one exact verb to an abstract noun phrase: “the window misses the earlier frame” is
   better than “a limitation in historical accessibility is observed.”
8. Use occasional restrained judgment: “What matters here is...”, “The comparison is informative,
   but...”, or “This result is weaker than the aggregate score suggests.” Such statements must be
   supported, not decorative.

### 4.4 Paragraph construction rules

- Start with the scientific point, not with “It is important to note,” “Before discussing,” or
  “In the modern world.”
- One paragraph should normally do one job. A long paragraph may link two ideas only when the link
  itself is the point.
- Do not explain a claim, restate it, and then summarize it again.
- When an equation appears, introduce the problem it solves before the equation and interpret at
  least one term or implication afterward.
- When a figure appears, tell the reader what relationship to inspect. Do not repeat every label in
  the caption and prose.
- When a table appears, discuss the comparison or pattern, not each cell sequentially.
- Vary citation position. Some citations support a clause; others close a synthesis paragraph.
- A paragraph containing an important limitation may be short. It need not be padded to match its
  neighbors.

### 4.5 Repetition controls

The following expressions should be rare in the scientific body:

- “It is important to note that”;
- “within the framework of this dissertation”;
- “this dissertation does not claim”;
- “not a universal...”;
- “rather than...” repeated as a rhetorical pattern;
- “the present study” in consecutive paragraphs;
- “strand,” “crosswalk,” “binding source,” “claim ledger,” and “evidence boundary.”

Scope must be handled locally and once. For example, the V-BLT subsection contains one explicit
measured-versus-proposed paragraph; later chapters should not repeat that qualification unless it
directly affects an interpretation.

### 4.6 Citation rule

Each externally checkable technical or historical claim should have a suitable source. Citations
must be attached to the claim they support rather than collected at the end of a long paragraph.
Primary papers and official documentation are preferred.

The literature target is determined by coverage, not a quota. A final bibliography of 170-230
well-used references is acceptable. Adding papers that are never synthesized in the prose is not.

### 4.7 Manual prose-quality pass

The paragraph rhythms in this blueprint are prompts, not a template that should become visible in
the final manuscript. After a subsection is scientifically complete, run a separate manual pass:

1. Read only the first sentence of every paragraph. Together they should form a clear argument,
   not a sequence of interchangeable topic sentences.
2. Compare the first four words of adjacent sentences. Rewrite clusters that open in the same way.
3. Mark every sentence that merely restates the previous sentence. Delete it unless it adds a new
   condition, example, consequence, or limitation.
4. Vary paragraph size because the reasoning varies, not because a detector or numerical rule asks
   for artificial variation.
5. Replace generic transitions with the actual relation: contradiction, consequence, limitation,
   mechanism, or change of scale.
6. Keep one authorial judgment where it helps the reader understand a decision. Then support it
   with evidence; do not add conversational opinion for decoration.
7. Check every repeated central phrase across chapters. Define it once, use a shorter reference
   later, and avoid concluding the same result in both the preceding and following paragraphs.
8. Read equations, figures, and tables with their surrounding paragraphs. The prose must explain
   why the artifact is present and what conclusion is justified, without narrating every symbol or
   cell.
9. Preserve uneven but purposeful rhythm: a long derivation may end with a short result; a short
   limitation may interrupt an otherwise extended discussion.
10. Finish with an author review of intent, terminology, numbers, citations, and claim scope.

No automatic humanizer is part of this workflow. The desired naturalness comes from rewriting the
reasoning structure and from genuine author decisions, not from disguising provenance or replacing
words mechanically.

## 5. Detailed dissertation structure

## 5.1 Annotation

**Artifact rule:** plain text only; no figure, table, or formula.

**Target:** 1,000-1,400 words for the English annotation, excluding the publication list and
mandatory bibliographic material.

| Paragraph block | Rhythm | Content |
|---|---|---|
| A1 | M | Bibliographic opening and one-sentence identification of the scientific problem. |
| A2 | L | Why long-horizon video prediction is a systems problem: partial observability, bounded resources, and accumulating inconsistency. |
| A3 | S | The unresolved gap. Existing quality improvements do not isolate training horizon, spatial allocation, and retained-history reach. |
| A4 | M | Aim and system-level decomposition into the three bottlenecks. No equation is shown. |
| A5 | L | Consistency criteria and the role of the first analytical study. Keep this conceptual, not chronological. |
| A6 | L | Reversible temporal method, its memory result, and applicability boundary. |
| A7 | L | V-BLT, RoPE-ND, static-patch validation, and entropy maps. State that dynamic entropy regioning is future work. |
| A8 | L | Final hierarchical-context model and exact-observability instrument. |
| A9 | L | Principal frontier results, reach efficiency, robustness, and support-identity boundary. |
| A10 | M | Scientific novelty in compact prose. Avoid a repeated numbered novelty list if the institutional template does not require it here. |
| A11 | S | Practical significance, software, and implementation scope. |
| A12 | X or S | Final design rule and limits. End directly. |

The Ukrainian annotation should later be translated from the scientifically approved English
version, while preserving meaning rather than sentence order.

## 5.2 Introduction

**Artifact rule:** plain text only; no figure, table, or displayed formula.

**Target:** 2,800-3,600 words, approximately 8-11 formatted pages.

### 5.2.1 Relevance and justification of the topic

**Target:** 750-950 words.  
**Paragraph rhythm:** `M-L-S-M-L-S`.

Content sequence:

1. Begin with the concrete long-horizon video problem, not with the history of artificial
   intelligence.
2. Explain why disappearance, re-entry, identity, trajectory, and fine appearance cannot be
   handled by short-frame quality alone.
3. Introduce bounded training and inference resources in words.
4. Separate the three bottlenecks: activation storage, spatial allocation, and retained context.
5. State the knowledge gap: existing studies rarely isolate these mechanisms under compatible
   budgets and observability conditions.
6. Conclude with the need for a systems-analysis methodology and a controlled final model study.

The first paragraph may be longer and scene-setting. The third or sixth should be short enough to
create a deliberate pause.

### 5.2.2 Relationship to research programmes, plans, and topics

**Target:** 180-280 words.  
**Paragraph rhythm:** `M-S`.

State the verified programme titles, dates, project numbers, and the applicant's role. Do not use
this section to explain the scientific narrative. Unverified fields remain explicit author inputs.

### 5.2.3 Aim and research tasks

**Target:** 400-550 words.  
**Paragraph rhythm:** `S-M-L-S` plus a six-item task list.

The aim is to improve task-relevant consistency and resource efficiency in long-horizon
hierarchical video prediction through systems analysis, separately addressing temporal training,
spatial representation, retained context, and evaluation.

The six tasks are:

1. analyze the system, uncertainty classes, and consistency requirements;
2. develop and delimit reversible long-horizon temporal training;
3. develop hierarchical visual representation, multidimensional position encoding, and
   entropy-based spatial-complexity estimation;
4. develop a hierarchical long-context predictor under explicit budgets;
5. develop exact-observability, complete-metric, and statistically valid evaluation methods;
6. evaluate the methods and identify their operating regions and limitations.

Do not add a contribution/evidence table after the task list.

### 5.2.4 Object and subject of research

**Target:** 120-180 words.  
**Paragraph rhythm:** `S-S`.

Use one paragraph for the object and one for the subject. Keep their scope narrower than “all
generative systems.”

### 5.2.5 Research methods

**Target:** 300-450 words.  
**Paragraph rhythm:** `M-L-S` plus a compact list if required.

For each method family, state what it was used to investigate. Cover systems decomposition,
information theory, reversible transformations, latent-variable modelling, diffusion, controlled
simulation, assignment-based object evaluation, cluster-aware resampling, and simultaneous
inference.

### 5.2.6 Scientific novelty

**Target:** 450-650 words.  
**Paragraph rhythm:** one short setup followed by four or five novelty items of deliberately
unequal length.

Novelty items:

1. consistency criteria and their trajectory-level operationalization;
2. reversible temporal computation and its measured applicability boundary;
3. V-BLT, RoPE-ND, and entropy-map-based spatial-complexity estimation, with dynamic regioning
   clearly identified as future work;
4. hierarchical context packaging and the measured reach-fidelity frontier;
5. exact-observability and paired clip-cluster evaluation for memory-dependent prediction.

Each item must say what differs from known work. Avoid using “for the first time” unless it can be
defended against the final literature review.

### 5.2.7 Practical significance

**Target:** 220-340 words.  
**Paragraph rhythm:** `M-S-M`.

Describe the software toolkit, reusable datasets/instruments, configuration validation, evidence
generation, and documented implementation acts. Do not imply deployment validation in domains
that were not tested.

### 5.2.8 Reliability and validity of results

**Target:** 230-350 words.  
**Paragraph rhythm:** `M-L-S`.

Mention analytical checks, controlled generators, untouched and protected populations, explicit
comparators, cluster-aware inference, multiple seeds where available, robustness studies, and
negative-result reporting. This is the only broad reliability summary before Chapter 4.

### 5.2.9 Personal contribution

**Target:** 220-320 words.  
**Paragraph rhythm:** one direct setup plus publication-specific bullets or short paragraphs.

Article labels may appear here. Distinguish co-authored and sole-author contributions. Keep
dissertation-only extensions separate from published claims.

### 5.2.10 Approbation and publications

**Target:** determined by mandatory records, normally 180-300 words plus bibliographic entries.

Use verified administrative facts only.

### 5.2.11 Dissertation structure and volume

**Target:** 180-260 words.  
**Paragraph rhythm:** `M-S`.

Describe the four chapters in scientific language without listing every subsection. Insert final
page, figure, table, reference, and appendix counts only after final typesetting.

## 5.3 Chapter 1 — Systems Analysis and Consistency Requirements of Long-Horizon Video Prediction

**Target:** 9,700-12,750 words, 30-40 pages.  
**Artifacts:** 18 core figures plus one conditional published interface figure, 3-5 tables, and
8-14 displayed equations.

### 5.3.1 Video prediction as a partially observed dynamic system

#### 1.1.1 System boundary, state, observations, and controls

**Target:** 700-900 words.  
**Rhythm:** `M-F-L-V-S`.

Define latent world state (x_t), observed frame (y_t), optional action (u_t), stochastic
disturbance, and generated prediction. Explain that the predictor sees observations, not the full
state.

**Equations:** state transition, observation equation, multi-step prediction objective.  
**Figures 1.1-1.2:** white-background system-boundary diagram showing world state, viewport
observation,
control, predictor, and feedback. This is the figure previously proposed for the Introduction.  
Add a separate hidden-state-versus-observation viewport storyboard.  
**Table:** none.

#### 1.1.2 Self-fed prediction and the growth of uncertainty

**Target:** 500-700 words.  
**Rhythm:** `S-L-F-M`.

Distinguish teacher-conditioned training from autoregressive evaluation. Show how an output becomes
the next input and why later errors are distributional as well as numerical.

**Equation:** autoregressive rollout recurrence.  
**Figure 1.3:** four-stage self-fed error-propagation storyboard.  
**Table:** none.

#### 1.1.3 Resource-constrained multi-criteria formulation

**Target:** 600-750 words.  
**Rhythm:** `M-F-F-S`.

Introduce context/state memory, activation memory, computation, and a vector of consistency losses.
The existing total-memory decomposition may be retained here rather than in the Introduction.

**Equations:** total memory decomposition; constrained multi-objective design problem.  
**Figure 1.4:** resource locations along training and inference, with context and activation memory
shown on different paths.  
**Table:** compact resource-term definitions if the prose alone becomes cumbersome.

### 5.3.2 Consistency as a multi-criteria property

#### 1.2.1 Why one quality score is insufficient

**Target:** 500-650 words.  
**Rhythm:** `S-L-V-M`.

Explain why aggregate MSE, perceptual quality, identity, trajectory continuity, and hallucination
penalties answer different questions.

**Figures 1.5 and 1.8:** conceptual consistency landscape and one video strip with four
deliberately different failure modes. The landscape is a dissertation-native figure; it is not
claimed to come from the 2024 publication.  
**Equation:** vector-valued consistency criterion.  
**Table:** none.

#### 1.2.2 Appearance, object, and conceptual consistency

**Target:** 700-850 words.  
**Rhythm:** `M-L-S-M`.

Synthesize the 2024 consistency publication and later literature. Present style/appearance,
identity, shape/geometry, and semantic/concept criteria as overlapping requirements, not mutually
exclusive universal classes. Attribute conceptual consistency as the author's earlier proposed
category rather than settled field taxonomy.

**Figure 1.6:** new multidimensional consistency taxonomy informed by the 2024 paper.  
**Equation:** only if a similarity or conceptual-consistency criterion is used later.  
**Table 1.1:** criteria, failure, representative metric, and limitation.

#### 1.2.3 Temporal and trajectory consistency

**Target:** 650-800 words.  
**Rhythm:** `S-L-F-V-S`.

Distinguish local smoothness from correct long-term trajectory, re-entry, and identity recovery.
Introduce the need for self-fed evaluation without yet describing LoopMom in detail.

**Equation:** generic trajectory loss.  
**Figure 1.14:** trajectory-consistency operationalization. This is a dissertation-native video
extension, not a figure from the 2024 static-image paper.

#### 1.2.4 Hierarchical and cross-level consistency

**Target:** 500-650 words.  
**Rhythm:** `M-L-S`.

Define cross-level agreement as a criterion for models that explicitly contain multiple levels.
Avoid implying that any hierarchy automatically guarantees this property.

**Figure 1.15:** macro/meso/micro example with one consistent and one contradictory cross-level
prediction.  
**Equation:** optional cross-level discrepancy term.

#### 1.2.5 From static control criteria to video requirements

**Target:** 500-700 words.  
**Rhythm:** `S-V-L-V-S`.

Explain which static criteria survive unchanged, which acquire a temporal dimension, and why
geometric support can improve while identity fidelity worsens. The ControlNet "how versus what"
distinction is used as a conceptual precursor to the later support-identity boundary, not as the
same mechanism or evidence.

**Figures 1.7 and 1.10-1.13:** static-to-video extension diagram; unchanged
published ControlNet and StyleAligned figures; a new five-interface comparison; and a new
support-versus-identity bridge.  
**Conditional Figure 1.9:** unchanged published image-to-image interface when permissions and page
balance permit.  
**Table:** method, controlled invariant, control interface, benefit, and limitation.

### 5.3.3 Three bottlenecks behind long-horizon inconsistency

#### 1.3.1 Activation growth during temporal learning

**Target:** 450-600 words.  
**Rhythm:** `S-M-F-S`.

Explain BPTT activation growth and checkpointing at a conceptual level. Detailed reversibility is
reserved for Chapter 2.

**Equation:** asymptotic activation-memory expression.  
**Figure 1.16:** short versus long unrolled training graph.

#### 1.3.2 Uniform spatial allocation

**Target:** 450-600 words.  
**Rhythm:** `M-V-S-M`.

Show how a uniform grid spends equal token capacity on static background and dynamic boundaries.
Do not yet claim a dynamic solution.

**Figure 1.17:** fixed grid over a rotating object, paired with an entropy or complexity overlay.
The variable-allocation branch is labelled proposed, not evaluated.  
**Equation:** token count for fixed spatial patches.

#### 1.3.3 Limited retained-context reach

**Target:** 500-700 words.  
**Rhythm:** `S-V-L-F-S`.

Introduce addressable history, fidelity, disappearance, and re-entry. Distinguish “stored,”
“addressable,” and “reliably reusable.”

**Figure 1.18:** flat short window versus coarse long history for a re-entering object.  
**Equation:** generic effective-reach definition.

#### 1.3.4 Causal map from bottlenecks to interventions

**Target:** 300-500 words.  
**Rhythm:** `V-M-S`.

Connect each bottleneck to an intervention without saying that the interventions form one tested
model.

**Figure 1.19:** the main dissertation problem map, moved out of the Introduction: consistency
requirements at the top, three
bottlenecks in the middle, developed methods below, and final operating-region evaluation at the
bottom. Solid borders indicate executed methods; a distinct annotation indicates separately
validated components.  
**Table:** none.

### 5.3.4 Existing methods and unresolved gap

#### 1.4.1 Long-horizon temporal modelling and memory-efficient training

**Target:** 700-900 words.  
**Rhythm:** `L-M-S-L`.

Synthesize recurrent, transformer, checkpointing, reversible, state-space, and rollout-training
approaches by the resource they control. Avoid a paper-by-paper catalogue.

#### 1.4.2 Adaptive and hierarchical visual representation

**Target:** 700-850 words.  
**Rhythm:** `M-L-M-S`.

Cover fixed patches, multiscale encoders, adaptive tokens, byte-level latent patching, and
positional representation.

#### 1.4.3 Context memory, retrieval, and multirate hierarchy

**Target:** 750-950 words.  
**Rhythm:** `S-L-L-M`.

Cover hierarchical latent video models, compressed memories, retrieval, sparse temporal context,
and modern long-memory benchmarks. Lead toward the need for matched package budgets and exact
observability.

**Table 1.2:** literature grouped by bottleneck, intervention, controlled resource, evaluation
instrument, and unresolved limitation.  
The comparison table carries the literature map. Do not add a decorative chronology merely to
increase the figure count.

### 5.3.5 Problem decomposition and requirements for the developed methods

#### 1.5.1 Derived design requirements

**Target:** 450-600 words.  
**Rhythm:** `S-M-L-S` plus a short requirements list.

Derive requirements for causal processing, explicit resource accounting, multi-criteria
evaluation, long-horizon access, and comparison validity.

#### 1.5.2 Research hypotheses

**Target:** 350-500 words.  
**Rhythm:** `M-S-M`.

State the chapter-level hypotheses that will be tested later. Keep the hierarchy hypothesis
conditional and distinguish training-memory, spatial-allocation, and context-reach predictions.

#### 1.5.3 Chapter conclusions and transition

**Target:** 400-550 words.  
**Rhythm:** `M-S-M-X`.

Summarize findings, not subsection titles. End by directing the reader to the two enabling methods
in Chapter 2 and the retained-context model in Chapter 3.

## 5.4 Chapter 2 — Methods for Feasible Temporal Training and Adaptive Spatial Representation

**Target:** 13,200-16,850 words, 43-56 pages.  
**Artifacts:** 23 core figures plus four conditional temporal-pattern figures, 7-10 tables, and
28-46 displayed equations.

### 5.4.1 Shared controlled experimental foundation

#### 2.1.1 Why controlled procedural video is used

**Target:** 400-550 words.  
**Rhythm:** `S-L-M`.

Explain isolation of dynamics, reproducibility, effectively unlimited sequences, and the limits of
synthetic evidence.

#### 2.1.2 Rotating 2-D shapes

**Target:** 350-500 words.  
**Rhythm:** `V-M-S`.

Describe geometry, rotation regimes, rendering, context, and its role in the two papers. Call it a
shared procedural rotation family until generator code, splits, rendering, conditioning, and
targets are verified as identical.

**Figure 2.2:** unchanged published V-BLT 2-D rotation sequence. Variable TempVerseFormer regimes
are shown later with their results.  
**Equation:** angular-state update if needed.

#### 2.1.3 Rotating 3-D polycubes

**Target:** 350-450 words.  
**Rhythm:** `V-L-S`.

Describe structural diversity, three-axis rotation, self-occlusion, shading, and its V-BLT role.

**Figure 2.3:** unchanged published V-BLT polycube progression and occlusion example.

#### 2.1.4 Dataset progression across the dissertation

**Target:** 300-400 words.  
**Rhythm:** `T-M`.

**Table 2.1:** dataset, controlled phenomenon, models evaluated, metrics, evidence role, and
limitations. Include 2-D rotations, 3-D polycubes, OpenSTL Moving-MNIST, TaxiBJ, LoopMom, and
persistent long Moving-MNIST.  
**Figure 2.1:** visual dataset ladder from simple dynamics to explicit long-memory demand. Include
the controlled rotation family, OpenSTL datasets, LoopMom, and persistent long Moving-MNIST without
implying that every method was tested on every dataset.

### 5.4.2 Reversible temporal modelling

#### 2.2.1 Activation-memory problem in long temporal chains

**Target:** 550-650 words.  
**Rhythm:** `S-F-L-S`.

Formalize the activation tape and distinguish parameters, optimizer state, recurrent state, and
local computation graphs.

**Equations:** BPTT memory and computational cost.  
**Figure 2.4:** standard forward/backward tape.

#### 2.2.2 Reversible transformer block

**Target:** 750-900 words.  
**Rhythm:** `M-F-F-V-S`.

Present additive coupling, forward equations, inverse equations, assumptions, and numerical
considerations.

**Figure 2.5:** published reversible-block figure, unchanged.  
**Equations:** complete forward and inverse coupling.
**Table 2.2:** verified reversible-state contract: tensor, operation, invertible status, retained
boundary value, replay requirement, and inclusion in the activation-memory claim.

#### 2.2.3 Temporal chaining and TempVerseFormer

**Target:** 750-900 words.  
**Rhythm:** `S-V-L-F-M`.

Explain how the reversible transition is reused through time and how the spatial encoder/decoder
interfaces with it. Before calling the chain reversible, define the persistent temporal state and
reconcile the paper's cloned streams and dimensional fusion with the implementation's actual split,
feedback tensor, loss placement, and retained boundary values.

**Figures 2.6-2.7:** published complete TempVerseFormer architecture and temporal-chaining figure,
unchanged.
The tall detailed architecture may occupy a full page.  
**Equations:** temporal recurrence and prediction map.

#### 2.2.4 Time-agnostic backpropagation

**Target:** 900-1,100 words.  
**Rhythm:** `M-V-F-L-F-S`.

Describe tape-free forward execution, reverse state reconstruction, local graph rebuilding, and
gradient accumulation.

**Figure 2.8:** published time-agnostic-backpropagation figure, unchanged. A separate
dissertation-native companion is added only if it explains a verified implementation detail that
the published figure does not.  
**Equations:** gradient recursion and memory-complexity lemma.  
**Algorithm table or pseudocode:** one compact algorithm, not both a long table and repeated prose.

#### 2.2.5 Controlled temporal-pattern results

**Target:** 950-1,150 words.  
**Rhythm:** `V-T-M-S-V-L-S`.

Report constant, accelerated, decelerated, oscillating, interrupted, and combined rotations. Make
the mixed result visible: TempVerseFormer is strongest in some regimes, while LSTM or TempFormer
wins other metrics. Separate temporal chaining's apparent stability from any claim that
reversibility itself improves predictive quality.

**Figures 2.9-2.11 and 2.16:** unchanged published constant-pattern qualitative comparison,
constant-pattern FID curve, 11-step extrapolation examples, and combined-pattern FID curve.  
**Conditional Figures 2.12-2.15:** unchanged accelerated, decelerated, oscillating, and interrupted
prediction galleries when page balance permits; otherwise move all four together to an appendix.  
**Table 2.3:** one model-by-dynamics matrix preserving the best reported FID, MSE, and SSIM for each
regime. The original six detailed tables may remain in an appendix.

#### 2.2.6 External OpenSTL validation

**Target:** 700-850 words.  
**Rhythm:** `S-T-V-L-S`.

Report Moving MNIST, Moving MNIST-CIFAR, KTH Action, and TaxiBJ as external competence evidence.
Use "competitive among the reported methods" and name the exact best result only where it occurs.
State once that the paper reports no multi-seed uncertainty, confidence intervals, or qualitative
selection rule.

**Figure 2.19:** four-source OpenSTL gallery using the unchanged published Figures 16-19. Each
source image remains intact and separately attributed; use a landscape spread if necessary.  
**Table 2.4:** exact published OpenSTL MSE/SSIM comparisons and dataset-specific winner.

#### 2.2.7 Activation-memory and computation frontier

**Target:** 600-800 words.  
**Rhythm:** `V-T-M-V-X`.

Report temporal-model and total-pipeline memory separately, followed immediately by training time.
The principal conclusion is a memory-computation exchange, not a free saving. Preserve the
qualification on any 1,024-step points until their measurement/extrapolation provenance is
confirmed.

**Figures 2.17-2.18:** unchanged published memory-versus-horizon and
training-time-versus-horizon graphs.  
**Table 2.5:** exact 1-, 4-, and 16-step memory values plus the matching runtime measurement where
available.

#### 2.2.8 Dissertation extension to stochastic reversible recurrence

**Target:** 750-950 words.  
**Rhythm:** `S-F-L-F-M-S`.

Only after the deterministic published method and results are complete, present store-the-noise
conditioning as a separately measured dissertation extension. Define the replay metadata, inverse,
reconstruction tolerance, gradient-direction validation, and metadata cost. Do not attribute this
extension to the published article.

**Equations:** stochastic transition, inverse, metadata cost.  
**Figure:** dissertation-native noise-conditioned forward/reverse path may be placed in the
appendix unless it materially clarifies the main applicability decision.

#### 2.2.9 Applicability relative to BPTT and checkpointing

**Target:** 600-800 words.  
**Rhythm:** `T-L-S-M`.

Explain where reversibility pays and where recomputation or checkpointing is preferable. The paper
does not provide a direct checkpointing experiment, so identify checkpointing as a theoretical and
practical comparator unless separate dissertation evidence is cited. Label the inner-loop negative
result as a later dissertation experiment, not TempVerseFormer article evidence.

**Table 2.6:** intervention selection by binding resource, including memory scope, recomputation,
metadata, and evidence type.  
**Figure:** applicability plane moves to the system-selection figure in Chapter 4 to avoid a
premature second conclusion.

### 5.4.3 Hierarchical spatial representation and entropy estimation

#### 2.3.1 Fixed-patch allocation problem

**Target:** 450-600 words.  
**Rhythm:** `S-V-L-S`.

Move from the Chapter 1 problem to the V-BLT response. Distinguish its local-global
representational hierarchy from Chapter 3's multirate retained-context hierarchy. The reported
experiment uses fixed `4 x 4 x 4` patches.

**Figure 2.22:** fixed-patch tensor-shape and local-to-global processing flow. Any variable-size
branch is visually separated and labelled proposed only.

#### 2.3.2 V-BLT architecture

**Target:** 800-950 words.  
**Rhythm:** `M-V-L-F-S`.

Describe the local encoder, global latent transformer, and local decoder over uncompressed raster
pixel-channel bytes. Define tensor sizes and distinguish hidden latent compression from the
categorical byte output interface.

**Figure 2.20:** published V-BLT architecture, unchanged. Do not use the current dissertation
schematic's "entropy-guided variable patches" label as a description of the measured model.  
**Equations:** tensor flattening and local/global/local mapping.  
**Table 2.7:** component, input/output, function, implemented status, and evidence level.

#### 2.3.3 Causal attention and categorical byte objective

**Target:** 550-750 words.  
**Rhythm:** `V-F-L-F-S`.

Define full attention within an observed frame, attention to permitted past positions, masked
future positions, 256-way byte logits, categorical cross-entropy, and channel handling.

**Figure 2.21:** unchanged published causal attention-mask figure.  
**Equations:** attention mask, categorical distribution, and pixel-channel cross-entropy.

#### 2.3.4 Multidimensional rotary positional encoding

**Target:** 800-1,000 words.  
**Rhythm:** `S-F-F-L-V-S`.

Derive RoPE-ND from one-dimensional RoPE, define axis allocation, and explain relative-coordinate
dependence. Retain mathematical detail.

**Figure 2.23:** dissertation-native RoPE-ND axis-subspace schematic, verified against the actual
implementation. The source paper contains equations, not a published RoPE-ND figure.  
**Equations:** complex rotation, axis-specific frequencies, concatenated rotated subspaces,
relative attention phase. Reconstruct the derivation with disjoint axis subspaces and complete
frequency-index domains; do not copy the paper's internally ambiguous indexing.

#### 2.3.5 Predictive entropy as a spatial-complexity signal

**Target:** 600-800 words.  
**Rhythm:** `M-F-L-S-V`.

Define per-byte entropy and explain what it measures. Distinguish predictive dispersion from
calibrated aleatoric or epistemic uncertainty.

**Equation:** Shannon entropy of the categorical byte prediction.  
**Figure 2.24:** new categorical-logit-to-entropy-to-pixel-map pipeline, including the documented
channel aggregation.  
**Figure 2.25:** the unchanged published 2-D and 3-D prediction/entropy images placed as two intact
source panels with precise row descriptions. No calibration or representativeness claim is added.

#### 2.3.6 Experimental results on 2-D and 3-D rotations

**Target:** 700-850 words.  
**Rhythm:** `T-V-M-V-S`.

Report computational comparisons, quality metrics, and visual entropy results. Discuss where the
evidence is strongest and where the first low-context predictions fail.

**Figure 2.26:** derived two-panel point-estimate comparison: model size/GFLOPs and quality
metrics. Do not add error bars. Verify MSE normalization, PSNR aggregation, sample counts, and
parameter precision before using it as a headline graph.  
**Table 2.8:** exact published model size and GFLOP values.  
**Table 2.9:** exact published MSE, SSIM, and PSNR values. Keep the 3-D ViViT PSNR win visible and
do not claim that V-BLT wins every metric.

Add one explicit experimental-completeness paragraph: the paper reports no seed uncertainty,
statistical inference, real-video validation, component ablation, scaling sweep, entropy
calibration, or declared qualitative selection rule.

#### 2.3.7 Measured method versus proposed dynamic regioning

**Target:** 350-500 words.  
**Rhythm:** `S-M-V-X`.

State the boundary plainly:

- static spatio-temporal patches were used in the reported experiment;
- entropy maps were measured;
- a variable-size allocation policy is a justified design proposal;
- its end-to-end compute/quality effect was not measured.

**Figure 2.27:** two-column "implemented/evaluated" and "proposed future extension" diagram.  
**Table:** none.

### 5.4.4 Design implications for the final hierarchical-context model

#### 2.4.1 Which results transfer conceptually

**Target:** 350-500 words.  
**Rhythm:** `M-S-L`.

Transfer only the general lessons: compute should follow information demand; temporal training and
retained context are different memory problems; coordinate systems matter when token geometry
changes.

#### 2.4.2 Which components are not executed in Chapter 3

**Target:** 200-300 words.  
**Rhythm:** `T-S`.

**Table 2.10:** method, bottleneck, validated result, possible interface, and integration status.
This is the single body-text integration-status table. Do not repeat it later.

#### 2.4.3 Chapter conclusions and transition

**Target:** 450-600 words.  
**Rhythm:** `M-S-M-X`.

Conclude that Chapter 2 provides two independent resource-aware methods. The final sentence should
introduce the unresolved retained-context problem without claiming architectural inheritance.

## 5.5 Chapter 3 — Hierarchical Context Model for Long-Horizon Video Prediction

**Target:** 11,950-15,500 words, 38-50 pages.  
**Artifacts:** 16 core figures plus one conditional persistent-object schematic, 5-8 tables, and
24-40 displayed equations.

### 5.5.1 Retained context as the final model problem

#### 3.1.1 From consistency requirements to memory reach

**Target:** 450-600 words.  
**Rhythm:** `S-L-V-S`.

Explain why a correct objective and feasible training do not decide which historical observations
remain accessible at inference.

**Figure 3.1:** a re-entry sequence in which the relevant appearance leaves the dense window.

#### 3.1.2 Context-package design space

**Target:** 550-700 words.  
**Rhythm:** `M-F-V-L-S`.

Define a package as streams with resolution level, stride, count, and anchor geometry.

**Equations:** package definition, token budget, effective reach.  
**Figure 3.2:** flat dense, spatial hierarchy, and temporal subsampling shown with identical visual
grammar.  
**Table 3.1:** package fields and semantics.

#### 3.1.3 Reach-fidelity hypothesis

**Target:** 400-550 words.  
**Rhythm:** `S-M-F-S`.

Define the expected mechanism without presenting results. Reach determines whether relevant
history is present; fidelity determines what information survives once it is present.

**Equation:** conceptual risk decomposition or two-factor notation if it is used consistently.  
**Figure 3.3:** reach and fidelity as separate axes, with no fitted crossing.

### 5.5.2 Hierarchical latent state

#### 3.2.1 Two-level frozen autoencoder

**Target:** 650-800 words.  
**Rhythm:** `M-V-F-L-S`.

Define fine and coarse codes, spatial dimensions, independent decodability, and the role of the
frozen interface.

**Equations:** encoder and decoder maps, latent shapes.  
**Figure 3.4:** two-level encoder/decoder with exact tensor dimensions.  
**Table 3.2:** fine/coarse resolution, channels, tokens per frame, payload.

#### 3.2.2 Stream construction and embeddings

**Target:** 600-750 words.  
**Rhythm:** `S-F-V-M`.

Describe token projection, level/role/frame embeddings, context and target roles, and ordering.

**Equation:** token sequence construction.  
**Figure 3.5:** token assembly before joint attention.

#### 3.2.3 Joint-attention context transfer

**Target:** 750-900 words.  
**Rhythm:** `M-V-F-L-S`.

Explain the implemented joint-attention model, the causal mask, and why the superseded macro-first
description must not reappear.

**Equation:** joint-attention update and mask.  
**Figure 3.6:** one transformer block with context, noisy targets, and conditioning.

#### 3.2.4 Gradient boundaries and information asymmetry

**Target:** 450-600 words.  
**Rhythm:** `S-F-M-S`.

Explain frozen codes, detached boundaries, trainable predictor components, and what receives
gradient.

**Figure 3.7:** gradient-flow diagram using solid and blocked arrows.

### 5.5.3 Generative prediction core

#### 3.3.1 Windowed latent-diffusion objective

**Target:** 750-900 words.  
**Rhythm:** `M-F-F-L-S`.

Define the noisy target, timestep conditioning, v-parameterization, and loss.

**Equations:** forward diffusion, v target, training objective.
**Figure 3.8:** regression, diffusion, and energy-objective interfaces, used here only to define the
mechanism choices later evaluated in Chapter 4.

#### 3.3.2 Transformer backbone and conditioning

**Target:** 450-650 words.  
**Rhythm:** `S-V-L-M`.

Describe the WindowDiT backbone, adaptive normalization, action/control conditioning, and shared
capacity across package comparisons.

**Figure 3.9:** complete predictor architecture from latent package to predicted latent window.

#### 3.3.3 DDIM sampling and inference work

**Target:** 500-650 words.  
**Rhythm:** `M-F-L-S`.

Specify deterministic sampling, number of function evaluations, candidates, and work accounting.

**Equations:** DDIM update and inference-work count.
**Figure 3.10:** deterministic DDIM denoising timeline with conditioning inputs and function
evaluation count.

#### 3.3.4 Regression control head

**Target:** 300-400 words.  
**Rhythm:** `S-F-M`.

Present it as a controlled alternative head and explain the co-change in objective and inference.

### 5.5.4 Autoregressive rollout and memory update

#### 3.4.1 Ring-buffer recurrence

**Target:** 600-750 words.  
**Rhythm:** `M-F-V-L-S`.

Define fine and coarse buffers, update rules, and self-fed replacement.

**Equation:** buffer recurrence.  
**Figure 3.11:** ring-buffer update over three rollout windows.

#### 3.4.2 Anchor geometry and causal semantics

**Target:** 550-700 words.  
**Rhythm:** `S-F-V-M`.

Define stream anchors independently of hard-coded stride assumptions and show which historical
frames each token represents.

**Figure 3.12:** anchor timeline with fine, coarse, and target slots.  
**Equation:** anchor-time mapping and reach.

#### 3.4.3 Context, target, and total cost

**Target:** 500-650 words.  
**Rhythm:** `F-T-M-S`.

Keep context tokens, target tokens, total sequence length, latent payload, and attention cost
separate.

**Table 3.3:** exact token and byte arithmetic for principal packages.  
**Figure 3.13:** token-budget bars at equal nominal reach.

### 5.5.5 Evaluation instruments and estimands

#### 3.5.1 Why short benchmarks cannot identify the reach mechanism

**Target:** 400-550 words.  
**Rhythm:** `S-L-M`.

Use standard Moving-MNIST as the motivating counterexample without dismissing its competence role.

#### 3.5.2 LoopMom generator and re-entry geometry

**Target:** 750-950 words.  
**Rhythm:** `M-V-F-L-V-S`.

Describe the toroidal world, viewport, actions, loop period, disappearance, and re-entry.

**Equations:** world/viewport dynamics and coverage set.  
**Figures 3.14-3.15:** world/viewport schematic and multi-frame re-entry storyboard.

#### 3.5.3 Exact observability and pair-specific reach populations

**Target:** 650-850 words.  
**Rhythm:** `S-F-L-V-S`.

Define geometric observability, unobserved foreground, and the population whose relevant evidence
lies between two package reaches.

**Equations:** unobserved fraction and pair-specific band.  
**Figure 3.16:** shaded reach-differential population on a timeline.

#### 3.5.4 Complete frame and object metrics

**Target:** 750-900 words.  
**Rhythm:** `M-F-T-L-S`.

Define full-frame, foreground, background, union-mask, precision/recall, miss/extra, overlap,
coverage, identity, and trajectory metrics. Explain their blind spots.

**Table 3.4:** endpoint, estimand population, direction, empty-case rule, and failure detected.

#### 3.5.5 Persistent-object and benchmark boundaries

**Target:** 450-650 words.  
**Rhythm:** `V-M-T-S`.

Introduce long Moving-MNIST, canonical OpenSTL Moving-MNIST, and TaxiBJ with their distinct roles.

**Conditional Figure 3.17:** persistent-object assignment and identity-tracking schematic.  
**Table 3.5:** context/prediction length, task property, memory demand, and evidentiary role.

#### 3.5.6 Statistical protocol

**Target:** 600-800 words.  
**Rhythm:** `S-F-L-F-S`.

Specify clip-cluster resampling, paired effects, training seeds as a separate uncertainty level,
registered comparison families, and simultaneous maximum-statistic intervals.

**Equations:** paired clip estimand, bootstrap statistic, simultaneous interval.  
Do not add a generic bootstrap cartoon. The equations and endpoint table should carry this section
unless a final reader test shows that the nested clip/seed uncertainty levels remain unclear.

### 5.5.6 Research hypotheses and chapter conclusions

#### 3.6.1 Registered hypotheses and expected directions

**Target:** 400-550 words.  
**Rhythm:** `M-S-M`.

State reach, fidelity, package-shape, robustness, and self-fed reliability hypotheses before the
results chapter.

#### 3.6.2 Chapter conclusions

**Target:** 450-650 words.  
**Rhythm:** `M-S-L-X`.

Summarize the implemented model and evaluation design. Do not report experimental results here.

## 5.6 Chapter 4 — Experimental Validation and Operating-Region Decisions

**Target:** 10,650-13,650 words, 35-47 pages.  
**Artifacts:** 16 core figures plus one conditional optimization-stage figure, 7-10 main-text
tables, and 6-12 displayed equations.

### 5.6.1 Experimental map and comparison logic

#### 4.1.1 Research questions, datasets, and evidence levels

**Target:** 450-600 words.  
**Rhythm:** `S-T-M-S`.

**Table 4.1:** one compact map from research question to intervention, population, seeds, and
terminal role.  
**Table 4.2:** executed component/configuration matrix modelled on the useful CW-VAE comparison
style: codec, context package, coarse/fine counts, temporal stride, prediction head, target
interface, sampler/NFE, dataset, seeds, and evidence level. Include every scientifically relevant
executed combination and visibly mark non-executed cells; run IDs and secondary endpoints remain
in the appendix.  
**Figure 4.1:** experiment flow from mechanism selection to confirmation, robustness, and boundary
tests.

#### 4.1.2 Budget vocabulary and comparison rules

**Target:** 400-500 words.  
**Rhythm:** `M-F-S-M`.

Define matched, near-matched, and whole-configuration comparisons. Do not reproduce the full arm
registry in the main chapter.

**Equation:** effect direction convention.  
**Appendix pointer:** complete configuration and evidence registry.

### 5.6.2 Selection of the generation mechanism

#### 4.2.1 Regression and diffusion under controlled capacity

**Target:** 550-650 words.  
**Rhythm:** `T-M-V-S`.

Report single-step best-of-(k) and best-of-1 rollout outcomes as different operating regimes.

**Table 4.3:** concise mechanism comparison.  
**Figures 4.2-4.3:** objective/inference comparison and the identical single-step qualitative
rows for regression and diffusion.

#### 4.2.2 Energy-based descent as a characterized negative

**Target:** 400-500 words.  
**Rhythm:** `S-V-M-X`.

Explain landscape ordering and reachability without carrying historical recipe detail into the
main text.

**Figure 4.4:** compact EBT panel combining the corrected candidate-update loop, ordering, and
descent sharpness. Do not reuse the current loop if its feedback arrow still returns to context
rather than candidate state.  
**Conditional Figure 4.5:** optimization-stage memory-gate effects after its annotation and source
values are reverified.  
**Appendix:** full diagnostic receipts.

#### 4.2.3 Mechanism decision

**Target:** 150-200 words.  
**Rhythm:** `S-X`.

State why diffusion is retained for stochastic coverage and regression remains a deterministic
control. End without another general mechanism review.

### 5.6.3 Hierarchical reach under fixed context budget

#### 4.3.1 Untouched reach-differential confirmation

**Target:** 550-650 words.  
**Rhythm:** `S-V-F-M-X`.

Lead with the two untouched seed effects and simultaneous intervals.

**Figure 4.6:** per-seed reach-band effects with intervals.  
**Equation:** pair-specific paired effect if not already sufficient in Chapter 3.

#### 4.3.2 Whole-population and endpoint support

**Target:** 400-550 words.  
**Rhythm:** `T-M-S`.

Show supporting endpoints without allowing aggregate MSE to become the headline.

**Table 4.4:** compact complete-metric support table.
**Figure 4.9:** selection-rule-controlled qualitative rollout strip, including late degradation in
both arms rather than a visually dramatic anecdote.

#### 4.3.3 Interpretation of reach

**Target:** 250-350 words.  
**Rhythm:** `M-X`.

Explain why concentration in the pair-derived band supports the proposed mechanism and what it
does not prove.

### 5.6.4 Reach-fidelity frontier and memory efficiency

#### 4.4.1 Simultaneous tested frontier

**Target:** 550-650 words.  
**Rhythm:** `S-V-L-S`.

Report hierarchy at 896/1,152, unresolved 1,792, and flat at 2,432/3,072.

**Figure 4.7:** current confirmed frontier, regenerated only if its visual language must match the new
dissertation style. Scientific values and unresolved status remain frozen.

#### 4.4.2 Exact token and payload arithmetic

**Target:** 400-500 words.  
**Rhythm:** `T-F-M-X`.

**Table 4.5:** 48 fine, 48 coarse + 6 fine, and 48 coarse + 2 fine.  
**Figure 4.8:** simple token-count bars or icon grid.  
**Equation:** relative token reduction.

#### 4.4.3 Reach versus fidelity as an operating-region interpretation

**Target:** 400-500 words.  
**Rhythm:** `M-S-L-X`.

Interpret sampled regions without inventing a continuous crossing. Use one concrete example of a
task that values identity and one that values support.

### 5.6.5 Context-package shape and model interaction

#### 4.5.1 Spatial hierarchy versus temporal subsampling

**Target:** 450-600 words.  
**Rhythm:** `T-V-M-S`.

Report the near-matched package-shape comparison and explain why spatial and temporal compression
are not interchangeable.

**Table 4.7:** context-form evidence map covering reach, fixed-span spatial abstraction,
temporal-versus-spatial packaging, the stride-2 family, and the qualified depth probe. State what
each comparison isolates and what remains confounded.

#### 4.5.2 Prediction-head interaction

**Target:** 400-500 words.  
**Rhythm:** `V-M-S`.

Show that the diffusion direction is more stable than the regression direction across the tested
factorial.

**Figure 4.10:** head-by-package interaction small multiples.  
**Table 4.6:** full five-seed diffusion effect, regression effect, and head-by-package interaction.
This factorial stays in the main chapter; only raw endpoint cells move to the appendix.

### 5.6.6 Robustness of the principal directions

#### 4.6.1 Sampler allocation and randomness

**Target:** 400-500 words.  
**Rhythm:** `V-M-X`.

Focus on whether direction persists and where precision weakens.

#### 4.6.2 Independently trained codec

**Target:** 400-500 words.  
**Rhythm:** `S-V-M-S`.

Report codec-directional robustness and the unresolved upper inversion.

**Figures 4.11-4.12:** sampler/NFE robustness and independently trained codec robustness. Keep
them separate because they intervene at different system interfaces.  
**Table 4.8:** direction/status matrix across seeds, samplers, and codec.

### 5.6.7 Boundary and negative evidence

#### 4.7.1 Failed self-fed trust-boundary replication

**Target:** 400-550 words.  
**Rhythm:** `S-V-L-X`.

Report the replicated early benefit and failed late inversion. Do not turn the failure into a new
theory.

**Figure 4.13:** failed self-fed-boundary replication, with verdict text outside all intervals.

#### 4.7.2 Privileged raw-memory reference

**Target:** 350-450 words.  
**Rhythm:** `V-M-S`.

Use the mosaic to distinguish storage/addressability from learned reuse. Keep it visually separate
from matched-budget neural comparisons.

**Figure 4.14:** privileged pose-indexed raw-memory reference.

#### 4.7.3 Persistent-object support-identity boundary

**Target:** 600-750 words.  
**Rhythm:** `T-V-L-S`.

Report both relative effects and absolute task difficulty.

**Table 4.9:** absolute metrics for both systems plus paired effects.  
**Figure 4.15:** two-panel support/continuity versus identity summary.

#### 4.7.4 Short-horizon and real-data competence boundaries

**Target:** 400-500 words.  
**Rhythm:** `T-M-X`.

Present canonical Moving-MNIST and TaxiBJ as competence or negative-control evidence, not hierarchy
confirmation or leaderboard claims.

### 5.6.8 System-level method-selection rule

#### 4.8.1 Diagnose the binding constraint

**Target:** 400-550 words.  
**Rhythm:** `S-V-L-S`.

Define a practical sequence: establish task criterion; determine whether history is missing;
determine whether fine detail is limiting; determine whether activation storage blocks training;
then choose an intervention.

**Figure 4.16:** two-resource synthesis with context-state quality/reach and activation-memory
scaling on separate axes. Do not normalize them into a joint saving or imply multiplication.  
**Figure 4.17:** final decision flow with separate branches for context hierarchy, dense context,
checkpointing/reversibility, and entropy-informed spatial allocation. The entropy branch must say
“candidate adaptive policy; end-to-end dynamic benefit not yet measured.”

#### 4.8.2 Package applicability matrix

**Target:** 300-400 words.  
**Rhythm:** `T-M`.

**Table 4.10:** dense flat, spatial hierarchy, temporal subsampling, adaptive spatial
representation, checkpointing, and reversible temporal replay. Compare controlled resource,
benefit, cost, evidence level, and suitable regime.

#### 4.8.3 Relationship among the developed methods

**Target:** 200-300 words.  
**Rhythm:** `M-X`.

One paragraph explains that the methods occupy different interfaces. One short paragraph states
that joint execution remains future work. Do not repeat the Chapter 2 integration-status table.

### 5.6.9 Software toolkit and practical application

#### 4.9.1 Software architecture

**Target:** 400-500 words.  
**Rhythm:** `V-M-S`.

Describe supported modules, reproducibility controls, and public interfaces.

**Table:** optional short mapping from module to purpose. The existing software/data/evidence
pipeline moves to an appendix so the main figure programme remains scientific rather than
administrative.

#### 4.9.2 Implementation and educational use

**Target:** 250-350 words.  
**Rhythm:** `M-S`.

Report only documented implementation acts and their actual covered methods.

### 5.6.10 Limitations and threats to validity

#### 4.10.1 Internal and construct validity

**Target:** 350-450 words.  
**Rhythm:** `S-L-M`.

Address co-changing interfaces, metric blind spots, synthetic priors, and comparison licences.

#### 4.10.2 Statistical and external validity

**Target:** 350-450 words.  
**Rhythm:** `M-L-S`.

Address seed coverage, codec coverage, task diversity, natural-video limits, and generalization.

#### 4.10.3 Chapter conclusions

**Target:** 500-650 words.  
**Rhythm:** `M-S-L-X`.

Conclude with the operating regions and practical decision rule. Detailed numerical receipts stay
in the preceding sections and appendices.

## 5.7 General Conclusions

**Artifact rule:** plain text preferred; no figure or table. A displayed equation is unnecessary
unless the institution explicitly expects the central formulation to be repeated.

**Target:** 1,500-2,200 words, approximately 4-7 pages.

### 5.7.1 Opening synthesis

**Target:** 160-240 words.  
**Rhythm:** `M-S`.

State the solved systems problem and the fact that the methods address distinct bottlenecks. Do
not repeat the full scope defence.

### 5.7.2 Six task-aligned conclusions

Use six numbered conclusions with unequal lengths:

1. **System decomposition and consistency criteria:** 180-260 words.
2. **Reversible temporal computation:** 220-320 words.
3. **V-BLT, RoPE-ND, and entropy estimation:** 220-320 words; explicitly preserve the static versus
   dynamic boundary.
4. **Hierarchical context model:** 220-320 words.
5. **Evaluation methodology:** 180-260 words.
6. **Operating regions and final empirical results:** 280-400 words.

Each conclusion should use a different internal shape. For example, one may lead with a method,
another with a measured result, and another with a limitation that changes the interpretation.

### 5.7.3 Practical significance

**Target:** 160-240 words.  
**Rhythm:** `M-S`.

### 5.7.4 Limitations and future work

**Target:** 220-320 words.  
**Rhythm:** `S-L-X`.

Prioritize natural-video validation, genuinely dynamic entropy regioning, and an optional
multi-window training objective if future integration is pursued.

## 5.8 Appendices

The appendices should contain:

1. applicant publications and personal-contribution mapping;
2. implementation acts and administrative documents;
3. complete package/configuration registry;
4. complete seed-by-seed and endpoint tables;
5. supplementary derivations and proofs;
6. energy-based negative-result details;
7. sampler and codec robustness receipts;
8. experiment authorization and provenance records;
9. software commands, schemas, and reproduction information;
10. supplementary qualitative sequences selected by a declared rule;
11. the four unchanged variable-rotation TempVerseFormer galleries if they do not fit Chapter 2;
12. detailed published rotation-regime tables and the stochastic reversible extension;
13. optional BLT/ViViT literature figures only after original-source permission checks;
14. figure provenance, licence/permission, extraction, hash, and modification-status ledger;
15. administrative software/data/evidence pipeline and full API mapping.

## 6. Detailed fourth-article structure

The article remains a focused context-packaging study. It must not become a summary of the entire
dissertation.

**Target:** 6,500-8,000 words for abstract through conclusion, plus references and a 2,000-4,000
word supplement where the venue permits.  
**Artifacts:** 7-10 main figures, 5-8 main tables, detailed registries in supplement.

The dissertation's minimum visual programme does not apply to the article. Do not reuse
TempVerseFormer, V-BLT, or 2024 consistency-paper figures in this focused manuscript. Its visual
density is controlled by the context-packaging argument and the venue format.

### 6.1 Abstract

**Target:** 230-290 words.  
**Rhythm:** one problem sentence; two method sentences; three deliberately unequal result
paragraphs or blocks; one scope/design-rule ending.

No dissertation, publication-strand, EBM, V-BLT, or reversibility discussion.

### 6.2 Introduction

#### 1.1 Fixed-budget memory problem

**Target:** 450-600 words.  
**Rhythm:** `S-L-M-S`.

#### 1.2 Reach, fidelity, and evaluation failure modes

**Target:** 450-600 words.  
**Rhythm:** `M-L-S-M`.

#### 1.3 Research questions and predictions

**Target:** 300-450 words.  
**Rhythm:** short setup plus RQ paragraphs of unequal length.

#### 1.4 Contributions and positioning

**Target:** 700-950 words.  
**Rhythm:** `S-L-T-M-S`.

Use “this work,” not “the fourth article.” Keep the closest prior work and protocol-positioning
table. Remove the sentence about other dissertation strands.

### 6.3 Controlled design

#### 2.1 LoopMom and exact geometric observability

**Target:** 600-800 words.  
**Artifacts:** generator schematic and re-entry timeline; 2-4 equations.

#### 2.2 Context packages and budgets

**Target:** 650-850 words.  
**Artifacts:** package timeline, cost table, package and budget equations.

#### 2.3 Predictor and rollout

**Target:** 550-750 words.  
**Artifacts:** one compact architecture figure; diffusion objective and rollout recurrence.

#### 2.4 Development and protected populations

**Target:** 350-500 words.  
**Artifacts:** one population table. Detailed access chronology moves to the supplement.

### 6.4 Endpoints and inference

#### 3.1 Complete frame and object metrics

**Target:** 500-700 words.  
**Artifacts:** compact endpoint table.

#### 3.2 Pair-specific reach estimand

**Target:** 350-500 words.  
**Artifacts:** reach-band equation and explanatory timeline.

#### 3.3 Cluster-aware simultaneous inference

**Target:** 400-550 words.  
**Artifacts:** estimand and maximum-statistic equations.

### 6.5 Results

#### 4.1 Untouched confirmation and frontier

**Target:** 650-850 words.  
**Artifacts:** primary frontier figure and compact table.

#### 4.2 Context-package shape

**Target:** 450-600 words.  
**Artifacts:** package-shape and head-interaction figure.

#### 4.3 Sampler and codec robustness

**Target:** 500-700 words.  
**Artifacts:** combined robustness figure and direction table.

#### 4.4 Failed self-fed-boundary replication

**Target:** 350-500 words.  
**Artifacts:** negative-replication figure.

#### 4.5 Raw-access and persistent-object boundaries

**Target:** 650-850 words.  
**Artifacts:** raw-reference figure, support-identity figure, absolute metric table.

Development evidence and the B1 whole-configuration result may remain concise here or move to the
supplement, depending on venue length.

### 6.6 Discussion

#### 5.1 Reach-fidelity design rule

**Target:** 400-550 words.  
**Rhythm:** `S-L-X`.

#### 5.2 Relationship to prior hierarchical-memory work

**Target:** 400-550 words.  
**Rhythm:** `M-L-S`.

#### 5.3 Limitations and external validity

**Target:** 400-550 words.  
**Rhythm:** `S-L-M-X`.

### 6.7 Reproducibility, availability, and conclusion

Keep reproducibility operational and concise. End the conclusion with the conditional design rule,
not a universal hierarchy claim.

### 6.8 Article supplement

Move or retain here:

- development evidence;
- full B1 description;
- complete package/interface registry;
- full endpoint tables;
- sampler and codec secondary results;
- evidence/provenance details;
- supplementary qualitative sequences.

## 7. Section-transition rules

Every major transition must answer why the next section follows.

Preferred transition logic:

- **Introduction to Chapter 1:** the problem requires formal system and consistency definitions.
- **Chapter 1 to Chapter 2:** two identified bottlenecks concern training and spatial
  representation.
- **Within Chapter 2:** after making long temporal differentiation feasible, the analysis turns
  to where spatial capacity is spent. These are sibling methods, not dependent stages.
- **Chapter 2 to Chapter 3:** neither method decides which past observations remain available at
  inference; Chapter 3 isolates retained context.
- **Chapter 3 to Chapter 4:** the model creates explicit competing packages, so their operating
  regions can now be measured.
- **Chapter 4 to General Conclusions:** experimental results are converted into method-selection
  rules and task-aligned conclusions.

Avoid generic transitions such as “The next section will discuss...” when the scientific reason
can be stated directly.

## 8. Later execution sequence

This section records the future implementation order. It does not authorize new experiments.

1. Freeze this blueprint after author review.
2. Extract the selected published figures, record original figure/page/source/licence, and hash-lock
   every unchanged asset.
3. Resolve the TempVerseFormer reversible-state contract against the implementation: split versus
   clone, persistent state, projections/fusion, feedback tensor, loss placement, boundary values,
   replay metadata, and numerical tolerance.
4. Audit the V-BLT implementation and published tables: fixed patch size, causal mask, RoPE-ND
   indexing, channel aggregation for entropy, MSE/PSNR aggregation, model-size precision, and
   GFLOP calculations.
5. Create a section-level claim matrix from the frozen evidence and published articles.
6. Correct V-BLT static/dynamic wording before broader prose rewriting.
7. Rewrite the English annotation as plain text.
8. Rewrite the Introduction as plain text and remove the crosswalk and systems figure.
9. Rebuild Chapter 1 around the system problem, consistency criteria, and three bottlenecks.
10. Reorder Chapter 2: common testbeds, reversibility, V-BLT/entropy, measured-versus-proposed
   boundary, short design implications.
11. Rebuild Chapter 3 as the central model chapter and expand its explanatory visual sequence.
12. Refactor Chapter 4 into research questions and operating-region decisions; retain the complete
   scientifically relevant component-comparison tables while moving run-level registries and raw
   endpoint cells to appendices.
13. Rewrite General Conclusions from the six tasks, not from the article list.
14. Clean the fourth article of dissertation-specific wording and move secondary material to its
    supplement.
15. Create or revise only non-published diagrams. Keep all reused published figures unchanged.
16. Select and combine figures by argumentative value, then remove any visual that merely repeats
    the surrounding prose or causes the 200-page English ceiling to be exceeded.
17. Expand and bind the literature by claim, prioritizing primary sources.
18. Run a scientific consistency audit across annotation, Introduction, chapter conclusions,
    novelty statements, and General Conclusions.
19. Run a prose audit for repeated sentence openings, repeated disclaimer language, uniform
    paragraph size, filler transitions, and unnecessary triplets.
20. Render the complete manuscript and test every figure at final size, greyscale, and a common
    color-vision simulation; inspect page flow so visuals do not break the argument.
21. Reassemble Markdown and only then regenerate review PDFs/DOCX if requested.

## 9. Acceptance checklist

### Narrative

- [ ] The dissertation reads as one research argument, not four article summaries.
- [ ] The hierarchical-context model is visibly the empirical core.
- [ ] Earlier methods are presented as sibling interventions at different system interfaces.
- [ ] The body rarely uses article labels or publication chronology.
- [ ] Each chapter begins from the unresolved question left by the previous chapter.
- [ ] Scope disclaimers appear once where needed instead of being repeated throughout.

### Scientific scope

- [ ] V-BLT static patching is distinguished from proposed dynamic entropy regioning.
- [ ] The reversible temporal state boundary is verified; no non-invertible VAE/projection interface
      is silently included in an exact-reversibility claim.
- [ ] TempVerseFormer memory savings are paired with their computation premium and mixed quality
      results.
- [ ] V-BLT uses the verified fixed-patch implementation and "raster pixel-channel byte" wording.
- [ ] Predictive entropy is not called calibrated uncertainty or a validated dynamic allocator.
- [ ] The 2024 taxonomy is identified as a static-image conceptual source extended to video.
- [ ] OpenSTL is not falsely described as shared published evidence from the earlier model papers.
- [ ] Reversibility is not claimed as implemented in the final predictor.
- [ ] The 1,792-token point remains unresolved.
- [ ] Hierarchy and flat-context results remain conditional on tested operating regions.
- [ ] External synthetic boundaries are not presented as natural-video replication.

### Writing

- [ ] Annotation and Introduction contain no figures, tables, or displayed equations.
- [ ] Paragraph length varies deliberately without becoming erratic.
- [ ] No three consecutive sentences use the same opening pattern.
- [ ] Filler phrases and repeated “not a/does not” constructions are reduced.
- [ ] Each equation is motivated and interpreted.
- [ ] Figures are discussed for their scientific relationship, not described label by label.
- [ ] Tables are interpreted by pattern rather than read cell by cell.
- [ ] Authorial judgments are supported and technically restrained.
- [ ] The author has reviewed every rewritten section for meaning and voice.

### Visuals and document design

- [ ] New schematics use white backgrounds and consistent visual grammar.
- [ ] Arrows and cycles are unambiguous.
- [ ] Chapter 3 has a complete visual sequence from package definition to rollout and evaluation.
- [ ] Published TempVerseFormer and V-BLT figures are unchanged.
- [ ] Selected 2024 consistency-paper figures are unchanged.
- [ ] Every main-text figure performs a distinct scientific or explanatory job; no minimum or
      maximum count is enforced, and the rendered English edition remains at or below 200 pages.
- [ ] Every reused figure has source, original figure number, permissions status, and hash receipt.
- [ ] Derived charts contain no invented uncertainty and identify their numeric source.
- [ ] Dense published figures are given full or landscape width instead of being simplified.
- [ ] Main-text tables are decision-relevant; audit and registry tables are in appendices.
- [ ] Chapter 4 contains the executed component/configuration matrix, the five-seed head-package
      factorial, and the context-form evidence map.
- [ ] Captions state the inference or relationship shown.

### Article

- [ ] The article contains no dissertation-strand discussion.
- [ ] The untouched frontier appears before development history.
- [ ] Reach, fidelity, and self-fed reliability remain distinct.
- [ ] Secondary configuration and provenance material is in the supplement where possible.
- [ ] The article closes with a conditional design rule.

## 10. Requirements traceability against the earlier proposed structure

| Earlier recommendation | Current blueprint status |
|---|---|
| Introduce three bottlenecks without article labels | Implemented in Introduction prose plan and Chapter 1 formalization. |
| Six research tasks | Preserved in Introduction and General Conclusions. |
| Four or five calibrated novelty claims | Preserved; dynamic allocation remains future work. |
| One Introduction progression figure | Deliberately overridden by the author's later instruction: Introduction is plain text; the systems map is now Figure 1.21. |
| Chapter 1 organised by system, consistency, failures, literature, and decomposition | Implemented and expanded with the static-to-video bridge. |
| Reversibility before spatial representation in Chapter 2 | Implemented. |
| Detailed TempVerseFormer architecture retained | Implemented as unchanged full/landscape Figure 2.6. |
| Explicit fixed-patch versus dynamic-regioning boundary | Implemented in 2.3.7 and Figure 2.27. |
| Chapter 3 as dissertation centre | Implemented with 16-17 planned figures and full cost/evaluation formalism. |
| Chapter 4 led by untouched evidence and operating regions | Implemented. |
| CW-VAE-style component comparison tables | Implemented through Tables 4.2, 4.6, and 4.7. |
| Article remains narrow | Implemented; earlier article figures and dissertation synthesis stay out. |
| Six task-aligned General Conclusions | Implemented. |
| High but purposeful formula density | Implemented through chapter-specific equation budgets and derivations. |
| More visual explanation | Preserved through a purposeful candidate pool, with overlapping panels combined or omitted when required by the hard page ceiling. |

## 11. Final writing principle

Do not try to make the prose sound human by exchanging words for synonyms. Change the reasoning
shape. Begin some paragraphs with evidence, others with a problem or a limitation. Allow a short
sentence to carry an important conclusion. Let complex explanations take more space when they
need it, and stop when the point has been made.

Accuracy remains the first constraint. Natural rhythm follows from clear decisions about what the
author wants the reader to understand next.
