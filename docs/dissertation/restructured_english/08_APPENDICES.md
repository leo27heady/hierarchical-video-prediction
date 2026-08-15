# Appendices

The appendices preserve audit material that would interrupt the main scientific argument if it
were repeated there. They do not enlarge the scope of the reported experiments. Where an item
depends on an institutional record, permission decision, or final venue status, the required
author input is stated explicitly rather than inferred from the repository.

## Appendix A — Publications, dissertation results, and personal contribution

The labels in this appendix follow the publication list in the front matter. They are not the
order in which the scientific body is organized.

| Publication | Dissertation role | Results carried into the dissertation | Applicant contribution | Qualification retained |
|---|---|---|---|---|
| **[A1]** Kulyk and Mokin, *Scaling Video Prediction with Spatio-Temporal Patches* | Chapter 2 spatial-representation method | fixed `4 x 4 x 4` raster-byte packages, local/global/local V-BLT processing, RoPE-ND, categorical prediction, entropy maps, published cost and quality point estimates | method and software development, procedural experiments, quantitative and qualitative analysis, manuscript preparation; supervisor contribution to systems framing and review | dynamic entropy-driven regioning was proposed but not executed; the available default source configuration does not independently establish the published temporal patch length |
| **[A2]** Kulyk, *Temporal Modeling with Reversible Transformers* | Chapter 2 activation-memory method | reversible transformer core, temporal chaining, controlled rotation results, OpenSTL comparisons, memory and runtime evidence | sole-author design, implementation, experiments, analysis, and manuscript | exact reconstruction applies to the persistent doubled transformer state; encoder/decoder, learned projections, fusion, and loss lie outside the reversible boundary; quality is mixed and replay costs computation |
| **[A3]** Kulyk and Mokin, *Methods for Ensuring Consistent Generation in Diffusion Models* | Chapter 1 conceptual starting point | appearance/style, identity, geometry/shape, and conceptual consistency; comparison of input-, weight-, structure-, inversion-, and attention-based controls | technical review, synthesis, and proposal of the conceptual-consistency category; supervisor contribution to scientific framing and editorial review | qualitative static-image review, not a systematic review or temporal experiment; the dissertation extends its criteria to video instead of treating them as ready-made temporal evidence |
| **[A4]** Kulyk, *When Does Hierarchical Video Memory Pay?* | Chapters 3–4 empirical core | explicit context packages, LoopMom observability, complete endpoints, clip-cluster inference, tested reach–fidelity frontier, robustness, negative replication, and persistent-object boundary | model, generators, evaluation and statistical software, experiment design and execution, artifact curation, analysis, and manuscript | **[AUTHOR INPUT: verify the final authorship, venue, bibliographic status, and contribution statement before submission.]** |

Dissertation-only work includes the cross-method systems synthesis, the implementation-boundary
audit of TempVerseFormer and V-BLT, the explicit distinction between context/state and activation
memory, the stochastic reversible-replay extension where identified in Chapter 2, and the
publication-independent operating-region decision rule. These additions must not be attributed
retroactively to the published papers.

## Appendix B — Administrative records and implementation acts

The final institutional version must insert verified records for the two state-budget research
projects, the applicant's role in each project, six conference publications, and the aggregate
publication statement. It must also reproduce the signed implementation acts rather than
paraphrasing them from memory.

| Required record | Repository status | Required author action |
|---|---|---|
| UDC and title-page wording | provisional value in the front matter | verify against the approved VNTU record |
| State-budget project 1 | no authoritative number or signed wording in the repository | insert title, number, dates, and applicant role |
| State-budget project 2 | no authoritative number or signed wording in the repository | insert title, number, dates, and applicant role |
| Conference publications | incomplete administrative list | supply all six entries and confirm whether DSMSI-2023 belongs |
| Article [A4] status | manuscript complete; venue field open | insert final venue, author list, and status |
| ORNETIKA LLC implementation act | date and scientific scope described in Chapter 4 | insert the signed scan and assigned appendix letter; preserve the act's exact scope |
| VNTU educational implementation act | date and scientific scope described in Chapter 4 | insert the signed scan and assigned appendix letter; preserve the act's exact scope |
| Final document statistics | unavailable before institutional typesetting | insert total and main-text pages, figure/table counts, reference count, and appendix letters |
| Reused-figure licences | author ownership declaration and source licences recorded | no further permission action for the retained author-created CC BY figures; preserve attribution in final typesetting |

## Appendix C — Complete configuration and evidence registry

The complete arm-level registry is maintained in
[`APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md`](../rewrite/appendices/APPENDIX_CH4_CONFIGURATION_AND_EVIDENCE_REGISTRY.md).
It is part of the review package and should be appended without changing its numerical cells when
the institutional document is assembled. The registry contains:

1. comparison licences and prohibited interpretations;
2. the observed LoopMom package grid and structurally absent cells;
3. exact context, target, total-token, payload, optimization, and available hardware receipts;
4. the untouched simultaneous frontier and pair-derived reach effects;
5. the five-seed head-by-package factorial;
6. package-shape, depth, and stream-blinding diagnostics;
7. sampler/NFE and independently trained codec families;
8. the failed self-fed replication, explicit-access reference, persistent-object boundary, and
   short-horizon competence studies;
9. artifact paths, access records, and terminal programme states.

The registry is normative for run identities and secondary numerical receipts. Tables in Chapter
4 are curated views of it. A discrepancy must be resolved by the frozen JSON/NPZ artifact and its
manifest, not by selecting whichever prose value is more convenient.

## Appendix D — Evidence and provenance index

| Evidence role | Human-readable record | Machine-readable or access record |
|---|---|---|
| Untouched E2 confirmation | `docs/confirmation/E2_RESULTS.md` | `artifacts/confirmation_e2/` and its dataset manifest |
| E5a registered replication | `docs/confirmation/E5A_RESULTS.md` | `artifacts/e5a_trust_replication/results.json` and completion validation |
| E6 persistent-object boundary | `docs/confirmation/E6_PHASEC_V2_RESULTS.md` | protected E6 result artifacts referenced by the release manifest |
| E7a codec robustness | `docs/confirmation/E7A_RESULTS.md` | experiment manifest, completion validation, and frozen result artifacts |
| E7c sampler/NFE robustness | `docs/confirmation/E7C_RESULTS.md` | `artifacts/e7c_sampler_nfe/results.json`, access log, and completion validation |
| E8a explicit-access reference | `docs/confirmation/E8A_RESULTS.md` | experiment manifest and frozen reference artifacts |
| Programme closure | `docs/FINAL_PROGRAMME_REPORT_2026-07-18.md` | `artifacts/release_manifest_2026-07-19.json` and `experiments/authorization.json` |
| Release checks | `docs/RELEASE_QA_REPORT_2026-07-18.md` | source revision and file hashes in the release manifest |

Development, untouched confirmation, registered replication, robustness, privileged reference,
and external boundary are different evidence roles. Their artifacts may be stored in one release,
but their inferential licences do not merge.

## Appendix E — Statistical reporting contract

Every paired analysis retains the clip as the resampling unit. Frames within a sampled clip remain
together, and training seeds are never replaced by a larger number of rollout frames. The minimum
report for an inferential contrast consists of: candidate and comparator; comparison licence;
population filter; clip and record counts; endpoint and favorable direction; point estimate;
interval type; registered family; training seed; sampler allocation; codec identity; and terminal
decision. Empty-mask rules and assignment denominators travel with the endpoint definition.

The maximum-statistic families are frozen separately for the untouched frontier, sampler study,
codec study, and persistent-object study. An interval containing zero is reported as unresolved,
not equivalent. A seed-level summary uses seeds as its units and is visibly distinguished from a
clip-cluster interval conditional on one trained model.

## Appendix F — Supplementary method and negative-result material

The following material remains available for a full institutional appendix or article
supplement:

- the six detailed controlled-rotation metric tables and the four unchanged variable-motion
  TempVerseFormer galleries;
- the verified reversible-state contract, numerical reconstruction tolerance, gradient-direction
  tests, stochastic replay metadata, and the distinction between measured and extrapolated memory
  points;
- exact V-BLT tensor, mask, RoPE-ND, entropy-channel aggregation, published-table, and
  source-configuration audit receipts;
- the complete energy-based transformer ordering and descent logs, including the corrected
  candidate-state update diagram;
- secondary sampler, codec, foreground, component, and raw-access endpoint tables;
- additional qualitative sequences chosen by fixed indices or a declared aggregate-quantile rule.

No supplementary gallery may be reduced to only the motion pattern or seed that looks best. A
derived chart must name its numeric source and must not add uncertainty that the source did not
report.

## Appendix G — Software and reproduction map

The public `hvp` namespace defines stable context, evaluation, provenance, and validation
interfaces. The frozen `hvp_rev` namespace retains the evidence-producing implementations and
compatibility paths. `experiments/` contains authorization and scientific configurations;
`artifacts/` contains immutable evidence; `scripts/` contains controlled runners and figure/table
generators. Local artifacts, rather than an experiment-tracking service, are the source of truth.

The operational reproduction guide is
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md), and the more detailed supplementary instructions
are in [`REPRODUCIBILITY_APPENDIX.md`](../../supplement/REPRODUCIBILITY_APPENDIX.md). Before a
public archival release, a clean-clone check must additionally verify downloadable dataset and
checkpoint URLs, expected sizes, hashes, licences, and third-party notices. A local passing test
suite cannot substitute for public retrievability.

Three public repositories preserve the earlier-method reproduction routes:

| Component | Public source | Reproduction role |
|---|---|---|
| Procedural 2-D/3-D rotation data | [simple-shape-dataset-toolbox](https://github.com/leo27heady/simple-shape-dataset-toolbox) [171] | generators, metadata, and command-line dataset creation |
| Reversible temporal modelling | [TempVerseFormer](https://github.com/leo27heady/TempVerseFormer) [172] | model code, train/evaluate/memory-test configurations, checkpoints, and experiment links |
| Byte-level visual modelling | [visionBLT](https://github.com/leo27heady/visionBLT) [173] | V-BLT implementation and executable configuration surface |

Repository availability and historical identity are recorded separately. A new run is reproducible
when its commit, environment, data manifest, configuration, seeds, and checkpoint hashes are fixed.
That run need not be described as byte-identical to a published run whose exact historical manifest
was not archived. The distinction protects the published receipt while giving a concrete route for
independent execution.

## Appendix H — Figure provenance and permissions

The machine-readable figure records are split between the
[Chapter 1 extraction and generation ledger](../../figures/restructured/figure_provenance.json),
the [Chapter 2–4 extraction and generation ledger](../../figures/restructured/ch2_ch4_figure_provenance.json),
and the [pre-existing repository-figure ledger](../../figures/restructured/existing_figure_provenance.json).
All three use the same versioned record contract: figure and file identity, asset class, byte size,
SHA-256 digest, raster or vector dimensions, modification and rights status, permission status,
caption attribution, and scientific role. Every reused published figure additionally records the
source paper, original figure number and page,
extraction method, SHA-256 hash, pixel dimensions, modification status, caption attribution, and
permission status. Author-generated figures retained from earlier dissertation passes are also
hash-locked and linked to their generator and numerical or conceptual source.
An extracted source panel retains the published appearance. Direct image objects remain
byte-identical where the PDF permits direct extraction; figures with a soft mask are reconstructed
losslessly into PNG without cropping or redrawing. Placement, surrounding whitespace, and the
dissertation caption may change; labels, content, and evidentiary meaning may not.

![Historical published illustration of reverse diffusion](../../figures/restructured/published_2024_fig3_reverse_diffusion.png)

**Appendix Figure H.1 — Historical source illustration of reverse diffusion.** The published drawing communicates progressive denoising but uses simplified $q$-notation. In the dissertation's formal specification, the learned generative transition is written $p_\theta(x_{t-1}\mid x_t)$, distinct from forward-process and posterior $q$-distributions. *Source: Kulyk and Mokin [133], original Figure 3; author-created artwork reused unchanged under CC BY 4.0. Conceptual illustration, not the authoritative mathematical definition.*

The provenance tooling recomputes file hashes, byte sizes, and dimensions, checks each
linked image against exactly one ledger record and numbered caption, verifies generator hashes,
and compares unchanged published assets with the supplied source image object or author-source
file. The retained published artwork is author-created and is covered by the recorded CC BY 4.0
source licences or the author's source-artwork declaration. The two LoopMom qualitative composites
are also author-created; their MNIST-derived glyphs are dataset content rather than copied publisher
artwork, and the Moving-MNIST citation plus local data manifests record that origin. No separate
publisher-figure permission remains open for these assets.

Dissertation-native schematics are generated in both high-resolution PNG and vector PDF where
practical. Their source notes distinguish conceptual diagrams from artifact-derived plots. The
final typeset manuscript should test them at actual page width, in greyscale, and under a common
color-vision simulation before deposit.

## Appendix I — Terminal open items

The scientific programme is closed for the claims made in the dissertation. E9 is recorded as
`SKIPPED_POST_FREEZE_NO_INCREMENTAL_UTILITY`; the B1 fixed-target control remains unexecuted and is
not needed while B1 is described permanently as a whole-configuration comparison. No additional
GPU experiment is required.

Open work is authorial or institutional: confirm the fields in Appendix B, translate the approved
English text if required, apply the VNTU template, insert signed
records, update live cross-references, and perform page-by-page render inspection. These steps can
change presentation and administrative completeness. They must not change the frozen experiment
verdicts or broaden the claim scope.
