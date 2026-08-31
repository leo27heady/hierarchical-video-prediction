# Articles 1--3 content and figure transfer ledger

## Transfer policy

Published text is rewritten into a unified dissertation voice and cited to its publication. Published author-owned figures may be reused only with preserved internal content, attribution, DOI/license information, and provenance. Cropping, external margins, and dissertation captions may change; the plotted or drawn content must not be silently redrawn. New dissertation synthesis is labeled as such.

Status values:

- `KEEP`: transfer into the new main text.
- `SELECT`: choose the listed subset, not the whole current gallery.
- `APPENDIX`: retain for provenance or detail outside the main narrative.
- `REGENERATE`: create from new v2 experiments; current version is pilot/background only.
- `BOUNDARY`: transfer the limitation as well as the positive result.

## Article 1 -- consistency and generative-control review

Source: Kulyk and Mokin (2024), DOI `10.31649/1997-9266-2024-175-4-75-85`; official source PDF `docs/figures/restructured/sources/Kulyk_Mokin_2024_official.pdf`.

| Material | Current asset/source | Destination | Status | Transfer note |
|---|---|---|---|---|
| consistency vocabulary and task interfaces | current Chapter 1 and publication [133] | 1.1.2, 1.3 | KEEP | compress around failure modes relevant to hierarchy/memory |
| visual perception concept | `published_2024_fig5_visual_perception.png` | 1.1.2 or appendix | SELECT | use only if it adds information beyond the taxonomy |
| diffusion pipeline | `published_2024_fig4_diffusion_pipeline.png` | 1.3 or appendix | SELECT | one generative-model illustration is sufficient |
| image-to-image conditioning | `published_2024_fig6_img2img.png` | appendix | APPENDIX | not central to the hierarchy claim |
| ControlNet/geometry conditioning | `published_2024_fig7_controlnet.*` | appendix | APPENDIX | historical/interface context only |
| inversion-based control | `published_2024_fig8_inversion.png` | appendix | APPENDIX | historical/interface context only |
| style/attention sharing | `published_2024_fig9_stylealigned.jpg` | appendix | APPENDIX | historical context; omit from main flow unless directly used |
| SDXL examples and video-prediction composite | `published_2024_fig1_sdxl.png` plus current Chapter 1 composite | 1.1 or appendix | SELECT | prefer the new system-boundary/taxonomy figures in main text |
| forward/reverse diffusion illustrations | `published_2024_fig2_forward_diffusion.png`, `published_2024_fig3_reverse_diffusion.png` | appendix | APPENDIX | standard method; cite rather than spend two main figures |

Article 1 contributes the literature/system context to Chapter 1. It is not experimental support for the Article 4 hierarchy frontier. Chapter 1 must also cite Article 4/New Chapter 2 when stating the measured hierarchy result, so the conceptual chapter is based on both the review and the dissertation's central experiment.

## Article 2 -- TempVerseFormer

Source: *Temporal Modeling with Reversible Transformers*, Neural Computing and Applications 38, article 30 (2026), DOI `10.1007/s00521-025-11707-3`; public implementation `https://github.com/leo27heady/TempVerseFormer`.

| Material | Current asset/source | Destination | Status | Transfer note |
|---|---|---|---|---|
| reversible additive-coupling block | `ch2_fig2_5_tempverse_reversible_block_published.png` | 3.2.1 | KEEP | preserve published pixels; group forward/inverse equations |
| full architecture | `ch2_fig2_6_tempverse_architecture_published.png` | 3.2.1 | KEEP | explain exact reversible boundary and non-reversible VAE parts |
| temporal chaining | `ch2_fig2_7_tempverse_chaining_published.png` | 3.2.2 | KEEP | retain if not duplicating replay figure |
| time-agnostic/reversible backward | `ch2_fig2_8_tempverse_time_agnostic_published.png` | 3.2.2 | SELECT | choose chaining or backward as primary; other to appendix |
| controlled motion examples/curves | `ch2_fig2_9...` through `ch2_fig2_16...` | 3.2.3 and appendix | SELECT | keep one representative quality summary; move pattern gallery to appendix |
| OpenSTL sample/results panels | `ch2_fig2_17a...17d...` and published tables | 3.2.3 | SELECT | retain representative result figure and empirical table; disclose historical protocol |
| memory versus horizon | `ch2_fig2_18_tempverse_memory_published.png` | 3.2.3 | KEEP | exact numeric claims only for tabulated points; far-curve coordinates are unavailable |
| runtime versus horizon | `ch2_fig2_19_tempverse_time_published.png` | 3.2.3 | KEEP | show compute cost next to, not combined with, memory |
| paper/code loss difference | current equations 2.14--2.15 and provenance | 3.2.2 or appendix | BOUNDARY | retain the distinction explicitly; do not merge inconsistent definitions |
| stochastic reversible extensions | current Chapter 2 extension | appendix or 4.2.4 if executed | BOUNDARY | main text only if used and tested by Article 5 |

TempVerseFormer supports activation-memory reconstruction and published external feasibility. It does not prove that the complete VAE path is reversible, that reversibility improves quality, or that the current Article 5 long-rollout wrapper preserves the same temporal dynamics.

## Article 3 -- V-BLT and RoPE-ND

Source: Kulyk and Mokin, *Scaling Video Prediction with Spatio-Temporal Patches* (2025), DOI `10.31649/1997-9266-2025-182-5-129-139`; implementations `https://github.com/leo27heady/visionBLT` and `https://github.com/leo27heady/simple-shape-dataset-toolbox`.

| Material | Current asset/source | Destination | Status | Transfer note |
|---|---|---|---|---|
| 2-D/3-D controlled sequences | `ch2_fig2_1_vblt_2d_sequence_published.png`, `ch2_fig2_2_vblt_3d_sequence_published.png` | 3.3.4 or appendix | SELECT | one compact composite is enough in main text |
| local--global--local tensor flow | `ch2_fig2_20_fixed_patch_flow.png` | 3.3.1 | SELECT | use as explanatory redraw if the published architecture is too dense |
| published architecture | `ch2_fig2_21_vblt_architecture_published.png` | 3.3.1 | KEEP | preserve pixels and attribution |
| causal mask | `ch2_fig2_22_vblt_causal_mask_published.png` | 3.3.2 or appendix | SELECT | retain only if it clarifies time/space causality beyond prose |
| RoPE-ND | publication/code and current formulas 2.35--2.39 | 3.3.2 and 4.2.2 | KEEP | compact to one multi-axis equation; adapt concept, not unverified results, for dynamic routing |
| entropy pipeline | `ch2_fig2_24_entropy_pipeline.png` | 3.3.3 | KEEP | make clear that entropy is diagnostic, not calibrated uncertainty |
| 2-D/3-D entropy maps | `ch2_fig2_25a...`, `ch2_fig2_25b...` | 3.3.3 | KEEP | published qualitative result; preserve panels |
| point-estimate visualization | `ch2_fig2_26_vblt_point_estimates.png` | 3.3.4 | SELECT | empirical tables may be clearer; avoid duplication |
| implemented versus proposed dynamic partition | `ch2_fig2_27_measured_vs_proposed.png` | 3.3.4 or 4.2.3 | BOUNDARY | fixed V-BLT is measured; adaptive routing is a new Article 5 hypothesis |
| published 2-D/3-D quality and cost tables | current tables 2.9--2.10 | 3.3.4 | KEEP | preserve mixed ranking and point-estimate limitations |

V-BLT supplies a fixed hierarchical local/global representation, RoPE-ND, and an entropy diagnostic. It does not establish calibrated uncertainty or a memory/quality win from dynamic region allocation.

## Current Article 4 / hierarchy material

The current hierarchy article and Chapters 3--4 are sources for architecture language, TRB mechanism, metric definitions, provenance controls, and pilot findings. The new Article 4 results are regenerated under the v2 protocol.

| Material | Destination | Status | Transfer note |
|---|---|---|---|
| hierarchy architecture and invariant interfaces | 2.1--2.2 | KEEP/REWRITE | preserve the scientifically executed core; align names and target interface |
| TRB/legacy LoopMom generator | 2.3 | KEEP | preserve generator identity and legacy hashes; use TRB for new artifacts |
| observability and pair-specific estimands | 2.3.2--2.3.3 | KEEP | retain rigorous claim boundaries; simplify equations |
| current budget frontier and E2/E4/E5a/E6/E7a/E7c/E8a | 2.5, appendix | BOUNDARY | label as prior/pilot unless a v2 run exactly satisfies the new protocol |
| current combined result figures | appendix or provenance | REGENERATE | new main figures have one question each and show seed-level replication |
| current integrated prototype | Chapter 4 background/appendix | BOUNDARY | feasibility only; not the final integrated architecture |

## Migration order

1. Draft Chapter 1 with selected Article 1 material and the new hierarchy/memory synthesis.
2. Draft Chapter 3's published-method sections from Articles 2 and 3 while Article 4 experiments run.
3. Draft Chapter 2 methods/protocol immediately, but insert results only from locked v2 artifacts.
4. Draft Chapter 4 architecture after A5 correctness gates pass; insert comparative claims only after OpenSTL confirmation.
5. Generate a final figure/equation/table inventory and compare it with the target budget before assembling the manuscript.
