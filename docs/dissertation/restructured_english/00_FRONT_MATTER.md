# Front Matter

## Title-page information

**Approved topic in Ukrainian:** «Системний аналіз та інтелектуальні технології консистентного моделювання процесів ієрархічної генерації відеоданих».

**English title:** *Systems Analysis and Intelligent Technologies for Consistent Modelling of Hierarchical Video-Data Generation Processes*.

**Applicant:** Leonid Ruslanovych Kulyk.  
**Degree field:** 12 Information Technologies.  
**Specialty:** 124 System Analysis.  
**Academic supervisor:** Oleksandr Borysovych Mokin, Doctor of Technical Sciences, Professor.  
**Institution:** Vinnytsia National Technical University, Vinnytsia, 2026.

**UDC:** 004.054:[004.032.26+004.85] **[AUTHOR INPUT: verify the classification and every title-page field against the approved institutional record.]**

## English annotation

Kulyk L. R. *Systems Analysis and Intelligent Technologies for Consistent Modelling of Hierarchical Video-Data Generation Processes*. Qualifying scientific work submitted as a manuscript for the degree of Doctor of Philosophy in specialty 124 System Analysis, field 12 Information Technologies.

The dissertation addresses long-horizon video prediction as a resource-constrained systems problem. A useful prediction must do more than resemble the next frame. It may need to preserve an object's appearance after temporary disappearance, continue a plausible trajectory, avoid introducing unsupported foreground, and remain stable when its own outputs become later inputs. Those requirements become harder with horizon because the predictor observes only a viewport of an evolving world and must decide what history to retain. Training introduces a different difficulty: differentiating through a long temporal computation stores intermediate activations even when the inference state itself is compact. Spatial representation adds a third constraint, since a uniform grid assigns equal capacity to regions whose information content may differ greatly.

Taken together, these difficulties are often compressed into one informal notion of “memory”. That description is too coarse for method selection. Activation storage during training, spatial allocation inside a representation, and addressable context during inference are different resources; reducing one does not automatically reduce the others. Existing video models frequently improve overall fidelity without isolating which resource binds, while evaluations based on average frame error can overlook object loss, false foreground, identity substitution, or failures concentrated exactly where distant evidence is required. The unresolved task is therefore not simply to build a larger generator. It is to formulate the interfaces, budgets, and consistency criteria so that a claimed improvement can be attributed to a specific mechanism.

Accordingly, the work aims to improve task-relevant consistency and resource efficiency in long-horizon hierarchical video prediction through systems analysis and separately validated interventions for temporal training, spatial representation, retained context, and evaluation. Generated video is first treated as the output of a partially observed dynamic system. Appearance, identity, geometry, semantic content, temporal continuity, trajectory correctness, and cross-level agreement are modelled as overlapping criteria rather than substitutes for one another. This vocabulary extends an earlier qualitative analysis of consistency control in static diffusion-based image generation. The earlier taxonomy supplies criteria and control interfaces without constituting temporal evidence.

Activation memory is addressed first. TempVerseFormer applies additive reversible coupling inside a temporal transformer chain so that intermediate transformer states can be reconstructed during the backward pass instead of being retained as a conventional activation tape. Inspection of the implementation fixes the exact scope of that statement. The persistent doubled transformer state is reversible; variational encoding and decoding, learned input projection, output fusion, and the loss interface lie outside the reversible boundary. On the reported controlled rotation task, the temporal-model memory remains approximately 0.89 GB at the one-, four-, and sixteen-step measurements, whereas matched non-reversible transformer variants rise to approximately 2.60–2.61 GB at sixteen steps. The long-horizon profile remains near 1 GB through the reported 1,024-step range. This saving has a cost: reverse reconstruction increases computation, and predictive quality is competitive but not best for every motion regime or external OpenSTL dataset. Reversibility is consequently an option when activation storage binds, not a general explanation for better prediction.

Spatial representation poses the next question. Vision Byte Latent Transformer processes raster pixel-channel bytes through a local encoder, a global latent transformer, and a local decoder, with a 256-way categorical objective for each predicted byte. N-dimensional rotary positional encoding preserves separate temporal and spatial coordinates after flattening. The published proof of concept uses fixed 4 × 4 × 4 spatio-temporal packages on procedural 2-D rotations and 3-D polycubes; it reports lower computation than the listed ViViT and UNet-3D comparators and strong, although not uniformly best, image metrics. Categorical predictive entropy is visualized as a per-byte spatial diagnostic and is concentrated around visually changing regions in the selected examples. No adaptive allocation loop was evaluated. Dynamic entropy-driven regioning remains a proposed extension, and the available default source configuration records a different temporal patch length from the publication.

Neither intervention determines which observations should remain available during inference. The dissertation therefore makes hierarchical retained context the central empirical problem. A frozen two-level latent codec supplies coarse and fine spatial streams to one joint-attention diffusion predictor. Competing packages are defined by resolution, count, temporal stride, and stream-anchor geometry; their context, target, payload, and total-token costs are accounted separately. Autoregressive rollouts update coarse and fine ring buffers with generated states. TempVerseFormer, V-BLT, and RoPE-ND remain outside the final predictor: they occupy different interfaces in the same systems methodology and were evaluated separately.

A purpose-built LoopMom environment isolates memory-dependent prediction. The underlying toroidal scene continues outside a moving viewport, so the exact frames that contain relevant evidence—and the interval during which that evidence is no longer directly observable—can be calculated analytically. Evaluation includes complete frame and foreground endpoints, explicit miss and extra-foreground measures, persistent-object assignment, identity and trajectory criteria, paired clip-level effects, and simultaneous maximum-statistic intervals for registered comparison families. Training seeds form a separate uncertainty level from correlated frames within a clip. Untouched, purpose-locked, and protected populations are distinguished from development evidence, and negative results remain part of the terminal record.

The principal result is conditional but clear. On the untouched reach-differential population, hierarchical context improves the registered endpoint by 0.0980 and 0.0914 for two training seeds when the matched flat fine-resolution window cannot include the relevant past. Across the simultaneously analysed tested frontier, hierarchy is confirmed at 896 and 1,152 context tokens, the 1,792-token comparison is unresolved, and flat fine-resolution context is confirmed at 2,432 and 3,072 tokens. At equal nominal 48-frame reach, a dense fine history uses 3,072 context tokens. Forty-eight coarse frames combined with six or two fine frames use 1,152 or 896 tokens, respectively—2.67 and 3.43 times fewer. These findings establish reach efficiency and a fixed-budget advantage in the tested reach-limited region. Equal information content, universal superiority, and a continuously fitted Pareto crossing remain outside the evidence.

Robustness studies preserve the principal directions under alternative sampler allocations and an independently trained codec, although some upper-region intervals lose precision. A registered self-fed study replicates an early hierarchical benefit across three seeds but fails to replicate the late inversion first seen at seed zero. A privileged pose-indexed raw-memory reference distinguishes storing information from learning to use it. Finally, a protected persistent-object long Moving-MNIST study exposes a useful boundary: hierarchical context better preserves spatial support and trajectory continuity, while dense flat context better preserves digit identity. Absolute identity, coverage, and trajectory performance also show that neither system solves the external task.

Scientific novelty lies in the combination of a system-level resource decomposition with mechanism-specific evaluation: trajectory-level consistency criteria; a bounded reversible temporal-state contract and its measured memory–computation frontier; a hierarchical raster-byte representation with multidimensional positional structure and entropy-based complexity estimation; a matched-budget hierarchical context model with a measured reach–fidelity frontier; and an exact-observability, paired clip-cluster protocol for effects that depend on historical reach. The novelty is methodological and empirical; priority over hierarchical video models, reversible networks, or adaptive tokenization as broad classes is outside the claim.

Practical outputs include controlled video generators, validated configuration schemas, fail-closed experiment authorization, immutable evidence artifacts, complete endpoint implementations, figure and table generation from frozen results, and a public research-software interface. Together, the tools support reproducible investigation and teaching; deployment in safety-critical or natural-video applications has not been validated. A direct design rule follows: compress retained state when distant reach is missing, preserve dense detail when fidelity is limiting, and reconstruct activations when the training tape—not inference context—is the dominant cost.

**Keywords:** systems analysis; long-horizon video prediction; hierarchical context; consistency; reach–fidelity trade-off; reversible computation; activation memory; raster-byte modelling; predictive entropy; latent diffusion; exact observability; cluster-aware inference.

## Applicant publications

- **[A1]** Kulyk L. R., Mokin O. B. “Scaling Video Prediction with Spatio-Temporal Patches.” *Visnyk of Vinnytsia Politechnical Institute*, 2025, No. 5, pp. 129–139. DOI: 10.31649/1997-9266-2025-182-5-129-139.
- **[A2]** Kulyk L. “Temporal Modeling with Reversible Transformers.” *Neural Computing and Applications*, 2026, Vol. 38, article 30. DOI: 10.1007/s00521-025-11707-3. Sole author; Scopus-indexed.
- **[A3]** Kulyk L. R., Mokin O. B. “Methods for Ensuring Consistent Generation in Diffusion Models.” *Visnyk of Vinnytsia Politechnical Institute*, 2024, No. 4, pp. 75–85. DOI: 10.31649/1997-9266-2024-175-4-75-85.
- **[A4]** Kulyk L. R. “When Does Hierarchical Video Memory Pay? Reach–Fidelity Trade-offs under Fixed Context Budgets.” **[AUTHOR INPUT: insert target venue and terminal status.]**
- **Conference publications:** **[AUTHOR INPUT: insert the six verified bibliographic entries and confirm whether DSMSI-2023 belongs in the list.]**

## Symbols and abbreviations

| Symbol or abbreviation | Meaning |
|---|---|
| AR | autoregressive |
| BPTT | backpropagation through time |
| CI | confidence interval |
| CRN | common random numbers |
| DDIM | denoising diffusion implicit model |
| DDPM | denoising diffusion probabilistic model |
| DiT | diffusion transformer |
| EBM / EBT | energy-based model / energy-based transformer |
| EMA | exponential moving average |
| FID / FVD | Fréchet inception distance / Fréchet video distance |
| fg-MSE | foreground mean squared error |
| LPIPS | learned perceptual image patch similarity |
| MSE | mean squared error |
| NFE | number of function evaluations |
| RoPE-ND | N-dimensional rotary positional embedding |
| SSIM | structural similarity index measure |
| V-BLT | Vision Byte Latent Transformer |
| VAE | variational autoencoder |

## Contents

The final institutional version should generate a live table of contents after pagination. Its scientific body comprises the Introduction; Chapter 1, *Systems Analysis and Consistency Requirements of Long-Horizon Video Prediction*; Chapter 2, *Methods for Feasible Temporal Training and Adaptive Spatial Representation*; Chapter 3, *Hierarchical Context Model for Long-Horizon Video Prediction*; Chapter 4, *Experimental Validation and Operating-Region Decisions*; General Conclusions; References; and Appendices.
