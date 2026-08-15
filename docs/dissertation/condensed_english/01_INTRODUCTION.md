# Introduction

## Relevance and justification of the research topic

A video predictor is often judged one frame at a time, yet many of its consequential errors are visible only after a longer interval. Camera motion can reveal a region that was present earlier while the model fills it with plausible but unsupported texture. These are not merely reductions in image sharpness. They show that the generated process has failed to preserve information that the task treats as invariant.

Prior research provides strong components but rarely isolates the resulting mechanisms under compatible budgets. Checkpointing and reversible networks trade storage for recomputation [30–32, 92, 108–113]. Hierarchical latent models, recurrent state, compressed memory, retrieval, sparse attention, and state-space models extend temporal access in different ways [14–24, 33–37, 79–83]. Patch-based visual transformers and byte-latent models alter the unit of computation [69–78]. Their reported gains are informative, but comparisons often change architecture, token count, objective, target interface, or evaluation population simultaneously. A better score then identifies a configuration, not necessarily the reason for its advantage.

That separation leads to a controlled systems methodology. It should define consistency as a vector of task criteria, account separately for training and inference resources, make context packages comparable, and construct populations in which the proposed mechanism can actually be identified. The final empirical model in this dissertation is designed for that purpose. Earlier methods address activation growth and spatial representation independently; they explain how two other resource constraints can be relieved while remaining outside the final predictor.

## Relationship to research programmes, plans, and topics

Work on the dissertation was carried out at Vinnytsia National Technical University within the scientific direction of systems analysis and intelligent information technologies. **[AUTHOR INPUT: insert the official title, number, dates, and applicant role for state-budget project 1.]** **[AUTHOR INPUT: insert the corresponding verified information for state-budget project 2.]** The final wording must follow the signed institutional reports; project names or responsibilities are not inferred from software history.

## Aim and research tasks

The aim of the dissertation is to improve task-relevant consistency and resource efficiency in long-horizon hierarchical video prediction through systems analysis, separately addressing temporal training, spatial representation, retained context, and evaluation.

Achieving this aim requires more than combining existing modules. Each intervention must be tied to the resource it changes, and its evidence must distinguish an actual mechanism from a coincident architectural difference. For this reason, the work moves from formal criteria to two enabling methods and then to a central matched-budget context study. The sequence is conceptual rather than a chronology of publications.

Accordingly, six research tasks are set:

1. Analyse long-horizon video prediction as a partially observed dynamic system; classify relevant uncertainty and consistency failures; and derive requirements for causal, resource-accounted comparison.
2. Develop and delimit reversible temporal computation for long-horizon training, verify its reconstructable state and backward procedure, and characterize the resulting activation-memory and computation trade-off.
3. Develop a hierarchical raster-byte representation, multidimensional rotary positional encoding, and entropy-based estimation of spatial predictive complexity, while separating the evaluated fixed-patch method from proposed dynamic allocation.
4. Develop a hierarchical long-context video predictor whose coarse and fine streams, anchors, token costs, target interface, and autoregressive update are explicit.
5. Develop exact-observability instruments, complete frame and object endpoints, and paired clip-cluster inference for comparisons whose effect depends on retained-history reach.
6. Evaluate the methods, robustness, negative cases, and external boundaries; identify the operating regions in which hierarchical context, dense context, checkpointing or reversibility, and entropy-informed spatial allocation are appropriate.

Together, these tasks answer one systems question: how should a long-horizon predictor allocate spatial detail, retained history, and training memory so that the consistency criterion relevant to the task can be maintained under bounded resources? The wording matters. It asks for a conditional allocation rule, not for an architecture that wins every benchmark.

## Object and subject of research

The object of research is the process of neural video prediction over long self-fed horizons under finite computational and memory resources. It includes the formation, retention, and reuse of visual information during training and autoregressive inference, when generated frames enter the model's later inputs and relevant scene content may leave the current viewport or context window.

The subject of research is the set of system models, reversible temporal transformations, hierarchical spatial and context representations, positional encodings, evaluation instruments, and statistical procedures used to preserve task-relevant consistency and to identify reach–fidelity operating regions. The subject therefore covers relations between an intervention, the resource it directly changes, and the consistency endpoints used to judge it. It excludes broad claims about all generative video systems and deployment performance in untested application domains.

## Research methods

Systems decomposition is used to define state, observation, control, disturbance, prediction feedback, and the locations of the principal resource costs [5–11, 41]. Multi-criteria analysis separates appearance, identity, geometry, trajectory, false-foreground, and cross-level consistency instead of collapsing them into one score. Information theory supplies the categorical entropy used to describe the dispersion of raster-byte predictions [9]. It is treated as a diagnostic unless calibration is measured [44, 45].

Controlled procedural simulation isolates temporal dynamics and memory demand. LoopMom supplies analytically known world and viewport geometry; long persistent-object Moving-MNIST provides a different synthetic boundary; OpenSTL datasets are used only where the corresponding published experiment directly evaluated them [1, 124, 126]. Object-level correspondence is established with Hungarian assignment [125]. Effects are computed on paired clips, clips rather than rollout frames are resampled, and simultaneous maximum-statistic intervals control registered comparison families [120–123]. Content-addressed configurations, untouched or protected populations, and artifact-derived reporting reduce opportunities for post-hoc reinterpretation; preregistration and benchmark-variance analyses motivate the same separation between planned claims and observed variability [168, 170].

## Scientific novelty

Here, scientific novelty is methodological and empirical. It concerns how a resource-constrained video-prediction system is decomposed, measured, and compared; it does not rely on claiming priority for broad architectural families.

1. A multi-criteria formulation of consistency is developed for long self-fed video prediction. Static appearance, identity, geometry, and conceptual requirements are extended to temporal continuity, trajectory correctness, re-entry recovery, hallucination-sensitive foreground support, and agreement between explicit representational levels. The extension links each criterion to an observable failure and an appropriate endpoint, avoiding the assumption that low average frame error implies process consistency.

Relative to the earlier static-image taxonomy and conventional frame-aggregate evaluation, the difference is the trajectory-level operationalization: delayed events, assignment failures, false foreground, and cross-level contradictions are made separately observable.

2. Reversible temporal computation is delimited by an explicit state contract. Additive coupling is applied to the persistent doubled transformer state, while non-invertible encoding, projection, fusion, decoding, and loss interfaces remain outside the exact claim. The work combines algebraic reconstruction, numerical replay, gradient comparison, activation-memory measurement, and computation accounting to identify when reverse replay is useful and when checkpointing or ordinary backpropagation is the more suitable choice.

The distinction from a generic reversible-network construction is this verified applicability boundary: reconstruction, retained state, replay metadata, and computation premium are evaluated together instead of inferring end-to-end reversibility from one invertible block.

3. A hierarchical visual representation is developed over uncompressed raster pixel-channel bytes using local–global–local processing, causal prediction, and N-dimensional rotary positional encoding. Categorical predictive entropy provides a model-internal spatial-complexity signal without requiring a separate learned uncertainty head. Unlike a stronger adaptive-token claim, the validated method uses the fixed spatio-temporal packages reported in the publication; using entropy to change region size or compute on the fly is defined as a future controller and is not counted as an experimental result.

4. A two-level hierarchical context model is developed for matched-budget long-horizon prediction. Its package specification separates spatial level, temporal stride, frame count, anchor geometry, context tokens, target tokens, and payload. This enables a tested reach–fidelity frontier: hierarchy is evaluated where coarse history extends beyond a dense fine window, while flat context is also evaluated where reach is saturated and fine identity detail can dominate.

Hierarchical latent architectures already exist [20, 35–37]. The new element claimed here is consequently the controlled package-by-budget intervention and its measured lower and upper inversions, not the use of multiple scales itself.

5. An evaluation methodology is developed for memory-dependent video prediction. Exact geometric observability defines pair-specific reach-differential populations rather than relying only on aggregate clips. Complete foreground metrics penalize both missing and extra content; persistent-object assignment separates support, continuity, identity, and trajectory. Paired clip-cluster resampling and simultaneous inference then preserve rollout dependence and multiplicity. This combination makes a claim about retained reach falsifiable at the population where the compared packages differ.

## Practical significance

In practical terms, the work provides a reproducible research toolkit rather than a deployment product. Stable command-line interfaces support training, evaluation, analysis, and validation. These provisions follow the broader reproducibility principle that claims should be traceable to data, code, configurations, and a declared evaluation procedure [166].

For future systems, the methods also provide a decision procedure. If the needed past lies outside a dense window, a lower-resolution long stream can buy reach. If the relevant history is already present but identity detail is lost, dense fine context is preferable. When the backward activation tape dominates training, checkpointing or a verified reversible core can exchange computation for memory. Predictive entropy can motivate a spatial allocation policy, although its end-to-end benefit still requires direct testing.

Documented implementation acts and educational use are reported only to the extent supported by signed records. **[AUTHOR INPUT: insert the verified institutions, dates, covered software components, and appendix letters for the implementation acts.]**

## Reliability and validity of results

Reliability is supported at several levels. Algebraic properties are checked against implementation boundaries; numerical replay and gradient directions are compared where reversibility is claimed. Controlled generators expose the latent mechanism instead of relying exclusively on natural scenes with unknown history. The central context study separates development data from hash-pinned untouched, purpose-locked, and protected populations, following the logic of holdout protection in adaptive analysis [167]. Comparators share capacity where the research question requires it, while whole-configuration comparisons are labelled as such.

Reproducibility checks complement those inferential safeguards. Configurations and checkpoints are content-addressed, headline artifacts are regenerated from frozen evidence tables, and compatibility checks reject comparisons with mismatched interfaces. Published point estimates that lack seed-level records remain labelled as such; they are not retrofitted with uncertainty. This separation between verified provenance and estimated sampling variation is important because the two establish different kinds of reliability.

## Personal contribution of the applicant

Within this programme, the applicant formulated the systems decomposition, developed the software and controlled generators, implemented the model and evaluation components attributed to the dissertation, conducted the experiments, curated the evidence artifacts, and prepared the scientific analysis. Contributions associated with joint publications are distinguished from dissertation-only extensions.

The applicant also carried out the dissertation-level synthesis: aligning terminology across the separate studies, auditing implementation boundaries against published descriptions, and narrowing claims where the available experiment supports only a fixed configuration or a conditional operating region. Administrative statements about co-author roles remain subject to approval by the corresponding authors and the institution.

- For **[A1]**, the applicant developed and implemented the V-BLT method, RoPE-ND integration, procedural experiments, and analysis; the co-author contributed scientific supervision, systems framing, and manuscript review. The dissertation narrows the published adaptive wording to the fixed-patch evidence actually reported.
- **[A2]** is a sole-author work. The applicant developed TempVerseFormer, the reversible temporal backward procedure, controlled motion experiments, external comparisons, and memory/time evaluation. The verified reversible-state contract and later stochastic-replay analysis are stated separately where they extend the article.
- For **[A3]**, the applicant performed the technical review and proposed the conceptual-consistency category; the co-author contributed scientific supervision and editorial review. The dissertation extends that static-image discussion into operational video criteria without attributing temporal evidence to the paper.
- For **[A4]**, the applicant developed the hierarchical-context study, exact-observability instrument, endpoint suite, statistical protocol, robustness programme, figures, and reproducible artifact chain. **[AUTHOR INPUT: verify the final author list and contribution statement at submission.]**

## Approbation and publications

Principal results have been presented in three journal articles already identified as [A1]–[A3], with the hierarchical-context manuscript [A4] prepared for submission. **[AUTHOR INPUT: supply the complete six conference records, event names, locations or online status, dates, page ranges, and the final aggregate publication statement. Confirm whether DSMSI-2023 is included.]**

## Dissertation structure and volume

The dissertation consists of an English annotation, this Introduction, four chapters, General Conclusions, References, and Appendices. Two independent enabling methods occupy Chapter 2: reversible temporal computation for activation-limited training and hierarchical raster-byte representation for nonuniform spatial complexity. Experimental evidence in Chapter 4 selects the generative mechanism, tests retained reach and fidelity across budgets, examines robustness and boundary cases, and yields an operating-region decision rule.

General Conclusions return to the six research tasks and state the resulting methods, operating regions, and limits without reconstructing the publication sequence. References bind the technical and empirical claims used in the argument. Material needed for audit or reproduction moves to the appendices: complete configurations, seed-level and endpoint tables, supplementary derivations, negative-result receipts, provenance records, software commands, figure permissions, administrative documents, and the publication-to-result mapping. **[AUTHOR INPUT: insert total pages, main-text pages, figures, tables, references, appendix count, and final appendix letters after institutional typesetting.]**
