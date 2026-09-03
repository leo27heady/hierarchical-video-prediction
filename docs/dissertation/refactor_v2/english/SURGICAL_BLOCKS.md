<!-- BLOCK abstract -->
This dissertation examines resource-constrained video prediction as a problem of retaining and reusing task-relevant evidence. Appearance, identity, geometry, concept, trajectory, temporal continuity, and cross-level consistency are related but not interchangeable endpoints. Memory accounting separates retained context/state, internal activations, parameters and optimizer state, and temporary workspace. Spatial compression can free retained capacity for older observations, but that opportunity does not guarantee better prediction.

The four-chapter structure connects system requirements, published supporting methods, the controlled retained-context architecture, and empirical operating regions. Chapter 1 preserves the consistency synthesis of Article [A3], including the distinction between static conditioning and autonomous temporal prediction. Chapter 2 preserves the published TempVerseFormer [A2] and V-BLT [A1] methods and results. TempVerseFormer reconstructs a specified internal state to reduce activation storage at a recomputation cost; it does not enlarge the inference context. V-BLT implements fixed local–global–local byte processing with RoPE-ND and predictive-entropy diagnostics; those diagnostics did not themselves implement adaptive token routing.

Chapter 3 specifies the registered TRB allocation study: 43 unique configurations, two training seeds, and a common 30,000-update endpoint. The frontier, allocation, coarse-expansion, level-depth, ownership, and space–time studies are long-context/distant-evidence experiments with a four-frame prediction endpoint. The entire matrix is registered development evidence. Fresh sealed confirmation is restricted to selected, hash-locked contrasts; the whole frontier is not thereby confirmed. Long autoregressive behavior remains supported by the distinct historical E2/E5/E6 protocols, including the non-replicated late reversal and the coverage-versus-identity boundary.

The OpenSTL-native composition combines an end-to-end fine/coarse pyramid, genuine fixed-quota gathering, physical x/y/time/scale/level metadata, and an identical-forward reversible Transformer core. Static, density, and hybrid inference policies are separated from mixed route-aware training. The proposed external evaluations are Moving MNIST 10-to-10 and TaxiBJ 4-to-4 with seeds 0 and 1. CPU compatibility does not establish competitive quality, total-GPU-memory savings, or a multiplied gain from individual components. Matched baselines, checkpointing profiles, routing stability, and smaller-quota quality remain distinct evidence cells.

{{CLAIM_A4_SUMMARY}}

{{CLAIM_INT_SUMMARY}}

The dissertation's six tasks remain system/consistency analysis; reversible temporal computation; V-BLT/RoPE-ND and spatial-complexity diagnosis; hierarchical retained-context design; observability/evaluation/statistics; and empirical operating regions plus integrated composition. Articles [A1]–[A3] retain their published attribution. Hierarchy and integration now belong to one combined manuscript in preparation [A4]; historical A4/A5 experiment prefixes do not denote two promised publications. Final comparative sentences, counts, and publication status are filled from the same locked claim ledger as Chapter 4 and the general conclusions.

Keywords: video prediction; task-specific consistency; retained context; spatial and temporal hierarchy; reversible computation; physical token metadata; dynamic routing; resource-constrained evaluation.
<!-- END BLOCK -->

<!-- BLOCK intro_relevance -->
Memory is treated as four separately measured resources: retained context/state, intermediate activations, parameters with gradients/optimizer state, and temporary workspace/allocator overhead. Spatial compression can release retained capacity for a longer observed history; temporal allocation decides where that capacity is spent. Reversibility changes the internal training tape, not the age of available evidence. Routing at an unchanged quota changes information allocation rather than producing an additional memory multiplier.

The controlled TRB study asks whether distant observed evidence improves a four-frame prediction endpoint under specified context budgets. This is different from a long self-fed forecast, where prediction errors re-enter memory. Historical rollout evidence remains necessary for that second claim. OpenSTL evaluation asks a separate external question: whether a composed architecture is competitive under matched Moving MNIST and TaxiBJ protocols and provides a measured quality–memory advantage. None of these questions is settled by analytic token counts alone.
<!-- END BLOCK -->

<!-- BLOCK tasks -->
## Мета та дослідницькі завдання

The aim is to improve task-relevant consistency and resource efficiency of hierarchical video prediction by analyzing temporal computation, spatial representation, retained context, and evaluation. The main architectural focus is retention of relevant distant evidence. Resource efficiency always names the resource saved and the quality/computation cost incurred.

The six dissertation-level tasks are preserved:

1. Analyze prediction as a partially observed dynamic system; distinguish uncertainty and consistency failures; define causal comparison and resource-accounting requirements.
2. Develop and delimit reversible temporal computation; verify state reconstruction and reverse differentiation; characterize the activation-memory/recomputation trade-off.
3. Develop hierarchical raster-byte representation, RoPE-ND, and entropy-based spatial-complexity diagnosis, distinguishing fixed-patch evidence from proposed adaptive allocation.
4. Specify a hierarchical predictor with explicit streams, ages, spatial identities, token cost, target interface, and causal update. The new budget, allocation, expansion, level, ownership, and space–time questions are nested under this task.
5. Establish exact observability, frame/object metrics, clip-aware paired inference, development selection, and protected confirmation. Distinguish distant-context access from autoregressive rollout stability.
6. Evaluate operating regions, robustness, negative cases, and external boundaries; test whether hierarchy, reversible core execution, and spatial routing can compose under a matched OpenSTL quality–memory protocol.

These tasks retain the contributions of Articles [A1]–[A3]; they are not replaced by a list of experiment codes. The new questions H1–H7 in Chapter 1 connect to the exact configurations in Chapter 3 and the evidence cells in Chapter 4. The intended result is a conditional design rule rather than an architecture assumed to win every comparison.
<!-- END BLOCK -->

<!-- BLOCK novelty -->
## Наукова новизна

The system contribution is a task-linked separation of retained evidence, internal training activations, spatial detail allocation, and the remaining parameter/workspace costs. This separates methods that are often grouped under the single label “memory efficient,” and makes each comparative claim testable at its own interface.

The published methodological contributions remain the implemented reversible temporal state/backward procedure in TempVerseFormer [A2], and the fixed local–global–local byte hierarchy, RoPE-ND, and predictive-complexity diagnostics in V-BLT [A1]. Article [A3] supplies the appearance/style, identity, geometry, and concept-consistency synthesis; the dissertation extends its reasoning to temporal observability without attributing video experiments to that review.

The retained-context contribution is an explicit package and observability contract together with controlled allocation experiments. The new campaign varies budget, distribution, coarse expansion, registered level constructions, ownership representations, and space–time allocation. CEX/LVL/OVR are compound interventions and are described accordingly. The complete matrix is development evidence; only evaluated selected roles can receive sealed confirmation.

The integrated architectural contribution is the specified composition of real packing, physical metadata, route-aware exposure, and a same-forward reversible core. The superseded fixed-region/second-order prototype is historical feasibility evidence, not the final contribution. New memory and competitive-quality claims remain conditional on their artifact cells.

{{CLAIM_A4_SUMMARY}}

{{CLAIM_INT_SUMMARY}}
<!-- END BLOCK -->

<!-- BLOCK practical -->
## Практичне значення

The work provides controlled dataset generators, published implementations for TempVerseFormer and V-BLT, an explicit retained-context allocator and config-driven training campaign, and a separate OpenSTL-native integrated research model [171–173]. Their practical roles differ: reproducible data, reversible activation reconstruction, structured spatial diagnosis, controlled allocation, and a composition under evaluation.

The campaign controllers save optimizer, random-number, model/EMA, and progress state for resumption; local JSON/JSONL/checkpoints are authoritative and W&B mirrors the record. Exact branches, source hashes, commands, dataset manifests, and verification receipts are in Appendix Ж. CPU integration checks demonstrate bounded compatibility; practical GPU memory/quality advantages remain result-gated. Existing institutional implementation records are preserved verbatim in Appendix Б and the software-use subsection of Chapter 4; unverified administrative information is not inferred from repository activity.
<!-- END BLOCK -->

<!-- BLOCK validity -->
## Достовірність результатів

Published results, historical protected experiments, new development estimates, selected confirmation, and CPU mechanism tests retain separate provenance. Every new numerical statement must resolve to an immutable artifact, metric implementation, population, configuration, checkpoint, and uncertainty unit. Both training seeds are displayed; clip-level intervals are conditional on a trained checkpoint. A broad training-seed confidence claim is not inferred from two runs.

No incomplete development result is reported as confirmatory. E5a's non-replicated late reversal and E6's coverage/identity conflict remain visible rather than being removed for an optimistic narrative. TRB, the historical long Moving-MNIST probe, and the planned standard OpenSTL protocols answer different questions. The current 86 training runs are neither restarted nor reinterpreted as long-rollout training by this editorial change.
<!-- END BLOCK -->

<!-- BLOCK publication_update -->
The redesigned hierarchy study and OpenSTL integration form two evidence blocks of one combined manuscript in preparation [A4], whose bibliographic record remains provisional [136]. Historical `A4_*`/`A5_*` labels remain artifact identifiers, not a commitment to two additional papers. The author/contributor declarations above are retained; the final author list, institutional approval, and publication status still require author-supplied confirmation.
<!-- END BLOCK -->

<!-- BLOCK intro_structure -->
## Структура дисертації

The four chapter titles and semantic order are retained. Chapter 1 establishes the system problem, consistency taxonomy, resource terms, and falsifiable questions. Chapter 2 preserves published reversible-computation and V-BLT evidence while relocating standard derivations. Chapter 3 defines the controlled retained-context method, exact experiment programme, and separately implemented OpenSTL architecture. Chapter 4 organizes allocation results, selected confirmation, historical long-rollout boundaries, component profiles, and external quality–memory tests. General conclusions align with the six tasks. Existing appendices retain their Ukrainian letter sequence, administrative records, contribution declarations, and permissions while gaining the configuration, claim, formula, and migration registries.

This is an evidence-gated mixed-language draft: retained foundation text is Ukrainian and new complete blocks are English. Final language harmonization, institutional counts, numbering, and result-dependent abstract/conclusion sentences follow evidence lock rather than being guessed in advance.
<!-- END BLOCK -->

<!-- BLOCK static_bridge -->
### 1.2.5 From static consistency controls to video requirements

Article [A3, 133] contributes a conceptual synthesis, not merely an illustration collection. Appearance/style consistency concerns surface characteristics; identity concerns which particular object is preserved; geometry concerns shape and spatial arrangement; concept consistency concerns the intended semantic entity or relation. A method can preserve one while changing another. In video, the same distinctions must be tested across time, disappearance, re-entry, and self-feedback.

The review's control interfaces organize where evidence enters a generative system. Input conditioning keeps a reference available during generation. Weight adaptation stores instance or style information in parameters. Structural conditioning supplies geometry such as an edge or pose. Inversion connects editing to a recovered source trajectory. Attention sharing couples features across generated samples. These are alternative information interfaces, not consecutive stages of one algorithm. A supplied future pose is privileged information, not a learned memory of an unseen future.

Static diffusion step and physical video time are separate coordinates. Successful denoising of one image does not establish temporal consistency. Appearance can persist while trajectory fails, and coverage can improve while identity deteriorates. This is why the dissertation measures the information level that a hierarchy is expected to preserve: coarse support/trajectory and fine identity/texture need distinct endpoints.

The detailed diffusion, image-to-image, ControlNet, inversion, and StyleAligned tutorial sequence and its original publication assets move to Appendix Е, with their original sources and permissions. The synthesis stays here. The resource map below connects these ideas to the dissertation's own interventions without claiming that a static control method solves autonomous long-term video memory.
<!-- END BLOCK -->

<!-- BLOCK state_notation -->
### 1.1.1 System boundary, state, observation, and control

System boundaries and hierarchy follow the systems-analysis viewpoint [5–8, 41]; retained model state is distinguished from the physical process [10, 11]. Let $x_t$ denote the physical/semantic world state, $y_t$ the observed frame, $u_t$ a known action when the protocol supplies one, and $w_t,v_t$ process/observation uncertainty. The predictor retains $m_t$, not the entire world. A compact system contract is

$$
x_{t+1}=f(x_t,u_t,w_t),\quad y_t=h(x_t,v_t),\quad
m_t=C_t(y_{\le t},u_{\le t}),\quad
p_\theta(\widehat y_{t+1:t+K}\mid m_t,u_{t:t+K-1}).
\tag{1.S1}
$$

This notation distinguishes physical state, available observation, retained model context, and conditional prediction; it is a system definition rather than a fitted dynamical law. Action-conditioned TRB is a controlled instrument, not the definition of all video prediction. In protocols without supplied actions, the corresponding condition is absent. The resource-constrained objective is specified in Section 1.1.3; the expanded source notation is retained in Appendix Д.

{{LEGACY_FIGURE:1.1}}

Partial observation matters when a region leaves the viewport but continues to exist in the world. A later re-entry can require evidence older than a short dense context. Retaining a coarse older observation may extend potential access while losing identity detail. This distinction motivates both the controlled reach population and the separate identity endpoints; it does not presume that longer memory is always better. Comparable state/observation distinctions arise in physically constrained forecasting [47–49].
<!-- END BLOCK -->

<!-- BLOCK intro_method_update -->
Controlled procedural generation supplies interpretable motion and representation tests. TRB supplies exact viewport/world observability; historical long Moving-MNIST supplies a separate persistent-object rollout boundary. Published OpenSTL results retain their original protocols, while the new composition plans matched Moving MNIST and TaxiBJ evaluation. Hungarian object assignment, clip-aware inference, preregistration, and training-seed reporting remain distinct parts of the evaluation contract [1, 120–126, 168, 170].

The current integrated checks concern real packing, physical metadata, and same-forward internal reversible execution. Future matched GPU profiles compare ordinary autograd, checkpointing, and reversible storage. They do not inherit the old second-order temporal-prototype results. Configuration, data, code, checkpoint, and metric hashes define which empirical statement each artifact supports.
<!-- END BLOCK -->

<!-- BLOCK objective_contract -->
In equation (1.7), $\theta$ denotes model parameters, $C$ the context-allocation policy, and $a$ the execution/storage schedule. The feasible set contains only configurations satisfying the declared context-state, activation and computational-work budgets. Chapters 2–4 distinguish these interventions before considering composition. Sections 3.7 and 4.8 now specify the new integrated architecture and its pending evidence cells: they do not report a completed composition-performance test. Historical feasibility results refer to the superseded prototype archived in Appendix Е and do not transfer to the new model.
<!-- END BLOCK -->

<!-- BLOCK hierarchy -->
### 1.3.4 Spatial, temporal, and computational hierarchy as distinct interventions

Spatial hierarchy provides coarse coverage and selected detail. Temporal hierarchy allocates resolution and slots by age, spending fewer tokens on some older observations. Temporal overlap describes whether an age is represented at more than one level: nested ownership duplicates scale coverage, whereas exclusive ownership separates supports; a residual representation changes what the fine stream means.

Computational hierarchy and reversibility concern stored internal activations. A reversible core reconstructs intermediate values for differentiation; it neither extends the observed history nor restores details discarded by a codec. Dynamic space–time routing assigns a fixed fine-detail quota to physical locations and times. It changes allocation quality at equal quota, and yields a memory advantage only if a smaller real retained state preserves acceptable quality.

The opportunity is to use spatial savings for temporal reach, not to conflate spatial and temporal abstraction. A coarse older token can retain coverage while losing identity. Likewise, a long observed context and a long autoregressive forecast are different axes. The former is tested by the new TRB endpoint study; the latter requires actual self-feeding, supplied by the historical rollout evidence or a separately registered new evaluation.

An analytic token ratio is not a GPU-memory ratio. Parameters/optimizer and temporary workspace coexist with retained state and activations. The same memory resource must be compared under a matched protocol; independently measured component ratios do not multiply into an integrated saving.
<!-- END BLOCK -->

<!-- BLOCK system_map -->
### 1.3.5 System map and intervention boundaries

![Which memory resource is directly changed by each intervention?](../../../figures/dissertation_v2/resource_interventions.svg)

Figure D1-resources — Retained evidence, internal activations, parameter/optimizer storage, and workspace are distinct. Spatial and temporal allocation act on the first; reversibility acts on a specified part of the second. Routing overhead and dense boundary tensors remain in the measured total. This is a conceptual accounting map, not a result plot.

For partially observed prediction, first ask whether relevant evidence remains available; then ask whether its detail and learned use suffice. When the bottleneck is training activations, compare ordinary autograd, checkpointing, and same-forward reversible execution. When it is retained context, compare explicit token/byte budgets. The intervention and the evaluation endpoint must name the same resource.
<!-- END BLOCK -->

<!-- BLOCK literature -->
Spatial token sparsification such as DynamicViT changes which image tokens reach later processing [176]; it does not by itself establish temporal memory. Temporal subsampling changes the ages retained at a fixed spatial representation. Multiscale memory and clocked latent models distribute representation across scales or update rates [15, 20]. Activation checkpointing recomputes graph segments, whereas reversible blocks reconstruct specified internal states [30, 31, 175]. These methods act at different interfaces and should not be compared by one unnamed “memory reduction.” The unresolved question here is the observed quality–resource operating region when those interfaces are specified and, later, composed.
<!-- END BLOCK -->

<!-- BLOCK hypotheses -->
### 1.5.2 Falsifiable hypotheses and their evidence

H1 asks whether, at low retained budgets, hierarchy reduces horizon-4 foreground error when relevant distant observed evidence would otherwise be excluded. H2 asks whether the best cross-level distribution depends on budget rather than on a single hand-selected package. H3 asks whether registered additional levels and ownership packages justify their displaced detail and other architectural changes. H4 asks how fine space–time placement preserves recent motion, distributed anchors, or information-rich regions.

H5 asks whether internal reversibility reduces training activation storage for the same forward function. H6 asks whether dynamic routing improves allocation at fixed quota and permits a smaller measured state within a declared quality tolerance. H7 asks whether full composition is competitive under matched OpenSTL protocols and improves the measured quality–memory frontier. These are hypotheses, not assumed positive results.

A separate rollout question asks whether selected context policies remain useful after their own predictions update memory over many windows. H1 does not answer it. Historical E5/E6 provide self-fed evidence, including negative boundaries; a selected-checkpoint v2 rollout would need separate registration and authorization without retraining the 86-run matrix.

H1–H4 connect to the controlled FRT/ALC/CEX/LVL/OVR/STS families and selected confirmation. H5–H7 connect to same-forward profiles, separately trained development ablations, quota/routing diagnostics, and matched external results. A missing comparison narrows the claim. The configuration/evidence registries make these dependencies explicit before results are inserted.
<!-- END BLOCK -->

<!-- BLOCK procedural -->
### 2.1.1 Controlled procedural generation

The procedural toolkit defines object geometry, motion, visibility, and rendering conditions so that controlled 2-D rotations and 3-D polycube sequences can test published methods [171]. These datasets form a ladder of representation and motion difficulty rather than interchangeable samples from natural video. The representative sequences and dataset-ladder figure remain below; exhaustive generator equations, parameters, and the secondary gallery are preserved in Appendix Е/В. The published numerical comparisons retain their original protocol and cannot be retroactively pooled with TRB or OpenSTL integration results.
<!-- END BLOCK -->

<!-- BLOCK reversible_block -->
### 2.2.2 Reversible transformation and its exact boundary

TempVerseFormer's additive-coupling block reconstructs its complete two-stream internal input, not an arbitrary image or a compressed projection. For fixed conditions and deterministic modules, forward and inverse relations form one scientific object:

$$
\begin{aligned}
y_1=x_1+F_\theta(x_2),&\quad y_2=x_2+G_\theta(y_1),\\
x_2=y_2-G_\theta(y_1),&\quad x_1=y_1-F_\theta(x_2).
\end{aligned}
\tag{2.R1}
$$

The left-hand quantities are the paired block states; $F$ and $G$ themselves need not be invertible. The inverse is exact in real arithmetic, with finite-precision error checked separately. The VAE encoder/decoder, projections, normalization/merge boundaries, and loss are not made bijective by this construction. They retain ordinary differentiation or separately stored boundary information. The published architecture in the next subsection is the main diagram; the redundant block schematic and detailed state-boundary table are in Appendix Е/В.

Unit Jacobian determinant and the full algebraic expansion are supporting derivations, not an independent empirical contribution. If compression or forgetting is required, it must occur outside the reversible state or be represented by additional retained information. Algebraic reversibility alone does not guarantee stable repeated temporal prediction. The published measurements below determine the actual memory–compute and quality boundary [A2].
<!-- END BLOCK -->

<!-- BLOCK replay -->
### 2.2.4 Reverse replay and gradient accumulation

Reverse replay reconstructs the preceding complete state, rebuilds one local differentiable transition, accumulates its parameter gradient and state adjoint, and releases that local graph. The final state, required boundary conditions, and any deterministic replay information remain available. The standard chain-rule expansion is preserved in Appendix Е with the original formulas and implementation references.

$$
\bar h_{t-1}=(\partial\Phi_\theta/\partial h_{t-1})^\top\bar h_t+\partial\ell_{t-1}/\partial h_{t-1},\qquad
g_\theta\mathrel{+}=(\partial\Phi_\theta/\partial\theta)^\top\bar h_t.
\tag{2.R2}
$$

The quantities are the state adjoint and accumulated parameter gradient for the executed temporal objective. The relation is exact differentiation for the specified replayed function; its practical validity depends on reconstruction and gradient parity within the tested tolerance. A compact replay timeline replaces separate forward-chain and backward-storage diagrams. The published quality, memory, and runtime evidence remains three distinct objects because the same horizon can improve one endpoint and worsen another.

![What is stored and recomputed during reverse differentiation?](../../../figures/dissertation_v2/reversible_replay.svg)

Figure D2-replay — Reconstruct one prior state, replay one local graph, accumulate its gradient, then release it. Ordinary encoder/decoder/loss boundaries remain outside this storage schedule. This is a mechanism diagram, not a GPU profile.
<!-- END BLOCK -->

<!-- BLOCK stochastic -->
### 2.2.8 Stochastic replay and numerical extensions

The deterministic published boundary must not be enlarged by a general derivation. Stochastic transitions additionally require reproducible random draws or saved generator state; nondeterministic recomputation can change the function being differentiated. Anchor checkpoints can limit accumulated reconstruction drift, but consume memory and change the recomputation schedule. The original replay, Lipschitz, anchor, and error-propagation derivations are preserved in Appendix Е as extensions with their own implementation status.

These extensions do not establish that every stochastic component of a video pipeline is reversible. Nor is a theoretical stability bound a measured rollout-quality result. The later integrated core disables dropout and keeps routing outside its inverse boundary; CPU bfloat16 checks and future CUDA checks are reported separately rather than using this discussion as a blanket guarantee.
<!-- END BLOCK -->

<!-- BLOCK rope -->
### 2.3.4 RoPE-ND and physical axes

V-BLT's raster-byte sequence preserves time, row, and column identities through multi-axis rotary position encoding rather than treating linear sequence rank as the only coordinate. With channel pairs allocated to physical axes, the rotation can be written compactly as

$$
R(p)=\operatorname{blockdiag}_{a,k}\operatorname{Rot}(p_a\omega_{a,k}),\qquad
\langle R(p)q,R(p')k\rangle=\langle q,R(p'-p)k\rangle.
\tag{2.R3}
$$

The relation describes the relative-phase property for the declared axis decomposition, not a measured isolated gain of RoPE-ND. Elementary sine/cosine derivations and the original coordinate examples remain in Appendix Е. The V-BLT result compares complete configurations; it does not disentangle positional encoding from the byte interface and hierarchy.

{{LEGACY_FIGURE:2.15}}

The integrated model in Section 3.7 uses continuous Fourier features plus a level embedding, not this rotary operator. What transfers is the requirement to preserve physical identity after packing, not identical positional code or pretrained weights.
<!-- END BLOCK -->

<!-- BLOCK measured_router -->
### 2.3.7 Measured V-BLT evidence and the later routing hypothesis

V-BLT implements fixed hierarchical patches and measures predictive byte entropy. Its local–global–local architecture, causal objective, RoPE-ND, categorical output, and published 2-D/3-D results remain part of the dissertation's main evidence. Predictive entropy shows spatially uneven uncertainty in the tested representation; it does not demonstrate an executed dynamic partition controller.

The later integrated router is independently specified. It scores fine–coarse residual and temporal change, optionally adding a learned correction, then gathers a real fixed quota with physical metadata. It does not consume the V-BLT entropy map. Equal-quota routing is an allocation-quality comparison; memory benefit requires a smaller real retained tensor at acceptable quality. A static-only checkpoint cannot be assumed robust to changed membership without route-aware exposure and a switch evaluation.

![What transfers from V-BLT to the integrated router?](../../../figures/dissertation_v2/vblt_router_boundary.svg)

Figure D2-boundary — Published fixed-patch processing and entropy diagnosis motivate an allocation question. The implemented integrated router and its future quality/memory measurements are separate evidence, not retroactive V-BLT results. The old hypothetical dynamic-partition formula (source 2.42) is retained only as a historical design proposal in Appendix Е.
<!-- END BLOCK -->

<!-- BLOCK synthesis -->
## 2.4 Наслідки для проєктування ієрархічно-контекстної моделі

### 2.4.1 What transfers across the methods

The published methods solve adjacent scaling constraints, not the entire retained-context question. TempVerseFormer provides a specified reconstructed state and reverse differentiation schedule. The changed resource is the stored activation tape of its temporal core; the unchanged resource is the content and age of the observations available to the predictor. A compact derivative computation cannot retrieve an observation that has already left the context.

V-BLT provides a local–global–local byte hierarchy and a physical-axis position contract. The changed interface is the spatial representation presented to a global processor; the diagnostic is predictive entropy. The unchanged boundary is that patch membership in the published experiments is fixed. Entropy concentration near difficult boundaries motivates further allocation work but is not an adaptive-router ablation.

The retained-context predictor asks a different question: which ages and levels should be present in a finite package? Spatial reduction can finance older coarse evidence, but identity may depend on detail that the same reduction discards. This makes coverage, trajectory, appearance, and identity separate endpoints rather than interchangeable descriptions of consistency. The new space–time study explicitly tests redistribution of detail over ages at fixed payload; it does not learn content-dependent membership.

The integrated model then tests composition, using an end-to-end encoder rather than the frozen controlled codec. Real fine-token gathering changes the core input length; physical metadata preserves position after membership changes; internal reversibility changes activation storage for the same forward computation. Encoder candidates, routing, decoder, parameters, optimizer state, and workspace remain outside that reversible saving. This boundary explains why component ratios cannot be multiplied and why a token-count ratio is insufficient evidence for total-memory superiority.

### 2.4.2 Published findings retained and new claims deferred

The published TempVerseFormer quality, memory, runtime, and OpenSTL results remain in Sections 2.2.5–2.2.7 with their original configurations. Favorable quality in several motion regimes does not establish dominance on every metric or dataset. The published V-BLT 2-D/3-D results remain in Section 2.3.6: they support feasibility and the reported configuration-level trade-offs, not isolated causal gains from every component. Article [A3]'s static-consistency taxonomy remains in Chapter 1, while exhaustive tutorial assets and standard derivations are preserved in the appendices.

Conceptually, the mechanism/resource map has three columns expressed in prose. Reversible replay changes internal stored activations and leaves retained evidence unchanged. Spatial hierarchy changes the units processed globally and does not guarantee useful temporal reach. Retained-context allocation changes which past evidence is available and leaves the numerical differentiation method unchanged. Dynamic routing adds a fourth conditional intervention: it changes membership at a fixed quota, and only a validated smaller quota changes the retained cost.

These distinctions lead to Chapter 3's method contract and Chapter 4's operating-region analysis. Legacy E5/E6 test self-fed accumulation and identity boundaries. The new TRB matrix tests distant observed context at a four-frame endpoint. The combined manuscript in preparation [A4] connects that matrix to separately measured OpenSTL composition without erasing the published contributions or assuming the integrated hypothesis succeeds.
<!-- END BLOCK -->
