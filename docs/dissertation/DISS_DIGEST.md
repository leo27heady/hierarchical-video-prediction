# DISS_DIGEST — Dissertation Restructure Blueprint

**Candidate:** Kulyk Leonid Ruslanovych · **Specialty:** 124 System Analysis · **Supervisor:** Prof. O. B. Mokin, VNTU
**Current draft topic (OLD):** «Системний аналіз та інформаційні технології консистентного моделювання процесів ітеративної генерації відеоданих» — built around hierarchical **energy-based** approaches (HVEBT).
**Approved topic (NEW, verbatim, fixed):** «Системний аналіз та інтелектуальні технології консистентного моделювання процесів **ієрархічної** генерації відеоданих» / *Systems Analysis and Intelligent Technologies for Consistent Modeling of Hierarchical Video Data Generation Processes* — built around **discrete latent spaces (VQ-VAE-class) + hierarchical decomposition + autoregressive prediction**, with continuous-latent diffusion cited as the motivating limitation.
**Actual 4th article (replaces the hoped-for EBT article):** *"Hierarchy earns its keep where the memory horizon demands it"* — hierarchical latent video model with **long-span multi-stream context** (rate x resolution packaging family; the coarse level is its per-level-decodable spatial form — packaging reframe 2026-07-10, `docs/CLAIM_LEDGER.md`); purpose-built looping **memory-horizon testbed** with re-entry-stratified evaluation; matched-compute mechanism study (diffusion > regression; **EBT = two-sided characterized negative**); single-step long-coarse-memory win (+13% hard re-entry, 2 seeds, emerges at 20k steps); 200-frame AR rollout where hier-long beats a token-matched flat control exactly at the flat window's horizon boundary, and window length orders everything (48>18>6); plus a reversibility chapter (exact invertible stochastic recurrence, store-the-noise, grad-cos 1.0 vs BPTT, constant-memory rollout frontier, three-axis taxonomy proving the inner-optimization-loop axis cannot pay vs checkpointing).

Purpose of this document: the single working blueprint for the rewrite. Sections: (1) annotated TOC of the current draft; (2) chapter↔article map; (3) full inventory of EBT/"Article-4" dependencies with dispositions; (4) new-topic gap analysis; (5) UA–EN terminology table; (6) proposed new chapter structure; (7) open questions for the author.

---

## 1. Annotated TOC of the current draft

Sizes are approximate word counts of body prose (docx→md export; total written ≈ 25k words; Ch. 4 body, ВИСНОВКИ, and real references/appendices are essentially absent).

### Front matter
| Section | ~Words | Summary | Status |
|---|---|---|---|
| Title page | 120 | Kulyk L.R., topic (OLD wording), spec. 124, supervisor Mokin, VNTU 2026, UDC 004.054:[004.032.26+004.85] | Needs new topic wording; stray «-quote typo |
| АНОТАЦІЯ (UA) | 1150 | Goal = accuracy + resource efficiency via **hierarchical energy-based** approaches; summarizes 4 chapters (Ch.1 taxonomy, Ch.2 V-BLT/RoPE-ND/TempVerseFormer, Ch.3 HVEBT + Reversible MCMC, Ch.4 experiments incl. TaxiBJ/KTH); 7 results, 4 novelty items, implementation acts, copyright certificate «TempVerseFormer та HVEBT» | Full rewrite (energy → hierarchical/discrete; Ch.3–4 claims ahead of evidence) |
| ABSTRACT (EN) | 1000 | English mirror; states 10 publications: 3 UA professional (spec. 124) + 1 Scopus + 6 theses | Full rewrite in sync with UA |
| СПИСОК ПУБЛІКАЦІЙ | 750 | **All 10 items are Ya. O. Isaienkov's** (WGAN, ID-card CEUR, U-GAN, dental X-ray topics) — unreplaced donor template | Replace wholesale with Kulyk's real list |
| ЗМІСТ | 380 | 4-chapter TOC; §§4.2–4.4 literally titled "NAME"; appendix titles reference dental X-rays | Regenerate after restructure |
| ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ | 90 | GAN-era template (WGAN, DCGAN, HYPE, IDEF0, UML…); none of the actual abbreviations (EBM, HVEBT, V-BLT, RoPE-ND, BPTT, MCMC, VQ-VAE, SSIM…) | Replace wholesale |
| ВСТУП | 1250 | Contradiction: multi-level consistency needs long trajectories/deep hierarchies vs O(T) BPTT memory; **EBM positioned as the promising direction**; goal, 7 tasks, object/subject, 4 novelty items, personal contribution mapping Articles 1–4, 6 conferences, 185 pp/145 sources (placeholder numbers) | Major rewrite: motivation swaps energy→discrete-hierarchical-AR; task 5 and novelty 1 rewritten; placeholders [X], [НОМЕР НДР], [НОМЕР СВІДОЦТВА] |

### РОЗДІЛ 1 — Системний аналіз процесів ітеративної генерації відеоданих та їх консистентності (~5.6k words)
| Section | ~Words | Summary | Rewrite load |
|---|---|---|---|
| 1.1 Генеративні моделі як складні системи управління | 700 | Iterative generative models (diffusion, AR, EBM+MCMC) as open dynamic control systems; state-transition recurrence (1.1); macro/meso/micro emergence; inconsistency destroys emergence | Light (drop/trim EBM strand) |
| 1.1.1 Генерація як задача оптимального керування | 500 | Global energy E(x,u) as consistency criterion; generation = argmin E (1.2) via Langevin (1.3); energy vs entropy split; memory barrier → reversibility motivation; explicit scaffolding "буде покладено в основу HVEBT у Розділі 3" | **Heavy** — energy-as-criterion is the OLD spine |
| 1.1.2 Прикладні сфери | 420 | Medical imaging, remote-sensing monitoring, AV simulators; 6 system requirements | Keep as-is |
| 1.2 [mistitled duplicate of 1.1.2] Декомпозиція системи | 100 | Announces structural / hierarchical / temporal decompositions | Fix heading; keep |
| 1.2.1 Структурна декомпозиція | 220 | Encoder / Iterative Generator / Decoder macro-subsystems | Keep |
| 1.2.2 Ієрархічна декомпозиція простору станів | 285 | Macro (semantics) / meso (geometry) / micro (texture) levels; single-level VAE latents mix levels → anomalies | Keep — **core of the NEW topic** |
| 1.2.3 Часова декомпозиція | 230 | Two-dimensional temporal grid: pseudo-time k (denoise/MCMC steps) × physical time t | Keep (generalize k to per-frame generation steps) |
| 1.2.4 Класифікація видів невизначеності | 520 | Informational / structural-algorithmic / computational / target uncertainty; concludes: entropy estimation + reversible compute + "new hierarchical energy control apparatus" | Light (final clause energy→hierarchical-discrete) |
| 1.3 Аналіз існуючих підходів | 120 | Frames review of diffusion vs patch/AR vs EBM | Reframe ordering of paradigms |
| 1.3.1 Дифузійні моделі | 520 | DDPM/DDIM (1.4–1.7), LDM, DiT/Sora; flaws: single-level latent, target uncertainty/exposure bias | Keep — now the explicit **motivating limitation** per new topic; refresh 2024-era claims |
| 1.3.2 Патч-орієнтовані та авторегресійні | 400 | ViT/ViViT/Swin O(L²) + rigid tokenization; BLT entropy patching → announces V-BLT | Keep; **expand with VQ-VAE + VideoGPT-class discrete AR** (currently missing!) |
| 1.3.3 Енергетичні моделі (EBM) та ресурсоефективність | 480 | Gibbs (1.8), score matching (1.9), energy as anomaly indicator; BPTT-through-MCMC blocker; checkpointing/Neural ODE/RevNet survey; "reversibility = foundation of HVEBT" | **Heavy** — demote EBM from "most fundamental paradigm" to reviewed-and-characterized-negative; keep resource-efficiency survey |
| 1.4 Критерії консистентності | 260 | Consistency as system-theoretic stability criterion; taxonomy announced | Keep |
| 1.4.1 Просторова та об'єктна | 430 | ControlNet/DreamBooth etc. as local patches | Keep |
| 1.4.2 Часова | 160 | Optical-flow warping loss (1.10); 3D-RoPE/block-causal forward-refs | Keep |
| 1.4.3 Стильова та концептуальна | 330 | Own notion of conceptual consistency (Article on criteria), CLIP metric (1.11) | Keep |
| 1.4.4 Ієрархічна консистентність | 190 | New criterion "introduced for HVEBT (Ch.3)"; enforced by unidirectional stop-gradient control | Retarget from HVEBT to the new hierarchical model (mechanism survives) |
| 1.4.5 Траєкторіальна консистентність | 270 | Stability of the iterative optimization in pseudo-time K; energy-gap ΔE metric (1.12) | Rewrite metric: ΔE→ rollout/re-entry-stratified degradation metrics; ΔE survives only inside the EBT mechanism study |
| 1.5 Висновки до розділу + постановка задач | 720 | Contradictions, goal "…hierarchical **energy** approaches", 7 tasks; publications = six [X] placeholders | Rewrite goal/task 5; resolve refs |

### РОЗДІЛ 2 — Математичне моделювання та розкриття невизначеності (~4.9k words)
| Section | ~Words | Summary | Rewrite load |
|---|---|---|---|
| 2.1 Ентропійний метод / V-BLT | 1100 | 3-tier byte-level V-BLT (Local Encoder → Global Latent Transformer w/ time-causal mask (2.1) → Local Decoder); 256-way byte classification (2.2); per-pixel Shannon entropy maps (2.3) drive variable patching; contrasts local entropy vs global energy of HVEBT | Light (retarget the HVEBT contrast; note the discrete-classification decoder is a natural bridge to VQ/discrete-space framing) |
| 2.2 RoPE-ND | 900 | N-dim rotary PE (2.4–2.7), D divisible by 6, relative (Δt,Δy,Δx); coordinate scaling for cross-level attention "in HVEBT (Ch.3)" | Light (retarget cross-level role to new model) |
| 2.3 TempVerseFormer | 1100 | BPTT memory M=O(N·T·L·D) (2.8); reversible two-branch additive coupling (2.9–2.12); Temporal Chaining V_t=Φ(V_{t−1}) (2.13); Scopus solo publication | Keep |
| 2.4 Часонезалежне зворотне розповсюдження | 1300 | Two-phase algorithm, Lemma 1: peak memory O(L·D)=O(1) in T; Table 2.1 vs BPTT/checkpointing; Selective Backpropagation (last K_opt steps); closing bridge: **T↔K isomorphism → foundation of HVEBT MCMC** | Keep core; **replace bridge** with article-4 truth: constant-memory AR rollout + three-axis taxonomy (inner-loop axis cannot beat checkpointing) |
| 2.5 Висновки до розділу | 450 | 5 conclusions; item 5 = enabler of HVEBT; "published in [1],[2],[3],[4]" | Rewrite items 5 + refs |

### РОЗДІЛ 3 — Інформаційна технологія та методи оптимального керування ієрархічними ЕНЕРГЕТИЧНИМИ моделями (~6.2k words) — **the chapter to be replaced**
| Section | ~Words | Summary |
|---|---|---|
| 3.1 Формалізація як багаторівнева оптимізація з енергетичним критерієм | 150 | Energy criterion vs goal uncertainty |
| 3.1.1 Енергетична функція як критерій консистентності | 550 | x*=argmin E (3.1), Langevin (3.2); rejects pixel-space MCMC; figure placeholders |
| 3.1.2 Піраміда ознак як простір станів | 400 | Frozen MobileCLIP pyramid: 8×8×512 / 16×16×256 / 32×32×128; per-level energies E_l |
| 3.1.3 Міжрівнева перехресна увага | 300 | Child queries → parent KV; RoPE-ND scaling + hard 2×2 spatial mask |
| 3.1.4 Односпрямоване керування з відсіканням градієнтів | 500 | Detached-KV; top-down macro→meso→micro sequencing (contradicted by Alg. 3) |
| 3.2 HVEBT каскадна модель | 300+300+550+400+250 | Cascade of per-level energy stages; Table 3.1; reversible transformer core, per-token scalar energy head (3.6); K-step Langevin training loop (3.8–3.13), loss SmoothL1+λ·LPIPS (3.14); detached Pixel Decoder |
| 3.3 Reversible MCMC | 1100 | Algorithm 2; T↔K isomorphism claim; selective backprop of last K_opt Langevin steps; learnable α_l |
| 3.4 Алгоритм прогнозування відеопослідовностей | 300+350+700 | Block-causal masks, 3D-RoPE routing; Algorithm 3 cascaded per-stage MCMC over future features; claims done Python/PyTorch implementation |
| 3.5 Висновки до розділу | 450 | Credits Tasks 5–6; O(K·L·D)→O(D) claim; "published in [4],[5],[6]" |

**Salvageable from Ch.3 regardless of paradigm:** feature-pyramid state space, cross-level cross-attention with RoPE-ND scaling + hard spatial masks, unidirectional stop-gradient control, block-causal masking, 3D-RoPE temporal routing, detached pixel decoder, autonomous (no-text) AR video-prediction framing. **Dies with the paradigm:** scalar energy head, Langevin/MCMC generation loop, Reversible-MCMC Algorithm 2, T↔K isomorphism, energy-gap trajectory criterion as the operational metric.

### РОЗДІЛ 4 — Експериментальне дослідження (~1.8k words written; body ~90% missing)
| Section | ~Words | Summary |
|---|---|---|
| 4.1 Інформаційна база | 1100 | Synthetic: rotating 2D polygons 64×64 (TempVerseFormer base test), 3D polycubes (PyVista, occlusion/non-uniform rotation); real: Moving MNIST, MMNIST-CIFAR, KTH, TaxiBJ; hardware RTX 4060Ti 16GB + "A100 80GB cloud"; AdamW, lr 1e-4 cosine, batch 8–32, 100–300 epochs; claims time-agnostic backprop enables consumer-GPU training |
| 4.2 / 4.3 / 4.4 | 0 | Literal "NAME"/"TEXT" stubs |
| 4.5 Впровадження у навчальний процес | 0 | "TEXT" stub |
| 4.6 Висновки до розділу | 15 | Stub + "published in [4],[5],[10]" (resolves into the alien Isaienkov list) |

### Back matter
| Section | Summary |
|---|---|
| ВИСНОВКИ | "TEXT" stub |
| СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ | Only 5 entries, **all Isaienkov/Mokin** (donor bibliography); none of the ~40+ body citations resolve |
| ДОДАТОК А/Б | Dental panoramic X-ray GAN evaluation — donor content, unrelated |
| ДОДАТОК В | 10 publications, **all Isaienkov's** — must be replaced with Kulyk's list |
| ДОДАТОК Г/Д | Implementation-act headings only, no content |

---

## 2. Map: chapters/sections ↔ the 3 published articles

Reconciled numbering (the draft's internal «Стаття N» labels ≠ the real list order; fix globally):

| Real article | Draft's label | Content | Where it feeds the draft |
|---|---|---|---|
| **[A1] Hierarchy / BLT-adaptation article** (UA professional journal): entropy-based adaptive spatio-temporal decomposition, V-BLT two-stage hierarchical video prediction, procedural data synthesis | «Стаття 2», cited once as [26] | V-BLT architecture, entropy maps, RoPE-ND usage, synthetic data generators | Ch.1 §1.3.2 (BLT motivation); **Ch.2 §2.1, §2.2**; Novelty item 4; §4.1 procedural datasets; abstract result on adaptive tokenization |
| **[A2] TempVerseFormer** (Springer NCA 2025, Scopus, solo): reversible transformer + temporal chaining, time-agnostic BPTT, deterministic 2D-rotation testbed | «Стаття 3» (but §1.4.5 mislabels it «Стаття 2») | Reversible coupling, constant-memory backprop, Lemma 1 | **Ch.2 §2.3, §2.4**; Novelty item 2; Ch.1 §1.1.1/§1.3.3 memory-barrier motivation; §4.1 (2D-shapes = its testbed; consumer-GPU feasibility claim); Ch.3 §3.3 (via now-defunct isomorphism) |
| **[A3] Consistency-criteria article** (UA professional journal): criteria apparatus, "conceptual consistency" of generative models | «Стаття 1» | Conceptual/hierarchical/trajectory consistency taxonomy, CLIP-space metric (1.11) | **Ch.1 §1.4 (esp. 1.4.3–1.4.5)**; Novelty item 3; task 2 |
| **[A4] ACTUAL 4th article (in finalization)** — hierarchy earns its keep at the memory horizon (see header) | «Стаття 4» = the *hoped-for* HVEBT/EBT article (never existed) | Hierarchical latent video model w/ long coarse memory; looping memory-horizon testbed; matched-compute mechanism study incl. EBT two-sided negative; +13% hard re-entry; 200-frame AR rollout; reversibility theory chapter | Replaces **all of Ch.3**, fills **Ch.4 §§4.2–4.4**, Novelty item 1 (rewritten), and upgrades the §2.4 bridge |

Conference theses (6 claimed) are unmapped anywhere in the draft — all six Ch.1 self-citations are [X].

---

## 3. Complete inventory of EBT/"Article-4" dependencies + dispositions

Legend: **REWRITE** = keep location, new content from [A4]; **CUT** = delete; **DEMOTE** = keep as reviewed literature / characterized negative; **RETARGET** = keep mechanism, point it at the new hierarchical model.

### Front matter
1. **UA+EN abstract, goal** — "…на основі ієрархічних енергетичних підходів". → **REWRITE** to the new-topic goal verbatim: efficiency + temporal/spatial consistency via hierarchical modeling in discrete feature spaces.
2. **Abstract Ch.1 summary** — "енергетичні" among analyzed paradigms. → keep (they *are* analyzed), but as reviewed paradigm, not direction.
3. **Abstract Ch.3 summary** — HVEBT, cascaded decomposition, unidirectional control, Reversible MCMC. → **REWRITE**: hierarchical latent model with long-memory coarse level; unidirectional control survives; Reversible MCMC → constant-memory rollout theory + taxonomy.
4. **Abstract Ch.4 summary** — "Доведено, що HVEBT перевершує…" on TaxiBJ/KTH. → **REWRITE** with actual [A4] results: memory-horizon testbed, matched-compute study, +13% hard re-entry (2 seeds, 20k steps), 200-frame rollout win at the flat window's horizon boundary, window ordering 48>18>6, EBT as characterized negative.
5. **Main result 5** (hierarchical energy model, "mathematically guarantees" consistency). → **REWRITE**: hierarchical model with coarse-level long memory; soften "guarantees" to empirically demonstrated + stop-gradient design argument.
6. **Main result 6** (IT combines tokenization + resource control + hierarchical *energy* optimization). → **REWRITE** (energy → hierarchical discrete-latent generation).
7. **Novelty claim 1** ("Вперше… на основі енергетичного критерію"). → **REWRITE**: first hierarchical video-generation method in which the coarse level carries the long memory horizon, validated on a purpose-built re-entry-stratified testbed. (Keep "вперше" defensible.)
8. **Copyright certificate title** «…TempVerseFormer та HVEBT», number placeholder. → **OPEN QUESTION** (Q5): if not yet filed, rename software; if filed under this title, keep the name as a software artifact label only.
9. **Keywords** — "енергетичні моделі, марковські ланцюги". → **REWRITE**: дискретні приховані простори, векторне квантування, авторегресійні моделі, ієрархічна генерація, горизонт пам'яті, консистентність, оборотні перетворення.

### ВСТУП
10. **Topic-justification paragraph** ("Перспективним напрямком… є перехід до енергетичних математичних моделей (EBM)… енергетичний критерій — об'єктивна міра системної консистентності"). → **REWRITE** with the new justification's argument: continuous-latent diffusion limitations (attention memory blow-up, blur on dynamics, instability) → discrete spaces + hierarchy + AR.
11. **Task 5** (hierarchical energy model with controlled Markov optimization + gradient detachment). → **REWRITE**: develop a hierarchical mathematical model of multi-level video generation in discrete feature spaces in which the upper (coarse) level carries long-range temporal memory, with unidirectional inter-level context transfer (gradient detachment).
12. **Methods list** — "Markov process theory and SDEs for Langevin dynamics and MCMC control". → **TRIM/REWRITE** to new-topic methods (probability/statistics for discrete distributions); keep Markov chains only insofar as AR generation is Markovian and the EBT mechanism study is described.
13. **НДР 2 attribution** — candidate developed "ієрархічну енергетичну модель генерації" in the state theme. → **REWRITE** wording (hierarchical generative model); verify against the actual НДР report text (Q6).
14. **Особистий внесок, Article [4]** — "формалізовано ієрархічну енергетичну модель HVEBT з методом односпрямованого керування". → **REWRITE** to the real [A4] contribution: hierarchical long-memory latent video model, memory-horizon testbed + re-entry-stratified protocol, matched-compute mechanism study incl. the two-sided EBT capability study, reversibility theory.

### РОЗДІЛ 1
15. **§1.1** — EBM+MCMC listed among iterative models. → keep (accurate as taxonomy).
16. **§1.1.1** — E(x,u) as global consistency criterion, eqs. (1.2)–(1.3), "буде покладено в основу HVEBT у Розділі 3". → **REWRITE**: keep the optimal-control framing (it is the systems-analysis spine and spec-124 language), but the criterion becomes consistency functionals over hierarchical discrete states; energy/Langevin math **moves** into §1.3.3 (literature) and the Ch.4 mechanism-study section. Forward reference retargeted to the new Ch.3.
17. **§1.2.4 conclusion** — needs "нового ієрархічного енергетичного апарату управління". → light **REWRITE** ("ієрархічного апарату керування в дискретних просторах станів").
18. **§1.3.3** — EBM as "most fundamental paradigm", reversibility as HVEBT foundation. → **DEMOTE + REWRITE**: keep the EBM review + score-matching identity as literature; add one honest paragraph forward-referencing the Ch.4 result that at matched compute/reachable scale the energy-descent objective fails two-sidedly (mis-ordered landscape in one recipe; ordered-but-unreachable in the other) — this is now a *finding of the systems analysis*, not the direction. Keep the resource-efficiency survey (checkpointing/Neural ODE/RevNet) intact.
19. **§1.4.4** — hierarchical consistency "у рамках розроблення архітектури HVEBT". → **RETARGET** to the new hierarchical model; stop-gradient enforcement survives verbatim.
20. **§1.4.5** — trajectory consistency defined via energy-gap ΔE (1.12) for "розробленої енергетичної моделі". → **REWRITE**: redefine trajectory consistency operationally as stability of the AR rollout trajectory (re-entry-stratified error, horizon-boundary behavior); ΔE survives only as the diagnostic used inside the EBT mechanism study (where landscape ordering is exactly what was measured). Fix the Article-2/3 misattribution here too.
21. **§1.5 goal + task 5** — "hierarchical energy approaches". → **REWRITE** as items 1/11.

### РОЗДІЛ 2
22. **§2.1** — entropy (local) vs energy of HVEBT (global) contrast. → **RETARGET**: contrast local entropy with global consistency criteria of the hierarchical model (and note entropy's role in *discrete* byte/token classification — a bridge to VQ framing).
23. **§2.2** — "Системне значення методу для ієрархічних архітектур (HVEBT)": coordinate scaling aligns cross-attention of the cascaded *energy* system. → **RETARGET** to the new model's cross-level attention (mechanism identical).
24. **§2.4 closing bridge** — "Зв'язок із марковською оптимізацією в енергетичних моделях": K-step MCMC chain graph-isomorphic to temporal chain T; time-agnostic backprop = "критичний фундамент HVEBT". → **REPLACE** with [A4]'s reversibility chapter content: exact invertible stochastic recurrence (store-the-noise; grad-cos 1.0 vs BPTT), the constant-memory **rollout** frontier, and the **three-axis taxonomy** whose honest conclusion is that the inner-optimization-loop axis (the very T↔K isomorphism claimed here) **cannot pay vs checkpointing**. This flips the bridge from over-claim to a rigorous scoping theorem — stronger science, and it inoculates the defense.
25. **§2.5 item 5** — integration into MCMC "усуває головний системний бар'єр… відкриваючи шлях до HVEBT". → **REWRITE** per item 24: reversibility enables constant-memory long-horizon *temporal* rollout (where it does pay), with the taxonomy delimiting where it cannot.

### РОЗДІЛ 3 (wholesale)
26. **Chapter title** ("…ієрархічними енергетичними моделями"). → **REWRITE**: «Математичні моделі та методи консистентної ієрархічної генерації відеоданих у дискретних просторах ознак» (or similar within new-topic vocabulary).
27. **§3.1.1** (energy criterion, eqs. 3.1–3.2, figure placeholders). → **CUT** from Ch.3; the diffusion-vs-EBM objective comparison moves to the Ch.4 mechanism-study section as its setup.
28. **§3.1.2 feature pyramid** (frozen MobileCLIP, per-level E_l). → **RETARGET**: keep hierarchical state-space design; reconcile with the actual [A4] architecture (discrete/latent codes; coarse level = long-memory carrier). Drop per-level *energies*; per-level *generators* instead. (Q2: MobileCLIP pyramid vs the actual tokenizer used.)
29. **§3.1.3 cross-level cross-attention** → **KEEP/RETARGET** (context-signal transfer between levels is literally a committed scope item of the new topic).
30. **§3.1.4 unidirectional control + Detached KV** → **KEEP/RETARGET**; **FIX** the top-down vs bottom-up contradiction (§3.1.4 says macro-first; Table 3.1/Algorithm 3 run micro-first; §3.5 says "висхідний"). Decide the true ordering from the [A4] implementation (Q3).
31. **§3.2 HVEBT core** (scalar energy head eq. 3.6, non-zero head init, Langevin training loop 3.8–3.13, SmoothL1+LPIPS). → **REPLACE** with the [A4] model core (hierarchical latent denoiser/predictor, single-step objective + AR rollout). The energy head + Langevin loop text **moves, condensed,** into the Ch.4 EBT mechanism-study section as the description of what was tested (S1/S2 recipes).
32. **§3.3 Reversible MCMC (Algorithm 2)** → **CUT** as a contribution (it is exactly the inner-loop axis that the [A4] taxonomy proves cannot pay vs checkpointing — and as drafted it stores all K states anyway, i.e. it *is* checkpointing). Salvage: one paragraph in the reversibility section as the negative branch of the taxonomy.
33. **§3.4 + Algorithm 3** (cascaded per-stage MCMC video prediction; block-causal masks; 3D-RoPE; "implementation done"). → **REWRITE**: same skeleton (autonomous AR video prediction, block-causal masking, 3D-RoPE, per-frame hierarchy) with the [A4] generation procedure replacing per-stage MCMC; memory claims re-derived for AR rollout.
34. **§3.5 conclusions + refs [4],[5],[6]** → **REWRITE**; fix O(K·L·D)→O(D) vs O(K·T·L·D)→O(T·L·D) inconsistency by re-deriving for the actual algorithm; publications retargeted to [A4] + real theses.

### РОЗДІЛ 4
35. **§4.1** — datasets chosen to verify HVEBT/TempVerseFormer; TaxiBJ as "key real benchmark"; A100 claim. → **REWRITE/EXTEND**: add the **looping memory-horizon testbed with re-entry-stratified evaluation** as the purpose-built primary instrument (it operationalizes "trajectory consistency" measurement); keep 2D-shapes (=[A2] testbed) and MMNIST/KTH if runs exist; TaxiBJ and A100: keep only if real results/receipts exist (Q4).
36. **§§4.2–4.4 "NAME" stubs** → **WRITE NEW** from [A4] results — this is where the actual 4th article's experimental sections slot in one-to-one: 4.2 matched-compute mechanism study (regression vs diffusion vs EBT; two-sided EBT capability study as an honest characterized negative), 4.3 single-step long-coarse-memory win (+13% hard re-entry, seed/step analysis), 4.4 200-frame AR rollout (hier-long vs token-matched flat; horizon-boundary switch-on; window ordering 48>18>6) + constant-memory rollout demonstration.
37. **§4.6 refs [4],[5],[10]** → **REWRITE** with real list.
38. **Implementation acts** (ORNETIKA 14.10.2025, VNTU 19.09.2025) mention only TempVerseFormer + entropy tokenization, *not* HVEBT. → **KEEP** — already consistent with reality; do not add unverified HVEBT claims. Appendix Г/Д content to be inserted.

**Bottom line of the inventory:** ~38 dependency sites. Ch.3 is ~85% replace (with ~4 sections' mechanisms salvaged), abstracts/Вступ ~50% rewrite, Ch.1 ~20% targeted rewrites, Ch.2 ~10% (one bridge + conclusions), Ch.4 is net-new writing that [A4] fully funds.

---

## 4. New-topic gap analysis

New scope (verbatim constraints): discrete feature spaces (VQ-VAE-class), hierarchical abstraction-level decomposition, autoregressive prediction with inter-level context transfer, software-implemented system, experimental comparison on efficiency/stability/consistency; diffusion = motivating limitation; reversibility and EBT **not mentioned** in the justification.

### Fits as-is (minor retitling only)
- Ch.1 §§1.1, 1.1.2, 1.2–1.2.4 (systems framing, decompositions, uncertainty taxonomy) — the hierarchical decomposition section is now the topic's core.
- Ch.1 §1.4 consistency taxonomy (incl. [A3] conceptual consistency) — the new topic is literally "consistent modeling"; the taxonomy *is* the criteria apparatus.
- Ch.2 §2.1 V-BLT and §2.2 RoPE-ND — adaptive decomposition + positional coherence serve "optimize attention / reduce computational load" (committed scope item c). Bonus: V-BLT's 256-way byte *classification* decoder is already a discrete-distribution model — use it as the conceptual on-ramp to VQ/discrete latents.
- Ch.4 §4.1 synthetic datasets + protocol skeleton.

### Needs reframing (content survives, spine changes)
- Ch.1 §1.1.1: optimal-control formalization stays; energy-as-THE-criterion goes.
- Ch.1 §1.3: reorder the paradigm review to match the new argument: diffusion (SOTA quality, continuous-space limits) → discrete AR hierarchy (the remedy; **add VQ-VAE/VideoGPT/latent-token literature — currently absent from the draft**) → EBM (reviewed; later characterized negative).
- Ch.2 §§2.3–2.4 TempVerseFormer/reversibility: not mentioned in the justification, but defensible under "reduce computational load" and task-4-style resource efficiency; must be re-homed as the constant-memory **rollout** enabler (temporal axis), with the taxonomy honestly closing the inner-loop axis. Needs supervisor sign-off (Q1b).
- Ch.3 wholesale → the [A4] model chapter (see §6).
- Novelty items 1 and (partly) 2 restated in new-topic vocabulary.

### Falls OUT of scope
- HVEBT as engine: scalar energy criterion, Langevin/MCMC generation, Reversible MCMC (Alg. 2), energy-gap ΔE as the operational trajectory metric, the T↔K isomorphism bridge. Survives only as: (i) reviewed literature in §1.3.3, (ii) the **characterized-negative mechanism study** in Ch.4, (iii) one taxonomy paragraph in the reversibility section.
- Continuous-latent diffusion **as the primary paradigm** per the justification's own framing. ⚠️ **TENSION (top risk):** [A4]'s winning generator is *hierarchical latent video diffusion*. Reconciliation path: the dissertation's system-level claims (hierarchy, discrete/compact latent states, AR rollout across frames, coarse-level long memory, inter-level context transfer) are paradigm-level; per-frame generation uses an iterative refinement core, and the matched-compute study *selects* it empirically (diffusion > regression > EBT). The justification's anti-diffusion clause targets *monolithic continuous pixel/latent-space video diffusion with long-range attention*, which is still what Ch.1 criticizes. This needs explicit framing and the author's/supervisor's confirmation (Q1a) — it is the single biggest narrative decision of the rewrite.
- Donor-template back matter (Isaienkov publications/references/appendices А–В) — remove entirely, regardless of scope.

### MISSING (must be created)
1. VQ-VAE / discrete-latent-space literature + math (van den Oord 2017; VideoGPT, Yan 2021; token-based video AR) — nowhere in the draft despite being the committed paradigm.
2. The memory-horizon looping testbed: formal description, re-entry stratification, why it is the right instrument for "trajectory/temporal consistency" measurement (this is also a systems-analysis contribution — a *criterion-driven testbed*).
3. All of Ch.4 body (4.2–4.6), ВИСНОВКИ, real reference list (~145 sources), abbreviation list, appendices (publication list, acts, code certificate).
4. Reconciliation of the [A4] architecture description with Ch.3's MobileCLIP-pyramid text (what the final model actually uses for tokenization/levels).
5. Six conference theses: identify and slot the [X] citations.
6. Software-implementation chapter/sections proving the "designed and software-implemented system" deliverable (repo: `hvp_dlh/`, `hvp_rev/`, testbed generators `scripts/gen_looping.py` — map code to the IT description).

---

## 5. UA–EN terminology table (canonical for all rewrites)

| Українською | English | Notes |
|---|---|---|
| консистентність (просторова / об'єктна / часова / стильова / концептуальна / ієрархічна / траєкторіальна) | consistency (spatial / object / temporal / style / conceptual / hierarchical / trajectory) | taxonomy from [A3] + Ch.1; keep all seven |
| ієрархічна генерація відеоданих | hierarchical video data generation | NEW topic term — use everywhere the old text said «ітеративна генерація» when referring to the contribution |
| дискретний простір ознак / прихований простір | discrete feature/latent space | new-topic core |
| векторно-квантований автокодувальник (VQ-VAE) | vector-quantized autoencoder (VQ-VAE) | to be introduced in §1.3.2 |
| авторегресійне прогнозування | autoregressive prediction | VideoGPT-style |
| передача контекстних сигналів між рівнями | inter-level context-signal transfer | verbatim scope item; = cross-level conditioning |
| міжрівнева перехресна увага | cross-level cross-attention | mechanism (survives from Ch.3) |
| односпрямоване керування з відсіканням градієнтних зв'язків | unidirectional control with gradient detachment (stop-gradient) | survives; Detached KV |
| горизонт пам'яті | memory horizon | NEW ([A4]); define at first use |
| циклічний тестовий стенд / тестбед із повторними входженнями | looping testbed with re-entry(-stratified) evaluation | [A4]; «повторне входження» = re-entry |
| стратифікація за складністю повторного входження | re-entry-stratified evaluation | [A4] protocol |
| авторегресійне розгортання (розгортка) на T кадрів | T-frame autoregressive (AR) rollout | [A4]; «200-кадрове розгортання» |
| вікно контексту / довжина вікна | context window / window length | 48>18>6 ordering result |
| грубий (верхній) рівень із довготривалою пам'яттю | coarse level carrying long-range memory | the [A4] spine |
| зіставлені за обчисленнями (при однаковому бюджеті обчислень) | matched-compute | mechanism-study protocol |
| охарактеризований негативний результат | characterized negative (result) | status of EBT in the dissertation |
| енергетичні моделі (EBM); енергетична функція; енергетичний ландшафт | energy-based models; energy function; energy landscape | now literature/mechanism-study vocabulary only |
| стабільно неправильно впорядкований енергетичний ландшафт | stably mis-ordered energy landscape | EBT S2 finding |
| недосяжність шляхом градієнтного спуску | descent-reachability failure | EBT S1 finding |
| динаміка Ланжевена; марковські ланцюги Монте-Карло (MCMC) | Langevin dynamics; MCMC | mechanism study only |
| оборотні перетворення / оборотні блоки | reversible (invertible) transformations / blocks | [A2] |
| часонезалежне зворотне розповсюдження помилки | time-agnostic backpropagation | [A2]; constant-memory BPTT alternative |
| зворотне розповсюдження помилки в часі (BPTT) | backpropagation through time (BPTT) | |
| точна оборотна стохастична рекурентність («збереження шуму») | exact invertible stochastic recurrence (store-the-noise) | [A4] reversibility chapter |
| контрольні точки (градієнтні) | gradient checkpointing | fix Table 2.1: O(√T·L·D) |
| трирівнева таксономія осей оборотності | three-axis reversibility taxonomy | [A4]: temporal-rollout axis pays; inner-optimization-loop axis cannot beat checkpointing |
| ентропійні показники / піксельні карти ентропії | entropy metrics / per-pixel entropy maps | [A1]/V-BLT |
| адаптивне пакування (токенізація) | adaptive patching (tokenization) | [A1] |
| N-вимірне ротаційне позиційне кодування (RoPE-ND / 3D-RoPE) | N-dimensional rotary positional embedding | [A1]/Ch.2 |
| блочно-каузальна маска уваги | block-causal attention mask | survives |
| макро- / мезо- / мікрорівень абстракції | macro / meso / micro abstraction level | FIX ordering globally: macro = HIGHEST abstraction (semantics), micro = lowest (pixels) — §1.1 currently inverts it |
| псевдочас k / фізична вісь часу t | pseudo-time k (per-frame generation steps) / physical frame axis t | keep the 2-D temporal grid |
| невизначеність: інформаційна / структурно-алгоритмічна / обчислювальна / цільова | uncertainty: informational / structural-algorithmic / computational / target | Ch.1 taxonomy, keep |
| розкриття невизначеності | uncertainty resolution/disclosure | spec-124 idiom |
| інтелектуальна інформаційна технологія | intelligent information technology | NEW topic says «інтелектуальні технології» — align all headings |
| ідентифікація параметрів | parameter identification (training) | genre term, keep |
| експозиційне зміщення (exposure bias) | exposure bias | keep EN in parentheses |
| зіткнення з межею горизонту вікна | (win) turning on at the window's horizon boundary | Ch.4 rollout finding |
| форма ієрархії | hierarchy shape | shapes program 2026-07-10; §4.4 umbrella |
| просторова / часова абстракція | spatial / temporal abstraction | the two downsampling axes |
| крок (проріджування) грубого потоку | coarse-stream (temporal) stride | k_t; slot anchors at block ends |
| сімейство конфігурацій | configuration family | (k_s, k_t)-indexed; edges are per-family |
| закон краю смуги | band-edge law | edge(family) = span cost + fine floor |
| потік контекстних токенів | context-token stream | (resolution, stride, count) triple |
| нейтральність глибини щодо ємності остова | backbone-capacity-neutral depth | +1.4% per-level I/O params disclosed; vs CW-VAE's full-cell-per-level |
| засліплення рівня (потоку) | level (stream) blinding | CW-VAE §5.1 analog, causal version |
| стовпець контролю | control column | standard-MMNIST long-horizon null |
| перерозподіл бюджету | budget re-allocation | what depth-3 does on single-scale data |

---

## 6. Proposed NEW chapter structure

**НОВИЙ РОЗДІЛ 1 — Системний аналіз процесів ієрархічної генерації відеоданих та критеріїв їх консистентності** *(old Ch.1, ~75% survives)*
Systems framing of video generators as complex control systems; structural/hierarchical/temporal decomposition; four-uncertainty taxonomy; paradigm review reordered as diffusion (SOTA, continuous-space limits = motivation) → discrete latent + hierarchical AR (VQ-VAE/VideoGPT — NEW subsection replacing the "direction" role of old §1.3.3) → EBM (reviewed, resource barrier noted); consistency-criteria apparatus (spatial/object/temporal/style/conceptual/hierarchical/trajectory) with trajectory consistency redefined via rollout/re-entry metrics; goal + 7 tasks restated in new-topic vocabulary. Maps: old 1.1–1.5 minus energy spine; old §1.3.3 demoted; [A3] anchors §1.4.

**НОВИЙ РОЗДІЛ 2 — Математичні методи оцінювання невизначеності та ресурсоефективного моделювання багатокрокових відеопроцесів** *(old Ch.2, ~90% survives)*
V-BLT entropy-based adaptive decomposition (uncertainty estimation, discrete byte-classification decoder as the bridge to discrete states) [A1]; RoPE-ND with cross-level coordinate scaling [A1]; TempVerseFormer reversible blocks + temporal chaining [A2]; time-agnostic backprop with Lemma 1 [A2]; NEW closing section replacing the MCMC bridge: exact invertible stochastic recurrence (store-the-noise, grad-cos 1.0 vs BPTT), constant-memory AR-rollout frontier, and the three-axis taxonomy that scopes where reversibility pays (temporal rollout) and proves where it cannot (inner optimization loops vs checkpointing) [A4-reversibility]. Maps: old 2.1–2.5; §2.4 bridge replaced; Table 2.1 checkpointing entry fixed.

**НОВИЙ РОЗДІЛ 3 — Ієрархічна математична модель генерації відеопослідовностей із довготривалою пам'яттю грубого рівня** *(old Ch.3 replaced by [A4] model chapters; ~4 mechanisms salvaged)*
The core contribution: hierarchical latent video model in discrete/compact feature spaces where the coarse (macro) level carries the long memory horizon; per-frame iterative-refinement generation core selected by the matched-compute mechanism analysis (design rationale here, evidence in Ch.4); inter-level context-signal transfer via cross-level cross-attention with RoPE-ND scaling and unidirectional stop-gradient control (salvaged §§3.1.3–3.1.4, ordering fixed per implementation); block-causal masking + 3D-RoPE spatio-temporal routing; the autonomous AR video-prediction algorithm (salvaged §3.4 skeleton, MCMC replaced); detached pixel decoder; formal definition of the **memory-horizon looping testbed and re-entry-stratified evaluation protocol** as the criterion-driven measurement instrument (systems-analysis deliverable); memory/complexity derivations for constant-memory rollout. Maps: old 3.1.2–3.1.4, 3.4 retargeted; 3.1.1, 3.2 energy core, 3.3 cut/moved.

**НОВИЙ РОЗДІЛ 4 — Експериментальне дослідження та практичне застосування** *(old §4.1 + [A4] results fill 4.2–4.6)*
4.1 Information base: synthetic 2D-shapes [A2] + 3D polycubes + looping memory-horizon testbed (primary) + MMNIST(-CIFAR)/KTH as available; hardware/protocol. 4.2 Matched-compute mechanism study: regression vs diffusion vs energy-descent on the shared backbone; the two-sided EBT capability study (S2: stably mis-ordered landscape at paper-xxs scale/20k steps; S1: correctly-ordered landscape but descent-reachability failure) presented as a characterized negative with the ΔE/landscape-ordering diagnostics — the honest systems-analysis justification of the chosen core. 4.3 Single-step evaluation: long-coarse-memory win (+13% hard re-entry, 2 seeds, emerges only at 20k steps — training-budget analysis). 4.4 Long-horizon 200-frame AR rollout: hier-long vs token-matched flat control (win switches on exactly at the flat window's horizon boundary); window-length ordering 48>18>6; constant-memory rollout demonstration tying back to Ch.2. 4.5 Implementations (ORNETIKA act, VNTU educational process — as per existing acts: TempVerseFormer + entropy tokenization). 4.6 Conclusions. Maps: old 4.1 extended; "NAME" stubs replaced by [A4] sections; TaxiBJ kept only if runs exist (Q4).

**ВИСНОВКИ** — 7 results restated against the 7 (rewritten) tasks; honest scoping: hierarchy validated *where the memory horizon demands it*; EBT characterized negative; reversibility scoped by taxonomy.

**Novelty items, rewritten:**
1. (вперше) Hierarchical video-generation method with a coarse level carrying the long memory horizon, + the re-entry-stratified memory-horizon evaluation methodology [A4].
2. (вперше) Time-agnostic backprop via reversible transforms, extended with exact invertible stochastic recurrence and the three-axis applicability taxonomy [A2]+[A4].
3. (удосконалено) Consistency-criteria apparatus (conceptual/hierarchical/trajectory), with trajectory consistency operationalized by rollout/re-entry metrics [A3]+[A4].
4. (удосконалено) Entropy-based adaptive spatio-temporal decomposition with RoPE-ND [A1].

---

## 7. Open questions for the author

**Q1 (blocking, narrative):** (a) The approved justification commits to *discrete VQ-VAE + autoregressive (VideoGPT-style)* and casts diffusion as the limitation — but [A4]'s engine is hierarchical **latent diffusion** with AR rollout across frames. Confirm the reconciliation frame: "hierarchical AR generation over compact/discrete latent states; per-frame refinement core chosen empirically (diffusion beat regression and EBT at matched compute)" — is Prof. Mokin on board with this reading of the topic, or must the final model literally use VQ token AR? (b) Does the reversibility chapter (absent from the justification) stay as a full contribution under "reduce computational load," per his view?
**Q2 (architecture ground truth):** What does the final [A4] model actually use for the level pyramid/tokenization — the frozen MobileCLIP pyramid described in old Ch.3, a VQ/discrete tokenizer, or the conv-AE latents from the repo? Ch.3's rewrite copies the real implementation, so I need the definitive stack (and the true level ordering: coarse-first or fine-first).
**Q3 (publications):** The real, final list of 10 items — exact bibliographic entries for [A1] (venue/year/pages), [A2] (Springer NCA 2025 — is it Scopus **Q1** as Вступ claims, or should the claim be softened to "Scopus-indexed"?), [A3] (venue), [A4] (target venue/status by defense), and the 6 conference theses (incl. whether DSMSI-2023 is real). Does the count 3 UA-professional + 1 Scopus + 6 theses still hold with [A4] added?
**Q4 (evidence inventory):** Which of the claimed experiments actually exist or will exist by defense: TaxiBJ (any runs at all?), KTH, MMNIST-CIFAR, A100 cloud runs? Anything without receipts gets cut from §4.1 and the abstract.
**Q5 (formalities):** Copyright-certificate status: filed or not? If not, drop "HVEBT" from the software title; if filed, exact number + title. Also the two НДР numbers, and whether the НДР-2 report text says "енергетичну модель" (needs consistency with the rewrite).
**Q6 (EBT visibility):** How prominent may the EBT characterized negative be? Options: (a) full §4.2 subsection with both recipes' diagnostics (recommended — it's real science and justifies the core choice), or (b) compressed to a paragraph + appendix, keeping energy models purely as literature. Supervisor/defense-culture call.
**Q7 (scope of Ch.4 datasets):** Is the looping memory-horizon testbed acceptable to the committee as the *primary* benchmark (purpose-built, synthetic), with KTH/MMNIST as secondary? This determines how much §4.1 argues for testbed validity.

---

## 8. AUTHOR DECISIONS (2026-07-03) — all Q1–Q7 resolved

- **Q1 RESOLVED:** paradigm = **continuous-latent hierarchical diffusion** (author empirically found
  hierarchical VQ video models unstable to train — usable as motivation in Ch.1). DROP the
  "diffusion as the limitation" thesis; new motivation spine = flat long-context attention memory
  blow-up + consistency degradation -> hierarchy + coarse-level memory + reversibility as remedies;
  per-frame refinement core selected empirically (mechanism study). EBT included as the full
  characterized study. Reversibility = memory-optimization emphasis, stays a contribution.
- **Q2 RESOLVED:** drop MobileCLIP pyramid AND VQ-VAE from Ch.3 entirely; architecture = the
  MilliVid-style stack as implemented (2-level conv-AE + joint-attention hierarchical DiT +
  AR window rollout; hvp_rev/rhvd.py).
- **Q3 RESOLVED:** publications list unchanged for now.
- **Q4 RESOLVED:** Ch.4 evidence: TempVerseFormer receipts exist for 2D/3D rotations, MMNIST,
  MMNIST-CIFAR, TaxiBJ, KTH — keep as [A2] results. New model: looping memory-horizon testbed
  PRIMARY + OpenSTL MMNIST bridge (+ MMNIST-CIFAR as stretch); NO new TaxiBJ/KTH commitment.
- **Q5 RESOLVED:** no copyright certificate filed -> drop "HVEBT" naming everywhere, freely.
- **Q6 RESOLVED:** option (a) — full §4.2 EBT subsection with both recipes' diagnostics.
- **Q7 RESOLVED:** looping testbed as primary benchmark accepted; additionally include the
  TempVerseFormer-style memory-benchmark-on-noise + GPU-memory-trend projection device for the
  constant-memory rollout claim (Panel A frontier + projection).
