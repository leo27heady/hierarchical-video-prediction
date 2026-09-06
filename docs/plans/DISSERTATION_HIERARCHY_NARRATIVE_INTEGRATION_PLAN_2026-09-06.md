# Plan for a coherent hierarchy study across Chapters 3 and 4

Date: 6 September 2026. Scope: assessment and implementation plan requested by the author; this turn does not rewrite the manuscript or launch experiments.

This plan supersedes the local-insertion narrative strategy in the [5 September remediation plan](DISSERTATION_REMEDIATION_AND_MINIMAL_POST_A4_PLAN_2026-09-05.md). Its code corrections, evidence safeguards and minimal experiment budget remain applicable. It builds on the [refactoring report](../reviews/DISSERTATION_REFACTORING_AND_FINAL_REVIEW_2026-09-05.md), but corrects that report's overly favorable assessment of narrative integration.

## 1. Assessment

The author's concern is justified. The dissertation has the ingredients for a coherent research argument, but the arrangement still exposes the editing history:

- Chapter 3, §§3.6.3–3.6.8, mostly enumerates configurations. It explains what varies more fully than why each comparison is necessary, what mechanism it distinguishes, and what remains confounded.
- Chapter 4 places the FRT table and examples in a subsection about completed trajectories after the system-level decision rule. Evidence should precede the synthesis that interprets it.
- The titles “Попередній прототип”, “Контрольоване інтеграційне розширення”, “Протокол і незаповнені результати остаточної архітектури”, and the software paragraph beginning “Нова експериментальна гілка” foreground development chronology.
- Running and planned families receive a collective pending sentence, not individual evidence destinations. The integrated table has result slots, but the full program does not.
- The same subsection number **4.9.3 occurs twice**. The current four gates did not detect this; their passing status is insufficient for structural acceptance.
- Chapter 3 includes a calibration account inside a hypotheses section, and Chapter 4 places its counterpart under generator selection. A common optimization-budget protocol needs its own methodological home.

Recommendation: retain one dissertation argument about allocating limited representation and computation across hierarchical aspects. Give the systematic design study an explicit methods subchapter and a matching results subchapter. Explain the integrated predictor as the implementation in which these resource choices interact.

One qualification is essential: conceptual unity must not imply that all experiments use the same architecture, forward computation, dataset, metric aggregation or evidential status. Describe these distinctions by scientific role and protocol, without “old/new” narrative labels. Do not retrospectively call an explanatory hypothesis preregistered when it was formulated after results were available.

## 2. Argument that will organize the text

The central question is: **under a finite resource budget, what information should be retained, at what resolution and age, and how can the resulting predictor be trained and evaluated reliably?**

The argument proceeds through six related dimensions:

1. **Capacity:** how much context is retained and where additional capacity ceases to help.
2. **Allocation:** how a fixed capacity is divided between temporal reach and spatial detail.
3. **Representation:** how hierarchy depth and overlap change what the same token budget represents.
4. **Placement:** where detail is spent in space and time, and whether content-adaptive selection is operationally useful.
5. **Computation:** whether activation reconstruction preserves the computation and reduces measured training memory.
6. **Validity:** whether observations survive independent evaluation, changed evaluation coverage and, conditionally, recursive rollout.

Chapters 1–2 motivate these dimensions and establish their concepts. Chapter 3 defines mechanisms and discriminating tests. Chapter 4 answers the questions in that order, then gives the system-level synthesis. Negative and mixed results define the limits of the method and remain part of the argument.

## 3. Proposed Chapter 3 structure

Retain §§3.1–3.5, with targeted transitions and a compact design-axis map at §3.1. Consolidate shared metrics and inference rules in §3.5 rather than repeating them beside each table.

| Section | Proposed title / purpose | Source and change |
|---|---|---|
| 3.6 | **Методика дослідження структури ієрархічного контексту** | Rebuild current §3.6 around mechanisms, not a campaign inventory. |
| 3.6.1 | Дослідницькі питання та контрольовані чинники | Connect the six dimensions to H1–H5 where applicable; distinguish registered hypotheses from explanatory questions. |
| 3.6.2 | Спільні умови навчання та оцінювання | Fixed 30k budget, model/data controls, seeds, validation versus confirmation, reference to metric definitions. No convergence claim. |
| 3.6.3 | Загальний бюджет і розподіл між рівнями | FRT followed by ALC: first vary capacity, then distinguish allocations at the same capacity. |
| 3.6.4 | Часова підтримка та просторово-часова деталізація | CEX and STS: adding coarse history versus redistributing fixed detail. Explicitly identify the different controls. |
| 3.6.5 | Глибина ієрархії та перекриття представлень | LVL and OVR with a shared geometric example and selection dependencies. |
| 3.6.6 | Незалежна перевірка та діагностика механізму | Registered held-out evaluation, mechanism roles, full-validation robustness and conditional rollout. |
| 3.7 | **Ієрархічний предиктор з адаптивним пакуванням і реверсивним ядром** | Replace chronological integration framing with a functional architecture description. |
| 3.7.1–3.7.5 | Representation, physical packing, routing, coordinates, reversible core/decoder | Reuse the implemented contracts and architecture figure. Explain how fixed allocations motivate studying content-dependent membership without asserting that their results validate it. |
| 3.7.6 | Протокол перевірки коректності, ресурсів і якості | Separate numerical parity, resource profiling, bounded development controls, and four full-model runs. Link every test to an inferential limit. |
| 3.8 | Висновки до розділу | Summarize the questions that Chapter 4 will answer; no result promises. |

The fixed-region predictor and temporal-recurrence experiment remain as a compact subsection or comparison box identified by their actual mechanisms. Move lengthy prototype-specific derivation to an appendix if it obstructs the main architecture account, preserving equations, source attribution and cross-references. Do not delete its negative evidence or imply its forward function matches the internal reversible-core comparison.

For **every experiment family**, write approximately 200–350 substantive Ukrainian words following one sequence: motivating limitation → competing explanations → controlled and varied factors → worked example → measured outcome → interpretation of favorable, null or adverse evidence. Link to shared protocol details instead of copying them. The integrated verification protocol needs approximately 500–700 words because correctness, resources and predictive quality are distinct questions.

## 4. Proposed Chapter 4 structure and movement map

Keep §§4.2–4.7 as the established mechanism, reach/fidelity, robustness and boundary evidence, with stronger transitions and explicit protocol identifiers. Move common schedule discussion from current §4.2.4 to §4.1. Do not pool incompatible experiments numerically.

| Section | Proposed title / contents |
|---|---|
| 4.1 | Експериментальна карта та правила порівняння: question-to-evidence map, datasets, budgets, training schedule, evaluation populations and evidence levels. |
| 4.8 | **Вплив структури ієрархії на якість і ресурси**: the dedicated structural experiment results subchapter. |
| 4.8.1 | Budget and allocation: FRT and ALC, with direct references to §3.6.3. |
| 4.8.2 | Coarse-context extension and detail placement: CEX and STS. |
| 4.8.3 | Hierarchy depth and overlap: LVL and OVR. |
| 4.8.4 | Held-out evaluation, full-validation robustness and mechanism diagnostics; conditional rollout as a clearly labeled diagnostic block. |
| 4.9 | **Якість і ресурсні властивості ієрархічного предиктора**: representation/recurrence feasibility evidence, same-computation parity and profiling, bounded controls, two-task integrated quality. |
| 4.10 | Правило вибору методу на системному рівні: synthesize evidence from all preceding sections, preserving unresolved branches. |
| 4.11 | Набір програмних засобів та практичне застосування: concise reproducibility description; operational run inventory in appendix. |
| 4.12 | Обмеження та загрози валідності, followed by chapter conclusions. |

Move current FRT table 4.10 and qualitative figure 4.15 from current §4.9.3 to proposed §4.8.1. Move current §4.8 into proposed §4.9 and restructure it by the property tested. Remove the “completed trajectories” subsection as a narrative unit: dates, hashes and execution status belong in captions/provenance, not the main argument.

Introduce each results subsection with its question and a short reminder of the distinguishing comparison. Then present evidence or its explicit slot, interpretation and a bridge to the next question. A pending subsection can fully explain design and interpretation but must stop short of an empirical conclusion.

Example transition in Ukrainian: «Залежність помилки від загального бюджету не визначає, як саме цей бюджет слід розподіляти. Для відокремлення ефекту місткості від ефекту розподілу далі зіставляються пакети однакового обсягу з різною часткою грубого та детального контексту.»

## 5. Complete experiment-to-evidence matrix

The IDs below are stable editorial keys, not final table numbers. The manuscript uses descriptive names; operational IDs remain available for traceability.

| Key | Scientific necessity and control | Required evidence destination |
|---|---|---|
| FRT | Capacity can improve any representation. Compare flat, linear-age and geometric-age policies over six budgets, both seeds; do not assume a monotone crossover. | Quality-versus-budget plot with individual seeds and a compact table; retain all 36 verified measurements, separating their validation protocol from confirmation. |
| ALC | Equal token counts can encode different temporal reach/detail. Seven policies at B=1792; selection and nonflat dependency reference defined explicitly. | Seven-policy, two-seed table; allocation diagram; selected-role slots remain empty until verified. |
| CEX | Does older coarse history add useful information when recent detail is fixed? Six detail frames fixed; coarse ages extend support. Budget and computation increase too. | Six-condition, two-seed table; error versus reach and token/byte cost. Do not call this a fixed-resource causal comparison. |
| STS | At fixed coarse support and global detail quota, is concentrated spatial detail preferable to wider temporal coverage? K=64/32/16/8 with recent/uniform/geometric placement. | 4×3 policy matrix with both seeds available; support diagram; physical-plan aliases linked rather than counted as independent designs. |
| LVL | Does another scale help after redistributing the same total budget? One/two/three levels; reuse references and resolve dependent three-level design from ALC. | Three-design table with realized allocations, two seeds and reference provenance. State that changing levels also redistributes representation. |
| OVR | Is same-time coarse/fine information complementary, redundant, or better represented residually? Nested/exclusive/residual at fixed budget. | Three-design table and overlap schematic. Explain that exclusive support changes coverage and residual encoding changes features. |
| CONF | Do development-selected roles retain their behavior on protected data? Registered best-flat, best-hierarchy and memory-Pareto roles, 500 clips, both seeds. | Role table, paired differences and uncertainty, horizon-4 metric, deduplicated checkpoint references. The 5% selection tolerance is not automatically a noninferiority margin. |
| MECH | What behavior explains success and failure? Registered best-flat, best-hierarchy and failure-hierarchy roles. | Compact mechanism summary and paired examples under a declared selection rule; state which diagnostics the finalizer actually supplies before fixing detailed columns. |
| VAL300 / E1 | Does limited validation coverage materially affect the frozen shortlist's descriptive behavior? At most three configurations × two seeds, all 300 validation clips. | Robustness table in §4.8.4; no winner reselection, no independent-confirmation label. |
| ROLL64 / E2 | Does the observed short-window behavior persist under recursive feedback? Conditional frozen pair × two seeds, 100 diagnostic clips, horizons 4/16/32/64. | Conditional plot/table and example slots in §4.8.4; fully explain feedback, initial context and failure modes. If omitted, mark not performed and narrow long-horizon claims. |
| PARITY / E3 | Does changing differentiation preserve the same computation? Eager/checkpoint/reversible, both dataset shapes, FP32 and intended precision. | Output/input/parameter-gradient discrepancy table, tolerances and pass/fail slots. Preserve already verified CPU evidence; CUDA cells remain pending. |
| RESOURCE / E3 | Do logical savings translate into measured device savings? Same weights/batch/precision; inference and training, whole-model and core-operation scope. | Main whole-model peak/time table; appendix covers flat/static/hybrid packing × three execution modes and diagnostic storage/saved-tensor detail. |
| SCREEN / E4 | Is the recipe trainable and measurable before full runs? Four bounded trajectories: full model on both tasks and flat/static controls on Moving MNIST. | Four-row development table: loss/gradient health, throughput, budget, configuration and status. Not a full-training ablation. |
| QUALITY / E5 | Does the locked integrated predictor work on both task structures? INT_MM_S00/S01 and INT_TB_S00/S01, 10→10 and 4→4. | Four-row quality table, per-horizon appendix, paired visual examples for each dataset with appropriate units; no invented image-like semantics for TaxiBJ channels. |
| COMP / E6 | Is a compatible comparator available for a stronger quality claim? Reuse compatible checkpoints first; no added training by default. | Conditional comparator/provenance row; if absent, explicitly limit the claim to feasibility. |

As inspected on 6 September, the controller records wave-one entry 57/80, an STS run. That is an operational observation, not verification of every preceding result. Refresh and hash completed artifacts during implementation before adding measurements beyond the existing frozen FRT snapshot. Integrated confirmation remains prelock with launch authorization false. This plan changes neither state.

## 6. Placeholder contract

Give every matrix row an explicit destination now, including dependent designs, post-training confirmation/mechanism analysis, E1–E5 and conditional E2/E6. An aggregate sentence saying that other work remains pending is insufficient.

- Use one inventory containing key, methods section, results section, figure/table slot, configuration/role, seed, metric, status, dependency, artifact and admissible claim.
- Distinguish **заплановано**, **виконується**, **очікує залежності**, **завершено, не перевірено**, **перевірено**, and **умовний дослід**. Show concise status in the draft; keep changing operational details in the inventory.
- Numeric cells use **очікується** or an em dash with an explicit caption defining it as unavailable, never zero. Existing verified cells stay populated.
- Figure slots contain a labeled frame with the intended variables, units, comparison and required data. No hypothetical favorable curves, synthetic predictions or simulated measurements may resemble results.
- Pending interpretation states what different outcomes would mean. It does not assert which outcome occurred.
- Conditional slots are retained in the working draft. At release they become measured evidence or a concise statement that the diagnostic was not performed and the corresponding claim is excluded.
- Derive pending counts from the inventory rather than preserving the present count of 14. The count must expand to reflect actual coverage.

Keep main-text presentation compact: three paired-family result blocks for the six structural families, one confirmation/robustness block, and integrated correctness/resource/quality blocks. Exhaustive run tables and profiling combinations belong in appendices; each family still receives individual motivation and interpretation.

## 7. Figures, examples and formulas

Add **at most two explanatory multipanel figures**, generated as repository-native SVG/vector diagrams:

1. **Allocation and temporal support:** shared age axis, coarse/fine token costs, fixed-budget ALC contrast, CEX extension and STS placement. Worked exact example: B=1792 is 28×64 flat tokens, or 56×16 coarse + 896 detail tokens; STS uses 96×16 coarse + 256 detail tokens spread over 4/8/16/32 positions. Mark nested support explicitly so counts are not mistaken for disjoint coverage.
2. **Levels and overlap:** one/two/three-level alternatives beside nested/exclusive/residual encoding of the same illustrated time point. The third-level example reserves 96×4=384 tokens, leaving 1408; the remainder is dependency-selected, not invented in the drawing. Show residual as a feature operation, not a claim of smaller error.

Reuse/refine the architecture figure, resource-axis figure and decision flow. Place the verified qualitative panel beside the corresponding budget comparison. Reserve a compact multipanel results figure for budget/extension/placement; populate only supported panels and explicitly mark unavailable panels. Integrated qualitative and conditional rollout panels remain evidence slots until actual outputs exist.

Reuse the established token-budget identity and tensor notation. Define physical time, spatial index, level, quotas and support sets before examples. Do not add equations merely to make descriptions appear deeper. Audit local versus global K, nested versus exclusive support, and logical bytes versus unique storage/device peaks. Any moved equation needs a source-to-destination mapping; the historical recurrence remains correctly attributed and scientifically qualified.

## 8. Cross-chapter editing and technical support

Chapter 1 needs a concise question map, not a second detailed experiment catalog. Chapter 2 should end relevant representation/memory discussions with the unresolved design question taken up in Chapter 3. Rewrite chapter openings, transitions, system synthesis, abstract and conclusions against the same question-to-evidence inventory.

Remove chronology-driven wording from the authoritative narrative, while retaining dates and revisions in provenance and necessary methodological distinctions. Audit for “нова гілка”, “попередній/остаточний” when used as editorial sequence, “історичний” as a substitute for protocol description, and “поточний зріз” as a section organizing principle. Do not blanket-delete words when they convey an actual temporal variable or necessary attribution.

No additional model refactoring is implied by this editorial plan. Required tooling work is limited to: experiment/result-slot inventory; unique heading/caption/equation and cross-reference checks; coverage checks from the inventory; and preventing snapshot generators from reinserting a results section at its obsolete location. Update hardcoded audit counts and build/provenance mappings after restructuring, never simply to suppress a failing check.

## 9. Implementation order and effort

| Step | Deliverable | Estimated focused effort |
|---|---|---|
| 1 | Freeze outline, protocol map, move/renumber ledger and complete evidence-slot inventory | 2–3 hours |
| 2 | Rewrite Chapter 3 motivations, comparisons, examples and integrated verification rationale | 5–7 hours |
| 3 | Restructure Chapter 4; relocate verified evidence; create all pending destinations and transitions | 5–7 hours |
| 4 | Two explanatory diagrams and reused-figure/caption integration | 3–4 hours |
| 5 | Align Chapters 1–2, abstract/conclusions, appendices, generator and structural audits | 3–4 hours |
| 6 | Build and review the complete Ukrainian PDF; correct layout and final narrative defects | 3–5 hours |

Total: approximately **21–30 focused hours**, about 3–4 working days, excluding experiment execution. These are estimates, not measured completion commitments. Expected main-text growth is roughly 6–10 typeset pages after consolidation, with a deliberate review if it exceeds 12. Depth comes from explanation and comparison, not from adding experiment families.

The earlier compute ceiling remains: E1–E4 at most 9 additional GPU hours including conditional rollout; exactly four planned integrated confirmations with runtime measured from E4 before locking. No new dataset, sweep, third seed or mandatory full-training ablation. No GPU execution is necessary to implement this narrative plan.

## 10. Acceptance criteria

1. A reader can identify the need, mechanism, controlled factors, metric and inference limit for every experiment family without consulting the operational plan.
2. Chapter 3 and Chapter 4 have matching methods/evidence destinations. The system-level rule follows all evidence and marks unsupported branches.
3. All running, dependent and planned results have traceable slots; completed numbers retain verified provenance. Conditional studies are not presented as committed runs.
4. The manuscript reads as one study of hierarchical design dimensions without implying one unchanged model or pooled protocol.
5. Unique numbering and valid references are checked explicitly, including the duplicated 4.9.3 defect. Renumbering propagates through captions, prose, appendices, contents and figure provenance.
6. All four existing gates plus structural/coverage checks pass. Pending counts remain visible and are never treated as scientific completion.
7. The rebuilt PDF receives a full layout review and enlarged inspection of the experiment map, explanatory diagrams, formulas, result tables and placeholders. Finally read the introduction → chapter openings → subsection transitions → synthesis → conclusions continuously to assess the argument, not only page geometry.
8. Deliver a revised PDF, source changes, experiment-slot inventory and a concise report listing structural moves, remaining evidence gaps and validation. No experiment launch or code revision is silently bundled into editorial acceptance.
