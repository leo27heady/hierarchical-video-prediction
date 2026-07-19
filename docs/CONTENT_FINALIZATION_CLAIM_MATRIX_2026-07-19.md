# Content Finalization Claim Matrix — 19 July 2026

This matrix is the editorial working copy of the frozen scientific claim ledger. It does not
replace `docs/CLAIM_LEDGER.md`; it translates the ledger into manuscript wording and verification
checks for the final content pass.

| Topic | Frozen evidence | Permitted headline wording | Wording to reject |
|---|---|---|---|
| Reach-specific confirmation | `hier-min` versus `flat-min`, registered \(14<\Delta t\le48\) band: seed 0 \(+0.0980\ [0.0788,0.1179]\), seed 1 \(+0.0914\ [0.0736,0.1098]\) | Hierarchy is confirmed where the longer package contains the re-entry past that the comparator cannot address | Hierarchy is universally better; geometry proves information-theoretic necessity |
| 896-token point | Untouched E2, hierarchy confirmed over matched flat, seeds 0–1 | Confirmed tested hierarchy-favouring point | Exact crossover occurs at 896 |
| 1,152-token point | Untouched E2, hierarchy confirmed over matched flat, seeds 0–2 | Largest tested family-wise-confirmed hierarchy-favouring point | All budgets below 1,152 favour hierarchy |
| 1,792-token point | Untouched simultaneous interval unresolved | Tested point is unresolved | Hierarchy wins; point lies inside a continuous hierarchy band |
| 2,432-token point | Flat confirmed under B16x30k and B8x60k; sampler/codec variants directionally retain but do not always family-wise resolve it | Smallest tested family-wise-confirmed flat-favouring point in the primary frontier; robustness precision is qualified | Unique crossover at 2,432; universal flat threshold |
| 3,072-token point | Flat full-resolution context confirmed, seeds 0–1 | Confirmed tested flat-favouring point | Flat always wins once reach is saturated |
| Reach efficiency | 48 fine frames \(=48\times64=3,072\) context tokens; 48 coarse + 6 fine \(=48\times16+6\times64=1,152\); 48 coarse + 2 fine \(=896\) | Same nominal 48-frame addressable reach with 2.67–3.43 times fewer context tokens | Same performance with fewer tokens; total compute or wall-clock is reduced by the same factor |
| Tested bracket | Largest confirmed hierarchy point 1,152; smallest confirmed flat point 2,432 | The tested family-wise bracket lies between 1,152 and 2,432 | Monotone, unique, interpolated, or asymptotic crossing law |
| B1 stride-2 configuration | Untouched overall \(+0.0340/+0.0381\); reach band \(+0.0692/+0.0743\) | Whole-configuration advantage | Pure causal effect of stride or target-interface invariance |
| Package shape | Diffusion temporal advantages positive at seeds 0–4, mean \(+0.0098\), t95 \([+0.0023,+0.0173]\); regression and interaction heterogeneous | Temporal package advantage is stable under the tested diffusion head; cross-head generalization is seed-sensitive | Stable directional head-interaction law |
| Trust replication | Early h1–24 hierarchy effects positive at seeds 0–2; late h61–256 negative only at seed 0 and positive at seeds 1–2 | Early hierarchy benefit replicates; the proposed late trust boundary does not | Stable approximately-24-frame trust boundary |
| Sampler robustness | All 20 registered point estimates retain direction; 18 simultaneous lower bounds positive; 2,432 inversion unresolved at sampler seeds 1/2 | Directionally robust across the registered sampler/NFE grid with qualified upper-point precision | Sampler invariant |
| Codec robustness | Four alternate-codec point estimates retain direction; 2,432 inversion unresolved family-wise | Directionally robust to one independently trained codec, conditional on predictor seed 0 | Representation invariant; component-level codec causality |
| Explicit raw access | Raw K48 foreground MSE 0.0494; registered neural-minus-K48 effects all positive | Privileged pose-indexed lossless access is a lower-error reference and separates addressability from learned self-fed reliability | Token-matched model ranking; practical retrieval system; natural-video result |
| E6 persistent objects | Hierarchy-minus-flat convention: union-MSE benefit \(+0.0102/+0.0122/+0.0169\); trajectory benefit \(+2.4786/+1.4223/+4.1325\); coverage \(+0.0304/+0.0159/+0.0498\); identity \(-0.0762/-0.0561/-0.0647\); foreground MSE unresolved | Hierarchy better preserves support/continuity; flat better preserves identity; task remains difficult in absolute terms | External replication of the LoopMom mechanism; natural-video generalization; task solved |
| Reversible computation | Exact reverse replay and constant activation memory relative to BPTT in the measured recurrence | Reversibility reduces the activation term for the tested long temporal process | Cause of Chapter 4 hierarchy gains; automatically multiplicative with context savings |
| Overall integration | V-BLT/RoPE-ND, TempVerseFormer, and hierarchical context prediction are separately measured dissertation strands | Integrated systems methodology and research software toolkit | One experimentally validated end-to-end integrated model pipeline |

## Editorial checks

Before accepting a manuscript revision, search for and inspect:

- `1,792` near *win*, *confirmed*, *band*, *crossover*, or *threshold*;
- `same performance` near token-efficiency claims;
- `pure stride`, `causal stride`, or `target matched` near B1;
- `trust boundary` without *not replicated*;
- `natural video`, `generalizes`, or `universal`;
- `integrated pipeline` or an unqualified `integrated software complex`;
- `frame-level bootstrap` or bootstrap wording that omits clip clusters;
- epsilon-prediction notation in descriptions of the executed diffusion head;
- pending/projected/in-queue states for completed or terminally skipped studies.

