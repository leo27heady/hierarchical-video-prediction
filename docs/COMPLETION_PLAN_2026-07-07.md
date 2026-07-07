# COMPLETION PLAN — everything between "draft" and "publishable/defendable" (2026-07-07)

> **STATUS UPDATE (2026-07-07 late):** E1 text fixes COMPLETE (batches 1+2, commits 91abb7f +
> dc3aaba — all 12 audit items done incl. ROZDIL2 math fixes, 124 decision-theory/object,
> 0.26% sync, Ch3-summary consistency). E2 article completion COMPLETE as prose: CH1_INTRO
> (finalized abstract), CH6_REVERSIBILITY, CH7_RELATED (+non-claims), CH8_CONCLUSION written;
> CH4 BRANCH markers resolved (§4.5.6 closure, §4.6 measured resolution) + caption fix; chapter
> numbering canonical (1-8). [A1]-as-first-hierarchy-adaptation framing added (VSTUP, ROZDIL2
> §2.1, CH1 arc) per author instruction. **E6 bibliography BUILT (2026-07-07):**
> `07_REFERENCES.md` — master list (~85 external entries + author slots) from the full 104-
> descriptor [REF:] inventory, with the descriptor→key binding table and assembly instructions;
> remaining E6 work = author slots + [VERIFY]-flag clearing against PDFs + number substitution
> at E8. REMAINING: E4 figures (placeholders in place), E5 generalization runs (TaxiBJ chain
> RUNNING; R-A/R-B/R-C queued), E6 verification pass, E7 author-blocking cluster, E8 assembly,
> E9 UA pass, E10 packaging.

**Definition of done.**
*Article [A4]:* a submittable manuscript — all 7 skeleton chapters as prose, final abstract,
every figure rendered (or an explicit placeholder box), final tables, verified bibliography,
limitations + reproducibility statements, journal formatting.
*Dissertation:* the assembled UA document — all chapters + front matter + ВИСНОВКИ, real
reference list, appendices, TOC/abbreviations/counts, Ukrainian language throughout, merged into
the submission .docx (current .docx = donor template, never edited; rewrite lives in
`docs/dissertation/rewrite/`).

**Marker census (measured 2026-07-07):** dissertation 00–06: **58 [AUTHOR:]**, **95 [REF:]**,
**47 [FIG]**, 16 EXPECTED + 5 FALLBACK (generalization shells), 49 [UA-PASS:]. Article CH2–CH5 +
skeleton: 10 [AUTHOR:], 11 [FIG], 5 [BRANCH], 30 EXPECTED + 18 PENDING + 9 FALLBACK (all in CH5).

---

## A. ARTICLE — the gap list

**A1. Chapters that do not exist as prose yet** (skeleton bullets only):
- **Ch. 1 Introduction** — from skeleton §1 (motivation, the three claims, contributions list,
  dissertation arc). ~2–3 pp.
- **Ch. 5 Reversibility** — content EXISTS (`docs/THEORY_CHAPTER_D.md`, T6 panels,
  `mem_panel.md`); needs conversion to article register + the two figures. ~3–4 pp.
- **Ch. 6 Related work + honest scoping** — skeleton §6 lists the prior-art axes (CW-VAE/DLH/
  TECO/MSPred/NUWA-XL/MilliVid; Self-Forcing; AlignProp/checkpointing; EBT; OpenSTL). Every
  mechanism claim must be verified against the PDFs (standing debt from FINAL_ARTICLE_REVIEW §5)
  + fresh prior-art sweep at submission. ~2 pp.
- **Ch. 7 Conclusion** — from skeleton §7 + VYSNOVKY structure. ~1 p.
- **Abstract** — 180-word draft exists in skeleton §0.1; finalize after frontier lower edge.
- **Appendices** — A: T7 bridge (text exists); B: EBT config-gap table (exists in CH3); C:
  reproducibility/artifact statement (write from HANDOFF §7).
- **Title decision** — candidates listed in skeleton §0; the surviving branch is
  "When Does Hierarchy Pay? A Memory-Horizon Study at Matched Token Budgets" (pair-RED-safe).

**A2. CH4 integration pass** (single focused edit session):
- Merge the §4.5 pair-outcome block into the section body; resolve the **5 [BRANCH:] markers**
  (single-step PASS branch survives; rollout closure = RED text).
- Add the frontier lower-edge row/sentence when tonight's pair lands (4th fig9 point).
- Fix the over-claiming [FIG ch4-4] caption → fig7's honest reading (audit-confirmed).
- Length trim: CH4 is 627 lines; candidates to compress = per-iteration H-grid narrative
  (§4.5.2–4.5.3 → one table + one paragraph), duplicated floor recitations.

**A3. CH5 generalization** — 30 EXPECTED / 18 PENDING / 9 FALLBACK:
- NOW (author-authorized happy path): pre-fill each dataset section with tendency-explicit
  [EXPECTED] prose (see §D) so measured numbers become one-bracket swaps.
- AFTER T1 runs: swap numbers, delete FALLBACKs or fire them, strike-or-keep KTH per its
  pre-registered option (a)/(b).
- [AUTHOR: A2 result tables ×2] for the rotation sections; [VERIFY: 3D render timing].

**A4. Figures** (exists: fig1–fig9 + README; placeholders allowed per author):
- MISSING-render (data exists, scriptable): EBT landscape/reachability (from
  `ebt_scaled_s{1,2}.log`), reversibility memory-frontier panel (from `mem_panel.md` + Panel A),
  mechanism qualitative grid ([FIG ch3-1] from `step1_mech_clean2_*.png` assets).
- MISSING-draw (schematics): testbed world/viewport/re-entry diagram ([FIG ch2-1]), 2-level
  AE + WindowDiT architecture ([FIG 3-1/3-2 in dissertation, same asset reused by article]).
- Update fig9 with the 4th point tonight; regenerate fig3 matrix if hier-min/flat-min added.

**A5. Bibliography** — build the real .bib; verify every claim-bearing citation against the
paper PDFs; fresh 2024–26 sweep (state-space/long-video/memory-augmented world models) — the
same list feeds dissertation §1.3.

**A6. Journal packaging** — venue choice ([AUTHOR]), formatting, author info, data/code
availability statement (repo + artifact inventory exists in HANDOFF §7).

## B. DISSERTATION — the gap list

**B1. Audit-confirmed text fixes** (12 items, PLAN_NEXT T2; the critical one = novelty item 1
re-scope). Status: being executed starting 2026-07-07 (this session: novelty re-scope,
200-frame, MSE-range, trustworthy-clause, Положення block, §1.4.4 alignment).

**B2. [AUTHOR:] flags — 58, clustered** (blocking-ranked):
1. Publications/references binding (biggest risk): real 10-item list replacing the donor
   (Isaienkov) list; [A1]–[A4] exact bibliographic entries; [A4] venue/status by defense
   (the 3+1+6 count depends on it); 6 conference theses incl. DSMSI-2023 verification.
2. Formalities: НДР numbers ×2 (+wording check vs НДР-2 report), UDC confirmation, [A2]
   Q1-vs-Scopus-indexed, page/figure/table/source counts (after assembly).
3. Evidence receipts: VQ-instability receipts for §1.3.2 (the motivation finding — locate
   logs/wandb of the author's VQ attempts or soften to "preliminary experiments"); [A2]
   dataset/result tables for §4.1.2; exact param count confirmation (04_ROZDIL3).
4. Decisions already made, mechanical: copyright-certificate sentence removal (Q5) in 00/01/06;
   acts appendix content (Г/Д).

**B3. [REF:] placeholders — 95** → the reference list (~145 sources target). Same .bib as A5;
§1.3's currency gap (named 2024–26 long-video/world-model/SSM works) is part of this.

**B4. [FIG] placeholders — 47.** Most map to existing assets (fig1–9, training grids, EBT probe
PNGs, mem_panel data); genuinely new: the two schematics (B/A4 shared) + per-chapter conceptual
sketches in ROZDIL1 (decomposition/uncertainty diagrams — old draft had figure slots; redraw
~4–6 simple diagrams). Produce a figure manifest: placeholder-ID → asset path or DRAW.

**B5. §4.6 generalization shells** — 12 EXPECTED + 3 FALLBACK: happy-path pre-fill now (§D),
measure in T1, strike-or-fill per Q4 rule before defense.

**B6. Missing document components** (not yet written at all):
- СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ (from B3's .bib, ДСТУ 8302:2015 format).
- ДОДАТКИ: А publication list; Б implementation acts (scans); В code fragments of the software
  complex; Г experiment configs + the OpenSTL-bridge appendix (per 00_FRONT §5 TOC).
- ЗМІСТ regeneration (section-level), final ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ (base table exists),
  title-page fields, «Положення, що виносяться на захист» (added this session).
- Structure/volume paragraph counts (after assembly).

**B7. UA translation** — the single biggest remaining task (~10k lines of EN working text),
deliberately LAST on frozen content; the 49 [UA-PASS:] markers are the pre-placed genre idioms.
Then merge into the submission .docx (replacing donor content wholesale).

**B8. Defense-process items (author-side, outside the text):** фаховий семінар/висновок
кафедри, керівник's висновок, академдоброчесність check, opponents — track separately.

## C. What I will draft on the happy path (author-authorized 2026-07-07)

Pre-fill CH5 + §4.6 dataset sections with tendency-explicit prose, keeping every unmeasured
number behind an [EXPECTED] tag (one bracket per claim; fallback clause retained). The
tendencies, derived from measured results:
- **2D rotations:** regression head dominates (deterministic dynamics — the T1/rollout-reversal
  pattern); hier ≈ flat (no memory horizon — T7-pattern null #2); large margin over persistence;
  [EXPECTED] competitive-to-better vs [A2] baselines at small Δ, graceful degradation to Δ=12;
  diffusion twin pays a small best-of-1 stochasticity overhead (the reg-vs-diff rollout gap
  ~0.01–0.02 scale).
- **3D polycubes:** same regime + the ONE dataset where diffusion best-of-8 should show the
  coverage win (T1's oracle-coverage signature) on occlusion-driven multimodality.
- **MMNIST-CIFAR:** hier ≈ flat null #3; AE round-trip ceiling becomes the prominent row
  (CIFAR texture vs 1.85M AE — decoder-capacity disclosure, T7 sentence pattern); LPIPS is the
  favorable column; specialist-MSE gap disclosed, no leaderboard claim.
- **TaxiBJ:** recipe-transfer claim only (4→4 = no window axis): trains stably, lands in the
  leaderboard's neighborhood [EXPECTED: within ~1.5–3× of TAU's 0.31 MSE, clearly beating
  persistence]; hier arm = token-efficiency read; LPIPS dropped with stated reason.
- **KTH-64:** recommend option (b) SKIP with the resolution-incomparability sentence (chapter
  is complete without it; TaxiBJ carries real-data transfer).

## D. Implementation plan (ordered)

| # | phase | content | owner | est |
|---|---|---|---|---|
| E0 | tonight | frontier lower-edge fill (fig9 4th point, CH4/Розділ4 sentences) + TaxiBJ B2 build + launch @~14GB BS | me | done on wake |
| E1 | text fixes | the 12 audit items (B1) — started this session | me | 1 session |
| E2 | article completion | Ch.1/5/6/7 prose from skeleton + THEORY_CHAPTER_D; CH4 integration pass (5 BRANCH, pair-block merge, trim); abstract; appendices | me | 1–2 sessions |
| E3 | happy-path pre-fill | CH5 + §4.6 tendency drafts (§C) | me | 1 session |
| E4 | figures | render EBT + reversibility + mechanism grid; draw 2 schematics + ROZDIL1 diagrams; figure manifest for all 47+11 placeholders | me | 1 day CPU |
| E5 | generalization runs | T1 campaign (TaxiBJ → rot2D → poly3D → MMNIST-CIFAR); swap EXPECTED→measured; strike-or-fill | me+GPU | ~28–34 GPU-h |
| E6 | bibliography | .bib build + citation verification + 2024–26 sweep; resolve 95 [REF:] + article refs | me (draft) + author (verify own pubs) | 1–2 sessions |
| E7 | author-blocking | B2 cluster: publications list, [A4] status, НДР/UDC/theses/quartile, VQ receipts, [A2] tables, acts | **author** | — |
| E8 | assembly | ЗМІСТ, abbreviations, appendices, counts; author review pass over everything | me + author | 1 session |
| E9 | UA pass | translate frozen EN → UA (49 UA-PASS idioms pre-placed); merge into .docx | me + author final read | 2–3 sessions |
| E10 | article packaging | venue format, title decision, submission | author + me | 0.5 session |

Dependencies: E9 strictly after E1–E8 freeze; E5 gates the final CH5/§4.6; E6 feeds both texts;
E7 is the critical path for the defense (publication list) — start it in parallel NOW.
