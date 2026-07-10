# PLAN — texts refactor for the "SHAPES OF HIERARCHY" program (2026-07-10)

**What this is.** The complete article + dissertation change plan for the CW-style-table program
(pre-registered in `SPATIAL_FACTORIAL.md` final block; Phase B running). Every pending number
carries an **[EXPECTED: …]** placeholder — these are the expectation baseline: when an npz lands,
the measured value replaces the placeholder and is *compared against it*; a measured value outside
its expected range triggers the pre-registered ALTERNATIVE branch text (drafted below), never a
silent rewrite. Discipline: **both branches are drafted before results exist; no soft reads.**

**The one framing correction to the author's sketch (agreed rationale):** we draft *expected*
outcomes, not only *happy* outcomes — for B2 especially, the "unhappy" branch (span, not coarse
level, is the currency) is scientifically fine and its wording is pre-drafted here so neither
branch tempts a soft read. Second correction: the shapes program did not start today — spat18
(spatial shape at fixed window), testbed B d1 (temporal shape at long loop), and the frontier
(allocation) already populate the matrix. The texts should present ONE designed investigation
that the new runs complete, not a bolt-on.

---

## 1. THE NEW NARRATIVE THREAD (drafted prose, reusable in CH1/CH4/Розділ 4)

> Hierarchy in a latent video predictor is not a single design but a **family of shapes**. A
> context-token budget can be packaged along at least four axes: **spatial abstraction** (keep
> every frame, at lower resolution), **temporal abstraction** (keep full resolution, at a coarser
> tick), **depth** (how many nested levels of either kind), and **allocation** (how the budget is
> split between long-span and recent-detail streams). Each shape buys memory span at a different
> token price and pays a different information cost. The question this article's final study
> answers is not "does hierarchy help?" but **"which shape of hierarchy pays, where, and why —
> and what does each shape cost?"** The instrument is the same throughout: matched token budgets,
> the exact re-entry-labelled testbed, and paired per-frame comparisons — applied to a
> configuration grid presented in the manner of the Clockwork VAE benchmark [REF: CW-VAE], whose
> presentation we adopt and whose controls we tighten (§7.1).

Retro-coverage sentence (place where the grid is introduced): "Three cells of this grid were
measured in earlier sections — the spatial factorial (§4.4.1), the slow-clock boundary study
(§4.5), and the allocation frontier (§4.3.6); the runs added here complete the temporal-family,
control, and depth cells at the headline budgets."

---

## 2. NEW TABLE — the shape taxonomy (CH4 §4.4 opener; Розділ 4 Табл. 4.x)

| axis | shape | buys | costs | evidence |
|---|---|---|---|---|
| spatial | coarse 4×4 stream, stride 1 | 4× span/token; per-level decodability; dense temporal coverage | abstraction cost −0.008 (diff) / ~0 (reg) at fixed window | G-S1 measured |
| temporal | full-res history, stride 4 | 4× span/token; full detail at anchors | inter-anchor blindness (momentum reversals); no cheap-level decodability | **B2 [EXPECTED: ≈ hier-long ±0.01]** |
| spatial × temporal | coarse stream, stride 2 | 8× span/token — the cheapest loop-coverer | both costs above | **B1 [EXPECTED: hier win at 512 tok]** |
| depth | + 2×2 top stream | a third scale to own | dead weight if the data has fewer scales; slower fixed-step convergence | **B3 [EXPECTED: null vs 2-stage]**; testbed-B d1; capacity-dump precedents |
| allocation | span-first sizing | the win band + its edges | — | frontier §4.3.6 measured |

---

## 3. ARTICLE — per-chapter changes

### CH1 (intro)
- §1.2: + one sentence introducing the shapes question (thread ¶ above, compressed).
- §1.4 contributions: + item **only if B1 primary confirms**: "the win-band lower edge is
  *family-parametric* and predictable from token arithmetic for every abstraction family tested
  (measured: 896 for the stride-1 coarse family, **[EXPECTED: 512]** for the stride-2 family)."
- Abstract: + one clause, same condition, e.g. "…with edges that move with the abstraction
  family exactly as token geometry predicts." **Do not touch the abstract until B1 is two-seed.**

### CH2 (testbed/protocol)
- §2.4 or §2.6: one harness sentence — the predictor consumes an arbitrary set of latent token
  streams parameterized by (resolution, stride, count); levels share one joint attention; depth
  adds tokens, **not parameters** (contrast CW-VAE, where depth adds parameters — our depth axis
  is capacity-neutral).
- §2.8: the carve-out sentence for Phase C: "One exception: for the standard-MMNIST control
  column we *display* an SSIM-vs-horizon curve for literature comparability, with the
  persistence and all-black floors drawn on the same axes; it is calibration, not evidence."

### CH3 — no content change (pointer sentence to §4.4.4 blinding probes at most).

### CH4 (the main site)
1. **Fold §4.3.x → §4.3.6** (already planned) + R-F/R-G upper-edge bracket text (landed in
   commits 1915a49); regenerate fig9/fig10 — fig10 switches to the **plateau-band presentation**
   (hier plotted as a band ≈0.166–0.182 with regime markers 30k/60k distinguished; the
   fixed-step convergence disclosure in the caption, per `hier-plateau-diagnosis`).
2. **Retitle §4.4** from "Claim 3 — spatial abstraction…" to **"Claim 3 — the shape of the
   abstraction: spatial, temporal, deep"**; current content becomes §4.4.1 (the spatial
   factorial, unchanged). New subsections:
   - **§4.4.2 The temporal family and the moving edge (B1).** Token arithmetic per family:
     edge(family) = (loop/k_t)·tok/frame(k_s) + fine floor. Stride-1 coarse: 48·16+2·64=896
     (measured, two-seed). Stride-2 coarse: 24·16+2·64=512 → **[EXPECTED: hier-s2-min ALL
     ≈0.170–0.180; Δ vs flat-short ≈ +0.04 to +0.05, CI>0, win in every cell; dt≤48 win /
     dt>96 null signature]**. Primary text: "the band's lower edge is not a constant of the
     testbed but of the family — it moves from 896 to 512 when the coarse tick halves, landing
     at its geometric prediction both times." ALT branch (null/loss): "the stride-2 family's
     edge lies in (512, 896]: covering the loop is necessary but the family's fine floor /
     inter-anchor aliasing sets the rest" — reported as a measured refinement, thread intact.
     Caption note for fig9: the 512 stride-2 hier point coexists with the 512 stride-1 flat win —
     different families at the same budget (prevents a "contradiction" misread).
   - **§4.4.3 Is it the coarse level, or just span? (B2).** The strongest attack on the word
     "hierarchy," run as a control: strided full-res history (span 48, no coarse level, 1088
     tok) vs the two-seed 1152 pair. **[EXPECTED (primary, from G-S1): strided-flat ALL
     ≈0.170–0.185 — within the hier plateau band; CI-clean win vs flat-budget; |Δ| vs hier-long
     < 0.01 (null-ish)]**. Primary wording change (apply everywhere listed in §6 below):
     "hierarchical packaging" → "**long-span context packaging**; the coarse level is the
     cheapest per-level-decodable implementation of it, and the only one that also keeps dense
     temporal coverage." ALT branch (hier-long CI-win): keep current wording and add: "temporal
     subsampling at the same budget does NOT reproduce the win — the coarse level's dense
     coverage carries value beyond span," and B2 becomes Claim 1's strongest control.
   - **§4.4.4 What each level stores (blinding probes).** CW-VAE §5.1 analog, strengthened by
     per-level decodability. **[EXPECTED: blind-coarse collapses the dt≤48 HARD retrieval bins
     toward flat-budget-like error (ALL → ≈0.20+), blind-fine costs mainly h1–8 detail and
     sharpness]**. Report-only; feeds fig14 and one mechanism paragraph.
   - **§4.4.5 The comparison table and degradation curves (B3 + master view).** Master table
     (auto-generated `docs/CWSTYLE_TABLE.md`) as Table 4.X + fig12. Depth row: **[EXPECTED:
     hier3 ALL ≈0.175–0.190; |Δ| vs hier-long within seed spread 0.016 → NULL]** with the
     span-cap explanation ("a third stage on a single-scale task can only re-allocate — depth
     must match the data's memory-scale count"); mirrors CW-VAE's own finding that a 2-level
     model sufficed for their maze dataset. ALT (win): escalate to seed 1, report as a real
     re-allocation benefit. ALT (loss): the fixed-step context-cost explanation (plateau
     diagnosis), reported as the depth cost at fixed budget. EBT column: cite §4.2.3/§3.5 —
     characterized negative, no rollout cells by design.
3. §4.6 chapter conclusions: + two bullets (family edge law [branch-gated]; depth boundary).

### CH5 (generalization)
- New **§5.3.1 "Long-horizon control: 300-frame rollouts on standard MMNIST"** (Phase C).
  **[EXPECTED: hier-long-mm ≈ flat-budget-mm within ±0.005 ALL fg-MSE at every horizon bin —
  the long-horizon extension of the §5.3 null]**. Disclosures: zero-action conditioning;
  hardness axis inert; SSIM display per the §2.8 carve-out; fig15. ALT (hier win): contradicts
  T7 — investigate (data/protocol) before ANY reporting; pre-registered as a stop-and-examine.

### CH7 (related work) — the CW-VAE context expansion the author asked for
Extend §7.1 with a methodology paragraph (draft):
> Our final study adopts the **comparison format** of the Clockwork VAE benchmark [REF: CW-VAE]:
> a configuration grid summarized in one table, quality-retention curves over long open-loop
> rollouts, and level-content probes (their §5.1 prior-following experiment; our blinding
> probes, sharpened by per-level decodability). We adopt the format while tightening three
> controls it lacks. First, CW-VAE's models differ in parameter count (RSSM 5M vs CW-VAE 12M in
> their Table; only NoTmpAbs is matched), whereas our depth axis is capacity-neutral — levels
> add tokens to a shared backbone, not weights. Second, an RNN state has no token-budget axis,
> so their hierarchy-vs-flat comparisons cannot hold context cost fixed; our grid is
> budget-matched by construction. Third, their metrics are aggregate SSIM/PSNR/LPIPS/FVD — on
> sparse-foreground video we measured a constant all-black frame outperforming a trained CW-VAE
> on aggregate SSIM/PSNR (§2.8), so our cells are foreground-masked and re-entry-stratified.
> Notably, CW-VAE's depth evidence is an information-allocation (KL) analysis, not a prediction
> -quality dose-response; our table carries the quality-per-depth cells directly — including the
> measured single-scale null **[EXPECTED: B3]**, which agrees with their own observation that a
> two-level model sufficed for GQN mazes. VTA [REF] and the clockwork lineage
> [REF: Koutník et al. 2014, Clockwork RNN] are the temporal-abstraction relatives; NoTmpAbs is
> the ancestor of our "same tokens, no abstraction" flat controls.
- §7.5 non-claims: + "we do not claim depth helps on single-scale data (measured null, §4.4.5)"
  [branch-gated].

### CH8 (conclusion)
- §8.1 one-claim: + family clause **only if B1 primary** ("…and the budgets move with the
  abstraction family as token geometry predicts").
- §8.2 boundary list: + depth bullet (single-scale null) + family bullet [branch-gated].
- §8.4 sizing rule upgrade: "spend tokens on span until the horizon is covered **using the
  cheapest family that covers it** — then on resolution/recency."
- §8.5 future work: + the multi-scale testbed with a *visible slow factor* (dodges the
  trustworthy-memory boundary because slow state is continuously refreshed — the CW-VAE
  slowness regime, not occluded recall) as the arena where depth-3 can *win*; + the re-based
  factor-4 pyramid (16×16/4×4/1×1) as its architecture.

---

## 4. DISSERTATION — per-file changes (docs/dissertation/rewrite/)

| file | change |
|---|---|
| 02_ROZDIL1 | §1.3.x review: UA version of the CH7 CW-VAE methodology paragraph (comparison-methodology justification is spec-124-friendly); add VTA/NoTmpAbs/clockwork-RNN mentions; one sentence positioning «форми ієрархії» as the systems-analysis decomposition of the design space |
| 04_ROZDIL3 | new subsection «Узагальнення моделі на сімейство ієрархічних конфігурацій»: token-stream parameterization (роздільна здатність, крок дискретизації, глибина), the frozen-trunk 2×2 top level, depth = tokens not parameters, complexity note; NO reopening of the 2-level core sections |
| 05_ROZDIL4 | mirror CH4: §4.4 retitle + §4.4.2–4.4.5 (UA), master table as Табл. 4.X, fig12/13/14 as рис.; §4.6.5 gains the MMNIST long-horizon control paragraph (Phase C); ВИСНОВКИ до розділу + two items |
| 06_VYSNOVKY | ONE sentence extension of the Claim-1/Claim-3 result items [branch-gated]; **no new novelty items** (too late to add safely; all new results slot under existing items as robustness/generalization) |
| 00_FRONT_MATTER | abstract clause synced with CH1 (LAST, at the UA pass) |
| 01_VSTUP | no change (unless B1 two-seed by then: optional clause in result 5) |

UA–EN terminology additions (append to DISS_DIGEST §5): форма ієрархії = hierarchy shape;
просторова/часова абстракція = spatial/temporal abstraction; крок (проріджування) грубого потоку
= coarse-stream stride; сімейство конфігурацій = configuration family; закон краю смуги =
band-edge law; засліплення рівня = level blinding; стовпець контролю = control column;
глибина ієрархії = hierarchy depth; перерозподіл бюджету = budget re-allocation.

---

## 5. FIGURES & TABLES master plan

| asset | status | action |
|---|---|---|
| fig9 frontier | exists (6-pt) | + caption family note after B1 (512 stride-2 hier point ≠ contradiction of 512 stride-1 flat win) |
| fig10 bars/deltas | exists (6-budget) | REDESIGN panel A hier side as plateau BAND (0.166–0.182) + 30k/60k regime markers; caption = fixed-step disclosure |
| fig12 degradation curves | exists | regenerate post-Phase-B (B1/B2/B3 auto-join); caption drafted in script |
| **fig13 family edge law** | NEW | bars/points: predicted vs measured lower edge per family — (s2,k1)→896 measured; (s2,k2)→512 **[EXPECTED measured]**; upper bracket (1792,2432) drawn; caption: "edges are arithmetic, not accidents" |
| **fig14 level blinding** | NEW | 3 curves (unblinded / blind-coarse / blind-fine) fg-MSE vs h + a dt≤48-HARD inset; hier3 blind-top added if B3 interesting |
| **fig15 MMNIST control** | NEW (Phase C) | fg-MSE vs h, flat vs hier overlapping **[EXPECTED]** + floors; optional SSIM calibration panel (carve-out) |
| Table 4.X master (CWSTYLE_TABLE.md) | exists, auto | placement CH4 §4.4.5 + Розділ 4; regenerates as cells land |
| Table B2-trio | NEW | hier-long / strided-flat / flat-budget @≈1152: ALL + paired Δ + CIs **[EXPECTED above]** |
| Table depth column | NEW | depth 1/2/3 @1152 (diff; reg where exists) **[EXPECTED B3 null]** |

---

## 6. AT-RISK SENTENCES (exact sites; both branches ready)

B2-dependent ("hierarchical packaging" → "long-span context packaging" IF primary/null branch):
1. CH4 §4.3.2 closing: "…the claim defended here is the *hierarchical packaging* of it at
   matched budget, which is exactly what the flat-budget control isolates."
2. CH8 §8.1 one-claim sentence ("coarse level … earns its token budget…" — keep; add the
   family qualifier clause).
3. CH1 abstract + §1.4 contribution 1 wording.
4. Розділ 4 mirror of (1); ВИСНОВКИ result item; АНОТАЦІЯ clause (front matter, UA pass).
Note: dissertation novelty item 1 is SAFE on either branch (it claims the testbed methodology +
matched-budget demonstration, not coarse-level exclusivity — the 2026-07-07 re-scope pays off).

Phase-C-dependent: CH2 §2.8 carve-out (add regardless — it is a protocol statement).
B1-dependent: every "band [896, …)" phrase gains "for the stride-1 coarse family" — sites:
CH1 abstract, CH4 §4.3.6, CH8 §8.1, Розділ 4 §4.3.6, VYSNOVKY, fig9/fig10 captions.

---

## 7. EXPECTATION BASELINE (the comparison sheet — fill measured values in-place)

| run | expected primary | measured | verdict vs expectation |
|---|---|---|---|
| B1 hier-s2-min @512 | ALL 0.170–0.180; Δ +0.04..+0.05 CI>0 all cells; dt-gate signature | ALL **0.1893**; Δ **+0.0330** [+0.0246,+0.0419]; 14/14 cells WIN; dt-signature exact | **primary CONFIRMED**; magnitude below range → family plateau offset ≈+0.017 discovered (sparser anchors); s1 queued |
| B2 strided-flat @1088 | ALL 0.170–0.185; ≥ flat-budget CI-win; vs hier-long null-ish (|Δ|<0.01) | ⟨npz⟩ | ⟨⟩ |
| B3 hier3 @1152 | ALL 0.175–0.190; vs hier-long |Δ| < 0.016 (null) | ⟨npz⟩ | ⟨⟩ |
| blind-coarse (hier-long) | ALL → ≈0.20+; dt≤48 HARD collapse | ⟨npz⟩ | ⟨⟩ |
| blind-fine (hier-long) | h1–8 + sharpness cost; long-h mild | ⟨npz⟩ | ⟨⟩ |
| Phase C MMNIST pair | hier ≈ flat, |Δ| ≤ 0.005 every bin | ⟨npz⟩ | ⟨⟩ |
| E-320 hier-s4-min @320 | hier 0.19–0.22 vs flat-320 0.23–0.245; Δ +0.02..+0.05 CI>0 → 3-pt edge law | ⟨npz⟩ | ⟨⟩ |
| REG twins ×3 (512/1088/1152) | diffusion shape verdicts reproduce/enlarge under reg; no cross-head sign flips | ⟨npz⟩ | ⟨⟩ |

Escalation rules (unchanged from the pre-registration): CI-clean surprising sign → seed-1 before
any text lands; Phase-C hier win → stop-and-examine, no reporting.

## 8. SEQUENCING
Write NOW (no results needed): thread ¶, taxonomy table, CH7 paragraph, CH2 sentences, §4.4
retitle + subsection skeletons with [EXPECTED] slots, Розділ 1/3 additions, terminology rows,
fig13/14/15 spec stubs, fig10 redesign. AFTER each npz: swap placeholders, pick branch, regen
figures/table. LAST: abstract/front-matter clauses (only on two-seed), UA pass on frozen text.
Estimated text effort: ~1.5–2 working days total, parallel to the GPU queue.
