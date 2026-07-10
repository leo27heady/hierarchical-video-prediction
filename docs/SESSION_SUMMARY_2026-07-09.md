# SESSION SUMMARY — comprehensive record (2026-07-07 → 2026-07-09)

**Scope of this session.** A multi-day working session that took the project from "content-complete
drafts + one measured frontier point" to: a fully two-seed six-point token-budget frontier (last
point resolving), all eight article chapters written as prose, every audit fix applied, an
external review answered, a master bibliography built, and a second generalization dataset
(TaxiBJ) measured inside the leaderboard band. The scientific story converged and hardened; the
remaining work is administrative + editorial. This document records everything — insights,
discussions, problems, decisions, results, plans — so a cold start loses nothing.

Companion docs: `HANDOFF_2026-07-07.md` (state snapshot), `FINALIZATION_PLAN_2026-07-07.md`
(remaining-work plan with code), `COMPLETION_PLAN_2026-07-07.md` (E-phase tracker),
`SPATIAL_FACTORIAL.md` (all frontier pre-registrations + measured blocks — the primary evidence
trail), `07_REFERENCES.md` (bibliography). Memory: `full-audit-2026-07-07.md`.

---

## 1. THE ARC IN ONE PARAGRAPH

A 13-reviewer audit opened the session and found NO invalidated claim (every headline number
re-verified against raw npz), one critical text fix (novelty item 1 self-contradiction), and a
batch of editorial items — all fixed same-day. The author then asked why hierarchy only won at
one budget on the frontier; investigating that produced the session's central scientific result:
a **win band with geometrically predicted edges**. A pre-registered lower-edge experiment at 896
tokens landed the campaign's largest effect exactly where the span-threshold arithmetic said it
would. The author flagged an anomaly (hier-min appearing to beat hier-long/hier-rich); resolving
it launched a four-run **robustness program** (R-A/B/C/D) — all confirmed their pre-registered
branch, dissolving the anomaly on two independent axes (seed and budget). In parallel: the four
missing article chapters were written, TaxiBJ was measured (landing inside the OpenSTL leaderboard
band on MAE/PSNR), and an external solo review was answered (9 of 11 findings accepted, 2 rejected
with evidence). The session closed on two more pre-registered runs (R-E completing the 512 point;
R-F/R-G a bracket that measures the upper edge instead of interpolating it) — R-E and R-F confirmed,
R-G resolving after a mid-session PC restart that cost almost nothing.

---

## 2. THE COMPREHENSIVE AUDIT (session open)

A 13-domain multi-agent audit (dissertation front-matter/Ch1/Ch2/Ch3/Ch4+conclusions, article
Ch2–5, model code, eval code, experiments, 124-specialty alignment, cross-consistency) with
adversarial verification, plus firsthand reading and direct npz verification.

**Verdict: sound and defense-trackable. NO invalidated claim, NO fabricated number.** Grades:
code 9/10 (leakage-free, token-matched, EBT-faithful), ROZDIL4/VYSNOVKY 9/10, cross-consistency
8.5/10, front matter 8/10, ROZDIL1 8/10, ROZDIL2 7.5/10, 124-alignment PASS (borderline-solid).

**The one CRITICAL fix:** dissertation novelty item 1 («вперше … coarse level carries the long
memory») was contradicted by its own §1.3.2 (CW-VAE/MilliVid cited as adopted prior art).
Re-scoped so the «вперше» is carried by the genuinely-new part — the memory-horizon testbed
methodology + matched-budget demonstration — with coarse-memory credited to the adopted paradigm.

**Major fixes (all applied, commits 91abb7f + dc3aaba):** "200-frame"→"152-frame"; added
«Положення, що виносяться на захист»; §1.4.4 stop-gradient wording aligned to the joint-denoising
model; ROZDIL2 inner-loop numbers → measured T6c panel (0.26%); ROZDIL2 store-the-noise
single-construction fix (eq. 2.16 + explicit inverse 2.16′); "reversibility used in Ch.3-4" →
"enabling instrument"; decision theory added to 124 methods; specialist-MSE range 23.97 → 22.06
(true floor, PredRNN++); trustworthy-memory clause synced. Two reviewers failed the format cap;
their domains (claims-artifacts, experiments) were covered by direct verification.

---

## 3. THE FRONTIER SCIENCE (the session's central thread)

### 3.1 The author's question and the win-band insight
The author observed hierarchy won only at the 1152-token mid-budget on the 3-point frontier
(512 flat / 1152 hier / 3072 flat) and asked why it wasn't winning at the bottom and top, and
whether it was an implementation problem. The answer — argued from the measured bins, not just
asserted — is that it is **lawful, not buggy**: hierarchy's only currency is "a coarse frame
costs 16 tokens vs 64 for fine → 4× more memory span per token." It pays iff that extra span
crosses the ~48-frame loop *while flat's same-budget window doesn't*. Below ~500 tokens no
hierarchical config can span the loop; at 3072 tokens flat affords the full 48-frame window at
full resolution, which information-dominates any coarse record of the same span (per-level
decodability). So hierarchy wins a bounded **band**, not everywhere — and that band's edges are
predictable from geometry.

**Framing insight (defense-relevant):** a bounded, falsifiable win band is a *stronger* claim
than unqualified dominance. If a lossy coarse compression beat full-resolution context even at
saturation, a reviewer would suspect a rigged testbed. The top-end loss is the upper edge of the
law and is load-bearing.

### 3.2 The lower edge — geometric prediction, then measurement
Pre-registered BEFORE the run: hierarchy should switch on at ~896 tokens (the cheapest budget
whose 48-slot coarse span covers the loop: 48×16 + a minimal fine tail), where flat's 896 tokens
buy only a 14-frame window. **Measured (hier-min 48c+2f vs flat-min 14f):** hier wins EVERY
horizon × hardness cell, ALL **+0.0399 [+0.0299, +0.0501]** — the campaign's largest delta —
with the lower edge landing exactly at the geometric prediction. This upgraded "hier wins at one
budget" to "hier wins the band [896, ≈2059) with a predicted lower edge."

### 3.3 The anomaly and its resolution (why the robustness program exists)
The author flagged that hier-min (0.1706) appeared to beat hier-long (0.1848) and match
hier-rich (0.1695) — "strange," possibly a training outlier, and asked to rerun hier-long/
hier-rich with more seeds/budget expecting them to improve. Key analysis given:
- The "dip" was substantially a **seed-0 artifact**: hier-long already had a second seed
  (s1 = 0.1690) better than hier-min. The hier curve from 896 on is a **plateau at ≈0.17**, not
  a monotone descent; hier arms are ~25× more seed-variable than flat arms.
- Honest expectation set BEFORE running: hier-long/hier-rich would *not* jump above the plateau;
  hier-rich would stay *below* flat-long only in the sense of the plateau, and would still LOSE
  the pair at 3072 (structural, information-theoretic — not fixable by seeds/budget).
- **Anti-cherry-picking guardrail installed:** all robustness runs are symmetric pairs (hier +
  its flat partner), seed-means reported, never best-of-seeds. This is the one move a hostile
  reviewer hunts for; the pre-registration trail is the campaign's strongest asset.

### 3.4 The robustness program — all confirmed
| run | tests | measured | verdict |
|---|---|---|---|
| **R-A** | 896 lower edge, seed 1 | +0.0358 [+0.0267,+0.0454], every cell | ✅ hier win **two-seed** |
| **R-B** | 3072 upper edge, seed 1 | −0.0107 [−0.0163,−0.0058], 12/14 loss | ✅ flat win **two-seed** |
| **R-D** | reg pairing, seed 1 (review-flagged) | +0.0501 [+0.0396,+0.0612], every cell | ✅ mechanism-agnostic **two-seed** |
| **R-C** | 1152 budget axis (both arms 60k) | Δ +0.0189→**+0.0366** (doubled) | ✅ **T3-emergence confirmed** |
| **R-E** | 512 below-threshold, seed 1 | −0.0051 [−0.0079,−0.0024] | ✅ flat win **two-seed** |

**R-C is the definitive resolution of the author's worry:** at 2× budget flat-budget is
saturated (0.2038→0.2032, no memory circuit to mature) while hier-long keeps improving
(0.1848→0.1666, reaching the plateau floor). The seed-0-30k "dip" is explained on TWO independent
axes: seed (s1 0.1690) AND budget (60k 0.1666). Both move it to the ≈0.17 plateau.

### 3.5 The upper-edge bracket (R-F/R-G, in progress)
Author asked to fill the 1152→3072 gap with two experiments. Designed to be more than filler:
1792 and 2432 tokens **bracket the interpolated crossing (≈2059)**, so the sign flip between
them *measures* the upper edge instead of interpolating it. Both keep the 48-frame coarse span
fixed, varying only fine-frame count. **R-F @1792: hier wins +0.0077 [+0.0016,+0.0142]**
(confirmed — narrower than 1152's win, as predicted below the crossing). **R-G @2432:** resolving
(hier-c2 0.1820 done; flat-c2 retraining after the restart — see §7). Predicted: flat win →
crossing bracketed to (1792, 2432).

### 3.6 What "a token" is (recorded because it framed the whole frontier discussion)
A token = one spatial cell of a latent frame, projected to width d=384. The frozen AE gives each
64×64 frame a coarse map (16 ch × 4×4 = **16 tokens/frame**, ~16×16-px layout regions) and a fine
map (16 ch × 8×8 = **64 tokens/frame**, ~8×8-px detail). Budget = n_ctx_c·16 + n_ctx_f·64. In the
hierarchy the **coarse stream is cheap wide memory** (48 frames × 16 = 768 tok span the whole
loop), the **fine stream is expensive recent detail**; flat has no coarse stream so its budget
buys a shorter window. That price asymmetry IS the mechanism.

---

## 4. TEXT COMPLETION

**Article — all 8 chapters now exist as journal prose** (were 4 at session start):
- NEW CH1_INTRO (finalized ~230-word abstract with the frontier band), CH6_REVERSIBILITY
  (theorem + correct inverse + three-axis taxonomy), CH7_RELATED (positioning + explicit
  non-claims box), CH8_CONCLUSION (one claim + boundary conditions + two negatives).
- CH4: all 5 [BRANCH] markers resolved (disambiguation PASS → pair RED → trustworthy-memory
  boundary, §4.5.6); over-claiming fig-ch4-4 caption retracted.
- Canonical chapter order fixed (1–8); title = "When Does Hierarchy Pay?…" (the branch data
  defends). [A1] V-BLT framed as the author's **first two-stage hierarchical adaptation** (author
  instruction) in three places → the arc is [A1] two-stage hierarchy → [A2] reversible machinery
  → [A4] coarse-memory hierarchy.

**Dissertation — all 12 audit fixes applied**; 00–07 rewrites content-complete (07 = master
bibliography). 58 [AUTHOR:] flags in 4 clusters remain (the critical path); 49 [UA-PASS:] idioms
pre-placed for the final translation.

**Bibliography built** (`07_REFERENCES.md`): ~85 external entries from the full 104-descriptor
[REF:] census + author-publication slots + a descriptor→key binding table + numbering policy
(numbers assigned at assembly by order of first mention; ДСТУ 8302:2015 at the UA pass). ~12
[VERIFY] flags for PDF-checking; a `bind_refs.py` sample script drafted.

---

## 5. GENERALIZATION: TaxiBJ MEASURED (2 of 5 suites now done)

Built `taxibj_bridge.py` (the disclosed ×2-upsample-equivalent so "the unmodified WindowDiT"
stays literally true; OpenSTL [0,1] convention verified in their loader; verbatim `metric()`
scoring; author's ~14 GB regime → probe chose B=64/9.8 GB). **Result: the unchanged 22M
generalist stack lands INSIDE the OpenSTL leaderboard band on MAE (14.96, ties MIM for 2nd) and
PSNR (39.83, above all quoted rows)**, at the band's lower edge on MSE (0.3642 vs PhyDNet
0.3622), below band on SSIM — at comparable epochs, best-of-1, one GPU. Persistence beaten 10.5×;
AE ceiling 0.0192 (tokenizer not the limiter — fallback branch moot); reg>diff replicated on
real data; hier=economy-only at 4→4 as pre-registered. Filled CH5 §5.7 + Розділ 4 §4.6.5, no
leaderboard-superiority headline. (T7 MMNIST = the other measured suite, the predicted null.)

---

## 6. THE EXTERNAL SOLO REVIEW AND HOW IT WAS ANSWERED

An independent solo review (re-derived every headline number, matching our npz verification)
returned "residual issues are disclosure-grade, not validity-grade." **9 of 11 findings
accepted, 2 rejected with evidence** (commit ea9f9a7):
- **Accepted, its best catch:** the T3 "emergence" claim had two confirmed-but-undisclosed
  confounds (separate OneCycle schedules; the 20k run was a post-6k-null escalation). Added a
  four-disclosure paragraph in CH4 §4.3.1 + CH8 + Розділ 4; wording changed to "emerges under a
  larger optimization budget." (R-C later confirmed the rollout-level version cleanly.)
- **Accepted:** R-D inserted (mechanism-agnosticism was single-seed — the weakest link in a
  *stated* claim); "≈2,059" qualified as interpolated; multiplicity + tj-range scope notes;
  loop-80 A′ re-ranked above G-iii; hash-pin datasets instead of regenerating; pycache untracked.
- **Rejected with reasons:** the CH8 "stale date" (it was today's date, review ran mid-update);
  regenerating loopmom with a seed (would create a *different* dataset, invalidating all
  comparability — hash-pin instead).

---

## 7. PROBLEMS ENCOUNTERED (and how handled)

1. **npz-overwrite hazard (R-C):** `--compare` reads `rhvd_eval_{arm}_s0.npz`, which the 60k
   retraining overwrites. Flagged in the plan BEFORE launch; archived the 30k npz to
   `logs_rhcp/archive_30k/`, restored canonical 30k names after, saved 60k as `*_s0_60k.npz`.
   The T4 rollout matrix and every doc reference the 30k versions unchanged.
2. **PC restart mid-session** (during the R-F/R-G chain): lost almost nothing — hier-c1/flat-c1/
   hier-c2 all evaluated, R-F compare done. Only flat-c2 was interrupted.
3. **My flat-c2 misdiagnosis (owned + corrected):** I initially said flat-c2 "only needs eval"
   because a `_t30000.pt` ckpt existed — but that filename is the *target*-step label,
   overwritten every 1k steps; flat-c2 had only reached ~16k before the reboot. It correctly
   resumed training from there (~2.5 h, not the 15 min I first told the author). Corrected the
   ETA honestly. This is a lesson: checkpoint *filenames* encode target steps, not progress.
4. **Batch-probe granularity:** the ~14 GB target repeatedly landed at a smaller batch (TaxiBJ
   B=64/9.8 GB because B=96 overshot to 14.6; bracket arms B16/11.6 GB) — disclosed each time,
   not disobedience.

---

## 8. KEY INSIGHTS & DISCUSSIONS (portable lessons)

- **A bounded win band beats claimed dominance.** The frontier's inversions at both ends make the
  claim falsifiable and mechanistic — the strongest defense posture.
- **Seed-means, symmetric pairs, never best-of-seeds.** Hier (memory) arms are seed-variable;
  flat arms are stable. The pre-registration + symmetric-pairing guardrail is the anti-cherry-
  picking discipline that makes the two-seed claims defensible.
- **Memory is a late capability.** T3/R-C: the flat arm saturates with budget, the memory arm
  keeps maturing — so short-budget ablations spuriously kill memory mechanisms, and the
  matched-budget delta *grows* with training.
- **Structural losses are not bugs.** At window saturation, full-resolution context information-
  dominates a coarse record of the same span — no seed or budget fixes it, and it shouldn't.
- **Reversibility is the most finished strand:** experiments concluded, theory verified, framing
  honest (capability vs BPTT + enabler + two scoped negatives incl. the measured 0.26% inner-loop
  no-go). It is positioned as the enabler of future distribution-matching rollout training, not a
  demonstrated end-to-end win.
- **The remaining risk is administrative, not scientific** (two independent reviews agree).

---

## 9. CURRENT STATE (2026-07-09)

**Experiments:** three claims complete; capability boundary closed; reversibility complete;
frontier **6 points, all two-seed except the bracket** (R-F confirmed, R-G resolving); 2 of 5
generalization suites measured (T7, TaxiBJ). GPU: flat-c2 retraining (~24k/30k), then R-G compare.

**Frontier as measured (diffusion head, ALL fg-MSE, seed-mean where 2 seeds):**
| tokens | flat | hier | Δ(flat−hier) | verdict |
|---|---|---|---|---|
| 512 | 0.2215 | 0.2258 | −0.0036/−0.0051 | flat (below threshold) |
| 896 | 0.2104 | 0.1725 | +0.0399/+0.0358 | **hier** (band lower edge) |
| 1152 | 0.2041 | 0.1769 | +0.0189/+0.0354 | **hier** (headline; +0.0366 @60k) |
| **1792** | **0.1876** | **0.1799** | **+0.0077** | **hier** (R-F) |
| 2432 | (resolving) | 0.1820 | (resolving) | R-G — predicted flat |
| 3072 | 0.1572 | 0.1691 | −0.0131/−0.0107 | flat (saturation) |

**Texts:** article 8/8 chapters prose-complete; dissertation 00–07 content-complete + bibliography.
**Figures:** 10 exist (fig1–10; fig9 frontier curve + fig10 two-panel flat-vs-hier comparison);
6 renders + ~10 drawings pending. **Commits this session:** ~30 (from `2e4a961` through `28ed646`).

---

## 10. FORWARD PLAN

**GPU queue:** finish R-G (flat-c2 → compare) → fill the 6-point frontier (fig9 + fig10 regen,
swap "≈2×10³ interpolated" for the measured bracket in abstract/CH8/VYSNOVKY) → generalization
(G-i 2D rotations needs a build + the author's [A2] tables; G-ii polycubes; G-iii MMNIST-CIFAR;
KTH recommend-skip) → optional loop-80 A′ (the one experiment that answers "single instrument").

**Author-blocking (the critical path):** publication list + [A4] venue/status (existential for
the 3+1+6 count); [A2] result tables (blocks G-i/G-ii comparison rows — send early); НДР/UDC/
theses/quartile; VQ-instability receipts decision; supervisor sign-offs (reversibility scope,
UA sources, «Положення» template).

**Mechanical (mine, no blockers):** 6 figure renders (data on disk, sample code in the plan) +
~10 drawings; bibliography [VERIFY] pass → `bind_refs.py` numbering; CH4 trim + appendix C;
ALL-fg row + `sharp_img` (after the queue drains — eval-code freeze while runs are active); then
the UA translation LAST on frozen text + .docx merge.

**Honest bottom line:** the science is finished and the frontier story is *stronger* than the
original hope — a lawful band with predicted-and-measured edges, a negative control on the
field's own benchmark, and a trustworthy-memory boundary. What remains is figures, the
bibliography verification, the author's publication cluster, and the Ukrainian pass.
