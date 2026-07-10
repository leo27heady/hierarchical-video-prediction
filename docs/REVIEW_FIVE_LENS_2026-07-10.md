# FIVE-LENS REVIEW — 2026-07-10 (late), post-B3 / post-packaging-reframe state

**What this is.** A five-reviewer audit (article, dissertation, evidence forensics, code harness,
adversarial defense-opponent) of the entire project at commit `a8e99d9`, run at the author's
request against the question: *are the points strong, is the topic and implicit story good and
defendable now?* Read-only; no files were changed by the review. This document is the archival
record; the ranked action list is at the bottom.

**Headline verdict.** The science is in better shape than the framing; the framing is what gets
attacked. Evidence integrity is verified SOUND (30/30 headline numbers recomputed from raw npz
match exactly; pre-registration verified real at commit-timestamp granularity). The code that
produced every published number is trustworthy (adversarially tested: no GT leakage, exact token
budgets, correct cluster bootstrap). Two genuinely new problems were found: a CRITICAL latent
bug in the un-launched Phase C chain, and one place where two of the project's own measured
facts contradict each other in print (the upper-edge bracket vs R-C).

---

## 1. NEW CRITICAL FINDINGS (not previously known)

### 1.1 Phase C would launch contaminated — FIX BEFORE LAUNCH
`scripts/run_phaseC_mmnist.sh` (C0 block) generates BOTH `mmnistL_train_3000x100.npy` and
`mmnistL_test_500x348.npy` with `gen_moving.py`'s default `--seed 1234` (no `--seed` passed).
Per-clip RNG consumption is frame-count independent (8 draws/clip; bounce loop consumes no RNG;
`p_turn=0` → deterministic dynamics), so **test clip *i* is bit-identical to train clip *i* over
the first 100 frames and frames 100–347 are the deterministic continuation of a memorized
training clip**. Verified empirically (5/5 simulated clips identical over the shared prefix).
Eval clips (test 250–349) sit inside the training pool. Because Phase C's pre-registered primary
read is an *equivalence* verdict, the contamination biases directly toward the expected
conclusion — a silent false "REPLICATED_EQUIVALENCE". No `data/mmnistL_*` exists yet; nothing
published is affected. **Fix: pass a distinct seed for test generation + add a frame-0
byte-collision assert to the chain.** (Existing datasets checked clean: loopmom and all hgrid
train/test pairs have zero frame-0 collisions.)

### 1.2 The upper band edge (1,792, 2,432) is 30k-conditional, and R-C contradicts its defense
Ledger L2 / CH8 L2 / VYSNOVKY state the flat/hier crossover is "measured to lie in
(1,792, 2,432)" with no training-budget qualifier. The on-record defense (SPATIAL_FACTORIAL
R-F/R-G "convergence fairness": sequence-length-matched pairs → fair paired delta) is refuted by
the project's own R-C: the 1,152 pair is also sequence-length-matched, yet at 2× budget hier
moved −0.0182 while flat moved −0.0006 — matched sequence length does NOT imply matched
distance-from-asymptote; memory arms are systematically less converged at 30k. Applying R-C's
shift to hier-c2 (0.1820 @2,432) puts it ≈0.164, below flat-c2's 0.1703 — at 60k the 2,432
verdict plausibly flips and the edge moves toward (2,432, 3,072). Also: the 3,072 frontier point
is a 60k regime while 1,792/2,432 are 30k — the regime changes inside the region where the
crossing lives. **Fix (choose one): (a) budget-qualify L2 everywhere ("measured at the 30k
budget; R-C implies the asymptotic edge lies at or above this bracket"), or (b) ~10 GPU-h:
hier-c2/flat-c2 at 60k.** Derived exposure if unfixed: CH8 §8.4's "≈2.3–3× budget" multiple and
every band sentence in abstract/CH8/VYSNOVKY.

### 1.3 CRN common-noise claim is scoped narrower than stated
`sample`'s draw order makes initial noise bit-identical across arms only for same-target-shape
pairs (verified: entire 1,152 trio, hier3, min/c1/c2 pairs share noise). Cross-stride pairs do
NOT (hier-s2-min vs flat-short; hier-s4-min vs flat-320) — fine-stream noise differs. No bias
(noise is data-independent; pairing/CIs remain valid), only loss of variance shielding at
exactly the B1 (512) and E-320 cells. Any text asserting common eval noise across arms must
scope the claim.

---

## 2. EVIDENCE INTEGRITY — VERIFIED SOUND (forensics lens)

- 24 arm/seed aggregates recomputed from `logs_rhcp/rhvd_eval_*.npz`: **30/30 checks match to 4
  decimals** (strided-flat 0.1652/0.0919 exact; hier-long 0.1848; hier3 0.2044/0.0564; TaxiBJ
  0.36416/14.958/0.9781/39.832 exact from raw artifact; archive/restore cycle bit-identical).
- Ledger L1–L10 ↔ SPATIAL_FACTORIAL ↔ chapters: all values, CIs, signs, seed qualifiers
  consistent everywhere checked. The retired ≈2059 edge survives nowhere claim-bearing.
- **Pre-registration is real, not retrofitted** — commit-granularity forensics: lower-edge
  pre-reg 441076e (00:42) → npz 03:25 → measured 1e0f1ca (06:13); B2 amendment d311816 (14:54)
  → B2 npz 19:07; B3 disclosure same commit → npz 22:57; Phase-C amendment dc68ad3 (22:38) with
  zero Phase-C artifacts on disk. Expectation-baseline misses honestly recorded (B1 magnitude
  below range; B2/frontier-top "third outcome").
- Every arm the texts assert as measured has its npz; the 7 absent arms are exactly the
  [EXPECTED]/queued ones. No unmarked gap.
- Freshness gaps (LOW): CWSTYLE_TABLE.md footer still lists hier3 as pending (regen needed);
  PLAN_TEXTS §7:227 carries the superseded Phase-C "no reporting" escalation rule (ledger has
  the amended rule — stale mirror only). Blinding npz (23:12/23:27) measured but unswapped.
- Operational risk (this review's addition): the 18 GB of `logs_rhcp/` npz/ckpt artifacts that
  back every published number are untracked local files — establish a backup story before the
  defense.

## 3. CODE TRUST — published numbers stand (code lens)

Positively verified under adversarial dynamic testing (not just reading):
- **No GT leakage**: marked-prediction canary — every context frame ≥ WARM was a prediction,
  every frame < WARM was GT, all arms/streams, full rollouts; ring-buffer phase preserved.
- **Token budgets exact**: frame-exact audit of anchors/slots/targets/frame-ids for 8 arms
  (incl. hier3 48×4+24×16+9×64=1,152; strided-flat 1,088; hier-s2-min 512); `expected_seq`
  asserted every forward. Stride-2 generalization phase-correct; stride-4 bit-identical legacy.
- **Paired cluster CI correct**: identical record order asserted across arms; clip-resampling
  bootstrap with ratio-of-sums estimator; no cross-seed frame pooling anywhere.
- **hier3 bit-compat proven**: same frozen AE object/code path; `attach_top` only adds zt;
  smoke asserts `torch.equal`.
- Metrics arm-symmetric ([STD-FIX] stats shared; fg masks GT-derived). Smoke's 46 checks real
  (construction/budgets/compat/plumbing, described honestly).
- Hygiene (MINOR): `make_cwstyle_table.py` silently drops seed≥2 (S2-trio will be excluded —
  print the exclusion); resume replays batch stream from start (unbiased, not run-reproducible);
  GO-gate default ceiling is hgrid's; fg_sharp docstring 0.1 vs code 0.5; `sharp` is a repeated
  batch-mean (never CI it); Phase C identity metric conditions on detection (quote `id_scored`
  fractions); hardness/dt bins vacuous on Phase C; ~20 dead EBT/RHCP modules alongside the live
  five-file stack.

## 4. ARTICLE (docs/draft/) — reframe ~90% carried; residues are exactly the dangerous kind

Consistency sweep: band notation, all headline numbers, token arithmetic verified consistent
across every chapter. No undisclosed single-seed headline found. "Matched compute" reserved
correctly to the mechanism study. B3 wording rule respected except one stale spot.

**CRITICAL residues (each hands a hostile reviewer their opening quote):**
- CH8_CONCLUSION.md:17-18 — inside the One-claim paragraph: "≈500 tokens: no packaging can cover
  the ~48-frame loop" — refuted by the measured 512-token stride-2 WIN three paragraphs later.
  Fix: "no packaging *in the stride-1 family* / no flat window".
- CH2_TESTBED.md:5-6 — pre-reframe thesis verbatim as "the central claim of this work". CH2 is
  the only chapter with no reframe pass.

**MAJOR:**
- CH4_RESULTS.md:828-830 + CH8:99-101 — "cheapest strong configuration = reg-long (0.1682)" is
  stale post-B2 (strided-flat 0.1652 @1,088 tok s0) and the surviving comparison is
  best-of-seeds vs seed-mean — disallowed by the campaign's own rule. Retire the superlative.
- CH8:114 — "the shape study's depth null [EXPECTED-B3]" stale on two axes (measured; CI-clean
  loss not null); contradicts L5 in the same file.
- CH5 §5.1:33-38 — recommends coarse packaging with no B2 caveat; chapter never re-swept after
  2026-07-10 (header dated 07-06).
- Table 4.7 bottom row missing the R-E seed-1 annotation (understates evidence vs CH8/ledger).
- Bare "hierarchy wins" bound to coarse span in CH1 C4/§1.3 move 3 and the fig9 caption —
  defensible under the §1.1 definition but quote-vulnerable; use "long-span packaging".
- ARTICLE_SKELETON.md:3,25,97,237-239 — old thesis in spine/title/§7; integration hazard.
- Shape: still 8 chapters; CH4:3 literally says "claims of the dissertation"; §4.3.6/§4.5 carry
  commit hashes, pre-reg block quotes, chain sentinels in main text (lab-notebook register);
  CH1 front matter still gives EBT/reversibility 2/5 contribution slots + ~20% of the abstract
  despite the NON-CORE triage. The 6-section restructure exists as a plan (CH1 §1.5) only.
- Submission blockers = exactly the ledger queue (S1-a/S1-b, reg twins, Phase C) + mechanical
  (Table 4.6 paste, A2 rows, [REF:] resolution, ~8 queued plots).
- MINOR: table numbering out of order (4.7 before 4.4-4.6); three colliding figure-ID schemes;
  CH8 status header stale; hier3-vs-flat null quoted w/o CI; abstract ~330 words vs advertised
  ~250; all-black-beats-CW-VAE claim still owes its citable appendix table.

**Venue verdict:** after the queue lands + fixes: major-revision → accept at TMLR-class /
measurement track; borderline at a flagship main track ("one synthetic testbed, one frozen AE,
one backbone" draws a reject vote regardless of rigor). As drafted: major revision.

## 5. DISSERTATION (docs/dissertation/rewrite/) — split personality; framing files lag the reframe

**CRITICAL:**
- C1 — the reframe never reached the захисні положення, Task 5, result 5, or Chapter 1's
  synthesis: 01_VSTUP.md:235-240 (положення 1: coarse level carries long memory — B2 refutes it
  as written), 01_VSTUP.md:128-133 + 02_ROZDIL1.md:1183-1187 (Task 5), 00_FRONT_MATTER.md:177-183
  (result 5 — while VYSNOVKY conclusion 5 is fully reframed: the two "must-sync" files assert
  different theses), 02_ROZDIL1.md:810-814, 1081-1084, 1117-1120, 1202-1203. ~10 sites, wording
  level. An opponent who reads §4.4.3 next to положення 1 wins the exchange.
- C2 — RoPE-ND contradiction: 03_ROZDIL2.md:341-359 + 942-949 claim RoPE-ND is "the mathematical
  basis of the inter-level transfer developed in Chapter 3"; 04_ROZDIL3.md:642-649 says the
  implemented model uses learned absolute frame-index embeddings (RoPE-ND lives in [A1]'s
  V-BLT). Положення 4 (01_VSTUP.md:254-257) also binds RoPE-ND to hierarchy levels. One
  quotation at the defense collapses novelty item 4. Fix via the option-(a) marker
  ("level-appropriate instruments"; §2.2/§2.5/положення 4 rewrites).
- C3 — [EXPECTED-B2-s1] inside the «вперше» novelty formula itself (00_FRONT_MATTER.md:214-215,
  01_VSTUP.md:192-193) and 7 markers in numbered VYSNOVKY results. A novelty claim cannot be
  conditional on a queued seed; the queue (S1-a/S1-b) is exactly what resolves them — the text
  must not freeze before it drains (or retreat to the seed-0-scoped VYSNOVKY form).
- C4 — the Ukrainian dissertation does not exist yet: all 8 files are English working text with
  [UA-PASS] headers; §4.4 is mixed-language mid-sentence (05_ROZDIL4.md:779-800) and needs
  authoring, not mechanical translation. This is the true bulk of remaining text work.

**MAJOR:** FRONT_MATTER desync (ЗМІСТ:360-361 old ROZDIL3 title; Chapter-4 abstract has no band /
no inversion / no shape study / asserts the seed-0 h≥25 onset unqualified while ROZDIL4:625-629
scopes it; keywords lead with coarse-memory); the sync-direction note (06_VYSNOVKY.md:247-249
"resolve toward the front matter") is written backwards post-reframe — executing it would undo
the reframe; head-selection inconsistency (02_ROZDIL1.md:797-798 "measurement selects diffusion"
vs §4.2.5/§4.6.1 regime-dependent + reg deployment pick); upper-edge bracket quoted without the
single-seed-interior disclosure at 06_VYSNOVKY.md:195-196 and 05_ROZDIL4.md:762 (ledger L2
requires it); novelty item 1 is a ~150-word single sentence — split into (i) вперше =
methodology (safe, genuinely unclaimed) + (ii) the packaging demonstration as its result;
MilliVid citation still a [VERIFY] placeholder (07_REFERENCES.md:76) while novelty 1's
adopted-prior-art defense leans on it; 0.2%-vs-0.26% drift at 03_ROZDIL2.md:974.

**Positive findings:** ROZDIL3/ROZDIL4 are in good post-reframe shape; the §3.1 stream-family
vocabulary IS used downstream (taxonomy, edge-law formula, blinding) — not defined-then-
abandoned; number consistency across VYSNOVKY/ROZDIL4/FRONT_MATTER verified; no encoding damage;
VYSNOVKY conclusion 7 correctly carries the conditional decision rule. Novelty judgment:
item 1 defensible after the split; item 2 (reversibility) real and defensible (published,
honestly scoped); item 3 real, modest, correctly graded; item 4 weakest — defensible at
«удосконалено» only after C2. The bundle is one dissertation by the 124 genre (competent
backbone-question staple; results 1→2→5→6→7 form a chain; 3 and 4 instrumental branches) —
but VSTUP's "the memory barrier ... is removed by reversible transformations" overpromises
(reversibility is used nowhere in the headline campaign; VYSNOVKY 4 answers this honestly —
soften the VSTUP verb to "creates the resource precondition").

**Defense verdict:** pass, most likely with minor revisions — IF C1/C2 are fixed and the queue
lands/retreat-branches execute. The three questions most likely in the room (strided-flat beat
your model; where does reversibility touch Chapter 4's numbers; two seeds + one synthetic stand
→ "laws"?) are all answerable from existing text once the framing files stop contradicting it.

## 6. THE STORY — adversarial verdict

- **Tautology charge: survives, with a bruise.** Near-analytic parts ("span wins where span is
  needed") are saved by the non-analytic findings: immediate switch-on at the affordability
  threshold (T3 shows retrieval is a late capability — this was genuinely uncertain), effect
  peaking at the lower edge, flat winning at saturation, the +0.017 family plateau offset, the
  loop-213 rollout failure, and head regime-dependence. The saturation inversion is the
  strongest anti-tautology card (a rigged testbed would not produce it).
- **B2 absorption: half-legitimate.** The rate/resolution taxonomy is factually grounded
  (CW-VAE's levels ARE temporal) and was defined before B1/B2 landed. But under the expanded
  definition, everything except a contiguous dense window is "hierarchy" — the hierarchy claim
  is no longer falsifiable; the falsifiable content correctly lives in the span/edge/saturation
  laws. Finish the retreat: "context packaging/allocation" as the load-bearing noun everywhere;
  "hierarchy" demoted to the taxonomy section. Article ≈80% there; dissertation ≈50%
  (title, Chapter 3 framing, положення 1 lag).
- **What would falsify the thesis (state this in one place in the texts):** (i) a matched-budget
  long-span packaging that covers the horizon and loses inside the band; (ii) band edges failing
  the family token arithmetic (E-320 is a live shot); (iii) long-span winning below its own
  threshold; (iv) hier winning at saturation — (iv) already fired in the project's favor.
- **Novelty:** the instrument (exact analytic re-entry labels + matched-budget causal design) and
  the measured band with pre-predicted family-parametric edges are genuinely new; the saturation
  inversion and (if REG twins confirm) the head×shape interaction are the most citable laws.
  BUT strided/keyframe conditioning is the field's default practice (FDM, video-LM frame
  subsampling, world-model anchor frames) and CH7 does not say so — add the one honest
  paragraph ("our contribution is measuring it against spatial coarsening at matched budget with
  exact memory labels, not proposing it") before a reviewer writes it less kindly.
- **Wording:** "lands exactly at the geometric prediction" overstates — the switch-on is
  bracketed (512, 896], nothing probed between; say "consistent with the arithmetic prediction
  at both tested budgets". Prefer "empirical regularities under stated scope" (UA: «емпіричні
  закономірності за фіксованих умов») over "laws" — costs no content, removes a target.
- **Statistics:** cluster-CI discipline exemplary; seed inference thin (2 seeds, magnitudes swing
  ~2×, shared CRN eval set, 1,792 point inside one seed-sigma of zero). The queue (S1s, S2 trio,
  Phase C 3-seed, equivalence margins) is the right minimum; report seed-level effects as the
  primary unit.
- **Deepest scope hole:** every edge is measured at loop≈48 — the span-threshold law has never
  been tested by varying the horizon. A single loop-96 point (predicted hier edge
  96×16+128 = 1,664 tokens) would convert L1 from one testbed's constant into a dose-response
  in the variable the law is about. Arguably more valuable than E-320.
- **Pre-registration:** real discipline (unwelcome outcomes kept and promoted; amendments
  verifiably precede runs). Keep describing it concretely ("pre-committed analysis plans,
  version-controlled before runs, commit X") — do not oversell as registry-grade.

## 7. RANKED ACTION LIST

**Before any further GPU spend:**
1. Fix the Phase C seed bug + collision assert (§1.1) — one flag; MUST precede Phase C launch.
2. Reconcile R-C vs the upper-edge bracket (§1.2): budget-qualify L2 wording everywhere, or
   queue the c2 pair at 60k (~10 GPU-h).

**Wording surgery (half a day, all sites enumerated above):**
3. Article: CH8:17 one-claim fix; CH2:5 opener; reg-long superlative (CH4:828/CH8:99);
   CH8:114 stale B3; CH5 §5.1 B2 caveat; Table 4.7 R-E annotation; C4/fig9 "long-span
   packaging"; skeleton spine.
4. Dissertation: положення 1 + Task 5 + result 5 + ROZDIL1 synthesis (~10 sites);
   RoPE-ND joint (option a); FRONT_MATTER resync (title/abstract/keywords) + invert the
   sync-direction note; VSTUP reversibility verb; novelty-1 split; upper-edge disclosure ×2;
   0.26% fix.
5. CH7: the sparse-conditioning-is-standard-practice paragraph. State the falsification
   conditions once. "Exactly at the prediction" → "consistent with". Consider laws→regularities.

**Queue (unchanged, correct):** S1-a → S1-b → REG twins + E-320 → S2 trio → Phase C (after fix)
→ G-i/G-ii. Then strip [EXPECTED] from novelty/VYSNOVKY per branch outcomes.

**Structural (before submission/defense):** article 6-section restructure + register scrub of
§4.3.6/§4.5; UA pass (the true bulk; §4.4 needs authoring); MilliVid + [VERIFY] citations;
all-black appendix table; figure-ID unification; CWSTYLE regen (hier3 row) + seed≥2 exclusion
print; npz backup story (18 GB untracked artifacts back every number).

**Recommended additions (cheap, high-value):** loop-96 edge point (dose-response in the horizon
variable); the one rehearsed one-sentence UA answer to "what did you discover" (drafted in the
opponent report, §S7).
