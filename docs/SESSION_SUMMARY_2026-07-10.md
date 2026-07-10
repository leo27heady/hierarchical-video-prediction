# SESSION SUMMARY — the shapes-of-hierarchy session (2026-07-10)

**Scope.** One working day that took the project from "frontier complete, texts content-complete"
to: the frontier's upper edge *measured* (R-G bracket), the author's hier-trend concern resolved
into a **diagnosed law** (plateau + convergence artifact), a full **CW-VAE-style shapes program**
designed, pre-registered, infrastructured, and half-measured — including the campaign's biggest
surprise (B2: strided full-res history *beats* the coarse hierarchy) — with the article and
dissertation refactored around a new "shapes of hierarchy" narrative under an
expectation-baseline discipline (predictions committed before results, both branches pre-drafted,
placeholder-swap on landing). Predecessor: `docs/SESSION_SUMMARY_2026-07-09.md`. Primary evidence
trail: `docs/SPATIAL_FACTORIAL.md` (all pre-registrations + measured blocks, in order).

---

## 1. THE ARC

R-G resolved the frontier bracket (flat wins at 2432; upper edge measured in (1792, 2432)) →
the author asked why the hier curve "oscillates ≈0.17" while flat descends → CPU-only diagnosis
from existing logs/npz: **the hier side is a lawful plateau** (span protocol-capped at the
48-frame loop; extra fine frames ~94% redundant) plus a **fixed-step under-convergence artifact
that scales with context length** (proven from teacher-forced val-h4: 0.027→0.034→0.044 at
2f→16f→26f; disappears at 60k) plus regime mixing in the figures → the author proposed an
81-cell factorial (temporal × spatial × depth × head, CW-VAE-style) → review showed ~60 cells
degenerate/impossible/predetermined; carved down to a pre-registered program of claim-bearing
cells → infrastructure built and smoke-tested in an afternoon → Phase B launched → **B1
confirmed the family edge law** (stride-2 coarse wins at 512 = its arithmetic prediction) →
**B2 surprised** (strided full-res history beats the coarse hierarchy at fewer tokens) → texts
swapped per pre-registered branches, claim softening applied under the pre-committed amendment,
seed-1 replications auto-queued. B3 (depth) + blinding probes resolve overnight; REG twins +
E-320 tomorrow.

## 2. MEASURED THIS SESSION (all seed 0 unless noted; cluster CIs; loopmom, diffusion head)

| result | numbers | meaning |
|---|---|---|
| **R-G @2432** (prior session's chain, resolved today) | flat wins −0.0116 [−0.0162, −0.0073] | with R-F (+0.0077 @1792): the win band's upper edge is **measured** to lie in (1792, 2432); the ≈2059 interpolation retired across all texts |
| **B1 `hier-s2-min` @512** (24c@stride2+2f) | **+0.0330 [+0.0246, +0.0419]** vs flat-short; WIN 14/14 cells; dt≤48 HARD +0.0643 → dt97+ null (exact span signature) | **the band's lower edge is family-parametric**: moved 896→512 when the coarse tick halved, at the token-arithmetic prediction both times. Honest refinement vs the committed expectation (Δ predicted +0.04..+0.05): the stride-2 family sits ≈+0.017 above the stride-1 plateau — **each family carries its own plateau level** (sparser anchors cost). Seed-1 queued (binding rule) |
| **B2 `strided-flat` @1088** (11 full-res@stride4+6f, NO coarse level) | ALL **0.1652** / sharp 0.0919; beats hier-long **+0.0197 [+0.0143, +0.0252]**, beats flat-budget +0.0386 — at a 64-token deficit; wins hard re-entry bins at every horizon (h1–8 HARD +0.0482) | **third outcome, stronger than the pre-registered primary**: span-per-token is the currency, and temporal subsampling of full-res history is the *better* packaging under the diffusion head — the G-S1 abstraction cost expressed at rollout scale. Advantage peaks at h25–60 (+0.0344) and decays to +0.009 by h121+ (echo of compounding self-feeding quality, not a late-horizon mechanism). Single-seed caveats disclosed (hier-long s0 is its weak seed; margin vs seed-mean ≈0.012) → **S1-b required, queued** |
| B3 `hier3` @1152 (48top(2×2)+24c+9f) | resolving (~23:50) | pre-registered: null vs hier-long (depth bounded by the data's single memory scale); loss-dilution confound pre-disclosed |
| A2 blinding probes | queued after B3 | level-content attribution (CW-VAE §5.1 analog); OOD caveat binding |

**The plateau diagnosis (zero GPU, from existing artifacts):** hier arms ≥896 tok all share the
protocol-capped 48-frame span → plateau ≈ flat-long's floor + abstraction cost ≈ 0.169 ✓; the
wiggle = fixed-step convergence penalty growing with context length (R-C: 60k moves hier −0.0182,
flat −0.0006) + seed spread (≤0.016) + 30k/60k regime mixing in fig9/10. Memory:
`hier-plateau-diagnosis`.

## 3. THE INTERPRETIVE FRAME (author-approved)

- **"Hierarchy" = multi-RATE and/or multi-RESOLUTION context** (defined in CH4 §4.4 opener).
  The axes are independent; strided-flat is a two-rate temporal hierarchy (the clockwork/CW-VAE
  quadrant — CW-VAE's own levels are temporal, not spatial). B2 therefore reads "the multi-rate
  shape beats the multi-resolution shape at this operating point," NOT "hierarchy is unnecessary."
- **Three binding scope qualifiers on B2** (§4.4.3 both texts): static world = maximally kind to
  temporal subsampling (nothing changes between anchors; dynamic content would alias); the
  abstraction cost is a property of THIS frozen AE's 4×4 layout code; the winner is itself a
  hierarchy in the rate sense.
- **Claim-1 wording softened at s0 per the pre-committed amendment** ("hierarchical packaging" →
  "long-span packaging" at CH4 §4.3.2 + CH8 §8.1); the POSITIVE span claim stays s1-gated.
- **No conflict with CW-VAE** — arguably a vindication of its (temporal) axis over the adopted
  MilliVid-style (spatial) axis, at this operating point, under the diffusion head. The
  head × shape interaction is the open question the REG twins answer (G-S1: abstraction is FREE
  under regression → coarse may win back a head).

## 4. PRE-REGISTERED & QUEUED (all committed BEFORE their runs)

Queue (one job at a time): B3 → blinding → `run_cwstyle_seed1.sh` (S1-a hier-s2-min s1;
S1-b strided-flat s1 — REQUIRED after B2) → `run_cwstyle_regtwins.sh` (E-320: hier-s4-min
12c@stride4+2f = 320 tok vs flat-320, the third edge-law point, GO-gate first; REG twins:
hier-s2-min-reg + trained flat-short-reg partner, strided-flat-reg, hier3-reg — the head axis at
every new cell; ~13–16 GPU-h) → Phase C (standard-MMNIST long-horizon control column, ~1 GPU-day)
→ G-i/G-ii generalization (author's [A2] tables gate the comparison rows) → G-iii-vs-loop-80-A′
decision (lean A′). Amendments in force: B2 seed asymmetry (softening ok at s0; positive claim
needs s1), B3 loss-dilution disclosure, blinding OOD caveat. Expected outcomes with numeric
ranges: `docs/PLAN_TEXTS_HIERARCHY_SHAPES_2026-07-10.md` §7 (the expectation-baseline table,
filled as results land — B1 and B2 rows already show measured-vs-expected).

## 5. TEXTS — what changed today

Article: CH4 §4.4 retitled **"the shape of the abstraction: spatial, temporal, deep"** (opener
thread + hierarchy definition + Table 4.5 taxonomy + §4.4.1 factorial demoted intact + new
§4.4.2 edge law [B1 measured] + §4.4.3 span control [B2 measured + 3 qualifiers] + §4.4.4
blinding + §4.4.5 master grid/depth); CH1 program thread + six-budget fixes; CH2 stream
generality + §2.8 SSIM carve-out; CH5 §5.3.1 Phase-C shell; CH7 CW-VAE-methodology paragraph
(adopted format / three tightened controls: budget matching, capacity-neutral depth, fg-masked
metrics) + non-claim 3a; CH8 shape bullets + sizing rule + multi-scale visible-slow-factor
future work. Dissertation: ROZDIL1 methodology paragraph; ROZDIL3 §3.3.5 configuration-family
formalization; ROZDIL4 full §4.4 mirror + §4.6.2 Phase-C shell; DIGEST +11 UA-EN terms;
bibliography +Kim2019vta +Koutnik2014. Figures: fig10 redesigned (hier plateau BAND + regime
hatching — the answer to the trend concern, drawn), fig12 (degradation curves, auto-regen),
fig13 (family edge law, built from B1). New tooling: `scripts/check_text_consistency.py`
(phrase-lint; caught 6 stale refs on day one; numeric-invariant layer removed as
false-positive-prone — numbers are protocol-scoped).

## 6. INFRASTRUCTURE BUILT TODAY (all CPU-smoked green, 46 checks, legacy bit-compat asserted)

7-field ARMS (optional top stream `n_ctx_t, stride_t`); 2×2 top AE level as a FROZEN-TRUNK
extension (`hvp_rev/hier3.py` — base latents bit-identical, every old measurement stays valid);
**two real bugs caught pre-launch**: the strided-slot rule (one slot/window) was stride-4-only —
generalized to WIN//stride (stride-2 phase-correct, stride-4 bit-identical); coarse frame-id
anchoring assumed coarse is the deepest stream — generalized; GO-gate empty-bin nan fix +
`--go_ceiling`; `--blind coarse|fine|top` eval probe (distinct npz tags; deliberately NOT
wandb-logged — the run-id resume scheme would contaminate canonical cloud runs); wandb ON for
all training arms (project `hgrid`, per-(arm,seed) runs, config incl. new stream fields).

## 7. HONEST ASSESSMENT

- **B2 is the finding of the campaign so far.** It converts the article from "hierarchy wins a
  band" into "the band belongs to long-span context; *which shape* of hierarchy is
  operating-point- and head-dependent" — a stronger, more falsifiable story. Its three risks
  (seed, head, static-world) are each covered by an already-queued run or an already-written
  qualifier. If strided-flat-reg shows coarse ≥ strided under regression, the head × shape
  interaction becomes the paper's most memorable law.
- **The pre-registration + expectation-baseline discipline carried the day.** B1 landing outside
  its predicted magnitude range produced a *discovery* (family plateau offset) instead of a
  quiet rewrite; B2 landing outside both branches produced a pre-committed escalation path
  (amendment → S1-b) instead of a scramble. This discipline is the project's moat; keep it.
- **The author's instincts have been repeatedly load-bearing**: the trend concern → the plateau
  law; the CW-VAE table request → the shapes program; the temporal-axis intuition → B1/B2. The
  agent's role was mostly to carve requests into falsifiable, budget-honest form.
- **Watch-items:** memory-arm seed variance (~0.016) remains the biggest statistical hazard —
  never quote single-seed margins without the spread; CH4 is heavy (fold §4.3.x→§4.3.6 + ~15%
  trim in the same pass as B3's swap); [EXPECTED] tags ≈ 105 across texts (most resolve with the
  queue; CH5's 62 await G-i/G-ii); the **author cluster is the true critical path to the
  defense** (publications list, [A4] venue/status, [A2] tables, НДР/UDC, supervisor sign-offs)
  — no GPU run substitutes for it.
- **Not final yet:** after the queue drains, remaining = figure renders/drawings (E4), the
  all-black-beats-CW-VAE citable appendix table, bibliography [VERIFY] pass, CH4 fold+trim,
  assembly (E8), UA pass (E9), packaging (E10). Article ≈ 1 week of text work + ~3 GPU-days
  from submission-ready-except-venue; dissertation additionally runs through the author cluster.

## 8. FILES TO REVIEW (cold-start map)

**Evidence & plans (read in this order):**
1. `docs/SPATIAL_FACTORIAL.md` — THE evidence trail: every pre-registration + measured block,
   chronological, including today's B1/B2 blocks and amendments.
2. `docs/SESSION_SUMMARY_2026-07-09.md` → this file — the two-day narrative.
3. `docs/PLAN_TEXTS_HIERARCHY_SHAPES_2026-07-10.md` — the text-change plan + expectation-baseline
   table (measured-vs-expected, filled live).
4. `docs/CWSTYLE_TABLE.md` — the auto-generated shape grid (regen:
   `scripts/make_cwstyle_table.py`).
5. `docs/HGRID_DESIGN.md` — testbed B (temporal boundary study, trustworthy-memory result).
6. `docs/COMPLETION_PLAN_2026-07-07.md` — the E-phase tracker to "final".

**Code (the working stack):**
- `hvp_rev/rhvd.py` — the whole campaign harness: ARMS registry (~line 51; 7-field configs),
  `WindowDiT` (joint-attention multi-stream DiT), `WindowDiffusion`/`WindowRegression` heads,
  `window_batch` (stream slicing), `rollout_eval` (ring-buffer AR rollout, CRN, GO-gate oracle,
  `--blind`), `compare` (paired cluster-CI tables). Single most important file.
- `hvp_rev/hier3.py` — frozen-trunk 2×2 top AE level + `attach_top` (zt latents).
- `hvp_rev/faithful_nets.py` — tokenizer (n_levels), DiT blocks, EBT nets (mechanism study).
- `hvp_rev/hier_convae.py` — the frozen 2-level AE (coarse 4×4 / fine 8×8, per-level decodable).
- `hvp_rev/step1_mech.py` — `ActionHierData` (latent encoding + [STD-FIX] standardization),
  metrics (fg-MSE, fg-sharp, `decode01`).
- `scripts/gen_looping.py` — the loopmom testbed generator (192-torus, momentum pan, actions).
- Chains: `scripts/run_cwstyle_phaseB.sh` / `run_cwstyle_seed1.sh` / `run_cwstyle_regtwins.sh`;
  smoke `tests/smoke_cwstyle.py`; lint `scripts/check_text_consistency.py`; figures
  `scripts/make_fig{9,10,13}*.py`, `make_cwstyle_table.py` (fig12 inside).
- Data/artifacts: `logs_rhcp/rhvd_eval_{arm}_s{seed}[{tag}].npz` (per-frame records: h, hard,
  dt, seen, fg, sharp, pers_fg, clip), `logs_rhcp/cwstyle_*.log` (chain logs), wandb project
  `hgrid`, checkpoints `logs_rhcp/rhvd_{arm}_s{seed}_t{steps}.pt` (filename = TARGET step).

**Texts:** article `docs/draft/CH1..CH8` (CH4 = the results spine; CH2 = the testbed/protocol);
dissertation `docs/dissertation/rewrite/00..07` + `DISS_DIGEST.md` (locked author decisions §8,
UA-EN terminology §5). Memory index: `~/.claude/.../memory/MEMORY.md` (start-here pointers).
