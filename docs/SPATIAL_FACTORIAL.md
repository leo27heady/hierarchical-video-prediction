# Spatial-abstraction factorial — Claim-3 isolation (pre-registered 2026-07-04)

**Claim 3 (author's vision):** spatially downsampled (coarse) history is a *more abstract, less
distracted* memory representation. Until now no arm isolated spatial resolution: every hierarchy
arm confounded coarse resolution with a longer window bought by its cheaper tokens (external
review finding, accepted).

## Design: window × history-resolution grid on loopmom (testbed A, shared frozen AE, seed 0)

Recency is held equal (all coarse-history arms keep the 6 most recent fine frames, exactly the
hier-long design); the factor is the RESOLUTION of the history representation at a FIXED window.

| window | full-res history (8×8/frame) | coarse history (4×4/frame) |
|---|---|---|
| 18 frames | **flat-budget** (18f, 1152 tok) — EXISTS | **spat18** (18c+6f, 672 tok) — NEW |
| 48 frames | **flat-long** (48f, 3072 tok) — EXISTS | **hier-long** (48c+6f, 1152 tok) — EXISTS |

New runs: `spat18` (diff) + `spat18-reg` (mechanism-agnosticism twin of the W18 contrast,
vs the existing reg-flat). Protocol identical to the existing campaign (B16 × 30k, bf16,
loopmom train/test, WARM=48, rollout h≤152, 100 clips, cluster CIs, empty-fg excluded) so all
existing npz records pair exactly.

## Pre-registered reads (estimation-first, not pass/fail)

- **G-S1 (abstraction sufficiency, W18):** paired delta flat-budget − spat18, ALL + HARD bins,
  cluster CI. Claim 3 is supported if the coarse-history arm loses little (delta CI within ~1
  persistence-normalized point) or wins HARD bins ("less distracted"), at **58% of the tokens**.
- **G-S2 (interaction):** does the abstraction cost shrink/grow from W18 to W48?
  (flat-budget−spat18) vs (flat-long−hier-long), both CIs reported.
- **G-S3 (window at fixed representation):** hier-long vs spat18 (48 vs 18 window, coarse
  history, 1152 vs 672 tok) — complements the existing window-ordering law.
- Honest risk carried: the per-pixel spatial-gate null on MMNIST (project memory) — a clean
  isolated null here is a sayable Claim-3 outcome; the claim is TESTED either way.

## Omitted corner (declared): temporal-only (strided full-res history)

The strided slow-clock mechanism is documented as nulling on testbed B (HGRID_DESIGN addendum);
re-importing stride into testbed A would test a mechanism, not the resolution claim. The W×R
grid above answers Claim 3 without it.

## Cost

spat18: ~736-token seq, ~0.25 s/step × 30k ≈ 2.2h + eval. spat18-reg: same. Total ≤ 4.5h GPU,
both today; seed-1 replication of the Claim-1 headline pair (hier-long + flat-budget, original
protocol) runs overnight after.

---

## MEASURED RESULTS (2026-07-04/05, seed 0; logs: rhvd_eval_spat18{,-reg}_s0.npz, compare runs)

Full factorial (fg-MSE ALL over 152 AR frames, cluster CIs; tokens in parentheses):

| window | full-res history | coarse history | paired delta (full − coarse) |
|---|---|---|---|
| 18f, diff | flat-budget 0.2038 (1152) | spat18 0.2120 (672) | **−0.0082 [−0.0115,−0.0050] LOSS** (short-h concentrated, null by h>120; sharp 0.058 vs 0.050) |
| 18f, reg | reg-flat 0.2015 (1152) | spat18-reg 0.2008 (672) | **+0.0008 [−0.0013,+0.0029] null** (h61-120 HARD +0.0047 WIN; tiny h1-8 loss) |
| 48f, diff | flat-long 0.1564 (3072) | hier-long 0.1848 (1152) | −0.0284 (from T4; 2.7× token gap) |

**G-S1 verdict (head-dependent):** for the DIFFUSION head, coarse history is a small, real,
short-horizon-concentrated cost (−0.008 ALL at 42% token savings). For the REGRESSION head —
the better rollout head (H2 reversal) — coarse history is **statistically free** (null ALL at
42% savings, long-horizon HARD slightly better). The "intrinsically-better abstraction"
hypothesis is NOT supported; the supported Claim-3 form is ECONOMIC + head-conditional:
abstraction is lossy compression whose savings buy the memory window, and with the regression
head the loss term is ~zero.
**G-S2 (interaction):** the abstraction cost per saved token is similar at both windows for
diffusion (~0.007-0.008 per 480 tokens); no evidence the cost shrinks with window.
**G-S3 (window at fixed representation):** hier-long 0.1848 (W48) vs spat18 0.2120 (W18):
window extension via coarse tokens buys −0.027 — dwarfs the abstraction cost, same law as T4.
**Combined-optimum readout:** spat18-reg (672 tok) 0.2008 ≤ flat-budget-diff (1152 tok) 0.2038 —
regression + coarse history at 58% tokens matches the diffusion flat arm; with W48 (reg-long
0.1682 @1152) the recipe "reg head + coarse history + long window" is the cheapest strong
configuration measured.

**Claim-1 replication (same night):** hier-long vs flat-budget at seed 1 = WIN in EVERY cell
(ALL +0.0354 [+0.0267,+0.0447]; HARD h25-60 +0.0622; sharp 0.088 vs 0.057). Two-seed status:
CONFIRMED (seed-0 +0.0189). Honest nuance: onset boundary is seed-variable (s0 null h<25, s1
wins from h1-8; hier-long seed spread 0.1848→0.1690 over a stable flat baseline) — the robust
claim is the win and its mid-horizon peak, not the exact onset frame.

---

## FRONTIER COMPLETION (pre-registered 2026-07-06, author request: every flat point gets a
## same-budget hierarchical partner)

Gap: flat-long (3072 tok) and hier-short (480 tok) had no budget-matched partners. New arms:
**hier-rich** (48c+36f = 3072 tok, B8x60k sample-matched to flat-long's disclosed regime) and
**flat-short** (8f = 512 tok ~ hier-short's 480, B16x30k). PRE-REGISTERED READING: at 3072
tokens the window is SATURATED at 48 frames by the T=100 train clips — hierarchy cannot buy
more window, so per the factorial's abstraction cost the honest expectation is hier-rich ~<=
flat-long (CONVERGENCE at the frontier top). The efficiency claim's correct form is "the
hierarchy curve reaches a given quality at fewer tokens and converges with flat where the
window saturates" — a frontier-from-below, not dominance at every budget. At the low end,
hier-short vs flat-short tests the frontier's bottom. Diffusion heads first; reg optional.

### FRONTIER MEASURED (2026-07-07; frontier_{top,bottom}.log, rhvd_eval_{hier-rich,flat-short}_s0.npz)

| budget | flat | hier | paired ALL delta (hier-favorable >0) | verdict |
|---|---|---|---|---|
| ~500 | flat-short 0.2224 (512, sharp 0.0432) | hier-short 0.2260 (480, 0.0392) | -0.0036 [-0.0063,-0.0010] | flat WINS |
| 1152 | flat-budget 0.2038 | hier-long 0.1848 | +0.0189 s0 / +0.0354 s1 | HIER WINS (2 seeds) |
| 3072 | flat-long 0.1564 (sharp 0.0996) | hier-rich 0.1695 (0.0898) | -0.0131 [-0.0201,-0.0069] | flat WINS |

READING (resolves the pre-registered dual branch to the third, measured outcome — "inverts at
both ends", stricter than the pre-registered convergence): hierarchy pays in the MID-budget
regime where coarse tokens buy memory window that (a) exceeds what fine tokens afford and (b)
matters (window >= memory horizon scale). Below the memory-relevance threshold (~500 tok, 6-8f
window) coarse slots are redundancy and flat's extra fine frames win; at window saturation
(3072 tok, 48f = clip-bounded) coarse slots displace full-res frames and pay the abstraction
cost (-0.0131, consistent with G-S1's diffusion penalty). The 1152 headline is untouched.
Claim-1 wording upgrade: "hierarchy dominates the token-budget frontier exactly where window
extension is both possible and valuable" — a bounded, dose-response-supported efficiency claim.

---

## FRONTIER LOWER EDGE (pre-registered 2026-07-07, written BEFORE the runs)

**Motivation (author question):** the 3-point frontier shows hier winning only at 1152. The
win-band theory is geometric: hier wins iff its budget affords a coarse span >= the memory
horizon (~48 frames = 48x16 = 768 coarse tokens + a minimal fine tail) while flat's same budget
affords a window < 48. That predicts the lower edge of the win band at ~896 tokens — currently
UNMEASURED (the gap between 512 LOSS and 1152 WIN). Nailing it upgrades "hier wins at one
budget" to "hier wins across the band [~900, 3072) — flat needs up to 3.4x the tokens to
match", with both band edges predicted from geometry BEFORE measurement.

**New arms (standard regime, B16x30k bf16, seed 0, loopmom, diffusion head):**
- **hier-min** = 48c + 2f = 48*16 + 2*64 = **896 ctx tokens** (span 48 >= loop ~48; minimal
  recent-fine tail; actions are per-target-frame so motion does not depend on fine history).
- **flat-min** = 14f = **896 ctx tokens** (window 14 << loop 48; exact budget match 896=896).

**PRE-REGISTERED PRIMARY READING:** hier-min WINS vs flat-min on ALL and on HARD h>=25 bins
(cluster CI > 0) — the win band's lower edge is at or below 896, matching the geometric
prediction; the frontier figure gains a 4th point and the efficiency claim its measured band.

**PRE-REGISTERED ALTERNATIVE (fine-context floor):** if 2 recent fine frames are insufficient
for local detail continuity, hier-min may go null/LOSS despite spanning the loop — then the
lower edge sits between 896 and 1152 and is set by the fine floor, not the coarse span. This is
reported as a measured boundary refinement, NOT a failure of the band theory (the span
mechanism is already isolated by mem_gate T3 and the R2 reach-causality read). Contingency (run
only in this branch, not by default): one pair at 1024 tokens (48c+4f vs 16f).

**Decision rule (binding):** report whichever branch measures, no soft reads; two-seed rule
before any claim escalation; no other arms authorized by this block.

### FRONTIER LOWER EDGE MEASURED (2026-07-07; `logs_rhcp/frontier_low.log`;
### npz `rhvd_eval_{hier-min,flat-min}_s0.npz`; chain marker FRONTIER_LOW_DONE)

**PRIMARY branch measured — stronger than predicted: hier-min WINS EVERY cell.**
- hier-min ALL fg-MSE **0.1706** / sharp 0.0861; flat-min 0.2106 / 0.0516 (persistence 0.2552).
- Paired Δ ALL **+0.0399 [+0.0299, +0.0501]** — the LARGEST hierarchy delta of the campaign
  (vs +0.0189 s0 / +0.0354 s1 at 1152 tokens).
- WIN in every horizon × hardness bin (from h1-8 easy +0.0128 [+0.0049,+0.0228] to h25-60 HARD
  **+0.0790 [+0.0561,+0.1016]**). dt-gates: dt≤48 HARD +0.0750 WIN; dt 49–96 HARD +0.0282 WIN;
  dt 97–192 null — beyond the 48-frame span, the expected reach signature.
- **Band statement:** measured win band = **[896, ≈2059) tokens** (upper crossing interpolated
  from the 4-point curves; CI-clean flat win at 3072; lower edge measured exactly at the
  geometric span-threshold prediction ~48·16+2·64). The effect PEAKS at the lower edge: the
  cheaper the budget — provided the coarse span still covers the loop — the larger the
  hierarchy win, because flat's window shrinks away from the horizon faster than the
  abstraction cost grows.
- Notable measured arm ordering (single-seed, REPORT-ONLY, no claim): hier-min 0.1706 @896 tok
  beats hier-long 0.1848 @1152 tok — fewer recent-fine frames measured BETTER at fewer tokens;
  the pre-registered "fine-context floor" alternative branch is refuted at w_f = 2.
- Disclosures: seed 0 only; per the binding rule, **seed-1 replication is QUEUED** (behind the
  TaxiBJ generalization launch per author GPU priority) before any headline escalation.

---

## FRONTIER SEED/BUDGET ROBUSTNESS PROGRAM (pre-registered 2026-07-07 evening, author request;
## written BEFORE any run)

**Motivation (author, on fig9):** the hier curve's non-monotonicity (hier-min 0.1706 < hier-long
0.1848 s0; hier-rich 0.1695 ≈ hier-min) looks like a possible training outlier; check its
persistence with more seeds and longer training.

**What the EXISTING data already says (no new GPU):** hier-long has TWO measured seeds — s0
0.1848, s1 **0.1690** (spread 0.0158, seed-mean 0.1769) — while flat-budget is seed-stable
(0.2038/0.2044, spread 0.0006). hier-long s1 is already BETTER than hier-min s0 (0.1706), so
"hier-min beats hier-long" is a seed-0 artifact, not a stable ordering. Under current data the
hier curve from 896 tokens on is a PLATEAU at ≈0.169–0.177 (hier-min 0.1706, hier-long seed-mean
0.1769, hier-rich 0.1695 — all within the measured seed spread), i.e. **hierarchy saturates near
the span threshold; flat scales with budget** — the plateau reading. Hier arms are markedly more
seed-variable than flat arms (an honest observation to report).

**Runs (symmetric pairs only — the anti-cherry-picking guardrail; ALL seeds reported, means with
spreads quoted, never best-of-seeds):**
- **R-A (binding, already queued):** hier-min + flat-min seed 1, B16×30k. Read: does the 896
  peak replicate (CI>0 at seed 1) and where does hier-min s1 land vs the hier-long seed band?
- **R-B:** hier-rich + flat-long seed 1, both B8×60k (~18–20 h). Read: top-pair delta at seed 1;
  does hier-rich stay on the ≈0.17 plateau?
- **R-C (the legitimate "train longer", hypothesis-driven):** hier-long + flat-budget at
  B8×60k, seed 0 — budget-matched to the flat-long regime (~18 h). HYPOTHESIS (from the T3
  emergence result, null@6k → win@20k): the hier−flat delta at 1152 tokens GROWS with training
  budget, and hier-long-60k moves toward the plateau floor (≈0.169). If instead the delta is
  flat/shrinks, that is reported as the budget-saturation of the mid-point.
**PRE-REGISTERED READINGS:** primary = the plateau reading above survives seed averaging (hier
plateau ≈0.17 from 896 on; flat needs ≈3,072 to reach 0.156); the win band and its edges are
claimed on seed-means with all per-seed values disclosed. Alternative = if seed-1 runs move any
pair's delta sign, the affected edge is re-bracketed and reported as measured — no soft reads.
**Explicitly forbidden:** selecting the best seed per arm; asymmetric seed counts within a
quoted pair; extending training of one arm without its budget-matched partner.
**Execution order (one job at a time, after TAXIBJ_ALL_DONE):** R-A → R-B → R-C.

### R-A MEASURED (2026-07-08; `logs_rhcp/frontier_low_s1.log`; RA_DONE) — **REPLICATED**

Seed 1: hier-min 0.1744 / sharp 0.0835 vs flat-min 0.2102 / 0.0514; paired Δ ALL **+0.0358
[+0.0267, +0.0454]** — WIN in EVERY horizon × hardness cell again, with the same dt-gate
signature (dt≤48 HARD +0.0666 WIN; dt 49–96 HARD +0.0263 WIN; dt 97–192 null — beyond-span).
**The 896-token lower-edge point is TWO-SEED CONFIRMED** (Δ s0 +0.0399 / s1 +0.0358; seed-mean
≈ +0.0379 — the campaign's largest two-seed effect). Seed spreads: hier-min 0.0038 (tight, vs
hier-long's 0.0158); flat-min 0.0004 (flat arms stable, as everywhere). Plateau reading holds:
hier-min seed-mean 0.1725 ≈ hier-long seed-mean 0.1769 ≈ hier-rich 0.1695 (~0.17 band).
Per the binding rule, claim escalation is now permitted: the win band's lower edge is
headline-grade. R-B launched on completion (hier-rich/flat-long seed 1, B8×60k).

### R-D AMENDMENT (pre-registered 2026-07-08, BEFORE the run; external-review recommendation
### accepted): mechanism-agnosticism seed-1 replication

**Motivation:** the reg-long vs reg-flat pairing (+0.0334 [+0.0248,+0.0423], every bin) is
load-bearing — it is what licenses "both heads / mechanism-agnostic" in Claim 1 and novelty
item 1 — and it exists only at seed 0. R-A/R-B replicate the diffusion pairings; nothing
replicates the regression pairing. Protecting a stated claim outranks testing a losable
hypothesis, so **R-D is inserted AFTER R-B and BEFORE R-C.**
**Arms:** reg-long seed 1 + reg-flat seed 1 (B16×30k each, standard regime, loopmom) + compare.
**PRE-REGISTERED PRIMARY READING:** reg-long beats reg-flat at seed 1 with ALL cluster-CI > 0
(replication → mechanism-agnosticism is two-seed on both heads).
**ALTERNATIVE:** a null/loss at seed 1 demotes the wording everywhere from
"mechanism-agnostic (both heads)" to "replicated under the diffusion head; single-seed under
regression, seed-disagreement disclosed" — pre-drafted, no soft reads.
**Rules:** symmetric pair; both seeds reported; no claim escalation beyond two-seed wording.
**Queue after this amendment:** R-B (running) → **R-D** → R-C → generalization G-i/G-ii/(G-iii
vs loop-80 A′ re-ranked: A′ now ABOVE G-iii per external-validity value).

### R-B MEASURED (2026-07-08; `logs_rhcp/frontier_top_s1.log`; RB_DONE) — **REPLICATED**

Seed 1: flat-long 0.1580 / sharp 0.0991 vs hier-rich 0.1687 / 0.0888; paired Δ ALL **−0.0107
[−0.0163, −0.0058]** (flat wins; s0 was −0.0131 [−0.0201, −0.0069]) — the same uniform-loss
pattern (LOSS in 12/14 cells, 2 nulls, no hier win anywhere). Seed spreads: hier-rich 0.0008
(0.1695/0.1687 — exactly ON the ≈0.17 plateau), flat-long 0.0016 (stable). **The band's upper
edge (window-saturation inversion at 3,072 tokens) is TWO-SEED confirmed**, completing the
frontier: both edges now carry two seeds (896 hier-win, 3,072 flat-win) alongside the two-seed
1,152 interior point. The plateau reading survives seed averaging in full: hier means 0.1725
(896) / 0.1769 (1,152) / 0.1691 (3,072). R-D launched on completion (reg-long/reg-flat seed 1).
