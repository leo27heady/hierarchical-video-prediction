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

### R-D MEASURED (2026-07-08; `logs_rhcp/mech_agnostic_s1.log`; RD_DONE) — **REPLICATED, STRONGER**

Seed 1: reg-long 0.1503 / sharp 0.0865 vs reg-flat 0.2004 / 0.0515; paired Δ ALL **+0.0501
[+0.0396, +0.0612]** — WIN in every cell, LARGER than seed 0 (+0.0334 [+0.0248, +0.0423]).
**Mechanism-agnosticism is TWO-SEED on both heads** (diffusion pairing +0.0189/+0.0354; regression
pairing +0.0334/+0.0501 — all four runs win every-or-nearly-every cell). Seed spreads: reg-long
0.0179 (0.1682/0.1503 — the familiar memory-arm seed variance), reg-flat 0.0011 (stable).
Report-only observation (cross-head, single comparison, NO claim): reg-long s1 (0.1503) is the
best absolute rollout number of the campaign, below even flat-long-diff (0.1564/0.1580) at 37%
of its tokens — quote only as a measured arm value with the seed-variance caveat; the recipe
statement stays at seed-mean level (reg-long mean 0.1593 @ 1,152 tok). Next: R-C (running;
30k npz archived to `logs_rhcp/archive_30k/` before launch, per the overwrite caveat).

### R-C MEASURED (2026-07-09; `logs_rhcp/frontier_mid_60k.log`; RC_DONE) — **T3-EMERGENCE HYPOTHESIS CONFIRMED**

The legitimate "train longer": hier-long AND flat-budget both retrained at B8×60k (the flat-long
regime), seed 0, so the pair is same-regime and budget-matched within itself. Result:
hier-long **0.1666** / sharp 0.0899 vs flat-budget **0.2032** / 0.0592; paired Δ ALL **+0.0366
[+0.0272, +0.0465]** — WIN in every horizon × hardness cell.

**The pre-registered primary reading is confirmed** (the hier−flat delta at 1,152 tokens GROWS
with the optimization budget): comparing to the 30k-step seed-0 pair (archived
`logs_rhcp/archive_30k/`), the delta nearly DOUBLED (+0.0189 → +0.0366), and the mechanism is
exactly the T3 emergence story — **flat-budget is budget-saturated** (0.2038 → 0.2032, unchanged
at 2× budget: no memory circuit to mature) while **hier-long keeps maturing** (0.1848 → 0.1666,
−0.0182). At 60k the mid-budget hier point (0.1666) sits at/below the ≈0.17 plateau floor (below
even hier-rich 0.1687/0.1695), so the plateau reading survives the budget axis too. Provenance:
default-named npz restored to the 30k canonical versions (the T4 matrix and every doc reference
them unchanged); the 60k data saved as `rhvd_eval_{hier-long,flat-budget}_s0_60k.npz`.

### R-E (pre-registered 2026-07-09, BEFORE the run): the 512-token pair seed-1 (completeness)

**Motivation:** 896/1152/3072 all carry two seeds; only the 512-token below-threshold point
(hier-short/flat-short) is single-seed. Completing it makes every frontier point two-seed.
**Arms:** hier-short + flat-short seed 1 (B16×30k) + compare.
**PRE-REGISTERED PRIMARY READING:** 512 tokens is BELOW the geometric span threshold (a 6-frame
coarse window cannot cover the ~48-frame loop), so the expectation is **flat still wins or ties**
(the seed-0 −0.0036 persists or narrows). Hier arms are seed-variable, so hier-short s1 may
improve and narrow the gap; a CI-clean flip to a hier win would be a genuine surprise and would
trigger a re-examination of the threshold arithmetic (reported, not buried).
**ALTERNATIVE:** if hier-short s1 flips the sign, the lower band edge is re-bracketed below 512
and the geometric-threshold claim is softened to "consistent with, not exactly at, 512–896".
**Rule:** symmetric pair, both seeds reported, no soft reads.

### R-E MEASURED (2026-07-09; `logs_rhcp/frontier_bottom_s1.log`; RE_DONE) — **REPLICATED (below-threshold flat win)**

Seed 1: hier-short 0.2257 / sharp 0.0397 vs flat-short 0.2206 / 0.0419; paired Δ ALL **−0.0051
[−0.0079, −0.0024]** — flat wins, replicating seed 0 (−0.0036). The 512-token below-threshold
point is now two-seed (flat wins both), as pre-registered (a 6-frame coarse window cannot cover
the 48-frame loop → coarse tokens are redundancy). Seed spreads: hier-short 0.0003 (tight — no
memory circuit to be seed-variable about, unlike the loop-covering hier arms), flat-short 0.0018.
**Every frontier point is now two-seed** (512 flat / 896 hier / 1,152 hier / 3,072 flat).

### R-F/R-G (pre-registered 2026-07-09, BEFORE the runs): the upper-edge BRACKET (1792 & 2432 tok)

**What a token is (for the record):** a token = one spatial cell of a latent frame projected to
width d. The frozen AE encodes each 64×64 frame into a coarse map (16 ch × 4×4 = **16 tokens/
frame**) and a fine map (16 ch × 8×8 = **64 tokens/frame**); a coarse token summarizes ~16×16 px
(layout), a fine token ~8×8 px (detail). Context-token budget = n_ctx_c·16 + n_ctx_f·64. In the
hierarchy the coarse stream is cheap wide memory (48 coarse frames = 768 tok span the whole
loop) and the fine stream is expensive recent detail; a flat arm has no coarse stream, so the
same budget buys a shorter window (e.g. flat 1152 tok = 18 fine frames = 18-frame window < loop).

**Motivation (author):** the frontier has a gap between 1,152 and 3,072 tokens, and the upper
band edge (where hier→flat inverts) is currently only *interpolated* at ≈2,059 tok. Two points
inside the gap BRACKET the crossing and measure the edge directly. Fixed 48-frame coarse span
(covers the loop); the varied axis is the number of recent fine frames.

**Arms (diffusion head; budget matches exact):**
- **R-F @ 1,792 tok:** hier-c1 (48c+16f) vs flat-c1 (28f). 1,792 < 2,059 → **predict HIER win**
  (positive Δ, narrower than the 1,152 win as the curve approaches the crossing).
- **R-G @ 2,432 tok:** hier-c2 (48c+26f) vs flat-c2 (38f). 2,432 > 2,059 → **predict FLAT win**
  (negative Δ, narrower than the 3,072 loss).

**PRE-REGISTERED PRIMARY READING:** the sign of Δ flips between 1,792 and 2,432 — locating the
measured upper band edge inside that interval and replacing the "≈2×10³ interpolated" qualifier
with a measured bracket. **ALTERNATIVE:** if BOTH are hier-wins, the edge is >2,432 and the band
is wider than interpolated (re-bracket upward); if BOTH are flat-wins, the edge is <1,792
(re-bracket downward). Either way the interpolation is replaced by measurement — no soft reads.
**Regime:** matched WITHIN each pair; batch by VRAM probe to ~14 GB (seq 2,112 / 2,752 tokens —
smaller than hier-rich's 3,392, so B16×30k likely fits; the paired delta cancels regime within
each pair). Seed 0 first; seed 1 only if a point ends up claim-bearing at the edge.
**Queue:** after R-E → R-F → R-G (one job at a time).

### R-F/R-G MEASURED (2026-07-09/10; `frontier_c1.log`, `frontier_c2.log`) — **UPPER EDGE BRACKETED**

The sign of the paired delta flips between the two bracket points, exactly as pre-registered:

| budget | flat | hier | paired Δ(flat−hier) ALL | verdict |
|---|---|---|---|---|
| **1,792 (R-F)** | flat-c1 0.1876 | hier-c1 0.1799 | **+0.0077 [+0.0016, +0.0142]** | **HIER wins** (below crossing) |
| **2,432 (R-G)** | flat-c2 0.1703 | hier-c2 0.1820 | **−0.0116 [−0.0162, −0.0073]** | **FLAT wins** (above crossing) |

**The upper band edge is therefore MEASURED to lie in the open interval (1,792, 2,432) tokens**
— the earlier interpolated estimate (≈2,059) sits inside it. The full six-point frontier:
512 flat / **896 hier** / **1,152 hier** / **1,792 hier** / [edge] / **2,432 flat** / 3,072 flat.
Replace the "≈2×10³ interpolated" qualifier (abstract, CH8, VYSNOVKY, ROZDIL4, CH1) with the
measured bracket "(1,792, 2,432)".

**Convergence fairness (verified):** each bracket pair is sequence-length-matched (hier-c1/flat-c1
seq 2,112; hier-c2/flat-c2 seq 2,752 — same context+target token count within each pair), so the
per-step training compute and hence the convergence state at the shared 30k-step budget are
matched WITHIN each pair → the paired delta is fair regardless of absolute convergence.

**Threats note (cross-budget hier-line shape, NOT the paired verdicts — cf. the plateau
diagnosis).** The absolute hier values are non-monotone across budgets (hier-min 0.1706 @896 <
hier-c1 0.1799 @1,792 < hier-c2 0.1820 @2,432; hier-rich 0.1695 @3,072). This is the
fixed-step under-convergence effect: at a constant 30k steps, longer-context arms are less
converged (train-log val-h4 rises 0.027→0.034→0.044 as fine frames go 2→16→26), so the hier
*line* wobbles ≈±0.01 on the plateau; hier-rich (3,072) additionally used a 60k regime. This
does NOT touch any paired win/loss (each is seq-matched), but the frontier *curve* should be
captioned as regime/convergence-mixed on the hier line — a fair-disclosure item, not a defect.

**This closes the frontier-robustness program (R-A/R-B/R-D/R-C confirmed their primary branch;
R-E completes the 512 point; R-F/R-G bracket the upper edge to (1,792, 2,432)). The six-point
two-sided frontier — win band with BOTH edges measured — is complete.** It also fully explains, on two independent axes, the seed-0-30k "dip" that motivated
the whole program: hier-long's seed-0-30k value (0.1848) is high both because of seed variance
(s1 = 0.1690) AND because of budget (60k → 0.1666); the true ≈0.17 plateau is robust to both.
The two-seed 30k pairing (§4.3.3) remains the headline anchor; R-C is a budget-robustness
addendum, single-seed by design (a budget-axis test, not a seed test).

---

## CW-STYLE COMPARISON TABLE PROGRAM (pre-registered 2026-07-10, BEFORE any run;
## author direction: depth x downsampling-factor x head table + degradation curves,
## presented CW-VAE-style [Saxena et al. 2021] but on OUR protocol — fg-masked metrics,
## matched token budgets, paired cluster CIs; NO aggregate SSIM/PSNR as claims)

**Scope decisions (author, 2026-07-10):** temporal factor axis limited to {1, 2} (4 only where
it already exists as machinery); spatial re-based pyramids (16x16 bottom / 1x1 top) deferred to
the multi-scale next-paper arena; EBT column = the existing characterized negative, NO new runs;
default-MMNIST joins as the CONTROL column (Phase C), never as the primary arena. Launch only
after R-G resolves (flat-c2 + compare). Infra landed ahead of runs: 7-field ARMS (top stream,
`hvp_rev/rhvd.py` arm_cfg), frozen-trunk 2x2 top AE branch (`hvp_rev/hier3.py`), generalized
strided-slot rule (WIN//stride slots per window — stride-4 unchanged, stride-2 now phase-correct),
`--blind` eval probe, CPU smoke `tests/smoke_cwstyle.py` (ALL GREEN incl. legacy bit-compat),
auto-table `scripts/make_cwstyle_table.py` (docs/CWSTYLE_TABLE.md + fig12).

### Phase B arms (loopmom, standard regime B16x30k bf16, seed 0 first; WARM=48; one job at a time)

- **B1 `hier-s2-min`** (24c@stride2 + 2f = **512 tok**, span 48) vs **EXISTING** flat-short npz
  (512 tok, s0+s1 — symmetric-pair rule satisfied by reuse; no new flat run). PRIMARY: hier WINS
  at 512 (ALL cluster-CI > 0) — the win-band lower edge MOVES with the temporal factor exactly as
  the token arithmetic predicts (24*16+2*64 = 512 < the stride-1 family's 896). ALTERNATIVE (a):
  null/loss -> the stride-2 family's edge is between 512 and 896; report as a measured family-edge
  refinement (fine-floor or aliasing mechanisms discussed, momentum reversals between stride-2
  anchors disclosed). A CI-clean hier win triggers the two-seed rule (seed 1 before any headline
  use; flat-short s1 already exists).
- **B2 `strided-flat`** (11 full-res hist@stride4 + 6f = **1088 tok**, span 48, NO coarse level)
  vs the EXISTING 1152 pair (hier-long, flat-budget; both two-seed). Its 64-token deficit favors
  the control. PRIMARY (honest, from G-S1): strided-flat lands ~AT hier-long (within CI of the
  hier plateau) and clearly beats flat-budget -> the frontier currency is SPAN-PER-TOKEN; coarse
  tokens are one implementation (the per-level-decodable one), temporal subsampling another.
  Claim-1 wording then softens per the pre-drafted branch: "hierarchical packaging" ->
  "long-span context packaging (coarse level = the cheapest per-level-decodable form)".
  ALTERNATIVE: strided-flat clearly BELOW hier-long (hier win CI>0) -> the coarse level adds
  value beyond span (denser temporal coverage at equal budget); Claim-1 wording stands and gains
  this control as its strongest defense. Either branch is reported; no soft reads.
- **B3 `hier3`** (48 top(2x2) + 24c + 9f = **1152 tok**; factor-2 spatial pyramid 2x2/4x4/8x8;
  needs `hier_ae_mom_top.pt` trained first + GO-gate oracle PASS) vs hier-long AND flat-budget
  (both existing, two-seed). PRIMARY: ~null vs hier-long (|delta| CI spans 0 or < seed spread
  0.016) — on a single-scale task with span protocol-capped at 48, a third stage can only
  re-allocate, not extend; the depth column's honest loopmom entry is "2 stages suffice; depth
  must match the task's memory-scale count" (mirrors the capacity-dump precedents). ALTERNATIVE
  (a): hier3 CI-clean WIN -> re-allocation/optimization benefit is real; escalate to seed 1 and
  report as a genuine depth finding. ALTERNATIVE (b): CI-clean LOSS -> the extra stream's
  optimization cost dominates (consistent with the fixed-step context-length effect); reported
  as the depth cost at fixed budget.
- **A2 blinding probes** (eval-only on EXISTING ckpts, no training): hier-long s0 with
  `--blind coarse` and `--blind fine`; hier3 (after B3) with `--blind top`. Read: per-level
  content attribution (CW-VAE §5.1 analog) — blinding coarse should destroy the dt<=48 retrieval
  bins specifically; blinding fine should cost recency/detail bins. Report-only (mechanism
  figure), no claims gate on it.

**Rules (binding):** symmetric pairs (reused partners count — both seeds already exist for every
partner); seed-means, never best-of-seeds; two-seed before any claim escalation; regime B16x30k
matched to every partner (NO 60k cells in this block); GO-gate oracle PASS required before
training any arm with new buffer machinery (hier-s2-min, hier3); anything not covered above
defaults to report-only.

### AMENDMENTS (pre-registered 2026-07-10 ~15:00, BEFORE B2/B3 launch; B1 mid-training,
### untouched — the same before-the-run discipline as the R-D insertion)

- **B2 seed asymmetry rule (clarifies the general two-seed rule for the null branch):** the
  CONSERVATIVE claim-1 wording softening ("hierarchical" -> "long-span context packaging") is
  permitted on a seed-0 null (weakening one's own claim needs no replication), but the POSITIVE
  assertion "temporal subsampling REPRODUCES the matched-budget win / span is the currency"
  is itself a claim and stays REPORT-ONLY until strided-flat seed 1 confirms (hier-arm seed
  spread ~0.016 can manufacture a one-seed null). Text sites carry the distinction explicitly.
- **B3 disclosed confound (loss-weight dilution):** the 3-level loss sums THREE per-level MSE
  terms (vs two in every 2-level arm), so hier3's effective fine-level loss weight is ~2/3 of
  the 2-level arms' — and ALL metrics decode the FINE latents. A small hier3 LOSS is therefore
  attributable to loss dilution as well as depth per se; ALTERNATIVE (b) reports both mechanisms
  and does NOT claim "depth hurts" without a weight-matched follow-up. (Equal-sum weighting IS
  the registered design — changing it now would invalidate the pairing; the confound is
  disclosed, not repaired mid-program.)
- **A2 blinding OOD caveat (binding on interpretation):** zeroed context streams are
  out-of-distribution inputs (never seen in training), so a collapse under blinding is an UPPER
  bound on brittleness and a LOWER bound on the stream's information contribution — the probe
  attributes content causally only in the direction "stream removed -> capability lost"; the
  magnitude may include OOD shock. Reported with this caveat in every use; a less-OOD variant
  (freeze-last-slot instead of zeros) is noted as optional follow-up, not run by default.

### Phase C (default-MMNIST control column; after Phase B)

Standard bouncing-digit MMNIST (OpenSTL-matched `scripts/gen_moving.py`), NEW frozen AE trained
on its frames, zero-action conditioning (disclosed; the action interface is inert on actionless
data), train 3000x100 / test 500x348, WARM=48, ~300-frame rollouts. Arms: flat-budget-mm,
hier-long-mm (+ hier3-mm optional) at 1152 tok, B16x30k, seed 0. PRIMARY (the thesis's own
prediction, extending the measured T7 null): hier ~= flat on ALL fg-MSE (no memory horizon ->
nothing for coarse span to buy); the value of the column is the MEASURED null in the same table
where loopmom shows the band — "where hierarchy pays and where it cannot". Disclosures: hardness
axis inert (nothing exits the frame); fg-threshold trivial (black bg); an SSIM-vs-horizon curve
may be SHOWN for literature comparability with persistence + all-black floors drawn, under the
CH2 §2.8 carve-out (calibration, not claim). A hier win here would contradict T7 and trigger
re-examination before any reporting.

### Cost ledger (estimates; VRAM all << the 11.8GB flat-c2 point)
B1 ~2h train+~1h eval; B2 ~2.5h+1h; AE-top ~0.25h; B3 ~3h+1h; blinding 3x ~0.5h; compares
minutes. Phase B total ~11-12 GPU-h. Phase C: AE ~0.3h + 2-3 arms x ~3-4h + evals ~ 1 GPU-day.

### B1 MEASURED (2026-07-10 15:42; `logs_rhcp/cwstyle_phaseB.log`, B1_DONE;
### npz `rhvd_eval_hier-s2-min_s0.npz`) — **PRIMARY BRANCH CONFIRMED: the edge moves with the
### temporal factor**

hier-s2-min (24c@stride2 + 2f = 512 tok, span 48) vs flat-short (512 tok, s0):
- ALL fg-MSE **0.1893** / sharp 0.0697 vs 0.2224 / 0.0432; paired Δ ALL **+0.0330
  [+0.0246, +0.0419]** — **WIN in EVERY horizon × hardness cell (14/14 + bg-only)**, from h1-8
  easy +0.0077 to h25-60 HARD +0.0640 [+0.0450, +0.0827].
- dt-gate span signature, textbook: dt≤48 HARD **+0.0643** [+0.0496,+0.0791] WIN; dt 49–96 HARD
  +0.0228 WIN; dt 97–192 null — beyond-span, exactly the hier-min signature.
- **The pre-registered primary reading is confirmed:** the win-band lower edge is
  family-parametric — at 512 tokens, where the stride-1 coarse family LOSES two-seed
  (−0.0036/−0.0051), the stride-2 family WINS CI-clean, at its token-arithmetic prediction
  (24·16 + 2·64 = 512).
- **Expectation-baseline comparison (honest):** sign + every-cell structure + dt signature as
  predicted; magnitude BELOW the predicted range (Δ +0.0330 vs expected +0.04..+0.05; ALL 0.1893
  vs expected 0.170–0.180). The stride-2 family sits ~+0.017 ABOVE the stride-1 plateau
  (hier-s2-min 0.1893 vs hier-min seed-mean 0.1725) — a real FAMILY COST (24 sparser anchors vs
  48 dense; inter-anchor blindness under momentum reversals). Refines the law: the edge moves
  with the family arithmetic AND each family carries its own plateau level. Reported as
  measured, not smoothed away.
- Per the binding rule + amendment: CI-clean win → **seed-1 replication QUEUED** (flat-short s1
  npz exists) before any headline escalation; CH1/CH8 clauses stay gated until then.

### REG TWINS + 320-TOKEN EDGE CELL (pre-registered 2026-07-10 evening, author-approved,
### BEFORE any run; B2 mid-training, B3/blinding queued — this block launches AFTER S1-a)

**Motivation (author):** complete the shape table along the axes originally requested — the
head crossing at the new shape cells, and a third point on the edge-law dose–response.

**Arms (standard B16×30k bf16, seed 0, loopmom; 6 training runs ≈ 13–16 GPU-h incl. evals —
honest cost update vs the ~11 h quoted in chat: the symmetric-pairs rule requires TRAINING
flat-short-reg and flat-320 as partners, not just the 4 named cells):**

- **E-320 (edge law, third family):** `hier-s4-min` (12c@stride4 + 2f = **320 tok**, span 48 —
  the cheapest loop-coverer of any family; one coarse slot per window, the proven d1 stride-4
  rule) vs `flat-320` (5f = 320 tok exact). GO-gate oracle PASS required first (first stride-4
  coarse ROLLOUT on loopmom). PRIMARY: hier-s4-min WINS (ALL CI>0) → the edge law becomes a
  3-point dose–response (896 / 512 / 320, all at token-arithmetic predictions). Expected values:
  hier-s4-min ALL ≈ 0.19–0.22 (family plateau offset GROWS with stride — s2 cost was ≈+0.017;
  s4 anchors are 4× sparser), flat-320 ≈ 0.23–0.245 (5-frame window, extrapolating the flat
  curve below flat-short 0.2224@8f), Δ ≈ +0.02..+0.05. ALTERNATIVE (null/loss): stride-4
  inter-anchor aliasing (momentum reversals unresolvable within 4-frame gaps) breaks the family
  — reported as the measured STRIDE LIMIT of the edge law, an equally sayable boundary.
- **REG-SHAPE (head axis at the new cells):** `hier-s2-min-reg` vs `flat-short-reg` (TRAINED,
  512 tok pair); `strided-flat-reg` (1088) vs reg-flat AND reg-long (both exist two-seed);
  `hier3-reg` (1152) vs reg-flat AND reg-long. PRIMARY (from G-S1 "coarse ≈ free under reg" +
  the reg-head H2 reversal): the diffusion-head shape verdicts REPRODUCE or ENLARGE under
  regression — (i) hier-s2-min-reg beats flat-short-reg CI-clean; (ii) strided-flat-reg lands
  within ~0.01 of reg-long and CI-clean above reg-flat; (iii) hier3-reg ≈ reg-long (null,
  |Δ| < seed spread 0.018). Expected absolute values in the reg band (reg-long mean 0.1593,
  reg-flat 0.2009). ALTERNATIVES mirror the diffusion-head branches cell-for-cell; a
  head × shape INTERACTION (any verdict flipping sign across heads) is the headline-grade
  surprise and triggers seed-1 on the flipped cell before any wording.

**Rules (binding, inherited + amendments):** symmetric pairs (flat-short-reg and flat-320
trained as partners; reg-flat/reg-long reused at both seeds); seed-means, never best-of-seeds;
two-seed before claim escalation; the B2 seed-asymmetry amendment applies to strided-flat-reg
verbatim; GO-gate before hier-s4-min; loss-dilution disclosure applies to hier3-reg verbatim;
anything uncovered defaults to report-only.
**Queue:** CWSTYLE_PHASE_B_ALL_DONE → S1-a (hier-s2-min s1) → **this block**
(`scripts/run_cwstyle_regtwins.sh`) → Phase C → G-i/G-ii. S1-b (strided-flat s1) inserts before
Phase C if its amendment condition fires.

### B2 MEASURED (2026-07-10 19:08; `logs_rhcp/cwstyle_phaseB.log`, B2_DONE;
### npz `rhvd_eval_strided-flat_s0.npz`) — **A THIRD OUTCOME, STRONGER THAN THE PRIMARY:
### strided full-res history BEATS the coarse hierarchy at fewer tokens**

strided-flat (11 full-res@stride4 + 6f = 1,088 tok, span 48, NO coarse level), seed 0:
- ALL fg-MSE **0.1652** / sharp **0.0919** — vs hier-long s0 (0.1848/0.0729): paired Δ
  **+0.0197 [+0.0143, +0.0252] WIN**; vs flat-budget (0.2038): **+0.0386 [+0.0290, +0.0488]
  WIN** — at a 64-token DEFICIT to both partners.
- Win across bins INCLUDING hard re-entry at every horizon (h1–8 HARD +0.0482; h25–60 HARD
  +0.0389; h61–120 HARD +0.0180): the full-resolution anchors carry re-entry identity detail
  that the coarse code loses — the G-S1 abstraction cost (−0.008 at fixed window) expressed at
  rollout scale. Sharpest 1,152-class arm measured (0.0919; only flat-long 0.0996 is higher).
- **Neither pre-registered branch as written:** PRIMARY expected ≈hier-long (null-ish,
  |Δ|<0.01); ALTERNATIVE expected hier-long to win. Measured = the primary's DIRECTION at
  greater strength: span is the currency, and at this operating point temporal subsampling of
  full-resolution history is the BETTER implementation under the diffusion head — not merely an
  equal one. Reported as measured; no soft reads.
- **Honest single-seed caveats (why escalation is gated):** hier-long s0 (0.1848) is that arm's
  weaker seed (s1 0.1690; seed-mean 0.1769); strided-flat is 1 seed and memory arms spread
  ~0.016. Against the hier-long SEED-MEAN the margin is ≈0.012; against hier-long's best seed
  ≈0.004. Per the pre-registered amendment, the POSITIVE claim ("temporal subsampling matches
  or exceeds the coarse packaging") requires strided-flat seed 1 — **S1-b now FIRES as
  REQUIRED** (uncommented in `run_cwstyle_seed1.sh`). The conservative Claim-1 wording
  softening ("hierarchical packaging" -> "long-span context packaging") is permitted at s0 and
  is applied at the four registered sites.
- Consequences if s1 confirms: the sizing rule gains "prefer full-res strided anchors over
  coarse anchors when the head is diffusion"; the coarse level's defense narrows to per-level
  decodability + dense coverage + (per G-S1) the regression head, where abstraction is free —
  exactly what the pre-registered REG twin (strided-flat-reg) now tests directly.
