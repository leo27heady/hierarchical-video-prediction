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
