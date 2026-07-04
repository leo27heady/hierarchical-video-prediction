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
