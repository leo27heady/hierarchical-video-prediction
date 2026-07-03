# H-grid design memo (depth × mechanism) — SPEC ONLY, launch is the author's call

Goal: convert the hierarchy claim from *efficiency at matched budget* (proven, T4) into
*capability* — a regime where NO affordable flat window covers the memory horizon, so temporal
abstraction is necessary, not just cheaper. Fixes both gaps of the external diagnosis
(docs/HANDOFF_2026-07-03.md §6): the missing mechanism (coarse-FIRST multi-rate rollout;
strided slow-clock levels) and the regime cap (loop ≤ flat window).

## 1. Regime (the crux — do this before any architecture work)

New dataset via `scripts/gen_looping.py` (parameterized): world ~768 px (viewport 64 → ~92%
off-screen), momentum walk tuned so the revisit/loop period is ~192 frames; clips T=480;
small counts for speed (train ~1200 clips, test ~300). GUARANTEE to verify after generation:
median re-entry interval ≥ 4× the largest affordable flat fine window (flat at 3072 tokens
affords 48 fine frames ≪ 192-frame loop → flat CANNOT span the horizon at any tested budget).
Measure and record: actual loop-period distribution, off-screen fraction, p_hold; SAVE THE SEED.

## 2. Depth axis (all inside the MilliVid-style stack — NOT a CW-VAE clockwork rebuild)

- d0 flat: fine 8×8 per frame only.
- d1: + coarse 16ch@4×4 at temporal STRIDE 4 (one coarse latent per 4 frames → 48 coarse slots
  span 192 frames = the slow-clock effect). AE: reuse hier_convae spatially; coarse stream simply
  encodes every 4th frame (no AE retrain needed for d1 if the coarse encoder is per-frame — the
  stride lives in the DATA/rollout layer).
- d2: + top level 16ch@2×2 at stride 16 (12 slots span 192). Needs a 3-level AE variant
  (extend `hier_convae.py`; ~0.5 day + ~1h GPU train on the new train file).

## 3. Rollout (the missing MilliVid mechanism)

Coarse-FIRST multi-rate: per macro-step, (a) top level rolls out its strided latents over a long
span; (b) mid level generates conditioned on top-level FUTURE + own past; (c) fine level
generates the next 4 frames conditioned on both levels' (partly future) context + last fine
frames + actions. Training layouts must match rollout layouts (the same train==eval discipline
as rhvd). Implementation: extend `hvp_rev/rhvd.py` (WindowDiT already supports per-token
level/frame/role conditioning; the scheduler + strided buffers are the new code, ~1–1.5 days).

## 4. Arms (grid) — pre-registered

| arm | depth | mech | note |
|---|---|---|---|
| d0-reg / d0-diff | 0 | reg / diff | flat baselines, largest affordable window |
| d0-strided-flat-diff | 0 | diff | full-res frames every 4th (MANDATORY control: separates temporal abstraction from context subsampling) |
| d1-reg / d1-diff | 1 | reg / diff | the main comparison |
| d2-diff (+d2-reg if budget) | 2 | diff | does more depth pay at 192-frame horizon? |
| d1-ebt | 1 | EBT | ONE spot-cell only (mechanism already characterized; cite T2) |
| persistence + AE-ceiling | — | — | floors |

Matched token budgets within each comparison (report tokens/arm explicitly). B=16 where VRAM
allows; sample-matched otherwise (disclose). 30k steps; EMA; ckpt/resume; spill guards — all
existing infrastructure.

## 5. Metrics & gates (pre-registered)

Horizon curves to h≥400 (fg-MSE with CLUSTER CIs, sharpness-vs-h, full-frame-MSE column,
re-entry-stratified vs the d0 window); identity retention at re-entry. GATES:
- G-H1 (capability): d1-diff beats d0-diff at EVERY tested budget on hard re-entry beyond the
  d0 window (CI>0, cluster bootstrap) — i.e., no flat point on the frontier reaches it.
- G-H2 (mechanism-agnostic): same for d1-reg vs d0-reg.
- G-H3 (abstraction vs subsampling): d1-diff beats d0-strided-flat at matched budget.
- G-H4 (depth): d2 ≥ d1 at h≥192-horizon bins (else depth saturates at 1 — honest result).
Any null → the current matched-budget headline stands; grid reported as boundary conditions.

## 6. Cost & order of execution (for the author)

1. Dataset gen + regime verification (hours, CPU+light GPU).
2. d1 path first (no AE retrain): strided coarse buffers + coarse-first scheduler + d0/d1 arms
   (~1.5–2 days build, ~1 GPU-day runs) → early read on G-H1/G-H2/G-H3.
3. d2 (3-level AE + arm) only if d1 shows capability signal (~0.5 day + 0.5 GPU-day).
4. Seeds on the winning comparison (1 GPU-night).
Total: ~2–3 days build + ~2–2.5 GPU-days. All runs <15 GB with existing guards.
