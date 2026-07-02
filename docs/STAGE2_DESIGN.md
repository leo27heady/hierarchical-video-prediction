# Stage-2 design memo: minimal hierarchical video diffusion with long-coarse memory (RHVD-mini)

Status: written 2026-07-03, immediately after gate results. G0b PASSED (diffusion commits with
standardized latents and beats regression on every column). G0a PASSED **at 20k training steps**
(hard re-entry delta +0.0135, CI [+0.0080,+0.0196], seed-1 confirmation in flight) after a clean
null at 6k — the retrieval capability emerges late; every Stage-2 arm therefore trains >=20k steps.

## Claim under test (pre-registered)

**H1 (headline):** in autoregressive rollout on the momentum looping testbed, a 2-level model whose
COARSE level carries a long context (>= loop period, 48 frames) sustains lower re-entry-stratified
foreground error over 200-frame rollouts than (a) a short-context hierarchical model and (b) a
FLAT fine-only model with the SAME total context-token budget. (b) is the arm that makes the claim
"hierarchy earns its keep", not merely "long context helps" — CW-VAE/TECO ancestry demands it.

**H2 (supporting):** the generative mechanism earns its keep at rollout: an action-conditioned
REGRESSOR of matched capacity, rolled out with the same scheduler, fades toward the mean (E1
precedent) while the diffusion model holds sharpness. (Pre-registered expectation; if regression
rolls out fine, we say so — that would re-scope the generative claim honestly.)

## Arms (all trained 30k steps, B=16-32, bf16, EMA 0.999, constant-LR + 2k warmup, save/resume)

| arm | coarse ctx | fine ctx | ctx tokens | role |
|---|---|---|---|---|
| hier-long   | 48 | 6 | 48*16+6*64 = 1152 | THE system |
| hier-short  | 6  | 6 | 6*16+6*64 = 480   | memory ablation |
| flat-budget | —  | 18 | 18*64 = 1152      | token-budget-matched flat control |
| reg-long    | 48 | 6 | 1152              | regression-rollout control (H2) |
| persistence + AE-roundtrip | — | — | — | floors/ceilings |

flat-budget needs `_build_sequence` to accept n_ctx_c=0 (skip coarse context tokens; keep the
coarse TARGET tokens so the AE interface and losses stay identical across arms — the flat arm
simply gets no coarse memory). Implementation: allow ctx_c with shape (B,0,LAT,4,4).

## Training protocol (single-step, multi-offset)

Keep the leakage-free single-step protocol of step1_mech (GT context -> predict frame at +k), but
sample k uniformly from {1..8} per example with k embedded via the action sequence (the k bridging
actions already encode horizon; add a horizon embedding only if sampling shows it's needed).
Rationale: the AR rollout advances 4 frames per step (k=4 fixed at eval); training on k in {1..8}
is cheap layout diversity that regularizes exposure at little complexity cost. NO self-generated
context at train time in v1 (clean-context first, per plan; context-noise augmentation is the
single budgeted fallback if rollout drifts, +2 GPU-days cap, G2 text).

## Rollout (eval-time AR)

state = ring buffers of the last 48 coarse + 6 fine latents (GT for the first 48 warmup frames,
then self-generated). Each rollout step: sample (yc, yf) with 16 DDIM steps conditioned on the
buffers + the 4 recorded actions; append yc/yf; advance 4 frames. 200-frame clips => 38 rollout
steps after warmup. Metrics per horizon h in {4,8,...,152}: fg-MSE, fg-sharp, per-digit centroid
L2 (long_eval), stratified by re-entry hardness *computed against the SHORT window* (novel_fg6,
as in mem_gate) and by "content last seen >= m frames ago". Best-of-1 (no oracle) for the headline
tables; best-of-8 reported as coverage diagnostic only. 100 test clips, bootstrap CIs, 2 seeds
for the headline arms (hier-long vs flat-budget), 1 seed for the rest + CI-on-one-seed.

## Model

DiTDenoiser d=384, depth=8, heads=6 (~20-25M), bf16, dropout 0. Frame embedding sized 49.
Timing probe REQUIRED before the first full run (feasibility attack: 0.6-1.2 s/step realistic at
seq ~1232; >1.2 s/step or >14 GB allocated => shrink B / accum). EMA decay 0.999, eval on EMA
weights. Checkpoint every 1k steps, atomic.

## Kill-gates

- G2a (smoke, 2k steps): losses decreasing at both levels, samples show digit structure by 2k
  (standardized latents did this at 1.5k in the clean run), no NaN, s/step within budget.
- G2b (after first 30k hier-long run + rollout eval): 200-frame rollout does not diverge
  (bounded centroid drift; sharpness stays > 0.5x single-step value). One context-noise-
  augmentation iteration allowed (+2 GPU-days), else downgrade to short-rollout (<=40 frames)
  evaluation and say so.
- G2c (headline): hier-long beats flat-budget on the hard re-entry bin at horizon >= 24 with CI
  excluding 0 (paired, CRN). If flat-budget matches hier-long, the honest conclusion is "context
  length, not hierarchy" — write it; the mechanism study + EBT chapter + Ch. D still carry.

## Non-goals (v1)

CFG, 3-level pyramid, FVD, renderer upgrade, Self-Forcing-style rollout objective (Stage-3b memo
first), any EBT arm (Stage 1 owns that), MilliVid-repo code reuse (pure-PyTorch path is proven).

## Estimated cost

5 model-arms x 30k steps at ~0.6-1.0 s/step ~= 5-8 h each => ~2 GPU-days incl. rollout evals and
the 2-seed headline replication; build ~3-5 person-days on top of the existing faithful_nets /
step1_mech / mem_gate machinery (universal k-sampling batch fn, ring-buffer rollout loop, eval
glue are new; everything else exists).
