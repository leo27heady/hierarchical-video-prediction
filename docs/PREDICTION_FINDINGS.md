# Single-step prediction: blob diagnosis, the working regime, and the motion wall

2026-06-20, branch `feat/uncertainty-maps-128`. Tooling: `scripts/debug_prediction.py` (decodes the
predicted next latent several ways), `scripts/gen_data_*.py` (motion-magnitude datasets).

## TL;DR
- **The tokenizer/decoder are perfect** — decoding the *true* next-frame latents (oracle) is sharp.
- **The blob is the BOT next-latent prediction mean-regressing, and it scales with per-frame motion.**
- **It works at moderate motion**: 8–16°/frame Z-only → recognizable, fairly sharp rotating shapes
  (pred ≈ 22–23 dB vs a 24.7 dB oracle ceiling). See `logs/debug_warp_moderate/`, `logs/debug_l2_moderate/`.
- **The warp head is only a marginal win** (+0.3–0.9 dB); **the `pred_bot+current-coarse` decode is the
  bigger, free win** (+1–2 dB, ~oracle).
- **Fast / SO(3) motion stays blobby** because *inferring the large/3D rotation from context* is the
  bottleneck — not the output head, not the decode, not the tokenizer. Neither warp nor GMM fixed it.

## The motion–sharpness wall (single-step pixel PSNR, 64px Z-only unless noted)

| data (per-frame motion) | head | recon | oracle* | pred (all) | pred_bot+cur** | look |
|---|---|---|---|---|---|---|
| slow 4–10°, 1–2 cubes | L2 | 30.3 | 22.5 | 21.1 | 22.7 | recognizable |
| **moderate 8–16°, 1–3 cubes** | **L2** | 31.3 | 24.8 | 21.1 | 22.4 | **recognizable/sharp** |
| **moderate 8–16°, 1–3 cubes** | **warp** | 31.2 | 24.7 | 21.5 | **23.3** | **recognizable/sharp** |
| fast 15–45°, 2–6 cubes | L2 | 30.3 | 22.2 | 17.1 | 17.2 | rounded blob |
| fast 15–45°, 2–6 cubes | warp | 29.9 | 23.7 | 18.0 | 17.2 | rounded blob |
| SO(3) 15–45°, 128px | GMM | 37.8 | 19.8 | 12.6 | 12.8 | formless blob |

*oracle = decode the true next latents (the achievable ceiling through this decode path).
**pred_bot+cur = decode predicted-BOT + CURRENT-time coarse latents.

## Root cause (with the disproven alternatives)
Asked to predict a large, hard-to-infer rotation in one step, the predictor minimizes MSE → the
**conditional mean** = the average of plausible next frames = a centered blob. Blur ∝ motion spread.
- **Not the tokenizer/decoder** — oracle decode is sharp.
- **Not the multi-rate hierarchy / coarse stages** — substituting oracle coarse doesn't rescue the BOT
  blob; the entire oracle→pred collapse is the BOT prediction itself.
- **Not the output representation** — the GMM head collapsed (samples = blobs; the motion is ~deterministic
  per clip, so there are no real modes to capture); the explicit **warp** head only helps marginally
  because *estimating the flow is as hard as the original regression*.

## The free win: `pred_bot + current-coarse` decode
Across slow/moderate data, decoding **predicted BOT + current-time (un-shifted) coarse** beats decoding
all-predicted stages by +1–2 dB and lands ~at the oracle ceiling. The coarse stages change slowly, so
their current value is a better next-frame context than a noisy prediction of them. This matches the BLT
intuition (coarse = slowly-varying global context, predict only the fine detail). Recommended as the
default prediction decode for slow/moderate motion. (On *fast* motion it doesn't help — coarse changes
too much per frame.)

## Working recipe (stable training, normal-looking predictions)
- Data: Z-only (in-plane), **≤ ~16°/frame**, 1–3 cubes, 64px. `scripts/gen_data_simple_z.py` (slow) or the
  moderate render (`shape_angle_min/max = 8/16`, hashes `a34e023fd69e`/`cc5bd1b9e7c2`).
- Model: `configs/warp_z.yaml` (warp head) or plain `base_64.yaml` (L2 — nearly identical, simpler),
  `--set data.train_hash=a34e023fd69e data.val_hash=cc5bd1b9e7c2`. 3000 steps ≈ 30 min, recon ~31 dB.
- Decode/eval: prefer `pred_bot+current-coarse`. Inspect with `scripts/debug_prediction.py --ckpt ...`.

## Open problem (for harder / SO(3) motion) — needs more than an output head
The wall is *motion inference*. Candidate directions, untested here:
1. **More temporal context** for the predictor (longer window / better temporal modeling) so the rotation
   axis+speed is inferable; today early causal positions have little context.
2. **Larger / BOT-only predictor** — drop the coarse predictive loss (they can't extrapolate), spend the
   capacity on the finest stage's motion inference.
3. **Explicit 3D-aware motion** (warp can't hallucinate newly-revealed faces under SO(3) self-occlusion);
   pair an in-plane warp with a generative residual / diffusion head only for the disoccluded region.
4. **Slower frame rate** for the SO(3) demo (the pragmatic option — moderate motion already looks normal).

## Research update — R1 cracks FAST in-plane motion (`configs/research_r1.yaml`)
Combining the levers above on the previously-blobby fast Z-only case (15–45°/frame): **more temporal
context (T=17)** + **larger BOT-focused predictor** (`stage_dims=[256,192,128]`, `stage_layers=[6,4,3]`,
`stage_pred_weight=[1,0.25,0.1]`) + **warp head**, 6000 steps.

| fast Z-only (15–45°/frame) | recon | oracle | pred | look |
|---|---|---|---|---|
| baseline L2/warp (T=9, dim128) | 30 | 22–24 | **17–18** | rounded blob |
| **R1 (T=17, big BOT, warp)** | **34.2** | 27.2 | **~21 (20.8)** | **recognizable structured shapes** |

So the levers work: fast in-plane prediction goes from a formless blob to recognizable, oriented shapes
(+3–4 dB, ~54% of the oracle headroom; see `logs/debug_r1_final/`). This shows the fast-motion blob was a
*model-capacity/context* problem after all (for in-plane motion) — not an intrinsic single-step wall.
**R2** applies the same recipe to moderate **SO(3)** (`configs/research_r2.yaml`) to test transfer to real
3D rotation (where self-occlusion is the extra difficulty the warp can't resample).

### R2 — recipe transfers to real 3D rotation (moderate SO(3), 64px)
Same recipe, moderate SO(3) (8–16°/frame, x+y+z), 6000 steps: recon **34.5 dB**, pred **~20 dB**
(oracle 28.0 → ~30% of headroom), and a notably **stronger hierarchy lift `mid_casc +0.10`** (vs ~+0.03
in-plane) — the coarse cross-attention earns its keep on 3D rotation. Predictions are **recognizable
rotating cuboids** (`logs/debug_r2_final/`), vs the original SO(3) blob at 12.6 dB. Self-occlusion leaves
a larger oracle gap than in-plane (the warp can't resample newly-revealed faces — the residual must, and
that part still softens).

### Scaling to 128px / 2→8→32 — a memory wall on a 16 GB GPU
The two capacity/context levers that make R1/R2 sharp **do not fit at 128**: BOT is already 32²×9 = 9216
self-attention tokens, so T=17 (→17408 tokens) and a larger BOT both OOM at bs≤4. The feasible 128 recipe
(`configs/so3_128_recipe.yaml`) keeps only **warp + BOT-focus + moderate motion** (original BOT size, T=9,
bs=4×accum4). To get the full recipe at 128 you need **true windowed/Flex attention on BOT** (the dense
SDPA mask saves no memory) or a bigger GPU — the clear next engineering step for full-res 2→8→32.

## Bottom line for the dissertation
- The blob was **mean-regression under hard-to-infer motion**, not a tokenizer/representation defect.
- It is **fixable**: moderate motion alone makes single-step prediction recognizable; for *fast* motion,
  **more temporal context + a larger BOT-focused predictor + a warp head** recover recognizable, oriented
  predictions (proven at 64px on both in-plane and SO(3)).
- The hierarchy (coarse→fine cross-attention) **measurably helps, more so on 3D rotation**.
- Remaining frontier: **full-res 2→8→32** needs windowed attention to afford the recipe; **SO(3)
  self-occlusion** needs a stronger generative residual than the warp (diffusion head) to fully sharpen.
