# 2→8→32 (128px) single-shift GMM run — handoff

Branch: `feat/uncertainty-maps-128`. Date: 2026-06-20. Author: automated session (user away).

This documents the entropy-map + 2→8→32 work, the long single-shift training run launched while you were
out, and how to read / resume / extend it.

## What was built

1. **Entropy / uncertainty maps** (your stated need — "review uncertainty maps", *not* VQ, which a deep
   review confirmed would re-hit the old composite-index blob):
   - `hvp/eval/uncertainty.py` — sample N next-frame predictions from the distributional (GMM) head,
     decode each, take the **per-pixel std** = predictive uncertainty. Rendered **B&W: white=certain,
     black=uncertain**.
   - `scripts/uncertainty_map.py --ckpt <ckpt>` — standalone maps from any checkpoint (rebuilds the model
     from the *saved* config, so GMM/128 checkpoints load correctly).
   - **In-training**: every validation writes stacked PNGs (GT / Recon / Pred / **Uncertainty**) to
     `logs/so3_128/val_strips/stepNNNNNN_imgKK.png`. Verified on the 64px `so3_gmm` checkpoint: the map
     correctly whitens the static background and blackens the moving/rotating shape + its edges.

2. **2→8→32 spatial pyramid** (`configs/so3_128.yaml`): `topology.image_size=128` with the existing tap
   schedule yields **BOT 32 / MID 8 / TOP 2** (verified). New SO(3) 128px data rendered
   (`scripts/gen_data_so3_128.py`; hashes wired into the config). 24.5M params.
   - Memory probe: BOT is 32²×9 = **9216 attention tokens** (the bottleneck). bs=4 fits at 10.25 GB / ~1
     it/s; bs≥6 thrashes the 16 GB GPU. Windowing-via-mask does **not** cut SDPA memory. So the run uses
     **bs=4 + gradient accumulation ×4 = effective batch 16** (matches the proven 64px regime).

3. **Review-driven correctness/robustness fixes** (from a 15-agent component review; merged with a
   concurrent checkpoint-durability change):
   - **S1 explained (not a bug)**: the old "recon 30 dB but pred 13.5 dB" gap is a metric-framing artifact
     — the single-step pixel *ceiling* through the multi-rate shifted decode is ~22.8 dB, not 30. New
     honest metrics: `val/pred_psnr_oracle` (ceiling), `val/pred_oracle_frac` (fraction captured),
     `val/pred_changed_*` (moving-region), `val/pred_psnr_bot` (finest-stage with oracle coarse), and the
     pixel metric now uses the **MMSE mixture mean** + **clamps** the decode.
   - **S2 fixed**: periodic checkpoints are now **stamped + pruned** (`max_checkpoints_keep`) + **atomic**
     (`.tmp`+rename) + a **wall-clock safety net** (`save_every_min`). A killed run now leaves recoverable
     step checkpoints (the flagship left only `last.pt`).
   - **Stability**: consecutive-NaN **stall canary** (aborts after 50), a **reshuffling** infinite loader
     (replaces `itertools.cycle`, which replayed a fixed order and cached an epoch in RAM), zero-spatial &
     sequence-length **validate guards**.
   - Rollout/AR stays **off** (single-shift only, per your scope).

## The run

```
configs/so3_128.yaml  (wandb disabled for robustness — the prior long run likely died via a wandb hang;
                       all metrics print to logs/so3_128/train.log and strips/checkpoints go to disk)
launched:  PYTHONUNBUFFERED=1 venv/Scripts/python.exe -u -m hvp.train --config configs/so3_128.yaml \
             --set run.wandb=false  2>&1 | tee logs/so3_128/train.log
```

- ~6000 optimizer steps (each = 4 microbatches), ~7h target. Checkpoints: `logs/so3_128/oneshot/last.pt`
  + `stepNNNNNN.pt` (3 newest kept) every 250 steps / 20 min.
- **Resume**: `... --config configs/so3_128.yaml --resume logs/so3_128/oneshot/last.pt`
- **Extend**: re-launch with `--set optim.max_steps=<more>` and `--resume`.

## How to read progress (in `logs/so3_128/train.log`)

Per validation line:
```
[val@STEP] recon R dB ssim S | pred P (lift L, changed-lift C, oracle-frac F) | bot B dB | mid_casc M
```
- **recon** should climb steadily (target ~30+ dB) — primary tokenizer health.
- **oracle-frac F** = fraction of the *achievable* single-step headroom captured (the honest predictor
  score; the absolute `pred` dB is capped by the ~22.8 dB decode ceiling, so don't expect ~30).
- **changed-lift C** = moving-pixel PSNR over persistence (the honest motion signal; >0 = beating copy).
- **mid_casc M** = cross-attention hierarchy lift (the dissertation claim).
- **erank** per stage (final summary) — collapse canary; must stay near the latent dim.

## Entropy maps

`logs/so3_128/val_strips/` — newest `stepNNNNNN_*.png` = latest. Bottom row of each is the B&W
uncertainty map (white=certain). Or regenerate from any checkpoint:
`venv/Scripts/python.exe scripts/uncertainty_map.py --ckpt logs/so3_128/oneshot/last.pt --out logs/unc_128`

## Disk

C: was at 92% (35 GB free) at launch. The run caps its own checkpoints at 3 (~0.8 GB). Old experiment
checkpoints in `logs/so3_*` (~2.8 GB) were left intact; clear them if you need space.
