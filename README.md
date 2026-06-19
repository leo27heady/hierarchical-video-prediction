# hierarchical-video-prediction (`hvp`)

A clean, falsification-first research codebase for **hierarchical video prediction** — the core of the
dissertation. It is a **VQ-free** (continuous) hierarchical video autoencoder with **causal temporal
downsampling**, plus a **single-step, BLT-style cross-attention cascade** that predicts the next frame
coarse-to-fine. Trained **end-to-end in one shot** on synthetic rotating-shape video.

> **Status (v1, 2026-06-19):** all gates pass — see [`docs/RESULTS_V1.md`](docs/RESULTS_V1.md).
> The temporally-downsampled MID latent is highly predictable (held-out VE ≈ 0.84), the cross-attention
> hierarchy adds real lift over independent stages, and single-step prediction beats persistence in pixel
> space (no blob). Detailed design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · plan:
> [`docs/PLAN_VQFREE_HIER_PREDICTOR.md`](docs/PLAN_VQFREE_HIER_PREDICTOR.md) · story & next steps:
> [`docs/SESSION_SUMMARY.md`](docs/SESSION_SUMMARY.md).

---

## Idea in one picture

```
ENCODER (causal-conv 3D, NO quantization)            PREDICTOR (single-step BLT cascade, ONE forward pass)
────────────────────────────────────────            ──────────────────────────────────────────────────────
x [B,3,9,64,64] in [-1,1]                            per-stage continuous latent z_s  (detached as pred target)
  │ causal Conv3d, left-pad (K-1,0)                  predict TOP once  (3D-RoPE self-attn) ─┐ hidden h_top
  ▼ /4 spatial                                         cross-attn (leak-free RF windows) ───┤ conditions
 BOT  16×16 × 16ch  @ T=9                            predict MID once ◄────────────────────┘  the finer stage
  ▼ /4 spatial, /2 time (causal)                       cross-attn ──────────────────────────┐
 MID   4×4  ×  8ch  @ T=5                            predict BOT once ◄─────────────────────┘
  ▼ /4 spatial, /2 time (causal)                     decode(predicted BOT latent) → predicted pixels
 TOP   1×1  ×  8ch  @ T=3
DECODER: causal-conv 3D, deep-supervised, BOT→64×64  Loss = recon(pixels) + VICReg + λ·Σ MSE(ẑ_s, stopgrad(z_s,next))
```

- **No VQ.** Plain continuous latents (an L2 metric where rotation moves smoothly).
- **Causal temporal downsampling.** `9 → 5 → 3` token-frames via causal stride-2 convs (`1+s·t` mapping).
- **Single-step, not recursive.** Teacher-forced `t→t+1`, one pass; no envelope rollout.
- **Detached one-shot.** The encoder is shaped by reconstruction (+ a VICReg variance floor); the cascade
  predicts the encoder's own **detached** latents, so prediction can't collapse the representation.
- **Cross-attention hierarchy.** Coarse hidden conditions the finer stage via causal **receptive-field
  overlap** windows (built once, cached, leak-free).

## Results (64×64, held-out cross-split, one-shot 3000 steps; see `docs/RESULTS_V1.md`)

| signal | result |
|---|---|
| reconstruction (held-out) | **PSNR 32.3 dB, SSIM 0.975**, all stages near full effective rank (no collapse) |
| **MID predictability** (independent stage) | `ve_changed ≈ 0.73` (bot 0.84) — *temporal-downsampling-is-fatal worry falsified* |
| **cross-attention hierarchy lift** | BOT **+0.061**, MID **+0.022** VE over independent stages |
| single-step prediction (pixel) | **+5.10 dB / +0.126 SSIM** over persistence; tracks rotation, not a blob |

## Install

```powershell
# Windows, Python 3.12, CUDA
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[dev]"
.\venv\Scripts\python.exe -m pip install wandb      # optional, for logging
```

## Data

Rotating-shape clips (Z-axis rotation), `(C,T,H,W)∈[-1,1]`, `T=9`, 64×64. Rendering needs the
`simple-shape-dataset-toolbox` (PyVista). `scripts/gen_data_64.py` renders an **independently-rendered
train + val cross-split** into the shared cache and prints the two hashes (already wired into
`DataConfig`):

```powershell
# run with a venv that has pyvista (e.g. the SEED-Voken venv)
<pyvista-venv>\Scripts\python.exe scripts\gen_data_64.py
```

## Usage

```powershell
$PY = ".\venv\Scripts\python.exe"

# tests (CPU, Windows-safe; ~27 unit tests + RSS watchdog)
$PY -m pytest tests -q

# G0 — the predictability gate (decisive, < a GPU-hr): is the MID latent predictable?
$PY -m hvp.probes.continuous_predictability --ckpt logs/oneshot/last.pt --steps 800

# one-shot training from a config (tokenizer + cascade), wandb + image tables
$PY -m hvp.train --config configs/base_64.yaml                  # main 64x64 run
$PY -m hvp.train --config configs/dev.yaml                      # fast smoke (~1 min)
$PY -m hvp.train --config configs/big_64.yaml --set optim.lr=2e-4   # + CLI override

# pixel-space single-step eval (+ GT/Pred strip PNG)
$PY scripts/eval_predict.py --ckpt logs/oneshot/last.pt --strip logs/predict_strip.png

# autoregressive rollout / per-horizon drift eval (P1) on the long-horizon set
$PY scripts/eval_rollout.py --ckpt logs/oneshot/last.pt --strip logs/rollout_strip.png

# resume a killed run
$PY -m hvp.train --config configs/base_64.yaml --resume logs/oneshot/last.pt
```

### Configs
Every knob (model / training / data / logging / per-stage dims / depth / connections / optimizer /
schedule) is a config field. `configs/_base.yaml` documents all defaults with comments; presets
(`dev`, `base_64`, `big_64`) inherit via `_base_:` and override only what they change. Unknown keys are
rejected (typo guard). Override anything on the CLI: `--set predictor.stage_dims=[192,256,256] optim.lr_schedule=warmup_cosine`.

### wandb
Set `run.wandb=true` (online if `WANDB_API_KEY` is set, else `run.wandb_mode=offline`). Logged: all
train/val scalars (loss, recon PSNR/SSIM, per-stage predictive VE, cascade lift, effective rank, grad
norm, throughput) and a **GT / Recon / Pred image table** with columns `image_index`, `category`, `image`.

## Layout

```
hvp/
  config.py          frozen-dataclass tree; TopologyConfig derives every stage shape
  data/              shape_video.py — reads cached RGB clips (true cross-split)
  nn/conv3d.py       CausalConv3d, ResBlock3d, Upsampler3d, FrameNorm3d (per-frame, causal)
  tokenizer/         encoder.py · decoder.py (deep-supervised) · model.py
  attention/         rope.py (factorized 3D RoPE) · sdpa.py (qk_norm) · blocks.py · masks.py
  predictor/         continuous.py (stage worker + VE) · masks.py (leak-free RF cross-mask) · cascade.py
  objectives.py      deep-sup recon + VICReg + detached predictive
  metrics.py         PSNR / SSIM / frame strips
  train.py           one-shot trainer (+ wandb)
  wandblog.py        thin wandb wrapper (no-op when disabled)
  probes/            continuous_predictability.py  (G0 gate)
scripts/             gen_data_64.py · eval_predict.py
tests/               conftest.py (RSS watchdog + Windows OMP guard) + 27 tests
docs/                ARCHITECTURE · PLAN · RESULTS · SESSION_SUMMARY
```

## Roadmap

The full V2 roadmap (honest state-of-the-union, 12 prioritized gaps, phases P0–P8 with falsifiable
gates, eval-upgrade table, risks) is in **[`docs/PLAN_V2_ROADMAP.md`](docs/PLAN_V2_ROADMAP.md)**.
Binding constraints, front-loaded:

1. **P0 — eval + repro spine** ✅: per-horizon/LPIPS/FVD/changed-region metrics + resumable checkpoints.
2. **P1 — autoregressive rollout** ✅: stable to H=24, beats persistence 24/24, TF→AR gap +1.61 dB (bounded). See [`docs/RESULTS_P0_P1.md`](docs/RESULTS_P0_P1.md).
3. **P2 — anti-drift training**: scheduled sampling + noised-history conditioning (Diffusion-Forcing/Self-Forcing).
4. **P3 — distributional head** (GMM→diffusion) wired through sampling + best-of-N (de-blur + multimodal honesty).
5. **P4 — hierarchy-anchors-horizon** (the money result) + multi-seed error bars.
6. **P5+ — SO(3) / windowed BOT attention / reversibility / joint-training escalation.**
