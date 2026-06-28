# E1 — on-the-fly rollout training on CW-VAE (v2: red-teamed + smoke-validated)

> E0 confirmed a large exposure gap (open-loop fg-MSE 0.49 vs teacher-forced 0.14 @+25). A 4-critic red-team
> returned **GO-WITH-CHANGES** (all blockers folded in below). A smoke (`scripts/e1_smoke.py`) validated the
> mechanism and surfaced the std-collapse failure mode. **No reversibility yet** — E1 isolates the method;
> reversibility (longer M) is the next stage.

## Smoke evidence (done)
- Gradient reaches **all** levels but is ~100× thinner at the slow ones and **grows with M** (L2 grad 3.3→7.0
  as M 4→40) → empirical motivation for longer windows (reversibility). VRAM **1.41 GB** (12× headroom →
  no checkpointing; raise batch). Rollout SE **−6.5 over 800 steps** (trainable), ELBO stable.
- **Latent-collapse confirmed:** free-run prior std drifts 0.257→0.155 → the std-collapse mitigation is
  mandatory (below), not optional.

## Corrected design (every red-team blocker applied)
- **Backbone/location:** CW-VAE, warm-started from `../cwvae/logs_torch/cwvae/last.pt`. New trainer **in this
  repo**: `hvp_rev/train_rollout.py` (imports `CWVAE` via sys.path like `e0_exposure_gap.py`); **`../cwvae`
  stays pristine**, its commit pinned in the trainer header.
- **`ctx_tf = 36`** (multiple of factor²=36; `ctx_tf=24` crashes via `_stack([])` on the empty top slice).
  Same ctx for baseline and treatment (apples-to-apples).
- **Phase B = `hierarchical_unroll(use_obs=[False]*L, initial=carried)` called DIRECTLY** (never `open_loop`,
  which is `@torch.no_grad` → would zero rollout grad = fake null). Free-run `post==prior` aliased → **NO KL in
  Phase B**; rollout loss = **Gaussian-NLL on decoded M frames only**.
- **Detach** Phase-A→B state for run 1 (train prior dynamics from a fixed start; bounded memory; clean isolation).
- **Per-frame normalized losses:** `loss = ELBO_A/ctx_tf + λ_roll · rollout_NLL/M` (so λ controls the true
  balance, not the 36:M frame-count ratio).
- **Std-collapse mitigation (confirmed needed):** compute the **rollout loss with `sample=False`** (prior mean →
  no variance-shrinkage incentive); keep `sample=True` for metric eval. Log free-run std + top-KL every eval.
- **Memory/throughput:** drop gradient checkpointing (unneeded); **batch 50–64**; lr `1e-4`–`3e-4`; startup
  tripwire `assert peak_reserved < 12 GB`.

## Arms & controls (matched compute, ≥3 TRAINING seeds each)
| arm | purpose |
|---|---|
| **frozen baseline** (E0) | reference |
| **λ_roll=0 warm-start** (continue teacher-forced, same steps) | rules out "just more fine-tuning" |
| **no-grad-rollout** (stop-grad on prior samples before decode; same exposure/budget) | isolates *gradient through the rollout* (the thesis) from mere exposure |
| **with-grad rollout** (the method) | must beat **no-grad** beyond the training-seed band |

Then a λ_roll ∈ {0.3,1,3} × M ∈ {4,6,8} sweep **only if** the first pass shows signal.

## Gate (post-training)
- **PRIMARY = in-window:** fg-MSE + fg-LPIPS at **+M/2 and +M** (where the loss has gradient), beating the
  λ=0 and no-grad controls by **> combined std on BOTH metrics** (`gate.py` threshold), over ≥3 training seeds
  (clip-bootstrap nested inside the seed band, reuse `a0_longhorizon_ci.py`).
- **+25/+50 = extrapolation diagnostics** (report, do not gate). **M-monotonicity sub-claim:** in-window
  closure grows with M. **Pre-registered:** in-window closure + positive M-trend = SUCCESS that motivates the
  longer-M reversibility stage; full +50 closure NOT required at E1.
- **Guards:** short-horizon (+5) not regressed; **lap-var not dropped** (sharpness); **top-KL ≥ 0.5× control**
  (not just KL>0); free-run std not collapsing.

## Live tracking
Every 500 steps: loss/elbo/rollout, per-level prior grad-norm, free-run std, peak VRAM. Every 2000 steps:
open-loop eval (≥48 clips) → fg-MSE/fg-LPIPS@{+M/2,+M,+25,+50}, **lap-var**, +5; **save GT-vs-pred strip** to
`logs_rollout/strips/`; metric-vs-step plots; Δ vs E0.

## Resource discipline (disk 13 GB free, 97% full; GPU 16 GB)
- **One `_last.pt`**, written via temp-file + `os.replace()` (no half-written orphans), `ckpt_every=2000`,
  never keep history. Resumable ckpt ≈ **150 MB** (model + 2 Adam moments).
- Strip cap **in code** (glob+sort+unlink, keep last ~10). JSON evals only. No wandb image dumps.
- `df`/`nvidia-smi` at start/mid/end; **abort if free disk < 3 GB**, checked at startup AND before every ckpt write.
- ⚠ Standing risk: the **15 GB `logs/` pile** (prior research) — pending user decision to prune (E1 itself
  needs < 200 MB, so this is margin, not a blocker).

## Open decisions (for the user)
1. **Sweep scope:** full matrix (~10–20 GPU-h) vs **reduced first pass** (M=8: frozen / λ=0 / no-grad /
   with-grad, 3 seeds — recommended), ablate λ/M only if signal.
2. **Disk:** prune stale `logs/` runs to restore margin? (E1 runs fine without it.)
(Autonomous calls already taken: detach=ON for run 1; trainer lives in this repo `hvp_rev/`.)
