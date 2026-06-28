# DLH improvements — gated implementation plan (v2, red-teamed)

> Branch `reproduce/dlh`. Builds on the faithful port (`hvp_dlh/`) and [RESULTS_DLH_REPRODUCTION.md](RESULTS_DLH_REPRODUCTION.md).
> **v2 status: GO after the blocker-fixes below are applied** (v1 was NO-GO; a 4-critic adversarial red-team found
> 6 blockers — all folded in here, each tagged `[RT]`). Every step has a smoke gate, a metric gate on the *right*
> metrics, and a decision rule. Read §0 (discipline) and §S (success criterion) before executing.

---

## §S. Global success criterion (what "the improvements worked" means)

The program succeeds if, **vs the frozen baseline**, on ≥64 held-out clips with mean±95%CI over ≥3 rollout seeds:
- **fg-MSE↓ and LPIPS-fg↓** at horizons +25/+50 (the in-training-range decision horizons), with **+100/+150 as
  diagnostics**, AND
- **no identity collapse** (defined: `fg_mse < 0.9·fg_mse_black` AND `pred_mean_intensity > 0.5·gt_mean_intensity`
  at +150), AND
- the hierarchy is **no worse than baseline** (top-level KL and `K_mean` not below baseline — engaging a *3rd* level
  is an *optional* bonus hypothesis, §1d, not a requirement; 2 effective levels is acceptable per DLH's
  minimum-levels behaviour).

A change that improves fg-MSE/LPIPS at the cost of the hierarchy is logged but **not** adopted as the new baseline
without an explicit note. SSIM/PSNR are reported for continuity but are **never** a decision metric (MMNIST SSIM is
93%-background; black scores 0.76).

---

## §0. Gating discipline (applies to every step)

1. **One change at a time**, measured against a **matched control** — not against a baseline that differs in >1
   variable. `[RT]` The frozen `baseline_iter8000.pt` is seq100/batch50/8k/**tanh**; the sweep runs are
   seq50/batch32/**sigmoid** — so they are NOT directly comparable. Every sweep step trains its own **matched
   control** (same seq/batch/iters, only the one variable changed) and reports Δ vs that control.
2. **Decision metrics = fg-MSE + LPIPS-fg only.** SSIM/PSNR are continuity-only.
3. **Statistical rigor `[RT]`**: `torch.manual_seed` before every rollout (reproducible best-of-N); report
   **mean ± 95% CI** over **≥3 rollout seeds** on **≥64 clips**; a winner must beat the control by **> the CI**
   (minimum effect size), not by a point estimate.
4. **best-of-N selection must match the decision metric `[RT]`.** `eval.py` currently picks the best-of-N sample by
   SSIM, then reports fg-MSE/LPIPS on that SSIM-optimal sample. Fix: select best-of-N by **fg-MSE** (or report
   mean-over-samples, no oracle, as primary + best-of-N separately).
5. **Track the hierarchy every run**: log `skl_per_level` and `K_mean` (wire them into `eval.py` `[RT]` — currently
   only in the train loop). Thresholds are **derived from the baseline**, not arbitrary (§1 gate).
6. **Judge at in-range horizons `[RT]`**: a seq-50-trained model has never compounded error past ~50 frames, so the
   sweep decision uses **+25/+50**; +100/+150 are diagnostics. Final candidates are re-ranked at the full seq-100
   regime before adoption.
7. **Smoke before launch**; **commit + tag** each change and its config (§R reproducibility).

---

## §B. Blocker-0: commit an auto-resume launcher BEFORE any multi-hour run `[RT]`

The runs have died at every session boundary; the "harness-tracked + --resume" mandate was prose only. **Deliver a
launcher first:**
- `scripts/run_until.sh` (committed): loops `python -m hvp_dlh.train --resume <run>_last.pt …` until `step==total`,
  launched via the harness-tracked background so it survives session teardown; re-invokes on non-zero exit.
- Set `ckpt_every=1000` (cheap; bounds lost work to ≤1000 iters).
- Verify `num_workers=0` in the actual run config (orphan-leak guard).
- **Gate:** kill the process mid-run, re-launch, confirm it resumes from the last checkpoint at the right step.

---

## Step 0 — Measurement + frozen baseline (PREREQUISITE)

**Goal.** See the effects of every later step; freeze the reference.

**Changes.**
- `cp logs_dlh/mmnist-3lvl-fix_last.pt logs_dlh/baseline_iter8000.pt`; `git tag dlh-baseline`.
- **LPIPS, done correctly `[RT]`.** `lpips` IS installed in the venv (verified). Recover the reference wrapper via
  `git show research/cwvae-backbone:hvp/eval/quality.py` (it exists on that branch; not on this one). Foreground-
  masked LPIPS needs a **spatial** map, not a scalar:
  ```python
  _lpips = lpips.LPIPS(net='alex', spatial=True).to(dev).eval()   # (B,1,h,w) distance map
  def lpips_fg(pred, gt, fg):                 # pred,gt (N,3,64,64) in [-1,1]; fg (N,1,64,64) bool
      dmap = _lpips(pred, gt)                  # (N,1,h,w)
      dmap = F.interpolate(dmap, size=fg.shape[-2:], mode='bilinear', align_corners=False)
      return (dmap * fg).sum() / fg.sum().clamp(min=1)
  # full-frame LPIPS = dmap.mean(); convert [0,1]->[-1,1] and gray->3ch before the call
  ```
- Compute LPIPS **once on the final `pred_keep`** (after best-of-N selection), batched/chunked — **never inside the
  best-of-N loop `[RT]`** (that would be 10× AlexNet passes and an OOM/throughput risk).
- Wire `skl_per_level`, `K_mean`, and the §S collapse proxies into `eval.py`'s JSON report.

**Gate `[RT]`.**
- *Smoke:* `import lpips` ok; `spatial=True` returns a finite (N,1,h,w) map; fg mask non-empty.
- *Sanity (validates the metric on the DECISION variant):* a 3-point monotonicity on **fg-LPIPS** (where black is
  clearly bad): `model < persist < black`. (Full-frame LPIPS of black-vs-mostly-black can be small — do NOT gate on
  it.) Measure the baseline's `skl_per_level[2]`, `K_mean` here to set §1 thresholds.

**Compute.** ~1–2 h (no training; eval is the cost).

---

## Step 1 — Faithfulness/config fixes + minimal sweep

**1a. sigmoid output** — config `dec_out_activation: tanh→sigmoid`. Hypothesis (not guarantee `[RT]`): fixes the
`tanh[-1,1]` vs `[0,1]` bright-pixel saturation; the gate decides, revert to tanh if it regresses.

**1b. dec_stddev** — config knob. **Corrected direction `[RT]`:** `NLL = 0.5·SE/var` is up-weighted as `var`
shrinks while `β·KL` is var-independent, so **smaller std → sharper recon AND *relatively weaker* KL pressure (KL
becomes cheap) → risks posterior/hierarchy collapse.** (v1 said "more KL pressure" — wrong.) So small-std must be
paired with the anti-collapse insurance in 1c and gated on `skl_per_level`.

**1c. Per-level free-nats = anti-collapse insurance ONLY `[RT]` (NOT a 3rd-level engager).** Free-bits relieves
downward KL pressure on **already-ticking** levels (collapse type a: latent→prior). It **cannot revive a frozen
level**: `d/dx max(x,c)=0` for `x<c`, and the clock `e` is hard/non-differentiable, so a sub-floor or clock-frozen
level gets zero gradient from the floor. Implement as a **single in-graph KL path** (the v1 sketch double-counted
with the global clamp):
```python
# config: free_nats_per_level: tuple = (0.0, 0.0, 0.0)   # nats/frame floor per level (off by default)
# loss(): keep skl_level[n] as in-graph (B,1) tensors across t -> sum over t -> mean over B -> (N,)
skl_pl = torch.stack([s_t_summed[n].mean(0) for n in range(N)])      # (N,), in graph
if any(cfg.free_nats_per_level):
    skl_pl = torch.maximum(skl_pl, torch.tensor(cfg.free_nats_per_level, device=dev) * T)
state_kl = skl_pl.sum()        # DELETE the old global clamp + skl.mean() so there is ONE KL path
```

**1d. (Optional) engage the 3rd level via the CLOCK, not free-nats `[RT]`.** Only if quality stalls AND we suspect
capacity is the limit (2 levels is otherwise acceptable, §S). The levers that actually drive ticking:
(i) **extend/anneal `force_tick`** (longer dense warmup → upper levels learn before selection takes over);
(ii) **anneal a tick-threshold margin**: `e = (kl_c < kl_s + m(t))`, `m` annealing `+→0` (tick more early, settle
later). Test as a separate small ablation; gate on `skl_per_level[2]` rising *without* fg-MSE/LPIPS regressing.

**Minimal sweep `[RT]`** (cut from v1's 4-point grid; all seq50 / batch32 / force_tick 1500 / **β-warmup 1500** so
post-warmup ≥ warmup `[RT]` / total 5000):
| run | activation | dec_stddev | free_nats/lvl | role |
|---|---|---|---|---|
| **A0** | **tanh** | 1.0 | (0,0,0) | **matched control** (the reference for all Δ) |
| A | sigmoid | 1.0 | (0,0,0) | isolates sigmoid vs A0 |
| C | sigmoid | 0.3 | (0,0,0) | brackets the recon-vs-KL tradeoff |
| D | sigmoid | (winner of A/C) | (0,1,1) | anti-collapse insurance on the winner |

**Gate.**
- *Smoke:* sigmoid∈[0,1]; per-level floor verified (`skl_per_level[n] ≥ floor`); 50 iters, loss finite, single KL path.
- *Metric gate:* a run beats **A0** if fg-MSE **and** LPIPS-fg drop at **+25/+50** beyond the 95% CI (§0.3), with no
  §S collapse and `skl_per_level[2]`/`K_mean` ≥ baseline.
- *Decision:* carry the **top-2** configs (not 1 `[RT]`) into Step 2, because dec_stddev × batch-size interact.

**Compute.** 4 runs × ~2 h + eval ≈ **~9–10 h** sequential.

---

## Step 2 — Gradient accumulation → effective batch 100, final-regime A/B

**Goal.** Remove the batch deviation; pick the final config in the real regime.

**Change.** `train.py` accumulate over `accum_steps` micro-batches; **average the logged metrics across
micro-batches `[RT]`** (else logs reflect one micro-batch):
```python
opt.zero_grad(set_to_none=True); accm = []
for k in range(cfg.accum_steps):
    obs = next(it).to(dev); loss, met = model.loss(obs, beta=beta, force_tick=force_tick)
    (loss / cfg.accum_steps).backward(); accm.append(met)
gnorm = clip_grad_norm_(...); opt.step()
met = {k: sum(m[k] for m in accm)/len(accm) for k in accm[0]}
```

**Gate.**
- *Benchmark first `[RT]`:* time 50 steps WITH `accum_steps=2` at seq100 to get true s/optimizer-step; budget from
  the measured rate (do not assume 3.5 s/it is per-optimizer-step).
- *Metric gate:* full run (seq100, eff-batch 100, ~15k) of **both** carried configs; the §S winner becomes the new
  `baseline_v2.pt` (re-tag).

**Compute.** 2 full runs ~12–15 h each (measure first).

---

## Step 3 — Inter-level cross-attention (one architecture A/B)

**Goal.** Test the BLT-faithful local↔global connection vs the MLP top-down sweep. Expect a small effect.

**Change.** Flag `inter_level_attn`. **Construct the MHA only when the flag is on `[RT]`** (so flag-off preserves
RNG order → bit-match). Replace the top-down `c`-sweep with a masked MHA where level n's query attends over
`{sᵐ,dᵐ : m≥n}`; **a frozen level's own carried posterior stays an UNMASKED key `[RT]`** so no row is fully masked
(avoids NaN).

**Gate.**
- *Cheap first `[RT]`:* a seq50 reduced-iter relative A/B (attn vs MLP) to decide go/no-go before any seq100 pair.
- *Smoke:* flag-off bit-matches MLP baseline; no NaN when all upper levels frozen.
- *Metric gate:* ≥3 seeds, beat MLP beyond CI on fg-MSE/LPIPS, else revert (flag off) and record the null.

**Compute.** cheap A/B ~3 h; full pair only if it shows signal.

---

## Step 4 — Bigger bets (BLOCKED-ON-PORT `[RT]`; only after 0–3)

> `scripts/` on this branch has only `.pyc`; `discriminator.py`/`diffusion_head.py`/`quality.py` live on the
> research branch. **Recover them via `git show research/cwvae-backbone:<path>` and commit before scheduling Step 4.**
> Consider doing **4a before Step 3** — the soft decoder is the *headline* weakness and likely higher-value than the
> cross-attn A/B (decide after Step 2 results).

**4a. Sharp decoder (the headline weakness).** Freeze DLH latents; train a perceptual+adversarial decoder
(L1+LPIPS+hinge-PatchGAN) or small diffusion head on the frozen bottom output; ELBO untouched. **Gate:** oracle
headroom first (>2 dB / >0.05 LPIPS to gain?); then LPIPS/edge↑ at matched identity. **Lead metric LPIPS/FVD, never
PSNR** (a correct sharpener won't beat the blur on PSNR).

**4b. Per-pixel decision on UNIFORM 16×16 spatial latents (best idea, done right).**
1. *Spatial-latent refactor:* conv map (not flattened); 1×1-conv heads; `ConvGRU`; `s/d/c/x` = `(B,C,16,16)` at
   **uniform** resolution (NOT a 4/8/16 pyramid — that rebuilt the nulled spatial pyramid). **Memory gate `[RT]`:**
   forward+backward one seq100 batch, assert peak CUDA < ~14 GB before any run; likely forces seq≤50 or batch≤32.
   **Gate:** spatial-latent DLH must *match* vector DLH before adding gating.
2. *Per-pixel `e`:* channel-summed Gaussian KL → `e (B,1,16,16)`; 1×1-conv prior-factor; nested = same-location
   Hadamard; L1/entropy reg on the e-map. **Cheap pre-check:** offline per-location KL map must visibly segment the
   digits before building full gating. Judge on fg-MSE/LPIPS/segmentation overlay.

---

## Skip list (with reasons)
- **Spatial pyramid 4×4/8×8/16×16** — rebuilds the spatial pyramid already proven NULL on MMNIST (memory
  `hierarchy-is-null-on-mmnist`). The salvageable part is uniform-16×16 spatial latents (4b).
- **GRU→causal-transformer temporal carry** — its only prize (parallel-over-time training) is forfeited by the
  online dynamic clock; the temporal path isn't the bottleneck.

---

## §F. Faithfulness deviations — decision `[RT]`
From the audit, two deviations remain after this plan; decide explicitly:
- **Dropped bottom-level transition-KL** (`model.py` skips `n=0` in term c): `e¹≡1`, so it only omits a constant
  `−log p(e¹=1)` regulariser. **Decision: keep dropped** (negligible), documented here. (Add it back only if you
  want strict eq-5c parity — 2-line change.)
- **`force_tick` dense warmup**: non-paper bootstrap; anneals off by iter 3000. **Decision: keep** (needed for
  cold-start), documented. Confirm `force_tick_iters=0` still trains (so it's a convenience, not a crutch).

## §R. Reproducibility
Commit each change separately; `git tag` every baseline (`dlh-baseline`, `baseline_v2`, …); the wandb run name
encodes the config; the per-checkpoint JSON report stores the full config + metrics. One change per run.

---

## Sequencing, honest compute `[RT]`, and gate summary

**Two timelines, stated honestly:** HAPPY-PATH (0→2, no failures) ≈ **3 days**; REALISTIC (0→3, +eval overhead
~10–20 min/sweep-run & ~1 h/full-run, +1.3–1.5× crash contingency given the history) ≈ **5–8 days**.

| step | change | smoke gate | decision gate | compute |
|---|---|---|---|---|
| B0 | auto-resume launcher | kill→resume works | (infra) | ~1 h |
| 0 | LPIPS-fg(spatial) + metrics + freeze | fg-LPIPS monotone model<persist<black | (infra) | ~1–2 h |
| 1 | sigmoid + dec_stddev + free-nats(+clock 1d) | sigmoid∈[0,1]; single KL path | beat A0 control on fg-MSE/LPIPS@+25/+50 beyond CI | ~9–10 h |
| 2 | grad-accum batch 100, final A/B of top-2 | benchmarked s/step | §S winner → baseline_v2 | ~24–30 h |
| 3 | inter-level cross-attn (flag) | flag-off bit-match, no NaN | cheap A/B signal → beat MLP beyond CI | ~3 h +opt |
| 4a | sharp decoder (port first) | oracle headroom | LPIPS/edge↑ at matched identity | research |
| 4b | uniform-spatial + per-pixel e (port first) | mem<14GB; refactor matches baseline | KL-map segments → fg/LPIPS | research |

**Critical path to a better model:** B0→0→1→2. Step 3 is one bounded A/B (consider 4a instead). Step 4 is the
research frontier and is blocked on porting scripts.

---

## Open risks (carried)
- Small `dec_stddev` trades sharpness for hierarchy collapse — gate on **both** fg-MSE/LPIPS **and** `skl_per_level`.
- sigmoid likely needs fresh training (decoder tuned for tanh) — use fresh sweep runs, not warm-start, for 1a.
- All gains may be invisible on SSIM — expected; never let an SSIM non-improvement veto an fg-MSE/LPIPS win.
- Completeness red-team critic did not run (placeholder output) `[RT]` — before executing, do an end-to-end wiring
  check that every new config key flows config→loss/eval→logged scalar→gate.
