# DLH → match-and-surpass CW-VAE — gated plan (v4, red-teamed)

> Branch `reproduce/dlh`. **Supersedes** [PLAN_DLH_IMPROVEMENTS.md](PLAN_DLH_IMPROVEMENTS.md). A 4-critic
> adversarial red-team returned **GO-WITH-CHANGES**; all blockers folded in, tagged `[RT]`. Evidence on disk:
> [match_eval.json](../logs_dlh/match_eval.json), [compare_curves.json](../logs_dlh/compare_curves.json),
> memory `dlh-blur-is-decoder-bottleneck`. **Read §D and §G first** — the red-team showed the central question is
> already half-answered by data in the repo.

---

## §D. Diagnosis + what the data ALREADY says

**(1) Single-frame sharpness** (matched: 64 clips, ctx=36, single seeded sample, open-loop both):

| | GT | DLH(8k) | CW-VAE(ep299) |
|---|---|---|---|
| fg-MSE ↓ | — | 0.430 | 0.428 (tied) |
| Laplacian-var ↑ | 0.056 | 0.0030 | 0.0071 (CW-VAE 2.4× sharper) |
| LPIPS ↓ | — | 0.334 | 0.265 |

**(2) Horizon-resolved head-to-head** (`compare_curves.json`, best-of-5, 150 frames) — **the decisive thing was
already run** `[RT]`:

| fg-MSE window | DLH | CW-VAE | SimVP-AR | winner |
|---|---|---|---|---|
| +0–25 | 0.345 | **0.311** | 0.500 | CW-VAE |
| +25–50 | **0.445** | 0.437 | 0.574 | ~tie→DLH |
| +50–100 | **0.499** | 0.498 | 0.595 | tie |
| +100–150 | **0.520** | 0.530 | 0.602 | **DLH** |

DLH's fg-MSE **crosses below CW-VAE at frame ~24 and stays below through 150** (PSNR & full-MSE likewise). So
**DLH's long-horizon / bounded-drift thesis is already validated on disk** — it beats both CW-VAE and SimVP-AR
late. CW-VAE wins **early** (0–25) on accuracy *and* sharpness. SimVP-AR wins SSIM only (background pathology,
memory `mmnist-ssim-is-background-dominated`).

**(3) The gap is conditioning, not capacity** `[RT]`. The two decoders are **the same size** (DLH 3.76M vs
CW-VAE 3.78M); CW-VAE's whole 6.35M param surplus is its bottom-up **encoder**. So the 2.4× sharpness edge comes
purely from CW-VAE's decoder reading a 200-dim **deterministic** channel `[s,det]` vs DLH's stochastic-only
context. **Ruled out by experiment:** inference noise / `min_std` (mean-mode doesn't sharpen) and open-loop
dynamics (teacher-forced recon still blurry). Faithful to the paper (Eq.1/§A.1): DLH decodes from `f(s^{>n})`.

**Net:** the real, narrowed problem is **close the SHORT-horizon (0–25) sharpness+accuracy gap while keeping the
already-won long-horizon advantage** — and DLH is a **2× smaller** model, so matching CW-VAE is a *stronger* claim.

---

## §G. Goal & milestones (reframed by the red-team)

- **M0 — VALIDATE THE THESIS (≈free, from disk).** Establish DLH's long-horizon win **with a real CI + FVD**, not
  point estimates. Point estimates already pass; the gap at +100–150 is only ~0.01 fg-MSE on 64 clips with **no CI
  yet** `[RT]` — so this is a statistics task, not a compute task.
- **M1 — SHORT-HORIZON PARITY (the real target).** DLH ≥ CW-VAE on **LPIPS-fg + FVD** at +1…+25, via a
  **DLH-distinct renderer** (Phase C), **without regressing** the long-horizon win, at step-matched compute
  (DLH 2× smaller = a handicap, report `params_M`).
- **M2 — SHARP (stretch).** Approach GT sharpness (both models at 5–12%); pass CW-VAE decisively on **FVD/LPIPS**.
- **`det_carry` is a DIAGNOSTIC, not a milestone** `[RT]`. Concatenating the deterministic `d⁰` into the decoder
  **is literally CW-VAE's defining channel** — a "win" there means "I made DLH into CW-VAE," not "DLH beat
  CW-VAE." We run it once to *confirm* the §D cause, then build the genuine renderer in Phase C.

**Reality check:** CW-VAE is itself only ~12% of GT sharpness. M0/M1 = "as good as a blurry, 2× bigger model."
M2 = "actually sharp." Don't sell one as the other.

---

## §0. Gating discipline (fixed per red-team)

1. **Decision metrics = LPIPS-fg + FVD + fg-MSE.** **FVD is REQUIRED for any surpass claim** `[RT]` (temporal
   coherence is literally DLH's thesis; per-frame metrics miss it). **Laplacian-var is DIAGNOSTIC ONLY** `[RT]`
   (a transpose-conv/checkerboard artifact can inflate it) — never a headline, always cross-checked with LPIPS↓
   AND fg-MSE↓.
2. **Proper statistics** `[RT]`. Vary the **TRAINING seed (≥3)**, not only the rollout seed (training-seed noise
   is the dominant source `gate.py` currently ignores). Winner test = **paired bootstrap/t CI on the difference
   excluding 0**, not `std(a)+std(b)`. Fix `gate.py:67-69` to match this and to match the clock rule below.
3. **One selection metric across models** `[RT]`. DLH `eval.py` selects best-of-N by fg-MSE; CW-VAE `eval_torch.py`
   by SSIM — route **both** through one metric (assert in the gate). Report the **primary sharpness number at
   N=1** (no oracle); best-of-N is a secondary accuracy view only (selecting on fg-MSE then reporting lap-var
   biases sharpness).
4. **Clock invariant** `[RT]`. Adopt only if `skl_per_level[n≥1]` and `K_mean` stay within a **band derived from
   baseline training-seed variance** (1–2σ) — replace `gate.py`'s hardcoded top-level-only `top_kl ≥ 0.5·ctrl`.
   Applies to **B2 (`dec_stddev`) too** `[RT]`, not just `min_std`. A change that sharpens but breaks the clock is
   logged as a **DLH→CW-VAE regression**, not adopted.
5. **In-range horizons**, **smoke before launch**, **commit+tag** each change with its config + `params_M`.

---

## §B0. Auto-resume launcher (kept; hardened `[RT]`)
`scripts/run_until.sh` loops `--resume` until `step==total`; `ckpt_every=1000`; `num_workers=0`. **A 15–18 h run
crosses session boundaries the in-process retry can't recover** — launch it harness-tracked/background and verify
kill→resume actually survives a session teardown, not just a non-zero exit. Add a **50-step bs50/seq100 smoke**
before the first matched run (committed default is bs32; bs50 is untested end-to-end here `[RT]`).

---

## Phase A — surface what's on disk, then an honest baseline

**A0. Surface the existing long-horizon verdict (≈0 h compute) `[RT]` — do this FIRST.**
- Pull `compare_curves.json` into the scoreboard; add **fg-LPIPS** to those curves and a **per-training-seed CI**;
  state a **minimum-detectable-effect** at +100–150.
- *Decision gate:* is DLH's long-horizon win **real beyond the CI**? If a 0.01–0.03 fg-MSE gap is **inside** the
  MMNIST CI, **M-long is mis-specified — do not spend Phase B/C compute chasing it** `[RT]`; re-scope to
  short-horizon only. (Point estimates pass; this confirms or kills cheaply.)

**A1. Instrument eval (~2 h, no training).** Wire into `eval.py`'s report: **Laplacian-var/HF (diagnostic)**,
**FVD** (I3D, ≥256 clips), `params_M`, and a `hvp_dlh.compare_cwvae` harness (load `../cwvae` ckpt, **one shared
selection metric**, matched clips/ctx/horizon/seed). *Gate:* reproduces `match_eval.json`; black scores badly on
LPIPS/lap-var.

**A2. Matched-compute faithful baseline.** Train faithful DLH to **18k steps / bs50** (= CW-VAE budget) →
`baseline_matched.pt`, tag. **Single seed first** (add 2 more only when a claim needs the CI) `[RT]`. Report
`params_M` so every comparison is honestly **"DLH at half the params."** *Gate M0′:* the honest scoreboard.

---

## Phase B — confirm the diagnosis (CHEAP; not a milestone) `[RT]`

**B1. `dec_condition: 'stochastic'(default) | 'det_carry'`, cheap seq50 A/B.** Decoder reads `cat([c, d⁰])`.
- **Wiring** `[RT]`: build `ConvImageDecoder` with `feat_dim = det_dim + temporal_gru = 400` (currently hardcoded
  200, `modules.py:87`), selected by `cfg.dec_condition`. Flag-on is a **fresh-trained decoder** (baseline weights
  can't load); the smoke gate is "flag-off bit-matches baseline; flag-on trains, no NaN with all upper levels
  frozen."
- **Drop the `[c,d⁰,x⁰]` variant** `[RT]`: `x⁰ = img_enc(current frame)` is `None` in open-loop (would crash) and
  an autoencoder leak in training (fake sharpness that won't transfer). `d⁰` alone is past-only/causal — valid in
  both modes.
- **Purpose = CONFIRM §D**, not win. Expected: it recovers most of CW-VAE's sharpness → proves conditioning is the
  cause. Log in §F as a tautological deviation; **do not adopt as the M1 result.**

**B2/B3/B4 (optional tranche-2 knobs, only if a faithful-renderer sharpening is wanted):** `dec_stddev{1,0.5,0.3}`,
`min_std`-retrain `{0.1,0.03,0.01}` (inference-only ruled out → must retrain; **clock-risk, gate hard**),
`tanh→sigmoid`. Each one-change-at-a-time, §0 gates, clock invariant (incl. B2). Deferred behind Phase C.

---

## Phase C — the REAL sharpness path (DLH-distinct) `[RT]`

**C0. Oracle-headroom probe FIRST (~1 h).** On the **existing 8k latents**, fit/oracle the best possible decoder
output and measure ceiling: is there **>0.05 LPIPS / >2 dB** to gain? If not, **M2 is dead cheaply** — stop.

**C1. Perceptual/adversarial/diffusion decoder on the FROZEN STOCHASTIC code.** This is the genuine,
non-tautological renderer: keep DLH's stochastic-only code (the thing `det_carry` would have bypassed) and train
a sharp decoder (L1+LPIPS+hinge-PatchGAN, or a small diffusion head) on it; ELBO/clock untouched (decoder-only =
safe). **Lead LPIPS/FVD, never PSNR.** *Port first:* `discriminator.py`/`diffusion_head.py`/`quality.py` live on
`research/cwvae-backbone` — `git show … > …` and commit before scheduling. *Gate M1/M2:* LPIPS-fg + FVD beat
CW-VAE beyond CI at the target horizon, long-horizon fg-MSE not regressed, clock intact.

---

## Phase D — RETIRED `[RT]`
The long-horizon DLH-vs-CW-VAE-vs-SimVP head-to-head it proposed is **already in `compare_curves.json`** (§D/A0).
Don't re-run it; just add the CI (A0). The optional "engage the 3rd level via clock levers" survives as a parked
capacity experiment, not on the critical path.

---

## §F. Faithfulness ledger
**faithful-DLH** (paper repro, the reference proving we didn't cheat) and **improved-DLH** coexist via flags,
all default-off, reversible, each with its measured Δ and what it deviates from:
- `dec_condition=det_carry` — **diagnostic only**, imports CW-VAE's deterministic channel (explicitly *not* a
  faithful DLH; not adopted as a result).
- `dec_stddev<1.0` — faithful (paper tunes likelihood std per-dataset; we left it at 1.0).
- `min_std<0.1`, `dec_out_activation=sigmoid`, Phase-C sharp decoder — unspecified-by-paper choices, flagged.

## Skip / de-scope (unchanged): inter-level cross-attention (null on sharpness), 4/8/16 spatial pyramid (null
hierarchy), GRU→transformer (prize forfeited by the online clock).

---

## §S. Success, sequencing, honest compute, kill-criterion

**SUCCESS = M0 (long-horizon win established with CI+FVD) AND M1 (short-horizon LPIPS-fg/FVD parity via the
Phase-C renderer, long-horizon preserved, at 2× fewer params).** M2 (actually-sharp) is stretch.

**Value-of-information order `[RT]` (cheapest decisive first):**
`B0 → A0 (surface curves+CI, ~0 h) → A1 (instrument, ~2 h) → B1 (det_carry confirm, ~3 h) → C0 (headroom, ~1 h)
→ DECISION → A2 (matched baseline, ~15–18 h) → C1 (sharp decoder)`. **~6 h of work decides whether the expensive
runs are even needed** — do not put the 18 h run ahead of them.

**Honest compute `[RT]`:** measured **2.87 s/it** (bs50/seq100) → one matched 18k run ≈ **15–18 h**; the full plan
with ≥3 training seeds × Phases B+C is **7–21 GPU-days**, not "3–5 days." The descoped decisive subset
(A0+A1+B1+C0 then one A2) is **~20 h**.

**Kill-criterion / off-ramps `[RT]`:**
- *Stage 0 (A0):* if DLH's long-horizon win is inside the CI on the existing curves → the distinctive claim is
  unproven on MMNIST; pivot to a harder dataset or drop the "surpass" framing.
- *Stage 1 (C0):* if oracle headroom < 0.05 LPIPS / 2 dB → M2 (real sharpness) is unreachable on these latents;
  stop at M0/M1.
- *Strategic:* if the goal is purely a sharp+accurate MMNIST predictor, SimVP/gSTA already wins short-horizon —
  DLH is worth keeping **only** for the long-horizon/clock story (which §D shows it earns). Keep that framing honest.

## §R. Open risks
- B-knobs (`min_std`/`dec_stddev`) can break the clock — gate hard, revert fast.
- FVD infra (I3D) must be stood up for any surpass claim; budget it in A1.
- `det_carry` is seductive but tautological — keep it labelled diagnostic.
- "beat CW-VAE" is parity with a blurry 2× model; only C1 is real sharpness.
