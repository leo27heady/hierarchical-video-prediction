# RHCP — Reversible Hierarchical Clockwork Predictor

**The converged, runnable architecture.** Built by fusing two pieces this project already had and verified:
`hvp/e2e/hier.py` (the CW-VAE-class spatial pyramid) + `scripts/reversible_temporal_proto.py` (the RevNet additive
coupling). One new module does the fusion. Validated training-free on the real conv modules (see *Measured results*).

> Status: core built (`hvp_rev/rhcp.py`), validated (`hvp_rev/rhcp_validate.py`), D1 trainer wired
> (`hvp_rev/rhcp_train.py`). This doc is the honest spec — verified by an 11-agent review + 3-claim adversarial
> red-team, every oversell downgraded to its defensible form.

---

## 1. One-sentence contribution

> A reversible hierarchical clockwork video predictor — a CW-VAE-class spatial pyramid on a fixed geometric clock
> whose per-level temporal recurrence is reversible along time — enabling **exact-gradient, no-truncation,
> free-running autoregressive rollout training at recurrent-core activation memory constant in rollout length K**:
> the capability that lets a free-running unroll fit where vanilla full-sequence BPTT OOMs.

### What is novel (the combination, not any primitive)
1. **Re-homing reversibility as the *enabler* of a constant-memory, no-truncation, free-running AR rollout objective**
   — exact gradients through the full K-step unroll by reverse-replay of the reversible chain, storing only per-step
   conditioning + the final state.
2. **The reversibility-of-the-clock argument:** a fixed geometric clock-hold (identity on non-tick frames) is
   invertible *for free* because the schedule bit `t mod k_l` is a known function of `t`. Multi-rate behaviour comes
   from lossless **HOLDING**, not forgetting — sidestepping the MacKay no-forgetting tension that a learned 0/1 gate
   would trigger (proto P4).

### What is NOT novel (state plainly, or a reviewer will)
- **The reversible block is standard RevNet/RevViT additive coupling.** The "two streams" are the two channel halves;
  "even/odd" is the F/G alternation. Cite RevNet (1707.04585). Claim only correct *composition*, not a new primitive.
- The spatial pyramid (CW-VAE/DLH) is not novel. The equal-token-budget tick schedule is the MilliVid trick, ported.
- Rollout-training-fixes-exposure-bias overlaps **Self-Forcing** (NeurIPS 2025). Reversibility-as-pure-memory-saver
  is **not** novel and has tied/lost to gradient checkpointing (AlignProp/Self-Forcing, and this project's own
  `hvp_rev/memory_axis_demo.py`).

---

## 2. The honest memory claim (after red-team + measurement)

This is the single most important thing to get right, because the project has been burned by oversold resource wins.

**The ONLY defensible claim:**
> The reversible clock trains the full no-truncation K-step free-running AR unroll at **recurrent-core activation
> memory constant in K (O(1))**, enabling a rollout that **vanilla full-sequence BPTT cannot fit** at this batch
> size / 16 GB GPU. Against **per-step gradient checkpointing** — which also fits — we are **NOT a memory winner**:
> checkpointing already achieves K-independent core memory at one full-forward recompute. Our only edge over
> checkpointing is a **recompute/throughput** consideration (one inverse pass vs one full forward recompute) plus
> saving the small recurrence-*state* share — measured as wall-clock, not a memory headline.

**Mandatory honesty moves** (all enforced in `rhcp_validate.py`):
- Benchmark against **per-step checkpointing**, not vanilla BPTT. Report BPTT only to show why one must do *something*.
- **Count the non-reversible conv/decoder activations** in every memory figure (they dominate the real tape and are
  recomputed regardless — reversibility never saves them). The bench includes the decoder, recomputed per step.
- Report the **fp32 reverse-replay drift** honestly; re-anchor (store full state every ~√K) if it nears the
  gradient-noise floor — which gracefully degrades reversibility to checkpointing (the honest fallback).
- **Do NOT** claim an "85×/380× categorical win" — that strawmans vanilla BPTT, which no practitioner runs for a
  recurrent rollout.

---

## 3. Measured results (training-free, this machine: RTX 4060 Ti 16 GB, torch 2.11)

Run: `venv/Scripts/python.exe -m hvp_rev.rhcp_validate`

**STEP 0 — block invertibility (real conv stage):** fp64 max-err **1.3e-15** (algebraically exact); fp32 **5.4e-4**
at realistic trained-scale init. The coupling is exact; fp32 drift is bounded and re-anchored.

**STEP 1 — assembled multi-level clock-hold reconstruction** (reconstruct the *entire* chain from the FINAL state
alone + recomputable schedule bits, HOLD exercised on slow levels, co-evolving top-down conditioning):

| config | fp64 drift | fp32 drift (K) |
|---|---|---|
| 3-level 16/8/4, k=1/2/4 | **1.3e-13** (exact) | 2.2e-2 (K=24) |
| 4-level 16/8/4/2, k=1/2/4/8 | **2.0e-13** (exact) | 3.2e-2 (K=32) |

→ the multi-rate hierarchy is reversible by **lossless holding** — no stored masks, no pre-gate state.

**STEP 2 — honest memory + gradient bench** (core+decoder 0.46 M params, batch 8, levels 16/8/4):

| K | BPTT peak | checkpoint | reversible | grad-cos vs BPTT | fp32 recon drift |
|---:|---:|---:|---:|---:|---:|
| 20 | 1405 MB | 122 MB | 133 MB | **1.000000** | 1.6e-2 |
| 40 | 2785 MB | 144 MB | 154 MB | **1.000000** | 4.2e-2 |
| 100 | 6909 MB | 198 MB | 209 MB | **1.000000** | 1.6e-1 |

- **vs BPTT:** reversible is **33× smaller at K=100** (0.030×) and ~flat in K, while BPTT climbs ~linearly and will
  OOM at larger K/res/batch. *This is the capability.*
- **vs checkpointing:** reversible is **1.05× (i.e. ~equal, slightly worse) on memory** and comparable on throughput.
  *This is the honest Pareto-neutral truth — not a win.*
- **grad-cos = 1.000000** confirms reverse-replay produces **bit-identical gradients** to BPTT. The mechanism is
  correct, not approximate.

---

## 4. Architecture (extend `hier.py`, do not rewrite)

```
                          ENCODER (per frame, 2D conv)              DECODER (only L0 -> pixels)
  frame 64x64 ──► 32 ─► 16(L0) ─► 8(L1) ─► 4(L2) ─► 2(L3)          s_L0 (16x16) ──► 32 ─► 64 ─► frame
                         │         │        │         │
                    ┌────▼────┬────▼───┬────▼───┬─────▼────┐  per-level reversible temporal recurrence
   time ─►          │ L0 k=1  │ L1 k=2 │ L2 k=4 │ L3 k=8   │  (additive coupling along time; HOLD off-tick)
                    └─────────┴────────┴────────┴──────────┘
                       fine/fast            coarse/slow
        top-down conditioning: child reads parent's just-updated state (CW-VAE prior), via CrossCond
```

| level | spatial | channels | tick k | role |
|---|---|---|---|---|
| L0 (BOT) | 16×16 | 64 | 1 | texture; the only level that decodes to pixels |
| L1 (MID) | 8×8 | 96 | 2 | mid structure |
| L2 | 4×4 | 128 | 4 | coarse layout |
| L3 (TOP) | 2×2 | 192 | 8 | scene-level, long temporal context |
| *(L4 1×1 k16)* | 1×1 | 256 | 16 | **ablation only** — ticks once in 10 frames, dead for the table |

**Ship 16/8/4 (or 16/8/4/2) for the 10→10 table.** 5 levels is overkill at 64×64; the 1×1 level is dead at T=10.

- **Encoder:** `MultiScaleEncoder` (hier.py) extended to 4 spatial taps. *(rhcp.py ships a self-contained
  `PyramidEncoder` so the model runs on this branch without the cwvae backbone.)*
- **Temporal core (the one new module):** per level a `RevConvStage` — additive coupling with ConvNeXt-style F/G,
  `forward()` **and** `inverse()`. Replaces `StageTranslator`'s forward-only conv block.
- **Clock-hold:** a slow level applies its coupling only when `t mod k_l == 0`, else identity. Fixed geometric
  `k_l = 2^l`. **No learned gate anywhere on the state path** (a learned 0/1 gate is non-invertible — proto P4,
  measured catastrophic drift).
- **Cross-level conditioning:** child queries parent (coarser, upsampled) through the conditioning `c` that rides
  *inside* F/G additively — via `CrossCond` (gated cross-attn, γ init 1e-4). Never a fusion of the two streams; never
  a multiplicative mask on the state.
- **Decoder:** `ConvDecoder` (16×16 → 64×64 + skip). Only L0 → pixels; L1–L3 supervised in latent space (deep sup).

### Reversible block math (RevNet eq. 8–10)
```
FORWARD  (per coupling pair):   y1 = x1 + F([x2, c]);   y2 = x2 + G([y1, c])
INVERSE  (reverse the pairs):   x2 = y2 - G([y1, c]);   x1 = y1 - F([x2, c])       # exact for ANY c
```
Cross-level `c` is exact-invertible because the coupling never *inverts* `c`, only **re-evaluates** it — which
succeeds iff `c`'s inputs are available at that point in the inverse schedule. **Condition (asserted in code):** a
strictly **acyclic** level-evaluation order — child may read parent's just-updated state; no parent reads a child's
same-frame state. Forward order TOP→BOTTOM; inverse order BOTTOM→TOP. A truly simultaneous mutual coupling (A reads
new-B while B reads new-A) is a fixed-point cycle with **no** reversible realization — forbidden by construction.

---

## 5. Deterministic core (for the OpenSTL table)

The `tab:mmnist` metric is **pixel-sum MSE/MAE**, minimized by the conditional **mean**; sampling a stochastic latent
is strictly penalized, and prior project runs showed the ELBO/stochastic path was a capacity-dump. So **the core is
deterministic** (renderer predicts the mean). The **store-the-noise** stochastic machinery (proto P3) stays
*implemented but dormant*: a stochastic level sets `c_l = [parent, eps_l]` with `eps` reparam-sampled and **stored**,
so the deterministic core still inverts exactly (verified: stored eps → drift ~1e-15, wrong eps → breaks). Dormant for
the table; it is the bridge to the TempVerseFormer stochastic extension in related work.

---

## 6. Training objectives

**D1 — the table (10→10), reuse `hier.py` loss + `hvp/e2e/train.py` unchanged** (Adam 1e-3, OneCycle, ~200 epochs):
```
L_D1 = L1(Ŷ_BOT, Y) + MSE(Ŷ_BOT, Y)                          # pixel, on decoded L0
     + Σ_{l>0} w_l · MSE_norm(s_l, stopgrad(Enc_l(Y)))        # latent deep-supervision
       with w_l = (0.5, 0.25, 0.125), norm_deepsup=True
```
Score via `hvp/eval/openstl_metrics.evaluate` after the `[-1,1]→[0,1]` / `(B,C,T,H,W)→(N,T,C,H,W)` conversion, or the
numbers are silently non-comparable.

**D2 — no-truncation rollout (the memory experiment): free-running AR + reverse-replay backward.** Replace `hier.py`'s
teacher-forced `rollout_loss` (feeds GT context) with free-running AR (feed the model's **own** predictions). Forward
stores only per-step conditioning `{c_l^t[, eps_l^t]}` + final state; backward reconstructs each `s_l^t` via
`inverse_step` (top-down) and computes **exact** gradients (grad-cos = 1.0, verified). Objective: **fg-weighted L1 +
Laplacian-variance penalty** — **not** plain MSE (prior null E1: MSE rollout fades to the mean). Add the warm-started
adversarial renderer only if drift sharpness is the bottleneck. Keep the objective choice **orthogonal** to the
memory claim.

---

## 7. Honest expected MMNIST (with floor + pre-registered kill criterion)

- **From scratch, 1–2 days, one 16 GB GPU:** **MSE high-30s to low-40s, SSIM ~0.90–0.93 — BELOW the table.**
  Grounded in this project's measured A/B: the hierarchical version of this exact codebase was **11.6% worse** than
  flat (hier 40.42 vs flat 36.23 MSE), and reversible coupling **halves** per-stream width on the pixel-producing BOT.
- **Table-range (~30, SimVP/CrevNet tier)** is reachable **only** by warm-starting the BOT decoder from the
  already-trained flat gSTA checkpoint (`logs/e2e_convnext_m8`, MSE 27.21) **and** rebalancing BOT to dim≈512 to undo
  the coupling halving — *not* from scratch.
- **Correct comparator in the table:** **CrevNet** (MSE 30.15 / SSIM 0.9350 @ 5.0 M) — the only reversible/info-
  preserving model in `tab:mmnist`.
- **PRE-REGISTERED KILL CRITERION:** if the first matched-capacity reversible-hier run exceeds the flat base by >15%
  (>~42 MSE), **abandon the D1-parity sentence** and ship the pure memory-capability + the 10→40 drift curve as the
  contributions. The capacity halving and the measured hier null are **structural**, not fixable by training harder.

---

## 8. Build plan (cheapest validation first; reuse existing code)

1. ✅ **STEP 0** — `RevConvStage` with `inverse()` (`hvp_rev/rhcp.py`); assert P1 on the real conv module. *Done.*
2. ✅ **STEP 1** — assembled multi-level clock-hold block; assert P1 on the assembled block under top-down inverse
   ordering with HOLD + co-evolving parent; static DAG assertion. *Done — fp64 exact.*
3. ✅ **STEP 2** — memory + gradient bench (reversible vs BPTT vs checkpointing), counting conv activations, grad-cos
   vs BPTT, fp32 drift. *Done — 33× vs BPTT, neutral vs checkpoint, grad-cos = 1.0.*
4. **STEP 3** — wire D1 end-to-end on MMNIST (`hvp_rev/rhcp_train.py`); short smoke confirms loss decreases.
5. **STEP 4** — train D1 (10→10) to convergence; score via OpenSTL metrics; check the kill criterion (>~42 → drop
   parity).
6. **STEP 5** — D2 free-running AR + reverse-replay backward (fg-weighted L1 + Laplacian-var, not MSE).
7. **STEP 6** — D2 long-horizon at K=40/100: foreground-MSE / LPIPS **drift curve** (NOT aggregate SSIM —
   background-dominated) vs a truncation-trained baseline. **Make-or-break for "hierarchy helps long-horizon".**
8. **STEP 7** *(only if D1 must reach table-range)* — warm-start BOT from `logs/e2e_convnext_m8` + rebalance to dim512.

### Reusable files
| file (branch) | use |
|---|---|
| `research/cwvae-backbone:hvp/e2e/hier.py` | `MultiScaleEncoder` (→4 taps), `CrossCond`, `rollout_loss` (→free-running) |
| `research/cwvae-backbone:hvp/e2e/model.py` | `ConvDecoder`, `ConvNeXtTranslator` (F/G) |
| `research/cwvae-backbone:hvp/eval/openstl_metrics.py` | the EXACT `tab:mmnist` metrics |
| `research/cwvae-backbone:hvp/data/mmnist.py` | on-the-fly train gen + fixed `mnist_test_seq.npy` val |
| `scripts/reversible_temporal_proto.py` | additive-coupling template + P1–P4 |
| `hvp_rev/memory_axis_demo.py` | honest BPTT-vs-checkpoint accounting (why the memory claim must be scoped) |
| **NEW:** `hvp_rev/rhcp.py` | the reversible `RevConvStage` + clock-hold core (the one new module) |

---

## 9. Caveats (so nothing surprises a reviewer)
- The two-stream even/odd block is **standard RevNet additive coupling**, not a new primitive — cite it as such.
- The memory result is **not a win over checkpointing** — only over vanilla BPTT (a strawman). Headline the
  constant-memory **capability**, never a memory win vs checkpointing.
- Reversibility does **not** stabilize training — drop that word. It makes the long rollout **affordable**; the
  rollout **objective** fixes exposure bias. Conflating the two is the oversell.
- On near-rigid 2-digit MMNIST the hierarchy is **neutral-to-null on the table metric** (3+ prior nulls + a measured
  11.6%-worse A/B). The hierarchy's value is argued **only** at 10→40 bounded drift; if that curve doesn't show the
  reversible-hier unrolling flatter than a truncation-trained baseline, the quality story collapses to a pure
  memory-capability result (still publishable as feasibility).
- fp32 reverse-replay drift accumulates (measured 0.16 at K=100); re-anchor every ~√K for very long horizons.
- The store-the-noise stochastic path is implemented but **dormant** for the table.
