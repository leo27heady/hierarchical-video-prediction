# BLT decoder-gating + memory-axis experiments (branch `explore/blt-decoder-gating`)

Two decisive, training-free experiments testing whether the "adaptive resource allocation at quality parity" thesis
survives on the actual DLH architecture / a 16 GB RTX 4060 Ti — after the multi-agent red-team re-targeted the idea
(gate the EXPENSIVE high-res decoder, not the cheap latents) and the memory axis (persist = zero reverse-tape bytes).

## Experiment (a) — tile-sparse decoder wall-clock (`hvp_blt/tile_decoder_bench.py`)

The FLOP-dominant block is the decoder upsample (16→32→64 = 80% of the pipeline). Gate it tile-sparsely: only
re-render dirty 8×8-px tiles (gather active tiles → packed conv → scatter), copy the rest. Geometric kill-shot first
said p_tile is achievable (17–26% at small/medium tiles, 78% of tiles clean ≥4 frames). But the **realized wall-clock**:

| p_tile | sparse vs dense | (FLOP-ideal) |
|---|---|---|
| 10% | **0.88× (slower)** | 10× |
| 18% | 0.79× | 5.6× |
| 26% | 0.73× | 3.8× |
| 50% | 0.36× | 2× |
| 100% | 0.17× | 1× |

**Verdict: DEAD.** Even at the favorable 10–26% fill, the tile-sparse path is *slower* than dense. The dense decoder
is only **1.05 ms** (launch-bound at 64×64 on a 4060 Ti), so the unfold + gather + scatter + halo overhead exceeds the
conv it avoids. K0 a third time, now with the correct (decoder) orientation. **The compute-saving thesis does not
survive on this hardware/scale**, regardless of how low p_tile is.

## Experiment (b) — memory axis (`hvp_rev/memory_axis_demo.py`)

Measured peak GPU memory of the real SpatialDLH vs rollout length T (B=32), and the max trainable window on 14 GB:

| method | MB/step | max window T | note |
|---|---|---|---|
| BPTT (store everything) | 185 | 75 | |
| **gradient checkpointing** | 48 | **290** | the real, measured **4× win** — standard, correct grads |
| reversible-gated + ckpt, tick 0.25 | 46.6 | 300 | **+4% only** |

**Verdict: the real memory win is gradient CHECKPOINTING (4×, standard).** Reversibility + the gate's "persist = zero
reverse-tape" are REAL but **marginal (~4%) on this architecture**, because the latent recurrence is only ~2.1 of ~48
MB/step — the **per-frame convs dominate the tape**, and checkpointing already handles them. Reversibility/gating only
becomes a major memory lever when the RECURRENCE dominates memory (a deep/wide temporal core) — which fights DLH's
anti-drift small-latent choice (stoch_dim=8).

## Combined verdict

The "resource redistribution at quality parity" thesis is **empirically weak on the current DLH on BOTH axes**:
compute is dead (decoder too cheap, sparsity overhead), and memory is dominated by checkpointing (standard), with
reversibility/gating a small add-on. A resource axis only becomes a strong contribution if the gated/reversible
component *dominates* the resource — which the small-latent anti-drift design precludes.

**Where the defensible contribution actually is (not resource savings):**
1. **Emergent unsupervised motion segmenter** — the per-pixel gate localizes motion (seg-corr 0.57), for free from the
   generative objective. Novel, visual, no kernel required.
2. **A rigorous characterization of the update-or-persist primitive** — *when* a content-adaptive clock helps a
   hierarchical stochastic video model vs capacity-dumps (the unified null across temporal / σ-sweep / spatial; the
   field assumes the clock helps and never measured it).
3. **Reversibility as correctness-theory** extending TempVerseFormer into the stochastic-gated regime (the
   no-forgetting figure + a checkpointing-baselined memory curve), framed honestly as a constant-factor / specific-
   regime result, not a blanket constant-memory claim.

Resource savings are reported as honest secondary/negative results, never the headline.
