# Generated reach–fidelity tables

> Status: synchronized through terminal E7c on 2026-07-16. E2/E4/E5/E7c values below are bound
> to their immutable result artifacts.

## Configuration and interface

| Arm | Package | Reach | Context tok | Target tok | Total tok | Target interface | Status |
|---|---|---|---|---|---|---|---|
| hier-min | spatial coarse + fine tail | 48 | 896 | 320 | 1216 | 4 coarse + 4 fine | measured |
| flat-min | single-stream full resolution | 14 | 896 | 320 | 1216 | 4 coarse + 4 fine | measured |
| hier-s2-min | stride-2 coarse + fine tail | 48 | 512 | 288 | 800 | 2 coarse + 4 fine | measured-confounded |
| flat-short | single-stream full resolution | 8 | 512 | 320 | 832 | 4 coarse + 4 fine | measured |
| hier-s2-fixed-target | stride-2 coarse + fine tail | 48 | 512 | 320 | 832 | 4 coarse + 4 fine | prepared-not-authorized |
| flat-short-fixed-target | single-stream full resolution | 8 | 512 | 320 | 832 | 4 coarse + 4 fine | prepared-not-authorized |
| hier-c1 / flat-c1 | spatial versus flat | 48 / 28 | 1792 | 320 | 2112 | matched | largest tested B16x30k budget favouring spatial |
| hier-c2 / flat-c2 | spatial versus flat | 48 / 38 | 2432 | 320 | 2752 | matched | flat favoured at seed 0 under B16x30k and B8x60k |

## Claim and evidence snapshot

| Claim | Contrast | Seed | Effect | 95% interval | Analysis | Status |
|---|---|---|---|---|---|---|
| R1 | hier-min vs flat-min | 0 | 0.0474 | [0.0354, 0.0595] | E2 all registered records | confirmed |
| R1 | hier-min vs flat-min | 1 | 0.0436 | [0.0326, 0.0549] | E2 all registered records | confirmed |
| R2 | hier-min vs flat-min | 0 | 0.0980 | [0.0788, 0.1179] | E2 registered 14 < dt <= 48 | confirmed |
| R2 | hier-min vs flat-min | 1 | 0.0914 | [0.0736, 0.1098] | E2 registered 14 < dt <= 48 | confirmed |
| F2 | hier-s2-min vs flat-short (diffusion) | 0 | 0.0340 | E2 source artifact | configuration effect | confirmed |
| F2 | hier-s2-min vs flat-short (diffusion) | 1 | 0.0381 | E2 source artifact | configuration effect | confirmed |
| F2 | hier-s2-min-reg vs flat-short-reg | 0 | 0.0639 | [0.0516, 0.0763] | configuration effect | measured |
| F3 | temporal vs spatial (diffusion) | 0 | 0.0197 | [0.0143, 0.0252] | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 1 | 0.0046 | [0.0007, 0.0084] | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 2 | 0.0111 | [0.0070, 0.0151] | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 3 | 0.0075 | E4b source artifact | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 4 | 0.0060 | E4b source artifact | package effect | measured |
| F3 | temporal vs spatial (regression) | 0 | 0.0107 | [0.0062, 0.0158] | package effect | measured-sign-sensitive |
| F3 | temporal vs spatial (regression) | 1 | -0.0104 | [-0.0146, -0.0062] | package effect | measured-sign-sensitive |
| F3 | temporal vs spatial (regression) | 2 | -0.0208 | E4 source artifact | package effect | measured-sign-sensitive |
| F3 | temporal vs spatial (regression) | 3 | 0.0248 | E4b source artifact | package effect | measured-sign-sensitive |
| F3 | temporal vs spatial (regression) | 4 | -0.0166 | E4b source artifact | package effect | measured-sign-sensitive |
| F1 | hier-c2 vs flat-c2 (B16x30k) | 0 | -0.0116 | [-0.0162, -0.0073] | optimizer-regime-specific package effect | measured-regime-qualified |
| F1 | hier-c2 vs flat-c2 (B8x60k) | 0 | -0.0077 | [-0.0123, -0.0030] | matched-example changed-batch/update/noise/EMA control | measured-regime-qualified |
| T1 early | d1-diff vs d0-diff-long | 0/1/2 | +0.0204 / +0.0082 / +0.0230 | see E5a artifact | h1–24 | three-seed positive |
| T1 late | d1-diff vs d0-diff-long | 0/1/2 | −0.0064 / +0.0074 / +0.0087 | see E5a artifact | h61–256 | late inversion not replicated |
| F4 | four frozen diffusion contrasts | predictor seed 0 | 20/20 expected signs | 18/20 simultaneous lower bounds positive | E7c NFE × sampler robustness | directionally robust; upper inversion unresolved at sampler seeds 1/2 |

## E7c sampler/NFE robustness family

Positive effects favor the registered candidate. Intervals are simultaneous 95% intervals from
the frozen 20-effect studentized max-statistic clip-cluster bootstrap. Total NFE per 152-frame
rollout is 38 times the per-step NFE.

| Contrast | NFE/step | Sampler seed | Total NFE/clip | Effect | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|
| 896 reach | 8 | 0 | 304 | +0.0478 | [+0.0323,+0.0634] |
| 896 reach | 16 | 0 | 608 | +0.0465 | [+0.0310,+0.0621] |
| 896 reach | 32 | 0 | 1216 | +0.0471 | [+0.0317,+0.0625] |
| 896 reach | 16 | 1 | 608 | +0.0482 | [+0.0325,+0.0638] |
| 896 reach | 16 | 2 | 608 | +0.0466 | [+0.0314,+0.0618] |
| 1,152 interior | 8 | 0 | 304 | +0.0152 | [+0.0045,+0.0259] |
| 1,152 interior | 16 | 0 | 608 | +0.0146 | [+0.0041,+0.0251] |
| 1,152 interior | 32 | 0 | 1216 | +0.0140 | [+0.0036,+0.0245] |
| 1,152 interior | 16 | 1 | 608 | +0.0201 | [+0.0086,+0.0315] |
| 1,152 interior | 16 | 2 | 608 | +0.0200 | [+0.0087,+0.0313] |
| 1,152 package shape | 8 | 0 | 304 | +0.0276 | [+0.0171,+0.0381] |
| 1,152 package shape | 16 | 0 | 608 | +0.0283 | [+0.0172,+0.0394] |
| 1,152 package shape | 32 | 0 | 1216 | +0.0292 | [+0.0182,+0.0403] |
| 1,152 package shape | 16 | 1 | 608 | +0.0251 | [+0.0160,+0.0342] |
| 1,152 package shape | 16 | 2 | 608 | +0.0221 | [+0.0133,+0.0309] |
| 2,432 inversion | 8 | 0 | 304 | +0.0115 | [+0.0016,+0.0214] |
| 2,432 inversion | 16 | 0 | 608 | +0.0117 | [+0.0018,+0.0217] |
| 2,432 inversion | 32 | 0 | 1216 | +0.0122 | [+0.0022,+0.0221] |
| 2,432 inversion | 16 | 1 | 608 | +0.0098 | [−0.0001,+0.0196] |
| 2,432 inversion | 16 | 2 | 608 | +0.0077 | [−0.0013,+0.0167] |

Registered category: **DIRECTIONALLY_ROBUST**. The result is a checkpoint-only inference-robustness
test on seed-0 predictors and one codec; E7a separately tests codec/latent-geometry robustness.

## Prior-art positioning

| Work | Memory | Matched budget | Observability | Dynamic re-entry | Self-fed | Role here |
|---|---|---|---|---|---|---|
| WorldPack / LoopNav | compressed trajectory packing + retrieval | no | no | primarily spatial consistency | yes | strong retrieval/compression comparator |
| CaR / SceneFly | context compression + viewpoint-conditioned implicit retrieval | no | annotated synthetic geometry | camera trajectories | yes | primary recent positioning |
| MBench | benchmark: entity/environment/causal consistency | n/a | rule + VLM metrics | broad long video | model dependent | memory-focused benchmark |
| MemoBench | disappear/reappear under dynamic hidden evolution | n/a | diagnostic curated tasks | yes | model dependent | closest directional benchmark |
| Self Forcing | rollout-aware autoregressive training | no | no | not primary | trained explicitly | trust-boundary intervention |
| This study | controlled context packages | yes, context tokens where stated | exact geometric labels | static hidden state now; dynamic future | yes + teacher-forced boundary | mechanism and frontier instrument |
