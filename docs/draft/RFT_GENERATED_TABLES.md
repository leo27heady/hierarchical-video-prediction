# Generated reach–fidelity–trust tables

> Status: verified post-c2 evidence; untouched confirmation pending. Generated from `experiments/evidence_snapshot_2026-07-11.json`; pending values remain explicit.

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
| R1 | hier-min vs flat-min | 0 | 0.0399 | development headline interval in source artifact | all registered records | measured-development |
| R1 | hier-min vs flat-min | 1 | 0.0358 | development headline interval in source artifact | all registered records | measured-development |
| R2 | hier-min vs flat-min | 0 | 0.088 | [0.0668, 0.1083] | 14 < dt <= 48 | post-hoc |
| R2 | hier-min vs flat-min | 1 | 0.0809 | [0.0621, 0.0998] | 14 < dt <= 48 | post-hoc |
| F2 | hier-s2-min vs flat-short (diffusion) | 0 | 0.033 | [0.0246, 0.0419] | configuration effect | measured |
| F2 | hier-s2-min vs flat-short (diffusion) | 1 | 0.0338 | [0.0251, 0.0430] | configuration effect | measured |
| F2 | hier-s2-min-reg vs flat-short-reg | 0 | 0.0639 | [0.0516, 0.0763] | configuration effect | measured |
| F3 | temporal vs spatial (diffusion) | 0 | 0.0197 | [0.0143, 0.0252] | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 1 | 0.0046 | [0.0007, 0.0084] | package effect | measured |
| F3 | temporal vs spatial (diffusion) | 2 | 0.0111 | [0.0070, 0.0151] | package effect | measured |
| F3 | temporal vs spatial (regression) | 0 | 0.0107 | [0.0062, 0.0158] | package effect | measured-sign-sensitive |
| F3 | temporal vs spatial (regression) | 1 | -0.0104 | [-0.0146, -0.0062] | package effect | measured-sign-sensitive |
| F1 | hier-c2 vs flat-c2 (B16x30k) | 0 | -0.0116 | [-0.0162, -0.0073] | optimizer-regime-specific package effect | measured-regime-qualified |
| F1 | hier-c2 vs flat-c2 (B8x60k) | 0 | -0.0077 | [-0.0123, -0.0030] | matched-example changed-batch/update/noise/EMA control | measured-regime-qualified |

## Prior-art positioning

| Work | Memory | Matched budget | Observability | Dynamic re-entry | Self-fed | Role here |
|---|---|---|---|---|---|---|
| WorldPack / LoopNav | compressed trajectory packing + retrieval | no | no | primarily spatial consistency | yes | strong retrieval/compression comparator |
| CaR / SceneFly | context compression + viewpoint-conditioned implicit retrieval | no | annotated synthetic geometry | camera trajectories | yes | primary recent positioning |
| MBench | benchmark: entity/environment/causal consistency | n/a | rule + VLM metrics | broad long video | model dependent | memory-focused benchmark |
| MemoBench | disappear/reappear under dynamic hidden evolution | n/a | diagnostic curated tasks | yes | model dependent | closest directional benchmark |
| Self Forcing | rollout-aware autoregressive training | no | no | not primary | trained explicitly | trust-boundary intervention |
| This study | controlled context packages | yes, context tokens where stated | exact geometric labels | static hidden state now; dynamic future | yes + teacher-forced boundary | mechanism and frontier instrument |
