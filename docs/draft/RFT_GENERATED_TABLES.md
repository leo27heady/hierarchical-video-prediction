# Generated reach–fidelity evidence tables

> Generated only from frozen result JSONs. Effects use the convention printed in each section; clip-cluster intervals condition on the trained model, while t intervals use training seeds as units.

## Configuration and exact interface accounting

| Arm | Context package | Reach | Context tokens | Target tokens | Total tokens | Qualification |
|---|---|---:|---:|---:|---:|---|
| hier-min | spatial: dense coarse + 2 recent fine | 48 | 896 | 320 | 1,216 | B16×30k |
| flat-min | flat contiguous fine | 14 | 896 | 320 | 1,216 | B16×30k |
| hier-s2-min | spatiotemporal: stride-2 coarse + 2 fine | 48 | 512 | 288 | 800 | configuration effect; target interface differs |
| flat-short | flat contiguous fine | 8 | 512 | 320 | 832 | B16×30k |
| hier-long | spatial: dense coarse + 6 recent fine | 48 | 1,152 | 320 | 1,472 | B16×30k |
| flat-budget | flat contiguous fine | 18 | 1,152 | 320 | 1,472 | B16×30k |
| strided-flat | temporal: 11 stride-4 fine anchors + 6 fine | 48 | 1,088 | 320 | 1,408 | near-matched to hier-long (1,088 vs 1,152) |
| hier-c1 | spatial: dense coarse + 16 fine | 48 | 1,792 | 320 | 2,112 | frontier member |
| flat-c1 | flat contiguous fine | 28 | 1,792 | 320 | 2,112 | frontier member |
| hier-c2 | spatial: dense coarse + 26 fine | 48 | 2,432 | 320 | 2,752 | frontier + regime control |
| flat-c2 | flat contiguous fine | 38 | 2,432 | 320 | 2,752 | frontier + regime control |
| hier-rich | spatial: dense coarse + 36 fine | 48 | 3,072 | 320 | 3,392 | B8×60k |
| flat-long | flat contiguous fine | 48 | 3,072 | 320 | 3,392 | B8×60k |

The temporal `strided-flat` package has 1,088 context tokens, not 1,152. Its 64-token deficit relative to `hier-long` was preregistered as favoring the temporal control; the comparison is near-matched rather than exact.

## Untouched confirmation: mechanistic bands

| Pair | Seed | Registered population | Records | Clips | Effect | Paired 95% CI |
|---|---:|---|---:|---:|---:|---:|
| reach-48-vs-14 | 0 | 14 < dt <= 48; seen >= 0.5; GT foreground non-empty | 1006 | 96 | +0.0980 | [+0.0788, +0.1179] |
| reach-48-vs-14 | 1 | 14 < dt <= 48; seen >= 0.5; GT foreground non-empty | 1006 | 96 | +0.0914 | [+0.0736, +0.1098] |
| b1-config-48-vs-8 | 0 | 8 < dt <= 48; seen >= 0.5; GT foreground non-empty | 1452 | 99 | +0.0692 | [+0.0543, +0.0845] |
| b1-config-48-vs-8 | 1 | 8 < dt <= 48; seen >= 0.5; GT foreground non-empty | 1452 | 99 | +0.0743 | [+0.0588, +0.0899] |

Positive favors the longer-reach registered candidate. These pair-specific bands were frozen before the E2 holdout was opened.

## Untouched confirmation: tested frontier family

| Frozen contrast | Effect | Simultaneous 95% CI |
|---|---:|---:|
| b1152-s0 | +0.0195 | [+0.0070, +0.0320] |
| b1152-s1 | +0.0389 | [+0.0228, +0.0549] |
| b1152-s2 | +0.0307 | [+0.0159, +0.0455] |
| b1792-s0 | +0.0084 | [-0.0013, +0.0181] |
| b2432-30k-s0 | -0.0150 | [-0.0260, -0.0039] |
| b2432-60k-s0 | -0.0122 | [-0.0213, -0.0032] |
| b3072-s0 | -0.0129 | [-0.0217, -0.0041] |
| b3072-s1 | -0.0134 | [-0.0215, -0.0053] |
| b896-s0 | +0.0474 | [+0.0305, +0.0642] |
| b896-s1 | +0.0436 | [+0.0280, +0.0592] |
| near-bottom-s0 | -0.0038 | [-0.0081, +0.0005] |
| near-bottom-s1 | -0.0060 | [-0.0105, -0.0015] |

Studentized max-statistic clip-cluster family; critical value 2.741. No interpolation is performed between budgets.

## Head × package by training seed

| Seed | Diffusion | Clip 95% CI | Regression | Clip 95% CI | Interaction | Clip 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.0197 | [+0.0143, +0.0252] | +0.0107 | [+0.0062, +0.0158] | -0.0089 | [-0.0151, -0.0025] |
| 1 | +0.0046 | [+0.0007, +0.0084] | -0.0104 | [-0.0146, -0.0062] | -0.0150 | [-0.0199, -0.0104] |
| 2 | +0.0111 | [+0.0070, +0.0151] | -0.0208 | [-0.0270, -0.0150] | -0.0319 | [-0.0401, -0.0245] |
| 3 | +0.0075 | [+0.0037, +0.0114] | +0.0248 | [+0.0181, +0.0323] | +0.0173 | [+0.0106, +0.0245] |
| 4 | +0.0060 | [+0.0027, +0.0092] | -0.0166 | [-0.0229, -0.0108] | -0.0226 | [-0.0297, -0.0160] |

| Estimand | Seeds | Across-seed mean | SD | t95 interval |
|---|---:|---:|---:|---:|
| diffusion | 5 | +0.0098 | 0.0060 | [+0.0023, +0.0173] |
| regression | 5 | -0.0024 | 0.0195 | [-0.0266, +0.0217] |
| interaction | 5 | -0.0122 | 0.0186 | [-0.0353, +0.0109] |

Positive package effects favor temporally sparse full-resolution context. The stable diffusion result and sign-changing regression/interaction results must not be averaged into an equivalence claim.

## Addressability versus self-fed reliability

| Seed | Early h1–24 | 95% CI | Late h61–256 | 95% CI |
|---:|---:|---:|---:|---:|
| 0 | +0.0204 | [+0.0133, +0.0279] | -0.0064 | [-0.0095, -0.0032] |
| 1 | +0.0082 | [+0.0035, +0.0133] | +0.0074 | [+0.0038, +0.0112] |
| 2 | +0.0230 | [+0.0150, +0.0315] | +0.0087 | [+0.0035, +0.0139] |

Registered verdict: **NOT_REPLICATED**. Positive favors the long-reach hierarchy; the late negative direction appears only at seed 0.

## Sampler/NFE robustness family

| Contrast | NFE/step | Sampler seed | NFE/clip | Effect | Simultaneous 95% CI |
|---|---:|---:|---:|---:|---:|
| 896 reach | 8 | 0 | 304 | +0.0478 | [+0.0323, +0.0634] |
| 896 reach | 16 | 0 | 608 | +0.0465 | [+0.0310, +0.0621] |
| 896 reach | 32 | 0 | 1216 | +0.0471 | [+0.0317, +0.0625] |
| 896 reach | 16 | 1 | 608 | +0.0482 | [+0.0325, +0.0638] |
| 896 reach | 16 | 2 | 608 | +0.0466 | [+0.0314, +0.0618] |
| 1,152 interior | 8 | 0 | 304 | +0.0152 | [+0.0045, +0.0259] |
| 1,152 interior | 16 | 0 | 608 | +0.0146 | [+0.0041, +0.0251] |
| 1,152 interior | 32 | 0 | 1216 | +0.0140 | [+0.0036, +0.0245] |
| 1,152 interior | 16 | 1 | 608 | +0.0201 | [+0.0086, +0.0315] |
| 1,152 interior | 16 | 2 | 608 | +0.0200 | [+0.0087, +0.0313] |
| 1,088 vs 1,152 package shape | 8 | 0 | 304 | +0.0276 | [+0.0171, +0.0381] |
| 1,088 vs 1,152 package shape | 16 | 0 | 608 | +0.0283 | [+0.0172, +0.0394] |
| 1,088 vs 1,152 package shape | 32 | 0 | 1216 | +0.0292 | [+0.0182, +0.0403] |
| 1,088 vs 1,152 package shape | 16 | 1 | 608 | +0.0251 | [+0.0160, +0.0342] |
| 1,088 vs 1,152 package shape | 16 | 2 | 608 | +0.0221 | [+0.0133, +0.0309] |
| 2,432 inversion | 8 | 0 | 304 | +0.0115 | [+0.0016, +0.0214] |
| 2,432 inversion | 16 | 0 | 608 | +0.0117 | [+0.0018, +0.0217] |
| 2,432 inversion | 32 | 0 | 1216 | +0.0122 | [+0.0022, +0.0221] |
| 2,432 inversion | 16 | 1 | 608 | +0.0098 | [-0.0001, +0.0196] |
| 2,432 inversion | 16 | 2 | 608 | +0.0077 | [-0.0013, +0.0167] |

Registered verdict: **DIRECTIONALLY_ROBUST**.

## Alternate-codec robustness

| Codec | Contrast | Effect | Simultaneous 95% CI |
|---|---|---:|---:|
| original | b896-reach | +0.0494 | [+0.0339, +0.0649] |
| original | b1152-interior | +0.0283 | [+0.0169, +0.0397] |
| original | b1152-package-shape | +0.0253 | [+0.0163, +0.0343] |
| original | b2432-inversion | +0.0150 | [+0.0065, +0.0235] |
| alternate | b896-reach | +0.0521 | [+0.0374, +0.0669] |
| alternate | b1152-interior | +0.0439 | [+0.0310, +0.0569] |
| alternate | b1152-package-shape | +0.0112 | [+0.0039, +0.0184] |
| alternate | b2432-inversion | +0.0079 | [-0.0006, +0.0165] |

Registered verdict: **CODEC_DIRECTIONALLY_ROBUST**. This is a whole-codec-seed intervention conditional on predictor seed 0.

## Privileged pose-indexed raw-memory reference

| Effect | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| raw-reach-14-to-48 | +0.0723 | [+0.0548, +0.0898] |
| raw-reach-38-to-48 | +0.0210 | [+0.0096, +0.0325] |
| hier-min-to-raw48 | +0.1219 | [+0.1031, +0.1407] |
| hier-long-to-raw48 | +0.1368 | [+0.1190, +0.1545] |
| strided-flat-to-raw48 | +0.1134 | [+0.0944, +0.1324] |
| flat-c2-to-raw48 | +0.1310 | [+0.1122, +0.1498] |

Registered verdict: **RAW_ACCESS_CEILING_CI_CLEAN**. The mosaic receives exact poses and raw observed pixels; it is not token matched.

## Architecture/protocol positioning

| Work | Memory mechanism | Matched budget | Exact observability | Self-fed | Role |
|---|---|---|---|---|---|
| CW-VAE | clocked hierarchical latent state | no | no | yes | temporal-hierarchy lineage |
| FDM | flexible diffusion history conditioning | no | no | yes | diffusion-video baseline lineage |
| TECO | temporally consistent latent transformer | no | no | yes | long-horizon generation context |
| WorldPack | compressed trajectory packing + retrieval | no | no | yes | modern memory/compression comparator |
| CaR / SceneFly | compression + viewpoint-conditioned retrieval | no | annotated geometry | yes | modern retrieval comparator |
| MemLearner / GIM-World | learned query or geometry-aware implicit memory | no | annotated or distilled geometry | yes | learned-memory comparators |
| MBench / MemoBench | benchmark instruments | n/a | task dependent | model dependent | memory-evaluation context |
| Self Forcing | rollout-distribution training | no | no | explicitly targeted | self-fed mismatch intervention |
| This work | flat/spatial/temporal context packages | yes where stated | analytic LoopMom labels | yes | controlled reach–fidelity mechanism study |

Numerical cross-paper ranking is intentionally omitted because dataset, decoder, context, training horizon, rollout, NFE and metrics are not protocol-equivalent.

## Moving-MNIST full-frame metric sanity diagnostic

| Horizon | CW-VAE SSIM | Black SSIM | CW-VAE PSNR | Black PSNR |
|---|---:|---:|---:|---:|
| +1–48 | 0.6338 | 0.7679 | 13.944 | 13.943 |
| +49–192 | 0.5665 | 0.7679 | 12.625 | 13.942 |
| +193–964 | 0.5677 | 0.7682 | 12.909 | 13.941 |

Same archived population: 100 clips, 36 context frames and 964 open-loop targets. CW-VAE is best-of-5 by clip-average SSIM; black is deterministic. This is a background-dominance diagnostic, not a budget-matched model comparison, and the archived mean curves do not support paired inference.

## Artifact binding

| Frozen result artifact | SHA-256 |
|---|---|
| artifacts/confirmation_e2/results.json | 85d01085e9ac074cd099e7b611101b30b7b7112b1f0230ea9163d9825cf2166e |
| artifacts/e4b_head_package/results.json | eb917d948d852d022b872532b826bdb20fc1f17ecfddb55a76446f1821bc9e89 |
| artifacts/e5a_trust_replication/results.json | c38244c207c24c17d6081a2dc75004b6ae96c63c7814d1cfdfbd87bb8ab5414b |
| artifacts/e7c_sampler_nfe/results.json | 93eef5069598b099f62013384e6d91978d7c9996f8fd4f67ab790486758054e6 |
| artifacts/cwvae_black_baseline/results.json | 561b0b3e5eac23276199b601fd22f77303d193921241bd53275861e17617bc78 |
| artifacts/e7a_codec_robustness/results.json | 3f113628babb58cd9adf5223f04a28cc7bb54b59a451b6f5f9aa4ac8d2dd7173 |
| artifacts/e8a_pose_mosaic/results.json | 74b9357f950fbfa4e94160852c53205398f8efad8c2c453b1d8a726be22148be |
