# E8a pose-indexed raw-memory reference results

> Dataset: `loopmom-retrieval_reference_holdout-20260717-804b9f5d2c13`. Positive favors `pose-mosaic-k48`.

Registered verdict: **RAW_ACCESS_CEILING_CI_CLEAN**.

The pose mosaic receives exact positions and raw-pixel storage. It is a privileged access reference, not a token-matched neural competitor.

| Effect | Estimate | Simultaneous 95% CI |
|---|---:|---:|
| raw-reach-14-to-48 | +0.0723 | [+0.0548, +0.0898] |
| raw-reach-38-to-48 | +0.0210 | [+0.0096, +0.0325] |
| hier-min-to-raw48 | +0.1219 | [+0.1031, +0.1407] |
| hier-long-to-raw48 | +0.1368 | [+0.1190, +0.1545] |
| strided-flat-to-raw48 | +0.1134 | [+0.0944, +0.1324] |
| flat-c2-to-raw48 | +0.1310 | [+0.1122, +0.1498] |

| Method | FG MSE | Full MSE | Background MSE | Union MSE |
|---|---:|---:|---:|---:|
| pose-mosaic-k6 | 0.1419 | 0.0075 | 0.0027 | 0.1414 |
| pose-mosaic-k14 | 0.1217 | 0.0064 | 0.0022 | 0.1213 |
| pose-mosaic-k38 | 0.0704 | 0.0038 | 0.0013 | 0.0702 |
| pose-mosaic-k48 | 0.0494 | 0.0027 | 0.0010 | 0.0492 |
| hier-min | 0.1713 | 0.0111 | 0.0054 | 0.1823 |
| flat-min | 0.2056 | 0.0142 | 0.0074 | 0.2197 |
| hier-long | 0.1862 | 0.0121 | 0.0060 | 0.1976 |
| flat-budget | 0.2022 | 0.0138 | 0.0071 | 0.2159 |
| strided-flat | 0.1628 | 0.0106 | 0.0051 | 0.1715 |
| hier-c2 | 0.1852 | 0.0130 | 0.0069 | 0.1920 |
| flat-c2 | 0.1804 | 0.0119 | 0.0058 | 0.1872 |

| Pose mosaic | Stored K | Median stored columns | Mean crop coverage | Pixel payload | Pose payload |
|---|---:|---:|---:|---:|---:|
| pose-mosaic-k6 | 6 | 80.0 | 0.454 | 5120 B | 632 B |
| pose-mosaic-k14 | 14 | 100.0 | 0.550 | 6400 B | 664 B |
| pose-mosaic-k38 | 38 | 140.0 | 0.750 | 8960 B | 760 B |
| pose-mosaic-k48 | 48 | 154.0 | 0.809 | 9856 B | 800 B |

No memory span, fill rule, checkpoint, sampler, threshold or endpoint was added after the reference set was generated.
