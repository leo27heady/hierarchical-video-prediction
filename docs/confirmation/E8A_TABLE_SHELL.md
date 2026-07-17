# E8a pose-indexed raw-memory reference

**Status:** frozen empty shell. Populate only from the immutable E8a result artifact.

## Validity

| Check | Expected | Result |
|---|---|---|
| Purpose-locked independent set | `retrieval_reference_holdout` | pending |
| Static-world repeated pixels | byte-identical | pending |
| Training-only unknown-pixel prior | hash verified | pending |
| Prediction leakage | no post-warm-up pixels | pending |
| Aligned population | 100 clips x 152 frames x 11 methods | pending |
| Neural identity | frozen codec/checkpoints/NFE/sampler/geometry | pending |
| W&B and immutable artifacts | complete | pending |

## Memory and coverage

| Method | Stored warm frames | Median stored world columns | Raw pixel bytes | Pose/mask bytes | Mean current-crop coverage |
|---|---:|---:|---:|---:|---:|
| pose-mosaic-k6 | 6 | pending | pending | pending | pending |
| pose-mosaic-k14 | 14 | pending | pending | pending | pending |
| pose-mosaic-k38 | 38 | pending | pending | pending | pending |
| pose-mosaic-k48 | 48 | pending | pending | pending | pending |

## Registered simultaneous family

Positive effects favor `pose-mosaic-k48`. All six effects share one simultaneous family. The
four neural-to-K48 rows determine the ceiling category; K14→K48 is the raw-span dose response and
K38→K48 is the prespecified coverage-saturation diagnostic.

| Effect | Estimate | Simultaneous 95% CI | Decision |
|---|---:|---:|---|
| k14 minus k48 | pending | pending | pending |
| k38 minus k48 | pending | pending | pending |
| hier-min minus k48 | pending | pending | pending |
| hier-long minus k48 | pending | pending | pending |
| strided-flat minus k48 | pending | pending | pending |
| flat-c2 minus k48 | pending | pending | pending |

## Complete metrics

| Method | Foreground MSE | Full MSE | Background MSE | Union MSE | Precision | Recall | Misses | Extras |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pose-mosaic-k6 | pending | pending | pending | pending | pending | pending | pending | pending |
| pose-mosaic-k14 | pending | pending | pending | pending | pending | pending | pending | pending |
| pose-mosaic-k38 | pending | pending | pending | pending | pending | pending | pending | pending |
| pose-mosaic-k48 | pending | pending | pending | pending | pending | pending | pending | pending |
| hier-min | pending | pending | pending | pending | pending | pending | pending | pending |
| flat-min | pending | pending | pending | pending | pending | pending | pending | pending |
| hier-long | pending | pending | pending | pending | pending | pending | pending | pending |
| flat-budget | pending | pending | pending | pending | pending | pending | pending | pending |
| strided-flat | pending | pending | pending | pending | pending | pending | pending | pending |
| hier-c2 | pending | pending | pending | pending | pending | pending | pending | pending |
| flat-c2 | pending | pending | pending | pending | pending | pending | pending | pending |

## Registered verdict

`PENDING`

This table must state that pose mosaics receive exact positions and raw-pixel storage and are not
token-matched neural competitors. A zero K38→K48 effect does not fail the ceiling verdict when
the 38-frame mosaic has already saturated world-column coverage.
