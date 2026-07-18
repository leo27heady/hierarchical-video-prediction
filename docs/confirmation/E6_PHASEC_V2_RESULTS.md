# E6 persistent-object long Moving-MNIST results

Registered verdict: **MIXED_BOUNDARY**.

Positive effects favor the hierarchical 1,152-token package. Intervals are one simultaneous 95% max-statistic clip-cluster family over all 15 co-primary effects.

| Seed | Endpoint | Effect | Simultaneous 95% CI |
|---:|---|---:|---:|
| 0 | foreground_mse | +0.0036 | [-0.0074, +0.0146] |
| 0 | union_mse | +0.0102 | [+0.0019, +0.0185] |
| 0 | trajectory_error | +2.4786 | [+0.2130, +4.7443] |
| 0 | identity_accuracy | -0.0762 | [-0.1106, -0.0418] |
| 0 | object_coverage | +0.0304 | [+0.0016, +0.0591] |
| 1 | foreground_mse | -0.0031 | [-0.0127, +0.0065] |
| 1 | union_mse | +0.0122 | [+0.0052, +0.0192] |
| 1 | trajectory_error | +1.4223 | [-1.0646, +3.9092] |
| 1 | identity_accuracy | -0.0561 | [-0.0934, -0.0188] |
| 1 | object_coverage | +0.0159 | [-0.0163, +0.0482] |
| 2 | foreground_mse | -0.0024 | [-0.0131, +0.0083] |
| 2 | union_mse | +0.0169 | [+0.0093, +0.0245] |
| 2 | trajectory_error | +4.1325 | [+1.5271, +6.7380] |
| 2 | identity_accuracy | -0.0647 | [-0.0955, -0.0338] |
| 2 | object_coverage | +0.0498 | [+0.0161, +0.0834] |

| Seed | Arm | FG MSE | Union MSE | Identity | Trajectory px | Coverage | Missed | Extra |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | mmL-flat | 0.6393 | 0.6861 | 0.2139 | 42.19 | 0.6194 | 0.761 | 0.144 |
| 0 | mmL-hier | 0.6357 | 0.6759 | 0.1377 | 39.71 | 0.6498 | 0.701 | 0.144 |
| 1 | mmL-flat | 0.6261 | 0.6901 | 0.2437 | 40.20 | 0.6460 | 0.708 | 0.144 |
| 1 | mmL-hier | 0.6292 | 0.6779 | 0.1876 | 38.77 | 0.6619 | 0.676 | 0.153 |
| 2 | mmL-flat | 0.6310 | 0.6931 | 0.2223 | 43.79 | 0.5994 | 0.801 | 0.111 |
| 2 | mmL-hier | 0.6334 | 0.6763 | 0.1576 | 39.65 | 0.6491 | 0.702 | 0.131 |

Equivalence was disabled before protected access. Moving-MNIST is an external boundary, not evidence for LoopMom's exact re-entry mechanism or natural-video transfer.

## Interpretation

The registered `MIXED_BOUNDARY` verdict is not a null result and should not be compressed into one
average model score:

- hierarchy has lower union-mask error at every seed;
- hierarchy has lower trajectory error and higher coverage with clean simultaneous intervals at
  seeds 0 and 2;
- hierarchy misses fewer objects and has lower full-frame/background error at every seed as
  secondary descriptive evidence;
- flat context has higher digit-identity accuracy with clean simultaneous intervals at every
  seed;
- foreground-only MSE is unresolved at all seeds and changes sign.

The consistent reading is that the coarse long-memory stream better preserves spatial support and
continuity, while flat full-resolution storage better preserves fine glyph identity. This is an
external reach–fidelity boundary, not evidence that either architecture universally dominates.

## Completion and provenance

- Frozen geometry: 48 warm-up frames, 300 self-fed targets, 100 protected clips, NFE 16 and
  sampler seed 0.
- Inference: one studentized max-statistic clip-cluster family over 15 co-primary effects, 100
  clip clusters, 10,000 replicates and critical value 2.8599605.
- Completion validator:
  `artifacts/phasec_v2/completion_validation.json`,
  status `PHASEC_V2_COMPLETION_PASS`.
- Protected access: exactly one integrity open plus six unique evaluation RunSpecs.
- Cloud audit: 15 finished W&B runs and immutable artifacts,
  status `WANDB_GROUP_AUDIT_PASS`.
- Mechanical project GPU ownership: 20.5127031 hours, including failed preparation cells.
- Frozen numerical source: `artifacts/phasec_v2/results.json`,
  SHA-256 `75a628ca6f2fa87e74e67bbe7b95d4acee6e0f35969d28fe97e72b0b40e981c7`.
