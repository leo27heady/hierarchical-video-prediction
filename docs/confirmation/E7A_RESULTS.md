# E7a alternate-codec robustness results

> Dataset: `loopmom-codec_robustness_holdout-20260716-77a4ec5aee94`. Positive favors the registered candidate.

Registered verdict: **CODEC_DIRECTIONALLY_ROBUST**.

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

Descriptive alternate-minus-original effect changes:

| Contrast | Change | Clip-bootstrap 95% CI |
|---|---:|---:|
| b896-reach | +0.0027 | [-0.0014, +0.0069] |
| b1152-interior | +0.0156 | [+0.0113, +0.0199] |
| b1152-package-shape | -0.0142 | [-0.0190, -0.0096] |
| b2432-inversion | -0.0071 | [-0.0109, -0.0034] |

No codec, predictor seed, arm, threshold or retry was selected after results were observed.
