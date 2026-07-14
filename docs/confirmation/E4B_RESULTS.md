# E4b five-seed head × package factorial

> Positive package effects favor temporally sparse full-resolution context. Negative interactions mean regression shifts toward spatial coarsening.

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | -0.0089 [-0.0151, -0.0025] |
| 1 | +0.0046 [+0.0007, +0.0084] | -0.0104 [-0.0146, -0.0062] | -0.0150 [-0.0199, -0.0104] |
| 2 | +0.0111 [+0.0070, +0.0151] | -0.0208 [-0.0270, -0.0150] | -0.0319 [-0.0401, -0.0245] |
| 3 | +0.0075 [+0.0037, +0.0114] | +0.0248 [+0.0181, +0.0323] | +0.0173 [+0.0106, +0.0245] |
| 4 | +0.0060 [+0.0027, +0.0092] | -0.0166 [-0.0229, -0.0108] | -0.0226 [-0.0297, -0.0160] |

## Across-training-seed summaries

- diffusion: +0.0098 [+0.0023, +0.0173], SD 0.0060.
- regression: -0.0024 [-0.0266, +0.0217], SD 0.0195.
- interaction: -0.0122 [-0.0353, +0.0109], SD 0.0186.

Robust negative head shift: **False**.
Regression material-sign counts: **{'temporal': 2, 'spatial': 3, 'small': 0}**.

The development pool is adaptively used. Clip intervals condition on each trained model; the Student-t summaries use training seeds as units.
