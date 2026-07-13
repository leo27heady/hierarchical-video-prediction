# E4 head × package factorial

> Positive package effects favor temporally sparse full-resolution context. Negative interactions mean regression shifts the package contrast toward spatial coarsening.

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | +0.0197 [+0.0143, +0.0252] | +0.0107 [+0.0062, +0.0158] | -0.0089 [-0.0151, -0.0025] |
| 1 | +0.0046 [+0.0007, +0.0084] | -0.0104 [-0.0146, -0.0062] | -0.0150 [-0.0199, -0.0104] |
| 2 | +0.0111 [+0.0070, +0.0151] | -0.0208 [-0.0270, -0.0150] | -0.0319 [-0.0401, -0.0245] |

Consistent negative interaction across all seeds: **True**.
E4b training-seed heterogeneity trigger: **True** (opposite material seed effects; across-seed SD 0.0161 > 0.010).

Clip-bootstrap intervals condition on each trained model. The E4b trigger uses variation across training seeds, not those intervals.
