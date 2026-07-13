# E4b training-seed expansion — frozen table shell

> Frozen before seeds 3–4 training. Positive package effects favor temporally sparse
> full-resolution context. Negative interactions mean regression shifts the package contrast
> toward spatial coarsening. All cells use the existing adaptively used development pool.

| Seed | Diffusion temporal advantage | Regression temporal advantage | Head × package interaction |
|---:|---:|---:|---:|
| 0 | frozen prior | frozen prior | frozen prior |
| 1 | frozen prior | frozen prior | frozen prior |
| 2 | frozen prior | frozen prior | frozen prior |
| 3 | pending | pending | pending |
| 4 | pending | pending | pending |

## Outcome-independent reading

- Within-model intervals are paired 10,000-replicate clip-cluster bootstraps and condition on a
  trained model.
- Across-training-seed summaries use the five seed effects as units and report the arithmetic
  mean, sample SD and two-sided 95% Student-t interval with four degrees of freedom.
- A robust head shift requires all five interaction point estimates to be negative and the
  across-seed interval to lie entirely below zero.
- A regression package preference is seed-stable only if all five effects have the same material
  sign. Otherwise report sign counts and heterogeneity, not equivalence.
- E4b terminates after seeds 3–4 for every outcome. It cannot trigger seeds 5+.
- E4b replaces E8c's GPU allocation; the literature/protocol comparison table remains required
  CPU-only work.
