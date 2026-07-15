# E5a trust-boundary replication table shell

> Frozen before seed-1/2 training. Positive effects favor the 224-frame-reach hierarchical
> configuration. Primary rows exclude frames with empty ground-truth foreground.

| Training seed | Early h1–24 effect [95% CI] | Late h61–256 effect [95% CI] | Directional boundary | CI-clean boundary |
|---:|---:|---:|:---:|:---:|
| 0 (legacy reference) | PENDING REGISTERED ANALYSIS | PENDING REGISTERED ANALYSIS | PENDING | PENDING |
| 1 (E5a) | PENDING | PENDING | PENDING | PENDING |
| 2 (E5a) | PENDING | PENDING | PENDING | PENDING |

## Mechanistic secondary rows

| Seed | Early HARD h1–24 | Late HARD h61–256 | Answerable HARD dt>48 | Notes |
|---:|---:|---:|---:|---|
| 0 | PENDING REGISTERED ANALYSIS | PENDING REGISTERED ANALYSIS | PENDING REGISTERED ANALYSIS | Legacy metric artifact |
| 1 | PENDING | PENDING | PENDING | Complete-v2 corroboration available |
| 2 | PENDING | PENDING | PENDING | Complete-v2 corroboration available |

## Interpretation guard

E5a replicates—or fails to replicate—the self-fed early-benefit/late-inversion boundary. The
separately measured GT-fed single-step addressability result is an external prerequisite, not an
E5a endpoint. Secondary bins and complete metrics cannot rescue a failed primary gate. The H-grid3
pool is adaptively used development evidence, so even a successful result supports a measured
boundary in this protocol rather than a universal law.

## Terminal resolution — 2026-07-15

This file preserves the table shell frozen before training. The registered analysis populated the
authoritative generated table in `docs/confirmation/E5A_RESULTS.md`. The verdict is
**NOT_REPLICATED**: seed 0 is early +0.0204 / late −0.0064, seed 1 is early +0.0082 / late
+0.0074, and seed 2 is early +0.0230 / late +0.0087. Both new seeds fail the required
late-negative direction, so `FULL_E5B_NOT_AUTHORIZED` is binding.
