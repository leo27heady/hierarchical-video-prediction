# E7c sampler/NFE robustness table shell (archival preregistration)

> Frozen before evaluation. Positive effects favor the registered candidate. Fill only from the
> immutable E7c analysis artifact.

> **Terminal fulfillment (2026-07-16):** the generated 20-row table is
> `docs/confirmation/E7C_RESULTS.md`. This pre-result shell is retained to show that no cell was
> added or selected after observation.

| Contrast | NFE | Sampler seed | Effect | Simultaneous 95% CI | Direction retained? |
|---|---:|---:|---:|---:|---|
| b896 reach | 8 | 0 | PENDING | PENDING | PENDING |
| b896 reach | 16 | 0 | PENDING | PENDING | PENDING |
| b896 reach | 32 | 0 | PENDING | PENDING | PENDING |
| b896 reach | 16 | 1 | PENDING | PENDING | PENDING |
| b896 reach | 16 | 2 | PENDING | PENDING | PENDING |
| b1152 interior | 8/16/32 | 0/1/2 as frozen | PENDING | PENDING | PENDING |
| b1152 package shape | 8/16/32 | 0/1/2 as frozen | PENDING | PENDING | PENDING |
| b2432 inversion | 8/16/32 | 0/1/2 as frozen | PENDING | PENDING | PENDING |

Registered terminal verdict: **DIRECTIONALLY_ROBUST** — 20/20 point estimates retained direction;
18/20 simultaneous lower bounds were positive. The two non-clean intervals are the 2,432-token
inversion at NFE 16 with sampler seeds 1 and 2.
