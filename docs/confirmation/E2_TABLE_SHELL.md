# E2 untouched-confirmation table shell

**Status:** frozen empty shell; populate only through `scripts/analyze_e2_confirmation.py`.

## Selected primary contrasts

| Pair | Seed | Population | Metric | Candidate | Baseline | Effect | 95% clip-cluster CI | Records | Clips | Verdict branch |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| reach-48-vs-14 | 0 | legacy-all | legacy foreground | | | | | | | |
| reach-48-vs-14 | 1 | legacy-all | legacy foreground | | | | | | | |
| reach-48-vs-14 | 0 | 14 < dt ≤ 48, answerable, non-empty | complete metric rows | | | | | | | |
| reach-48-vs-14 | 1 | 14 < dt ≤ 48, answerable, non-empty | complete metric rows | | | | | | | |
| b1-config-48-vs-8 | 0 | legacy-all | legacy foreground | | | | | | | |
| b1-config-48-vs-8 | 1 | legacy-all | legacy foreground | | | | | | | |
| b1-config-48-vs-8 | 0 | 8 < dt ≤ 48, answerable, non-empty | complete metric rows | | | | | | | |
| b1-config-48-vs-8 | 1 | 8 < dt ≤ 48, answerable, non-empty | complete metric rows | | | | | | | |

Positive effect means the candidate package is better. B1 remains a whole-configuration result.

## Frozen frontier family

| Contrast | Budget | Regime | Seed | Candidate | Baseline | Effect | Simultaneous 95% CI | Status |
|---|---:|---|---:|---|---|---:|---:|---|
| near-bottom-s0 | 480 vs 512 | B16×30k | 0 | hier-short | flat-short | | | near-match |
| near-bottom-s1 | 480 vs 512 | B16×30k | 1 | hier-short | flat-short | | | near-match |
| b896-s0 | 896 | B16×30k | 0 | hier-min | flat-min | | | matched |
| b896-s1 | 896 | B16×30k | 1 | hier-min | flat-min | | | matched |
| b1152-s0 | 1,152 | B16×30k | 0 | hier-long | flat-budget | | | matched |
| b1152-s1 | 1,152 | B16×30k | 1 | hier-long | flat-budget | | | matched |
| b1152-s2 | 1,152 | B16×30k | 2 | hier-long | flat-budget | | | matched |
| b1792-s0 | 1,792 | B16×30k | 0 | hier-c1 | flat-c1 | | | matched |
| b2432-30k-s0 | 2,432 | B16×30k | 0 | hier-c2 | flat-c2 | | | matched |
| b2432-60k-s0 | 2,432 | B8×60k | 0 | hier-c2 | flat-c2 | | | matched, separate regime |
| b3072-s0 | 3,072 | B8×60k | 0 | hier-rich | flat-long | | | matched |
| b3072-s1 | 3,072 | B8×60k | 1 | hier-rich | flat-long | | | matched |

Do not interpolate a crossing or promote a post-result cell to primary.

## Required corroborating decomposition

For every selected pair/seed, generated rows must include:

- non-empty GT-foreground, union-mask, full-frame and GT-background MSE;
- precision, recall and F1;
- missed and extra components;
- matched IoU/centroid distance where defined;
- component/scoring coverage;
- empty-GT stratum;
- true record-level sharpness.
