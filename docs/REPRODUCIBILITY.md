# Reproducibility Guide

## What can be reproduced without protected artifacts

The CPU suite verifies metric semantics, Hungarian assignment, reach geometry, paired and
simultaneous cluster inference, dataset schemas, run identity, RNG restoration, immutable
artifacts and authorization gates:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[test]"
$env:CUDA_VISIBLE_DEVICES = "-1"
.\venv\Scripts\python.exe -m pytest -q
```

## Evidence regeneration

With the hash-matched result artifacts present:

```powershell
.\venv\Scripts\python.exe scripts\make_rft_final_tables.py
.\venv\Scripts\python.exe scripts\make_rft_result_figures.py
.\venv\Scripts\python.exe scripts\assemble_final_manuscripts.py
```

The canonical claim source is `docs/CLAIM_LEDGER.md`. The final evidence manifest records file
sizes and SHA-256 hashes. Historical development figures 9 and 10 are retained for provenance but
are not the terminal simultaneous frontier; use `docs/figures/article_confirmed_frontier.*`.

## Experiment reproduction

Training requires the exact dataset and codec hashes named by a `RunSpec`, the frozen arm
configuration and an explicitly launchable authorization entry. Check the gate without launching:

```powershell
.\venv\Scripts\hvp.exe gate STUDY_NAME --ledger experiments\authorization.json --root .
```

All terminal studies are deliberately blocked. A reproduction should be registered as a new study
instead of changing a terminal record.

## Statistical conventions

- comparisons are paired on shared clips and sampler geometry;
- intervals resample clips, never individual autoregressive frames;
- registered frontier families use simultaneous max-statistic intervals;
- foreground-only MSE is reported only on non-empty ground-truth foreground;
- full-frame, background, union-mask, missed/extra and coverage endpoints disclose false positives
  and missing detections;
- persistent identity uses generator object IDs and one-to-one assignment.

## Known limits

Legacy checkpoints before provenance format 2 do not include complete RNG/sampler state and are
distributionally, not trajectory, reproducible. The core evidence is synthetic; E6 is an external
synthetic boundary rather than natural-video generalization.
