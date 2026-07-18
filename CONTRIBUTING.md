# Contributing

The repository is both a software package and the evidence record for a completed scientific
programme. Contributions must preserve that distinction.

## Development setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e ".[test]"
$env:CUDA_VISIBLE_DEVICES = "-1"
.\venv\Scripts\python.exe -m pytest -q
```

The default test suite is CPU-only and must not require local datasets, W&B credentials or network
access.

## Compatibility rules

- Treat `hvp_rev` as the frozen experiment implementation. Avoid mechanical renames or algorithmic
  cleanup that could change tensor order, random-number consumption, optimizer state, EMA,
  sampling or artifact identity.
- Add reusable interfaces through `hvp`, with parity tests against the frozen implementation.
- Do not weaken clip-level clustering, persistent-object assignment, empty-frame semantics,
  provenance hashes or holdout-purpose checks.
- Do not overwrite completed content-addressed artifacts.
- Do not turn a historical development analysis into confirmatory evidence.

## Scientific changes

A new experiment requires a new entry in `experiments/authorization.json`, a frozen configuration,
explicit endpoints and a decision rule. Terminal or skipped studies must remain non-launchable.
Claim changes must update `docs/CLAIM_LEDGER.md`, the article/dissertation source and the evidence
manifest together.

## Pull-request checklist

1. CPU tests pass.
2. `python scripts/validate_scientific_configs.py` passes.
3. Text/claim consistency gates pass.
4. A wheel builds and imports in a clean environment.
5. No local absolute paths, credentials, datasets, checkpoints or W&B state are committed.
