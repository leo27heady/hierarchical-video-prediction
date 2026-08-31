# Version 2 experiment registry

This namespace holds the clean Path B experiment definitions. Legacy result and confirmation directories are never reused.

Repository JSON files are non-launchable templates. They are validated and hashed by:

```powershell
venv\Scripts\python.exe scripts\v2\validate_experiment_configs.py
```

Before a GPU launch, a resolved configuration must additionally have:

- immutable train/validation/test split manifest and dataset hash;
- exact git commit and environment manifest;
- hardware identity and memory-profiler mode;
- declared screening or confirmation status;
- authorization ledger entry for the resolved configuration hash;
- output directory outside protected legacy artifact paths.

The config validator intentionally rejects test-based checkpoint selection, dynamic routing with static slot positions, and an external second-order reversible rollout presented as a memory-only comparator.
