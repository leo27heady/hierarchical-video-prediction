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

## Article 4 primary 30k campaign

The executable campaign is
`campaigns/article4_primary_30k_2026-09-01.json`. It authorizes 80 immediately resolved runs and
declares six dependency-resolved runs, for 43 unique configurations and 86 two-seed training runs.
The single controller is `scripts/v2/run_article4_campaign.py`.

```powershell
$python = 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction\venv\Scripts\python.exe'
& $python scripts/v2/run_article4_campaign.py preflight
& $python -u scripts/v2/run_article4_campaign.py run --execute
```

Rerun the same launch command after interruption. Completed configuration hashes are skipped and
incomplete workers restore their atomic model/optimizer/EMA/RNG checkpoint. The controller also
runs the locked post-training mechanism and sealed-confirmation stages. See
`docs/plans/ARTICLE4_30K_CAMPAIGN_IMPLEMENTATION_AND_OPERATIONS_2026-09-01.md` for the complete
contract and evidence layout.
