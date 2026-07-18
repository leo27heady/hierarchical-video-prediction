# Experiment and Script Index

This index separates release commands, terminal scientific workflows and archived development
utilities. It does not grant launch authority; `experiments/authorization.json` does.

## Release and manuscript commands

| Purpose | Command |
|---|---|
| Complete CPU release gate | `python scripts/release_check.py` |
| Scientific configuration validation | `python scripts/validate_scientific_configs.py` |
| Claim/text consistency | `python scripts/check_text_consistency.py --gate` |
| Fourth-article consistency | `python scripts/check_rft_consistency.py` |
| Final tables | `python scripts/make_rft_final_tables.py` |
| Final figures | `python scripts/make_rft_result_figures.py` |
| Bind dissertation references | `python scripts/bind_dissertation_references.py` |
| Assemble English manuscripts | `python scripts/assemble_final_manuscripts.py` |
| Release evidence manifest | `python scripts/make_release_manifest.py` |

## Terminal registered studies

| Study | Preparation / launch | Analysis / validation | Terminal status |
|---|---|---|---|
| E2 untouched confirmation | `run_e2_confirmation.ps1` | `analyze_e2_confirmation.py`, `validate_e2_completion.py` | complete |
| E4/E4b head × package | `run_e4_factorial.ps1`, `run_e4b_factorial.ps1` | `analyze_e4_factorial.py`, `analyze_e4b_factorial.py` | complete |
| E5a trust replication | `run_e5a_trust_replication.ps1` | `analyze_e5a_trust_replication.py` | complete; proposed boundary not replicated |
| E7c sampler/NFE | `run_e7c_sampler_nfe.ps1` | `analyze_e7c_sampler_nfe.py` | complete |
| E7a alternate codec | `run_e7a_codec_robustness.ps1` | `analyze_e7a_codec_robustness.py` | complete |
| E8a explicit-access reference | `run_e8a_reference.ps1` | `analyze_e8a_pose_mosaic.py` | complete |
| E6 persistent-object boundary | `run_phasec_v2.ps1` | `analyze_phasec_v2.py`, `validate_phasec_completion.py` | complete; mixed boundary |
| E9 counterfactual loop world | none | utility gate in canonical ledger | skipped; no incremental post-freeze utility |

Every terminal launcher fails closed through the authorization ledger. Rerunning a terminal study
requires registration as a new reproduction, not a status edit.

## Prepared but unauthorized controls

- `run_b1_fixed_target.ps1` and `hvp_rev.fixed_target_control` isolate the B1 target-interface
  confound. They remain prepared but unauthorized because the article permanently scopes B1 as a
  configuration result.
- `run_trust_horizon_dose_response.ps1` is disabled. The seed-0 late inversion did not replicate,
  so localization is not authorized.

## Archived development workflows

- `run_newvision_queue.*` is terminally disabled. It encoded a superseded queue and referenced the
  retired Phase C workflow.
- `run_phaseC_mmnist.sh` is retained as provenance but fails before data generation or GPU work.
- `make_fig9_frontier.py`, `make_fig10_flat_vs_hier.py` and `make_fig13_family_edges.py` reproduce
  development-era visualizations only when `HVP_ALLOW_HISTORICAL_FIGURES=1`. They are not terminal
  evidence; use `make_rft_result_figures.py`.
- Early `e0_*`, `e1_*`, `e2_rollout_drift.py`, mechanism probes and shell sweeps are historical
  diagnostics. They are not entry points for a new study.

## Environment-specific inputs

Dataset-generation PowerShell scripts accept `-Mnist` or the `HVP_MNIST_IDX` environment variable.
The historical activation-memory demo accepts `HVP_MOVING_MNIST_NPY`. No release script contains a
developer-machine dataset path.
