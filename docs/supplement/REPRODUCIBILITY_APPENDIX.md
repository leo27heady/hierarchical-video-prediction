# Reproducibility appendix

**Frozen:** 2026-07-18  
**Scope:** fourth article and its dissertation Chapter 4 evidence  
**Scientific authority:** `docs/CLAIM_LEDGER.md`  
**Execution authority:** `experiments/authorization.json`

## A. Reproduction levels

The repository distinguishes three levels because protected datasets and trained checkpoints are
too large for the Git source archive.

1. **CPU semantic validation.** Tests metric definitions, Hungarian assignment, persistent-object
   metadata, reach-band construction, immutable provenance, resume semantics, experiment gates and
   CPU-safe model plumbing without local datasets or a GPU.
2. **Artifact analysis.** Recomputes conclusions and regenerates article tables/figures from
   hash-pinned result JSON/NPZ artifacts. The local evidence archive is authoritative; W&B is a
   cloud mirror and health record.
3. **Training reproduction.** Recreates a model only from the frozen dataset/codec/config
   identities and a content-addressed `RunSpec`. Legacy checkpoints without complete RNG state are
   distributionally reproducible rather than trajectory-identical; provenance-v2 runs record
   Python, NumPy and Torch RNG state.

## B. Frozen environment

The terminal QA environment was:

| Component | Version |
|---|---|
| OS | Windows 11, build 26200 |
| Python | 3.12.0, 64-bit |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| PyTorch | 2.11.0+cu126 |
| CUDA build | 12.6 |
| Matplotlib | 3.11.0 |
| W&B | 0.27.2 |
| pytest | 9.1.0 |
| GPU | NVIDIA GeForce RTX 4060 Ti, 16,380 MiB |
| Driver | 591.86 |

`pyproject.toml` defines the package bounds and optional dependencies. `requirements-cpu.txt`
pins the CPU validation core.

## C. CPU validation

From the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:WANDB_API_KEY=$null
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\check_rft_consistency.py
.\venv\Scripts\python.exe scripts\check_text_consistency.py
.\venv\Scripts\python.exe scripts\validate_scientific_configs.py
```

Terminal result: **81 tests passed**. Packaging/RFT consistency, text consistency and all ten
scientific-config/fail-closed authorization checks passed. A completed E6 launch request is
correctly rejected with status `COMPLETED_EVIDENCE_FROZEN`.

## D. Regenerating tables and figures

```powershell
.\venv\Scripts\python.exe scripts\make_rft_final_tables.py
.\venv\Scripts\python.exe scripts\make_rft_result_figures.py
```

The first command writes:

- `docs/draft/RFT_GENERATED_TABLES.md`;
- `artifacts/rft_claim_artifact_index.json`.

The second writes the result figures in `docs/figures/`. Figure 22 is generated directly from
`artifacts/phasec_v2/results.json`; its positive direction always favors hierarchy.

## E. Central immutable result artifacts

The generated index is the machine-readable authority for the current hashes. At the terminal E6
freeze, the primary result hashes are:

| Study | Frozen artifact | SHA-256 |
|---|---|---|
| E2 confirmation/frontier | `artifacts/confirmation_e2/results.json` | `85d01085e9ac074cd099e7b611101b30b7b7112b1f0230ea9163d9825cf2166e` |
| E4b head/package | `artifacts/e4b_head_package/results.json` | `eb917d948d852d022b872532b826bdb20fc1f17ecfddb55a76446f1821bc9e89` |
| E5a trust replication | `artifacts/e5a_trust_replication/results.json` | `c38244c207c24c17d6081a2dc75004b6ae96c63c7814d1cfdfbd87bb8ab5414b` |
| E7c sampler/NFE | `artifacts/e7c_sampler_nfe/results.json` | `93eef5069598b099f62013384e6d91978d7c9996f8fd4f67ab790486758054e6` |
| E7a alternate codec | `artifacts/e7a_codec_robustness/results.json` | `3f113628babb58cd9adf5223f04a28cc7bb54b59a451b6f5f9aa4ac8d2dd7173` |
| E8a pose mosaic | `artifacts/e8a_pose_mosaic/results.json` | `74b9357f950fbfa4e94160852c53205398f8efad8c2c453b1d8a726be22148be` |
| E6 persistent objects | `artifacts/phasec_v2/results.json` | `75a628ca6f2fa87e74e67bbe7b95d4acee6e0f35969d28fe97e72b0b40e981c7` |

The repository release also carries E6's completion and W&B audit JSONs. Large per-frame NPZs,
datasets and checkpoints remain in the local content-addressed archive and are identified by
their RunSpec/manifests rather than copied into Git.

## F. E6 independent completion command

```powershell
$env:WANDB_API_KEY=$null
.\venv\Scripts\python.exe scripts\validate_phasec_completion.py `
  --config experiments\configs\phase_c_v2.json `
  --manifest data\phasec_v2\mmnist_holdout_500x348.manifest.json `
  --results artifacts\phasec_v2\results.json `
  --gpu-ledger artifacts\phasec_v2\gpu_hours.jsonl `
  --access-log artifacts\phasec_v2\access.jsonl `
  --run-root artifacts\runs `
  --training-root artifacts\training `
  --wandb-audit artifacts\phasec_v2\wandb_validation.json `
  --out artifacts\phasec_v2\completion_validation.json
```

Expected status: `PHASEC_V2_COMPLETION_PASS`, six training cells, six evaluation cells, 30,000
paired records per cell, seven protected accesses, 15 W&B runs and 20.5127031 mechanically charged
GPU-hours.

The E6 cloud group is
[`phasec-v2-persistent-object-moving-mnist-phase_c_v2_holdout-test-20260720-198306f9eab1`](https://wandb.ai/leo27heady/hgrid/groups/phasec-v2-persistent-object-moving-mnist-phase_c_v2_holdout-test-20260720-198306f9eab1).
The cloud record is not used to recompute the inference.

## G. Protected-data policy

Protected datasets are purpose-bound. Access is logged before opening, and immutable dataset
identity must match the consuming RunSpec. E2 may not be reused for model or robustness selection.
E6 holdout access is permanently closed after the six frozen evaluations. Smoke runs use
run-specific paths and cannot overwrite canonical evidence.

## H. Interpretation safeguards

- Positive E2/E6 effects favor the registered candidate or hierarchy exactly as stated in each
  artifact; endpoint-specific sign definitions are not inferred from column names.
- Clip-bootstrap intervals condition on a trained model. Training-seed uncertainty is reported
  separately and cannot be replaced by a frame/clip bootstrap.
- Tested brackets are not continuous crossings.
- B1 is a configuration result because the target interface changes.
- E7a is a whole-codec intervention; E8a is a privileged access reference.
- E6 is `MIXED_BOUNDARY`, not equivalence: support/continuity and identity fidelity favor
  different packages.
- Legacy Phase C object metrics remain invalid and are never revived by corrected E6.
