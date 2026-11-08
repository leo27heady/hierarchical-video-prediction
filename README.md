# Hierarchical Video Prediction

Reproduction code for the concluded video-context experiments reported in the article and dissertation. The code covers controlled context allocation, the TRB/LoopMom testbed, long persistent-object Moving MNIST, regression/diffusion comparisons, robustness checks, and the integrated reversible predictor.

## Install and test

```bash
python -m venv .venv
# Activate .venv using your platform's activation command.
python -m pip install -e ".[test,vision]"
python -m pytest -q
python scripts/release_check.py
```

Use Python 3.11 or later. Install the appropriate PyTorch CPU/CUDA build for your hardware. GPU training is not launched by the tests. Full OpenSTL integration checks require its optional checkout. Experiment launchers expect a committed clean source tree.

## Reproduce the experiments

- Structural allocation and TRB: `hvp_hca/`, `scripts/v2/train_article4.py`, `scripts/v2/train_article4_campaign.py`, and `experiments/v2/`. Use each entry point's `--help` to configure datasets, codec checkpoints, run specifications and outputs.
- Confirmatory context comparisons: `hvp_rev/rhvd.py`, `hvp_rev/rhvd_train_v2.py`, `scripts/run_e2_confirmation.ps1` and `scripts/analyze_e2_confirmation.py`.
- Training-seed/head comparisons and robustness: the E4/E4b, E5a, E7a/E7c and E8a launch, analysis and validation scripts.
- Long persistent-object evaluation: `hvp_rev/data/moving_mnist.py`, `hvp_rev/phasec_v2.py` and Phase C v2 scripts. The invalid older evaluator is not an entry point for reported results.
- Integrated model training and memory profiling: `scripts/v2/train_integrated_openstl.py`, `scripts/v2/profile_integrated_openstl_memory.py`, and the OpenSTL code overlay under `integrations/openstl/`.

Run `python <entrypoint> --help` for the exact CLI. The experiment JSON files contain the original hyperparameters and scientific run definitions. Copy configurations to your own run directory and configure your local paths before training. Original run hashes/freeze attestations describe historical executions and must not be relabelled as authorization for modified runs.

The smaller `hvp_dlh` dependency subset is retained only where imported by the concluded model implementation. It is not a separate experimental programme in this release.

## External inputs and outputs

Datasets, codec/model checkpoints, logs, result payloads, images and manuscripts are not included. Supply required inputs explicitly. Existing CLI flags take precedence; supported integration scripts also use HVP_OPENSTL_ROOT and HVP_STORAGE_ROOT. Data generators are supplied for the synthetic tasks. Source datasets such as MNIST and TaxiBJ remain subject to their original access terms.

Published TempVerseFormer and V-BLT have separate repositories. This repository includes their integrated-use code where needed; it does not replace those standalone releases. Reversibility reduces activation storage for the defined component, and does not establish universally constant end-to-end memory or better predictive quality.
