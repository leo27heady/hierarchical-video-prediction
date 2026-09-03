### Ж.v2 Research branches, commands, and evidence gates

The scientific campaign worktree and the dissertation editing worktree are separate. Do not launch experiments from the editorial branch or alter the active campaign checkout. The new manuscript is assembled from frozen commit `cf3f661`, whose nine source files were byte-identical to the Path B manuscript snapshot. The original remains recoverable from Git. This early separate-worktree implementation is a safe deviation from the plan's wait-until-campaign-end ordering; final result insertion still waits for locked artifacts.

Article 4 controller, in the pinned `hierarchical-video-prediction-path-b` worktree:

```powershell
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/v2/run_article4_campaign.py status
# Execution command is documentation only; do not start a second active controller.
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/v2/run_article4_campaign.py run --execute
```

The controller validates configuration hashes, source commit, required storage, and completion state. Completed runs are skipped; interrupted runs resume saved training state. Local JSON/JSONL and checkpoint artifacts are authoritative. W&B uses the existing dedicated project `hierarchical-video-prediction-article4-v2`; mirroring failures do not authorize replacing local evidence. The post-training finalizer checks all 86 final-step records, writes selection before protected access, and evaluates only the union of selected confirmation/mechanism roles.

Integrated controller, in `hierarchical-video-prediction-integration`, with the separately configured `OpenSTL-integrated` checkout:

```powershell
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/v2/run_integrated_openstl_campaign.py status
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/v2/run_integrated_openstl_campaign.py preflight
# Run only after authorization, idle-GPU, data, storage, correctness, and quality gates.
& 'C:/Users/leoni/Documents/projects/hierarchical-video-prediction/venv/Scripts/python.exe' scripts/v2/run_integrated_openstl_campaign.py run
```

The integrated manifest declares the data, code, four final run cells, logging, and authorization checks. It does not schedule all component-quality ablations or matched baseline training. No experimental command is executed by this editorial refactor.

<!-- INTEGRATED_SOURCE_SNAPSHOT -->

The integrated model source is `openstl/models/hvp_integrated_model.py`. `PhysicalTokenMetadata` stores x/y, time, scale, and level; `_grid_metadata` uses cell centres and the protocol time denominator; `SpaceTimeRouter` gathers true membership; `ReversibleTransformerCore` changes execution storage; `decode_packed` constructs future queries. Input shape is enforced by configuration. Encoder, router, decoder, retained coarse grid, and dense loss boundaries must be accounted for outside the reversible core.

#### Ж.v2.1 Correctness versus uncompleted performance cells

| Cell | Current scope | Missing evidence / allowed conclusion |
|---|---|---|
| CPU quota and response | fixed quota, uniqueness, frame coverage, response to feature change | compatibility, not trained routing utility |
| CPU position/permutation | metadata follows tokens; decoder fp32 atol/rtol 3e-6 | CUDA/deployed mixed-precision permutation pending |
| Core fp32 parity | outputs atol/rtol 2e-6; input gradients 2e-5/2e-4; parameter gradients 3e-5/3e-4 | tested tiny core, not every full architecture/protocol |
| Core inverse | double-precision atol/rtol 1e-10 | measure deployed dtype/depth drift separately |
| CPU bfloat16 core | outputs 5e-3/5e-3; input gradients 5e-3/5e-2; parameters 1e-2/8e-2 | does not certify CUDA bfloat16 |
| Whole-model static path | output 3e-6/3e-6; input gradients 3e-5/4e-4; connectivity | complete parameter-gradient parity and dynamic/mixed path parity remain gates |
| Shape/checkpoint/OpenSTL smoke | both dataset shapes, checkpoint round trip, optimizer/training/validation interface | not a trained quality result |
| Inference route switch | mixed evaluation resolves to hybrid; policy interface exists | same trained checkpoint static/density/hybrid quality and stability pending |
| Ordinary/reversible profile | profile utility exists | matched trained-checkpoint CUDA measurements pending |
| Activation checkpointing | required practical comparator | current profile utility does not implement this cell; add separately or restrict claims to ordinary autograd |
| Flat/hierarchy quality | separately trained development ablation required | cannot use execution toggles to establish quality |
| Static/mixed training | separately trained development ablation required | four full-model runs do not supply this contrast |
| Smaller quota | bounded evaluations/training after routing gate | fixed-quota routing alone is not a memory gain |
| MM and TaxiBJ Full | four long conditional runs | no results until authorized and completed |
| Matched local baselines | existing compatible checkpoints or additional training | published results remain contextual if contracts differ |

The gate summary references the recorded test source and verification report, not an invented “all tests pass” claim for every future configuration. Adjacent-frame route churn is distinct from stability across autoregressive calls. Top-k equal-score ties and the unconstrained score margin remain limitations. A route switch changes selected-score feature addition as well as membership in the current implementation.

#### Ж.v2.2 Editorial reproduction

```powershell
python scripts/make_dissertation_v2_assets.py
python scripts/refactor_dissertation_v2.py
python scripts/register_dissertation_v2_evidence.py
python scripts/audit_dissertation_v2.py
# The release check must fail while experimental or institutional placeholders remain.
python scripts/audit_dissertation_v2.py --release
```

Run these in `hierarchical-video-prediction-dissertation`. They perform no experiment launches, use no CUDA, and do not modify active worktrees. The source fragments are the editing surface; the assembly script is a deterministic bulk migration. Do not edit an assembled section and then regenerate without first transferring the edit to its source fragment.

New result insertion requires an immutable accepted evidence bundle with configuration/checkpoint/data hashes, seed IDs, selection role, endpoint, uncertainty unit, and finalized status. A missing bundle produces a visible result slot and no plotted value. The calibration and historical transcriptions have distinct provenance; they cannot fill new performance cells.
