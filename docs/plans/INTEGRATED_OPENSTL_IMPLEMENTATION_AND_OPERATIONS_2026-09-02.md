# Integrated OpenSTL Implementation and Operations

Date: 2 September 2026  
Current state: CPU correctness and restart path verified; CUDA performance/memory gates deferred
until the Article 4 controller releases the GPU

## 1. Repository isolation

Do not edit or commit the live Article 4 worktree while its controller is running. It validates the
pinned commit before every worker.

| Purpose | Path | Branch |
|---|---|---|
| plans, manifest, controller, worker, profiler | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration` | `codex/integrated-openstl-v2` |
| clean OpenSTL-native model and loaders | `C:\Users\leoni\Documents\projects\OpenSTL-integrated` | `codex/hvp-integrated-v2` |
| datasets and user TempVerseFormer work, read-only | `C:\Users\leoni\Documents\projects\OpenSTL` | `OpenSTL-Lightning` |
| live Article 4 campaign, read-only | `C:\Users\leoni\Documents\projects\hierarchical-video-prediction-path-b` | `codex/hierarchy-allocation-v2` |

Datasets are not copied into either worktree. Commands use the existing OpenSTL data folder. Run
output is configured on D:, which had 74.9 GiB free on 2 September; C: had 33.1 GiB.

## 2. Implemented code

### OpenSTL branch

- `openstl/models/hvp_integrated_model.py`: pyramid, real space--time packer, coordinate positions,
  reversible/vanilla parity core, future queries, and full-frame decoder.
- `openstl/methods/hvp_integrated.py`: Lightning method, train/validation/routing metrics, gradient
  norm, validation panels, and streaming locked-test aggregation.
- `configs/mmnist/HVPIntegrated.py`, `configs/taxibj/HVPIntegrated.py`: provisional gate configs;
  capacity and long-run epochs remain unlocked.
- optimized Moving MNIST and TaxiBJ loaders plus
  `tools/prepare_data/prepare_hvp_validation_and_cache.py`.
- model/data contract tests under `tests/`.

### HVP integration branch

- `experiments/v2/campaigns/integrated_openstl_confirmation_2026-09-02.json`: exact four-run queue,
  currently prelock and unauthorized.
- `scripts/v2/run_integrated_openstl_campaign.py`: preflight/status/sequential controller.
- `scripts/v2/train_integrated_openstl.py`: resumable worker with local evidence and W&B mirror.
- `scripts/v2/profile_integrated_openstl_memory.py`: paired same-checkpoint component profiler.
- `tests/test_integrated_openstl_worker.py`: complete CPU train/validate/test/restart test.

## 3. Architecture data flow

```text
OpenSTL frames [B,T,C,H,W]
  -> shared CNN pyramid
       fine candidates [B,T,D,H/4,W/4]
       coarse coverage [B,T,D,H/8,W/8]
  -> fine score = coarse residual + temporal change + learned correction
  -> gather fixed K across space and time (non-reversible boundary)
  -> add physical x/y/time/scale/level encoding
  -> internally reversible additive-coupling transformer
  -> future coarse-grid queries cross-attend to packed context
  -> three-stage convolutional upsampler
  -> residual over last input frame [B,T_out,C,H,W]
```

Moving MNIST has 2,560 fine candidates. The provisional hierarchy retains 640 coarse plus 160
detail tokens, or 800 total: 3.2x fewer retained tokens. TaxiBJ has 256 candidates and retains 64
plus 16, also 3.2x. This is an analytic token ratio, not a CUDA peak-memory result.

The provisional width-128/depth-6 model has 1,912,386 parameters on Moving MNIST and 1,912,963 on
TaxiBJ. The local OpenSTL SimVP configs instantiate to 46,769,665 and 9,961,058 parameters.
Capacity matching is therefore a gate decision: external memory tables must separate parameter
memory from activations/context, while hierarchy ablations use identical integrated parameters.

## 4. Inference-time dynamic routing

`model.set_routing_mode(mode)` and the forward `routing_mode=` override accept:

- `static`: content-independent stratified membership;
- `density`: deterministic coarse-residual plus temporal-change scores;
- `hybrid`: density/change plus a learned correction;
- `mixed`: static/hybrid route augmentation in training, hybrid in evaluation; and
- `dense`: all fine candidates for controls.

Physical metadata makes switching structurally valid; it does not guarantee quality for an unseen
policy. Final training therefore uses `mixed`. Enabling dynamic routing after static-only training
is prohibited as a claim unless paired validation demonstrates non-inferiority.

At fixed K, static and dynamic retain the same bytes. Dynamic routing succeeds if it improves
quality/stability at that K or permits a smaller K at the same quality. It is not an extra memory
multiplier.

## 5. CPU verification

Run with CUDA hidden and one math thread:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'

Set-Location 'C:\Users\leoni\Documents\projects\OpenSTL-integrated'
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  -m pytest tests\test_models\test_hvp_integrated.py tests\test_hvp_data_contracts.py -q

Set-Location 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration'
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  -m pytest tests\test_integrated_openstl_worker.py -q
```

Verified on 2 September:

- 18 OpenSTL model/data tests passed in 7.01 seconds;
- five controller/worker tests passed in 9.82 seconds, including the complete restart path;
- no CUDA device was visible;
- reversible/vanilla output, input-gradient, and parameter-gradient parity passes in fp32 and CPU
  bf16 autocast tolerance;
- a capped segment resumes and reaches the exact original optimizer-update total; and
- panels, rolling/best checkpoints, JSONL evidence, and streaming test metrics are written.

These tests do not establish predictive competitiveness or CUDA-memory reduction.

## 6. Prepare validation/cache artifacts after Article 4

The source audit found `mnist_test_seq.npy` at `(20,10000,64,64)`, uint8, 819,200,096 bytes. TaxiBJ
is a 2,747,401,294-byte NPZ with two 1,340,932,096-byte float64 training arrays plus two
32,768,000-byte float64 test arrays. Do not run conversion while Article 4 needs host RAM/I/O.

After it finishes:

```powershell
Set-Location 'C:\Users\leoni\Documents\projects\OpenSTL-integrated'
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  tools\prepare_data\prepare_hvp_validation_and_cache.py `
  --data-root 'C:\Users\leoni\Documents\projects\OpenSTL\data' `
  --prepare both
```

The command refuses overwrites. Moving MNIST produces 2,000 deterministic validation clips at seed
260902 and a hash manifest. TaxiBJ produces four float32 memmaps and a source/output hash manifest.
Publication configs fail closed if artifacts are missing; they never reuse test as validation.
TaxiBJ validation is the newest 10% of training, separated by an eight-sample excluded gap.

## 7. Preflight and status

The campaign is intentionally unauthorized until CUDA gates and Leonid's confirmation.

```powershell
Set-Location 'C:\Users\leoni\Documents\projects\hierarchical-video-prediction-integration'

& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\run_integrated_openstl_campaign.py status

& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\run_integrated_openstl_campaign.py preflight `
  --allow-missing-prepared-data
```

After preparation, omit `--allow-missing-prepared-data`, then add `--check-wandb` and
`--full-data-hash`. Locked preflight checks clean repositories, configs, exact dataset/seed cells,
distinct validation/cache artifacts, D: reserve, CUDA availability, and W&B authentication. It
produces a lock fingerprint over the manifest, both repository commits, every config, and every
prepared/source data file. The authorization ledger must contain both `manifest_sha256` and this
`lock_fingerprint`; `run` recomputes all hashes and refuses a mismatch.

## 8. Gate and launch sequence

Do not authorize the four long runs directly. After Article 4:

1. overfit 32--64 clips on each dataset;
2. measure 200 updates per dataset to select batch/capacity/precision and estimate GPU hours;
3. run same-checkpoint vanilla/reversible and flat/hierarchical profiles;
4. compare static/density/hybrid routing on Moving MNIST validation;
5. run a 5--10 epoch two-dataset screen;
6. freeze configs, commits, metric, update count, and queue duration; and
7. obtain Leonid's confirmation before setting `launch_authorized: true` and writing the matching
   authorization ledger.

Then one command launches or resumes all four:

```powershell
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\run_integrated_openstl_campaign.py run `
  --authorization experiments\v2\campaigns\integrated_openstl_confirmation.authorization.json
```

Completed results are skipped. An interruption resumes `checkpoints/last.ckpt`. Total optimizer
updates, rather than only epoch count, enforce schedule completion after restart.

## 9. W&B and local evidence

Project: `hierarchical-video-prediction-integrated-v2`  
Group: `integrated-openstl-confirmation-2026-09`

The project is created by the first authorized online run. Local evidence is canonical: resolved
config; provenance; stdout/stderr; JSONL; status/failure/result; train/validation loss, frame MSE,
pixel MAE, LR, gradient norm, timing and CUDA memory; route entropy, coverage, churn and margin;
panels and indices; rolling/best checkpoints; and streaming test MSE/MAE/PSNR. W&B uses deterministic
IDs and resume, mirrors scalars/panels, and is disabled after a caught failure without losing local
evidence.

## 10. Paired memory profile

After a checkpoint exists:

```powershell
& 'C:\Users\leoni\Documents\projects\OpenSTL\venv\Scripts\python.exe' `
  scripts\v2\profile_integrated_openstl_memory.py `
  --openstl 'C:\Users\leoni\Documents\projects\OpenSTL-integrated' `
  --config 'C:\Users\leoni\Documents\projects\OpenSTL-integrated\configs\mmnist\HVPIntegrated.py' `
  --checkpoint 'D:\hierarchical-video-prediction-storage\integrated_openstl_v2\INT_MM_S00\checkpoints\best.ckpt' `
  --dataset mmnist `
  --output 'D:\hierarchical-video-prediction-storage\integrated_openstl_v2\profiles\INT_MM_S00.json'
```

It reloads identical weights for flat/vanilla, hierarchical/static/vanilla,
hierarchical/dynamic/vanilla, hierarchical/static/reversible, and full dynamic/reversible. It also
profiles the core alone and verifies forward/input-gradient parity. Defaults are five warmups and
twenty measured optimizer steps; OOM is recorded rather than hidden.

## 11. Claim boundary

No current evidence guarantees OpenSTL-level quality. The architecture is structurally capable of
the test, but only CUDA gates determine capacity, speed, peak memory, routing stability, and
convergence. If quality exceeds the registered 5% memory-focused degradation tolerance, adjust
capacity/quota or narrow the claim; never substitute analytic token counts for measured end-to-end
superiority.
