param(
    [ValidateSet("Prepare", "Run", "All")][string]$Mode = "All",
    [string]$Config = "experiments/configs/phase_c_v2.json",
    [string]$Log = "logs_rhcp/phasec_v2.log",
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$root = (Resolve-Path ".").Path
$ledger = "artifacts/phasec_v2/gpu_hours.jsonl"
$accessLog = "artifacts/phasec_v2/access.jsonl"
$maxHours = [double]$configValue.budget.authorized_max_gpu_hours
$sourceCommit = (git rev-parse HEAD).Trim()
$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "E6 controller requires a clean worktree: $($dirty -join '; ')" }
$cpuReport = Get-Content gates\phase_c_cpu_validation_report.json -Raw | ConvertFrom-Json
foreach ($property in $cpuReport.implementation_hashes.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $property.Name).Hash.ToLowerInvariant()
    if ($actual -ne $property.Value) {
        throw "E6 implementation changed after CPU validation: $($property.Name)"
    }
}

$processSnapshot = @(Get-CimInstance Win32_Process)
$ancestorPids = [System.Collections.Generic.HashSet[int]]::new()
$cursorPid = $PID
while ($true) {
    [void]$ancestorPids.Add([int]$cursorPid)
    $cursor = $processSnapshot | Where-Object { $_.ProcessId -eq $cursorPid } | Select-Object -First 1
    if ($null -eq $cursor -or $cursor.ParentProcessId -le 0 -or $ancestorPids.Contains([int]$cursor.ParentProcessId)) {
        break
    }
    $cursorPid = [int]$cursor.ParentProcessId
}
$projectProcesses = @($processSnapshot | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match 'run_phasec_v2|phasec_eval_v2|train_phasec_classifier|build_phasec_classifier_domains|validate_phasec_evaluator|hier_convae_v2|rhvd_train_v2|hvp_rev\.rhvd' -and
    -not $ancestorPids.Contains([int]$_.ProcessId)
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/trainer/evaluator is active" }

New-Item -ItemType Directory -Force -Path artifacts\phasec_v2 | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
$script:consumed = 0.0
if (Test-Path $ledger) {
    foreach ($line in Get-Content $ledger) {
        if ($line.Trim()) { $script:consumed += [double](($line | ConvertFrom-Json).gpu_hours) }
    }
}

function Invoke-PhaseCGpuCell {
    param(
        [string]$Phase,
        [string]$Arm,
        [int]$Seed,
        [double]$Limit,
        [double]$Expected,
        [scriptblock]$Command
    )
    if ($script:consumed + $Limit -gt $maxHours + 1e-9) {
        throw "E6 $Phase/$Arm/s$Seed cannot fit the remaining 40-hour programme budget"
    }
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -and
        $_.CommandLine -match 'phasec_eval_v2|train_phasec_classifier|build_phasec_classifier_domains|validate_phasec_evaluator|hier_convae_v2|rhvd_train_v2|hvp_rev\.rhvd' -and
        $_.ProcessId -ne $PID
    })
    if ($active.Count -ne 0) { throw "Another project GPU process appeared before $Phase/$Arm" }
    $started = Get-Date
    "=== E6 $Phase arm=$Arm seed=$Seed START $($started.ToString('o')) ===" |
        Tee-Object -FilePath $Log -Append
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $code = 0
    try {
        & $Command 2>&1 | Tee-Object -FilePath $Log -Append
        $code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    } catch {
        $_ | Out-String | Tee-Object -FilePath $Log -Append | Out-Null
        $code = 1
    }
    $ErrorActionPreference = $savedPreference
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $script:consumed += $hours
    [ordered]@{
        schema_version = 1; study = "phase-c-persistent-object-v2"; phase = $Phase;
        arm = $Arm; seed = $Seed;
        started = $started.ToUniversalTime().ToString("o");
        ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours;
        expected_gpu_hours = $Expected; cell_authorized_max_gpu_hours = $Limit;
        programme_authorized_max_gpu_hours = $maxHours; exit_code = $code;
        source_commit = $sourceCommit; resumed = [bool]$Resume
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E6 $Phase arm=$Arm seed=$Seed END code=$code hours=$($hours.ToString('F3')) total=$($script:consumed.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E6 cell failed: $Phase/$Arm/s$Seed" }
    if ($hours -gt $Limit) { throw "E6 cell exceeded its authorization: $Phase/$Arm/s$Seed" }
    if ($script:consumed -gt $maxHours) { throw "E6 exceeded the 40-hour programme maximum" }
}

function Assert-SourceFiles {
    foreach ($property in $configValue.official_mnist.PSObject.Properties) {
        if ($property.Name -eq "mirror") { continue }
        if (-not (Test-Path -LiteralPath $property.Value)) {
            throw "Official MNIST source is missing: $($property.Value)"
        }
    }
}

function Generate-DataIfMissing {
    param([object]$Dataset)
    $path = [string]$Dataset.path
    $manifest = [System.IO.Path]::ChangeExtension($path, ".manifest.json")
    if (Test-Path -LiteralPath $path) {
        if (-not $Resume -or -not (Test-Path -LiteralPath $manifest)) {
            throw "Existing E6 dataset requires explicit -Resume and a manifest: $path"
        }
        return
    }
    $images = if ($Dataset.source_split -eq "train") {
        $configValue.official_mnist.train_images
    } else { $configValue.official_mnist.test_images }
    $labels = if ($Dataset.source_split -eq "train") {
        $configValue.official_mnist.train_labels
    } else { $configValue.official_mnist.test_labels }
    $env:CUDA_VISIBLE_DEVICES = "-1"
    & .\venv\Scripts\python.exe scripts\gen_moving_v2.py `
        --images $images --labels $labels --source-split $Dataset.source_split `
        --purpose $Dataset.purpose --n $Dataset.clips --frames $Dataset.frames `
        --objects $configValue.datasets.common.objects --size $configValue.datasets.common.image_size `
        --step $configValue.datasets.common.step --seed $Dataset.seed `
        --generator-commit $sourceCommit --out $path
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Find-SingleFile {
    param([string]$Root, [string]$Name, [string]$Complete = "COMPLETE")
    $matches = @()
    if (Test-Path $Root) {
        foreach ($item in Get-ChildItem $Root -Recurse -Filter $Name) {
            if (Test-Path (Join-Path $item.Directory.FullName $Complete)) { $matches += $item.FullName }
        }
    }
    if ($matches.Count -ne 1) { throw "Expected exactly one completed $Name below $Root; found $($matches.Count)" }
    return $matches[0]
}

if ($Mode -in @("Prepare", "All")) {
    & .\venv\Scripts\python.exe scripts\check_experiment_gate.py phase-c-v2-preparation
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Assert-SourceFiles
    $prepGroup = "phasec-v2-preparation-$($sourceCommit.Substring(0,12))"
    & .\venv\Scripts\python.exe scripts\check_wandb_online.py `
        --project $configValue.evaluation.wandb_project --group $prepGroup `
        --run-id "phasec-v2-prep-gate-$($sourceCommit.Substring(0,12))-$(Get-Date -Format yyyyMMddHHmmss)"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($CheckOnly) {
        Write-Output "Phase C v2 preparation gate passed; no data or GPU process launched."
        exit 0
    }
    Generate-DataIfMissing $configValue.datasets.training
    Generate-DataIfMissing $configValue.datasets.development
    $trainManifest = [System.IO.Path]::ChangeExtension($configValue.datasets.training.path, ".manifest.json")
    $devManifest = [System.IO.Path]::ChangeExtension($configValue.datasets.development.path, ".manifest.json")
    if (-not (Test-Path gates\phase_c_train_development_validated.json)) {
        & .\venv\Scripts\python.exe scripts\validate_phasec_datasets.py `
            --config $Config --training-manifest $trainManifest --development-manifest $devManifest `
            --source-commit $sourceCommit --out gates\phase_c_train_development_report.json `
            --marker gates\phase_c_train_development_validated.json
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    $codecRoot = "$($configValue.codec.artifact_root)/phasec-v2-codec/seed-$($configValue.codec.seed)"
    $codecMatches = @()
    if (Test-Path $codecRoot) {
        $codecMatches = @(Get-ChildItem $codecRoot -Recurse -Filter checkpoint.pt | Where-Object {
            Test-Path (Join-Path $_.Directory.FullName "COMPLETE")
        })
    }
    if ($codecMatches.Count -eq 0) {
        Invoke-PhaseCGpuCell -Phase "codec-training" -Arm "hier-convae" -Seed $configValue.codec.seed `
            -Limit $configValue.codec.max_gpu_hours -Expected 0.45 -Command {
                & .\venv\Scripts\python.exe -u -m hvp_rev.hier_convae_v2 `
                    --study phasec-v2-codec --seed $configValue.codec.seed `
                    --data $configValue.datasets.training.path --steps $configValue.codec.steps `
                    --batch $configValue.codec.batch --lr $configValue.codec.lr `
                    --artifact_root $configValue.codec.artifact_root `
                    --source_commit $sourceCommit --provenance_tag phasec-v2-dataset-codec `
                    --dataset_tag moving-mnist --allow_seed_zero `
                    --wandb_project $configValue.evaluation.wandb_project --wandb_group $prepGroup `
                    --run_log $Log --authorized_gpu_hours $configValue.codec.max_gpu_hours
            }
    } elseif (-not $Resume -or $codecMatches.Count -ne 1) {
        throw "Completed E6 codec exists; use explicit -Resume (or audit duplicates)"
    }
    $codec = Find-SingleFile $codecRoot "checkpoint.pt"

    $classifierRoot = "artifacts/phasec_v2/classifier-v3"
    $classifierMatches = @()
    if (Test-Path $classifierRoot) {
        $classifierMatches = @(Get-ChildItem $classifierRoot -Recurse -Filter checkpoint.pt | Where-Object {
            Test-Path (Join-Path $_.Directory.FullName "COMPLETE")
        })
    }
    if ($classifierMatches.Count -eq 0) {
        Invoke-PhaseCGpuCell -Phase "classifier-training" -Arm "digit-classifier" `
            -Seed $configValue.classifier.seed -Limit $configValue.classifier.max_gpu_hours `
            -Expected 0.08 -Command {
                & .\venv\Scripts\python.exe -u scripts\train_phasec_classifier.py `
                    --images $configValue.official_mnist.train_images `
                    --labels $configValue.official_mnist.train_labels `
                    --seed $configValue.classifier.seed --steps $configValue.classifier.steps `
                    --batch $configValue.classifier.batch --lr $configValue.classifier.lr `
                    --source-commit $sourceCommit --wandb-project $configValue.evaluation.wandb_project `
                    --wandb-group $prepGroup `
                    --artifact-root $classifierRoot `
                    --authorized-gpu-hours $configValue.classifier.max_gpu_hours
            }
    } elseif (-not $Resume -or $classifierMatches.Count -ne 1) {
        throw "Completed E6 classifier exists; use explicit -Resume (or audit duplicates)"
    }
    $classifier = Find-SingleFile $classifierRoot "checkpoint.pt"

    if (-not (Test-Path gates\phase_c_evaluator_validated.json)) {
        Invoke-PhaseCGpuCell -Phase "metric-validation" -Arm "classifier+evaluator" -Seed 0 `
            -Limit $configValue.validation.evaluator_reliability_max_gpu_hours -Expected 0.12 -Command {
                & .\venv\Scripts\python.exe scripts\build_phasec_classifier_domains.py `
                    --test-images $configValue.official_mnist.test_images `
                    --test-labels $configValue.official_mnist.test_labels `
                    --development $configValue.datasets.development.path `
                    --development-metadata ([System.IO.Path]::ChangeExtension($configValue.datasets.development.path, ".metadata.npz")) `
                    --ae $codec --seed 20260718 --limit 2000 `
                    --out artifacts\phasec_v2\classifier_domains_v4.npz `
                    --report gates\phase_c_classifier_domains_report_v4.json
                if ($LASTEXITCODE -ne 0) { throw "classifier-domain construction failed" }
                & .\venv\Scripts\python.exe scripts\validate_phasec_classifier.py `
                    --checkpoint $classifier --domains artifacts\phasec_v2\classifier_domains_v4.npz `
                    --out gates\phase_c_classifier_report_v4.json --write-marker `
                    --source-commit $sourceCommit --wandb-project $configValue.evaluation.wandb_project `
                    --wandb-group $prepGroup
                if ($LASTEXITCODE -ne 0) { throw "classifier validation failed" }
                & .\venv\Scripts\python.exe scripts\validate_phasec_evaluator.py `
                    --development $configValue.datasets.development.path `
                    --metadata ([System.IO.Path]::ChangeExtension($configValue.datasets.development.path, ".metadata.npz")) `
                    --ae $codec --classifier $classifier --config $Config `
                    --out gates\phase_c_evaluator_report_v3.json `
                    --marker gates\phase_c_evaluator_validated.json `
                    --source-commit $sourceCommit --wandb-project $configValue.evaluation.wandb_project `
                    --wandb-group $prepGroup
            }
    }
    [ordered]@{
        schema_version = 1; status = "PHASEC_V2_PREPARATION_COMPLETE";
        source_commit = $sourceCommit; codec = $codec;
        codec_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $codec).Hash.ToLowerInvariant();
        classifier = $classifier;
        classifier_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $classifier).Hash.ToLowerInvariant();
        train_manifest = $trainManifest; development_manifest = $devManifest;
        total_gpu_hours = $script:consumed
    } | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 gates\phase_c_preparation_complete.json
    "=== PHASEC_V2_PREPARATION_DONE total_gpu_hours=$($script:consumed.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($Mode -eq "Prepare") { exit 0 }
    throw "All-mode intentionally pauses after preparation so its hash-bound gates can be reviewed and committed."
}

if ($Mode -eq "Run") {
    if ($configValue.status -ne "FROZEN") { throw "E6 scientific run requires status FROZEN" }
    & .\venv\Scripts\python.exe scripts\check_experiment_gate.py phase-c-v2
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $resolved = $configValue.resolved_artifacts
    foreach ($path in @($resolved.codec_path, $resolved.classifier_path, $resolved.holdout_manifest)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Frozen E6 artifact is missing: $path" }
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $resolved.codec_path).Hash.ToLowerInvariant() -ne
        $resolved.codec_sha256) { throw "Frozen E6 codec hash changed" }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $resolved.classifier_path).Hash.ToLowerInvariant() -ne
        $resolved.classifier_sha256) { throw "Frozen E6 classifier hash changed" }
    $holdout = Get-Content -LiteralPath $resolved.holdout_manifest -Raw | ConvertFrom-Json
    $group = "$($configValue.evaluation.wandb_group_prefix)-$($holdout.dataset_id)"
    & .\venv\Scripts\python.exe scripts\check_wandb_online.py `
        --project $configValue.evaluation.wandb_project --group $group `
        --run-id "phasec-v2-run-gate-$($sourceCommit.Substring(0,12))-$(Get-Date -Format yyyyMMddHHmmss)"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($CheckOnly) { Write-Output "Phase C v2 protected run gate passed; no GPU process launched."; exit 0 }

    foreach ($seed in $configValue.training.seeds) {
        foreach ($arm in $configValue.training.arms) {
            $complete = @()
            $partial = @()
            $base = "artifacts/training/phase-c-v2-predictors/$arm/seed-$seed"
            if (Test-Path $base) {
                foreach ($checkpoint in Get-ChildItem $base -Recurse -Filter checkpoint.pt) {
                    if (Test-Path (Join-Path $checkpoint.Directory.FullName "COMPLETE")) {
                        $complete += $checkpoint.FullName
                    } else { $partial += $checkpoint.FullName }
                }
            }
            if ($complete.Count -gt 1 -or $partial.Count -gt 1) {
                throw "Duplicate E6 predictor artifacts for $arm seed $seed"
            }
            if ($complete.Count -eq 1) {
                if (-not $Resume) { throw "Completed predictor exists; use -Resume: $arm s$seed" }
                continue
            }
            $resumeArgs = @()
            if ($partial.Count -eq 1) {
                if (-not $Resume) { throw "Partial predictor needs -Resume: $arm s$seed" }
                $resumeArgs = @("--resume")
            }
            Invoke-PhaseCGpuCell -Phase "predictor-training" -Arm $arm -Seed $seed `
                -Limit $configValue.training.per_cell_max_gpu_hours -Expected 3.45 -Command {
                    & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd_train_v2 `
                        --arm $arm --study phase-c-v2-predictors --seed $seed `
                        --steps $configValue.training.steps --batch $configValue.training.batch `
                        --lr $configValue.training.lr --warmup $configValue.training.warmup_updates `
                        --bf16 --train_only --train_path $configValue.datasets.training.path `
                        --test_path $configValue.datasets.development.path --world 64 --wspeed 0 `
                        --warm $configValue.evaluation.warmup_frames --n_val 250 `
                        --ae_path $resolved.codec_path --source_commit $sourceCommit `
                        --wandb_project $configValue.evaluation.wandb_project --wandb_group $group `
                        --dataset_tag moving-mnist --provenance_tag phase-c-v2-persistent-object `
                        --run_log $Log --authorized_gpu_hours $configValue.training.per_cell_max_gpu_hours `
                        @resumeArgs
                }
        }
    }

    foreach ($seed in $configValue.training.seeds) {
        foreach ($arm in $configValue.training.arms) {
            $checkpoint = Find-SingleFile "artifacts/training/phase-c-v2-predictors/$arm/seed-$seed" "checkpoint.pt"
            $evalComplete = @()
            $evalBase = "artifacts/runs/phase-c-v2-evaluation/$arm/seed-$seed"
            if (Test-Path $evalBase) {
                $evalComplete = @(Get-ChildItem $evalBase -Recurse -Filter COMPLETE)
            }
            if ($evalComplete.Count -gt 1) { throw "Duplicate E6 evaluations for $arm seed $seed" }
            if ($evalComplete.Count -eq 1) {
                if (-not $Resume) { throw "Completed evaluation exists; use -Resume: $arm s$seed" }
                continue
            }
            Invoke-PhaseCGpuCell -Phase "protected-evaluation" -Arm $arm -Seed $seed `
                -Limit $configValue.evaluation.per_cell_max_gpu_hours -Expected 0.75 -Command {
                    & .\venv\Scripts\python.exe -u -m hvp_rev.phasec_eval_v2 `
                        --arm $arm --seed $seed --train-path $configValue.datasets.training.path `
                        --holdout-manifest $resolved.holdout_manifest --ae $resolved.codec_path `
                        --classifier $resolved.classifier_path --checkpoint $checkpoint `
                        --source-commit $sourceCommit --access-log $accessLog `
                        --wandb-project $configValue.evaluation.wandb_project --wandb-group $group `
                        --nfe $configValue.evaluation.nfe `
                        --sampler-seed $configValue.evaluation.sampler_seed `
                        --warm $configValue.evaluation.warmup_frames `
                        --eval-start $configValue.evaluation.holdout_clip_start `
                        --n-eval-clips $configValue.evaluation.holdout_clips `
                        --chunk $configValue.evaluation.chunk `
                        --detection-threshold $configValue.evaluation.object_evaluator.detection_threshold `
                        --min-component-pixels $configValue.evaluation.object_evaluator.min_component_pixels `
                        --confidence-threshold $configValue.evaluation.object_evaluator.confidence_threshold `
                        --distance-weight $configValue.evaluation.object_evaluator.distance_weight `
                        --iou-weight $configValue.evaluation.object_evaluator.iou_weight `
                        --max-assignment-cost $configValue.evaluation.object_evaluator.max_assignment_cost `
                        --authorized-gpu-hours $configValue.evaluation.per_cell_max_gpu_hours
                }
        }
    }
    $env:CUDA_VISIBLE_DEVICES = "-1"
    & .\venv\Scripts\python.exe scripts\analyze_phasec_v2.py `
        --config $Config --manifest $resolved.holdout_manifest --run-root artifacts\runs `
        --out-json artifacts\phasec_v2\results.json `
        --out-md docs\confirmation\E6_PHASEC_V2_RESULTS.md `
        --wandb-project $configValue.evaluation.wandb_project --wandb-group $group
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .\venv\Scripts\python.exe scripts\audit_wandb_group.py `
        --project $configValue.evaluation.wandb_project --group $group --min-runs 14 `
        --min-job-type scientific-training=6 `
        --min-job-type phasec-v2-evaluation=6 `
        --min-job-type phasec-v2-analysis=1 `
        --require-artifact-job scientific-training `
        --require-artifact-job phasec-v2-evaluation `
        --require-artifact-job phasec-v2-analysis `
        --out artifacts\phasec_v2\wandb_validation.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .\venv\Scripts\python.exe scripts\validate_phasec_completion.py `
        --config $Config --manifest $resolved.holdout_manifest `
        --results artifacts\phasec_v2\results.json --gpu-ledger $ledger `
        --access-log $accessLog --run-root artifacts\runs --training-root artifacts\training `
        --wandb-audit artifacts\phasec_v2\wandb_validation.json `
        --out artifacts\phasec_v2\completion_validation.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    "=== PHASEC_V2_DONE total_gpu_hours=$($script:consumed.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
}
