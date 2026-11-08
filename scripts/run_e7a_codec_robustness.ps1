param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Config = "experiments/configs/alternate_codec_robustness.json",
    [string]$Log = "logs_rhcp/e7a_codec_robustness.log",
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "-1"
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$manifestValue = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E7a config is not frozen" }
if ($manifestValue.purpose -ne "codec_robustness_holdout") { throw "Wrong E7a dataset purpose" }
$testPath = $manifestValue.artifacts.clips.path
$group = "$($configValue.evaluation.wandb_group_prefix)-$($manifestValue.dataset_id)"
$maxHours = [double]$configValue.budget.authorized_max_gpu_hours
$ledger = "artifacts/e7a_codec_robustness/gpu_hours.jsonl"
$accessLog = "artifacts/e7a_codec_robustness/access.jsonl"
$floorReport = "gates/e7a_codec_floor_report.json"
$dispatchReport = "gates/e7a_evaluation_dispatch.json"
$executionRecord = "artifacts/e7a_codec_robustness/execution.json"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$currentCommit = (git rev-parse HEAD).Trim()
if ($Resume) {
    if (-not (Test-Path $executionRecord)) { throw "E7a resume requires its pinned execution record" }
    $execution = Get-Content $executionRecord -Raw | ConvertFrom-Json
    $sourceCommit = $execution.source_commit
    if ($execution.config_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant()) {
        throw "Frozen E7a config changed since the original launch"
    }
} else {
    if (Test-Path $executionRecord) { throw "E7a execution record exists; use explicit -Resume" }
    $sourceCommit = $currentCommit
}
$cpuReport = Get-Content gates/e7a_cpu_validation_report.json -Raw | ConvertFrom-Json
foreach ($property in $cpuReport.implementation_hashes.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $property.Name).Hash.ToLowerInvariant()
    if ($actual -ne $property.Value) { throw "E7a implementation changed after CPU validation: $($property.Name)" }
}
$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'hier_convae_v2|rhvd_train_v2|hvp_rev\.rhvd|run_e7a_codec' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/trainer/evaluator is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py alternate-autoencoder-robustness
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Resume) {
    & .\venv\Scripts\python.exe scripts\validate_e7a_design.py `
        --config $Config --manifest $Manifest --out gates/e7a_design_validation.json --allow-existing
} else {
    & .\venv\Scripts\python.exe scripts\validate_e7a_design.py `
        --config $Config --manifest $Manifest --out gates/e7a_design_validation.json
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wandbGateId = "e7a-gate-$($manifestValue.dataset_id)-$($sourceCommit.Substring(0,8))-$(Get-Date -Format yyyyMMddHHmmss)"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.evaluation.wandb_project --group $group --run-id $wandbGateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E7a prelaunch gates PASS; no GPU process launched."; exit 0 }

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts\e7a_codec_robustness | Out-Null
if (-not $Resume) {
    [ordered]@{
        schema_version = 1; study = "robustness-e7a"; source_commit = $sourceCommit;
        launched_from_commit = $currentCommit; config_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant();
        dataset_id = $manifestValue.dataset_id; created_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $executionRecord
}
$script:consumed = 0.0
if (Test-Path $ledger) {
    foreach ($line in Get-Content $ledger) {
        if ($line.Trim()) { $script:consumed += [double](($line | ConvertFrom-Json).gpu_hours) }
    }
}

function Invoke-E7aGpuCell {
    param(
        [string]$Phase,
        [string]$Arm,
        [string]$Codec,
        [double]$Limit,
        [double]$Expected,
        [scriptblock]$Command
    )
    if ($script:consumed + $Limit -gt $maxHours + 1e-9) {
        throw "E7a $Phase/$Codec/$Arm cannot fit its frozen allocation in the remaining GPU budget"
    }
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'hier_convae_v2|rhvd_train_v2|hvp_rev\.rhvd' -and
        $_.ProcessId -ne $PID
    })
    if ($active.Count -ne 0) { throw "Another project GPU process appeared before $Phase/$Arm" }
    $started = Get-Date
    "=== E7a $Phase codec=$Codec arm=$Arm START $($started.ToString('o')) ===" |
        Tee-Object -FilePath $Log -Append
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Command 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $script:consumed += $hours
    [ordered]@{
        schema_version = 1; study = "robustness-e7a"; phase = $Phase; arm = $Arm; codec = $Codec;
        started = $started.ToUniversalTime().ToString("o");
        ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours;
        expected_gpu_hours = $Expected; cell_authorized_max_gpu_hours = $Limit;
        tranche_authorized_max_gpu_hours = $maxHours; exit_code = $code;
        source_commit = $sourceCommit; dataset_id = $manifestValue.dataset_id; resumed = [bool]$Resume
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E7a $Phase codec=$Codec arm=$Arm END code=$code hours=$($hours.ToString('F3')) total=$($script:consumed.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E7a cell failed: $Phase/$Codec/$Arm" }
    if ($hours -gt $Limit) { throw "E7a cell exceeded its frozen allocation: $Phase/$Codec/$Arm" }
    if ($script:consumed -gt $maxHours) { throw "E7a exceeded its frozen 36-hour maximum" }
}

function Find-AlternateAe {
    $matches = @()
    $base = "artifacts/autoencoders/e7a-alternate-codec/seed-$($configValue.codecs.alternate.seed)"
    if (Test-Path $base) {
        foreach ($specPath in Get-ChildItem $base -Recurse -Filter run_spec.json) {
            $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
            if ($spec.training_data_sha256 -eq $configValue.codecs.alternate.training_data_sha256 -and
                [int]$spec.seed -eq [int]$configValue.codecs.alternate.seed -and
                [int]$spec.steps -eq [int]$configValue.codecs.alternate.steps -and
                (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
                $matches += (Join-Path $specPath.Directory.FullName "checkpoint.pt")
            }
        }
    }
    if ($matches.Count -gt 1) { throw "Duplicate completed alternate codec artifacts" }
    return $matches
}

$altAeMatches = @(Find-AlternateAe)
if ($altAeMatches.Count -eq 1) {
    if (-not $Resume) { throw "Completed alternate codec exists; explicit -Resume is required" }
    $altAe = $altAeMatches[0]
    "=== E7a alternate codec already COMPLETE; skipping ===" | Tee-Object -FilePath $Log -Append
} else {
    $aeLimit = [double]$configValue.codecs.alternate.max_gpu_hours
    $ae = $configValue.codecs.alternate
    Invoke-E7aGpuCell -Phase "codec-training" -Arm "hier-convae" -Codec "alternate" `
        -Limit $aeLimit -Expected 0.45 -Command {
            & .\venv\Scripts\python.exe -u -m hvp_rev.hier_convae_v2 `
                --study e7a-alternate-codec --seed $ae.seed --data $ae.training_data `
                --steps $ae.steps --batch $ae.batch --lr $ae.lr `
                --source_commit $sourceCommit --wandb_project $configValue.evaluation.wandb_project `
                --wandb_group $group --run_log $Log --authorized_gpu_hours $aeLimit
        }
    $altAeMatches = @(Find-AlternateAe)
    if ($altAeMatches.Count -ne 1) { throw "Alternate codec artifact not found after successful training" }
    $altAe = $altAeMatches[0]
}
$altAeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $altAe).Hash.ToLowerInvariant()

if (Test-Path $floorReport) {
    if (-not $Resume) { throw "E7a codec floor exists; explicit -Resume is required" }
    $floor = Get-Content $floorReport -Raw | ConvertFrom-Json
    if ($floor.status -ne "CODEC_FLOOR_PASS" -or $floor.alternate_codec_sha256 -ne $altAeHash) {
        throw "Existing E7a codec floor is not a valid pass for this alternate codec"
    }
    "=== E7a codec floor already PASS; skipping ===" | Tee-Object -FilePath $Log -Append
} else {
    $codecSpent = 0.0
    if (Test-Path $ledger) {
        foreach ($line in Get-Content $ledger) {
            if ($line.Trim()) {
                $row = $line | ConvertFrom-Json
                if ($row.phase -in @("codec-training", "codec-floor")) {
                    $codecSpent += [double]$row.gpu_hours
                }
            }
        }
    }
    $floorLimit = [double]$configValue.budget.codec_training_max_gpu_hours - $codecSpent
    if ($floorLimit -le 0) { throw "Codec training consumed the shared codec/floor allocation" }
    Invoke-E7aGpuCell -Phase "codec-floor" -Arm "hier-convae" -Codec "both" `
        -Limit $floorLimit -Expected 0.03 -Command {
            & .\venv\Scripts\python.exe -u scripts\evaluate_e7a_codec_floor.py `
                --config $Config --original $configValue.codecs.original.path --alternate $altAe `
                --out $floorReport --wandb-project $configValue.evaluation.wandb_project `
                --wandb-group $group
        }
}

function Find-AlternatePredictor([string]$Arm) {
    $matches = @()
    $base = "artifacts/training/e7a-alternate-codec-predictors/$Arm/seed-0"
    if (Test-Path $base) {
        foreach ($specPath in Get-ChildItem $base -Recurse -Filter run_spec.json) {
            $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
            if ($spec.ae_hash -eq $altAeHash -and [int]$spec.optimizer_updates -eq 30000 -and
                (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
                $matches += (Join-Path $specPath.Directory.FullName "checkpoint.pt")
            }
        }
    }
    if ($matches.Count -gt 1) { throw "Duplicate completed alternate predictor for $Arm" }
    return $matches
}

foreach ($run in $configValue.predictor_runs) {
    $matches = @(Find-AlternatePredictor $run.arm)
    if ($matches.Count -eq 1) {
        if (-not $Resume) { throw "Completed alternate predictor exists; explicit -Resume is required: $($run.arm)" }
        "=== E7a predictor $($run.arm) already COMPLETE; skipping ===" | Tee-Object -FilePath $Log -Append
        continue
    }
    $resumeArgs = @()
    $partialRoot = "artifacts/training/e7a-alternate-codec-predictors/$($run.arm)/seed-0"
    $partial = @(if (Test-Path $partialRoot) { Get-ChildItem $partialRoot -Recurse -Filter checkpoint.pt })
    if ($partial.Count -gt 0) {
        if (-not $Resume) { throw "Partial alternate predictor needs explicit -Resume: $($run.arm)" }
        $resumeArgs = @("--resume")
    }
    $limit = [double]$run.max_gpu_hours
    Invoke-E7aGpuCell -Phase "predictor-training" -Arm $run.arm -Codec "alternate" `
        -Limit $limit -Expected ([double]$limit * 0.86) -Command {
            & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd_train_v2 `
                --arm $run.arm --study e7a-alternate-codec-predictors --seed $run.seed `
                --steps $run.steps --batch $run.batch --lr $run.lr --warmup $run.warmup --bf16 `
                --train_only --ae_path $altAe --source_commit $sourceCommit `
                --wandb_project $configValue.evaluation.wandb_project --wandb_group $group `
                --dataset_tag loopmom --provenance_tag e7a-alternate-codec `
                --run_log $Log --authorized_gpu_hours $limit @resumeArgs
        }
}

# Resolve the exact argv for all fourteen rows without opening the locked array. The resulting
# hash-checked matrix is both the CPU dispatch gate and the command source used below.
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\resolve_e7a_evaluation_dispatch.py `
    --config $Config --manifest $Manifest --alternate-ae $altAe `
    --execution $executionRecord --training-root artifacts/training `
    --access-log $accessLog --run-log $Log --artifact-root artifacts/runs `
    --out $dispatchReport 2>&1 | Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$dispatch = Get-Content -LiteralPath $dispatchReport -Raw | ConvertFrom-Json
if ($dispatch.status -ne "E7A_EVALUATION_DISPATCH_PASS" -or @($dispatch.rows).Count -ne 14) {
    throw "E7a fourteen-cell evaluation dispatch gate failed"
}
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

# The locked array is opened only after codec validity, all predictor trainings and dispatch pass.
foreach ($evalRow in $dispatch.rows) {
        $codecId = [string]$evalRow.codec
        $armName = [string]$evalRow.arm
        $aePath = [string]$evalRow.ae_path
        $checkpointPath = [string]$evalRow.checkpoint_path
        $evalArgs = @($evalRow.argv | ForEach-Object { [string]$_ })
        if (-not $aePath -or -not $checkpointPath -or
            -not (Test-Path -LiteralPath $aePath) -or
            -not (Test-Path -LiteralPath $checkpointPath)) {
            throw "E7a dispatch contains an empty or missing path: $codecId/$armName"
        }
        $aeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $aePath).Hash.ToLowerInvariant()
        $checkpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpointPath).Hash.ToLowerInvariant()
        if ($aeHash -ne [string]$evalRow.ae_sha256 -or
            $checkpointHash -ne [string]$evalRow.checkpoint_sha256) {
            throw "E7a dispatch path hash changed: $codecId/$armName"
        }
        $existing = @()
        $evalRoot = "artifacts/runs/robustness-e7a/$armName/seed-0"
        if (Test-Path $evalRoot) {
            foreach ($specPath in Get-ChildItem $evalRoot -Recurse -Filter run_spec.json) {
                $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
                if ($spec.dataset_hash -eq $manifestValue.artifacts.clips.sha256 -and
                    $spec.ae_hash -eq $aeHash -and
                    (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
                    $existing += $specPath.FullName
                }
            }
        }
        if ($existing.Count -gt 1) { throw "Duplicate E7a evaluation: $codecId/$armName" }
        if ($existing.Count -eq 1) {
            if (-not $Resume) { throw "Completed E7a evaluation exists; explicit -Resume is required" }
            "=== E7a evaluation $codecId/$armName already COMPLETE; skipping ===" |
                Tee-Object -FilePath $Log -Append
            continue
        }
        $cellLimit = [double]$evalRow.max_gpu_hours
        $cellExpected = [double]$evalRow.expected_gpu_hours
        Invoke-E7aGpuCell -Phase "evaluation" -Arm $armName -Codec $codecId `
            -Limit $cellLimit -Expected $cellExpected -Command {
                & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd @evalArgs
            }
}

$env:CUDA_VISIBLE_DEVICES = "-1"
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& .\venv\Scripts\python.exe scripts\analyze_e7a_codec_robustness.py `
    --config $Config --manifest $Manifest --alternate-ae $altAe --run-root artifacts/runs `
    --out-json artifacts/e7a_codec_robustness/results.json `
    --out-md docs/confirmation/E7A_RESULTS.md `
    --wandb-project $configValue.evaluation.wandb_project --wandb-group $group 2>&1 |
    Tee-Object -FilePath $Log -Append
$analysisCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($analysisCode -ne 0) { exit $analysisCode }
& .\venv\Scripts\python.exe scripts\validate_e7a_completion.py `
    --config $Config --manifest $Manifest --alternate-ae $altAe --floor $floorReport `
    --results artifacts/e7a_codec_robustness/results.json --gpu-ledger $ledger `
    --access-log $accessLog --execution $executionRecord `
    --dispatch $dispatchReport `
    --training-root artifacts/training --run-root artifacts/runs `
    --out artifacts/e7a_codec_robustness/completion_validation.json 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
"=== E7A_DONE total_gpu_hours=$($script:consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
