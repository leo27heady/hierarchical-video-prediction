param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Config = "experiments/configs/pose_indexed_retrieval.json",
    [string]$Log = "logs_rhcp/e8a_pose_mosaic.log",
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "-1"
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$manifestValue = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E8a config is not frozen" }
if ($manifestValue.purpose -ne "retrieval_reference_holdout") {
    throw "Wrong E8a dataset purpose"
}
$group = "$($configValue.evaluation.wandb_group_prefix)-$($manifestValue.dataset_id)"
$maxHours = [double]$configValue.budget.authorized_max_gpu_hours
$ledger = "artifacts/e8a_pose_mosaic/gpu_hours.jsonl"
$accessLog = "artifacts/e8a_pose_mosaic/access.jsonl"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$sourceCommit = (git rev-parse HEAD).Trim()
$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match 'hvp_rev\.rhvd|run_e8a_reference|evaluate_e8a_pose_mosaic' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/evaluator is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py pose-indexed-retrieval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$designArgs = @(
    "scripts/validate_e8a_design.py", "--config", $Config,
    "--protocol", "docs/E8A_POSE_MOSAIC_PROTOCOL_2026-07-17.md",
    "--manifest", $Manifest, "--out", "gates/e8a_design_validation.json"
)
if ($Resume) { $designArgs += "--allow-existing" }
& .\venv\Scripts\python.exe @designArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wandbGateId = "e8a-gate-$($manifestValue.dataset_id)-$($sourceCommit.Substring(0,8))-$(Get-Date -Format yyyyMMddHHmmss)"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.evaluation.wandb_project --group $group --run-id $wandbGateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E8a prelaunch gates PASS; no evaluation launched."; exit 0 }

New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts\e8a_pose_mosaic | Out-Null

# Run the CPU-only static-world validity/reference path first. It can prevent unnecessary GPU
# evaluation, but cannot change the already frozen neural matrix.
$mosaicSpecs = @()
if (Test-Path "artifacts/runs/reference-e8a/pose-mosaic") {
    foreach ($specPath in Get-ChildItem "artifacts/runs/reference-e8a/pose-mosaic" -Recurse -Filter run_spec.json) {
        $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
        if ($spec.dataset_hash -eq $manifestValue.artifacts.clips.sha256 -and
            (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
            $mosaicSpecs += $specPath.FullName
        }
    }
}
if ($mosaicSpecs.Count -gt 1) { throw "Duplicate completed E8a mosaic runs" }
if ($mosaicSpecs.Count -eq 1) {
    if (-not $Resume) { throw "Completed E8a mosaic exists; explicit -Resume required" }
    "=== E8a pose-mosaic already COMPLETE; skipping ===" | Tee-Object -FilePath $Log -Append
} else {
    "=== E8a pose-mosaic START $(Get-Date -Format o) ===" | Tee-Object -FilePath $Log -Append
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & .\venv\Scripts\python.exe scripts\evaluate_e8a_pose_mosaic.py `
        --config $Config --manifest $Manifest --access-log $accessLog `
        --artifact-root artifacts/runs --source-commit $sourceCommit `
        --wandb-project $configValue.evaluation.wandb_project --wandb-group $group `
        --wandb-required 2>&1 | Tee-Object -FilePath $Log -Append
    $mosaicCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($mosaicCode -ne 0) { throw "E8a pose-mosaic reference failed" }
    "=== E8a pose-mosaic END code=0 ===" | Tee-Object -FilePath $Log -Append
}

$consumed = 0.0
if (Test-Path $ledger) {
    foreach ($line in Get-Content $ledger) {
        if ($line.Trim()) { $consumed += [double](($line | ConvertFrom-Json).gpu_hours) }
    }
}
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
foreach ($run in $configValue.neural_reference_runs) {
    $existing = @()
    $armRoot = "artifacts/runs/reference-e8a/$($run.arm)/seed-$($run.seed)"
    if (Test-Path $armRoot) {
        foreach ($specPath in Get-ChildItem $armRoot -Recurse -Filter run_spec.json) {
            $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
            if ($spec.dataset_hash -eq $manifestValue.artifacts.clips.sha256 -and
                (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
                $existing += $specPath.FullName
            }
        }
    }
    if ($existing.Count -gt 1) { throw "Duplicate completed E8a neural cell: $($run.arm)" }
    if ($existing.Count -eq 1) {
        if (-not $Resume) { throw "Completed E8a neural cell exists; explicit -Resume required" }
        "=== E8a $($run.arm) already COMPLETE; skipping ===" |
            Tee-Object -FilePath $Log -Append
        continue
    }
    $checkpointHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $run.checkpoint).Hash.ToLowerInvariant()
    if ($checkpointHash -ne $run.checkpoint_sha256) {
        throw "E8a checkpoint hash changed: $($run.checkpoint)"
    }
    $expected = [double]$run.reference_eval_hours_nfe16
    $cellLimit = [Math]::Min(1.5 * $expected, $maxHours - $consumed)
    if ($cellLimit -le 0 -or $consumed + $expected -gt $maxHours) {
        throw "Next E8a evaluation does not fit the frozen four-hour maximum"
    }
    $started = Get-Date
    "=== E8a neural arm=$($run.arm) START $($started.ToString('o')) ===" |
        Tee-Object -FilePath $Log -Append
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd `
        --arm $run.arm --seed $run.seed --steps $run.steps --batch $run.batch --bf16 `
        --eval_only --complete_metrics --metric_threshold $configValue.evaluation.metric_threshold `
        --nfe $configValue.evaluation.nfe --sampler_seed $configValue.evaluation.sampler_seed `
        --test_path $manifestValue.artifacts.clips.path `
        --warm $configValue.evaluation.warmup_frames --n_val 250 `
        --n_eval_clips $configValue.evaluation.evaluation_clips `
        --ae_path $configValue.codec.path --checkpoint_path $run.checkpoint `
        --retrieval_manifest $Manifest --retrieval_access_log $accessLog `
        --artifact_root artifacts/runs --source_commit $sourceCommit `
        --wandb_project $configValue.evaluation.wandb_project --wandb_group $group --wandb_required `
        --run_log $Log --authorized_gpu_hours $cellLimit 2>&1 |
        Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $consumed += $hours
    [ordered]@{
        schema_version = 1; study = "reference-e8a"; phase = "neural-evaluation";
        arm = $run.arm; seed = [int]$run.seed;
        started = $started.ToUniversalTime().ToString("o");
        ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours;
        expected_gpu_hours = $expected; cell_authorized_max_gpu_hours = $cellLimit;
        tranche_authorized_max_gpu_hours = $maxHours; exit_code = $code;
        source_commit = $sourceCommit; dataset_id = $manifestValue.dataset_id
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E8a neural arm=$($run.arm) END code=$code hours=$($hours.ToString('F3')) total=$($consumed.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E8a neural evaluation failed: $($run.arm)" }
    if ($hours -gt $cellLimit -or $consumed -gt $maxHours) {
        throw "E8a exceeded a frozen runtime authorization"
    }
}

$env:CUDA_VISIBLE_DEVICES = "-1"
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& .\venv\Scripts\python.exe scripts\analyze_e8a_pose_mosaic.py `
    --config $Config --manifest $Manifest --run-root artifacts/runs `
    --out-json artifacts/e8a_pose_mosaic/results.json `
    --out-md docs/confirmation/E8A_RESULTS.md `
    --wandb-project $configValue.evaluation.wandb_project --wandb-group $group 2>&1 |
    Tee-Object -FilePath $Log -Append
$analysisCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($analysisCode -ne 0) { exit $analysisCode }
& .\venv\Scripts\python.exe scripts\validate_e8a_completion.py `
    --config $Config --manifest $Manifest --results artifacts/e8a_pose_mosaic/results.json `
    --gpu-ledger $ledger --access-log $accessLog --run-root artifacts/runs `
    --out artifacts/e8a_pose_mosaic/completion_validation.json 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
"=== E8A_DONE total_gpu_hours=$($consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
