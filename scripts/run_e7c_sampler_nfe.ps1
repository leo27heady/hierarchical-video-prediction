param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Config = "experiments/configs/sampler_nfe_robustness.json",
    [string]$Log = "logs_rhcp/e7c_sampler_nfe.log",
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "-1"
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$manifestValue = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E7c config is not frozen" }
if ($manifestValue.purpose -ne "robustness_holdout") { throw "Wrong E7c dataset purpose" }
$testPath = $manifestValue.artifacts.clips.path
$group = "$($configValue.evaluation.wandb_group_prefix)-$($manifestValue.dataset_id)"
$maxHours = [double]$configValue.budget.authorized_max_gpu_hours
$multiplierStop = [double]$configValue.budget.single_run_runtime_multiplier_stop
$ledger = "artifacts/e7c_sampler_nfe/gpu_hours.jsonl"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$sourceCommit = (git rev-parse HEAD).Trim()
$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd|run_e7c_sampler_nfe' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/evaluator is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py sampler-nfe-robustness
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Resume) {
    & .\venv\Scripts\python.exe scripts\validate_e7c_design.py `
        --config $Config --manifest $Manifest --out gates/e7c_design_validation.json --allow-existing
} else {
    & .\venv\Scripts\python.exe scripts\validate_e7c_design.py `
        --config $Config --manifest $Manifest --out gates/e7c_design_validation.json
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wandbGateId = "e7c-gate-$($manifestValue.dataset_id)-$($sourceCommit.Substring(0,8))-$(Get-Date -Format yyyyMMddHHmmss)"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.evaluation.wandb_project --group $group --run-id $wandbGateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E7c prelaunch gates PASS; no GPU process launched."; exit 0 }

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts\e7c_sampler_nfe | Out-Null
$consumed = 0.0
if (Test-Path $ledger) {
    foreach ($line in Get-Content $ledger) {
        if ($line.Trim()) { $consumed += [double](($line | ConvertFrom-Json).gpu_hours) }
    }
}

foreach ($arm in $configValue.arms) {
    $checkpoint = "logs_rhcp/rhvd_$($arm.arm)_s$($arm.seed)_t$($arm.steps).pt"
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash.ToLowerInvariant()
    if ($actualHash -ne $arm.checkpoint_sha256) { throw "Checkpoint hash changed: $checkpoint" }
    foreach ($variant in $configValue.variants) {
        $existing = @()
        $armRoot = "artifacts/runs/robustness-e7c/$($arm.arm)/seed-$($arm.seed)"
        if (Test-Path $armRoot) {
            foreach ($specPath in Get-ChildItem $armRoot -Recurse -Filter run_spec.json) {
                $spec = Get-Content $specPath.FullName -Raw | ConvertFrom-Json
                if ($spec.dataset_hash -eq $manifestValue.artifacts.clips.sha256 -and
                    [int]$spec.sampler.nfe -eq [int]$variant.nfe -and
                    [int]$spec.sampler.sampler_seed -eq [int]$variant.sampler_seed -and
                    (Test-Path (Join-Path $specPath.Directory.FullName "COMPLETE"))) {
                    $existing += $specPath.FullName
                }
            }
        }
        if ($existing.Count -gt 1) { throw "Duplicate completed E7c cell" }
        if ($existing.Count -eq 1) {
            if (-not $Resume) { throw "Completed E7c cell exists; explicit -Resume is required" }
            "=== E7c $($arm.arm) $($variant.id) already COMPLETE; skipping ===" |
                Tee-Object -FilePath $Log -Append
            continue
        }

        $expectedHours = [double]$arm.reference_hours_nfe16 * ([double]$variant.nfe / 16.0)
        $perRunMax = $expectedHours * $multiplierStop
        if ($consumed + 1.25 * $expectedHours -gt $maxHours) {
            throw "Next E7c cell would threaten the frozen 8-hour maximum"
        }
        $projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd' -and $_.ProcessId -ne $PID
        })
        if ($projectProcesses.Count -ne 0) { throw "Another project evaluator/trainer appeared" }
        $cellLimit = [Math]::Min($perRunMax, $maxHours - $consumed)

        $started = Get-Date
        "=== E7c $($arm.arm) $($variant.id) START $($started.ToString('o')) ===" |
            Tee-Object -FilePath $Log -Append
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd `
            --arm $arm.arm --seed $arm.seed --steps $arm.steps --batch $arm.batch --bf16 `
            --eval_only --complete_metrics --metric_threshold $configValue.evaluation.metric_threshold `
            --test_path $testPath --warm $configValue.evaluation.warmup_frames --n_val 250 `
            --n_eval_clips $configValue.evaluation.evaluation_clips --nfe $variant.nfe `
            --sampler_seed $variant.sampler_seed --robustness_manifest $Manifest `
            --robustness_access_log artifacts/e7c_sampler_nfe/access.jsonl `
            --artifact_root artifacts/runs --source_commit $sourceCommit `
            --wandb_project $configValue.evaluation.wandb_project --wandb_group $group --wandb_required `
            --run_log $Log --authorized_gpu_hours $cellLimit 2>&1 | Tee-Object -FilePath $Log -Append
        $code = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorActionPreference
        $ended = Get-Date
        $hours = ($ended - $started).TotalHours
        $consumed += $hours
        [ordered]@{
            schema_version = 1; study = "robustness-e7c"; arm = $arm.arm;
            training_seed = [int]$arm.seed; nfe = [int]$variant.nfe;
            sampler_seed = [int]$variant.sampler_seed; variant = $variant.id;
            started = $started.ToUniversalTime().ToString("o");
            ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours;
            expected_gpu_hours = $expectedHours; per_run_stop_hours = $perRunMax;
            authorized_max_gpu_hours = $maxHours; exit_code = $code;
            source_commit = $sourceCommit; dataset_id = $manifestValue.dataset_id
        } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
        "=== E7c $($arm.arm) $($variant.id) END code=$code hours=$($hours.ToString('F3')) total=$($consumed.ToString('F3')) ===" |
            Tee-Object -FilePath $Log -Append
        if ($code -ne 0) { throw "E7c cell failed: $($arm.arm)/$($variant.id)" }
        if ($hours -gt $perRunMax) { throw "E7c cell exceeded twice its NFE-scaled reference time" }
        if ($consumed -gt $maxHours) { throw "E7c exceeded its frozen tranche maximum" }
    }
}

$env:CUDA_VISIBLE_DEVICES = "-1"
$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& .\venv\Scripts\python.exe scripts\analyze_e7c_sampler_nfe.py `
    --config $Config --manifest $Manifest --run-root artifacts/runs `
    --out-json artifacts/e7c_sampler_nfe/results.json `
    --out-md docs/confirmation/E7C_RESULTS.md `
    --wandb-project $configValue.evaluation.wandb_project --wandb-group $group 2>&1 |
    Tee-Object -FilePath $Log -Append
$analysisCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($analysisCode -ne 0) { exit $analysisCode }
& .\venv\Scripts\python.exe scripts\validate_e7c_completion.py `
    --config $Config --manifest $Manifest --results artifacts/e7c_sampler_nfe/results.json `
    --gpu-ledger $ledger --run-root artifacts/runs `
    --out artifacts/e7c_sampler_nfe/completion_validation.json 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
"=== E7C_DONE total_gpu_hours=$($consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
