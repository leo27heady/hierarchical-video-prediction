param(
    [string]$Config = "experiments/configs/regression_factorial_seed2.json",
    [string]$Log = "logs_rhcp/e4_head_package_seed2.log",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "-1"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$sourceCommit = (git rev-parse HEAD).Trim()
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E4 config is not frozen" }
$maximum = [double]$configValue.budget.authorized_max_gpu_hours
$allocated = ($configValue.new_training_runs | Measure-Object -Property max_gpu_hours -Sum).Sum
if ([double]$allocated -gt $maximum) { throw "E4 allocations exceed tranche maximum" }

$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd|run_e[24]_.*factorial' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/trainer is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py regression-factorial-seed2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\venv\Scripts\python.exe scripts\validate_e4_design.py --config $Config --out gates/e4_design_validation.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$gateId = "e4-gate-$($sourceCommit.Substring(0,8))-$(Get-Date -Format yyyyMMddHHmmss)"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.wandb.project --group $configValue.wandb.group --run-id $gateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E4 prelaunch gates PASS; no GPU process launched."; exit 0 }

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts\e4_head_package | Out-Null
$ledger = "artifacts/e4_head_package/gpu_hours.jsonl"
$consumed = 0.0

foreach ($run in $configValue.new_training_runs) {
    $projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd_train_v2' -and $_.ProcessId -ne $PID
    })
    if ($projectProcesses.Count -ne 0) { throw "Another project trainer appeared" }
    if (($consumed + [double]$run.max_gpu_hours) -gt $maximum) {
        throw "Next E4 run would exceed the authorized tranche maximum"
    }
    $started = Get-Date
    "=== E4 $($run.arm) seed=$($run.seed) START $($started.ToString('o')) ===" |
        Tee-Object -FilePath $Log -Append
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd_train_v2 `
        --arm $run.arm --study $configValue.study --seed $run.seed --steps $run.steps `
        --batch $run.batch --lr $run.lr --warmup $run.warmup --bf16 `
        --nfe $configValue.evaluation.nfe --n_eval_clips $configValue.evaluation.clips `
        --source_commit $sourceCommit --wandb_project $configValue.wandb.project `
        --wandb_group $configValue.wandb.group --run_log $Log `
        --authorized_gpu_hours $run.max_gpu_hours 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $consumed += $hours
    [ordered]@{
        schema_version = 1; study = $configValue.study; arm = $run.arm; seed = [int]$run.seed
        steps = [int]$run.steps; started = $started.ToUniversalTime().ToString("o")
        ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours
        authorized_max_gpu_hours = [double]$run.max_gpu_hours; exit_code = $code
        source_commit = $sourceCommit
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E4 $($run.arm) END code=$code hours=$($hours.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E4 run failed: $($run.arm)" }
    if ($hours -gt [double]$run.max_gpu_hours) {
        throw "E4 run exceeded its authorized maximum; stopping before the next run"
    }
}

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& .\venv\Scripts\python.exe scripts\analyze_e4_factorial.py `
    --config $Config --run-root artifacts/training `
    --out-json artifacts/e4_head_package/results.json `
    --out-md docs/confirmation/E4_RESULTS.md `
    --wandb-project $configValue.wandb.project --wandb-group $configValue.wandb.group 2>&1 |
    Tee-Object -FilePath $Log -Append
$analysisCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($analysisCode -ne 0) { exit $analysisCode }

"=== E4_FACTORIAL_DONE total_gpu_hours=$($consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
