param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Config = "experiments/configs/e2_untouched_confirmation.json",
    [string]$Log = "logs_rhcp/e2_confirmation.log",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
$env:CUDA_VISIBLE_DEVICES = "-1"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$sourceCommit = (git rev-parse HEAD).Trim()
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$manifestValue = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E2 config is not FROZEN" }
if ($manifestValue.purpose -ne "confirmation_holdout") { throw "Wrong dataset purpose" }
$testPath = $manifestValue.artifacts.clips.path
$group = "$($configValue.evaluation.wandb_group_prefix)-$($manifestValue.dataset_id)"
$maxHours = [double]$configValue.budget.authorized_max_gpu_hours
$allocated = ($configValue.runs | Measure-Object -Property max_gpu_hours -Sum).Sum
if ([double]$allocated -gt $maxHours) { throw "Run allocations exceed the tranche maximum" }

$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd|run_e2_confirmation' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/trainer is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py confirmation-holdout
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\venv\Scripts\python.exe scripts\validate_e2_design.py `
    --config $Config --manifest $Manifest --out gates/e2_design_validation.json
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$wandbGateId = "e2-gate-$($manifestValue.dataset_id)-$($sourceCommit.Substring(0,8))"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.evaluation.wandb_project --group $group --run-id $wandbGateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E2 prelaunch gates PASS; no GPU process launched."; exit 0 }

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts | Out-Null
$ledger = "artifacts/gpu_hours.jsonl"
$consumed = 0.0

foreach ($run in $configValue.runs) {
    $projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd' -and $_.ProcessId -ne $PID
    })
    if ($projectProcesses.Count -ne 0) { throw "Another project evaluator/trainer appeared" }
    $projected = $consumed + [double]$run.max_gpu_hours
    if ($projected -gt $maxHours) { throw "Next run would exceed E2's authorized maximum" }
    $checkpoint = "logs_rhcp/rhvd_$($run.arm)_s$($run.seed)_t$($run.steps).pt"
    $actualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $checkpoint).Hash.ToLowerInvariant()
    if ($actualHash -ne $run.checkpoint_sha256) { throw "Checkpoint hash changed: $checkpoint" }

    $started = Get-Date
    $header = "=== E2 $($run.arm) seed=$($run.seed) steps=$($run.steps) START $($started.ToString('o')) ==="
    $header | Tee-Object -FilePath $Log -Append
    & .\venv\Scripts\python.exe -u -m hvp_rev.rhvd `
        --arm $run.arm --seed $run.seed --steps $run.steps --batch $run.batch --bf16 `
        --eval_only --complete_metrics --metric_threshold $configValue.evaluation.metric_threshold `
        --test_path $testPath --warm $configValue.evaluation.warmup_frames --n_val 250 `
        --n_eval_clips $configValue.evaluation.evaluation_clips --nfe $configValue.evaluation.nfe `
        --confirmation_manifest $Manifest --confirmation_access_log artifacts/confirmation_e2/access.jsonl `
        --artifact_root artifacts/runs --source_commit $sourceCommit `
        --wandb_project $configValue.evaluation.wandb_project --wandb_group $group --wandb_required `
        --run_log $Log --authorized_gpu_hours $run.max_gpu_hours 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $consumed += $hours
    [ordered]@{
        schema_version = 1
        study = "confirmation-e2"
        arm = $run.arm
        seed = [int]$run.seed
        steps = [int]$run.steps
        started = $started.ToUniversalTime().ToString("o")
        ended = $ended.ToUniversalTime().ToString("o")
        gpu_hours = $hours
        authorized_max_gpu_hours = [double]$run.max_gpu_hours
        exit_code = $code
        source_commit = $sourceCommit
        dataset_id = $manifestValue.dataset_id
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E2 $($run.arm) seed=$($run.seed) END code=$code hours=$($hours.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E2 run failed: $($run.arm) seed $($run.seed)" }
    if ($hours -gt [double]$run.max_gpu_hours) {
        throw "E2 run exceeded its authorized per-run maximum; stopping before the next run"
    }
}

& .\venv\Scripts\python.exe scripts\analyze_e2_confirmation.py `
    --config $Config --manifest $Manifest --run-root artifacts/runs `
    --out-json artifacts/confirmation_e2/results.json `
    --out-md docs/confirmation/E2_RESULTS.md `
    --wandb-project $configValue.evaluation.wandb_project --wandb-group $group 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

"=== E2_CONFIRMATION_DONE total_gpu_hours=$($consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
