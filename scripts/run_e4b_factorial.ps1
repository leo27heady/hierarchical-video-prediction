param(
    [string]$Config = "experiments/configs/regression_factorial_seeds3_4.json",
    [string]$Log = "logs_rhcp/e4b_head_package_seeds3_4.log",
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
Remove-Item Env:WANDB_API_KEY -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "-1"

$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "Scientific launch requires a clean worktree: $($dirty -join '; ')" }
$currentCommit = (git rev-parse HEAD).Trim()
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E4b config is not frozen" }
$maximum = [double]$configValue.budget.authorized_max_gpu_hours
$allocated = ($configValue.new_training_runs | Measure-Object -Property max_gpu_hours -Sum).Sum
if ([double]$allocated -ne 36.0 -or $maximum -ne 36.0) {
    throw "E4b must remain inside its frozen 36-hour replacement allocation"
}

$ledger = "artifacts/e4b_head_package/gpu_hours.jsonl"
$entries = @()
if (Test-Path $ledger) {
    $entries = @(Get-Content -LiteralPath $ledger -Encoding utf8 | Where-Object { $_ } |
        ForEach-Object { $_ | ConvertFrom-Json })
}
$sourceCommit = $currentCommit
if ($Resume -and $entries.Count -gt 0) {
    $commits = @($entries | Select-Object -ExpandProperty source_commit -Unique)
    if ($commits.Count -ne 1) { throw "E4b ledger has multiple source commits" }
    $sourceCommit = [string]$commits[0]
    $scientificPaths = @(
        "hvp_rev/arm_registry.py", "hvp_rev/provenance.py", "hvp_rev/rhvd.py",
        "hvp_rev/rhvd_train_v2.py", $Config, "scripts/analyze_e4b_factorial.py",
        "scripts/validate_e4b_completion.py"
    )
    & git diff --quiet $sourceCommit $currentCommit -- $scientificPaths
    if ($LASTEXITCODE -ne 0) {
        throw "Scientific sources changed since the original E4b launch commit"
    }
}

$projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match 'hvp_rev\.rhvd_train_v2|run_e4b_factorial' -and
    $_.ProcessId -ne $PID
})
if ($projectProcesses.Count -ne 0) { throw "Another project controller/trainer is active" }

& .\venv\Scripts\python.exe scripts\check_experiment_gate.py head-package-seeds3-4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$designArgs = @("scripts/validate_e4b_design.py", "--config", $Config,
    "--out", "gates/e4b_design_validation.json")
if ($Resume) { $designArgs += "--allow-existing" }
& .\venv\Scripts\python.exe @designArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$gateId = "e4b-gate-$($currentCommit.Substring(0,8))-$(Get-Date -Format yyyyMMddHHmmss)"
& .\venv\Scripts\python.exe scripts\check_wandb_online.py `
    --project $configValue.wandb.project --group $configValue.wandb.group --run-id $gateId
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "E4b prelaunch gates PASS; no GPU process launched."; exit 0 }

Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
New-Item -ItemType Directory -Force -Path artifacts\e4b_head_package | Out-Null
$consumed = [double](($entries | Measure-Object -Property gpu_hours -Sum).Sum)

foreach ($run in $configValue.new_training_runs) {
    $runBase = Join-Path "artifacts/training/$($configValue.study)/$($run.arm)" "seed-$($run.seed)"
    $specs = @()
    if (Test-Path $runBase) {
        $specs = @(Get-ChildItem -LiteralPath $runBase -Recurse -Filter run_spec.json)
    }
    if ($specs.Count -gt 1) { throw "Multiple RunSpecs found for $($run.arm)/s$($run.seed)" }
    if ($specs.Count -eq 1) {
        $runDir = $specs[0].Directory.FullName
        if (Test-Path (Join-Path $runDir "COMPLETE")) {
            $successful = @($entries | Where-Object {
                $_.arm -eq $run.arm -and [int]$_.seed -eq [int]$run.seed -and $_.exit_code -eq 0
            })
            if ($successful.Count -eq 0) {
                throw "Completed run lacks a GPU-ledger row; manual conservative accounting required"
            }
            "=== E4b $($run.arm) seed=$($run.seed) already COMPLETE; skipping ===" |
                Tee-Object -FilePath $Log -Append
            continue
        }
        if (-not $Resume) { throw "Partial E4b run exists; explicit -Resume is required" }
    }

    $projectProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'hvp_rev\.rhvd_train_v2' -and
        $_.ProcessId -ne $PID
    })
    if ($projectProcesses.Count -ne 0) { throw "Another project trainer appeared" }
    $cellSpent = [double](($entries | Where-Object {
        $_.arm -eq $run.arm -and [int]$_.seed -eq [int]$run.seed
    } | Measure-Object -Property gpu_hours -Sum).Sum)
    $cellRemaining = [double]$run.max_gpu_hours - $cellSpent
    if ($cellRemaining -le 0) { throw "No authorized time remains for $($run.arm)/s$($run.seed)" }
    if (($consumed + $cellRemaining) -gt $maximum + 1e-9) {
        throw "Next E4b attempt could exceed the tranche maximum"
    }

    $started = Get-Date
    "=== E4b $($run.arm) seed=$($run.seed) START $($started.ToString('o')) ===" |
        Tee-Object -FilePath $Log -Append
    $trainArgs = @(
        "-u", "-m", "hvp_rev.rhvd_train_v2",
        "--arm", $run.arm, "--study", $configValue.study, "--seed", $run.seed,
        "--steps", $run.steps, "--batch", $run.batch, "--lr", $run.lr,
        "--warmup", $run.warmup, "--bf16", "--nfe", $configValue.evaluation.nfe,
        "--n_eval_clips", $configValue.evaluation.clips, "--source_commit", $sourceCommit,
        "--wandb_project", $configValue.wandb.project, "--wandb_group", $configValue.wandb.group,
        "--run_log", $Log, "--authorized_gpu_hours", $cellRemaining
    )
    if ($specs.Count -eq 1) { $trainArgs += "--resume" }
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & .\venv\Scripts\python.exe @trainArgs 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    $ended = Get-Date
    $hours = ($ended - $started).TotalHours
    $consumed += $hours
    [ordered]@{
        schema_version = 1; study = $configValue.study; arm = $run.arm; seed = [int]$run.seed
        steps = [int]$run.steps; started = $started.ToUniversalTime().ToString("o")
        ended = $ended.ToUniversalTime().ToString("o"); gpu_hours = $hours
        attempt_authorized_gpu_hours = $cellRemaining
        cell_authorized_max_gpu_hours = [double]$run.max_gpu_hours
        tranche_authorized_max_gpu_hours = $maximum; exit_code = $code
        source_commit = $sourceCommit; resumed = [bool]($specs.Count -eq 1)
    } | ConvertTo-Json -Compress | Add-Content -Encoding utf8 -Path $ledger
    "=== E4b $($run.arm) seed=$($run.seed) END code=$code hours=$($hours.ToString('F3')) ===" |
        Tee-Object -FilePath $Log -Append
    if ($code -ne 0) { throw "E4b run failed: $($run.arm)/s$($run.seed)" }
    if ($hours -gt $cellRemaining) {
        throw "E4b cell exceeded its remaining authorization; stopping before the next run"
    }
    $entries = @(Get-Content -LiteralPath $ledger -Encoding utf8 | Where-Object { $_ } |
        ForEach-Object { $_ | ConvertFrom-Json })
}

$savedErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& .\venv\Scripts\python.exe scripts\analyze_e4b_factorial.py `
    --config $Config --run-root artifacts/training `
    --out-json artifacts/e4b_head_package/results.json `
    --out-md docs/confirmation/E4B_RESULTS.md `
    --wandb-project $configValue.wandb.project --wandb-group $configValue.wandb.group 2>&1 |
    Tee-Object -FilePath $Log -Append
$analysisCode = $LASTEXITCODE
$ErrorActionPreference = $savedErrorActionPreference
if ($analysisCode -ne 0) { exit $analysisCode }

& .\venv\Scripts\python.exe scripts\validate_e4b_completion.py `
    --config $Config --results artifacts/e4b_head_package/results.json `
    --run-root artifacts/training --gpu-ledger $ledger `
    --out artifacts/e4b_head_package/completion_validation.json 2>&1 |
    Tee-Object -FilePath $Log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

"=== E4B_FACTORIAL_DONE total_gpu_hours=$($consumed.ToString('F3')) ===" |
    Tee-Object -FilePath $Log -Append
