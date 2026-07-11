param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py phase-c-v2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($CheckOnly) {
    Write-Output "Phase C v2 gate passed; no GPU process launched."
    exit 0
}

throw "Phase C v2 training launcher is intentionally disabled until the validated evaluator is integrated with the post-c2 trainer."
