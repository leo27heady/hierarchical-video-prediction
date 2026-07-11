param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py b1-fixed-target
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($CheckOnly) {
    Write-Output "B1 fixed-target gate passed; no GPU process launched."
    exit 0
}

throw "Training launch remains disabled until c2 handoff, RunSpec integration, and explicit authorization."
