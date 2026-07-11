param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py regression-factorial-seed2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "Regression factorial gate passed; no GPU process launched."; exit 0 }
throw "Launcher disabled until c2 handoff and explicit authorization."
