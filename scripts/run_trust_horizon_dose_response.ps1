param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py trust-horizon-dose-response
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "Trust horizon gate passed; no GPU process launched."; exit 0 }
throw "Launcher disabled until trust protocol, c2 handoff and explicit authorization."
