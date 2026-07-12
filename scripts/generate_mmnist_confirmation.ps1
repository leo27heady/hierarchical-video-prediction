param(
    [Parameter(Mandatory=$true)][string]$Images,
    [Parameter(Mandatory=$true)][string]$Labels,
    [Parameter(Mandatory=$true)][int]$Seed,
    [Parameter(Mandatory=$true)][string]$GeneratorCommit,
    [string]$Out = "data/confirmation_mmnist.npy",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py phase-c-v2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "Phase-C-v2 generation gate passed; dataset not created."; exit 0 }

& .\venv\Scripts\python.exe scripts\gen_moving_v2.py `
    --images $Images --labels $Labels --source-split test --purpose phase_c_v2_holdout `
    --n 500 --frames 348 --objects 2 --seed $Seed --generator-commit $GeneratorCommit --out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Moving-MNIST Phase-C-v2 set created."
