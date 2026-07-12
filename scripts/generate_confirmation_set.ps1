param(
    [Parameter(Mandatory=$true)][int]$Seed,
    [Parameter(Mandatory=$true)][string]$GeneratorCommit,
    [Parameter(Mandatory=$true)][string]$DirtyHash,
    [string]$Mnist = "C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\train-images-idx3-ubyte.gz",
    [string]$Out = "data/confirmation/loopmom_confirmation_500x200.npy",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
& .\venv\Scripts\python.exe scripts\check_experiment_gate.py confirmation-holdout
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($CheckOnly) { Write-Output "Loopmom confirmation generation gate passed; dataset not created."; exit 0 }

& .\venv\Scripts\python.exe scripts\generate_loopmom_confirmation.py `
    --mnist $Mnist --source-split train --purpose confirmation_holdout `
    --n 500 --frames 200 --world 192 --digits 3 --speed 4 --p-cont 0.9 --p-hold 0.1 `
    --seed $Seed --generator-commit $GeneratorCommit --dirty-hash $DirtyHash --out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "Loopmom confirmation set created. Do not open it outside the frozen evaluation command."
