param(
    [string]$Config = "experiments/configs/sampler_nfe_robustness.json",
    [string]$Mnist = "C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\train-images-idx3-ubyte.gz",
    [string]$Out = "data/robustness/loopmom_e7c_500x200.npy",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E7c config is not frozen" }
if ($configValue.dataset.purpose -ne "robustness_holdout") { throw "Wrong E7c dataset purpose" }
if (-not (Test-Path -LiteralPath $Mnist)) { throw "MNIST source file missing: $Mnist" }
if ($CheckOnly) { Write-Output "E7c dataset-generation gate PASS; dataset not created."; exit 0 }

$sourceCommit = (git rev-parse HEAD).Trim()
$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "E7c dataset generation requires a clean worktree" }
& .\venv\Scripts\python.exe scripts\generate_loopmom_confirmation.py `
    --mnist $Mnist --source-split train --purpose robustness_holdout `
    --n $configValue.dataset.clips --frames $configValue.dataset.frames `
    --world $configValue.dataset.world --digits $configValue.dataset.digits `
    --speed $configValue.dataset.speed --p-cont $configValue.dataset.p_cont `
    --p-hold $configValue.dataset.p_hold --seed $configValue.dataset.seed `
    --generator-commit $sourceCommit --dirty-hash CLEAN --out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "E7c robustness set created without display or selection."
