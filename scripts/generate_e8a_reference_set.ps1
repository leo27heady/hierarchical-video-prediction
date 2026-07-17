param(
    [string]$Config = "experiments/configs/pose_indexed_retrieval.json",
    [string]$Mnist = "C:\Users\leoni\Documents\projects\OpenSTL\data\moving_mnist\train-images-idx3-ubyte.gz",
    [string]$Out = "data/reference/loopmom_e8a_reference_500x200.npy",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "-1"
$configValue = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
if ($configValue.status -ne "FROZEN") { throw "E8a config is not frozen" }
if ($configValue.dataset.purpose -ne "retrieval_reference_holdout") {
    throw "Wrong E8a dataset purpose"
}
if (-not (Test-Path -LiteralPath $Mnist)) { throw "MNIST source file missing: $Mnist" }
if ($CheckOnly) {
    Write-Output "E8a dataset-generation gate PASS; dataset not created."
    exit 0
}

$sourceCommit = (git rev-parse HEAD).Trim()
$dirty = @(git status --porcelain)
if ($dirty.Count -ne 0) { throw "E8a dataset generation requires a clean worktree" }
& .\venv\Scripts\python.exe scripts\generate_loopmom_confirmation.py `
    --mnist $Mnist --source-split train --purpose retrieval_reference_holdout `
    --n $configValue.dataset.clips --frames $configValue.dataset.frames `
    --world $configValue.dataset.world --digits $configValue.dataset.digits `
    --speed $configValue.dataset.speed --p-cont $configValue.dataset.p_cont `
    --p-hold $configValue.dataset.p_hold --seed $configValue.dataset.seed `
    --generator-commit $sourceCommit --dirty-hash CLEAN --out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "E8a reference set created without display, selection, training, or model access."
