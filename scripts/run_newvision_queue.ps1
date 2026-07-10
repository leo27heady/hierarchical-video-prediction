param(
    [ValidateSet('full', 'after-seed1', 'core-only', 'after-seed1-core')]
    [string]$Mode = 'full'
)

$ErrorActionPreference = 'Continue'  # native stderr is part of the training log, not a PS failure
$bash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path -LiteralPath $bash)) {
    throw "Git Bash not found at $bash. Do not use C:\Windows\System32\bash.exe; WSL is unavailable."
}

$log = 'logs_rhcp\newvision_queue.log'
& $bash scripts/run_newvision_queue.sh $Mode 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "New-vision queue stopped with exit code $code. Inspect $log before resuming."
}
