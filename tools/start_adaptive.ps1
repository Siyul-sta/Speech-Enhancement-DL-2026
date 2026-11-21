<#
Start adaptive architecture-switching Ralph V2 locally without time cutoff.
Before running:
1. Ctrl+C the old start_until_quota.ps1 controller.
2. Verify the last Kaggle kernel is no longer RUNNING/QUEUED.
3. Run: & .\tools\start_adaptive.ps1 -ConfirmedNoRemoteJob
#>
[CmdletBinding()]
param(
    [switch]$ConfirmedNoRemoteJob,
    [ValidateRange(2,100)]
    [int]$Epochs = 25,
    [ValidateRange(1,20)]
    [int]$StagnationRounds = 2
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
if (-not $ConfirmedNoRemoteJob) {
    throw "Confirm that the old local loop has stopped and the final Kaggle notebook is not RUNNING/QUEUED. Then re-run with -ConfirmedNoRemoteJob."
}
$existing = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^(py|python|pythonw)(?:\.exe)?$') -and
    ($_.CommandLine -match '(ralph_loop|ralph_v2)\.py')
})
if ($existing.Count -gt 0) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ","
    throw "A local Ralph process is still running (PID $pids). Stop old controller first."
}
$RunRoot = Join-Path $Root "runs\ralph_v2"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$log = Join-Path $RunRoot ("adaptive_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $log -Append | Out-Null
try {
    Write-Host "ADAPTIVE RALPH V2 START"
    Write-Host "First model: residual_phase. Switch after $StagnationRounds full-score plateaus."
    Write-Host "No wall-time cutoff; GPU quota and safety guards remain."
    Write-Host "LOG: $log"
    & py -m unittest discover -s tests -p "test_ralph_v2.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Adaptive offline tests failed; no GPU job submitted." }
    & py -m unittest discover -s tests -p "test_kaggle_v2.py" -v
    if ($LASTEXITCODE -ne 0) { throw "V2 architecture tests failed; no GPU job submitted." }
    $RalphArgs = @(
        "tools/ralph_v2.py",
        "--username", "siyul1122",
        "--dataset", "siyul1122/speech-data-private",
        "--codex-model", "gpt-6.1-sol",
        "--epochs", "$Epochs",
        "--stagnation-rounds", "$StagnationRounds",
        "--max-rounds", "0",
        "--min-gpu-quota-hours", "2",
        "--execute"
    )
    & py @RalphArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Adaptive controller ended with error ($LASTEXITCODE); check log."
    }
    Write-Host "Adaptive controller finished at GPU quota/safety boundary."
}
finally {
    Stop-Transcript | Out-Null
}
