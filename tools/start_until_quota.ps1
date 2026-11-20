<#
Local unattended Ralph Loop without an 8-hour deadline.
Waits for existing ralph_loop.py process, then runs until:
- Kaggle free GPU quota guard (default >= 2 hours remaining before new job)
- repeated unsafe/unrecoverable errors / uncertain remote job status
- explicit Ctrl+C / Windows process termination
No PESQ score target, no fixed iteration count or fixed wall-time limit.
Run ONLY ONCE in the SECOND PowerShell window, after STOPPING the old
start_overnight.ps1 watcher (Ctrl+C in that watcher window only).

Usage:
  & .\tools\start_until_quota.ps1
#>
[CmdletBinding()]
param(
    [ValidateRange(2, 100)]
    [int]$Epochs = 25
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Started = Get-Date
$LogDir = Join-Path $Root "runs\ralph"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("until_quota_" + $Started.ToString("yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $LogPath -Append | Out-Null

try {
    Write-Host "UNTIL-QUOTA START: $Started"
    Write-Host "No 8-hour deadline and no target PESQ cutoff."
    Write-Host "Stop condition: GPU quota guard or safety failure (or manual Ctrl+C)."
    Write-Host "LOG: $LogPath"
    Write-Host "Waiting for existing 3-round ralph_loop.py process to FINISH..."
    while ($true) {
        $Processes = @(
            Get-CimInstance Win32_Process | Where-Object {
                ($_.Name -match '^(py|python|pythonw)(?:\.exe)?$') -and
                ($_.CommandLine -match 'ralph_loop\.py')
            }
        )
        if ($Processes.Count -eq 0) { break }
        Write-Host ("{0}: existing Ralph Loop still running (PID {1}); waiting..." -f
            (Get-Date -Format "HH:mm:ss"), (($Processes | ForEach-Object { $_.ProcessId }) -join ","))
        Start-Sleep -Seconds 30
    }

    Write-Host "Previous Ralph process exited. Running offline safety tests..."
    & py -m unittest discover -s tests -p "test_ralph_loop.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Offline safety tests failed; not submitting any GPU jobs." }

    Write-Host "Resuming best verified model, continuing until Kaggle GPU quota/safety cutoff."
    $RalphArgs = @(
        "tools/ralph_loop.py",
        "--username", "siyul1122",
        "--dataset", "siyul1122/speech-data-private",
        "--codex-model", "gpt-6.1-sol",
        "--rounds", "0",
        "--epochs", "$Epochs",
        "--max-total-minutes", "0",
        "--max-job-minutes", "80",
        "--min-gpu-quota-hours", "2",
        "--no-pesq-target",
        "--max-stagnant", "0",
        "--execute"
    )
    & py @RalphArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Ralph Loop failed (exit $LASTEXITCODE); see logfile. Automatic retry avoided."
    }
    Write-Host "FINISHED: GPU quota/safety boundary or stop condition reached."
}
finally {
    Stop-Transcript | Out-Null
}
