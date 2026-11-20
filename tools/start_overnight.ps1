<#
8-hour LOCAL Ralph Loop scheduler. Start once in a SECOND PowerShell window.
It waits for the existing 3-round ralph_loop.py Python process to finish.
The 8-hour deadline begins NOW, not after the old experiment completes.
Does not require GitHub Actions, Windows Task Scheduler, or new API keys.

Usage:
  powershell -ExecutionPolicy Bypass -File .\tools\start_overnight.ps1 -Hours 8

Leave this PowerShell and the existing experiment window open, PC awake.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 10)]
    [double]$Hours = 8,
    [ValidateRange(1, 40)]
    [int]$Rounds = 40,
    [ValidateRange(2, 100)]
    [int]$Epochs = 25
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Started = Get-Date
$Deadline = $Started.AddHours($Hours)
$LogDir = Join-Path $Root "runs\ralph"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("overnight_" + $Started.ToString("yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $LogPath -Append | Out-Null

try {
    Write-Host "OVERNIGHT START: $Started"
    Write-Host "DEADLINE: $Deadline (local time)"
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
        if ((Get-Date) -ge $Deadline) {
            Write-Host "8-hour overnight window expired while waiting. No new GPU job submitted."
            exit 0
        }
        Write-Host ("{0}: existing Ralph Loop still running (PID {1}); waiting..." -f
            (Get-Date -Format "HH:mm:ss"), (($Processes | ForEach-Object { $_.ProcessId }) -join ","))
        Start-Sleep -Seconds 30
    }

    # Run preflight only when previous controller has stopped changing source.
    Write-Host "Existing local Ralph process finished. Running offline tests..."
    & py -m unittest discover -s tests -p "test_ralph_loop.py" -v
    if ($LASTEXITCODE -ne 0) {
        throw "Offline unit tests failed. No new GPU job submitted."
    }
    $RemainingMinutes = [int][math]::Floor(($Deadline - (Get-Date)).TotalMinutes)
    if ($RemainingMinutes -lt 80) {
        Write-Host "Less than 80 minutes remain; not starting more GPU jobs."
        exit 0
    }
    Write-Host "Resuming best verified model. Remaining overnight budget: $RemainingMinutes minutes."
    Write-Host "No PESQ target, no stagnation early stop. GPU quota safety remains active."
    $RalphArgs = @(
        "tools/ralph_loop.py",
        "--username", "siyul1122",
        "--dataset", "siyul1122/speech-data-private",
        "--codex-model", "gpt-6.1-sol",
        "--rounds", "$Rounds",
        "--epochs", "$Epochs",
        "--max-total-minutes", "$RemainingMinutes",
        "--max-job-minutes", "60",
        "--min-gpu-quota-hours", "2",
        "--no-pesq-target",
        "--max-stagnant", "0",
        "--execute"
    )
    & py @RalphArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Ralph Loop returned exit code $LASTEXITCODE; see log above."
    }
    Write-Host "OVERNIGHT COMPLETE. Check the newest runs\ralph\run_*\best.json"
}
finally {
    Stop-Transcript | Out-Null
}
