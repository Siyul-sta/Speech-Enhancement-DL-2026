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
    [switch]$NoDashboard,
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
if (-not $NoDashboard) {
    try {
        $dashPort = 8765
        $dashUrl = "http://127.0.0.1:$dashPort"
        $tcp = New-Object System.Net.Sockets.TcpClient
        $alreadyOpen = $false
        try {
            $tcp.Connect("127.0.0.1", $dashPort)
            $alreadyOpen = $true
        }
        catch {
            $alreadyOpen = $false
        }
        finally {
            $tcp.Dispose()
        }
        if ($alreadyOpen) {
            Write-Host "Existing Ralph dashboard: $dashUrl" -ForegroundColor Cyan
            Start-Process $dashUrl
        }
        else {
            $dashboardScript = Join-Path $PSScriptRoot "start_dashboard.ps1"
            $launchArgs = '-NoProfile -ExecutionPolicy Bypass -File "' + $dashboardScript + '"'
            Start-Process -FilePath "powershell.exe" -ArgumentList $launchArgs -WorkingDirectory $Root -WindowStyle Minimized
            Write-Host "Starting Ralph visual dashboard: $dashUrl" -ForegroundColor Cyan
        }
    }
    catch {
        Write-Warning "Dashboard couldn't start, but the experiment controller can continue: $_"
    }
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
    Write-Host "Checking interrupted Kaggle jobs and recovering verified terminal results..."
    & py tools/reconcile_ralph_v2.py --resolve
    if ($LASTEXITCODE -ne 0) {
        throw "Remote recovery failed; no new GPU job submitted."
    }
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
