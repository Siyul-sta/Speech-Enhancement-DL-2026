<#
Open the Ralph V2 visual dashboard on your own PC.
Only localhost; no Kaggle GPU job is submitted by this script.
#>
[CmdletBinding()]
param([ValidateRange(1024,65535)][int]$Port = 8765)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Url = "http://127.0.0.1:$Port"

Write-Host ""
Write-Host "  RALPH / SPEECH LAB" -ForegroundColor Cyan
Write-Host "  Live dashboard: $Url" -ForegroundColor White
Write-Host "  Read-only. Close this window or press Ctrl+C to stop dashboard." -ForegroundColor DarkGray
Write-Host ""

& py tools/ralph_dashboard.py --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "Dashboard couldn't start (port $Port may already be occupied)."
}
