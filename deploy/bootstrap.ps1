# face2face bootstrap wrapper for Windows PowerShell
# Double-click or run: powershell -ExecutionPolicy Bypass -File bootstrap.ps1
#
# This script downloads and runs bootstrap.py. All flags are forwarded.

$ErrorActionPreference = "Stop"

# Find Python
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    $python = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($python) { break }
}
if (-not $python) {
    Write-Host "Error: Python not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/"
    Write-Host 'Make sure to check "Add Python to PATH" during installation.'
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Using $($python.Source)"

$branch = "claude/webcam-screen-http-proxy-jpxC1"
$url = "https://raw.githubusercontent.com/jtexp/face2face/$branch/deploy/bootstrap.py"
$tmp = Join-Path $env:TEMP "f2f_bootstrap.py"

Write-Host "Downloading bootstrap.py ..."
Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing

Write-Host "Running bootstrap ..."
& $python.Source $tmp @args

Remove-Item $tmp -ErrorAction SilentlyContinue
