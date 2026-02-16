#!/usr/bin/env pwsh
# Test the face2face HTTP proxy with a minimal HEAD request.
#
# Usage:
#   .\test-proxy.ps1
#   .\test-proxy.ps1 -Port 9090

param(
    [string]$Host_ = "127.0.0.1",
    [int]$Port = 8080
)

$proxy = "http://${Host_}:${Port}"
Write-Host "HEAD http://1.1.1.1/ via $proxy" -ForegroundColor Cyan
curl.exe --proxy $proxy -v --max-time 120 -H "User-Agent:" -H "Accept:" http://1.1.1.1/
