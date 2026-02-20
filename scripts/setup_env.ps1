[CmdletBinding()]
param(
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1"
    Write-Host ""
    Write-Host "What it does:"
    Write-Host "  1) Create venv .venv"
    Write-Host "  2) Install requirements.txt"
    Write-Host "  3) Run quick compileall sanity"
    Write-Host ""
    Write-Host "Next step:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -DisableLegacyEquityFallback"
    exit 0
}

if (-not (Test-Path -LiteralPath "requirements.txt")) {
    Write-Error "requirements.txt not found."
}

if (-not (Test-Path -LiteralPath ".venv")) {
    python -m venv .venv
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "venv python not found at .venv\Scripts\python.exe"
}

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m compileall quant_refactor_skeleton scripts tools docs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete."
Write-Host "Next: powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -DisableLegacyEquityFallback"
