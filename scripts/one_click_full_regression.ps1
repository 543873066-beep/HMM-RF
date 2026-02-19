[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\full_regression",
    [ValidateSet("legacy", "new")]
    [string]$Route = "new",
    [switch]$Legacy,
    [switch]$DisableLegacyEquityFallback
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"


$resolvedRoute = if ($Legacy) { "legacy" } else { $Route }
$bfText = if ($env:QRS_LEGACY_BACKFILL -and $env:QRS_LEGACY_BACKFILL -ne "0") { "on" } else { "off" }
$dfText = if ($DisableLegacyEquityFallback) { "on" } else { "off" }
Write-Host ("[one-click-full] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2}" -f $resolvedRoute, $bfText, $dfText)
if (-not (Test-Path -LiteralPath $InputCsv)) {
    Write-Error ("Input CSV not found: {0}" -f $InputCsv)
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$root = Join-Path $OutRoot $timestamp
New-Item -ItemType Directory -Force -Path $root | Out-Null

function Latest-CompareRoot([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) { return $null }
    return Get-ChildItem -LiteralPath $RootDir -Directory |
        Where-Object { (Test-Path (Join-Path $_.FullName "legacy")) -and (Test-Path (Join-Path $_.FullName "new")) } |
        Sort-Object Name -Descending |
        Select-Object -First 1
}

function Fail-Step([string]$Step, [string]$Info) {
    Write-Host ("FAIL step={0}" -f $Step)
    if (-not [string]::IsNullOrWhiteSpace($Info)) {
        Write-Host ("FAIL info={0}" -f $Info)
    }
    exit 2
}

# 1) engine compare
$compareArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", "scripts\run_qrs.ps1",
    "-Mode", "compare", "-Route", $resolvedRoute, "-InputCsv", $InputCsv, "-OutRoot", $root
)
if ($DisableLegacyEquityFallback) {
    $compareArgs += "-DisableLegacyEquityFallback"
}
powershell @compareArgs
if ($LASTEXITCODE -ne 0) {
    $cmp = Latest-CompareRoot $root
    Fail-Step "ENGINE_COMPARE" ("compare_root=" + ($cmp.FullName))
}

# 2) stage-diff
powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode stage-diff -InputCsv $InputCsv -OutRoot $root
if ($LASTEXITCODE -ne 0) {
    $cmp = Latest-CompareRoot $root
    Fail-Step "STAGE_DIFF" ("compare_root=" + ($cmp.FullName))
}

# 3) rolling compare
$rollRoot = Join-Path $root "rolling"
$rollArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", "scripts\one_click_rolling_compare.ps1",
    "-InputCsv", $InputCsv, "-OutRoot", $rollRoot
)
if ($DisableLegacyEquityFallback) {
    $rollArgs += "-DisableLegacyEquityFallback"
}
powershell @rollArgs
if ($LASTEXITCODE -ne 0) {
    Fail-Step "ROLLING_COMPARE" ("rolling_root=" + $rollRoot)
}

Write-Host "FULL_REGRESSION=PASS"
exit 0
