[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\rolling_compare",
    [ValidateSet("legacy", "new")]
    [AllowNull()]
    [AllowEmptyString()]
    [string]$Route = $null,
    [switch]$Legacy,
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,
    [switch]$DisableLegacyEquityFallback
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"


$routeSource = "default"
if ($Legacy) {
    $resolvedRoute = "legacy"
    $routeSource = "param"
} elseif (-not [string]::IsNullOrWhiteSpace([string]$Route)) {
    $resolvedRoute = $Route
    $routeSource = "param"
} elseif (-not [string]::IsNullOrWhiteSpace($env:QRS_ROLLING_ROUTE)) {
    $resolvedRoute = $env:QRS_ROLLING_ROUTE
    $routeSource = "env"
} else {
    $resolvedRoute = "new"
}
$bfText = if ($env:QRS_LEGACY_BACKFILL -and $env:QRS_LEGACY_BACKFILL -ne "0") { "on" } else { "off" }
$dfText = if ($DisableLegacyEquityFallback) { "on" } else { "off" }
Write-Host ("[one-click-rolling] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2} route_source={3}" -f $resolvedRoute, $bfText, $dfText, $routeSource)
if (-not (Test-Path -LiteralPath $InputCsv)) {
    Write-Error ("Input CSV not found: {0}" -f $InputCsv)
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$leaf = Split-Path -Path $OutRoot -Leaf
if ($leaf -match '^\d{8}_\d{6}$') {
    $baseRoot = $OutRoot
} else {
    $baseRoot = Join-Path $OutRoot $timestamp
}
New-Item -ItemType Directory -Force -Path $baseRoot | Out-Null

powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy -InputCsv $InputCsv -OutRoot $baseRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$newArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", "scripts\run_qrs.ps1",
    "-Mode", "rolling", "-InputCsv", $InputCsv, "-OutRoot", $baseRoot
)
if ($routeSource -eq "param") {
    $newArgs += @("-Route", $resolvedRoute)
}
if ($DisableLegacyEquityFallback) {
    $newArgs += "-DisableLegacyEquityFallback"
}
powershell @newArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python tools\rolling_compare.py --legacy-root $baseRoot --new-root $baseRoot --tol-abs $TolAbs --tol-rel $TolRel --topn $TopN
$rc = $LASTEXITCODE
if ($rc -eq 0) {
    Write-Host "[one-click-rolling] PASS"
    exit 0
}

Write-Host "[one-click-rolling] FAIL"
exit 2
