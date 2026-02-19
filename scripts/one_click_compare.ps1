[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\final_compare",
    [ValidateSet("legacy", "new")]
    [AllowNull()]
    [AllowEmptyString()]
    [string]$Route = $null,
    [switch]$Legacy,
    [switch]$DisableLegacyEquityFallback,
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,
    [switch]$SkipStageDiff
)

$ErrorActionPreference = "Stop"


$routeSource = "default"
if ($Legacy) {
    $resolvedRoute = "legacy"
    $routeSource = "param"
} elseif (-not [string]::IsNullOrWhiteSpace([string]$Route)) {
    $resolvedRoute = $Route
    $routeSource = "param"
} elseif (-not [string]::IsNullOrWhiteSpace($env:QRS_PIPELINE_ROUTE)) {
    $resolvedRoute = $env:QRS_PIPELINE_ROUTE
    $routeSource = "env"
} else {
    $resolvedRoute = "new"
}
$bfText = if ($env:QRS_LEGACY_BACKFILL -and $env:QRS_LEGACY_BACKFILL -ne "0") { "on" } else { "off" }
$dfText = if ($DisableLegacyEquityFallback) { "on" } else { "off" }
Write-Host ("[one-click-compare] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2} route_source={3}" -f $resolvedRoute, $bfText, $dfText, $routeSource)

if (-not (Test-Path -LiteralPath $InputCsv)) {
    Write-Error ("Input CSV not found: {0}`nSuggestions:`n1) Put CSV under data\ (required columns: time/open/high/low/close/volume)`n2) Or pass -InputCsv with a full path" -f $InputCsv)
    exit 2
}

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\run_qrs.ps1",
    "-Mode", "compare",
    "-InputCsv", $InputCsv,
    "-OutRoot", $OutRoot,
    "-TolAbs", $TolAbs,
    "-TolRel", $TolRel,
    "-TopN", $TopN
)
if ($SkipStageDiff) { $argsList += "-SkipStageDiff" }
if ($routeSource -eq "param") { $argsList += @("-Route", $resolvedRoute) }
if ($DisableLegacyEquityFallback) { $argsList += "-DisableLegacyEquityFallback" }

& powershell @argsList

$rc = $LASTEXITCODE
if ($rc -eq 0) {
    exit 0
}
exit 2
