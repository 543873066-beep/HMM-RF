[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\final_compare",
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,
    [switch]$SkipStageDiff
)

$ErrorActionPreference = "Stop"

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

& powershell @argsList

$rc = $LASTEXITCODE
if ($rc -eq 0) {
    exit 0
}
exit 2
