[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\rolling_compare_one_click",
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,
    [switch]$DisableLegacyEquityFallback
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath $InputCsv)) {
    Write-Error ("Input CSV not found: {0}" -f $InputCsv)
}

$legacyRoot = Join-Path $OutRoot "legacy"
$newRoot = Join-Path $OutRoot "new"
New-Item -ItemType Directory -Force -Path $legacyRoot, $newRoot | Out-Null

powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy -InputCsv $InputCsv -OutRoot $legacyRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$newArgs = @(
    "-ExecutionPolicy", "Bypass", "-File", "scripts\run_qrs.ps1",
    "-Mode", "rolling", "-Route", "new", "-InputCsv", $InputCsv, "-OutRoot", $newRoot
)
if ($DisableLegacyEquityFallback) {
    $newArgs += "-DisableLegacyEquityFallback"
}
powershell @newArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python tools\rolling_compare.py --legacy-root $legacyRoot --new-root $newRoot --tol-abs $TolAbs --tol-rel $TolRel --topn $TopN
$rc = $LASTEXITCODE
if ($rc -eq 0) {
    Write-Host "[one-click-rolling] PASS"
    exit 0
}

Write-Host "[one-click-rolling] FAIL"
exit 2
