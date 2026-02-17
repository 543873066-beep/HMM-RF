[CmdletBinding()]
param(
    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\final_compare",
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $InputCsv)) {
    Write-Error ("Input CSV not found: {0}`n建议：`n1) 把 CSV 放到 data\ 下（字段至少含 time/open/high/low/close/volume）`n2) 或使用 -InputCsv 指定路径" -f $InputCsv)
    exit 2
}

& powershell -ExecutionPolicy Bypass -File "scripts\run_qrs.ps1" `
    -Mode compare `
    -InputCsv $InputCsv `
    -OutRoot $OutRoot `
    -TolAbs $TolAbs `
    -TolRel $TolRel `
    -TopN $TopN

$rc = $LASTEXITCODE
if ($rc -eq 0) {
    exit 0
}
exit 2
