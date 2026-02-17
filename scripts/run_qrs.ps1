[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("engine", "rolling", "compare")]
    [string]$Mode,

    [ValidateSet("legacy", "new")]
    [string]$Route = "legacy",

    [string]$InputCsv = "data\sh000852_5m.csv",

    [string]$OutRoot = "outputs_rebuild\qrs_runs",

    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,

    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine  -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare [-InputCsv <csv>] [-OutRoot <dir>] [-TolAbs 1e-10] [-TolRel 1e-10] [-TopN 20]"
}

function Ensure-InputCsv([string]$CsvPath) {
    if (-not (Test-Path -LiteralPath $CsvPath)) {
        Write-Error ("Input CSV not found: {0}`n建议：`n1) 把 CSV 放到 data\ 下（字段至少含 time/open/high/low/close/volume）`n2) 或使用 -InputCsv 指定路径" -f $CsvPath)
    }
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )
    & python @Args | Out-Host
    return $LASTEXITCODE
}

function Get-BestEquityCsv([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) {
        return $null
    }
    $best = $null
    $bestScore = -1
    $bestRows = -1

    Get-ChildItem -LiteralPath $RootDir -Recurse -File -Filter *.csv | ForEach-Object {
        $file = $_
        $name = $file.Name.ToLowerInvariant()
        $score = 0
        if ($name -match "equity") { $score += 4 }
        if ($name -match "curve") { $score += 2 }

        $header = ""
        try {
            $header = (Get-Content -LiteralPath $file.FullName -TotalCount 1)
        } catch {
            $header = ""
        }
        $headerLower = $header.ToLowerInvariant()
        if ($headerLower -match "equity|eq|nav") { $score += 3 }
        if ($headerLower -match "time") { $score += 1 }

        $rows = 0
        try {
            $rows = [Math]::Max(0, ((Get-Content -LiteralPath $file.FullName | Measure-Object -Line).Lines - 1))
        } catch {
            $rows = 0
        }

        if (($score -gt $bestScore) -or (($score -eq $bestScore) -and ($rows -gt $bestRows))) {
            $best = $file.FullName
            $bestScore = $score
            $bestRows = $rows
        }
    }
    return $best
}

if ($Help) {
    Show-Usage
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $OutRoot $timestamp
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$oldUtf8 = $env:PYTHONUTF8
$oldPipeRoute = $env:QRS_PIPELINE_ROUTE
$oldRollRoute = $env:QRS_ROLLING_ROUTE
$env:PYTHONUTF8 = "1"

try {
    switch ($Mode) {
        "engine" {
            Ensure-InputCsv $InputCsv
            $outDir = Join-Path $runRoot $Route
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null

            if ($Route -eq "legacy") {
                $rc = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", $outDir)
            } else {
                $env:QRS_PIPELINE_ROUTE = "new"
                $rc = Invoke-Python @("scripts\run_engine_compat.py", "--route", "new", "--", "--input_csv", $InputCsv, "--out_dir", $outDir)
            }
            if ($rc -ne 0) { exit $rc }
            Write-Host ("[QRS] mode=engine route={0} out_dir={1}" -f $Route, (Resolve-Path $outDir).Path)
            exit 0
        }

        "rolling" {
            $outDir = Join-Path $runRoot $Route
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null
            if ($Route -eq "legacy") {
                $rc = Invoke-Python @("rolling_runner.py")
            } else {
                $env:QRS_ROLLING_ROUTE = "new"
                $rc = Invoke-Python @("scripts\run_rolling_compat.py", "--route", "new")
            }
            if ($rc -ne 0) { exit $rc }
            Write-Host ("[QRS] mode=rolling route={0} out_root={1}" -f $Route, (Resolve-Path $runRoot).Path)
            exit 0
        }

        "compare" {
            Ensure-InputCsv $InputCsv
            $legacyDir = Join-Path $runRoot "legacy"
            $newDir = Join-Path $runRoot "new"
            New-Item -ItemType Directory -Force -Path $legacyDir, $newDir | Out-Null

            $rcLegacy = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", $legacyDir)
            if ($rcLegacy -ne 0) { Write-Error ("Legacy run failed with code {0}" -f $rcLegacy) }

            $env:QRS_PIPELINE_ROUTE = "new"
            $rcNew = Invoke-Python @("scripts\run_engine_compat.py", "--route", "new", "--", "--input_csv", $InputCsv, "--out_dir", $newDir)
            if ($rcNew -ne 0) { Write-Error ("New run failed with code {0}" -f $rcNew) }

            $oldEq = Get-BestEquityCsv $legacyDir
            $newEq = Get-BestEquityCsv $newDir
            if ([string]::IsNullOrWhiteSpace($oldEq) -or [string]::IsNullOrWhiteSpace($newEq)) {
                Write-Error ("Failed to locate equity CSV. legacy='{0}', new='{1}'" -f $oldEq, $newEq)
            }

            $diffPath = Join-Path $runRoot "regression_diff.csv"
            $rcCmp = Invoke-Python @(
                "tools\regression_compare.py",
                "--old-equity", $oldEq,
                "--new-equity", $newEq,
                "--tol-abs", "$TolAbs",
                "--tol-rel", "$TolRel",
                "--topn", "$TopN",
                "--out", $diffPath
            )

            if ($rcCmp -eq 0) {
                Write-Host "[QRS] compare=PASS"
            } else {
                Write-Host "[QRS] compare=FAIL"
            }
            Write-Host ("[QRS] legacy_equity={0}" -f $oldEq)
            Write-Host ("[QRS] new_equity={0}" -f $newEq)
            Write-Host ("[QRS] diff_csv={0}" -f $diffPath)
            exit $rcCmp
        }
    }
} finally {
    if ($null -ne $oldUtf8) { $env:PYTHONUTF8 = $oldUtf8 } else { Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue }
    if ($null -ne $oldPipeRoute) { $env:QRS_PIPELINE_ROUTE = $oldPipeRoute } else { Remove-Item Env:\QRS_PIPELINE_ROUTE -ErrorAction SilentlyContinue }
    if ($null -ne $oldRollRoute) { $env:QRS_ROLLING_ROUTE = $oldRollRoute } else { Remove-Item Env:\QRS_ROLLING_ROUTE -ErrorAction SilentlyContinue }
}
