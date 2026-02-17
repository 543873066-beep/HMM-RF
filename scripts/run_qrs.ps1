[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("engine", "rolling", "compare", "stage-diff")]
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
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare [-InputCsv <csv>] [-OutRoot <dir>] [-TolAbs 1e-10] [-TolRel 1e-10] [-TopN 20]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode stage-diff [-InputCsv <csv>] [-OutRoot <dir>]"
}

function Ensure-InputCsv([string]$CsvPath) {
    if (-not (Test-Path -LiteralPath $CsvPath)) {
        Write-Error ("Input CSV not found: {0}`nSuggestions:`n1) Put CSV under data\ (required columns: time/open/high/low/close/volume)`n2) Or pass -InputCsv with a full path" -f $CsvPath)
    }
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    & python @Args | Out-Host
    return $LASTEXITCODE
}

function Get-BestEquityCsv([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) { return $null }
    $best = $null
    $bestScore = -1
    $bestRows = -1
    Get-ChildItem -LiteralPath $RootDir -Recurse -File -Filter *.csv | ForEach-Object {
        $score = 0
        $name = $_.Name.ToLowerInvariant()
        if ($name -match "equity") { $score += 4 }
        if ($name -match "curve") { $score += 2 }
        $header = ""
        try { $header = (Get-Content -LiteralPath $_.FullName -TotalCount 1) } catch {}
        $h = $header.ToLowerInvariant()
        if ($h -match "equity|eq|nav") { $score += 3 }
        if ($h -match "time") { $score += 1 }
        $rows = 0
        try { $rows = [Math]::Max(0, ((Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines - 1)) } catch {}
        if (($score -gt $bestScore) -or (($score -eq $bestScore) -and ($rows -gt $bestRows))) {
            $best = $_.FullName
            $bestScore = $score
            $bestRows = $rows
        }
    }
    return $best
}

function Get-LatestCompareRun([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) { return $null }
    return Get-ChildItem -LiteralPath $RootDir -Directory |
        Where-Object { (Test-Path (Join-Path $_.FullName "legacy")) -and (Test-Path (Join-Path $_.FullName "new")) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

if ($Help) {
    Show-Usage
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = $null
if ($Mode -ne "stage-diff") {
    $runRoot = Join-Path $OutRoot $timestamp
    New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
}

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
                Ensure-InputCsv $InputCsv
                $env:QRS_ROLLING_ROUTE = "new"
                $foldDir = Join-Path $outDir "fold_001"
                New-Item -ItemType Directory -Force -Path $foldDir | Out-Null
                $rc = Invoke-Python @(
                    "scripts\run_rolling_compat.py",
                    "--route", "new",
                    "--",
                    "--input_csv", $InputCsv,
                    "--out_dir", $foldDir
                )
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
            if ($rcCmp -eq 0) { Write-Host "[QRS] compare=PASS" } else { Write-Host "[QRS] compare=FAIL" }
            Write-Host ("[QRS] legacy_equity={0}" -f $oldEq)
            Write-Host ("[QRS] new_equity={0}" -f $newEq)
            Write-Host ("[QRS] diff_csv={0}" -f $diffPath)
            exit $rcCmp
        }
        "stage-diff" {
            Ensure-InputCsv $InputCsv
            $latest = Get-LatestCompareRun $OutRoot
            if ($null -eq $latest) {
                Write-Host ("[QRS] no compare run under {0}, bootstrap compare first..." -f $OutRoot)
                $compareRc = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", (Join-Path (Join-Path $OutRoot $timestamp) "legacy"))
                if ($compareRc -ne 0) { Write-Error ("Bootstrap legacy run failed: {0}" -f $compareRc) }
                $env:QRS_PIPELINE_ROUTE = "new"
                $newRc = Invoke-Python @("scripts\run_engine_compat.py", "--route", "new", "--", "--input_csv", $InputCsv, "--out_dir", (Join-Path (Join-Path $OutRoot $timestamp) "new"))
                if ($newRc -ne 0) { Write-Error ("Bootstrap new run failed: {0}" -f $newRc) }
                $latest = Get-LatestCompareRun $OutRoot
                if ($null -eq $latest) { Write-Error ("No valid compare run found after bootstrap under {0}" -f $OutRoot) }
            }
            $legacyDir = Join-Path $latest.FullName "legacy"
            $newDir = Join-Path $latest.FullName "new"
            if ((-not (Test-Path -LiteralPath $legacyDir)) -or (-not (Test-Path -LiteralPath $newDir))) {
                Write-Error ("Stage artifacts missing. legacy='{0}', new='{1}'" -f $legacyDir, $newDir)
            }
            $rcStage = Invoke-Python @("tools\stage_diff.py", "--legacy-dir", $legacyDir, "--new-dir", $newDir, "--input-csv", $InputCsv)
            exit $rcStage
        }
    }
}
finally {
    if ($null -ne $oldUtf8) { $env:PYTHONUTF8 = $oldUtf8 } else { Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue }
    if ($null -ne $oldPipeRoute) { $env:QRS_PIPELINE_ROUTE = $oldPipeRoute } else { Remove-Item Env:\QRS_PIPELINE_ROUTE -ErrorAction SilentlyContinue }
    if ($null -ne $oldRollRoute) { $env:QRS_ROLLING_ROUTE = $oldRollRoute } else { Remove-Item Env:\QRS_ROLLING_ROUTE -ErrorAction SilentlyContinue }
}
