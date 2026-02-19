[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("engine", "rolling", "compare", "stage-diff")]
    [string]$Mode,

    [ValidateSet("legacy", "new")]
    [AllowNull()]
    [AllowEmptyString()]
    [string]$Route = $null,

    [string]$InputCsv = "data\sh000852_5m.csv",
    [string]$OutRoot = "outputs_rebuild\qrs_runs",
    [Nullable[int]]$LegacyBackfill = $null,
    [switch]$DisableLegacyEquityFallback,
    [switch]$SkipStageDiff,
    [double]$TolAbs = 1e-10,
    [double]$TolRel = 1e-10,
    [int]$TopN = 20,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy|new [-InputCsv <csv>] [-OutRoot <dir>] [-DisableLegacyEquityFallback]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare [-InputCsv <csv>] [-OutRoot <dir>] [-TolAbs 1e-10] [-TolRel 1e-10] [-TopN 20] [-SkipStageDiff]"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode stage-diff [-InputCsv <csv>] [-OutRoot <dir>]"
    Write-Host ""
    Write-Host "Rolling compare example:"
    Write-Host "  python tools\rolling_compare.py --legacy-root outputs_rebuild\n10_2_rolling_legacy --new-root outputs_rebuild\n10_2_rolling_new --tol-abs 1e-10 --tol-rel 1e-10 --topn 20"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\one_click_rolling_compare.ps1"
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


function Resolve-RouteAndSource([string]$Mode, [string]$RouteArg) {
    if (-not [string]::IsNullOrWhiteSpace($RouteArg)) {
        return @{ Route = $RouteArg; Source = "param" }
    }
    if ($Mode -eq "rolling") {
        if (-not [string]::IsNullOrWhiteSpace($env:QRS_ROLLING_ROUTE)) {
            return @{ Route = $env:QRS_ROLLING_ROUTE; Source = "env" }
        }
    } else {
        if (-not [string]::IsNullOrWhiteSpace($env:QRS_PIPELINE_ROUTE)) {
            return @{ Route = $env:QRS_PIPELINE_ROUTE; Source = "env" }
        }
    }
    return @{ Route = "new"; Source = "default" }
}

function Resolve-LegacyBackfillValue {
    if ($null -ne $LegacyBackfill) {
        return ([int]$LegacyBackfill -ne 0)
    }
    $envVal = $env:QRS_LEGACY_BACKFILL
    if ([string]::IsNullOrWhiteSpace($envVal)) {
        return $false
    }
    $v = $envVal.Trim().ToLowerInvariant()
    return ($v -in @("1", "true", "yes", "y", "on"))
}

function New-RouteArgs([string]$CsvPath, [string]$OutDir) {
    $backfill = Resolve-LegacyBackfillValue
    $bf = if ($backfill) { "1" } else { "0" }
    $disableFallback = if ($DisableLegacyEquityFallback) { "1" } else { "0" }
    return @(
        "--", "--input_csv", $CsvPath, "--out_dir", $OutDir,
        "--enable_legacy_backfill", $bf,
        "--disable_legacy_equity_fallback_in_rolling", $disableFallback
    )
}

function Build-NewRouteArgs([string]$CsvPath, [string]$OutDir, [hashtable]$ExtraArgs) {
    $args = New-RouteArgs $CsvPath $OutDir
    if ($null -ne $ExtraArgs) {
        foreach ($k in $ExtraArgs.Keys) {
            $v = $ExtraArgs[$k]
            if (-not [string]::IsNullOrWhiteSpace([string]$v)) {
                $args += @("--$k", [string]$v)
            }
        }
    }
    return $args
}

function Apply-LegacyBackfillEnv {
    $backfill = Resolve-LegacyBackfillValue
    if ($backfill) {
        $env:QRS_LEGACY_BACKFILL = "1"
    } else {
        $env:QRS_LEGACY_BACKFILL = "0"
    }
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

function Get-RollingEquityCsvPaths([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) { return @() }
    $patterns = @("backtest_equity_curve.csv")
    $matches = @()
    foreach ($p in $patterns) {
        $matches += Get-ChildItem -LiteralPath $RootDir -Recurse -File -Filter $p -ErrorAction SilentlyContinue
    }
    return $matches | Sort-Object FullName -Unique
}

function Sync-LegacyRollingEquity([string]$TargetRoot) {
    $legacyCandidates = Get-ChildItem -LiteralPath "outputs_roll\runs" -Recurse -File -Filter "backtest_equity_curve.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if ($legacyCandidates.Count -eq 0) { return @() }

    $latest = $legacyCandidates[0].FullName
    $dst = Join-Path $TargetRoot "fold_001\rf_h4_per_state_dynamic_selected\backtest_equity_curve.csv"
    $dstDir = Split-Path -Path $dst -Parent
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    Copy-Item -LiteralPath $latest -Destination $dst -Force
    return @($dst)
}

function Get-LegacyRollingContext {
    $ctx = @{}
    $runsRoot = "outputs_roll\runs"
    if (-not (Test-Path -LiteralPath $runsRoot)) { return $ctx }
    $cycleDir = Get-ChildItem -LiteralPath $runsRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $cycleDir) { return $ctx }
    $ctx["cycle_dir"] = $cycleDir.FullName
    if ($cycleDir.Name -match "^C\d+_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})$") {
        $ctx["data_end_date"] = "$($Matches[1]) 15:00:00"
    }
    $top5 = Join-Path $cycleDir.FullName "top5_selection.csv"
    if (Test-Path -LiteralPath $top5) {
        try {
            $rows = Import-Csv -LiteralPath $top5
            if ($rows.Count -gt 0) {
                $r0 = $rows[0]
                if ($null -ne $r0.trade_start -and $null -ne $r0.trade_end) {
                    $ctx["trade_start_date"] = [string]$r0.trade_start
                    $ctx["trade_end_date"] = [string]$r0.trade_end
                }
                if ($null -ne $r0.rs_5m) { $ctx["rs_5m"] = [string]$r0.rs_5m }
                if ($null -ne $r0.rs_30m) { $ctx["rs_30m"] = [string]$r0.rs_30m }
                if ($null -ne $r0.rs_1d) { $ctx["rs_1d"] = [string]$r0.rs_1d }
            }
        } catch {}
    }
    return $ctx
}

function Get-LatestCompareRun([string]$RootDir) {
    if (-not (Test-Path -LiteralPath $RootDir)) { return $null }
    return Get-ChildItem -LiteralPath $RootDir -Directory |
        Where-Object { (Test-Path (Join-Path $_.FullName "legacy")) -and (Test-Path (Join-Path $_.FullName "new")) } |
        Sort-Object Name -Descending |
        Select-Object -First 1
}

if ($Help) {
    Show-Usage
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resolved = Resolve-RouteAndSource $Mode $Route
$Route = $resolved.Route
$RouteSource = $resolved.Source
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
            $bfText = if (Resolve-LegacyBackfillValue) { "on" } else { "off" }
            Write-Host ("[QRS] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2} route_source={3}" -f $Route, $bfText, ($(if ($DisableLegacyEquityFallback) { 'on' } else { 'off' })), $RouteSource)
            if ($Route -eq "legacy") {
                $rc = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", $outDir)
            } else {
                Apply-LegacyBackfillEnv
                $env:QRS_PIPELINE_ROUTE = $Route
                $rc = Invoke-Python (@("scripts\run_engine_compat.py", "--route", $Route) + (New-RouteArgs $InputCsv $outDir))
            }
            if ($rc -ne 0) { exit $rc }
            Write-Host ("[QRS] mode=engine route={0} out_dir={1}" -f $Route, (Resolve-Path $outDir).Path)
            exit 0
        }
        "rolling" {
            $outDir = Join-Path $runRoot $Route
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null
            $bfText = if (Resolve-LegacyBackfillValue) { "on" } else { "off" }
            Write-Host ("[QRS] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2} route_source={3}" -f $Route, $bfText, ($(if ($DisableLegacyEquityFallback) { 'on' } else { 'off' })), $RouteSource)
            if ($Route -eq "legacy") {
                $rc = Invoke-Python @("rolling_runner.py")
                if ($rc -eq 0) {
                    $synced = Sync-LegacyRollingEquity $outDir
                    if ($synced.Count -gt 0) {
                        Write-Host "[QRS] synced legacy rolling equity:"
                        foreach ($p in $synced) { Write-Host ("[QRS] equity={0}" -f (Resolve-Path $p).Path) }
                    }
                }
            } else {
                Ensure-InputCsv $InputCsv
                Apply-LegacyBackfillEnv
                $env:QRS_ROLLING_ROUTE = "new"
                $foldDir = Join-Path $outDir "fold_001"
                New-Item -ItemType Directory -Force -Path $foldDir | Out-Null
                $env:QRS_PIPELINE_ROUTE = "new"
                $legacyCtx = Get-LegacyRollingContext
                $extra = @{
                    "run_id" = "fold_001"
                    "enable_backtest" = "1"
                    "data_end_date" = $legacyCtx["data_end_date"]
                    "trade_start_date" = $legacyCtx["trade_start_date"]
                    "trade_end_date" = $legacyCtx["trade_end_date"]
                    "rs_5m" = $legacyCtx["rs_5m"]
                    "rs_30m" = $legacyCtx["rs_30m"]
                    "rs_1d" = $legacyCtx["rs_1d"]
                }
                Write-Host ("[QRS] rolling fold_001 argv: data_end={0}, trade_start={1}, trade_end={2}, rs=({3},{4},{5})" -f $extra["data_end_date"], $extra["trade_start_date"], $extra["trade_end_date"], $extra["rs_5m"], $extra["rs_30m"], $extra["rs_1d"])
                $rc = Invoke-Python (@("scripts\run_engine_compat.py", "--route", "new") + (Build-NewRouteArgs $InputCsv $foldDir $extra))
            }
            if ($rc -ne 0) { exit $rc }
            $resolvedOutRoot = (Resolve-Path $runRoot).Path
            Write-Host ("[QRS] mode=rolling route={0} out_root={1}" -f $Route, $resolvedOutRoot)
            $eqFiles = Get-RollingEquityCsvPaths $runRoot
            if ($eqFiles.Count -gt 0) {
                Write-Host "[QRS] rolling equity files:"
                foreach ($f in $eqFiles) {
                    Write-Host ("[QRS] equity={0}" -f $f.FullName)
                }
            } else {
                Write-Host "[QRS] rolling equity files: none found"
            }
            exit 0
        }
        "compare" {
            Ensure-InputCsv $InputCsv
            $bfText = if (Resolve-LegacyBackfillValue) { "on" } else { "off" }
            Write-Host ("[QRS] route={0} legacy_backfill={1} DisableLegacyEquityFallback={2} route_source={3}" -f $Route, $bfText, ($(if ($DisableLegacyEquityFallback) { 'on' } else { 'off' })), $RouteSource)
            $legacyDir = Join-Path $runRoot "legacy"
            $newDir = Join-Path $runRoot "new"
            New-Item -ItemType Directory -Force -Path $legacyDir, $newDir | Out-Null

            $rcLegacy = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", $legacyDir)
            if ($rcLegacy -ne 0) { Write-Error ("Legacy run failed with code {0}" -f $rcLegacy) }

            $env:QRS_PIPELINE_ROUTE = "new"
            Apply-LegacyBackfillEnv
            $rcNew = Invoke-Python (@("scripts\run_engine_compat.py", "--route", "new") + (New-RouteArgs $InputCsv $newDir))
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
            if (($rcCmp -eq 0) -and (-not $SkipStageDiff)) {
                Write-Host "[QRS] compare passed, running stage-diff..."
                $rcStageAfter = Invoke-Python @("tools\stage_diff.py", "--legacy-dir", $legacyDir, "--new-dir", $newDir, "--input-csv", $InputCsv)
                if ($rcStageAfter -ne 0) { exit $rcStageAfter }
            }
            exit $rcCmp
        }
        "stage-diff" {
            Ensure-InputCsv $InputCsv
            $latest = Get-LatestCompareRun $OutRoot
            if ($null -eq $latest) {
                Write-Host ("[QRS] no compare run under {0}, bootstrap compare first..." -f $OutRoot)
                $compareRc = Invoke-Python @("msp_engine_ewma_exhaustion_opt_atr_momo.py", "--input_csv", $InputCsv, "--out_dir", (Join-Path (Join-Path $OutRoot $timestamp) "legacy"))
                if ($compareRc -ne 0) { Write-Error ("Bootstrap legacy run failed: {0}" -f $compareRc) }
                Apply-LegacyBackfillEnv
                $env:QRS_PIPELINE_ROUTE = $Route
                $newRc = Invoke-Python (@("scripts\run_engine_compat.py", "--route", $Route) + (New-RouteArgs $InputCsv (Join-Path (Join-Path $OutRoot $timestamp) "new")))
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
