# HANDOFF: New Route Alignment

## Current status
- Default behavior stays legacy.
- New route is enabled only by explicit switch:
  - `QRS_PIPELINE_ROUTE=new`
  - `QRS_ROLLING_ROUTE=new`
- `scripts/run_qrs.ps1` is the main wrapper for engine/rolling/compare/stage-diff.

## Self-contained alignment (post N9-1)
- New route no longer reads legacy artifacts when backfill is off.
- Alignment is validated on this observable chain:
  - `features -> super_state -> rf_inputs -> equity`
- Why equity can still be 0-diff:
  - same sample domain
  - same feature set/order
  - same train/trade split
  - same gate/backtest rules



## ??????? new?
- ????? new route??? -Route ??? new??
- ???????? no-fallback??
  - `powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuildull_regression -DisableLegacyEquityFallback`

## legacy ????
- Engine ? legacy?
  - `powershell -ExecutionPolicy Bypass -File scripts
un_qrs.ps1 -Mode engine -Route legacy -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\legacy_run`
- Rolling ? legacy?
  - `powershell -ExecutionPolicy Bypass -File scripts
un_qrs.ps1 -Mode rolling -Route legacy -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\legacy_rolling`

## ??????no-fallback?
- ? compare/rolling/full regression ???`-DisableLegacyEquityFallback`
- ????????? legacy equity ???

## Routes and defaults
- `legacy` route: runs the original scripts unchanged (`msp_engine_ewma_exhaustion_opt_atr_momo.py`, `rolling_runner.py`).
- `new` route: runs the refactor pipeline under `quant_refactor_skeleton`.
- Default is still legacy. To switch:
  - Engine: set `QRS_PIPELINE_ROUTE=new` or use `scripts/run_qrs.ps1 -Mode engine -Route new`.
  - Rolling: set `QRS_ROLLING_ROUTE=new` or use `scripts/run_qrs.ps1 -Mode rolling -Route new`.

## One-click full regression (no-fallback)
Recommended (PowerShell):

`powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuildull_regression -DisableLegacyEquityFallback`

What it does:
- engine compare (legacy vs new)
- stage-diff (features / super_state / rf_inputs)
- rolling compare (fold-by-fold equity)

PASS criteria:
- engine compare: `rows_over_threshold=0` and `max_abs_diff=0`
- rolling compare: `overall=PASS`

## Common failures and how to check
- ENGINE_COMPARE FAIL:
  - open `regression_diff.csv` in the compare output root
  - confirm `rolling`-only logic is not applied in engine mode (see N12-1A guard)
- ROLLING FAIL:
  - check `rolling_diff_by_fold.csv` and `rolling_diff_summary.json`
- HMM warning `Model is not converging`:
  - this is a training warning, not an automatic failure; only compare results decide PASS/FAIL
- Windows encoding:
  - if you see garbled output, set `PYTHONUTF8=1` before running or use the wrapper (no legacy edits)


## Troubleshooting ???
1) full regression FAIL
- ?? `run_report.json` ? `compare` ???status/max_abs/rows_over/diff_csv??

2) engine compare FAIL
- ?? stage-diff?features/super_state/rf_inputs??
- ?? `run_report.json` ? inputs/time_range ? config? 

3) rolling FAIL
- ? `rolling_diff_by_fold.csv` ? `rolling_diff_summary.json`?
- ??? `fold_inputs_summary.json`????? fold ????? 

4) ?????PATH_OUT_OF_ROOT / LOW_COVERAGE / ZERO_ROWS_COMPARED?
- ?????????????????
- ????????? out_root ??? compare/rolling???? run_report.json ? inputs/time_range ???

## Output layout (full regression)
- `out_root/<timestamp>/legacy/...`
- `out_root/<timestamp>/new/...`
- Diff files:
  - `regression_diff.csv` (engine compare)
  - `rolling_diff_by_fold.csv` and `rolling_diff_summary.json` (rolling compare)
- Reproducibility:
  - `run_manifest.json` is written under each new-route out_dir and includes input hash/config/artifacts.

## Validation commands
- Stage-level diff:
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode stage-diff -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\stage_check`
- Equity compare:
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\compare_check`

## Rolling alignment quick run
- Run legacy rolling:
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\rolling_legacy`
- Run new rolling:
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route new -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\rolling_new`
- Fold-by-fold compare:
  - `python tools\rolling_compare.py --legacy-root outputs_rebuild\rolling_legacy --new-root outputs_rebuild\rolling_new --tol-abs 1e-10 --tol-rel 1e-10 --topn 20`

## Common failures and workaround
- HMM warning `Model is not converging`:
  - this is a training warning, not an automatic failure; use stage-diff/equity compare as the pass criteria.
- PowerShell GBK/UTF-8 issue:
  - wrapper sets `PYTHONUTF8=1` inside script process, no legacy script edit required.
- Input path missing:
  - put CSV in `data\` with `time/open/high/low/close/volume`, or pass `-InputCsv`.
- Fold mismatch in rolling compare:
  - ensure both roots come from runs on the same input/config and check `rolling_diff_summary.json`.
