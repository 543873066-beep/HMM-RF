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
