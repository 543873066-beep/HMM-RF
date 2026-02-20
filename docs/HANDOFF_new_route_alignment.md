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
## 推荐用法（默认 new）
- 默认已经是 new route（不传 -Route 时即为 new）。
- 一键全回归（强制 no-fallback）：
  - `powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\full_regression -DisableLegacyEquityFallback`

## legacy 回退命令
- Engine 走 legacy：
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine -Route legacy -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\legacy_run`
- Rolling 走 legacy：
  - `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\legacy_rolling`

## 强制无兜底（no-fallback）
- 在 compare/rolling/full regression 中加：`-DisableLegacyEquityFallback`
- 用于确保结果不依赖 legacy equity 兜底。

## Routes and defaults
- `legacy` route: runs the original scripts unchanged (`msp_engine_ewma_exhaustion_opt_atr_momo.py`, `rolling_runner.py`).
- `new` route: runs the refactor pipeline under `quant_refactor_skeleton`.
- Default is still legacy. To switch:
  - Engine: set `QRS_PIPELINE_ROUTE=new` or use `scripts/run_qrs.ps1 -Mode engine -Route new`.
  - Rolling: set `QRS_ROLLING_ROUTE=new` or use `scripts/run_qrs.ps1 -Mode rolling -Route new`.

## One-click full regression (no-fallback)
Recommended (PowerShell):

`powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\full_regression -DisableLegacyEquityFallback`

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


## Troubleshooting 决策树
1) full regression FAIL
- 先看 `run_report.json` 的 `compare` 字段（status/max_abs/rows_over/diff_csv）。

2) engine compare FAIL
- 先跑 stage-diff（features/super_state/rf_inputs）。
- 再看 `run_report.json` 的 inputs/time_range 与 config。 

3) rolling FAIL
- 看 `rolling_diff_by_fold.csv` 与 `rolling_diff_summary.json`。
- 如果有 `fold_inputs_summary.json`，优先对齐 fold 输入摘要。 

4) 护栏触发（PATH_OUT_OF_ROOT / LOW_COVERAGE / ZERO_ROWS_COMPARED）
- 说明：输出目录结构或时间域不一致。
- 修复：用相同输入与 out_root 重新跑 compare/rolling，并确认 run_report.json 的 inputs/time_range 一致。

## 第一次在新机器上安装/验证
1) 安装依赖与创建 venv：
   - `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`
2) 运行一次全回归（no-fallback）：
   - `powershell -ExecutionPolicy Bypass -File scripts\one_click_full_regression.ps1 -DisableLegacyEquityFallback`
3) 看到 `FULL_REGRESSION=PASS` 即完成验证。
4) 若需要 legacy 回退，显式加 `-Route legacy`。

## Output layout (full regression)
- `out_root/<timestamp>/legacy/...`
- `out_root/<timestamp>/new/...`
- Diff files:
  - `regression_diff.csv` (engine compare)
  - `rolling_diff_by_fold.csv` and `rolling_diff_summary.json` (rolling compare)
- Reproducibility:
  - `run_report.json` is written under each new-route out_dir and includes input hash/config/artifacts.

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
