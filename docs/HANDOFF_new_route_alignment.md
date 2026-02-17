# HANDOFF: New Route Alignment

## Current Status
当前仓库保持“默认 legacy，显式切换 new”的双路由模式：不设置环境变量时所有入口继续走旧脚本；设置 route/new 环境变量后走 `quant_refactor_skeleton` 新链路。新增 wrapper 的目的是把 Windows 下编码、目录组织、双跑对比和路径定位统一起来，减少手工误操作。

## Route Switches And Entrypoints
- `QRS_PIPELINE_ROUTE=new`：引擎入口切到 new route；默认未设置时为 legacy。
- `QRS_ROLLING_ROUTE=new`：rolling 入口切到 new route；默认未设置时为 legacy。
- 引擎兼容入口：`scripts/run_engine_compat.py`
- rolling 兼容入口：`scripts/run_rolling_compat.py`
- 统一 wrapper：`scripts/run_qrs.ps1`

## Wrapper Usage Examples
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine -Route legacy`
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode engine -Route new`
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route legacy`
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode rolling -Route new`
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare`
- `powershell -ExecutionPolicy Bypass -File scripts\run_qrs.ps1 -Mode compare -InputCsv data\sh000852_5m.csv -OutRoot outputs_rebuild\manual_compare`

## Output Layout Convention
`scripts/run_qrs.ps1` 使用 `OutRoot\<timestamp>\...` 组织输出。示例：

```text
outputs_rebuild/
  qrs_runs/
    20260217_140000/
      legacy/
      new/
      regression_diff.csv
```

## Alignment Strategy (Current And Next)
- 当前 N4 对齐策略：`rf.alignment_fallback_legacy`，new route 在骨架阶段后回落 legacy 产出 equity，确保回归对齐 PASS。
- 下一阶段目标：逐段替换并保持对齐，顺序建议为 `data -> features -> overlay -> super_state -> rf`，每段单独验收/回归。

## Known Issues And Workarounds
- `Model is not converging`：HMM 拟合告警，当前不作为回归阻断条件。
- Windows 编码（GBK/emoji）问题：通过 wrapper 在子进程内设置 `PYTHONUTF8=1` 规避；不改旧脚本。
