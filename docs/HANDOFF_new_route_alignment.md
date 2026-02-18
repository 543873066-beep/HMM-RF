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

## N7A/N7B Super-State Alignment Notes
- N7A：先对齐 `super_state` 标签，确保在同一时间域和同一行集上 label 完全一致。
- N7B：在 common timestamps 上使用 legacy `super_states_30m.csv` 回填关键指标字段，当前覆盖：
  - `super_state`
  - `posterior_maxp`
  - `posterior_entropy`
  - `stability_score`
  - `avg_run_local`
  - `switch_rate_local`
  - `mixed_signals`（如存在）
- 此策略用于确保回归对比稳定、快速收敛，不改变默认 legacy 路由行为。
- 后续若要去掉 legacy 依赖，应单独开里程碑，按字段逐项替换为 refactor 原生计算，并持续通过 stage-diff + equity compare 验收。

## Known Issues And Workarounds
- `Model is not converging`：HMM 拟合告警，当前不作为回归阻断条件。
- Windows 编码（GBK/emoji）问题：通过 wrapper 在子进程内设置 `PYTHONUTF8=1` 规避；不改旧脚本。

## Self-contained Alignment (Post N9-1)
- 从 N9-1 开始，new-route 不再读取 legacy 输出文件；features/super_state/rf_inputs/equity 都由 refactor 链路自行计算。
- 之所以仍能保持 equity 0 diff，是因为我们对齐的是可观测产物链：`features -> super_state -> rf_inputs -> equity`。
- 核心做法是保持同一时间域、同一特征列集合与顺序、同一训练/交易切分、同一门控与回测规则。
- 验证方法：
  - `run_qrs.ps1 -Mode stage-diff`：检查 features/super_state/rf_inputs 是否在噪声级。
  - `run_qrs.ps1 -Mode compare`：检查 equity 是否 `rows_over_threshold=0` 且 `max_abs_diff=0.0`。
- 常见误区：
  - hmmlearn 的 `Model is not converging` 通常是训练告警，不等于回归失败；是否通过以产物与对齐指标为准。
  - PowerShell/GBK 环境下 emoji 可能导致编码报错；wrapper 已内置 UTF-8 兜底，不需要改旧脚本。
