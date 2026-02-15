# Quant Refactor Structure Plan

Generated plan:

1. Create skeleton package `quant_refactor_skeleton/` mirroring `quant_refactor/`.
2. For each module, add TODOs mapping to the existing implementation (see mapping file).
3. Implement functions incrementally: copy function, add tests, delete original.
4. Update package imports to the new refactored modules once stable.

Notes:
- Skeleton files are intentionally minimal to act as targets for migration.
- Work iteratively per package (core -> data -> features -> pipeline -> rf -> hmm).

# Quant Refactor Structure Plan

## Scope
This document defines the target package structure for refactoring legacy scripts:
- `msp_engine_ewma_exhaustion_opt_atr_momo.py`
- `rolling_runner.py`

The refactor follows a staged migration with skeleton-first modules. Each skeleton keeps an explicit TODO reference to legacy functions.

## Target Structure

```text
quant_refactor/
  __init__.py
  core/
    __init__.py
    config.py
    types.py
    io_utils.py
  data/
    __init__.py
    loaders.py
    resample.py
  features/
    __init__.py
    indicators.py
    feature_builder.py
  hmm/
    __init__.py
    model.py
    states.py
    portrait.py
  overlay/
    __init__.py
    overlay_builder.py
  super_state/
    __init__.py
    lifecycle.py
    super_hmm.py
  rf/
    __init__.py
    dataset.py
    gates.py
    trainer.py
    backtest.py
  pipeline/
    __init__.py
    msp_pipeline.py
    rf_pipeline.py
    engine_compat.py
  live/
    __init__.py
    export.py
  eval/
    __init__.py
    metrics.py
    regression.py
  runner/
    __init__.py
    rolling_runner_adapter.py
```

## Migration Principles

1. Keep legacy scripts untouched during initial migration.
2. Use skeleton modules first (`NotImplementedError` + TODO pointer).
3. Migrate by function group, then wire integration.
4. Preserve behavior parity with regression checks before replacing entrypoints.

## Phase Plan

1. Phase 0: Skeleton landing
- Create package and interfaces.
- Keep `tools/regression_compare.py` and `scripts/run_engine_compat.py` as integration placeholders.

2. Phase 1: Data and features migration
- Move loading/resampling/indicator logic.
- Validate dataframe schemas and timestamp handling.

3. Phase 2: HMM and overlay migration
- Move HMM fitting/state alignment/portrait generation.
- Rebuild 5m->30m aggregation and 30m overlay composition.

4. Phase 3: Super-state and RF migration
- Move lifecycle stats + super HMM.
- Move RF feature engineering, gate thresholds, backtest.

5. Phase 4: Runner compatibility and cutover
- Migrate rolling orchestration helpers.
- Use regression tooling to compare equity and summary metrics.
- Cut over CLI/script entrypoints only after parity checks pass.

## Validation Checklist

- Same date-window slicing behavior.
- Same random-seed propagation behavior.
- Same state label alignment behavior.
- Same gating and trade filtering behavior.
- Equity-curve delta within agreed threshold.
- Summary metrics parity (Sharpe, drawdown, trade count).
