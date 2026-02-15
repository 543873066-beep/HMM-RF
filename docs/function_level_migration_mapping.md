# Function-level migration mapping

This file was generated automatically. It lists skeleton targets in
`quant_refactor_skeleton/` and maps them to the existing modules in
`quant_refactor/` for later function-level migration.

Mapping (skeleton -> source):

- quant_refactor_skeleton/core/config.py -> quant_refactor/core/config.py
- quant_refactor_skeleton/core/io_utils.py -> quant_refactor/core/io_utils.py
- quant_refactor_skeleton/core/types.py -> quant_refactor/core/types.py
- quant_refactor_skeleton/data/loaders.py -> quant_refactor/data/loaders.py
- quant_refactor_skeleton/data/resample.py -> quant_refactor/data/resample.py
- quant_refactor_skeleton/eval/metrics.py -> quant_refactor/eval/metrics.py
- quant_refactor_skeleton/eval/regression.py -> quant_refactor/eval/regression.py
- quant_refactor_skeleton/features/feature_builder.py -> quant_refactor/features/feature_builder.py
- quant_refactor_skeleton/features/indicators.py -> quant_refactor/features/indicators.py
- quant_refactor_skeleton/hmm/model.py -> quant_refactor/hmm/model.py
- quant_refactor_skeleton/hmm/portrait.py -> quant_refactor/hmm/portrait.py
- quant_refactor_skeleton/hmm/states.py -> quant_refactor/hmm/states.py
- quant_refactor_skeleton/live/export.py -> quant_refactor/live/export.py
- quant_refactor_skeleton/overlay/overlay_builder.py -> quant_refactor/overlay/overlay_builder.py
- quant_refactor_skeleton/pipeline/engine_compat.py -> quant_refactor/pipeline/engine_compat.py
- quant_refactor_skeleton/pipeline/msp_pipeline.py -> quant_refactor/pipeline/msp_pipeline.py
- quant_refactor_skeleton/pipeline/rf_pipeline.py -> quant_refactor/pipeline/rf_pipeline.py
- quant_refactor_skeleton/rf/backtest.py -> quant_refactor/rf/backtest.py
- quant_refactor_skeleton/rf/dataset.py -> quant_refactor/rf/dataset.py
- quant_refactor_skeleton/rf/gates.py -> quant_refactor/rf/gates.py
- quant_refactor_skeleton/rf/trainer.py -> quant_refactor/rf/trainer.py
- quant_refactor_skeleton/runner/rolling_runner_adapter.py -> quant_refactor/runner/rolling_runner_adapter.py
- quant_refactor_skeleton/super_state/lifecycle.py -> quant_refactor/super_state/lifecycle.py
- quant_refactor_skeleton/super_state/super_hmm.py -> quant_refactor/super_state/super_hmm.py

Each skeleton file contains a short TODO and an empty public API to start tests
and gradual migration.

# Function-Level Migration Mapping

## Legacy Source: msp_engine_ewma_exhaustion_opt_atr_momo.py

| Legacy Function | Target Module | Target Function | Status |
|---|---|---|---|
| `_safe_read_csv_maybe_empty` | `quant_refactor/core/io_utils.py` | `safe_read_csv_maybe_empty` | skeleton |
| `safe_log` | `quant_refactor/features/indicators.py` | `safe_log` | skeleton |
| `returns` | `quant_refactor/features/indicators.py` | `returns` | skeleton |
| `rolling_vol` | `quant_refactor/features/indicators.py` | `rolling_vol` | skeleton |
| `rsi` | `quant_refactor/features/indicators.py` | `rsi` | skeleton |
| `true_range` | `quant_refactor/features/indicators.py` | `true_range` | skeleton |
| `atr` | `quant_refactor/features/indicators.py` | `atr` | skeleton |
| `adx` | `quant_refactor/features/indicators.py` | `adx` | skeleton |
| `_load_ohlcv_csv` | `quant_refactor/data/loaders.py` | `load_ohlcv_csv` | skeleton |
| `load_1m` | `quant_refactor/data/loaders.py` | `load_1m` | skeleton |
| `load_5m` | `quant_refactor/data/loaders.py` | `load_5m` | skeleton |
| `normalize_input_5m` | `quant_refactor/data/loaders.py` | `normalize_input_5m` | skeleton |
| `resample_ohlcv` | `quant_refactor/data/resample.py` | `resample_ohlcv` | skeleton |
| `make_features` | `quant_refactor/features/feature_builder.py` | `make_features` | skeleton |
| `pick_fit_mask_by_date` | `quant_refactor/features/feature_builder.py` | `pick_fit_mask_by_date` | skeleton |
| `_force_random_init_hmm` | `quant_refactor/hmm/model.py` | `force_random_init_hmm` | skeleton |
| `align_states_by_volatility` | `quant_refactor/hmm/model.py` | `align_states_by_volatility` | skeleton |
| `fit_hmm_train_infer` | `quant_refactor/hmm/model.py` | `fit_hmm_train_infer` | skeleton |
| `transition_matrix_from_states` | `quant_refactor/hmm/states.py` | `transition_matrix_from_states` | skeleton |
| `mode_agg` | `quant_refactor/hmm/states.py` | `mode_agg` | skeleton |
| `build_state_portrait` | `quant_refactor/hmm/portrait.py` | `build_state_portrait` | skeleton |
| `aggregate_5m_to_30m` | `quant_refactor/overlay/overlay_builder.py` | `aggregate_5m_to_30m` | skeleton |
| `build_overlay_30m` | `quant_refactor/overlay/overlay_builder.py` | `build_overlay_30m` | skeleton |
| `compute_avg_run_local_expected_life` | `quant_refactor/super_state/lifecycle.py` | `compute_avg_run_local_expected_life` | skeleton |
| `compute_avg_run_local_expected_life_online_ewma` | `quant_refactor/super_state/lifecycle.py` | `compute_avg_run_local_expected_life_online_ewma` | skeleton |
| `compute_run_len_so_far` | `quant_refactor/super_state/lifecycle.py` | `compute_run_len_so_far` | skeleton |
| `compute_path_stats_seed_expected_life` | `quant_refactor/super_state/lifecycle.py` | `compute_path_stats_seed_expected_life` | skeleton |
| `run_super_hmm_from_overlay` | `quant_refactor/super_state/super_hmm.py` | `run_super_hmm_from_overlay` | skeleton |
| `write_run_summary_json` | `quant_refactor/live/export.py` | `write_run_summary_json` | skeleton |
| `export_live_pack` | `quant_refactor/live/export.py` | `export_live_pack` | skeleton |
| `run_msp_pipeline` | `quant_refactor/pipeline/msp_pipeline.py` | `run_msp_pipeline` | skeleton |
| `safe_log_return` | `quant_refactor/rf/dataset.py` | `safe_log_return` | skeleton |
| `evaluate_regression` | `quant_refactor/eval/metrics.py` | `evaluate_regression` | skeleton |
| `build_features_rf` | `quant_refactor/rf/dataset.py` | `build_features_rf` | skeleton |
| `fit_gate_thresholds_robust` | `quant_refactor/rf/gates.py` | `fit_gate_thresholds_robust` | skeleton |
| `gate_mask` | `quant_refactor/rf/gates.py` | `gate_mask` | skeleton |
| `backtest_using_full_timeline_df` | `quant_refactor/rf/backtest.py` | `backtest_using_full_timeline_df` | skeleton |
| `compute_daily_sharpe` | `quant_refactor/eval/metrics.py` | `compute_daily_sharpe` | skeleton |
| `run_rf_pipeline_strict_last_month` | `quant_refactor/rf/trainer.py` | `run_rf_pipeline_strict_last_month` | skeleton |
| `main` | `quant_refactor/pipeline/engine_compat.py` | `run_engine_compat` | skeleton |

## Legacy Source: rolling_runner.py

| Legacy Function | Target Module | Target Function | Status |
|---|---|---|---|
| `ensure_dir` | `quant_refactor/core/io_utils.py` | `ensure_dir` | skeleton |
| `default_seed_triplet` | `quant_refactor/runner/rolling_runner_adapter.py` | `default_seed_triplet` | skeleton |
| `run_engine_once` | `quant_refactor/runner/rolling_runner_adapter.py` | `run_engine_once` | skeleton |
| `update_latest_live_pack` | `quant_refactor/runner/rolling_runner_adapter.py` | `update_latest_live_pack` | planned |
| `load_run_summary_optional` | `quant_refactor/runner/rolling_runner_adapter.py` | `load_run_summary_optional` | planned |
| `find_equity_curve_csv_optional` | `quant_refactor/runner/rolling_runner_adapter.py` | `find_equity_curve_csv_optional` | planned |
| `read_equity_curve` | `quant_refactor/eval/regression.py` | `read_equity_curve` | planned |
| `rescale_equity_to_1` | `quant_refactor/eval/regression.py` | `rescale_equity_to_1` | planned |
| `make_flat_curve_from_input` | `quant_refactor/eval/regression.py` | `make_flat_curve_from_input` | planned |
| `safe_mean_combo` | `quant_refactor/runner/rolling_runner_adapter.py` | `safe_mean_combo` | planned |
| `resolve_njobs` | `quant_refactor/runner/rolling_runner_adapter.py` | `resolve_njobs` | skeleton |
| `_safe_get` | `quant_refactor/runner/rolling_runner_adapter.py` | `_safe_get` | planned |
| `extract_eval_metrics_from_summary` | `quant_refactor/runner/rolling_runner_adapter.py` | `extract_eval_metrics_from_summary` | skeleton |
| `_json_sanitize` | `quant_refactor/runner/rolling_runner_adapter.py` | `_json_sanitize` | planned |
| `robust_select_top5` | `quant_refactor/runner/rolling_runner_adapter.py` | `robust_select_top5` | skeleton |
| `read_trading_days_5m` | `quant_refactor/runner/rolling_runner_adapter.py` | `read_trading_days_5m` | planned |
| `pick_backstep_endpoints` | `quant_refactor/runner/rolling_runner_adapter.py` | `pick_backstep_endpoints` | skeleton |
| `find_first_cutoff_in_month` | `quant_refactor/runner/rolling_runner_adapter.py` | `find_first_cutoff_in_month` | planned |
| `parse_engine_cost` | `quant_refactor/runner/rolling_runner_adapter.py` | `parse_engine_cost` | planned |
| `record_experiment_log` | `quant_refactor/runner/rolling_runner_adapter.py` | `record_experiment_log` | planned |
| `main` | `scripts/run_engine_compat.py` | `main` | skeleton |

## Notes

- Legacy scripts remain the runtime source of truth until each mapped function is migrated and parity-tested.
- `tools/regression_compare.py` is the canonical parity harness entrypoint.
