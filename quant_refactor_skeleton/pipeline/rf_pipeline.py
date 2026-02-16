from __future__ import annotations

# assembly-only module imports for migration wiring
from quant_refactor_skeleton.rf import backtest as rf_backtest
from quant_refactor_skeleton.rf import dataset as rf_dataset
from quant_refactor_skeleton.rf import gates as rf_gates
from quant_refactor_skeleton.rf import trainer as rf_trainer


def run_rf_pipeline(super_df, cfg):
    return int(rf_trainer.run_rf_stage_placeholder(argv=[]))


def run_rf_pipeline_placeholder(argv=None) -> int:
    return int(rf_trainer.run_rf_stage_placeholder(argv=argv))


__all__ = ["rf_backtest", "rf_dataset", "rf_gates", "rf_trainer", "run_rf_pipeline", "run_rf_pipeline_placeholder"]
