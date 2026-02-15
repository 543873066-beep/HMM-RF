import pandas as pd

from quant_refactor.core.config import EngineConfig


def run_rf_pipeline(super_df: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    # TODO: bridge to quant_refactor.rf.trainer.run_rf_pipeline_strict_last_month
    raise NotImplementedError
