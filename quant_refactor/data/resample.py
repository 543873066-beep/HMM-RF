import pandas as pd

from quant_refactor.core.config import EngineConfig


def resample_ohlcv(df_base: pd.DataFrame, rule: str, cfg: EngineConfig) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::resample_ohlcv
    raise NotImplementedError
