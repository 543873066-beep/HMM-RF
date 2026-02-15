import pandas as pd

from quant_refactor.core.config import EngineConfig


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::_load_ohlcv_csv
    raise NotImplementedError


def load_1m(csv_path: str) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::load_1m
    raise NotImplementedError


def load_5m(csv_path: str) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::load_5m
    raise NotImplementedError


def normalize_input_5m(df_5m: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::normalize_input_5m
    raise NotImplementedError
