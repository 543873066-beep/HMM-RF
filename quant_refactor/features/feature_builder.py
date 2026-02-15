import pandas as pd

from quant_refactor.core.config import EngineConfig


def make_features(df: pd.DataFrame, cfg: EngineConfig, tf_name: str) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::make_features
    raise NotImplementedError


def pick_fit_mask_by_date(df: pd.DataFrame, cfg: EngineConfig, tf_name: str) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::pick_fit_mask_by_date
    raise NotImplementedError
