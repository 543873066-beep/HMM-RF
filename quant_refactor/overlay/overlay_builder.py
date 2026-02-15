import pandas as pd

from quant_refactor.core.config import EngineConfig


def aggregate_5m_to_30m(states_5m: pd.DataFrame, cfg: EngineConfig) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::aggregate_5m_to_30m
    raise NotImplementedError


def build_overlay_30m(states_30m: pd.DataFrame, agg5_to_30: pd.DataFrame, states_1d: pd.DataFrame) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::build_overlay_30m
    raise NotImplementedError
