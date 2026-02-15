import pandas as pd

from quant_refactor.core.config import EngineConfig


def fit_gate_thresholds_robust(train_df: pd.DataFrame, allowed_states: list, cfg: EngineConfig):
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::fit_gate_thresholds_robust
    raise NotImplementedError


def gate_mask(df: pd.DataFrame, th: dict, allowed_states: list) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::gate_mask
    raise NotImplementedError
