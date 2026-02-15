import numpy as np
import pandas as pd

from quant_refactor.core.config import EngineConfig


def force_random_init_hmm(model, x_fit_s: np.ndarray, seed: int, min_covar: float = 1e-3):
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::_force_random_init_hmm
    raise NotImplementedError


def align_states_by_volatility(model, x_sample=None):
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::align_states_by_volatility
    raise NotImplementedError


def fit_hmm_train_infer(df: pd.DataFrame, cfg: EngineConfig, tf_name: str) -> pd.DataFrame:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::fit_hmm_train_infer
    raise NotImplementedError
