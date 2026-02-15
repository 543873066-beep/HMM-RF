import pandas as pd


def safe_log(x: pd.Series) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::safe_log
    raise NotImplementedError


def returns(close: pd.Series) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::returns
    raise NotImplementedError


def rolling_vol(ret: pd.Series, win: int) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::rolling_vol
    raise NotImplementedError


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::rsi
    raise NotImplementedError


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::true_range
    raise NotImplementedError


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::atr
    raise NotImplementedError


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::adx
    raise NotImplementedError
