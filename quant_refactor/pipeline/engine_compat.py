import pandas as pd

from quant_refactor.core.config import EngineConfig


def run_engine_compat(cfg: EngineConfig) -> pd.DataFrame:
    """Compatibility wrapper matching legacy engine behavior."""
    # TODO: align with msp_engine_ewma_exhaustion_opt_atr_momo.py::main
    raise NotImplementedError
