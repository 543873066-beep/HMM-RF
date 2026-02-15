from dataclasses import dataclass
from typing import Optional


@dataclass
class EngineConfig:
    """Placeholder config for compatibility runner."""

    data_1m_csv: str
    out_dir: str
    random_seed: int = 42
    trade_start: Optional[str] = None
    trade_end: Optional[str] = None


# TODO: align fields with legacy Config in msp_engine_ewma_exhaustion_opt_atr_momo.py::Config
