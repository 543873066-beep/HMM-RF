from typing import Dict, Optional

import pandas as pd

from quant_refactor.core.config import EngineConfig


def write_run_summary_json(out_dir: str, cfg: EngineConfig, extra: Optional[Dict] = None) -> None:
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::write_run_summary_json
    raise NotImplementedError


def export_live_pack(out_dir: str, cfg: EngineConfig, trade_df_snapshot: Optional[pd.DataFrame] = None):
    # TODO: port from msp_engine_ewma_exhaustion_opt_atr_momo.py::export_live_pack
    raise NotImplementedError
