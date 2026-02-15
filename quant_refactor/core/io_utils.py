from pathlib import Path

import pandas as pd


def safe_read_csv_maybe_empty(path: str) -> pd.DataFrame:
    """Read CSV or return empty frame when file is absent/empty."""
    # TODO: port logic from msp_engine_ewma_exhaustion_opt_atr_momo.py::_safe_read_csv_maybe_empty
    raise NotImplementedError


def ensure_dir(path: str) -> Path:
    """Create directory if missing and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# TODO: align with rolling_runner.py::ensure_dir behavior if differences exist
