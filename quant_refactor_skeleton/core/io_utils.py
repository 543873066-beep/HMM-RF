import os

import pandas as pd


def _safe_read_csv_maybe_empty(path: str) -> pd.DataFrame:
    if (not path) or (not os.path.exists(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def safe_read_csv_maybe_empty(path: str) -> pd.DataFrame:
    return _safe_read_csv_maybe_empty(path)
