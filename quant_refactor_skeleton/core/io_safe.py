"""Safe I/O helpers migrated from legacy scripts."""

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


def read_equity_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    eq_col = None
    for c in ["equity", "eq", "equity_net", "equity_curve", "eq_net"]:
        if c in df.columns:
            eq_col = c
            break
    if eq_col is None:
        eq_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    return df[[tcol, eq_col]].rename(columns={tcol: "time", eq_col: "equity"})
