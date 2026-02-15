from typing import List

import pandas as pd


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


def rescale_equity_to_1(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    base = float(df["equity"].iloc[0])
    if base == 0:
        return df
    out = df.copy()
    out["equity"] = out["equity"] / base
    return out


def make_flat_curve_from_input(csv_path: str, trade_start_dt: pd.Timestamp, trade_end_dt: pd.Timestamp, value: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df[(df["time"] >= trade_start_dt) & (df["time"] <= trade_end_dt)].sort_values("time")
    if df.empty:
        t = pd.to_datetime([trade_start_dt, trade_end_dt])
        return pd.DataFrame({"time": t, "equity": [value, value]})
    return pd.DataFrame({"time": df["time"].values, "equity": value})


def safe_mean_combo(df: pd.DataFrame, cols: List[str], mode: str) -> pd.Series:
    if mode == "mean_fixed5":
        return df[cols].mean(axis=1)
    return df[cols].mean(axis=1, skipna=True)


__all__ = [
    "make_flat_curve_from_input",
    "read_equity_curve",
    "rescale_equity_to_1",
    "safe_mean_combo",
]
