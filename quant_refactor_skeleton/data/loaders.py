from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}, current columns: {df.columns.tolist()}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    df = df.set_index("time")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    return _load_ohlcv_csv(csv_path)


def load_1m(csv_path: str) -> pd.DataFrame:
    return _load_ohlcv_csv(csv_path)


def load_5m(csv_path: str) -> pd.DataFrame:
    return _load_ohlcv_csv(csv_path)


def _cfg_get(cfg: Optional[Any], key: str, default):
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def normalize_input_5m(df_5m: pd.DataFrame, cfg: Optional["Config"] = None) -> pd.DataFrame:
    df = df_5m.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    tf_5m = _cfg_get(cfg, "tf_5m", "5min")
    min_count_5m = int(_cfg_get(cfg, "min_count_5m", 1))
    bars = df.resample(tf_5m, label="right", closed="right").agg(AGG).dropna()
    cnt = df["close"].resample(tf_5m, label="right", closed="right").count()
    bars = bars[cnt.reindex(bars.index).fillna(0).astype(int) >= min_count_5m]
    return bars
