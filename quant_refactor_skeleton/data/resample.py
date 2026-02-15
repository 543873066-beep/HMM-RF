from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _cfg_get(cfg: Optional[Any], key: str, default):
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def resample_ohlcv(df_base: pd.DataFrame, rule: str, cfg: Optional["Config"] = None) -> pd.DataFrame:
    AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    rule_u = str(rule).upper()
    if rule_u in ["1D", "D"]:
        out = df_base.resample("1D", label="left", closed="left").agg(AGG).dropna()
        return out
    out = df_base.resample(rule, label="right", closed="right").agg(AGG)
    cnt = df_base["close"].resample(rule, label="right", closed="right").count()
    r = str(rule).lower()
    if r in ["30min", "30t"]:
        min_cnt = int(_cfg_get(cfg, "min_count_30m", 1))
    elif r in ["5min", "5t"]:
        min_cnt = int(_cfg_get(cfg, "min_count_5m", 1))
    else:
        min_cnt = 1
    out = out.dropna()
    out = out[cnt.reindex(out.index).fillna(0).astype(int) >= min_cnt]
    return out
