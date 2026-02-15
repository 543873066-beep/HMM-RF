from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quant_refactor_skeleton.features.indicators import adx, atr, returns, rolling_vol, rsi, safe_log


def make_features(df: pd.DataFrame, cfg: "Config", tf_name: str) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = returns(out["close"])
    out["log_ret_1"] = safe_log(out["close"]).diff()
    out["ma_fast"] = out["close"].rolling(cfg.ma_fast, min_periods=cfg.ma_fast).mean()
    out["ma_slow"] = out["close"].rolling(cfg.ma_slow, min_periods=cfg.ma_slow).mean()
    out["trend_strength"] = (out["close"] - out["ma_slow"]) / (out["ma_slow"] + 1e-12)
    out["vol_short"] = rolling_vol(out["ret_1"], cfg.vol_short)
    out["vol_long"] = rolling_vol(out["ret_1"], cfg.vol_long)
    out["vol_ratio"] = out["vol_short"] / (out["vol_long"] + 1e-12)
    out["rsi_14"] = rsi(out["close"], cfg.rsi_n)
    out["atr_14"] = atr(out["high"], out["low"], out["close"], cfg.atr_n)
    out["atr_pct_14"] = out["atr_14"] / (out["close"] + 1e-12)
    out["adx_14"] = adx(out["high"], out["low"], out["close"], cfg.adx_n)
    bars_15 = 15
    out["price_change_15"] = out["close"].pct_change(bars_15)
    out["volatility_15"] = rolling_vol(out["ret_1"], bars_15)
    out["volume_mean_15"] = out["volume"].rolling(bars_15, min_periods=bars_15).mean()
    out["volume_ratio_15"] = out["volume"] / (out["volume_mean_15"] + 1e-12)
    out["momentum_10"] = out["close"].pct_change(10)
    out["momentum_30"] = out["close"].pct_change(30)
    out["momentum_60"] = out["close"].pct_change(60)
    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)

    out["bar_hour"] = out.index.hour
    out["bar_minute"] = out.index.minute
    opening_mask = (
        ((out["bar_hour"] == 9) & (out["bar_minute"].isin([35, 40, 45, 50, 55])))
        | ((out["bar_hour"] == 10) & (out["bar_minute"] == 0))
    )
    shrink_factors = [0.25, 0.4, 0.5, 0.625, 0.714, 0.833]
    out["opening_order"] = np.nan
    opening_bars = out[opening_mask].index
    for i, idx in enumerate(opening_bars):
        out.at[idx, "opening_order"] = i
    vol_features = ["volume_ratio_15", "vol_short", "volatility_15", "volume"]
    for feat in vol_features:
        if feat in out.columns:
            mask = out["opening_order"].notna()
            order = out.loc[mask, "opening_order"].astype(int)
            shrink = np.array(
                [shrink_factors[min(o, len(shrink_factors) - 1)] if o < len(shrink_factors) else 1.0 for o in order]
            )
            out[feat] = out[feat].astype("float64")
            out.loc[mask, feat] *= shrink
    out = out.drop(columns=["bar_hour", "bar_minute", "opening_order"], errors="ignore")
    for c in out.columns:
        if c not in ["open", "high", "low", "close", "volume"]:
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    out["tf"] = tf_name
    return out


def pick_fit_mask_by_date(
    idx: pd.DatetimeIndex,
    exclude_last_day: bool,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=bool)
    if train_end_date is not None:
        end_dt = pd.to_datetime(train_end_date)
        start_dt = pd.to_datetime(train_start_date) if train_start_date else idx.min()
        mask = (idx >= start_dt) & (idx <= end_dt)
        if mask.sum() == 0:
            raise ValueError("empty fit mask from provided date range")
        return mask.astype(bool)
    else:
        if not exclude_last_day:
            return np.ones(len(idx), dtype=bool)
        last_date = idx.max().normalize()
        return (idx < last_date).astype(bool)
