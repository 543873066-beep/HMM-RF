import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CONT_FEATURES = [
    "log_ret_1",
    "momentum_10",
    "momentum_30",
    "momentum_60",
    "range_pct",
    "price_change_15",
    "ma_fast",
    "ma_slow",
    "vol_short",
    "vol_ratio",
    "atr_pct_14",
    "volatility_15",
    "trend_strength",
    "adx_14",
    "rsi_14",
    "volume_ratio_15",
    "posterior_maxp",
    "posterior_entropy",
    "stability_score",
    "mixed_signals",
    "avg_run_local",
    "switch_rate_local",
]


# migrated from: safe_log_return
def safe_log_return(close_future: pd.Series, close_now: pd.Series) -> pd.Series:
    return np.log(close_future / close_now)


# migrated from: evaluate_regression
def evaluate_regression(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "ic": ic}


# migrated from: build_features_rf
def build_features_rf(df: pd.DataFrame):
    work = df.copy()
    mf = pd.to_numeric(work["ma_fast"], errors="coerce")
    ms = pd.to_numeric(work["ma_slow"], errors="coerce")
    work["ma_gap"] = mf / ms - 1.0
    for c in CONT_FEATURES:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    cols = [c for c in CONT_FEATURES if c not in ["ma_fast", "ma_slow"]] + ["ma_gap"]
    return work[cols].copy(), cols
