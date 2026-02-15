import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from quant_refactor_skeleton.data.loaders import load_ohlcv_csv, normalize_input_5m
from quant_refactor_skeleton.features.feature_builder import make_features

cfg = SimpleNamespace(
    tf_5m="5min",
    min_count_5m=1,
    ma_fast=20,
    ma_slow=60,
    vol_short=20,
    vol_long=60,
    rsi_n=14,
    atr_n=14,
    adx_n=14,
)

p = Path("_tmp_l10_ohlcv_5m.csv")
df = pd.DataFrame({
    "time": pd.date_range("2026-01-01 09:30:00", periods=160, freq="5min"),
    "open": np.linspace(100, 110, 160),
    "high": np.linspace(100.2, 110.2, 160),
    "low": np.linspace(99.8, 109.8, 160),
    "close": np.linspace(100, 110, 160),
    "volume": np.arange(160) + 1,
})
df.to_csv(p, index=False)

x = load_ohlcv_csv(str(p))
y = normalize_input_5m(x, cfg)
feat = make_features(y, cfg, tf_name="5m")

print("columns(head30):", feat.columns.tolist()[:30])
print("shape:", feat.shape)
print(feat[["close", "log_ret_1", "atr_14", "momentum_10"]].tail(3))

p.unlink()
print("ok: L10 smoke")
