import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_refactor_skeleton.data.loaders import load_ohlcv_csv, normalize_input_5m
from quant_refactor_skeleton.features.feature_builder import make_features
from quant_refactor_skeleton.overlay.overlay_builder import build_overlay_superstate_minimal

p = Path("_tmp_l11_ohlcv_5m.csv")

n = 360
idx = pd.date_range("2026-01-01 09:30:00", periods=n, freq="5min")
base = np.linspace(100.0, 108.0, n)
wave = 0.5 * np.sin(np.linspace(0, 12, n))
close = base + wave
open_ = close + 0.02 * np.cos(np.linspace(0, 8, n))
high = np.maximum(open_, close) + 0.15
low = np.minimum(open_, close) - 0.15
vol = (np.arange(n) % 50) + 10

df = pd.DataFrame(
    {
        "time": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    }
)
df.to_csv(p, index=False)

cfg_data = SimpleNamespace(tf_5m="5min", min_count_5m=1)
cfg_feat = SimpleNamespace(
    ma_fast=20,
    ma_slow=60,
    vol_short=20,
    vol_long=60,
    rsi_n=14,
    atr_n=14,
    adx_n=14,
)

x = load_ohlcv_csv(str(p))
y = normalize_input_5m(x, cfg_data)
feat = make_features(y, cfg_feat, tf_name="5m")
out = build_overlay_superstate_minimal(y, feat, cfg=None)

print("columns(head40):", out.columns.tolist()[:40])
tail_df = out.reset_index().rename(columns={out.index.name or "index": "time"})
show = [c for c in ["time", "super_state", "stability_score", "avg_run_local", "switch_rate_local"] if c in tail_df.columns]
print(tail_df[show].tail(3))
print("ok: L11 smoke")

p.unlink()
