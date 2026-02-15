from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd


def aggregate_5m_to_30m(states_5m: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
    from quant_refactor_skeleton.hmm.states import mode_agg

    df = pd.DataFrame({"state_5m": states_5m["state_5m"].copy()})
    agg = df.resample(cfg.tf_30m, label="right", closed="right").agg(
        state_5m_mode=("state_5m", mode_agg),
        state_5m_last=("state_5m", "last"),
    )
    return agg


def build_overlay_30m(states_30m: pd.DataFrame, agg5_to_30: pd.DataFrame, states_1d: pd.DataFrame) -> pd.DataFrame:
    base = states_30m.copy().join(agg5_to_30, how="left")
    d = states_1d[["state_1d"]].copy()
    d["trade_date"] = pd.to_datetime(d.index).date
    d["state_1d_prev"] = d["state_1d"].shift(1)
    base2 = base.reset_index().rename(columns={base.index.name or "index": "time"})
    base2["time"] = pd.to_datetime(base2["time"])
    base2["trade_date"] = base2["time"].dt.date
    base2 = base2.merge(d[["trade_date", "state_1d_prev"]], on="trade_date", how="left").drop(columns=["trade_date"]).set_index("time")

    base2["overlay_s5"] = base2["state_5m_mode"].fillna(-1).astype(int)
    base2["overlay_s30"] = base2["state_30m"].fillna(-1).astype(int)
    base2["overlay_sd"] = base2["state_1d_prev"].fillna(-1).astype(int)

    base2["overlay_id"] = (base2["overlay_s5"] * 10000 + base2["overlay_s30"] * 100 + base2["overlay_sd"]).astype(int)

    base2["overlay_tuple"] = base2.apply(
        lambda r: f"({r['overlay_s5']},{r['overlay_s30']},{r['overlay_sd']})",
        axis=1,
    )
    return base2


def _default_cfg(cfg: Optional[object]) -> object:
    if cfg is not None:
        return cfg
    return SimpleNamespace(
        tf_30m="30min",
        exclude_last_day_from_fit=True,
        hmm_train_start_date=None,
        hmm_train_end_date=None,
        trade_start_date=None,
        num_inits=1,
        rs_super=123,
        min_covar=1e-3,
        super_infer_warmup_bars=200,
        avg_run_window=200,
        ewma_alpha_run=0.15,
        min_hist_runs=3,
        use_ewma_prior=True,
    )


def _build_minimal_state_series(values: pd.Series, n_bins: int = 6) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    try:
        q = pd.qcut(s.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
        return pd.Series(q, index=s.index).fillna(0).astype(int)
    except Exception:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)


def build_overlay_superstate_minimal(
    ohlcv_5m: pd.DataFrame,
    features_df: pd.DataFrame,
    cfg: Optional[object] = None,
) -> pd.DataFrame:
    """Minimal integration path for L11 smoke.

    Input:
      - ohlcv_5m: OHLCV dataframe indexed by time.
      - features_df: feature dataframe from make_features (same/similar index).
      - cfg: optional config with super-HMM fields.
    Output:
      - dataframe containing overlay and super-state fields.
    """
    from quant_refactor_skeleton.data.resample import resample_ohlcv
    from quant_refactor_skeleton.super_state.super_hmm import run_super_hmm_from_overlay

    cfg = _default_cfg(cfg)

    f5 = features_df.copy()
    if not isinstance(f5.index, pd.DatetimeIndex):
        raise ValueError("features_df index must be DatetimeIndex")
    f5 = f5.sort_index()

    # minimal pseudo 5m states from existing feature distribution
    src_col = "log_ret_1" if "log_ret_1" in f5.columns else "ret_1"
    if src_col not in f5.columns:
        f5[src_col] = pd.to_numeric(f5.get("close", 0.0), errors="coerce").fillna(0.0).diff().fillna(0.0)
    states_5m = pd.DataFrame(index=f5.index)
    states_5m["state_5m"] = _build_minimal_state_series(f5[src_col], n_bins=6)

    agg5_to_30 = aggregate_5m_to_30m(states_5m, cfg)

    o30 = resample_ohlcv(ohlcv_5m, rule=cfg.tf_30m, cfg=None)
    feat30 = (
        f5[[c for c in ["ret_1", "vol_short", "trend_strength", "adx_14", "rsi_14", "atr_14"] if c in f5.columns]]
        .resample(cfg.tf_30m, label="right", closed="right")
        .mean()
    )
    states_30m = o30.join(feat30, how="left")
    # simple pseudo 30m state from 5m aggregation mode to preserve overlay format
    states_30m["state_30m"] = agg5_to_30.reindex(states_30m.index)["state_5m_mode"].fillna(-1).astype(int)

    s1d_base = states_30m[["state_30m"]].copy()
    s1d = s1d_base.resample("1D", label="left", closed="left").agg({"state_30m": "last"}).dropna()
    states_1d = pd.DataFrame(index=s1d.index)
    states_1d["state_1d"] = s1d["state_30m"].astype(int)

    ov = build_overlay_30m(states_30m, agg5_to_30, states_1d)

    out, _, _ = run_super_hmm_from_overlay(ov, cfg)

    required = [
        "overlay_id",
        "overlay_tuple",
        "overlay_s5",
        "overlay_s30",
        "overlay_sd",
        "super_state",
        "posterior_maxp",
        "posterior_entropy",
        "stability_score",
        "avg_run_local",
        "switch_rate_local",
    ]
    for c in required:
        if c not in out.columns:
            # TODO(L11): backfill from upstream stages if absent in current minimal path.
            out[c] = np.nan

    return out


__all__ = [
    "aggregate_5m_to_30m",
    "build_overlay_30m",
    "build_overlay_superstate_minimal",
]
