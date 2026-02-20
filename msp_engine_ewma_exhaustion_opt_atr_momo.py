# -*- coding: utf-8 -*-
"""
msp_engine_ewma_exhaustion_opt_atr_momo.py

【修复版 v2】
1. 回测引擎：真正启用 Event-Driven 模式，支持 ATR 止损/止盈。
2. 参数优化：Take Profit 设为 8.0（放宽），Stop Loss 设为 2.0（严控）。
3. 稳定性：保留 HMM 状态对齐和自动降级容错。
"""

from __future__ import annotations
from joblib import Parallel, delayed
import os
import re
import math
import argparse
import pickle
import joblib
import json
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # -----------------
    # data
    # -----------------
    input_csv: str = r"data/sh000852_5m.csv"
    input_tf_minutes: int = 5
    out_dir: str = r"outputs"
    run_id: Optional[str] = None
    data_end_date: Optional[str] = None

    trade_start_date: Optional[str] = None
    trade_end_date: Optional[str] = None
    
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    
    hmm_train_start_date: Optional[str] = None
    hmm_train_end_date: Optional[str] = "2026-01-29 15:00:00"

    # -----------------
    # infer-only on trade (live-like decoding)
    # -----------------
    hmm_infer_warmup_5m: int = 400
    hmm_infer_warmup_30m: int = 200
    hmm_infer_warmup_1d: int = 60
    super_infer_warmup_bars: int = 200
    # avg_run_local: expected/average state lifetime in recent history (causal)
    avg_run_window: int = 200  # in bars of super_df (typically 30m bars)
    # --- Online EWMA prior for avg_run_local (production-like) ---
    use_ewma_prior: bool = True
    ewma_alpha_run: float = 0.15
    min_hist_runs: int = 3


    # -----------------
    # export artifacts for live use
    # -----------------
    export_live_pack: bool = False
    live_pack_dir: str = "live_pack"

    # runtime holders (not user parameters)
    _hmm_models: dict = None   # {"5m": (model, scaler), ...}
    _super_model: object = None
    _super_scaler: object = None
    _rf_models: dict = None    # {state: model}
    _last_good_states: list = None
    _last_gate_th: dict = None

    
    rf_train_start_date: Optional[str] = None
    rf_train_end_date: Optional[str] = "2025-05-06 10:00:00"
    
    trade_days = 44
    min_trade_rows_per_state = 1
    min_hist_bars = 1000
    
    strict_last_month: bool = True
    hist_train_frac: float = 0.8
    
    # -----------------
    # base HMM
    # -----------------
    n_states: int = 6
    tf_5m: str = "5min"
    tf_30m: str = "30min"
    tf_1d: str = "1D"

    min_count_30m: int = 6
    min_count_5m: int = 1

    # indicators
    rsi_n: int = 14
    atr_n: int = 14
    adx_n: int = 14
    ma_fast: int = 20
    ma_slow: int = 60
    vol_short: int = 20
    vol_long: int = 60

    w_change_15: int = 15
    w_vol_15: int = 15
    w_vol_ratio_15: int = 15

    exclude_last_day_from_fit: bool = True

    rs_5m: int = 52
    rs_30m: int = 53
    rs_1d: int = 54

    # super HMM
    k_candidates: Tuple[int, ...] = (6, 7, 8)
    rs_super: int = 123
    num_inits: int = 1
    min_covar: float = 1e-3

    # eval
    force_random_init: bool = True
    eval_small_state_frac: float = 0.005
    eval_rolling_window: int = 800
    eval_history_lookback: int = 30
    eval_q_good: float = 0.30
    eval_q_bad: float = 0.70
    eval_history_file: str = "eval_history.csv"

    # -----------------
    # RF stage
    # -----------------
    rf_subdir: str = "rf_h4_per_state_dynamic_selected"

    horizon_bars: int = 4
    test_ratio: float = 0.2

    min_train_rows_per_state: int = 300
    min_test_rows_per_state: int = 80

    # 保持宽松的筛选条件，保证开单
    min_ic: float = 0.005
    min_test_n: int = 80
    min_r2: float = -1.0
    max_states: Optional[int] = None

    gate_q_run: float = 0.65
    gate_q_switch: float = 0.65
    stability_static_min: float = 0.95
    clamp_avg_run_max: float = 4.0
    clamp_switch_min: float = 0.35
    target_coverage_hint: float = 0.15

    exhaustion_max: float = 1.2            # 新增：状态衰竭率上限（run_len_so_far / avg_run_local）
    exhaustion_min: float = 0.2            # 新增：状态衰竭率下限（用于过滤“太早期”）
    rf_params: Dict = None

    # backtest
    enable_backtest: bool = True
    cost_bps_per_unit: float = 3
    leverage_max: float = 1.0
    backtest_restrict_to_pred_range: bool = True
    save_trades_per_run: bool = True
    trades_runs_subdir: str = "trades_runs"
    
    # 【修复】这些参数现在真正被使用了
    stop_atr_mult: float = 2.0  # 止损：2倍 ATR
    take_atr_mult: float = 8.0  # 止盈：8倍 ATR (放宽，让利润奔跑)
    
    def __post_init__(self):
        if self.hmm_train_start_date is None:
            self.hmm_train_start_date = self.train_start_date
        if self.hmm_train_end_date is None:
            self.hmm_train_end_date = self.train_end_date
        if self.rf_train_start_date is None:
            self.rf_train_start_date = self.train_start_date
        if self.rf_train_end_date is None:
            self.rf_train_end_date = self.train_end_date
            
        if self.rf_params is None:
            self.rf_params = dict(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=50,
                min_samples_split=100,
                max_features="sqrt",
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            )

# ============================================================
# Utils
# ============================================================
def _safe_read_csv_maybe_empty(path: str) -> pd.DataFrame:
    if (not path) or (not os.path.exists(path)):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================
# Indicators
# ============================================================
def safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))

def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()

def rolling_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win, min_periods=win).std()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    tr_s = pd.Series(tr, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100.0 * (plus_dm_s / (tr_s + 1e-12))
    minus_di = 100.0 * (minus_dm_s / (tr_s + 1e-12))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx_val = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_val

# ============================================================
# Data load & resample
# ============================================================
def _load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}，当前列：{df.columns.tolist()}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    df = df.set_index("time")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df

def load_1m(csv_path: str) -> pd.DataFrame:
    return _load_ohlcv_csv(csv_path)

def load_5m(csv_path: str) -> pd.DataFrame:
    return _load_ohlcv_csv(csv_path)

def normalize_input_5m(df_5m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_5m.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    bars = df.resample(cfg.tf_5m, label="right", closed="right").agg(AGG).dropna()
    cnt = df["close"].resample(cfg.tf_5m, label="right", closed="right").count()
    bars = bars[cnt.reindex(bars.index).fillna(0).astype(int) >= int(cfg.min_count_5m)]
    return bars

def resample_ohlcv(df_base: pd.DataFrame, rule: str, cfg: Config) -> pd.DataFrame:
    AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    rule_u = str(rule).upper()
    if rule_u in ["1D", "D"]:
        out = df_base.resample("1D", label="left", closed="left").agg(AGG).dropna()
        return out
    out = df_base.resample(rule, label="right", closed="right").agg(AGG)
    cnt = df_base["close"].resample(rule, label="right", closed="right").count()
    r = str(rule).lower()
    if r in ["30min", "30t"]:
        min_cnt = int(cfg.min_count_30m)
    elif r in ["5min", "5t"]:
        min_cnt = int(cfg.min_count_5m)
    else:
        min_cnt = 1
    out = out.dropna()
    out = out[cnt.reindex(out.index).fillna(0).astype(int) >= min_cnt]
    return out

def make_features(df: pd.DataFrame, cfg: Config, tf_name: str) -> pd.DataFrame:
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
    # RF 特征用：ATR 标准化（百分比 ATR），避免价格尺度变化导致非平稳
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


# ============================================================
# HMM fit/predict + ALIGNMENT
# ============================================================

def pick_fit_mask_by_date(
    idx: pd.DatetimeIndex, 
    exclude_last_day: bool, 
    train_start_date: Optional[str] = None, 
    train_end_date: Optional[str] = None
) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=bool)
    if train_end_date is not None:
        end_dt = pd.to_datetime(train_end_date)
        start_dt = pd.to_datetime(train_start_date) if train_start_date else idx.min()
        mask = (idx >= start_dt) & (idx <= end_dt)
        if mask.sum() == 0:
            raise ValueError("指定日期范围导致拟合数据为空。请检查日期。")
        return mask.astype(bool)
    else:
        if not exclude_last_day:
            return np.ones(len(idx), dtype=bool)
        last_date = idx.max().normalize()
        return (idx < last_date).astype(bool)

def _force_random_init_hmm(model: GaussianHMM, X_fit_s: np.ndarray, seed: int, min_covar: float = 1e-3):
    if X_fit_s is None or len(X_fit_s) < model.n_components:
        return
    rng = np.random.default_rng(int(seed))
    k = int(model.n_components)
    n, d = X_fit_s.shape
    sp = rng.random(k)
    sp = sp / sp.sum()
    tm = rng.random((k, k))
    tm = tm / tm.sum(axis=1, keepdims=True)
    idx = rng.choice(n, size=k, replace=False)
    means = X_fit_s[idx].copy()
    cov = np.cov(X_fit_s, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + np.eye(d) * float(min_covar)
    covars = np.stack([cov.copy() for _ in range(k)], axis=0)
    model.startprob_ = sp
    model.transmat_ = tm
    model.means_ = means
    model.covars_ = covars
    model.init_params = ""

def align_states_by_volatility(model, X_sample=None):
    try:
        if model.covariance_type == "full":
            vols = np.array([np.mean(np.diag(c)) for c in model.covars_])
        elif model.covariance_type == "diag":
            vols = np.mean(model.covars_, axis=1)
        elif model.covariance_type == "spherical":
            vols = model.covars_.flatten()
        else: 
            return 

        sorted_idx = np.argsort(vols)
        
        if np.array_equal(sorted_idx, np.arange(len(vols))):
            return

        old_startprob = model.startprob_.copy()
        old_transmat = model.transmat_.copy()
        old_means = model.means_.copy()
        old_covars = model.covars_.copy()

        model.startprob_ = old_startprob[sorted_idx]
        model.transmat_ = old_transmat[sorted_idx][:, sorted_idx]
        model.means_ = old_means[sorted_idx]
        model.covars_ = old_covars[sorted_idx]
        
    except Exception:
        return

def fit_hmm_train_infer(
    feat_ready: pd.DataFrame,
    feature_cols: List[str],
    n_states: int,
    base_random_state: int,
    exclude_last_day_from_fit: bool,
    num_inits: int,
    force_random_init: bool = False,
    min_covar: float = 1e-3,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
    infer_start_date: Optional[str] = None,
    infer_warmup_bars: int = 0,
) -> Tuple[GaussianHMM, StandardScaler, np.ndarray, float]:
    """
    Fit HMM on a restricted (historical) slice, then infer states.

    - Fit is controlled by pick_fit_mask_by_date(idx, ..., train_start_date/train_end_date).
    - If infer_start_date is provided: states for rows >= infer_start_date are inferred in an
      infer-only manner with an optional warmup prefix (infer_warmup_bars) to preserve continuity.
      Rows < infer_start_date are inferred on the historical slice only (no leakage from future).
    """
    if feat_ready is None or len(feat_ready) == 0:
        raise ValueError("feat_ready is empty")

    X_all = feat_ready[feature_cols].values.astype(float)

    # -------- Fit mask (historical only) --------
    fit_mask = pick_fit_mask_by_date(
        feat_ready.index,
        exclude_last_day_from_fit,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )
    if fit_mask.sum() < max(200, int(n_states) * 50):
        fit_mask[:] = True

    X_fit = X_all[fit_mask]
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)

    best_model = None
    best_ll = -np.inf

    for i in range(int(num_inits)):
        current_seed = int(base_random_state) + i
        try:
            model = GaussianHMM(
                n_components=int(n_states),
                covariance_type="full",
                n_iter=500,
                random_state=current_seed,
                verbose=False,
                min_covar=float(min_covar),
            )
            if force_random_init:
                _force_random_init_hmm(model, X_fit_s, current_seed, min_covar=float(min_covar))
            model.fit(X_fit_s)
        except ValueError:
            try:
                model = GaussianHMM(
                    n_components=int(n_states),
                    covariance_type="diag",
                    n_iter=500,
                    random_state=current_seed,
                    verbose=False,
                    min_covar=1e-2,
                )
                model.fit(X_fit_s)
            except Exception:
                continue

        ll = model.score(X_fit_s)
        if ll > best_ll:
            best_ll = ll
            best_model = model

    if best_model is None:
        best_model = GaussianHMM(
            n_components=int(n_states),
            covariance_type="diag",
            n_iter=500,
            random_state=int(base_random_state),
            min_covar=0.1,
        )
        best_model.fit(X_fit_s)
        best_ll = best_model.score(X_fit_s)

    # Align states for interpretability (low-vol -> low index, etc.)
    align_states_by_volatility(best_model)

    # -------- Inference --------
    X_all_s = scaler.transform(X_all)
    idx = feat_ready.index

    if not infer_start_date:
        states = best_model.predict(X_all_s)
        return best_model, scaler, states, float(best_ll)

    infer_start_dt = pd.to_datetime(infer_start_date)
    hist_mask = (idx < infer_start_dt)
    trade_mask = ~hist_mask

    states = np.full(len(idx), -1, dtype=int)

    # Infer historical part WITHOUT seeing trade part (no leakage)
    if hist_mask.any():
        states[hist_mask] = best_model.predict(X_all_s[hist_mask])

    # Infer trade part with warmup to avoid "new sequence" jump
    if trade_mask.any():
        first_trade_pos = int(np.where(trade_mask)[0][0])
        warm = int(max(0, infer_warmup_bars))
        warm_start_pos = max(0, first_trade_pos - warm)

        X_seg = X_all_s[warm_start_pos:]
        seg_states = best_model.predict(X_seg)

        offset = first_trade_pos - warm_start_pos
        states[trade_mask] = seg_states[offset:]

    return best_model, scaler, states, float(best_ll)

def transition_matrix_from_states(states: np.ndarray, n_states: int) -> pd.DataFrame:
    mat = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        mat[int(a), int(b)] += 1.0
    row_sum = mat.sum(axis=1, keepdims=True)
    prob = mat / np.where(row_sum == 0, 1.0, row_sum)
    df = pd.DataFrame(prob, columns=[f"to_{i}" for i in range(n_states)])
    df.insert(0, "from_state", [f"{i}" for i in range(n_states)])
    return df

def mode_agg(x: pd.Series):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    vc = x.value_counts()
    return vc.index[0]

def build_state_portrait(df_states: pd.DataFrame, state_col: str, tf_name: str) -> pd.DataFrame:
    g = df_states.groupby(state_col)
    prof = g.agg(
        count=("ret_1", "size"),
        ret_mean=("ret_1", "mean"),
        ret_std=("ret_1", "std"),
        vol_short_mean=("vol_short", "mean"),
        trend_strength_mean=("trend_strength", "mean"),
    ).reset_index()
    prof.insert(0, "tf", tf_name)
    return prof

def aggregate_5m_to_30m(states_5m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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
    
    base2["overlay_id"] = (
        base2["overlay_s5"] * 10000 + 
        base2["overlay_s30"] * 100 + 
        base2["overlay_sd"]
    ).astype(int)
    
    base2["overlay_tuple"] = base2.apply(
        lambda r: f"({r['overlay_s5']},{r['overlay_s30']},{r['overlay_sd']})",
        axis=1
    )
    return base2


def compute_avg_run_local_expected_life(s_arr: np.ndarray, window: int) -> np.ndarray:
    """
    Causal "expected" lifetime for the *current* state based on recent completed runs.

    For each time t with state s_t:
      - maintain a deque of completed run lengths for each state, keyed by run end index
      - keep only runs whose end index is within the last `window` bars
      - avg_run_local[t] = mean(lengths of those recent completed runs for state s_t)
        If none exist yet, fall back to the current run length so far (>=1).

    This avoids future leak from "segment total length" and makes train/live consistent.
    """
    n = len(s_arr)
    out = np.full(n, np.nan, dtype=float)
    from collections import defaultdict, deque
    runs = defaultdict(deque)  # state -> deque[(end_idx, run_len)]

    prev_state = None
    cur_len = 0

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
        else:
            if s == prev_state:
                cur_len += 1
            else:
                # finalize previous run ended at t-1
                runs[prev_state].append((t - 1, int(cur_len)))
                prev_state = s
                cur_len = 1

        dq = runs[s]
        cutoff = t - int(window)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

        if dq:
            out[t] = float(np.mean([rl for _, rl in dq]))
        else:
            out[t] = float(cur_len)

    return out

def compute_avg_run_local_expected_life_online_ewma(
    s_arr: np.ndarray,
    window: int,
    ewma_alpha: float = 0.15,
    min_hist_runs: int = 3,
    use_ewma_prior: bool = True,
):
    """
    Online + causal expected-life with optional EWMA prior fusion.

    For each t with current state s_t:
      - completed runs for each state are stored as (end_idx, run_len) within last `window`
      - mean_runs = mean(completed run lengths) for s_t (within window)
      - if n_runs >= min_hist_runs: avg_run_local = mean_runs
      - else:
          - if use_ewma_prior and ewma_alpha>0 and ewma_prior exists: blend toward ewma_prior
          - otherwise fallback to current run length so far

    EWMA prior is updated ONLY when a run completes (no future leak).
    Returns:
      avg_run_local: np.ndarray[float]
      ewma_state: dict[str, dict] (json-serializable) containing per-state ewma mean/n
    """
    from collections import defaultdict, deque

    s_arr = np.asarray(s_arr).astype(int)
    n = len(s_arr)
    out = np.full(n, np.nan, dtype=float)

    window = int(max(20, window))
    min_hist_runs = int(max(1, min_hist_runs))
    ewma_alpha = float(ewma_alpha)

    runs = defaultdict(deque)     # state -> deque[(end_idx, run_len)]
    sums = defaultdict(float)     # state -> sum(run_len in deque)
    ewma = {}                     # state -> (mean, n)

    prev_state = None
    cur_len = 0

    def update_ewma(st: int, run_len: float):
        if not use_ewma_prior:
            return
        if ewma_alpha <= 0:
            return
        if st not in ewma:
            ewma[st] = (float(run_len), 1)
        else:
            m0, n0 = ewma[st]
            m1 = ewma_alpha * float(run_len) + (1.0 - ewma_alpha) * float(m0)
            ewma[st] = (float(m1), int(n0) + 1)

    def get_prior(st: int):
        if (not use_ewma_prior) or (ewma_alpha <= 0):
            return None
        if st in ewma and ewma[st][1] > 0 and ewma[st][0] > 0:
            return float(ewma[st][0])
        # optional global fallback (mean of existing priors)
        if ewma:
            ms = [m for (m, nn) in ewma.values() if nn > 0 and m > 0]
            if ms:
                return float(sum(ms) / len(ms))
        return None

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
        else:
            if s == prev_state:
                cur_len += 1
            else:
                # finalize previous run ended at t-1
                runs[prev_state].append((t - 1, int(cur_len)))
                sums[prev_state] += float(cur_len)
                update_ewma(prev_state, float(cur_len))
                prev_state = s
                cur_len = 1

        # prune runs for current state by end_idx within window
        cutoff = t - window
        dq = runs[s]
        while dq and dq[0][0] < cutoff:
            end_i, L = dq.popleft()
            sums[s] -= float(L)
        if dq and sums[s] < 0:
            sums[s] = float(sum(L for _, L in dq))

        n_runs = len(dq)
        mean_runs = (sums[s] / float(n_runs)) if n_runs > 0 else None
        prior = get_prior(s)

        if n_runs >= min_hist_runs and (mean_runs is not None):
            avg = float(mean_runs)
        else:
            if (prior is None) or (prior <= 0):
                avg = float(cur_len)
            else:
                if mean_runs is None or n_runs <= 0:
                    avg = float(prior)
                else:
                    w = float(max(1, min_hist_runs - n_runs))
                    avg = (float(mean_runs) * float(n_runs) + float(prior) * w) / (float(n_runs) + w)

        out[t] = max(1.0, float(avg))

    ewma_state = {str(k): {"mean": float(v[0]), "n": int(v[1])} for k, v in ewma.items()}
    return out, ewma_state



def compute_run_len_so_far(s_arr: np.ndarray) -> np.ndarray:
    """Causal current run length for each time step (age of current state segment)."""
    s_arr = np.asarray(s_arr).astype(int)
    out = np.empty(len(s_arr), dtype=float)
    prev = None
    cur = 0
    for i, s in enumerate(s_arr):
        if prev is None or s != prev:
            cur = 1
        else:
            cur += 1
        out[i] = float(cur)
        prev = s
    return out

def compute_path_stats_seed_expected_life(s_arr: np.ndarray, window: int = 200, switch_window: int = 50,
                                         ewma_alpha: float = 0.15, min_hist_runs: int = 3,
                                         use_ewma_prior: bool = True):
    """Build a *causal* seed for path-dependent metrics used by gate (avg_run_local / switch_rate_local).

    - avg_run_local mode: expected-life (mean of *completed* run lengths in recent history)
    - optional EWMA prior: updated ONLY when a run completes (no future leak)
    - switch_rate_local: rolling mean of switches over the last `switch_window` bars

    Returns a JSON-serializable dict that live loader can warm-start from.
    """
    from collections import defaultdict, deque

    s_arr = np.asarray(s_arr).astype(int)
    n = len(s_arr)
    window = int(max(20, window))
    switch_window = int(max(5, switch_window))
    min_hist_runs = int(max(1, min_hist_runs))
    ewma_alpha = float(ewma_alpha)

    runs = defaultdict(deque)  # state -> deque[(end_idx, run_len)] pruned by end_idx within `window`
    sums = defaultdict(float)

    ewma = {}  # state -> (mean, n)

    prev_state = None
    cur_len = 0

    sw_buf = deque(maxlen=switch_window)

    def _update_ewma(st: int, run_len: float):
        if (not use_ewma_prior) or (ewma_alpha <= 0):
            return
        if st not in ewma:
            ewma[st] = (float(run_len), 1)
        else:
            m0, n0 = ewma[st]
            m1 = ewma_alpha * float(run_len) + (1.0 - ewma_alpha) * float(m0)
            ewma[st] = (float(m1), int(n0) + 1)

    for t in range(n):
        s = int(s_arr[t])

        if prev_state is None:
            prev_state = s
            cur_len = 1
            sw_buf.append(0.0)
        else:
            if s == prev_state:
                cur_len += 1
                sw_buf.append(0.0)
            else:
                # finalize previous run ended at t-1
                dq = runs[prev_state]
                dq.append((t - 1, int(cur_len)))
                sums[prev_state] += float(cur_len)
                _update_ewma(prev_state, float(cur_len))

                prev_state = s
                cur_len = 1
                sw_buf.append(1.0)

        # prune per-state runs for current state
        cutoff = t - window
        dq = runs[s]
        while dq and dq[0][0] < cutoff:
            end_i, L = dq.popleft()
            sums[s] -= float(L)
        if dq and sums[s] < 0:
            sums[s] = float(sum(L for _, L in dq))

    seed = {
        "path_metric_mode": "expected_life_online_ewma" if (use_ewma_prior and ewma_alpha > 0) else "expected_life",
        "avg_run_window": int(window),
        "switch_window": int(switch_window),
        "switch_min_periods": 10,
        "min_hist_runs": int(min_hist_runs),
        "use_ewma_prior": bool(use_ewma_prior),
        "ewma_alpha_run": float(ewma_alpha),
        "per_state_runs": {str(k): list(v) for k, v in runs.items()},
        "per_state_sums": {str(k): float(sums.get(k, 0.0)) for k in runs.keys()},
        "ewma": {str(k): {"mean": float(v[0]), "n": int(v[1])} for k, v in ewma.items()},
        "last_state": int(prev_state) if prev_state is not None else None,
        "cur_run_len": int(cur_len) if prev_state is not None else 0,
        "switch_buf": list(sw_buf),
        "seed_n": int(n),
    }
    return seed



def run_super_hmm_from_overlay(df_overlay: pd.DataFrame, cfg: Config):
    super_feature_cols = [
        "ret_1", "vol_short", "trend_strength", "adx_14", "rsi_14", "atr_14",
        "overlay_s5", "overlay_s30", "overlay_sd"
    ]
    df2 = df_overlay.dropna(subset=super_feature_cols).copy()
    if len(df2) == 0:
        raise ValueError("overlay after dropna is empty")

    X_all = df2[super_feature_cols].values.astype(float)

    # --- Determine infer-only split point (trade start) ---
    trade_start_dt = None
    if getattr(cfg, "trade_start_date", None):
        try:
            trade_start_dt = pd.to_datetime(cfg.trade_start_date)
        except Exception:
            trade_start_dt = None

    # Fit mask: by default uses cfg.hmm_train_*; if trade_start_dt is provided,
    # enforce "HMM classification cutoff before trade_df" (fit uses only < trade_start_dt).
    fit_mask = pick_fit_mask_by_date(
        df2.index,
        cfg.exclude_last_day_from_fit,
        train_start_date=cfg.hmm_train_start_date,
        train_end_date=cfg.hmm_train_end_date,
    )
    if trade_start_dt is not None:
        fit_mask = fit_mask & (df2.index < trade_start_dt)

    if fit_mask.sum() < max(300, 7 * 50):
        fit_mask[:] = True

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_all[fit_mask])

    best_k = 7
    best_model = None
    best_ll = -np.inf

    for i in range(int(cfg.num_inits)):
        current_seed = int(cfg.rs_super) + i
        try:
            model = GaussianHMM(
                n_components=best_k, covariance_type="full", n_iter=500,
                random_state=current_seed, verbose=False, min_covar=float(cfg.min_covar)
            )
            model.fit(X_fit_s)
        except ValueError:
            try:
                model = GaussianHMM(
                    n_components=best_k, covariance_type="diag", n_iter=500,
                    random_state=current_seed, verbose=False, min_covar=1e-2
                )
                model.fit(X_fit_s)
            except Exception:
                continue

        ll = model.score(X_fit_s)
        if ll > best_ll:
            best_ll = ll
            best_model = model

    if best_model is None:
        best_model = GaussianHMM(
            n_components=best_k, covariance_type="diag", n_iter=500,
            random_state=int(cfg.rs_super), min_covar=0.1
        )
        best_model.fit(X_fit_s)
        best_ll = best_model.score(X_fit_s)

    align_states_by_volatility(best_model)

    # stash fitted super HMM model for optional export
    try:
        cfg._super_model = best_model
        cfg._super_scaler = scaler
        cfg._super_feature_cols = list(super_feature_cols)
    except Exception:
        pass

    X_all_s = scaler.transform(X_all)
    idx = df2.index

    # --- Infer super_state with "infer-only on trade" ---
    if trade_start_dt is None:
        df2["super_state"] = best_model.predict(X_all_s)

        # NOTE: predict_proba is a smoother over the whole sequence.
        # In backtests it's fine; for strict real-time you would implement online filtering.
        try:
            gamma = best_model.predict_proba(X_all_s)
            df2["posterior_maxp"] = gamma.max(axis=1)
            ent = -np.sum(gamma * np.log(np.clip(gamma, 1e-12, 1.0)), axis=1)
            df2["posterior_entropy"] = ent
            df2["stability_score"] = 1.0 - (ent / np.log(best_k))
            df2["mixed_signals"] = 1.0 - df2["stability_score"]
        except Exception:
            pass
    else:
        hist_mask = (idx < trade_start_dt)
        trade_mask = ~hist_mask

        states = np.full(len(idx), -1, dtype=int)

        if hist_mask.any():
            states[hist_mask] = best_model.predict(X_all_s[hist_mask])

        if trade_mask.any():
            warm = int(getattr(cfg, "super_infer_warmup_bars", 200))
            first_trade_pos = int(np.where(trade_mask)[0][0])
            warm_start_pos = max(0, first_trade_pos - max(0, warm))
            X_seg = X_all_s[warm_start_pos:]
            seg_states = best_model.predict(X_seg)
            offset = first_trade_pos - warm_start_pos
            states[trade_mask] = seg_states[offset:]

        df2["super_state"] = states

        # Compute posterior features segment-wise (reduces "new-sequence" effect at trade_start).
        try:
            post_maxp = np.full(len(idx), np.nan, dtype=float)
            post_ent = np.full(len(idx), np.nan, dtype=float)

            if hist_mask.any():
                g_hist = best_model.predict_proba(X_all_s[hist_mask])
                post_maxp[hist_mask] = g_hist.max(axis=1)
                ent = -np.sum(g_hist * np.log(np.clip(g_hist, 1e-12, 1.0)), axis=1)
                post_ent[hist_mask] = ent

            if trade_mask.any():
                warm = int(getattr(cfg, "super_infer_warmup_bars", 200))
                first_trade_pos = int(np.where(trade_mask)[0][0])
                warm_start_pos = max(0, first_trade_pos - max(0, warm))
                X_seg = X_all_s[warm_start_pos:]
                g_seg = best_model.predict_proba(X_seg)
                offset = first_trade_pos - warm_start_pos
                g_trade = g_seg[offset:]
                post_maxp[trade_mask] = g_trade.max(axis=1)
                ent = -np.sum(g_trade * np.log(np.clip(g_trade, 1e-12, 1.0)), axis=1)
                post_ent[trade_mask] = ent

            df2["posterior_maxp"] = post_maxp
            df2["posterior_entropy"] = post_ent
            df2["stability_score"] = 1.0 - (post_ent / np.log(best_k))
            df2["mixed_signals"] = 1.0 - df2["stability_score"]
        except Exception:
            pass

    # Local expected lifetime and switch rate (for gate)  [causal, no future leak]
    s_arr = df2["super_state"].values.astype(int)
    win = int(getattr(cfg, "avg_run_window", 200))
    win = max(20, win)  # keep sane lower bound
        # Online expected-life + optional EWMA prior (matches live behavior)
    ewma_alpha = float(getattr(cfg, "ewma_alpha_run", 0.0))
    min_hist_runs = int(getattr(cfg, "min_hist_runs", 3))
    use_prior = bool(getattr(cfg, "use_ewma_prior", True))
    avg_arr, ewma_state = compute_avg_run_local_expected_life_online_ewma(
        s_arr, window=win, ewma_alpha=ewma_alpha, min_hist_runs=min_hist_runs, use_ewma_prior=use_prior
    )
    df2["avg_run_local"] = avg_arr.astype(float)
    # --- NEW: current run length (age) + exhaustion ratio ---
    df2["run_len_so_far"] = compute_run_len_so_far(df2["super_state"].values)
    df2["exhaustion_ratio"] = df2["run_len_so_far"] / np.maximum(pd.to_numeric(df2["avg_run_local"], errors="coerce").astype(float).values, 1e-6)
    # stash ewma_state for export
    try:
        cfg._path_stats_ewma = ewma_state
    except Exception:
        pass

    # Switch rate: causal rolling mean of state changes (uses only past + current)
    df2["switch_rate_local"] = (
        (df2["super_state"] != df2["super_state"].shift(1)).astype(float).rolling(50, min_periods=10).mean()
    )

    return df2, best_k, float(best_ll)

def write_run_summary_json(out_dir: str, cfg: "Config", extra: Optional[Dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "run_time": str(pd.Timestamp.now()),
        "run_id": getattr(cfg, "run_id", None),
        "data_end_date": getattr(cfg, "data_end_date", None),
    }
    stats = getattr(cfg, "_last_rf_stats", None)
    if isinstance(stats, dict):
        payload["rf_stats"] = stats
    if extra:
        payload.update(extra)
    path = os.path.join(out_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_live_pack(out_dir: str, cfg: "Config", trade_df_snapshot: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Export artifacts needed for live trading.

    This function overwrites files in pack dir so only the last training artifacts are kept.
    """
    try:
        if not bool(getattr(cfg, "export_live_pack", False)):
            return None
        pack_dir = os.path.join(out_dir, str(getattr(cfg, "live_pack_dir", "live_pack")))
        os.makedirs(pack_dir, exist_ok=True)

        # --- meta ---
        meta = {
            "export_time": str(pd.Timestamp.now()),
            "run_id": getattr(cfg, "run_id", None),
            "data_end_date": getattr(cfg, "data_end_date", None),
            "trade_start_date": getattr(cfg, "trade_start_date", None),
            "trade_end_date": getattr(cfg, "trade_end_date", None),
            "hmm_train_start_date": getattr(cfg, "hmm_train_start_date", None),
            "hmm_train_end_date": getattr(cfg, "hmm_train_end_date", None),
            "rf_train_start_date": getattr(cfg, "rf_train_start_date", None),
            "rf_train_end_date": getattr(cfg, "rf_train_end_date", None),
            "good_states": getattr(cfg, "_last_good_states", None),
            "gate_th": getattr(cfg, "_last_gate_th", None),
            "path_metric_mode": "expected_life",
            "avg_run_window": int(getattr(cfg, "avg_run_window", 200)),
            "switch_window": 50,
            "switch_min_periods": 10,
            "use_ewma_prior": bool(getattr(cfg, "use_ewma_prior", True)),
            "ewma_alpha_run": float(getattr(cfg, "ewma_alpha_run", 0.0)),
            "min_hist_runs": int(getattr(cfg, "min_hist_runs", 3)),

        }
        with open(os.path.join(pack_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # --- base HMM models ---
        hmm_dir = os.path.join(pack_dir, "hmm")
        os.makedirs(hmm_dir, exist_ok=True)
        hmm_models = getattr(cfg, "_hmm_models", None)
        if isinstance(hmm_models, dict):
            for key, val in hmm_models.items():
                try:
                    model, scaler, feat_cols = val
                    joblib.dump(model, os.path.join(hmm_dir, f"{key}_model.joblib"))
                    joblib.dump(scaler, os.path.join(hmm_dir, f"{key}_scaler.joblib"))
                    with open(os.path.join(hmm_dir, f"{key}_feature_cols.json"), "w", encoding="utf-8") as f:
                        json.dump(list(feat_cols), f, ensure_ascii=False, indent=2)
                except Exception:
                    continue

        # --- super HMM model ---
        super_dir = os.path.join(pack_dir, "super_hmm")
        os.makedirs(super_dir, exist_ok=True)
        if getattr(cfg, "_super_model", None) is not None:
            try:
                joblib.dump(getattr(cfg, "_super_model"), os.path.join(super_dir, "super_model.joblib"))
                if getattr(cfg, "_super_scaler", None) is not None:
                    joblib.dump(getattr(cfg, "_super_scaler"), os.path.join(super_dir, "super_scaler.joblib"))
                if getattr(cfg, "_super_feature_cols", None) is not None:
                    with open(os.path.join(super_dir, "super_feature_cols.json"), "w", encoding="utf-8") as f:
                        json.dump(list(getattr(cfg, "_super_feature_cols")), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # --- RF models ---
        rf_dir = os.path.join(pack_dir, "rf")
        os.makedirs(rf_dir, exist_ok=True)
        rf_models = getattr(cfg, "_rf_models", None)
        if isinstance(rf_models, dict):
            for s, m in rf_models.items():
                try:
                    joblib.dump(m, os.path.join(rf_dir, f"rf_state_{int(s)}.joblib"))
                except Exception:
                    continue

        # --- GOODSTATES + dynamic thresholds ---
        if getattr(cfg, "_last_good_states", None) is not None:
            with open(os.path.join(pack_dir, "good_states.json"), "w", encoding="utf-8") as f:
                json.dump(list(getattr(cfg, "_last_good_states")), f, ensure_ascii=False, indent=2)
        if getattr(cfg, "_last_gate_th", None) is not None:
            with open(os.path.join(pack_dir, "gate_thresholds.json"), "w", encoding="utf-8") as f:
                json.dump(getattr(cfg, "_last_gate_th"), f, ensure_ascii=False, indent=2)

        # --- trade_df snapshot (for live comparison) ---
        if isinstance(trade_df_snapshot, pd.DataFrame) and (not trade_df_snapshot.empty):
            snap_path = os.path.join(pack_dir, "trade_df_snapshot.csv")
            try:
                trade_df_snapshot.to_csv(snap_path, index=False, encoding="utf-8-sig")
            except Exception:
                # fallback minimal
                cols = [c for c in ["time", "close", "super_state", "gate_on", "pred_y_ret_4", "y_ret_4"] if c in trade_df_snapshot.columns]
                trade_df_snapshot[cols].to_csv(snap_path, index=False, encoding="utf-8-sig")

        # --- path metrics seed (for live loader: stable avg_run_local / switch_rate_local without long warmup) ---
        try:
            if isinstance(trade_df_snapshot, pd.DataFrame) and (not trade_df_snapshot.empty) and ("super_state" in trade_df_snapshot.columns):
                s_seed = pd.to_numeric(trade_df_snapshot["super_state"], errors="coerce").dropna().astype(int).values
                if len(s_seed) >= 20:
                    win = int(getattr(cfg, "avg_run_window", 200))
                    seed = compute_path_stats_seed_expected_life(
                        s_seed,
                        window=win,
                        switch_window=50,
                        ewma_alpha=float(getattr(cfg, "ewma_alpha_run", 0.0)),
                        min_hist_runs=int(getattr(cfg, "min_hist_runs", 3)),
                        use_ewma_prior=bool(getattr(cfg, "use_ewma_prior", True)),
                    )
                    if "time" in trade_df_snapshot.columns:
                        try:
                            seed["seed_until_time"] = str(pd.to_datetime(trade_df_snapshot["time"]).dropna().iloc[-1])
                        except Exception:
                            seed["seed_until_time"] = None
                    with open(os.path.join(pack_dir, "path_stats_seed.json"), "w", encoding="utf-8") as f:
                        json.dump(seed, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


        return pack_dir
    except Exception:
        return None

def run_msp_pipeline(cfg: Config) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ----------------------------
    # 1) Load base timeframe data
    # ----------------------------
    if int(cfg.input_tf_minutes) == 1:
        df_base = load_1m(cfg.input_csv)
    else:
        df_5m_raw = load_5m(cfg.input_csv)
        cuthead = pd.Timestamp("2022-01-01 09:35:00")
        df_base = normalize_input_5m(df_5m_raw[df_5m_raw.index >= cuthead], cfg)

    # Visible data cutoff (runner / live alignment)
    if getattr(cfg, "data_end_date", None):
        end_dt = pd.to_datetime(cfg.data_end_date)
        df_base = df_base[df_base.index <= end_dt]

    # Trade window split point (for "infer-only on trade")
    trade_start_dt = None
    if getattr(cfg, "trade_start_date", None):
        try:
            trade_start_dt = pd.to_datetime(cfg.trade_start_date)
        except Exception:
            trade_start_dt = None

    # ----------------------------
    # 2) Resample to multi TF
    # ----------------------------
    df_5m = resample_ohlcv(df_base, cfg.tf_5m, cfg)
    df_30m = resample_ohlcv(df_base, cfg.tf_30m, cfg)
    df_1d = resample_ohlcv(df_base, cfg.tf_1d, cfg)

    # ----------------------------
    # 3) Feature engineering
    # ----------------------------
    feat_5m = make_features(df_5m, cfg, "5m").dropna()
    feat_30m = make_features(df_30m, cfg, "30m").dropna()
    feat_1d = make_features(df_1d, cfg, "1d").dropna()

    feature_cols = ["vol_ratio", "rsi_14", "atr_14", "adx_14", "ret_1", "trend_strength"]

    # Enforce "HMM cutoff before trade_df": fit end = min(cfg.hmm_train_end_date, trade_start_dt - eps)
    eff_hmm_end = getattr(cfg, "hmm_train_end_date", None)
    if trade_start_dt is not None:
        eff_hmm_end = str(trade_start_dt - pd.Timedelta(seconds=1))

    # ----------------------------
    # 4) Base HMM: fit(hist) + infer(trade)
    # ----------------------------
    m5, sc5, s5, _ = fit_hmm_train_infer(
        feat_5m,
        feature_cols,
        cfg.n_states,
        cfg.rs_5m,
        cfg.exclude_last_day_from_fit,
        cfg.num_inits,
        min_covar=cfg.min_covar,
        train_start_date=cfg.hmm_train_start_date,
        train_end_date=eff_hmm_end,
        infer_start_date=str(trade_start_dt) if trade_start_dt is not None else None,
        infer_warmup_bars=int(getattr(cfg, "hmm_infer_warmup_5m", 400)),
    )
    feat_5m["state_5m"] = s5

    m30, sc30, s30, _ = fit_hmm_train_infer(
        feat_30m,
        feature_cols,
        cfg.n_states,
        cfg.rs_30m,
        cfg.exclude_last_day_from_fit,
        cfg.num_inits,
        min_covar=cfg.min_covar,
        train_start_date=cfg.hmm_train_start_date,
        train_end_date=eff_hmm_end,
        infer_start_date=str(trade_start_dt) if trade_start_dt is not None else None,
        infer_warmup_bars=int(getattr(cfg, "hmm_infer_warmup_30m", 200)),
    )
    feat_30m["state_30m"] = s30

    m1d, sc1d, s1d, _ = fit_hmm_train_infer(
        feat_1d,
        feature_cols,
        cfg.n_states,
        cfg.rs_1d,
        cfg.exclude_last_day_from_fit,
        cfg.num_inits,
        min_covar=cfg.min_covar,
        train_start_date=cfg.hmm_train_start_date,
        train_end_date=eff_hmm_end,
        infer_start_date=str(trade_start_dt) if trade_start_dt is not None else None,
        infer_warmup_bars=int(getattr(cfg, "hmm_infer_warmup_1d", 60)),
    )
    feat_1d["state_1d"] = s1d

    # stash fitted base HMM models for optional export
    try:
        if getattr(cfg, "_hmm_models", None) is None:
            cfg._hmm_models = {}
        cfg._hmm_models["5m"] = (m5, sc5, feature_cols)
        cfg._hmm_models["30m"] = (m30, sc30, feature_cols)
        cfg._hmm_models["1d"] = (m1d, sc1d, feature_cols)
    except Exception:
        pass

    # ----------------------------
    # 5) Overlay + Super HMM (fit(hist) + infer(trade))
    # ----------------------------
    agg5_30 = aggregate_5m_to_30m(feat_5m, cfg)
    overlay = build_overlay_30m(feat_30m, agg5_30, feat_1d)

    super_df, _, _ = run_super_hmm_from_overlay(overlay, cfg)

    out_super = os.path.join(cfg.out_dir, "super_states_30m.csv")
    super_df.reset_index().rename(columns={"index": "time"}).to_csv(out_super, index=False, encoding="utf-8-sig")

    return super_df


# ============================================================
# RF Stage & Backtest
# ============================================================

CONT_FEATURES = [
    "log_ret_1", "momentum_10", "momentum_30", "momentum_60", "range_pct", "price_change_15", "ma_fast", "ma_slow",
    "vol_short", "vol_ratio", "atr_pct_14", "volatility_15", "trend_strength", "adx_14",
    "rsi_14", "volume_ratio_15", "posterior_maxp", "posterior_entropy", "stability_score",
    "mixed_signals", "avg_run_local", "switch_rate_local"
]

def safe_log_return(close_future: pd.Series, close_now: pd.Series) -> pd.Series:
    return np.log(close_future / close_now)

def evaluate_regression(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "ic": ic}

def build_features_rf(df: pd.DataFrame):
    work = df.copy()
    mf = pd.to_numeric(work["ma_fast"], errors="coerce")
    ms = pd.to_numeric(work["ma_slow"], errors="coerce")
    work["ma_gap"] = mf / ms - 1.0
    for c in CONT_FEATURES:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    cols = [c for c in CONT_FEATURES if c not in ["ma_fast", "ma_slow"]] + ["ma_gap"]
    return work[cols].copy(), cols

def fit_gate_thresholds_robust(train_df: pd.DataFrame, allowed_states: list, cfg: Config):
    df = train_df[train_df["super_state"].isin(allowed_states)].copy()
    runl = pd.to_numeric(df["avg_run_local"], errors="coerce")
    sw = pd.to_numeric(df["switch_rate_local"], errors="coerce")
    run_th = float(runl.dropna().quantile(cfg.gate_q_run))
    sw_th = float(sw.dropna().quantile(1.0 - cfg.gate_q_switch))
    run_th = min(run_th, cfg.clamp_avg_run_max)
    sw_th = max(sw_th, cfg.clamp_switch_min)
    return {"stability_min": cfg.stability_static_min, "avg_run_min": run_th, "switch_rate_max": sw_th, "exhaustion_min": float(getattr(cfg, "exhaustion_min", 0.0)), "exhaustion_max": float(getattr(cfg, "exhaustion_max", 1.2))}


def gate_mask(df: pd.DataFrame, th: dict, allowed_states: list) -> pd.Series:
    """Gate condition used both in backtest and live (must be reproducible from live_pack thresholds)."""
    m = (
        df["super_state"].isin(allowed_states)
        & (pd.to_numeric(df["stability_score"], errors="coerce") >= float(th.get("stability_min", 0.0)))
        & (pd.to_numeric(df["avg_run_local"], errors="coerce") >= float(th.get("avg_run_min", 0.0)))
        & (pd.to_numeric(df["switch_rate_local"], errors="coerce") <= float(th.get("switch_rate_max", 1.0)))
    )
    # Optional exhaustion constraint (only if column & threshold exist)
    if ("exhaustion_ratio" in df.columns) and ("exhaustion_max" in th):
        ex = pd.to_numeric(df["exhaustion_ratio"], errors="coerce")
        ex_min = float(th.get("exhaustion_min", -1e9))
        ex_max = float(th.get("exhaustion_max", 1e9))
        m = m & (ex >= ex_min) & (ex <= ex_max)
    return m.fillna(False)

def backtest_using_full_timeline_df(
    full: pd.DataFrame,
    pred_all: pd.DataFrame,
    out_dir: str,
    horizon_bars: int,
    cost_bps: float,
    leverage_max: float = 1.0,
    restrict_to_pred_range: bool = True,
    # 【核心修复】真正传入这些参数
    stop_atr_mult: float = 2.0,
    take_atr_mult: float = 8.0
):
    """
    【修复版 v2】真正的事件驱动引擎，启用ATR风控
    """
    os.makedirs(out_dir, exist_ok=True)

    full = full.copy()
    full["time"] = pd.to_datetime(full["time"], errors="coerce")
    full = full.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    
    # 补全OHLC (High/Low 用于ATR)
    if "high" not in full.columns: full["high"] = full["close"]
    if "low" not in full.columns: full["low"] = full["close"]
    if "atr_14" not in full.columns: 
        full["atr_14"] = (full["high"] - full["low"]).rolling(14).mean().fillna(0.0)

    pred = pred_all.copy()
    pred["time"] = pd.to_datetime(pred["time"], errors="coerce")
    pred = pred.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    m = full.merge(pred[["time", "gate_on", "pred_y_ret_4"]], on="time", how="left")
    m["gate_on"] = m["gate_on"].fillna(False)

    if restrict_to_pred_range and len(pred) > 0:
        t0, t1 = pred["time"].min(), pred["time"].max()
        m = m[(m["time"] >= t0) & (m["time"] <= t1)].reset_index(drop=True)

    # 生成开仓信号
    m["entry_signal"] = 0.0
    sig_mask = m["gate_on"] & m["pred_y_ret_4"].notna()
    m.loc[sig_mask, "entry_signal"] = np.sign(m.loc[sig_mask, "pred_y_ret_4"])

    # --- 核心循环：事件驱动模式 ---
    n = len(m)
    pos = np.zeros(n, dtype=float)
    
    close = m["close"].values
    high = m["high"].values
    low = m["low"].values
    atr = m["atr_14"].values
    entries = m["entry_signal"].values
    
    current_pos = 0.0
    entry_price = 0.0
    entry_bar = 0
    trades = [] # 记录交易明细
    
    for i in range(n - 1):
        # 1. 盘中风控 (仅当有持仓时)
        if current_pos != 0:
            elapsed = i - entry_bar
            
            # (A) 时间离场 (Time Exit)
            if elapsed >= horizon_bars:
                # 平仓
                exit_price = close[i]
                logret = np.log(exit_price / entry_price) * current_pos
                trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "time_exit"])
                current_pos = 0.0
                
            # (B) 价格离场 (Price Stop/Take)
            else:
                is_closed = False
                if current_pos > 0: # 多头
                    stop_p = entry_price - stop_atr_mult * atr[entry_bar]
                    take_p = entry_price + take_atr_mult * atr[entry_bar]
                    
                    if low[i] <= stop_p: # 止损
                        exit_price = stop_p # 假设按止损价成交
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "stop_loss"])
                        current_pos = 0.0
                        is_closed = True
                    elif high[i] >= take_p: # 止盈
                        exit_price = take_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "take_profit"])
                        current_pos = 0.0
                        is_closed = True
                        
                elif current_pos < 0: # 空头
                    stop_p = entry_price + stop_atr_mult * atr[entry_bar]
                    take_p = entry_price - take_atr_mult * atr[entry_bar]
                    
                    if high[i] >= stop_p: # 止损
                        exit_price = stop_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "stop_loss"])
                        current_pos = 0.0
                        is_closed = True
                    elif low[i] <= take_p: # 止盈
                        exit_price = take_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "take_profit"])
                        current_pos = 0.0
                        is_closed = True

        # 2. 开仓/反手 (Signal Entry/Flip)
        if entries[i] != 0:
            # 如果当前空仓 -> 开仓
            if current_pos == 0:
                current_pos = entries[i]
                entry_price = close[i]
                entry_bar = i
                
            # 如果当前持有反向仓位 -> 反手 (Flip)
            elif np.sign(entries[i]) != np.sign(current_pos):
                # 先平掉旧仓
                exit_price = close[i]
                logret = np.log(exit_price / entry_price) * current_pos
                trades.append([m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "signal_flip"])
                
                # 再开新仓
                current_pos = entries[i]
                entry_price = close[i]
                entry_bar = i
        
        # 记录持仓状态
        pos[i] = current_pos

    pos[-1] = current_pos # 补齐最后一位

    if leverage_max is not None:
        pos = np.clip(pos, -float(leverage_max), float(leverage_max))

    m["position"] = pos
    m["ret_1"] = np.log(m["close"] / m["close"].shift(1)).fillna(0.0)
    m["turnover"] = m["position"].diff().abs().fillna(0.0)
    m["cost"] = m["turnover"] * float(cost_bps) / 10000.0
    m["strategy_ret"] = m["position"].shift(1).fillna(0.0) * m["ret_1"] - m["cost"]
    m["equity"] = np.exp(m["strategy_ret"].cumsum())

    # 生成 Trades DF
    trades_df = pd.DataFrame(trades, columns=["entry_time", "exit_time", "dir", "entry_px", "exit_px", "logret", "exit_reason"])

    m.to_csv(os.path.join(out_dir, "backtest_equity_curve.csv"), index=False, encoding="utf-8-sig")
    trades_df.to_csv(os.path.join(out_dir, "backtest_trades.csv"), index=False, encoding="utf-8-sig")
    
    return m, trades_df

def compute_daily_sharpe(strategy_ret: pd.Series, times: pd.Series) -> float:
    try:
        t = pd.to_datetime(times, errors="coerce")
        r = pd.to_numeric(strategy_ret, errors="coerce").fillna(0.0)
        df = pd.DataFrame({"time": t, "r": r}).dropna(subset=["time"])
        if len(df) < 10: return float("nan")
        daily = df.groupby(df["time"].dt.date)["r"].sum()
        if daily.std(ddof=1) == 0: return float("nan")
        return float(daily.mean() / daily.std(ddof=1) * np.sqrt(252.0))
    except:
        return float("nan")

def run_rf_pipeline_strict_last_month(super_df: pd.DataFrame, cfg: Config):
    out_dir = os.path.join(cfg.out_dir, cfg.rf_subdir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    df = super_df.reset_index().rename(columns={"index": "time"}).copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    df["super_state"] = pd.to_numeric(df["super_state"], errors="coerce").astype("Int64")
    df["y_ret_4"] = safe_log_return(df["close"].shift(-cfg.horizon_bars), df["close"])
    X, feature_cols = build_features_rf(df)
    data = pd.concat([df[["time", "close", "super_state"]], X, df["y_ret_4"]], axis=1).dropna()

    trade_days = int(getattr(cfg, "trade_days", 22))
    unique_days = sorted(data["time"].dt.date.unique())
    
    if getattr(cfg, "trade_start_date", None) and getattr(cfg, "trade_end_date", None):
        trade_start = pd.to_datetime(cfg.trade_start_date)
        trade_end = pd.to_datetime(cfg.trade_end_date)
        trade_df = data[(data["time"] >= trade_start) & (data["time"] <= trade_end)].copy()
    else:
        if len(unique_days) < trade_days:
            raise ValueError(f"交易日不足 {trade_days}")
        trade_day_set = set(unique_days[-trade_days:])
        trade_df = data[data["time"].dt.date.isin(trade_day_set)].copy()
        trade_start = trade_df["time"].min()

    train_start_fixed = pd.Timestamp(getattr(cfg, "rf_train_start_date", None) or "2022-04-11 10:00:00")
    hist_df = data[(data["time"] >= train_start_fixed) & (data["time"] < trade_start)].copy()
    
    hist_df = hist_df.sort_values("time")
    cut = int(len(hist_df) * cfg.hist_train_frac)
    train_df = hist_df.iloc[:cut].copy()
    val_df = hist_df.iloc[cut:].copy()

    states = sorted(train_df["super_state"].unique().tolist())
    models = {}
    valid_metrics = []

    for s in states:
        tr = train_df[train_df["super_state"] == s]
        te = val_df[val_df["super_state"] == s] 
        
        if len(tr) < cfg.min_train_rows_per_state: continue
        
        model = RandomForestRegressor(**cfg.rf_params)
        model.fit(tr[feature_cols].values, tr["y_ret_4"].values)
        models[int(s)] = model
        
        if len(te) >= cfg.min_test_rows_per_state:
            pred = model.predict(te[feature_cols].values)
            m = evaluate_regression(te["y_ret_4"].values, pred)
            valid_metrics.append({
                "super_state": int(s), "ic": m["ic"], "test_n": len(te), "r2": m["r2"]
            })

    vm = pd.DataFrame(valid_metrics)
    if vm.empty:
        good_states = []
    else:
        cands = vm[(vm["ic"] > cfg.min_ic) & (vm["test_n"] >= cfg.min_test_n) & (vm["r2"] > cfg.min_r2)]
        if cfg.max_states:
            cands = cands.sort_values("ic", ascending=False).head(cfg.max_states)
        good_states = sorted(cands["super_state"].tolist())
    
    gate_th = fit_gate_thresholds_robust(train_df, good_states, cfg)

    # stash for export
    try:
        cfg._rf_models = models
        cfg._last_good_states = list(good_states)
        cfg._last_gate_th = dict(gate_th) if isinstance(gate_th, dict) else gate_th
    except Exception:
        pass
    
    trade_df = trade_df.copy()
    trade_df["gate_on"] = gate_mask(trade_df, gate_th, good_states)
    
    preds = []
    for s in good_states:
        if s in models:
            sub = trade_df[trade_df["super_state"] == s]
            if len(sub) > 0:
                p = models[s].predict(sub[feature_cols].values)
                tmp = sub.copy()
                tmp["pred_y_ret_4"] = p
                preds.append(tmp)
    
    if preds:
        pred_all = pd.concat(preds).sort_values("time")
        # snapshot for live comparison (trade window full rows with predictions)
        try:
            trade_snapshot = trade_df.merge(
                pred_all[["time", "super_state", "pred_y_ret_4"]],
                on=["time", "super_state"],
                how="left",
            )
        except Exception:
            trade_snapshot = trade_df.copy()
        full_tl = trade_df[["time", "close"]].copy()
        
        # 【核心修复】真正传入 Config 中的风控参数
        bt_df, trades_info = backtest_using_full_timeline_df(
            full_tl, pred_all, out_dir, 
            cfg.horizon_bars, cfg.cost_bps_per_unit,
            stop_atr_mult=cfg.stop_atr_mult, # 传入 2.0
            take_atr_mult=cfg.take_atr_mult  # 传入 8.0
        )
        sharpe = compute_daily_sharpe(bt_df["strategy_ret"], bt_df["time"])
        final_eq = bt_df["equity"].iloc[-1]
        
        n_trades = len(trades_info)
        if len(bt_df) > 0 and "equity" in bt_df.columns:
            eq_arr = bt_df["equity"].values
            running_max = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - running_max) / running_max
            max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        else:
            max_dd = 0.0
        
    else:
        sharpe = np.nan
        final_eq = 1.0
        max_dd = 0.0
        n_trades = 0
        bt_df = pd.DataFrame()
        trade_snapshot = trade_df.copy() if isinstance(trade_df, pd.DataFrame) else pd.DataFrame()

    try:
        cfg._last_rf_stats = {
            "sharpe_daily": float(sharpe),
            "final_equity": float(final_eq),
            # drawdown is negative (e.g. -0.08)
            "max_drawdown": float(max_dd),
            "n_trades": int(n_trades),
            # for rolling_runner robust selector filters
            "active_bars": int((bt_df["position"] != 0).sum()) if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("position" in bt_df.columns) else int((bt_df["strategy_ret"] != 0).sum()) if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("strategy_ret" in bt_df.columns) else 0,
            "gate_coverage": float(bt_df["gate_on"].mean()) if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("gate_on" in bt_df.columns) else float("nan"),
            "gated_n": int(bt_df["gate_on"].sum()) if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("gate_on" in bt_df.columns) else 0,
        }
        write_run_summary_json(out_dir, cfg)
        # export live pack if requested
        if bool(getattr(cfg, "export_live_pack", False)):
            export_live_pack(out_dir, cfg, trade_df_snapshot=trade_snapshot)
    except:
        pass

def main():
    p = argparse.ArgumentParser()
    # ... args same as before ...
    p.add_argument("--input_csv", type=str)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--run_id", type=str)
    p.add_argument("--rs_5m", type=int)
    p.add_argument("--rs_30m", type=int)
    p.add_argument("--rs_1d", type=int)
    p.add_argument("--data_end_date", type=str)
    p.add_argument("--trade_start_date", type=str)
    p.add_argument("--trade_end_date", type=str)
    p.add_argument("--enable_backtest", type=str)
    p.add_argument("--input_tf_minutes", type=str)
    # export artifacts for live use (overwrite within the same run_dir)
    p.add_argument("--export_live_pack", type=str, default="0")
    p.add_argument("--live_pack_dir", type=str, default="live_pack")
    
    args, unknown = p.parse_known_args()
    
    cfg = Config()
    if args.input_csv: cfg.input_csv = args.input_csv
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.run_id: cfg.run_id = args.run_id
    if args.rs_5m: cfg.rs_5m = int(args.rs_5m)
    if args.rs_30m: cfg.rs_30m = int(args.rs_30m)
    if args.rs_1d: cfg.rs_1d = int(args.rs_1d)
    if args.data_end_date: cfg.data_end_date = args.data_end_date
    if args.trade_start_date: cfg.trade_start_date = args.trade_start_date
    if args.trade_end_date: cfg.trade_end_date = args.trade_end_date
    cfg.export_live_pack = (str(args.export_live_pack).strip() == "1")
    cfg.live_pack_dir = str(args.live_pack_dir).strip() if args.live_pack_dir else "live_pack"

    super_df = run_msp_pipeline(cfg)
    run_rf_pipeline_strict_last_month(super_df, cfg)

if __name__ == "__main__":
    main()