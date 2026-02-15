import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quant_refactor_skeleton.rf.backtest import backtest_using_full_timeline_df, compute_daily_sharpe
from quant_refactor_skeleton.rf.dataset import build_features_rf, evaluate_regression, safe_log_return
from quant_refactor_skeleton.rf.gates import fit_gate_thresholds_robust, gate_mask


# migrated from: run_rf_pipeline_strict_last_month
def run_rf_pipeline_strict_last_month(super_df: pd.DataFrame, cfg: "Config"):
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
            raise ValueError(f"浜ゆ槗鏃ヤ笉瓒?{trade_days}")
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

        if len(tr) < cfg.min_train_rows_per_state:
            continue

        model = RandomForestRegressor(**cfg.rf_params)
        model.fit(tr[feature_cols].values, tr["y_ret_4"].values)
        models[int(s)] = model

        if len(te) >= cfg.min_test_rows_per_state:
            pred = model.predict(te[feature_cols].values)
            m = evaluate_regression(te["y_ret_4"].values, pred)
            valid_metrics.append({"super_state": int(s), "ic": m["ic"], "test_n": len(te), "r2": m["r2"]})

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

        # 銆愭牳蹇冧慨澶嶃€戠湡姝ｄ紶鍏?Config 涓殑椋庢帶鍙傛暟
        bt_df, trades_info = backtest_using_full_timeline_df(
            full_tl,
            pred_all,
            out_dir,
            cfg.horizon_bars,
            cfg.cost_bps_per_unit,
            stop_atr_mult=cfg.stop_atr_mult,  # 浼犲叆 2.0
            take_atr_mult=cfg.take_atr_mult,  # 浼犲叆 8.0
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
            "active_bars": int((bt_df["position"] != 0).sum())
            if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("position" in bt_df.columns)
            else int((bt_df["strategy_ret"] != 0).sum())
            if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("strategy_ret" in bt_df.columns)
            else 0,
            "gate_coverage": float(bt_df["gate_on"].mean())
            if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("gate_on" in bt_df.columns)
            else float("nan"),
            "gated_n": int(bt_df["gate_on"].sum())
            if isinstance(bt_df, pd.DataFrame) and (not bt_df.empty) and ("gate_on" in bt_df.columns)
            else 0,
        }
        from quant_refactor_skeleton.live.export import export_live_pack, write_run_summary_json

        write_run_summary_json(out_dir, cfg)
        # export live pack if requested
        if bool(getattr(cfg, "export_live_pack", False)):
            export_live_pack(out_dir, cfg, trade_df_snapshot=trade_snapshot)
    except:
        pass
