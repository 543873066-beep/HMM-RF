import os

import numpy as np
import pandas as pd


# migrated from: backtest_using_full_timeline_df
def backtest_using_full_timeline_df(
    full: pd.DataFrame,
    pred_all: pd.DataFrame,
    out_dir: str,
    horizon_bars: int,
    cost_bps: float,
    leverage_max: float = 1.0,
    restrict_to_pred_range: bool = True,
    # 銆愭牳蹇冧慨澶嶃€戠湡姝ｄ紶鍏ヨ繖浜涘弬鏁?
    stop_atr_mult: float = 2.0,
    take_atr_mult: float = 8.0,
):
    """
    銆愪慨澶嶇増 v2銆戠湡姝ｇ殑浜嬩欢椹卞姩寮曟搸锛屽惎鐢ˋTR椋庢帶
    """
    os.makedirs(out_dir, exist_ok=True)

    full = full.copy()
    full["time"] = pd.to_datetime(full["time"], errors="coerce")
    full = full.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)

    # 琛ュ叏OHLC (High/Low 鐢ㄤ簬ATR)
    if "high" not in full.columns:
        full["high"] = full["close"]
    if "low" not in full.columns:
        full["low"] = full["close"]
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

    # 鐢熸垚寮€浠撲俊鍙?
    m["entry_signal"] = 0.0
    sig_mask = m["gate_on"] & m["pred_y_ret_4"].notna()
    m.loc[sig_mask, "entry_signal"] = np.sign(m.loc[sig_mask, "pred_y_ret_4"])

    # --- 鏍稿績寰幆锛氫簨浠堕┍鍔ㄦā寮?---
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
    trades = []  # 璁板綍浜ゆ槗鏄庣粏

    for i in range(n - 1):
        # 1. 鐩樹腑椋庢帶 (浠呭綋鏈夋寔浠撴椂)
        if current_pos != 0:
            elapsed = i - entry_bar

            # (A) 鏃堕棿绂诲満 (Time Exit)
            if elapsed >= horizon_bars:
                # 骞充粨
                exit_price = close[i]
                logret = np.log(exit_price / entry_price) * current_pos
                trades.append(
                    [m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "time_exit"]
                )
                current_pos = 0.0

            # (B) 浠锋牸绂诲満 (Price Stop/Take)
            else:
                is_closed = False
                if current_pos > 0:  # 澶氬ご
                    stop_p = entry_price - stop_atr_mult * atr[entry_bar]
                    take_p = entry_price + take_atr_mult * atr[entry_bar]

                    if low[i] <= stop_p:  # 姝㈡崯
                        exit_price = stop_p  # 鍋囪鎸夋鎹熶环鎴愪氦
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append(
                            [
                                m.loc[entry_bar, "time"],
                                m.loc[i, "time"],
                                current_pos,
                                entry_price,
                                exit_price,
                                logret,
                                "stop_loss",
                            ]
                        )
                        current_pos = 0.0
                        is_closed = True
                    elif high[i] >= take_p:  # 姝㈢泩
                        exit_price = take_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append(
                            [
                                m.loc[entry_bar, "time"],
                                m.loc[i, "time"],
                                current_pos,
                                entry_price,
                                exit_price,
                                logret,
                                "take_profit",
                            ]
                        )
                        current_pos = 0.0
                        is_closed = True

                elif current_pos < 0:  # 绌哄ご
                    stop_p = entry_price + stop_atr_mult * atr[entry_bar]
                    take_p = entry_price - take_atr_mult * atr[entry_bar]

                    if high[i] >= stop_p:  # 姝㈡崯
                        exit_price = stop_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append(
                            [
                                m.loc[entry_bar, "time"],
                                m.loc[i, "time"],
                                current_pos,
                                entry_price,
                                exit_price,
                                logret,
                                "stop_loss",
                            ]
                        )
                        current_pos = 0.0
                        is_closed = True
                    elif low[i] <= take_p:  # 姝㈢泩
                        exit_price = take_p
                        logret = np.log(exit_price / entry_price) * current_pos
                        trades.append(
                            [
                                m.loc[entry_bar, "time"],
                                m.loc[i, "time"],
                                current_pos,
                                entry_price,
                                exit_price,
                                logret,
                                "take_profit",
                            ]
                        )
                        current_pos = 0.0
                        is_closed = True

        # 2. 寮€浠?鍙嶆墜 (Signal Entry/Flip)
        if entries[i] != 0:
            # 濡傛灉褰撳墠绌轰粨 -> 寮€浠?
            if current_pos == 0:
                current_pos = entries[i]
                entry_price = close[i]
                entry_bar = i

            # 濡傛灉褰撳墠鎸佹湁鍙嶅悜浠撲綅 -> 鍙嶆墜 (Flip)
            elif np.sign(entries[i]) != np.sign(current_pos):
                # 鍏堝钩鎺夋棫浠?
                exit_price = close[i]
                logret = np.log(exit_price / entry_price) * current_pos
                trades.append(
                    [m.loc[entry_bar, "time"], m.loc[i, "time"], current_pos, entry_price, exit_price, logret, "signal_flip"]
                )

                # 鍐嶅紑鏂颁粨
                current_pos = entries[i]
                entry_price = close[i]
                entry_bar = i

        # 璁板綍鎸佷粨鐘舵€?
        pos[i] = current_pos

    pos[-1] = current_pos  # 琛ラ綈鏈€鍚庝竴浣?

    if leverage_max is not None:
        pos = np.clip(pos, -float(leverage_max), float(leverage_max))

    m["position"] = pos
    m["ret_1"] = np.log(m["close"] / m["close"].shift(1)).fillna(0.0)
    m["turnover"] = m["position"].diff().abs().fillna(0.0)
    m["cost"] = m["turnover"] * float(cost_bps) / 10000.0
    m["strategy_ret"] = m["position"].shift(1).fillna(0.0) * m["ret_1"] - m["cost"]
    m["equity"] = np.exp(m["strategy_ret"].cumsum())

    # 鐢熸垚 Trades DF
    trades_df = pd.DataFrame(
        trades,
        columns=["entry_time", "exit_time", "dir", "entry_px", "exit_px", "logret", "exit_reason"],
    )

    m.to_csv(os.path.join(out_dir, "backtest_equity_curve.csv"), index=False, encoding="utf-8-sig")
    trades_df.to_csv(os.path.join(out_dir, "backtest_trades.csv"), index=False, encoding="utf-8-sig")

    return m, trades_df


# migrated from: compute_daily_sharpe
def compute_daily_sharpe(strategy_ret: pd.Series, times: pd.Series) -> float:
    try:
        t = pd.to_datetime(times, errors="coerce")
        r = pd.to_numeric(strategy_ret, errors="coerce").fillna(0.0)
        df = pd.DataFrame({"time": t, "r": r}).dropna(subset=["time"])
        if len(df) < 10:
            return float("nan")
        daily = df.groupby(df["time"].dt.date)["r"].sum()
        if daily.std(ddof=1) == 0:
            return float("nan")
        return float(daily.mean() / daily.std(ddof=1) * np.sqrt(252.0))
    except:
        return float("nan")


def run_backtest_placeholder(argv=None) -> int:
    if argv and any(a in ("-h", "--help") for a in argv):
        print("usage: qrs-new-backtest [--help]")
        print("placeholder: backtest stage")
        return 0
    return 0
