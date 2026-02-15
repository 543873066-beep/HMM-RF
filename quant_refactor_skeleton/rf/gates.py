import numpy as np
import pandas as pd


def fit_gate_thresholds_robust(train_df: pd.DataFrame, allowed_states: list, cfg: "Config"):
    df = train_df[train_df["super_state"].isin(allowed_states)].copy()
    runl = pd.to_numeric(df["avg_run_local"], errors="coerce")
    sw = pd.to_numeric(df["switch_rate_local"], errors="coerce")
    run_th = float(runl.dropna().quantile(cfg.gate_q_run))
    sw_th = float(sw.dropna().quantile(1.0 - cfg.gate_q_switch))
    run_th = min(run_th, cfg.clamp_avg_run_max)
    sw_th = max(sw_th, cfg.clamp_switch_min)
    return {
        "stability_min": cfg.stability_static_min,
        "avg_run_min": run_th,
        "switch_rate_max": sw_th,
        "exhaustion_min": float(getattr(cfg, "exhaustion_min", 0.0)),
        "exhaustion_max": float(getattr(cfg, "exhaustion_max", 1.2)),
    }


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
