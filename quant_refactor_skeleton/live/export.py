import json
import os
from typing import Dict, Optional

import joblib
import pandas as pd


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
                cols = [
                    c
                    for c in ["time", "close", "super_state", "gate_on", "pred_y_ret_4", "y_ret_4"]
                    if c in trade_df_snapshot.columns
                ]
                trade_df_snapshot[cols].to_csv(snap_path, index=False, encoding="utf-8-sig")

        # --- path metrics seed (for live loader: stable avg_run_local / switch_rate_local without long warmup) ---
        try:
            if isinstance(trade_df_snapshot, pd.DataFrame) and (not trade_df_snapshot.empty) and ("super_state" in trade_df_snapshot.columns):
                s_seed = pd.to_numeric(trade_df_snapshot["super_state"], errors="coerce").dropna().astype(int).values
                if len(s_seed) >= 20:
                    from quant_refactor_skeleton.super_state.runlife_utils import compute_path_stats_seed_expected_life

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


__all__ = ["export_live_pack", "write_run_summary_json"]
