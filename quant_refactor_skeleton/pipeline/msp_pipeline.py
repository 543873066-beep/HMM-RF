from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# assembly-only module imports for migration wiring
from quant_refactor_skeleton.core.config import build_pipeline_config
from quant_refactor_skeleton.data import loaders as data_loaders
from quant_refactor_skeleton.data import resample as data_resample
from quant_refactor_skeleton.features import feature_builder as feature_builder_mod
from quant_refactor_skeleton.hmm import model as hmm_model
from quant_refactor_skeleton.hmm import portrait as hmm_portrait
from quant_refactor_skeleton.hmm import states as hmm_states
from quant_refactor_skeleton.overlay import overlay_builder as overlay_builder_mod
from quant_refactor_skeleton.super_state import super_hmm as super_hmm_mod


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QRS new-route MSP pipeline (skeleton)")
    p.add_argument("--input_csv", type=str, default=r"data/sh000852_5m.csv")
    p.add_argument("--out_dir", type=str, default=r"outputs")
    p.add_argument("--run_id", type=str)
    p.add_argument("--rs_5m", type=int)
    p.add_argument("--rs_30m", type=int)
    p.add_argument("--rs_1d", type=int)
    p.add_argument("--data_end_date", type=str)
    p.add_argument("--trade_start_date", type=str)
    p.add_argument("--trade_end_date", type=str)
    p.add_argument("--enable_backtest", type=str)
    p.add_argument("--input_tf_minutes", type=int)
    p.add_argument("--export_live_pack", type=str)
    p.add_argument("--live_pack_dir", type=str)
    return p


def _normalize_argv(argv: Optional[Sequence[str]]) -> list[str]:
    return list(argv or [])


def _build_cfg(args: argparse.Namespace):
    return build_pipeline_config(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        run_id=args.run_id,
        input_tf_minutes=args.input_tf_minutes,
    )


def _save_with_time_index(df, out_csv: Path) -> None:
    tmp = df.copy()
    tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "time"})
    tmp.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _apply_super_export_domain(df, cfg):
    out = df.copy()
    start = getattr(cfg, "super_state_export_start_date", None)
    end = getattr(cfg, "super_state_export_end_date", None)
    if start:
        start_dt = np.datetime64(start)
        out = out[out.index >= start_dt]
    if end:
        end_dt = np.datetime64(end)
        out = out[out.index <= end_dt]
    return out


def _build_rf_inputs(super_df):
    from quant_refactor_skeleton.rf.dataset import CONT_FEATURES, safe_log_return

    rf_df = super_df.copy()
    for col in CONT_FEATURES:
        if col not in rf_df.columns:
            rf_df[col] = np.nan
    for col in ["ma_fast", "ma_slow"]:
        if col not in rf_df.columns:
            rf_df[col] = np.nan
    rf_df["y_ret_4"] = safe_log_return(rf_df["close"].shift(-4), rf_df["close"])
    if "gate_on" not in rf_df.columns:
        rf_df["gate_on"] = False
    keep = ["close", "super_state", "gate_on", "y_ret_4"] + ["ma_fast", "ma_slow"] + list(CONT_FEATURES)
    keep = [c for c in keep if c in rf_df.columns]
    return rf_df[keep].copy()


def _load_legacy_super_state_if_exists(out_dir: Path):
    legacy_path = out_dir / "super_states_30m.csv"
    if not legacy_path.exists():
        return None
    df = pd.read_csv(legacy_path)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return df


def run_msp_pipeline(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, _ = parser.parse_known_args(_normalize_argv(argv))

    input_csv = str(args.input_csv or "").strip()
    if not input_csv:
        print("[QRS:new] error: --input_csv is required")
        return 2

    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"[QRS:new] error: input_csv does not exist: {input_path}")
        return 2

    out_dir = Path(str(args.out_dir or "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = _build_cfg(args)

    print(f"[QRS:new] pipeline=msp input_csv={input_path}")
    print(f"[QRS:new] pipeline=msp out_dir={out_dir.resolve()}")
    print(f"[QRS:new] pipeline=msp run_id={run_id}")
    print("[QRS:new] stage=data.loaders")
    df_raw = data_loaders.load_ohlcv_csv(str(input_path))
    df_5m = data_loaders.normalize_input_5m(df_raw, cfg)
    df_30m = data_resample.resample_ohlcv(df_5m, "30min", cfg)
    df_1d = data_resample.resample_ohlcv(df_5m, "1D", cfg)

    print("[QRS:new] stage=features.make_features")
    feat_5m = feature_builder_mod.make_features(df_5m, cfg, "5m")
    feat_30m = feature_builder_mod.make_features(df_30m, cfg, "30m")
    feat_1d = feature_builder_mod.make_features(df_1d, cfg, "1d")
    if feat_5m.empty:
        print("[QRS:new] error: feature table is empty after make_features")
        return 4

    _save_with_time_index(feat_5m, out_dir / "features_5m.csv")
    _save_with_time_index(feat_30m, out_dir / "features_30m.csv")
    _save_with_time_index(feat_1d, out_dir / "features_1d.csv")

    core_cols = ["log_ret_1", "atr_14", "momentum_10"]
    missing = [c for c in core_cols if c not in feat_5m.columns]
    if missing:
        print(f"[QRS:new] error: missing core feature columns: {missing}")
        return 5

    print("[QRS:new] stage=overlay.super_state")
    overlay_super_df = overlay_builder_mod.build_overlay_superstate_minimal(df_5m, feat_5m, cfg=None)
    if overlay_super_df.empty:
        print("[QRS:new] error: overlay/super_state table is empty")
        return 6
    required_super_cols = [
        "overlay_id",
        "overlay_tuple",
        "super_state",
        "posterior_maxp",
        "stability_score",
        "avg_run_local",
        "switch_rate_local",
    ]
    missing_super = [c for c in required_super_cols if c not in overlay_super_df.columns]
    if missing_super:
        print(f"[QRS:new] error: missing super_state columns: {missing_super}")
        return 7
    overlay_super_df = _apply_super_export_domain(overlay_super_df, cfg)
    _save_with_time_index(overlay_super_df, out_dir / "super_state_overlay_30m.csv")
    _save_with_time_index(overlay_super_df, out_dir / "super_state.csv")
    rf_inputs = _build_rf_inputs(overlay_super_df)
    _save_with_time_index(rf_inputs, out_dir / "rf_inputs.csv")

    print(f"[QRS:new] features_5m rows={len(feat_5m)} cols={len(feat_5m.columns)}")
    print(f"[QRS:new] super_state rows={len(overlay_super_df)} cols={len(overlay_super_df.columns)}")
    print(f"[QRS:new] rf_inputs rows={len(rf_inputs)} cols={len(rf_inputs.columns)}")
    print("[QRS:new] stage=rf.pipeline")
    from quant_refactor_skeleton.pipeline import rf_pipeline as rf_pipeline_mod

    rc_rf = int(rf_pipeline_mod.run_rf_pipeline(rf_inputs, cfg, argv=list(_normalize_argv(argv))))
    if rc_rf != 0:
        print(f"[QRS:new] error: rf pipeline returned {rc_rf}")
        return rc_rf

    legacy_super_df = _load_legacy_super_state_if_exists(out_dir)
    if legacy_super_df is not None:
        legacy_super_df = _apply_super_export_domain(legacy_super_df, cfg)
        _save_with_time_index(legacy_super_df, out_dir / "super_state.csv")
        legacy_rf_inputs = _build_rf_inputs(legacy_super_df)
        _save_with_time_index(legacy_rf_inputs, out_dir / "rf_inputs.csv")
        print(f"[QRS:new] super_state backfilled from legacy rows={len(legacy_super_df)}")

    print("[QRS:new] N4 pipeline finished")
    return 0


__all__ = [
    "data_loaders",
    "data_resample",
    "feature_builder_mod",
    "hmm_model",
    "hmm_portrait",
    "hmm_states",
    "overlay_builder_mod",
    "run_msp_pipeline",
    "super_hmm_mod",
]
