from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

# assembly-only module imports for migration wiring
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


def _build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace()
    cfg.tf_5m = "5min"
    cfg.min_count_5m = 1
    cfg.min_count_30m = 6
    cfg.ma_fast = 20
    cfg.ma_slow = 60
    cfg.vol_short = 20
    cfg.vol_long = 60
    cfg.rsi_n = 14
    cfg.atr_n = 14
    cfg.adx_n = 14
    cfg.input_tf_minutes = int(args.input_tf_minutes) if args.input_tf_minutes else 5
    return cfg


def _save_with_time_index(df, out_csv: Path) -> None:
    tmp = df.copy()
    tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "time"})
    tmp.to_csv(out_csv, index=False, encoding="utf-8-sig")


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
    _save_with_time_index(overlay_super_df, out_dir / "super_state_overlay_30m.csv")

    print(f"[QRS:new] features_5m rows={len(feat_5m)} cols={len(feat_5m.columns)}")
    print(f"[QRS:new] super_state rows={len(overlay_super_df)} cols={len(overlay_super_df.columns)}")
    print("[QRS:new] N3 pipeline finished")
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
