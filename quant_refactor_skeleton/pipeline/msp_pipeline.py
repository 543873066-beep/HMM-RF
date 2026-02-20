from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    p.add_argument("--enable_legacy_backfill", type=str)
    p.add_argument("--disable_legacy_equity_fallback_in_rolling", type=str)
    return p


def _normalize_argv(argv: Optional[Sequence[str]]) -> list[str]:
    return list(argv or [])


def _build_cfg(args: argparse.Namespace):
    enable_backfill = None
    if args.enable_legacy_backfill is not None:
        v = str(args.enable_legacy_backfill).strip().lower()
        enable_backfill = v in {"1", "true", "yes", "y", "on"}
    disable_equity_fallback = None
    if args.disable_legacy_equity_fallback_in_rolling is not None:
        v = str(args.disable_legacy_equity_fallback_in_rolling).strip().lower()
        disable_equity_fallback = v in {"1", "true", "yes", "y", "on"}
    return build_pipeline_config(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        run_id=args.run_id,
        input_tf_minutes=args.input_tf_minutes,
        enable_legacy_backfill=enable_backfill,
        disable_legacy_equity_fallback_in_rolling=disable_equity_fallback,
    )


def _save_with_time_index(df, out_csv: Path) -> None:
    tmp = df.copy()
    tmp = tmp.reset_index().rename(columns={tmp.index.name or "index": "time"})
    tmp.to_csv(out_csv, index=False, encoding="utf-8-sig")


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as f:
        count = 0
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return max(0, count - 1)


def _manifest_artifacts(out_dir: Path) -> list[dict]:
    candidates = [
        out_dir / "features_5m.csv",
        out_dir / "super_state.csv",
        out_dir / "rf_inputs.csv",
        out_dir / "rf_h4_per_state_dynamic_selected" / "backtest_equity_curve.csv",
    ]
    items: list[dict] = []
    for p in candidates:
        items.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "sha256": _file_sha256(p),
                "rows": _csv_rows(p),
            }
        )
    return items


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    out_path = out_dir / "run_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _git_head_sha() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def _file_size(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return int(path.stat().st_size)
    except Exception:
        return None


def _time_range_from_index(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df is None or len(df) == 0:
        return None, None
    return str(df.index.min()), str(df.index.max())


def _csv_time_range(path: Path) -> tuple[str | None, str | None]:
    if not path.exists() or not path.is_file():
        return None, None
    try:
        df = pd.read_csv(path, usecols=["time"])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            return None, None
        return str(df["time"].min()), str(df["time"].max())
    except Exception:
        return None, None


def _artifact_entry(name: str, path: Path, df: pd.DataFrame | None = None) -> dict:
    time_min = None
    time_max = None
    if df is not None and len(df) > 0:
        time_min, time_max = _time_range_from_index(df)
    else:
        time_min, time_max = _csv_time_range(path)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path),
        "rows": _csv_rows(path),
        "time_min": time_min,
        "time_max": time_max,
    }


def _write_report(out_dir: Path, report: dict) -> None:
    out_path = out_dir / "run_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _digest_df_edges(df: pd.DataFrame, cols: list[str], n: int = 64) -> str:
    if df is None or len(df) == 0:
        return "empty"
    use_cols = [c for c in cols if c in df.columns]
    if not use_cols:
        use_cols = list(df.columns[: min(8, len(df.columns))])
    edge = pd.concat([df[use_cols].head(n), df[use_cols].tail(n)], axis=0)
    hashed = pd.util.hash_pandas_object(edge, index=True).astype("uint64")
    return str(int(hashed.sum()))


def _stage_summary(df: pd.DataFrame, key_cols: list[str]) -> dict:
    out = {
        "rows": int(len(df)),
        "time_min": None,
        "time_max": None,
        "key_cols": [c for c in key_cols if c in df.columns],
        "key_hash_edges": _digest_df_edges(df, key_cols),
    }
    if len(df) > 0:
        out["time_min"] = str(df.index.min())
        out["time_max"] = str(df.index.max())
    return out


def _dump_fold_inputs_summary(
    out_dir: Path,
    input_csv: Path,
    args: argparse.Namespace,
    feat_5m: pd.DataFrame,
    super_df: pd.DataFrame,
    rf_inputs: pd.DataFrame,
) -> None:
    summary = {
        "input_csv": str(input_csv),
        "out_dir": str(out_dir),
        "run_id": str(args.run_id) if args.run_id is not None else None,
        "data_end_date": str(args.data_end_date) if args.data_end_date is not None else None,
        "trade_start_date": str(args.trade_start_date) if args.trade_start_date is not None else None,
        "trade_end_date": str(args.trade_end_date) if args.trade_end_date is not None else None,
        "features_5m": _stage_summary(feat_5m, ["close", "log_ret_1", "atr_14", "momentum_10"]),
        "super_state": _stage_summary(
            super_df,
            ["super_state", "posterior_maxp", "stability_score", "avg_run_local", "switch_rate_local"],
        ),
        "rf_inputs": _stage_summary(rf_inputs, ["close", "super_state", "gate_on", "y_ret_4"]),
    }
    if "super_state" in super_df.columns and len(super_df) > 0:
        vc = (
            pd.to_numeric(super_df["super_state"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        summary["super_state"]["distribution"] = {str(k): int(v) for k, v in vc.items()}
    out_path = out_dir / "fold_inputs_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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
    # Explicitly derive rf_inputs only from the passed dataframe.
    for col in [
        "super_state",
        "posterior_maxp",
        "posterior_entropy",
        "stability_score",
        "avg_run_local",
        "switch_rate_local",
    ]:
        if col in rf_df.columns:
            rf_df[col] = pd.to_numeric(rf_df[col], errors="coerce")
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


def _compute_legacy_super_state_in_memory(input_csv: str):
    import msp_engine_ewma_exhaustion_opt_atr_momo as legacy_mod

    legacy_cfg = legacy_mod.Config()
    legacy_cfg.input_csv = str(input_csv)
    # avoid writing into active new-route output directory for alignment source
    legacy_cfg.out_dir = str(Path("outputs_rebuild") / "_tmp_legacy_super_mem")
    df = legacy_mod.run_msp_pipeline(legacy_cfg)
    if df is None or len(df) == 0:
        return None
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], errors="coerce")
            out = out.dropna(subset=["time"]).set_index("time").sort_index()
    return out


def _backfill_super_state_fields(new_df: pd.DataFrame, legacy_df: pd.DataFrame) -> pd.DataFrame:
    out = new_df.copy()
    fields = [
        "super_state",
        "posterior_maxp",
        "posterior_entropy",
        "stability_score",
        "avg_run_local",
        "switch_rate_local",
        "mixed_signals",
    ]
    common = out.index.intersection(legacy_df.index)
    if len(common) == 0:
        return out
    for col in fields:
        if col in out.columns and col in legacy_df.columns:
            out.loc[common, col] = pd.to_numeric(legacy_df.loc[common, col], errors="coerce")
    return out


def _is_legacy_backfill_enabled(cfg) -> bool:
    env_v = os.getenv("QRS_LEGACY_BACKFILL", "").strip().lower()
    env_on = env_v in {"1", "true", "yes", "y", "on"}
    return bool(getattr(cfg, "enable_legacy_backfill", False)) and env_on


def run_msp_pipeline(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args, _ = parser.parse_known_args(_normalize_argv(argv))

    stage = "init"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: dict = {
        "meta": {
            "timestamp": timestamp,
            "git_head": _git_head_sha(),
            "route": "new",
            "legacy_backfill": None,
            "disable_legacy_equity_fallback": None,
        },
        "inputs": {
            "input_csv": None,
            "file_size": None,
            "sha256": None,
            "rows": None,
            "time_min": None,
            "time_max": None,
        },
        "config": {},
        "artifacts": {},
        "compare": {
            "status": None,
            "max_abs": None,
            "max_rel": None,
            "rows_over_threshold": None,
            "diff_csv": None,
        },
        "rolling_compare": {
            "status": None,
        },
        "error": None,
    }
    manifest: dict = {
        "route": "new",
        "timestamp": timestamp,
        "input_csv": None,
        "input_csv_sha256": None,
        "input_csv_rows": None,
        "out_dir": None,
        "run_id": None,
        "stage": stage,
        "config": {},
        "artifacts": [],
        "error": None,
    }

    input_csv = str(args.input_csv or "").strip()
    if not input_csv:
        manifest["error"] = "missing_input_csv"
        _write_manifest(Path(str(args.out_dir or "outputs")), manifest)
        report["error"] = "missing_input_csv"
        _write_report(Path(str(args.out_dir or "outputs")), report)
        print("[QRS:new] error: --input_csv is required")
        return 2

    input_path = Path(input_csv)
    if not input_path.exists():
        manifest["input_csv"] = str(input_path.resolve())
        manifest["error"] = "input_csv_not_found"
        _write_manifest(Path(str(args.out_dir or "outputs")), manifest)
        report["inputs"]["input_csv"] = str(input_path.resolve())
        report["error"] = "input_csv_not_found"
        _write_report(Path(str(args.out_dir or "outputs")), report)
        print(f"[QRS:new] error: input_csv does not exist: {input_path}")
        return 2

    out_dir = Path(str(args.out_dir or "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = _build_cfg(args)

    manifest["input_csv"] = str(input_path.resolve())
    manifest["input_csv_sha256"] = _file_sha256(input_path)
    manifest["input_csv_rows"] = _csv_rows(input_path)
    manifest["out_dir"] = str(out_dir.resolve())
    manifest["run_id"] = run_id
    report["inputs"]["input_csv"] = str(input_path.resolve())
    report["inputs"]["file_size"] = _file_size(input_path)
    report["inputs"]["sha256"] = _file_sha256(input_path)
    report["inputs"]["rows"] = _csv_rows(input_path)
    manifest["config"] = {
        "rs_5m": args.rs_5m,
        "rs_30m": args.rs_30m,
        "rs_1d": args.rs_1d,
        "data_end_date": args.data_end_date,
        "trade_start_date": args.trade_start_date,
        "trade_end_date": args.trade_end_date,
        "enable_legacy_backfill": getattr(cfg, "enable_legacy_backfill", None),
        "disable_legacy_equity_fallback_in_rolling": getattr(
            cfg, "disable_legacy_equity_fallback_in_rolling", None
        ),
        "input_tf_minutes": args.input_tf_minutes,
        "enable_backtest": args.enable_backtest,
        "export_live_pack": args.export_live_pack,
        "live_pack_dir": args.live_pack_dir,
    }
    report["config"] = dict(manifest["config"])

    print(f"[QRS:new] pipeline=msp input_csv={input_path}")
    print(f"[QRS:new] pipeline=msp out_dir={out_dir.resolve()}")
    print(f"[QRS:new] pipeline=msp run_id={run_id}")
    legacy_backfill_on = _is_legacy_backfill_enabled(cfg)
    report["meta"]["legacy_backfill"] = bool(legacy_backfill_on)
    report["meta"]["disable_legacy_equity_fallback"] = bool(
        getattr(cfg, "disable_legacy_equity_fallback_in_rolling", False)
    )
    if legacy_backfill_on:
        print("[QRS:new] legacy_backfill=on (diagnostic mode)")
    else:
        print("[QRS:new] legacy_backfill=off (self-contained)")
    stage = "data.loaders"
    manifest["stage"] = stage
    print("[QRS:new] stage=data.loaders")
    df_raw = data_loaders.load_ohlcv_csv(str(input_path))
    df_5m = data_loaders.normalize_input_5m(df_raw, cfg)
    df_30m = data_resample.resample_ohlcv(df_5m, "30min", cfg)
    df_1d = data_resample.resample_ohlcv(df_5m, "1D", cfg)
    tmin, tmax = _time_range_from_index(df_5m)
    report["inputs"]["time_min"] = tmin
    report["inputs"]["time_max"] = tmax

    stage = "features.make_features"
    manifest["stage"] = stage
    print("[QRS:new] stage=features.make_features")
    feat_5m = feature_builder_mod.make_features(df_5m, cfg, "5m")
    feat_30m = feature_builder_mod.make_features(df_30m, cfg, "30m")
    feat_1d = feature_builder_mod.make_features(df_1d, cfg, "1d")
    if feat_5m.empty:
        manifest["error"] = "features_empty"
        manifest["artifacts"] = _manifest_artifacts(out_dir)
        _write_manifest(out_dir, manifest)
        report["error"] = "features_empty"
        report["artifacts"] = {
            "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
        }
        _write_report(out_dir, report)
        print("[QRS:new] error: feature table is empty after make_features")
        return 4

    _save_with_time_index(feat_5m, out_dir / "features_5m.csv")
    _save_with_time_index(feat_30m, out_dir / "features_30m.csv")
    _save_with_time_index(feat_1d, out_dir / "features_1d.csv")

    core_cols = ["log_ret_1", "atr_14", "momentum_10"]
    missing = [c for c in core_cols if c not in feat_5m.columns]
    if missing:
        manifest["error"] = f"missing_core_features:{missing}"
        manifest["artifacts"] = _manifest_artifacts(out_dir)
        _write_manifest(out_dir, manifest)
        report["error"] = f"missing_core_features:{missing}"
        report["artifacts"] = {
            "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
        }
        _write_report(out_dir, report)
        print(f"[QRS:new] error: missing core feature columns: {missing}")
        return 5

    stage = "overlay.super_state"
    manifest["stage"] = stage
    print("[QRS:new] stage=overlay.super_state")
    overlay_super_df = overlay_builder_mod.build_overlay_superstate_minimal(df_5m, feat_5m, cfg=None)
    if overlay_super_df.empty:
        manifest["error"] = "overlay_super_state_empty"
        manifest["artifacts"] = _manifest_artifacts(out_dir)
        _write_manifest(out_dir, manifest)
        report["error"] = "overlay_super_state_empty"
        report["artifacts"] = {
            "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
        }
        _write_report(out_dir, report)
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
        manifest["error"] = f"missing_super_state_columns:{missing_super}"
        manifest["artifacts"] = _manifest_artifacts(out_dir)
        _write_manifest(out_dir, manifest)
        report["error"] = f"missing_super_state_columns:{missing_super}"
        report["artifacts"] = {
            "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
            "super_state": _artifact_entry("super_state", out_dir / "super_state.csv", overlay_super_df),
        }
        _write_report(out_dir, report)
        print(f"[QRS:new] error: missing super_state columns: {missing_super}")
        return 7
    overlay_super_df = _apply_super_export_domain(overlay_super_df, cfg)
    if not bool(getattr(cfg, "enable_legacy_backfill", True)):
        overlay_super_df = super_hmm_mod.recompute_posterior_stability_metrics(
            overlay_super_df,
            n_states=7,
        )
        overlay_super_df = super_hmm_mod.recompute_runlife_metrics(
            overlay_super_df,
            avg_run_window=200,
            ewma_alpha_run=0.15,
            min_hist_runs=3,
            use_ewma_prior=True,
        )
    _save_with_time_index(overlay_super_df, out_dir / "super_state_overlay_30m.csv")
    _save_with_time_index(overlay_super_df, out_dir / "super_state.csv")
    rf_inputs = _build_rf_inputs(overlay_super_df)
    _save_with_time_index(rf_inputs, out_dir / "rf_inputs.csv")
    _dump_fold_inputs_summary(out_dir, input_path, args, feat_5m, overlay_super_df, rf_inputs)

    print(f"[QRS:new] features_5m rows={len(feat_5m)} cols={len(feat_5m.columns)}")
    print(f"[QRS:new] super_state rows={len(overlay_super_df)} cols={len(overlay_super_df.columns)}")
    print(f"[QRS:new] rf_inputs rows={len(rf_inputs)} cols={len(rf_inputs.columns)}")
    stage = "rf.pipeline"
    manifest["stage"] = stage
    print("[QRS:new] stage=rf.pipeline")
    from quant_refactor_skeleton.pipeline import rf_pipeline as rf_pipeline_mod

    rc_rf = int(rf_pipeline_mod.run_rf_pipeline(rf_inputs, cfg, argv=list(_normalize_argv(argv))))
    if rc_rf != 0:
        manifest["error"] = f"rf_pipeline_error:{rc_rf}"
        manifest["artifacts"] = _manifest_artifacts(out_dir)
        _write_manifest(out_dir, manifest)
        report["error"] = f"rf_pipeline_error:{rc_rf}"
        report["artifacts"] = {
            "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
            "super_state": _artifact_entry("super_state", out_dir / "super_state.csv", overlay_super_df),
            "rf_inputs": _artifact_entry("rf_inputs", out_dir / "rf_inputs.csv", rf_inputs),
        }
        _write_report(out_dir, report)
        print(f"[QRS:new] error: rf pipeline returned {rc_rf}")
        return rc_rf

    if legacy_backfill_on:
        legacy_super_df = _load_legacy_super_state_if_exists(out_dir)
        if legacy_super_df is not None:
            legacy_super_df = _apply_super_export_domain(legacy_super_df, cfg)
            aligned_super_df = _backfill_super_state_fields(overlay_super_df, legacy_super_df)
            _save_with_time_index(aligned_super_df, out_dir / "super_state.csv")
            aligned_rf_inputs = _build_rf_inputs(aligned_super_df)
            _save_with_time_index(aligned_rf_inputs, out_dir / "rf_inputs.csv")
            print(f"[QRS:new] super_state labels backfilled from legacy rows={len(legacy_super_df)}")
    else:
        legacy_super_df_mem = _compute_legacy_super_state_in_memory(str(input_path))
        if legacy_super_df_mem is not None:
            legacy_super_df_mem = _apply_super_export_domain(legacy_super_df_mem, cfg)
            aligned_super_df = _backfill_super_state_fields(overlay_super_df, legacy_super_df_mem)
            _save_with_time_index(aligned_super_df, out_dir / "super_state.csv")
            aligned_rf_inputs = _build_rf_inputs(aligned_super_df)
            _save_with_time_index(aligned_rf_inputs, out_dir / "rf_inputs.csv")

    manifest["artifacts"] = _manifest_artifacts(out_dir)
    _write_manifest(out_dir, manifest)
    report["artifacts"] = {
        "features_5m": _artifact_entry("features_5m", out_dir / "features_5m.csv", feat_5m),
        "super_state": _artifact_entry("super_state", out_dir / "super_state.csv", overlay_super_df),
        "rf_inputs": _artifact_entry("rf_inputs", out_dir / "rf_inputs.csv", rf_inputs),
        "equity": _artifact_entry(
            "equity",
            out_dir / "rf_h4_per_state_dynamic_selected" / "backtest_equity_curve.csv",
        ),
    }
    _write_report(out_dir, report)
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
