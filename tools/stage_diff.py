from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_refactor_skeleton.core.config import build_pipeline_config
from quant_refactor_skeleton.data.loaders import load_ohlcv_csv, normalize_input_5m
from quant_refactor_skeleton.features.feature_builder import make_features


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame, key_cols: list[str]) -> dict[str, float]:
    out = {}
    for c in key_cols:
        if c not in a.columns or c not in b.columns:
            continue
        aa = pd.to_numeric(a[c], errors="coerce")
        bb = pd.to_numeric(b[c], errors="coerce")
        mask = aa.notna() & bb.notna()
        if mask.any():
            out[c] = float((aa[mask] - bb[mask]).abs().max())
        else:
            out[c] = float("nan")
    return out


def _build_legacy_features(input_csv: str) -> pd.DataFrame:
    cfg = build_pipeline_config(input_csv=input_csv)
    raw = load_ohlcv_csv(input_csv)
    bars = normalize_input_5m(raw, cfg)
    feat = make_features(bars, cfg, "5m")
    return feat.reset_index().rename(columns={feat.index.name or "index": "time"})


def _build_legacy_rf_inputs(legacy_super: pd.DataFrame) -> pd.DataFrame:
    work = legacy_super.copy()
    if "time" in work.columns:
        work["time"] = pd.to_datetime(work["time"], errors="coerce")
    for c in ["close", "super_state"]:
        if c not in work.columns:
            work[c] = np.nan
    for c in ["gate_on", "y_ret_4"]:
        if c not in work.columns:
            work[c] = np.nan
    return work


def main() -> int:
    p = argparse.ArgumentParser(description="Stage diff summary for new-route alignment")
    p.add_argument("--legacy-dir", required=True)
    p.add_argument("--new-dir", required=True)
    p.add_argument("--input-csv", required=True)
    args = p.parse_args()

    legacy_dir = Path(args.legacy_dir)
    new_dir = Path(args.new_dir)

    new_features = pd.read_csv(new_dir / "features_5m.csv")
    new_super = pd.read_csv(new_dir / "super_state.csv")
    new_rf_inputs = pd.read_csv(new_dir / "rf_inputs.csv")

    legacy_features = _build_legacy_features(args.input_csv)
    legacy_super_path = legacy_dir / "super_states_30m.csv"
    legacy_super = pd.read_csv(legacy_super_path) if legacy_super_path.exists() else pd.DataFrame()
    legacy_rf_inputs = _build_legacy_rf_inputs(legacy_super)
    new_features["time"] = pd.to_datetime(new_features["time"], errors="coerce")
    legacy_features["time"] = pd.to_datetime(legacy_features["time"], errors="coerce")
    if not legacy_super.empty and "time" in legacy_super.columns:
        legacy_super["time"] = pd.to_datetime(legacy_super["time"], errors="coerce")
    if "time" in new_super.columns:
        new_super["time"] = pd.to_datetime(new_super["time"], errors="coerce")

    print("[STAGE-DIFF] features_5m")
    print(f"  legacy_rows={len(legacy_features)} new_rows={len(new_features)}")
    print(f"  columns_same_set={set(legacy_features.columns) == set(new_features.columns)}")
    print(f"  columns_same_order={list(legacy_features.columns) == list(new_features.columns)}")
    feature_key = ["log_ret_1", "atr_14", "momentum_10"]
    f_merge = pd.merge(
        legacy_features[["time"] + [c for c in feature_key if c in legacy_features.columns]],
        new_features[["time"] + [c for c in feature_key if c in new_features.columns]],
        on="time",
        suffixes=("_old", "_new"),
        how="inner",
    )
    f_old = pd.DataFrame({c: f_merge.get(f"{c}_old") for c in feature_key})
    f_new = pd.DataFrame({c: f_merge.get(f"{c}_new") for c in feature_key})
    print(f"  key_max_abs_diff={_max_abs_diff(f_old, f_new, feature_key)}")

    print("[STAGE-DIFF] super_state")
    super_key = ["super_state", "posterior_maxp", "stability_score", "avg_run_local", "switch_rate_local"]
    if legacy_super.empty:
        print("  legacy super_states_30m.csv not found")
    else:
        s_merge = pd.merge(
            legacy_super[["time"] + [c for c in super_key if c in legacy_super.columns]],
            new_super[["time"] + [c for c in super_key if c in new_super.columns]],
            on="time",
            suffixes=("_old", "_new"),
            how="inner",
        )
        s_old = pd.DataFrame({c: s_merge.get(f"{c}_old") for c in super_key})
        s_new = pd.DataFrame({c: s_merge.get(f"{c}_new") for c in super_key})
        print(f"  legacy_rows={len(legacy_super)} new_rows={len(new_super)} merged_rows={len(s_merge)}")
        print(f"  key_max_abs_diff={_max_abs_diff(s_old, s_new, super_key)}")

    print("[STAGE-DIFF] rf_inputs")
    print(f"  legacy_rows={len(legacy_rf_inputs)} new_rows={len(new_rf_inputs)}")
    if "time" in legacy_rf_inputs.columns and "time" in new_rf_inputs.columns:
        t_old_min = pd.to_datetime(legacy_rf_inputs["time"], errors="coerce").min()
        t_old_max = pd.to_datetime(legacy_rf_inputs["time"], errors="coerce").max()
        t_new_min = pd.to_datetime(new_rf_inputs["time"], errors="coerce").min()
        t_new_max = pd.to_datetime(new_rf_inputs["time"], errors="coerce").max()
        print(f"  legacy_time_range={t_old_min} -> {t_old_max}")
        print(f"  new_time_range={t_new_min} -> {t_new_max}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
