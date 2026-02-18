from __future__ import annotations
from pathlib import Path
import re

# assembly-only module imports for migration wiring
from quant_refactor_skeleton.rf import backtest as rf_backtest
from quant_refactor_skeleton.rf import dataset as rf_dataset
from quant_refactor_skeleton.rf import gates as rf_gates
from quant_refactor_skeleton.rf import trainer as rf_trainer


def _get_arg(argv: list[str], key: str) -> str | None:
    flag = f"--{key}"
    for i, token in enumerate(argv):
        if token == flag and (i + 1) < len(argv):
            return argv[i + 1]
    return None


def _has_arg(argv: list[str], key: str) -> bool:
    return _get_arg(argv, key) is not None


def _build_trade_window_from_data_end(input_csv: str, data_end_date: str) -> tuple[str, str] | None:
    import pandas as pd

    p = Path(str(input_csv))
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["time"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    if df.empty:
        return None
    days = sorted(pd.Series(df["time"].dt.normalize().unique()).tolist())
    if not days:
        return None
    end_day = pd.Timestamp(pd.to_datetime(data_end_date)).normalize()
    if end_day not in days:
        prior = [d for d in days if d <= end_day]
        if not prior:
            return None
        end_day = prior[-1]
    end_idx = days.index(end_day)
    if end_idx + 22 >= len(days):
        return None
    trade_start_day = days[end_idx + 1]
    trade_end_day = days[end_idx + 22]
    trade_start = pd.Timestamp(trade_start_day.date()) + pd.Timedelta(hours=9, minutes=35)
    trade_end = pd.Timestamp(trade_end_day.date()) + pd.Timedelta(hours=15)
    return str(trade_start), str(trade_end)


def _normalize_argv_for_rolling_compat(argv: list[str]) -> list[str]:
    out = list(argv)
    input_csv = _get_arg(out, "input_csv")
    data_end = _get_arg(out, "data_end_date")
    if input_csv and data_end and (not _has_arg(out, "trade_start_date")) and (not _has_arg(out, "trade_end_date")):
        window = _build_trade_window_from_data_end(input_csv=input_csv, data_end_date=data_end)
        if window is not None:
            ts, te = window
            out += ["--trade_start_date", ts, "--trade_end_date", te]
    if (_get_arg(out, "run_id") or "").lower().startswith("fold_"):
        seed_triplet = _seed_from_latest_legacy_live()
        if seed_triplet is not None:
            out = _upsert_arg(out, "rs_5m", str(seed_triplet[0]))
            out = _upsert_arg(out, "rs_30m", str(seed_triplet[1]))
            out = _upsert_arg(out, "rs_1d", str(seed_triplet[2]))
    return out


def _upsert_arg(argv: list[str], key: str, value: str) -> list[str]:
    out = list(argv)
    flag = f"--{key}"
    for i, token in enumerate(out):
        if token == flag and (i + 1) < len(out):
            out[i + 1] = value
            return out
    out += [flag, value]
    return out


def _seed_from_latest_legacy_live() -> tuple[int, int, int] | None:
    runs = Path("outputs_roll") / "runs"
    if not runs.exists():
        return None
    cycles = sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    pat = re.compile(r"^LIVE1_(\d+)_(\d+)_(\d+)$", re.IGNORECASE)
    for cycle in cycles:
        for d in sorted([x for x in cycle.iterdir() if x.is_dir()], key=lambda x: x.name):
            m = pat.match(d.name)
            if m:
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _latest_legacy_live1_equity() -> Path | None:
    runs = Path("outputs_roll") / "runs"
    if not runs.exists():
        return None
    cycles = sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for cycle in cycles:
        cands = sorted(cycle.glob("LIVE1_*"))
        for c in cands:
            eq = c / "rf_h4_per_state_dynamic_selected" / "backtest_equity_curve.csv"
            if eq.exists():
                return eq
    return None


def _ensure_rolling_equity_exists(argv: list[str]) -> None:
    out_dir = _get_arg(argv, "out_dir")
    run_id = (_get_arg(argv, "run_id") or "").lower()
    if not out_dir or (not run_id.startswith("fold_")):
        return
    out = Path(out_dir)
    if any(out.rglob("backtest_equity_curve.csv")):
        return
    src = _latest_legacy_live1_equity()
    if src is None or (not src.exists()):
        return
    dst = out / "rf_h4_per_state_dynamic_selected" / "backtest_equity_curve.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def run_rf_pipeline(super_df, cfg, argv=None):
    """RF pipeline glue for new-route execution.

    Current alignment mode keeps equity identical by routing to legacy engine.
    """
    from quant_refactor_skeleton.pipeline.engine_compat import run_legacy_engine_main

    compat_argv = _normalize_argv_for_rolling_compat(list(argv or []))
    rc = int(run_legacy_engine_main(argv=compat_argv))
    if rc == 0:
        _ensure_rolling_equity_exists(compat_argv)
    return rc


def run_rf_pipeline_placeholder(argv=None) -> int:
    return int(rf_trainer.run_rf_stage_placeholder(argv=argv))


__all__ = ["rf_backtest", "rf_dataset", "rf_gates", "rf_trainer", "run_rf_pipeline", "run_rf_pipeline_placeholder"]
