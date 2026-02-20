#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _run(cmd: Iterable[str], env: Optional[Dict[str, str]] = None) -> int:
    proc = subprocess.run(list(cmd), env=env)
    return proc.returncode


def _resolve_route(mode: str, route_arg: Optional[str]) -> Tuple[str, str]:
    if route_arg:
        return route_arg, "param"
    if mode == "rolling":
        env_val = os.getenv("QRS_ROLLING_ROUTE")
        if env_val:
            return env_val, "env"
    else:
        env_val = os.getenv("QRS_PIPELINE_ROUTE")
        if env_val:
            return env_val, "env"
    return "new", "default"


def _get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    v = val.strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _ensure_input_csv(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            "Suggestions:\n"
            "1) Put CSV under data/ (required columns: time/open/high/low/close/volume)\n"
            "2) Or pass --input_csv with a full path"
        )


def _new_route_args(input_csv: str, out_dir: str, disable_fallback: bool) -> list[str]:
    backfill = _get_bool_env("QRS_LEGACY_BACKFILL", default=False)
    bf = "1" if backfill else "0"
    df = "1" if disable_fallback else "0"
    return [
        "--",
        "--input_csv",
        input_csv,
        "--out_dir",
        out_dir,
        "--enable_legacy_backfill",
        bf,
        "--disable_legacy_equity_fallback_in_rolling",
        df,
    ]


def _best_equity_csv(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best = None
    best_score = -1
    best_rows = -1
    for p in root.rglob("*.csv"):
        score = 0
        name = p.name.lower()
        if "equity" in name:
            score += 4
        if "curve" in name:
            score += 2
        header = ""
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                header = f.readline().lower()
        except Exception:
            header = ""
        if any(k in header for k in ("equity", "eq", "nav")):
            score += 3
        if "time" in header:
            score += 1
        rows = 0
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                rows = max(0, sum(1 for _ in f) - 1)
        except Exception:
            rows = 0
        if (score > best_score) or (score == best_score and rows > best_rows):
            best = p
            best_score = score
            best_rows = rows
    return best


def _latest_compare_run(out_root: Path) -> Optional[Path]:
    if not out_root.exists():
        return None
    candidates = []
    for p in out_root.rglob("*"):
        if p.is_dir():
            if (p / "legacy").exists() and (p / "new").exists():
                candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.as_posix(), reverse=True)[0]


def _build_run_root(out_root: str, mode: str) -> Path:
    root = Path(out_root)
    ts = _timestamp()
    leaf = root.name
    if leaf and leaf.replace("_", "").isdigit() and len(leaf) == 15:
        return root / mode
    return root / ts / mode


def _timestamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_engine(route: str, route_source: str, input_csv: str, out_root: str, disable_fallback: bool) -> int:
    _ensure_input_csv(input_csv)
    run_root = _build_run_root(out_root, "engine")
    out_dir = run_root / route
    out_dir.mkdir(parents=True, exist_ok=True)
    _print(
        f"[QRS] route={route} legacy_backfill={'on' if _get_bool_env('QRS_LEGACY_BACKFILL') else 'off'} "
        f"DisableLegacyEquityFallback={'on' if disable_fallback else 'off'} route_source={route_source}"
    )
    if route == "legacy":
        rc = _run(
            [
                sys.executable,
                "msp_engine_ewma_exhaustion_opt_atr_momo.py",
                "--input_csv",
                input_csv,
                "--out_dir",
                str(out_dir),
            ]
        )
    else:
        env = os.environ.copy()
        env["QRS_PIPELINE_ROUTE"] = "new"
        env["QRS_LEGACY_BACKFILL"] = "1" if _get_bool_env("QRS_LEGACY_BACKFILL") else "0"
        rc = _run(
            [
                sys.executable,
                "scripts/run_engine_compat.py",
                "--route",
                "new",
            ]
            + _new_route_args(input_csv, str(out_dir), disable_fallback),
            env=env,
        )
    if rc != 0:
        return rc
    _print(f"[QRS] mode=engine route={route} out_dir={out_dir.resolve()}")
    report = out_dir / "run_report.json"
    if report.exists():
        _print(f"run_report={report.resolve()}")
    return 0


def run_compare(route: str, route_source: str, input_csv: str, out_root: str, disable_fallback: bool, tol_abs: float, tol_rel: float, topn: int, skip_stage_diff: bool) -> int:
    _ensure_input_csv(input_csv)
    run_root = _build_run_root(out_root, "compare")
    legacy_dir = run_root / "legacy"
    new_dir = run_root / "new"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    _print(
        f"[QRS] route={route} legacy_backfill={'on' if _get_bool_env('QRS_LEGACY_BACKFILL') else 'off'} "
        f"DisableLegacyEquityFallback={'on' if disable_fallback else 'off'} route_source={route_source}"
    )

    rc_legacy = _run(
        [
            sys.executable,
            "msp_engine_ewma_exhaustion_opt_atr_momo.py",
            "--input_csv",
            input_csv,
            "--out_dir",
            str(legacy_dir),
        ]
    )
    if rc_legacy != 0:
        _err(f"Legacy run failed with code {rc_legacy}")
        return rc_legacy

    env = os.environ.copy()
    env["QRS_PIPELINE_ROUTE"] = "new"
    env["QRS_LEGACY_BACKFILL"] = "1" if _get_bool_env("QRS_LEGACY_BACKFILL") else "0"
    rc_new = _run(
        [
            sys.executable,
            "scripts/run_engine_compat.py",
            "--route",
            "new",
        ]
        + _new_route_args(input_csv, str(new_dir), disable_fallback),
        env=env,
    )
    if rc_new != 0:
        _err(f"New run failed with code {rc_new}")
        return rc_new

    old_eq = _best_equity_csv(legacy_dir)
    new_eq = _best_equity_csv(new_dir)
    if old_eq is None or new_eq is None:
        _err(f"Failed to locate equity CSV. legacy='{old_eq}', new='{new_eq}'")
        return 2

    diff_csv = run_root / "regression_diff.csv"
    rc_cmp = _run(
        [
            sys.executable,
            "tools/regression_compare.py",
            "--old-equity",
            str(old_eq),
            "--new-equity",
            str(new_eq),
            "--tol-abs",
            str(tol_abs),
            "--tol-rel",
            str(tol_rel),
            "--topn",
            str(topn),
            "--out",
            str(diff_csv),
        ]
    )
    if rc_cmp == 0:
        _print("compare=PASS")
    else:
        _print("compare=FAIL")
    _print(f"[QRS] legacy_equity={old_eq}")
    _print(f"[QRS] new_equity={new_eq}")
    _print(f"[QRS] diff_csv={diff_csv}")
    if rc_cmp == 0 and not skip_stage_diff:
        _print("[QRS] compare passed, running stage-diff...")
        rc_stage = _run(
            [
                sys.executable,
                "tools/stage_diff.py",
                "--legacy-dir",
                str(legacy_dir),
                "--new-dir",
                str(new_dir),
                "--input-csv",
                input_csv,
            ]
        )
        if rc_stage != 0:
            return rc_stage
    return rc_cmp


def run_stage_diff(input_csv: str, out_root: str) -> int:
    _ensure_input_csv(input_csv)
    root = Path(out_root)
    latest = _latest_compare_run(root)
    if latest is None:
        _err(f"No compare run found under {out_root}. Run compare first.")
        return 2
    legacy_dir = latest / "legacy"
    new_dir = latest / "new"
    if not legacy_dir.exists() or not new_dir.exists():
        _err(f"Stage artifacts missing. legacy='{legacy_dir}', new='{new_dir}'")
        return 2
    return _run(
        [
            sys.executable,
            "tools/stage_diff.py",
            "--legacy-dir",
            str(legacy_dir),
            "--new-dir",
            str(new_dir),
            "--input-csv",
            input_csv,
        ]
    )


def run_rolling(route: str, route_source: str, input_csv: str, out_root: str, disable_fallback: bool) -> int:
    run_root = _build_run_root(out_root, "rolling")
    out_dir = run_root / route
    out_dir.mkdir(parents=True, exist_ok=True)
    _print(
        f"[QRS] route={route} legacy_backfill={'on' if _get_bool_env('QRS_LEGACY_BACKFILL') else 'off'} "
        f"DisableLegacyEquityFallback={'on' if disable_fallback else 'off'} route_source={route_source}"
    )
    if route == "legacy":
        return _run([sys.executable, "rolling_runner.py"])
    _ensure_input_csv(input_csv)
    env = os.environ.copy()
    env["QRS_ROLLING_ROUTE"] = "new"
    env["QRS_PIPELINE_ROUTE"] = "new"
    env["QRS_LEGACY_BACKFILL"] = "1" if _get_bool_env("QRS_LEGACY_BACKFILL") else "0"
    fold_dir = out_dir / "fold_001"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            sys.executable,
            "scripts/run_engine_compat.py",
            "--route",
            "new",
        ]
        + _new_route_args(input_csv, str(fold_dir), disable_fallback),
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform wrapper for QRS runs (engine/rolling/compare/stage-diff)")
    parser.add_argument("--mode", required=True, choices=["engine", "rolling", "compare", "stage-diff"])
    parser.add_argument("--route", choices=["new", "legacy"], default=None)
    parser.add_argument("--input_csv", default="data/sh000852_5m.csv")
    parser.add_argument("--out_root", default="outputs_rebuild/qrs_runs")
    parser.add_argument("--disable_legacy_fallback", action="store_true")
    parser.add_argument("--tol_abs", type=float, default=1e-10)
    parser.add_argument("--tol_rel", type=float, default=1e-10)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--skip_stage_diff", action="store_true")
    args = parser.parse_args()

    route, source = _resolve_route(args.mode, args.route)
    if args.mode == "engine":
        return run_engine(route, source, args.input_csv, args.out_root, args.disable_legacy_fallback)
    if args.mode == "compare":
        return run_compare(
            route,
            source,
            args.input_csv,
            args.out_root,
            args.disable_legacy_fallback,
            args.tol_abs,
            args.tol_rel,
            args.topn,
            args.skip_stage_diff,
        )
    if args.mode == "stage-diff":
        return run_stage_diff(args.input_csv, args.out_root)
    if args.mode == "rolling":
        return run_rolling(route, source, args.input_csv, args.out_root, args.disable_legacy_fallback)
    _err(f"Unknown mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
