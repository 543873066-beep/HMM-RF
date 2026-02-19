"""Fold-by-fold equity comparison for rolling outputs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from regression_compare import compare_curves, read_curve


FOLD_RE = re.compile(r"^fold_(\d+)$", re.IGNORECASE)
S_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
LIVE_RE = re.compile(r"^LIVE(\d+)_", re.IGNORECASE)


@dataclass
class FoldResult:
    fold: str
    legacy_csv: str
    new_csv: str
    rows_compared: int
    max_abs_diff: float
    max_rel_diff: float
    rows_over_threshold: int
    first_bad_time: str | None
    last_bad_time: str | None
    len_old: int
    len_new: int
    coverage: float
    fail_reason: str | None
    pass_fold: bool
    diff_csv: str


def _find_latest_run_root(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    children = [p for p in root.iterdir() if p.is_dir()]
    if not children:
        return root
    with_curves = [p for p in children if any(p.rglob("backtest_equity_curve.csv"))]
    if with_curves:
        return sorted(with_curves, key=lambda p: p.name, reverse=True)[0]
    stamped = sorted(children, key=lambda p: p.name, reverse=True)
    return stamped[0]


def _is_file_path(p: Path) -> bool:
    return p.exists() and p.is_file()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _resolve_run_root(root: Path, side: str) -> Path:
    if _is_file_path(root):
        return root
    run_root = _find_latest_run_root(root)
    side_root = run_root / side
    return side_root if side_root.exists() else run_root


def _discover_fold_curves(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    candidates = sorted(root.rglob("backtest_equity_curve.csv"))
    for i, csv in enumerate(candidates, start=1):
        parts = csv.parts
        fold = None
        for part in parts:
            m_fold = FOLD_RE.match(part)
            if m_fold:
                fold = f"fold_{int(m_fold.group(1)):03d}"
                break
            m_s = S_RE.match(part)
            if m_s:
                fold = f"fold_{int(m_s.group(1)) + 1:03d}"
                break
            m_live = LIVE_RE.match(part)
            if m_live:
                fold = f"fold_{int(m_live.group(1)):03d}"
                break
        if fold is None:
            fold = f"fold_{i:03d}"
        if fold not in out:
            out[fold] = csv
    return out


def _discover_with_fallback(run_root: Path, fallback_root: Path | None) -> Dict[str, Path]:
    found = _discover_fold_curves(run_root)
    if found:
        return found
    if fallback_root is None:
        return found
    return _discover_fold_curves(fallback_root)


def _compute_metrics(diff_df: pd.DataFrame, tol_abs: float, tol_rel: float) -> tuple[int, float, float, int]:
    valid = diff_df.dropna(subset=["abs_diff"]).copy()
    rows = int(len(valid))
    max_abs = float(valid["abs_diff"].max()) if rows > 0 else 0.0
    max_rel = float(valid["rel_diff"].max()) if rows > 0 else 0.0
    abs_bad = valid["abs_diff"] > float(tol_abs)
    rel_bad = valid["rel_diff"] > float(tol_rel)
    bad = valid[abs_bad.fillna(False) | rel_bad.fillna(False)]
    return rows, max_abs, max_rel, int(len(bad))


def _bad_rows(diff_df: pd.DataFrame, tol_abs: float, tol_rel: float) -> pd.DataFrame:
    valid = diff_df.dropna(subset=["abs_diff"]).copy()
    abs_bad = valid["abs_diff"] > float(tol_abs)
    rel_bad = valid["rel_diff"] > float(tol_rel)
    return valid[abs_bad.fillna(False) | rel_bad.fillna(False)].sort_values("time")


def _top_rows(diff_df: pd.DataFrame, topn: int) -> list[dict]:
    cols = ["time", "equity_old", "equity_new", "abs_diff", "rel_diff"]
    bad = diff_df.dropna(subset=["abs_diff"]).sort_values("abs_diff", ascending=False).head(int(max(1, topn)))
    rows: list[dict] = []
    for _, r in bad[cols].iterrows():
        rows.append(
            {
                "time": str(r["time"]),
                "equity_old": float(r["equity_old"]) if pd.notna(r["equity_old"]) else None,
                "equity_new": float(r["equity_new"]) if pd.notna(r["equity_new"]) else None,
                "abs_diff": float(r["abs_diff"]) if pd.notna(r["abs_diff"]) else None,
                "rel_diff": float(r["rel_diff"]) if pd.notna(r["rel_diff"]) else None,
            }
        )
    return rows


def _fail(reason: str) -> str:
    return reason


def run_compare(
    legacy_root: Path,
    new_root: Path,
    tol_abs: float,
    tol_rel: float,
    topn: int,
    out_json: Path | None = None,
) -> int:
    coverage_threshold = 0.95
    legacy_run = _resolve_run_root(legacy_root, "legacy")
    new_run = _resolve_run_root(new_root, "new")

    legacy_fallback = Path("outputs_roll") / "runs"
    legacy_curves = _discover_with_fallback(legacy_run, legacy_fallback if legacy_fallback.exists() else None)
    new_curves = _discover_with_fallback(new_run, None)

    common_folds = sorted(set(legacy_curves).intersection(new_curves))
    if not common_folds:
        print("rolling_compare: no common folds with backtest_equity_curve.csv found")
        return 2

    diff_dir = (out_json.parent if out_json else new_root) / "rolling_diff"
    diff_dir.mkdir(parents=True, exist_ok=True)

    results: list[FoldResult] = []
    overall_pass = True
    for fold in common_folds:
        legacy_csv = legacy_curves[fold]
        new_csv = new_curves[fold]
        reason = None
        if not legacy_csv.exists() or legacy_csv.stat().st_size == 0:
            reason = _fail("EQUITY_MISSING_OLD")
        if not new_csv.exists() or new_csv.stat().st_size == 0:
            reason = _fail("EQUITY_MISSING_NEW")
        if reason is None:
            if not _is_under(legacy_csv, legacy_run):
                reason = _fail("PATH_OUT_OF_ROOT")
            if not _is_under(new_csv, new_run):
                reason = _fail("PATH_OUT_OF_ROOT")

        if reason is None:
            old_curve = read_curve(str(legacy_csv))
            new_curve = read_curve(str(new_csv))
            len_old = int(len(old_curve))
            len_new = int(len(new_curve))
            if len_old == 0 or len_new == 0:
                reason = _fail("EQUITY_EMPTY")
            diff_df = compare_curves(old_curve, new_curve)
        else:
            old_curve = pd.DataFrame(columns=["time", "equity"])
            new_curve = pd.DataFrame(columns=["time", "equity"])
            len_old = 0
            len_new = 0
            diff_df = pd.DataFrame(columns=["time", "equity_old", "equity_new", "abs_diff", "rel_diff"])

        rows, max_abs, max_rel, rows_bad = _compute_metrics(diff_df, tol_abs=tol_abs, tol_rel=tol_rel)
        bad_df = _bad_rows(diff_df, tol_abs=tol_abs, tol_rel=tol_rel)
        first_bad_time = str(bad_df["time"].iloc[0]) if len(bad_df) > 0 else None
        last_bad_time = str(bad_df["time"].iloc[-1]) if len(bad_df) > 0 else None
        diff_csv = diff_dir / f"{fold}_diff.csv"
        diff_df.to_csv(diff_csv, index=False)
        min_len = max(0, min(len_old, len_new))
        coverage = float(rows) / float(min_len) if min_len > 0 else 0.0
        if reason is None and rows == 0:
            reason = _fail("ZERO_ROWS_COMPARED")
        if reason is None and coverage < coverage_threshold:
            reason = _fail("LOW_COVERAGE")
        fold_ok = (rows_bad == 0) and (rows > 0) and (reason is None)
        if not fold_ok:
            overall_pass = False
        results.append(
            FoldResult(
                fold=fold,
                legacy_csv=str(legacy_csv),
                new_csv=str(new_csv),
                rows_compared=rows,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                rows_over_threshold=rows_bad,
                first_bad_time=first_bad_time,
                last_bad_time=last_bad_time,
                len_old=len_old,
                len_new=len_new,
                coverage=coverage,
                fail_reason=reason,
                pass_fold=fold_ok,
                diff_csv=str(diff_csv),
            )
        )

    worst_abs = max((r.max_abs_diff for r in results), default=0.0)
    worst_rows = max((r.rows_over_threshold for r in results), default=0)
    summary = {
        "legacy_root": str(legacy_root),
        "new_root": str(new_root),
        "legacy_run_root": str(legacy_run),
        "new_run_root": str(new_run),
        "threshold_abs": float(tol_abs),
        "threshold_rel": float(tol_rel),
        "overall_pass": overall_pass,
        "max_abs_diff": worst_abs,
        "max_rows_over_threshold": worst_rows,
        "folds": [asdict(x) for x in results],
        "top_rows_by_fold": {
            r.fold: _top_rows(
                compare_curves(read_curve(r.legacy_csv), read_curve(r.new_csv)),
                topn=topn,
            )
            for r in results
            if not r.pass_fold
        },
    }

    summary_path = out_json if out_json else (new_root / "rolling_diff_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    by_fold_csv = summary_path.parent / "rolling_diff_by_fold.csv"
    by_fold_df = pd.DataFrame([asdict(x) for x in results])
    by_fold_df.to_csv(by_fold_csv, index=False)

    for r in results:
        status = "PASS" if r.pass_fold else "FAIL"
        print(f"[rolling-compare] {status} fold={r.fold}")
        print(f"[rolling-compare] old_path={Path(r.legacy_csv)}")
        print(f"[rolling-compare] new_path={Path(r.new_csv)}")
        print(
            f"[rolling-compare] len_old={r.len_old} len_new={r.len_new} "
            f"rows={r.rows_compared} coverage={r.coverage:.6f}"
        )
        print(
            f"[rolling-compare] max_abs={r.max_abs_diff} rows_over={r.rows_over_threshold} "
            f"first_bad={r.first_bad_time} last_bad={r.last_bad_time} reason={r.fail_reason}"
        )
        if not r.pass_fold:
            bad_df = _bad_rows(compare_curves(read_curve(r.legacy_csv), read_curve(r.new_csv)), tol_abs=tol_abs, tol_rel=tol_rel)
            cols = ["time", "equity_old", "equity_new", "abs_diff", "rel_diff"]
            top_df = bad_df.sort_values("abs_diff", ascending=False).head(int(max(1, topn)))
            print(f"[rolling-compare] top_bad_rows fold={r.fold}:")
            print(top_df[cols].to_string(index=False))
    print(f"[rolling-compare] by_fold_csv={by_fold_csv}")
    print(f"[rolling-compare] summary={summary_path}")
    print(f"[rolling-compare] overall={'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rolling fold equity CSVs between legacy and new routes")
    parser.add_argument("--legacy-root", required=True, help="Legacy rolling out root")
    parser.add_argument("--new-root", required=True, help="New rolling out root")
    parser.add_argument("--tol-abs", type=float, default=1e-10, help="Absolute threshold")
    parser.add_argument("--tol-rel", type=float, default=1e-10, help="Relative threshold")
    parser.add_argument("--topn", type=int, default=20, help="Top N rows for failed folds")
    parser.add_argument("--out", default="", help="Optional summary JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if str(args.out).strip() else None
    return run_compare(
        legacy_root=Path(args.legacy_root),
        new_root=Path(args.new_root),
        tol_abs=float(args.tol_abs),
        tol_rel=float(args.tol_rel),
        topn=int(args.topn),
        out_json=out_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
