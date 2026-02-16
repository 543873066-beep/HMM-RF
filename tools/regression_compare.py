"""Regression comparison tool for legacy vs refactor outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_curve(csv_path: str) -> pd.DataFrame:
    """Read equity curve CSV and normalize to columns: time, equity."""
    df = pd.read_csv(csv_path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    eq_col = None
    for c in ["equity", "eq", "equity_net", "equity_curve", "eq_net"]:
        if c in df.columns:
            eq_col = c
            break
    if eq_col is None:
        eq_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).sort_values(tcol)
    out = df[[tcol, eq_col]].rename(columns={tcol: "time", eq_col: "equity"}).copy()
    out["equity"] = pd.to_numeric(out["equity"], errors="coerce")
    out = out.dropna(subset=["equity"])
    return out


def compare_curves(old_curve: pd.DataFrame, new_curve: pd.DataFrame) -> pd.DataFrame:
    """Return point-wise aligned diff dataframe for review."""
    m = old_curve.merge(new_curve, on="time", how="outer", suffixes=("_old", "_new")).sort_values("time")
    m["equity_old"] = pd.to_numeric(m["equity_old"], errors="coerce")
    m["equity_new"] = pd.to_numeric(m["equity_new"], errors="coerce")
    m["abs_diff"] = (m["equity_new"] - m["equity_old"]).abs()
    denom = m["equity_old"].abs().replace(0, pd.NA)
    m["rel_diff"] = (m["abs_diff"] / denom).astype(float)
    return m


def _print_diagnostics(diff_df: pd.DataFrame, tol_abs: float, tol_rel: float, topn: int) -> bool:
    valid = diff_df.dropna(subset=["abs_diff"]).copy()
    rows = len(valid)
    max_abs = float(valid["abs_diff"].max()) if rows > 0 else float("nan")
    max_rel = float(valid["rel_diff"].max()) if rows > 0 else float("nan")

    abs_bad = valid["abs_diff"] > float(tol_abs)
    rel_bad = valid["rel_diff"] > float(tol_rel)
    bad = valid[abs_bad.fillna(False) | rel_bad.fillna(False)].copy()

    print(f"rows_compared={rows}")
    print(f"max_abs_diff={max_abs}")
    print(f"max_rel_diff={max_rel}")
    print(f"threshold_abs={tol_abs}")
    print(f"threshold_rel={tol_rel}")
    print(f"rows_over_threshold={len(bad)}")

    if len(bad) > 0:
        show = bad.sort_values("abs_diff", ascending=False).head(int(max(1, topn)))
        print("top_diff_rows:")
        cols = ["time", "equity_old", "equity_new", "abs_diff", "rel_diff"]
        print(show[cols].to_string(index=False))
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and refactor equity outputs")
    parser.add_argument("--old-equity", required=True, help="Legacy equity CSV path")
    parser.add_argument("--new-equity", required=True, help="Refactor equity CSV path")
    parser.add_argument("--out", default="artifacts/regression_diff.csv", help="Diff CSV output path")
    parser.add_argument("--tol-abs", type=float, default=1e-6, help="Absolute diff threshold")
    parser.add_argument("--tol-rel", type=float, default=1e-4, help="Relative diff threshold")
    parser.add_argument("--topn", type=int, default=10, help="Top-N rows to print when over threshold")
    args = parser.parse_args()

    old_path = Path(args.old_equity)
    new_path = Path(args.new_equity)
    out_path = Path(args.out)

    if not old_path.exists() or not new_path.exists():
        raise FileNotFoundError("Input equity CSV path does not exist")

    old_curve = read_curve(str(old_path))
    new_curve = read_curve(str(new_path))
    diff_df = compare_curves(old_curve, new_curve)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff_df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    ok = _print_diagnostics(diff_df, tol_abs=args.tol_abs, tol_rel=args.tol_rel, topn=args.topn)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
