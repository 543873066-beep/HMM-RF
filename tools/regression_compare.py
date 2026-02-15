"""Regression comparison skeleton for legacy vs refactor outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_curve(csv_path: str) -> pd.DataFrame:
    """Read equity curve CSV from a run directory."""
    # TODO: port and consolidate logic from rolling_runner.py::read_equity_curve
    raise NotImplementedError


def compare_curves(old_curve: pd.DataFrame, new_curve: pd.DataFrame) -> pd.DataFrame:
    """Return point-wise diff dataframe for review."""
    # TODO: wire to quant_refactor.eval.regression.compare_equity_curves after implementation
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and refactor equity outputs")
    parser.add_argument("--old-equity", required=True, help="Legacy equity CSV path")
    parser.add_argument("--new-equity", required=True, help="Refactor equity CSV path")
    parser.add_argument("--out", default="artifacts/regression_diff.csv", help="Diff CSV output path")
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


if __name__ == "__main__":
    main()
