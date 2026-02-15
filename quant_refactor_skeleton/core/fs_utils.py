"""Filesystem helpers migrated from legacy scripts."""

import os
from typing import Optional


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_equity_curve_csv_optional(run_dir: str) -> Optional[str]:
    for root, _, files in os.walk(run_dir):
        if "backtest_equity_curve.csv" in files:
            return os.path.join(root, "backtest_equity_curve.csv")
    return None
