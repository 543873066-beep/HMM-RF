from pathlib import Path

import pandas as pd


def compare_equity_curves(old_curve: pd.DataFrame, new_curve: pd.DataFrame) -> pd.DataFrame:
    """Produce point-wise diff for regression review."""
    # TODO: tie into tools/regression_compare.py checks
    raise NotImplementedError


def save_regression_report(report_df: pd.DataFrame, out_path: str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out, index=False)
    return out
