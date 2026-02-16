from typing import Optional

import numpy as np
import pandas as pd


def pick_fit_mask_by_date(
    idx: pd.DatetimeIndex,
    exclude_last_day: bool,
    train_start_date: Optional[str] = None,
    train_end_date: Optional[str] = None,
) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=bool)
    if train_end_date is not None:
        end_dt = pd.to_datetime(train_end_date)
        start_dt = pd.to_datetime(train_start_date) if train_start_date else idx.min()
        mask = (idx >= start_dt) & (idx <= end_dt)
        if mask.sum() == 0:
            raise ValueError("鎸囧畾鏃ユ湡鑼冨洿瀵艰嚧鎷熷悎鏁版嵁涓虹┖銆傝妫€鏌ユ棩鏈熴€?")
        return mask.astype(bool)
    else:
        if not exclude_last_day:
            return np.ones(len(idx), dtype=bool)
        last_date = idx.max().normalize()
        return (idx < last_date).astype(bool)


def train_split_placeholder(argv=None) -> int:
    if argv and any(a in ("-h", "--help") for a in argv):
        print("usage: qrs-new-train-split [--help]")
        print("placeholder: train-split stage")
        return 0
    return 0
