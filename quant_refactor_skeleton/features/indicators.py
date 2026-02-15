import numpy as np
import pandas as pd


def safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def rolling_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win, min_periods=win).std()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    tr_s = pd.Series(tr, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100.0 * (plus_dm_s / (tr_s + 1e-12))
    minus_di = 100.0 * (minus_dm_s / (tr_s + 1e-12))
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx_val = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx_val
