"""Feature scaling (Project-2 guidance).

Three regimes, applied in the right place:

  1. Meaning-preserving transforms (`meaning_preserving_transform`) — applied ONCE
     to the raw features so price-scale / positive-multiplicative indicators do not
     get distorted by later scaling. This is the independent-row idea for TA-lib
     features (RSI -> (x-50)/50, MACD -> asinh(MACD/close), positive levels -> log).
  2. Global StandardScaler — used INSIDE the Scikit-Learn grid pipeline for the
     tabular ML models (fit on train folds only).
  3. Rolling scaling (`asymmetric_rolling_scale`, verbatim from the assignment) —
     used for the neural pipeline, applied to the whole feature matrix before the
     chronological split. The target is already shifted (t -> t+1) in datasets.py.

Calendar features (month_sin/cos, weekday) are never scaled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------
# 1. Meaning-preserving per-feature transforms
# ----------------------------------------------------------------------------
def meaning_preserving_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix with independent-row / level transforms applied.

    `df` must contain the raw features plus `close` (for MACD normalization).
    Output columns == C.FEATURES, meaning preserved, roughly stationary.
    """
    close = df[C.COL_CLOSE].astype(float)
    out = pd.DataFrame(index=df.index)
    for f in C.FEATURES:
        x = df[f].astype(float) if f in df else None
        if f == "rsi_14":
            out[f] = (x - 50.0) / 50.0                     # bounded -> ~[-1,1]
        elif f == "macd_hist":
            out[f] = np.arcsinh(x / close)                 # price-scale -> asinh
        elif f in ("nvt", "puell", "dvol"):
            out[f] = np.log(x)                             # positive multiplicative
        elif f == "sopr":
            out[f] = x - 1.0                               # center around 0
        else:
            out[f] = x                                     # already stationary/bounded
    return out


# ----------------------------------------------------------------------------
# 2. Rolling scaling (verbatim from the assignment) + matrix helper
# ----------------------------------------------------------------------------
def asymmetric_rolling_scale(series: pd.Series, mean_window: int = C.ROLL_MEAN_WINDOW,
                             std_window: int = C.ROLL_STD_WINDOW, ddof: int = 0,
                             eps: float = 1e-12) -> pd.Series:
    """scaled[t] = (x[t] - short_rolling_mean[t]) / long_rolling_std[t].

    Deviation of the current value from its recent mean, in units of longer-term
    variability. Includes the current observation in both rolling windows, so the
    target must be shifted (done in datasets.py).
    """
    if mean_window <= 0 or std_window <= 0:
        raise ValueError("Window lengths must be positive integers.")
    series = series.astype(float)
    rolling_mean = series.rolling(window=mean_window, min_periods=mean_window).mean()
    rolling_std = series.rolling(window=std_window, min_periods=std_window).std(ddof=ddof)
    rolling_std = rolling_std.where(rolling_std > eps, np.nan)
    return (series - rolling_mean) / rolling_std


def rolling_scale_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling scaling to every non-calendar feature column."""
    out = X.copy()
    for col in X.columns:
        if col in C.CALENDAR_FEATURES:
            continue
        out[col] = asymmetric_rolling_scale(X[col])
    return out
