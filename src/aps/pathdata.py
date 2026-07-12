"""1-minute path data for barrier-aligned backtesting.

The modelling dataset stores forward stats measured from the hourly close. For an
*executable* backtest we enter at the next-minute open, so we recompute first-touch
directly from the 1-minute bars. This module builds a compact per-decision-hour
table (entry open, first up/down touch times, time-stop exit price) once per
threshold and caches it, so the expensive 1-minute scan never repeats.

Timestamp convention (matches scripts/build_dataset.py):
  * 1-minute bars are indexed by open_time; bar at minute m covers [m, m+59s].
  * A decision hour t owns the 1-minute bars with open_time in [t, t+1h).
  * Entry = open of the first bar of that window (open_time == t) — the price you
    can actually trade at once the hourly signal is known.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pcsv

from . import config as C

KLINES_1M_CSV = (
    C.ROOT / "raw_data" / "03_unzipped_feeds" / "klines" / "btc" / "binance"
    / "spot-BTCUSDT-1m" / "spot-BTCUSDT-1m.csv"
)


def load_1m() -> pd.DataFrame:
    """Load the merged 1-minute klines, indexed by open_time (deduped, sorted)."""
    cols = ["open_time", "open", "high", "low", "close"]
    tbl = pcsv.read_csv(
        str(KLINES_1M_CSV),
        convert_options=pcsv.ConvertOptions(include_columns=cols))
    df = tbl.to_pandas()
    df = df.drop_duplicates(subset="open_time", keep="last")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("ts").sort_index()
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        df = df[~bad]
    return df[["open", "high", "low", "close"]]


def _path_table_path(threshold: float) -> Path:
    return C.OUT_BACKTEST / f"path_table_{threshold:.2f}.parquet"


def build_path_table(threshold: float, decision_index: pd.DatetimeIndex,
                     m1: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-decision-hour first-touch table for a next-minute-open entry.

    Columns (indexed by decision hour t):
      entry_open      : open price at t (executable entry)
      first_up_time   : first minute high >= entry*(1+thr), else NaT
      first_dn_time   : first minute low  <= entry*(1-thr), else NaT
      time_exit_close : close of the last minute in [t, t+1h) (time-stop price)
    """
    if m1 is None:
        m1 = load_1m()
    dec = pd.DatetimeIndex(decision_index)

    m = m1.loc[(m1.index >= dec.min()) & (m1.index < dec.max() + pd.Timedelta(hours=1))].copy()
    m["t"] = m.index.floor("h")
    keep = m["t"].isin(set(dec))
    m = m[keep]

    entry = m.groupby("t")["open"].first()
    m["entry"] = m["t"].map(entry)

    up_mask = m["high"].to_numpy() >= (m["entry"].to_numpy() * (1 + threshold))
    dn_mask = m["low"].to_numpy() <= (m["entry"].to_numpy() * (1 - threshold))
    ts_vals = m.index.to_numpy()
    m["ts_up"] = np.where(up_mask, ts_vals, np.datetime64("NaT"))
    m["ts_dn"] = np.where(dn_mask, ts_vals, np.datetime64("NaT"))

    g = m.groupby("t").agg(
        entry_open=("entry", "first"),
        first_up_time=("ts_up", "min"),
        first_dn_time=("ts_dn", "min"),
        time_exit_close=("close", "last"),
    )
    return g.reindex(dec)


def get_path_table(threshold: float, decision_index: pd.DatetimeIndex,
                   rebuild: bool = False) -> pd.DataFrame:
    """Load the cached path table, building (and caching) it on first use."""
    path = _path_table_path(threshold)
    if path.exists() and not rebuild:
        tbl = pd.read_parquet(path)
        # ensure it covers the requested index
        if pd.DatetimeIndex(decision_index).isin(tbl.index).all():
            return tbl.reindex(decision_index)
    C.ensure_dirs()
    tbl = build_path_table(threshold, decision_index)
    tbl.to_parquet(path)
    return tbl
