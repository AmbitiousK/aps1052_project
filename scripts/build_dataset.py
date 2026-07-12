"""APS1052 project — build the model-ready hourly BTC dataset.

Pipeline (all timestamps UTC):
  1. Clean Binance spot 1m klines  -> hourly bars (index = bar CLOSE time,
     so at timestamp t every value with index <= t is fully known).
  2. Compute 8 plain, explainable price/trend/volatility/volume/flow
     features from the hourly bars.
  3. Clean the futures open-interest 5m archives (daily zips) and the 8h
     funding-rate feed; as-of merge them onto the hourly grid (backward,
     with a staleness tolerance) -> 2 derivatives features.
  4. Build the 3-class label for the NEXT hour:
       mode "path"  (default): +1 if price first touches close_t*(1+th)
                    within (t, t+1h], -1 if it first touches close_t*(1-th),
                    0 otherwise (first-touch resolved on 1m bars; the rare
                    same-minute double-touch is set to 0).
       mode "close": sign of the 1h close-to-close return vs +-th.
     The raw forward quantities (ret_fwd_1h, max_up_1h, max_dn_1h) are kept
     in the output so labels can be rebuilt at any threshold without rerunning.
  5. Inner-join features + labels, drop warm-up/invalid rows, save
     parquet + csv, print class balance per year.

Usage:
    python scripts/build_dataset.py                 # default: path label, th=3%
    python scripts/build_dataset.py --threshold 0.02 --mode close
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.csv as pcsv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "raw_data")
OUT_DIR = os.path.join(ROOT, "data", "processed")

KLINES_1M = os.path.join(
    RAW, "03_unzipped_feeds", "klines", "btc", "binance",
    "spot-BTCUSDT-1m", "spot-BTCUSDT-1m.csv")
OI_ZIP_DIR = os.path.join(RAW, "02_raw_feeds", "futures_oi_history", "btc")
FUNDING_CSV = os.path.join(
    RAW, "02_raw_feeds", "funding_rate", "btc", "binance",
    "um-BTCUSDT-8h", "um-BTCUSDT-8h.csv")

# An hourly bar is only trusted if it contains at least this many 1m bars
# (exchange outages leave partially-filled hours whose volume/flow numbers
# are not comparable to normal hours).
MIN_MINUTES_PER_HOUR = 45
# As-of merge staleness tolerances.
OI_TOLERANCE = pd.Timedelta(hours=2)       # OI feed is 5m; >2h gap = outage
FUNDING_TOLERANCE = pd.Timedelta(hours=9)  # funding prints every 8h


# ---------------------------------------------------------------- cleaning

def load_klines_1m() -> pd.DataFrame:
    """Read the merged 1m klines CSV, dedupe, sort, index by bar open time."""
    cols = ["open_time", "open", "high", "low", "close",
            "volume", "taker_buy_volume"]
    tbl = pcsv.read_csv(
        KLINES_1M, convert_options=pcsv.ConvertOptions(include_columns=cols))
    df = tbl.to_pandas()
    df = df.drop_duplicates(subset="open_time", keep="last")
    df["ts"] = pd.to_datetime(df["open_time"], unit="ms").astype("datetime64[ns]")
    df = df.set_index("ts").sort_index()
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        print(f"  dropped {bad.sum()} 1m rows with non-positive prices")
        df = df[~bad]
    return df


def resample_hourly(m1: pd.DataFrame) -> pd.DataFrame:
    """1m -> 1h bars. Index is moved to the bar CLOSE time."""
    h = m1.resample("1h").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        n_min=("close", "count"))
    h.index = h.index + pd.Timedelta(hours=1)
    h["valid_bar"] = h["n_min"] >= MIN_MINUTES_PER_HOUR
    return h


def load_oi_5m() -> pd.DataFrame:
    """Read all daily futures-metrics zips (5m open interest)."""
    files = sorted(glob.glob(os.path.join(OI_ZIP_DIR, "*.zip")))
    print(f"  reading {len(files)} daily OI archives ...")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(
                f, usecols=["create_time", "sum_open_interest"]))
        except Exception as e:  # a few archives can be truncated/corrupt
            print(f"    skipped {os.path.basename(f)}: {e}")
    oi = pd.concat(frames, ignore_index=True)
    oi["ts"] = pd.to_datetime(oi["create_time"]).astype("datetime64[ns]")
    oi = (oi.dropna(subset=["ts"])
            .drop_duplicates(subset="ts", keep="last")
            .set_index("ts").sort_index())
    oi = oi[oi["sum_open_interest"] > 0]
    return oi


def load_funding() -> pd.DataFrame:
    f = pd.read_csv(FUNDING_CSV, usecols=["funding_time", "funding_rate"])
    f["ts"] = pd.to_datetime(f["funding_time"], unit="ms").astype("datetime64[ns]")
    f = (f.dropna(subset=["funding_rate"])
          .drop_duplicates(subset="ts", keep="last")
          .set_index("ts").sort_index())
    return f


# ---------------------------------------------------------------- features

def build_price_features(h: pd.DataFrame) -> pd.DataFrame:
    """8 features from spot hourly bars. Everything at index t uses bars <= t.

    Deliberately plain, textbook-style indicators so every input is easy to
    explain: returns (momentum), SMA deviation (trend), RSI (overbought/
    oversold), realized vol + range (volatility), volume ratio and taker buy
    fraction (activity / order-flow sentiment).
    """
    c = h["close"]
    logret = np.log(c / c.shift(1))
    hi24 = h["high"].rolling(24).max()
    lo24 = h["low"].rolling(24).min()

    # Wilder's RSI(14) on hourly closes
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.where(loss > 0))
    rsi = rsi.where(loss > 0, 100.0)  # no losses in window -> fully overbought

    f = pd.DataFrame(index=h.index)
    f["ret_1h"] = logret                                  # last-hour return
    f["ret_24h"] = np.log(c / c.shift(24))                # 1-day trend
    f["close_vs_sma24"] = c / c.rolling(24).mean() - 1    # deviation from MA
    f["rsi_14"] = rsi                                     # overbought/oversold
    f["rvol_24h"] = logret.rolling(24).std()              # realized volatility
    f["range_24h"] = (hi24 - lo24) / c                    # 24h high-low range
    f["vol_ratio_24h"] = (h["volume"]                     # this hour's volume
                          / h["volume"].rolling(24).mean())  # vs 24h average
    f["taker_buy_frac_1h"] = (h["taker_buy_volume"] / h["volume"]
                              ).where(h["volume"] > 0)
    # invalidate features computed on untrusted bars
    f[~h["valid_bar"]] = np.nan
    return f


def build_deriv_features(hourly_index: pd.DatetimeIndex,
                         oi: pd.DataFrame,
                         funding: pd.DataFrame) -> pd.DataFrame:
    """2 features from the derivatives feeds, as-of merged (no lookahead)."""
    grid = pd.DataFrame(index=hourly_index)

    oi_asof = pd.merge_asof(
        grid.reset_index().rename(columns={"index": "ts"}),
        oi[["sum_open_interest"]].reset_index(),
        on="ts", direction="backward", tolerance=OI_TOLERANCE,
    ).set_index("ts")["sum_open_interest"]

    fund_asof = pd.merge_asof(
        grid.reset_index().rename(columns={"index": "ts"}),
        funding[["funding_rate"]].reset_index(),
        on="ts", direction="backward", tolerance=FUNDING_TOLERANCE,
    ).set_index("ts")["funding_rate"]

    f = pd.DataFrame(index=hourly_index)
    f["oi_chg_4h"] = oi_asof.pct_change(4)   # leverage build-up / unwind
    f["funding_rate"] = fund_asof            # long/short crowding sentiment
    return f


# ---------------------------------------------------------------- labels

def build_labels(m1: pd.DataFrame, h: pd.DataFrame,
                 threshold: float, mode: str) -> pd.DataFrame:
    """Label at time t describes the NEXT hour (t, t+1h].

    Also returns the raw forward quantities so the threshold can be changed
    later without rebuilding: ret_fwd_1h, max_up_1h, max_dn_1h.
    """
    c0 = h["close"]  # entry price known at t (bar closing exactly at t)

    # forward close-to-close return
    ret_fwd = c0.shift(-1) / c0 - 1

    # forward intra-hour extremes, from 1m bars belonging to (t, t+1h]
    m = m1[["open_time", "high", "low"]].copy()
    m["t_decision"] = m.index.floor("h")  # 1m bar in [t, t+1h) -> decision t
    entry = m["t_decision"].map(c0)
    up_frac = m["high"] / entry - 1
    dn_frac = m["low"] / entry - 1
    g = pd.DataFrame({
        "t": m["t_decision"], "up": up_frac, "dn": dn_frac,
        "ts_up": m.index.where(up_frac >= threshold),
        "ts_dn": m.index.where(dn_frac <= -threshold),
    }).groupby("t").agg(max_up=("up", "max"), max_dn=("dn", "min"),
                        first_up=("ts_up", "min"), first_dn=("ts_dn", "min"))

    lab = pd.DataFrame(index=h.index)
    lab["ret_fwd_1h"] = ret_fwd
    lab["max_up_1h"] = g["max_up"].reindex(h.index)
    lab["max_dn_1h"] = g["max_dn"].reindex(h.index)

    if mode == "close":
        lab["label"] = np.select(
            [ret_fwd > threshold, ret_fwd < -threshold], [1, -1], 0)
    else:  # path / first-touch
        fu = g["first_up"].reindex(h.index)
        fd = g["first_dn"].reindex(h.index)
        up_first = fu.notna() & (fd.isna() | (fu < fd))
        dn_first = fd.notna() & (fu.isna() | (fd < fu))
        both_same_minute = fu.notna() & fd.notna() & (fu == fd)
        n_amb = int(both_same_minute.sum())
        if n_amb:
            print(f"  {n_amb} hours touched both barriers in the same "
                  f"minute -> labelled 0 (ambiguous)")
        lab["label"] = np.select([up_first, dn_first], [1, -1], 0)

    # a label is only trustworthy if the forward hour bar is complete
    fwd_valid = h["valid_bar"].shift(-1).fillna(False).astype(bool)
    lab.loc[~fwd_valid, ["label", "ret_fwd_1h", "max_up_1h", "max_dn_1h"]] = np.nan
    return lab


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="label threshold as a fraction (default 0.03 = 3%%)")
    ap.add_argument("--mode", choices=["path", "close"], default="path",
                    help="path = first-touch on intra-hour extremes (default); "
                         "close = close-to-close return")
    args = ap.parse_args()

    print("[1/5] cleaning 1m klines ...")
    m1 = load_klines_1m()
    h = resample_hourly(m1)
    n_bad = int((~h["valid_bar"]).sum())
    print(f"  {len(h)} hourly bars, {n_bad} incomplete "
          f"(<{MIN_MINUTES_PER_HOUR} minutes) -> excluded")

    print("[2/5] price/volume/flow features ...")
    px = build_price_features(h)

    print("[3/5] derivatives features (OI, funding) ...")
    oi = load_oi_5m()
    funding = load_funding()
    dv = build_deriv_features(h.index, oi, funding)

    print(f"[4/5] labels (mode={args.mode}, threshold={args.threshold:.3%}) ...")
    lab = build_labels(m1, h, args.threshold, args.mode)

    print("[5/5] merging and writing output ...")
    ds = pd.concat([px, dv, lab], axis=1)
    ds.insert(0, "close", h["close"])  # kept for backtesting, NOT a feature
    n_before = len(ds)
    ds = ds.dropna()
    ds["label"] = ds["label"].astype(int)
    print(f"  {n_before} -> {len(ds)} rows after dropping warm-up / stale / "
          f"invalid hours ({ds.index.min()} .. {ds.index.max()})")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{args.mode}_{args.threshold:g}"
    pq = os.path.join(OUT_DIR, f"btc_1h_dataset_{tag}.parquet")
    csv = os.path.join(OUT_DIR, f"btc_1h_dataset_{tag}.csv")
    ds.to_parquet(pq)
    ds.to_csv(csv)
    print(f"  wrote {pq}\n  wrote {csv}")

    print("\nClass balance by year (label: -1 / 0 / +1):")
    by_year = (ds.groupby([ds.index.year, "label"]).size()
                 .unstack(fill_value=0))
    by_year["total"] = by_year.sum(axis=1)
    for cls in (-1, 0, 1):
        if cls in by_year:
            by_year[f"{cls:+d}%"] = (by_year[cls] / by_year["total"] * 100
                                     ).round(2)
    print(by_year.to_string())

    n = len(ds)
    print(f"\nOverall: {n} samples | "
          + " | ".join(f"label {c:+d}: {(ds['label'] == c).sum()} "
                       f"({(ds['label'] == c).mean() * 100:.2f}%)"
                       for c in (-1, 0, 1)))
    print("\nSuggested chronological split (no shuffling — time series!):")
    i70, i85 = int(n * 0.70), int(n * 0.85)
    print(f"  train: {ds.index[0]}  ..  {ds.index[i70 - 1]}")
    print(f"  val:   {ds.index[i70]}  ..  {ds.index[i85 - 1]}")
    print(f"  test:  {ds.index[i85]}  ..  {ds.index[-1]}")


if __name__ == "__main__":
    main()
