"""R1 — assemble the daily modelling dataset from the teammate raw_data base.

Reads each feed read-only, builds the 19 features (7 from the target OHLCV bar,
12 not), merges on a daily date index, and forms the regression target
`y_logret_fwd1` = log(close_{t+1}/close_t). Features at t use only data <= t; the
target is strictly future (t -> t+1), so X_t aligns with y_{t+1}. Weekly COT is
merged as-of backward (last report <= t) — no look-ahead.

The common window is bounded by the on-chain start (~2022-06) and the DVOL end
(~2025-05). Output: data/processed/btc_daily_dataset.parquet (+ .csv).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import config as C


def _to_date_index(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    return df.set_index(col).sort_index()


# ----------------------------------------------------------------------------
# Individual feeds
# ----------------------------------------------------------------------------
def load_price_and_ta() -> pd.DataFrame:
    """Close + TA features from the pre-built TA_plus file (target OHLCV bar)."""
    d = pd.read_csv(C.F_TA_PLUS)
    d = _to_date_index(d, "date")
    d = d[~d.index.duplicated(keep="last")]
    close = d["close"].astype(float)
    out = pd.DataFrame(index=d.index)
    out["close"] = close
    out["ret_1d"] = np.log(close / close.shift(1))
    out["ret_5d"] = np.log(close / close.shift(5))
    out["ret_20d"] = np.log(close / close.shift(20))
    out["rsi_14"] = d["RSI_14"].astype(float)
    out["macd_hist"] = d["MACD_hist"].astype(float)
    out["natr_14"] = d["NATR_14"].astype(float)
    lower, upper = d["BBANDS_lower_14"].astype(float), d["BBANDS_upper_14"].astype(float)
    out["bb_position"] = 2 * (close - lower) / (upper - lower) - 1
    return out


def load_onchain() -> pd.DataFrame:
    """5 daily on-chain metrics (NOT from the target OHLCV bar)."""
    files = {"mvrv_zscore": "mvrv_zscore.csv", "nupl": "nupl.csv", "nvt": "nvt.csv",
             "puell": "puell_multiple.csv", "sopr": "sopr.csv"}
    frames = []
    for name, fn in files.items():
        d = pd.read_csv(C.D_ONCHAIN / fn)
        d = _to_date_index(d, "date")
        col = [c for c in d.columns if c != "date"][0]
        frames.append(d[[col]].rename(columns={col: name}))
    return pd.concat(frames, axis=1)


def load_ntv() -> pd.DataFrame:
    """Net taker volume imbalance in [-1, 1] from the 1d NTV feed."""
    d = pd.read_csv(C.F_NTV)
    d = _to_date_index(d, "date")
    vol = d["volume"].replace(0, np.nan)
    ntv = (2 * d["taker_buy_volume"] - d["volume"]) / vol
    return pd.DataFrame({"ntv_ratio": ntv})


def load_funding() -> pd.DataFrame:
    """Daily mean perp funding rate from the 8h feed."""
    d = pd.read_csv(C.F_FUNDING, usecols=["datetime", "funding_rate"])
    d = _to_date_index(d, "datetime")
    daily = d["funding_rate"].astype(float).resample("1D").mean()
    return pd.DataFrame({"funding_rate": daily})


def load_dvol() -> pd.DataFrame:
    """Deribit implied volatility (DVOL close), daily."""
    d = pd.read_csv(C.F_DVOL, usecols=["date", "close"])
    d = _to_date_index(d, "date")
    return pd.DataFrame({"dvol": d["close"].astype(float)})


def load_cot_weekly() -> pd.DataFrame:
    """Weekly COT leveraged-money net position fraction (as-of merged later)."""
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(C.D_COT / "*.csv")))]
    d = pd.concat(frames, ignore_index=True)
    d = _to_date_index(d, "date")
    d = d[~d.index.duplicated(keep="last")]
    lo = d["Lev_Money_Positions_Long_All"].astype(float)
    sh = d["Lev_Money_Positions_Short_All"].astype(float)
    net = (lo - sh) / (lo + sh).replace(0, np.nan)
    return pd.DataFrame({"cot_net_frac": net}).sort_index()


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------
def assemble(save: bool = True) -> pd.DataFrame:
    """Build the full daily feature+target dataset."""
    price = load_price_and_ta()
    onchain = load_onchain()
    ntv = load_ntv()
    funding = load_funding()
    dvol = load_dvol()
    cot = load_cot_weekly()

    # daily-frequency block: inner-join on the intersection of daily feeds
    daily = price.join([onchain, ntv, funding, dvol], how="outer")

    # COT is weekly -> as-of backward on the daily grid (last report <= t)
    daily = daily.sort_index()
    daily.index.name = "date"
    cot.index.name = "date"
    cot_asof = pd.merge_asof(
        daily.reset_index()[["date"]], cot.reset_index(),
        on="date", direction="backward",
    ).set_index("date")
    daily["cot_net_frac"] = cot_asof["cot_net_frac"].values

    # calendar features
    daily["month_sin"] = np.sin(2 * np.pi * daily.index.month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily.index.month / 12)
    daily["weekday"] = daily.index.weekday

    # target: next-day log return (strictly future)
    daily[C.TARGET] = np.log(daily["close"].shift(-1) / daily["close"])

    keep = [C.COL_CLOSE] + C.FEATURES + [C.TARGET]
    daily = daily[keep]

    # restrict to the common window then drop warm-up / edge NaNs
    daily = daily.dropna()
    daily.index.name = "date"

    if save:
        C.ensure_dirs()
        daily.to_parquet(C.DATA_PROCESSED / "btc_daily_dataset.parquet")
        daily.to_csv(C.DATA_PROCESSED / "btc_daily_dataset.csv")
    return daily


if __name__ == "__main__":
    ds = assemble(save=True)
    non_ohlcv = [f for f in C.FEATURES if not C.FEATURE_SPEC[f]["bar"]]
    print(f"dataset: {ds.shape[0]} days, {len(C.FEATURES)} features "
          f"({len(non_ohlcv)} non-OHLCV)")
    print(f"window: {ds.index[0].date()} -> {ds.index[-1].date()}")
    print(f"target {C.TARGET}: mean={ds[C.TARGET].mean():.5f} std={ds[C.TARGET].std():.5f}")
    print("missing per column:", int(ds.isna().sum().sum()))
