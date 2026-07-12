"""Global configuration — single source of truth (v2, daily regression).

Project-2 requirements: daily BTC, regression target (next-day log return),
>=15 features with >=half NOT based on the target OHLCV bar, models limited to
Scikit-Learn / Keras-TensorFlow / LightGBM (no PyTorch, no XGBoost). The data
base is the teammate-provided raw_data/ export and is not changed.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 1052

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_data"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUT_AUDIT = OUTPUTS / "audit"
OUT_EDA = OUTPUTS / "eda"
OUT_MODELS = OUTPUTS / "models"
OUT_BACKTEST = OUTPUTS / "backtest"
OUT_STATS = OUTPUTS / "stats"
OUT_FIGURES = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"
_ALL_OUT = [OUT_AUDIT, OUT_EDA, OUT_MODELS, OUT_BACKTEST, OUT_STATS,
            OUT_FIGURES, OUT_TABLES]


def ensure_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for d in _ALL_OUT:
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Raw-data feed files (teammate data base — read only, never modified)
# ----------------------------------------------------------------------------
F_OHLCV_DAILY = RAW / "06_source_bundles" / "btc_ohlcv.csv"
F_TA_PLUS = RAW / "06_source_bundles" / "CryptoDataBundle1_btc" / "BTCUSDT_OHLCV_TA_plus.csv"
D_ONCHAIN = RAW / "02_raw_feeds" / "onchain_metrics" / "btc" / "bgeometrics" / "spot-BTC-1d"
F_NTV = RAW / "02_raw_feeds" / "net_taker_volume" / "btc" / "binance" / "um-BTCUSDT-1d" / "um-BTCUSDT-1d.csv"
F_FUNDING = RAW / "02_raw_feeds" / "funding_rate" / "btc" / "binance" / "um-BTCUSDT-8h" / "um-BTCUSDT-8h.csv"
F_FUT_METRICS = RAW / "02_raw_feeds" / "futures_metrics" / "btc" / "binance" / "um-BTCUSDT-5m" / "api" / "um-BTCUSDT-5m.csv"
F_DVOL = RAW / "02_raw_feeds" / "dvol" / "btc" / "deribit" / "spot-BTC-1d" / "BTC_Volatility_DVOL_plus.csv"
D_COT = RAW / "02_raw_feeds" / "cot" / "btc" / "cftc" / "futures-BITCOIN-1w"

# ----------------------------------------------------------------------------
# Target
# ----------------------------------------------------------------------------
# Regression target: next-day log return. Features at t predict return over
# (t, t+1]; the target is shifted so X_t aligns with y_{t+1}.
TARGET = "y_logret_fwd1"
COL_CLOSE = "close"      # kept for backtesting, not a feature

# ----------------------------------------------------------------------------
# Features (>=15, majority NOT from target OHLCV). Each maps to a scaling class
# handled in aps.scaling. `bar` marks whether it derives from the target's OHLCV.
# ----------------------------------------------------------------------------
# scaling classes: 'return' | 'bounded' | 'pricescale' | 'normalized' |
#                  'positive_log' | 'signed' | 'zscore' | 'ratio'
FEATURE_SPEC: dict[str, dict] = {
    # ---- based on the target's OHLCV bar (kept to <= half) ----
    "ret_1d":       {"bar": True,  "scale": "return",     "group": "price_trend"},
    "ret_5d":       {"bar": True,  "scale": "return",     "group": "price_trend"},
    "ret_20d":      {"bar": True,  "scale": "return",     "group": "price_trend"},
    "rsi_14":       {"bar": True,  "scale": "bounded",    "group": "momentum"},
    "macd_hist":    {"bar": True,  "scale": "pricescale", "group": "momentum"},
    "natr_14":      {"bar": True,  "scale": "normalized", "group": "volatility"},
    "bb_position":  {"bar": True,  "scale": "bounded",    "group": "volatility"},
    # ---- NOT based on the target's OHLCV bar (majority) ----
    "mvrv_zscore":  {"bar": False, "scale": "zscore",       "group": "onchain"},
    "nupl":         {"bar": False, "scale": "bounded",      "group": "onchain"},
    "nvt":          {"bar": False, "scale": "positive_log", "group": "onchain"},
    "puell":        {"bar": False, "scale": "positive_log", "group": "onchain"},
    "sopr":         {"bar": False, "scale": "signed",       "group": "onchain"},
    "ntv_ratio":    {"bar": False, "scale": "signed",       "group": "flow"},
    "funding_rate": {"bar": False, "scale": "signed",       "group": "leverage"},
    "cot_net_frac": {"bar": False, "scale": "signed",       "group": "positioning"},
    "dvol":         {"bar": False, "scale": "positive_log", "group": "options"},
    # ---- calendar (non-OHLCV, not scaled) ----
    "month_sin":    {"bar": False, "scale": "none", "group": "calendar"},
    "month_cos":    {"bar": False, "scale": "none", "group": "calendar"},
    "weekday":      {"bar": False, "scale": "none", "group": "calendar"},
}

FEATURES: list[str] = list(FEATURE_SPEC)
NON_OHLCV_FEATURES = [f for f, s in FEATURE_SPEC.items() if not s["bar"]]
OHLCV_FEATURES = [f for f, s in FEATURE_SPEC.items() if s["bar"]]
CALENDAR_FEATURES = [f for f, s in FEATURE_SPEC.items() if s["group"] == "calendar"]

FEATURE_DESCRIPTIONS = {
    "ret_1d": "1-day log return", "ret_5d": "5-day log return",
    "ret_20d": "20-day log return", "rsi_14": "RSI(14)",
    "macd_hist": "MACD histogram (asinh-normalized by close)",
    "natr_14": "Normalized ATR(14)", "bb_position": "Position within Bollinger Bands",
    "mvrv_zscore": "On-chain MVRV Z-score", "nupl": "On-chain net unrealized P/L",
    "nvt": "On-chain NVT ratio", "puell": "On-chain Puell multiple",
    "sopr": "On-chain SOPR", "ntv_ratio": "Net taker volume imbalance",
    "funding_rate": "Perp funding rate (daily mean)",
    "cot_net_frac": "COT leveraged-money net position fraction (weekly, ffilled)",
    "dvol": "Deribit implied volatility (DVOL)",
    "month_sin": "Calendar month (sin)", "month_cos": "Calendar month (cos)",
    "weekday": "Day of week (0=Mon)",
}

# ----------------------------------------------------------------------------
# Split (chronological, no shuffle)
# ----------------------------------------------------------------------------
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15   # test = remaining 0.15

# ----------------------------------------------------------------------------
# Trading (quantile strategy) & costs
# ----------------------------------------------------------------------------
N_QUANTILES = 5                 # trade the extreme quantiles
LONG_QUANTILE = 5               # top predicted return -> long
SHORT_QUANTILE = 1              # bottom predicted return -> short
COST_SCENARIOS = {"zero": 0.0000, "base": 0.0005, "high": 0.0010}  # per side
BASE_COST = COST_SCENARIOS["base"]
TRADING_DAYS = 365              # crypto trades every day (annualization)

# Rolling-scale windows (for the neural pipeline; per assignment guidance)
ROLL_MEAN_WINDOW = 10
ROLL_STD_WINDOW = 90
