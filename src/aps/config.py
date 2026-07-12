"""Global configuration — single source of truth for paths, features, seed, splits.

Every pipeline/module imports from here. Do not hard-code paths, feature names,
thresholds, or the split ratio anywhere else.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 1052  # used for every stochastic component (models, permutations, bootstrap)

# ----------------------------------------------------------------------------
# Paths (all relative to the repository root)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUT_AUDIT = OUTPUTS / "audit"
OUT_EDA = OUTPUTS / "eda"
OUT_MODELS = OUTPUTS / "models"
OUT_BACKTEST = OUTPUTS / "backtest"
OUT_STATS = OUTPUTS / "stats"
OUT_FIGURES = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"

_ALL_OUT_DIRS = [
    OUT_AUDIT, OUT_EDA, OUT_MODELS, OUT_BACKTEST,
    OUT_STATS, OUT_FIGURES, OUT_TABLES,
]


def ensure_dirs() -> None:
    """Create every output directory if missing. Call at the top of a pipeline."""
    for d in _ALL_OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Datasets — the frozen data pipeline emits one file per barrier threshold.
# ----------------------------------------------------------------------------
THRESHOLDS = (0.03, 0.02, 0.01)   # (main task, mid, control)
MAIN_THRESHOLD = 0.03             # primary barrier for the main experiment


def dataset_path(threshold: float = MAIN_THRESHOLD, ext: str = "parquet") -> Path:
    """Path to the modelling dataset for a given barrier threshold."""
    return DATA_PROCESSED / f"btc_1h_dataset_path_{threshold:.2f}.{ext}"


# ----------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------
# The 10 model input features, grouped by information dimension. The grouping
# drives the ablation study (Stage 18 of the plan).
FEATURE_GROUPS: dict[str, list[str]] = {
    "price_trend": ["ret_1h", "ret_24h", "close_vs_sma24"],
    "volatility": ["rvol_24h", "range_24h"],
    "volume": ["vol_ratio_24h"],
    "sentiment": ["rsi_14", "taker_buy_frac_1h"],
    "leverage": ["oi_chg_4h", "funding_rate"],
}
FEATURES: list[str] = [f for group in FEATURE_GROUPS.values() for f in group]

# Non-feature columns kept in the dataset.
COL_CLOSE = "close"                       # entry/exit reference price (backtest only)
COL_LABEL = "label"                       # target: {-1, 0, +1}
FORWARD_COLS = ["ret_fwd_1h", "max_up_1h", "max_dn_1h"]  # raw forward stats (relabelling)

# Human-readable feature descriptions (data dictionary / slides).
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret_1h": "Past 1h return",
    "ret_24h": "Past 24h return",
    "close_vs_sma24": "Close deviation from 24h SMA (%)",
    "rsi_14": "RSI(14) on hourly bars",
    "rvol_24h": "Realized vol: std of last 24 hourly returns",
    "range_24h": "(24h high - 24h low) / close",
    "vol_ratio_24h": "Last-1h volume / 24h average volume",
    "taker_buy_frac_1h": "Taker buy volume fraction over last 1h",
    "oi_chg_4h": "Futures open-interest 4h change rate",
    "funding_rate": "Latest funding rate (8h)",
}

# ----------------------------------------------------------------------------
# Chronological split — fractions of the (time-ordered) sample.
# ----------------------------------------------------------------------------
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15   # validation is (0.70, 0.85]; test is the final 0.15
# test fraction is implied = 1 - SPLIT_TRAIN - SPLIT_VAL

# Class labels in canonical order.
CLASSES = [-1, 0, 1]
CLASS_NAMES = {-1: "down", 0: "flat", 1: "up"}

# ----------------------------------------------------------------------------
# Trading / backtest configuration
# ----------------------------------------------------------------------------
# Cost per side = fee + slippage, as a fraction. Instrument: Binance USDT-M BTC
# perpetual futures (shorting required). Round-trip = 2 x per-side.
COST_SCENARIOS = {"zero": 0.0000, "base": 0.0005, "high": 0.0010}
BASE_COST = COST_SCENARIOS["base"]

# Confidence-threshold grid for the trading signal (calibrated on validation).
TAU_GRID = (0.40, 0.50, 0.60, 0.70)

# Entry convention: next-minute open after the hourly signal (see aps.pathdata).
# The model selected in Stage 4, recorded here after selection.
SELECTED_MODEL = "logistic"
