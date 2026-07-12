"""Quantile trading backtest (Project-2).

Strategy (described in English): each day, rank the model's predicted next-day
return against quantile thresholds learned on the training predictions. Go LONG
when the prediction falls in the top quantile (largest predicted up-move), go
SHORT in the bottom quantile, otherwise stay flat. Hold one day; the realized
return is the next-day log return. Costs are charged on turnover (per-side fee +
slippage). Focusing on the extreme quantiles targets the largest predicted moves.

Quantile thresholds are fit on TRAIN predictions and applied to validation/test,
so there is no look-ahead in the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def quantile_thresholds(pred_train, n_q: int = C.N_QUANTILES) -> np.ndarray:
    """Interior quantile edges of the training prediction distribution."""
    qs = np.linspace(0, 1, n_q + 1)[1:-1]
    return np.quantile(np.asarray(pred_train, dtype=float), qs)


def quantile_signal(pred_eval, edges: np.ndarray,
                    long_q: int = C.LONG_QUANTILE,
                    short_q: int = C.SHORT_QUANTILE,
                    n_q: int = C.N_QUANTILES) -> np.ndarray:
    """Map predictions to positions {-1,0,+1} using fixed (train) quantile edges."""
    buckets = np.digitize(np.asarray(pred_eval, dtype=float), edges) + 1  # 1..n_q
    pos = np.where(buckets == long_q, 1, np.where(buckets == short_q, -1, 0))
    return pos.astype(float)


def run_backtest(positions, actual_ret, cost_per_side: float,
                 index=None) -> pd.DataFrame:
    """Daily gross/net returns and equity from a position series.

    turnover_t = |p_t - p_{t-1}|; cost_t = cost_per_side * turnover_t (a flat->
    long->flat round trip pays two sides across its two transitions).
    """
    p = np.asarray(positions, dtype=float)
    r = np.asarray(actual_ret, dtype=float)
    prev = np.concatenate([[0.0], p[:-1]])
    turnover = np.abs(p - prev)
    gross = p * r
    cost = cost_per_side * turnover
    net = gross - cost
    bt = pd.DataFrame({
        "position": p, "actual_ret": r, "turnover": turnover,
        "gross_ret": gross, "cost": cost, "net_ret": net,
    }, index=index)
    bt["equity_gross"] = (1 + bt["gross_ret"]).cumprod()
    bt["equity_net"] = (1 + bt["net_ret"]).cumprod()
    return bt


def buy_and_hold(actual_ret, index=None) -> pd.Series:
    r = np.asarray(actual_ret, dtype=float)
    return pd.Series((1 + r).cumprod(), index=index, name="buy_hold")


def _profit_factor(r: np.ndarray) -> float:
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    if l == 0:
        return float("inf") if g > 0 else float("nan")
    return float(g / l)


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def equity_metrics(bt: pd.DataFrame, periods_per_year: int = C.TRADING_DAYS,
                   use: str = "net_ret") -> dict:
    """CAGR, Sharpe, Profit Factor, max drawdown, trade stats (Project-2)."""
    r = bt[use].to_numpy()
    eq = (1 + r).cumprod()
    n = len(r)
    traded = bt["position"].to_numpy() != 0
    n_trades = int((bt["turnover"].to_numpy() > 0).sum())
    years = n / periods_per_year
    total = float(eq[-1] - 1)
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else float("nan")
    sd = r.std(ddof=1) if n > 1 else 0.0
    sharpe = float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    tr = r[traded]
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "profit_factor": _profit_factor(r),
        "max_drawdown": _max_drawdown(eq),
        "n_days": n,
        "n_trades": n_trades,
        "n_long": int((bt["position"] == 1).sum()),
        "n_short": int((bt["position"] == -1).sum()),
        "win_rate": float((tr > 0).mean()) if len(tr) else float("nan"),
        "avg_ret_per_active_day": float(tr.mean()) if len(tr) else float("nan"),
    }
