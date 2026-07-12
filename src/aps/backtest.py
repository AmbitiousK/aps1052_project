"""Barrier-aligned backtest engine (plan Section 11).

Trading rule, aligned with the label so the model's prediction target and the
execution are identical:
  * enter at the next-minute open once the hourly signal is known;
  * take-profit and stop-loss both at the barrier magnitude (TP = SL = threshold);
  * maximum holding = 1 hour; first-touch on the 1-minute path decides the exit;
  * if neither barrier is touched, time-stop at the end of the hour;
  * at most one position per hour, so windows never overlap.

Returns are computed from the precomputed path table (aps.pathdata). Costs are a
per-side fraction (fee + slippage), charged on entry and exit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# exit reasons
TP, SL, TIME, AMBIG = "tp", "sl", "time", "ambig"


def simulate(signals: pd.Series, path_table: pd.DataFrame, threshold: float,
             cost_per_side: float) -> pd.DataFrame:
    """Run the backtest for a per-hour direction series (values in {-1,0,+1}).

    Returns a trade ledger (one row per non-zero signal) with entry/exit info and
    gross/net returns.
    """
    df = path_table.reindex(signals.index).copy()
    d = signals.to_numpy()
    traded = d != 0

    up, dn = df["first_up_time"], df["first_dn_time"]
    has_up, has_dn = up.notna().to_numpy(), dn.notna().to_numpy()
    up_t, dn_t = up.to_numpy(), dn.to_numpy()
    up_first = has_up & (~has_dn | (up_t < dn_t))
    dn_first = has_dn & (~has_up | (dn_t < up_t))
    same = has_up & has_dn & (up_t == dn_t)

    entry = df["entry_open"].to_numpy()
    tmove = df["time_exit_close"].to_numpy() / entry - 1.0

    # gross return by direction
    long_gross = np.where(up_first, threshold,
                 np.where(dn_first, -threshold,
                 np.where(same, -threshold, tmove)))
    short_gross = np.where(dn_first, threshold,
                  np.where(up_first, -threshold,
                  np.where(same, -threshold, -tmove)))
    gross = np.where(d == 1, long_gross, np.where(d == -1, short_gross, 0.0))

    # exit reason + holding minutes
    reason = np.where(same, AMBIG,
             np.where(up_first | dn_first,
                      np.where((d == 1) & up_first, TP,
                      np.where((d == 1) & dn_first, SL,
                      np.where((d == -1) & dn_first, TP, SL))),
                      TIME)).astype(object)
    touch_time = np.where(up_first, up_t, np.where(dn_first, dn_t,
                          np.datetime64("NaT")))
    entry_ts = df.index.to_numpy()
    hold_min = (touch_time - entry_ts) / np.timedelta64(1, "m")
    hold_min = np.where(np.isnan(hold_min), 60.0, hold_min)

    net = gross - 2 * cost_per_side

    ledger = pd.DataFrame({
        "direction": d,
        "entry_open": entry,
        "exit_reason": reason,
        "gross_ret": gross,
        "cost": 2 * cost_per_side,
        "net_ret": net,
        "hold_min": hold_min,
    }, index=signals.index)
    ledger = ledger[traded].copy()
    ledger.index.name = "entry_ts"
    return ledger


def equity_curve(ledger: pd.DataFrame, index: pd.DatetimeIndex,
                 use: str = "net_ret") -> pd.Series:
    """Compounded equity over the full decision index (flat between trades)."""
    r = pd.Series(0.0, index=index)
    if len(ledger):
        r.loc[ledger.index] = ledger[use].to_numpy()
    return (1.0 + r).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((peak - equity) / peak).max())


def buy_and_hold(close: pd.Series) -> pd.Series:
    """Benchmark: normalized buy-and-hold equity over the same index."""
    return close / close.iloc[0]


def trade_metrics(ledger: pd.DataFrame, index: pd.DatetimeIndex,
                  rf: float = 0.0) -> dict:
    """Return + risk-adjusted + trade-quality metrics for one backtest run."""
    eq = equity_curve(ledger, index, "net_ret")
    eq_gross = equity_curve(ledger, index, "gross_ret")
    n = len(ledger)
    out = {"n_trades": n}
    if n == 0:
        out.update({"total_net_return": 0.0, "total_gross_return": 0.0,
                    "sharpe": np.nan, "sortino": np.nan, "max_drawdown": 0.0,
                    "win_rate": np.nan, "profit_factor": np.nan})
        return out

    r = ledger["net_ret"].to_numpy()
    wins, losses = r[r > 0], r[r < 0]
    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    trades_per_year = n / years
    mu, sd = r.mean(), r.std(ddof=1) if n > 1 else 0.0
    downside = r[r < rf]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0

    out.update({
        "total_net_return": float(eq.iloc[-1] - 1),
        "total_gross_return": float(eq_gross.iloc[-1] - 1),
        "mean_ret_per_trade": float(mu),
        "median_ret_per_trade": float(np.median(r)),
        "sharpe": float((mu - rf) / sd * np.sqrt(trades_per_year)) if sd > 0 else np.nan,
        "sortino": float((mu - rf) / dsd * np.sqrt(trades_per_year)) if dsd > 0 else np.nan,
        "max_drawdown": max_drawdown(eq),
        "win_rate": float((r > 0).mean()),
        "loss_rate": float((r < 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
        "avg_hold_min": float(ledger["hold_min"].mean()),
        "n_long": int((ledger["direction"] == 1).sum()),
        "n_short": int((ledger["direction"] == -1).sum()),
        "long_net_return": float(ledger.loc[ledger["direction"] == 1, "net_ret"].sum()),
        "short_net_return": float(ledger.loc[ledger["direction"] == -1, "net_ret"].sum()),
        "trades_per_year": float(trades_per_year),
    })
    return out
