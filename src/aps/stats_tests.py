"""Statistical validation of the trading equity curve (Project-2).

  1. Stationary block bootstrap confidence intervals for the daily net-return
     series (total return, Sharpe, profit factor, max drawdown) — preserves
     autocorrelation (Politis & Romano 1994).
  2. Monte-Carlo permutation test using the PROFIT FACTOR as the statistic:
     circularly shift the position series against the fixed daily returns and
     recompute the profit factor; p = (1+#{PF_b >= PF_obs})/(B+1).
  3. White's Reality Check over a candidate family of models, versus a flat
     benchmark, using the stationary bootstrap.

Fully seeded. Wide intervals with few active days are a result, not a failure.
"""
from __future__ import annotations

import numpy as np

from . import config as C


# ----------------------------------------------------------------------------
# metric helpers on a daily net-return series
# ----------------------------------------------------------------------------
def _profit_factor(r: np.ndarray) -> float:
    g = r[r > 0].sum(); l = -r[r < 0].sum()
    if l == 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def _sharpe(r: np.ndarray, ppy: int = C.TRADING_DAYS) -> float:
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def _total_return(r: np.ndarray) -> float:
    return float(np.prod(1 + r) - 1)


def _max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak))


# ----------------------------------------------------------------------------
# stationary bootstrap
# ----------------------------------------------------------------------------
def _stationary_index(n: int, avg_block: float, rng) -> np.ndarray:
    p = 1.0 / avg_block
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        idx[i] = start
        i += 1
        while i < n and rng.random() >= p:
            start = (start + 1) % n
            idx[i] = start
            i += 1
    return idx


def bootstrap_ci(net_returns, avg_block: float = 10, B_iter: int = 5000,
                 alpha: float = 0.05, seed: int = C.SEED) -> dict:
    """Stationary-bootstrap 95% CIs for total return / Sharpe / PF / max DD."""
    rng = np.random.default_rng(seed)
    r = np.asarray(net_returns, dtype=float)
    n = len(r)
    stats_fns = {"total_return": _total_return, "sharpe": _sharpe,
                 "profit_factor": _profit_factor, "max_drawdown": _max_drawdown}
    boot = {k: np.empty(B_iter) for k in stats_fns}
    for b in range(B_iter):
        rb = r[_stationary_index(n, avg_block, rng)]
        for k, fn in stats_fns.items():
            boot[k][b] = fn(rb)
    out = {}
    for k, fn in stats_fns.items():
        arr = boot[k][np.isfinite(boot[k])]
        out[k] = {
            "point": fn(r),
            "ci_lo": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    out["p_total_return_le_0"] = float((boot["total_return"] <= 0).mean())
    return out


# ----------------------------------------------------------------------------
# Monte-Carlo permutation test (Profit Factor)
# ----------------------------------------------------------------------------
def mc_permutation_pf(positions, actual_ret, cost_per_side: float,
                      B_iter: int = 2000, seed: int = C.SEED) -> dict:
    """Permutation p-value using profit factor as the statistic.

    Circularly shift the position series against the fixed daily returns, recompute
    net returns (with turnover costs) and the profit factor. Preserves both the
    position autocorrelation and the return series while breaking their alignment.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(positions, dtype=float)
    r = np.asarray(actual_ret, dtype=float)
    n = len(p)

    def net_from_positions(pos):
        prev = np.concatenate([[0.0], pos[:-1]])
        turnover = np.abs(pos - prev)
        return pos * r - cost_per_side * turnover

    pf_obs = _profit_factor(net_from_positions(p))
    ge = 0
    for _ in range(B_iter):
        shift = int(rng.integers(1, n))
        pf_b = _profit_factor(net_from_positions(np.roll(p, shift)))
        ge += (pf_b >= pf_obs)
    return {"pf_obs": pf_obs, "p_value": (1 + ge) / (B_iter + 1), "B": B_iter}


# ----------------------------------------------------------------------------
# White's Reality Check
# ----------------------------------------------------------------------------
def whites_reality_check(returns_matrix: np.ndarray, avg_block: float = 10,
                         B_iter: int = 2000, seed: int = C.SEED) -> dict:
    """Reality-check p-value for the best of K candidate strategies vs flat.

    returns_matrix: (T, K) per-day net returns; benchmark = 0. Statistic
    V = max_k sqrt(T)*mean(f_k), null centers each strategy on its own mean.
    """
    rng = np.random.default_rng(seed)
    T, K = returns_matrix.shape
    means = returns_matrix.mean(axis=0)
    V_obs = np.sqrt(T) * means.max()
    V_boot = np.empty(B_iter)
    for b in range(B_iter):
        idx = _stationary_index(T, avg_block, rng)
        V_boot[b] = np.sqrt(T) * (returns_matrix[idx].mean(axis=0) - means).max()
    return {"V_obs": float(V_obs), "best_mean": float(means.max()),
            "reality_check_p": float((V_boot >= V_obs).mean()),
            "n_candidates": K, "B": B_iter}
