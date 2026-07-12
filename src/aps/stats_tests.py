"""Statistical validation (plan Sections 15-16).

Three procedures, all respecting the time-series structure of returns:

  1. Bootstrap confidence intervals
     * trade-level bootstrap: resample the (non-overlapping) trade returns;
     * stationary bootstrap: resample the per-hour return series in random blocks,
       preserving autocorrelation (Politis & Romano 1994).
  2. Permutation test: circular block shifts of the signal series against the
     fixed price paths — breaks the signal-outcome alignment while keeping the
     signal's own temporal clustering intact.
  3. White's Reality Check: does the best strategy in a candidate family beat the
     benchmark once the multiplicity of the search is accounted for, using the
     stationary bootstrap.

Every routine is fully seeded. With few trades the intervals are wide by design —
a wide interval is a result, not a failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import backtest as B


# ----------------------------------------------------------------------------
# Metric helpers on a trade-return vector
# ----------------------------------------------------------------------------
def _trade_stats(r: np.ndarray) -> dict:
    """Summary stats on a vector of per-trade returns (already net or gross)."""
    n = len(r)
    if n == 0:
        return {"total_return": 0.0, "mean_return": 0.0, "sharpe": np.nan,
                "profit_factor": np.nan, "win_rate": np.nan}
    wins, losses = r[r > 0], r[r < 0]
    sd = r.std(ddof=1) if n > 1 else 0.0
    return {
        "total_return": float(np.prod(1 + r) - 1),      # compounded
        "mean_return": float(r.mean()),
        "sharpe": float(r.mean() / sd) if sd > 0 else np.nan,  # per-trade
        "profit_factor": float(wins.sum() / -losses.sum())
                         if len(losses) and losses.sum() != 0 else np.nan,
        "win_rate": float((r > 0).mean()),
    }


# ----------------------------------------------------------------------------
# 1a. Trade-level bootstrap CI
# ----------------------------------------------------------------------------
def trade_bootstrap_ci(ledger: pd.DataFrame, B_iter: int = 10000,
                       alpha: float = 0.05, seed: int = C.SEED) -> pd.DataFrame:
    """Percentile CIs for trade metrics by resampling trades with replacement.

    Reports both net and gross. Also returns a one-sided bootstrap p-value for
    H0: total return <= 0 (fraction of resamples with total return <= 0).
    """
    rng = np.random.default_rng(seed)
    net = ledger["net_ret"].to_numpy()
    gross = ledger["gross_ret"].to_numpy()
    n = len(net)
    rows = []
    for name, r in [("net", net), ("gross", gross)]:
        point = _trade_stats(r)
        if n == 0:
            continue
        boot = {k: [] for k in point}
        for _ in range(B_iter):
            idx = rng.integers(0, n, n)
            s = _trade_stats(r[idx])
            for k in point:
                boot[k].append(s[k])
        for k, v in point.items():
            arr = np.array(boot[k], dtype=float)
            arr = arr[~np.isnan(arr)]
            lo, hi = (np.percentile(arr, 100 * alpha / 2),
                      np.percentile(arr, 100 * (1 - alpha / 2))) if len(arr) else (np.nan, np.nan)
            rows.append({"basis": name, "metric": k, "point": v,
                         "ci_lo": lo, "ci_hi": hi})
        # p-value H0: total_return <= 0
        tot = np.array(boot["total_return"])
        rows.append({"basis": name, "metric": "p_total_return_gt_0",
                     "point": float((tot <= 0).mean()), "ci_lo": np.nan, "ci_hi": np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 1b. Stationary bootstrap on the per-hour return series
# ----------------------------------------------------------------------------
def _stationary_bootstrap_index(n: int, avg_block: float, rng) -> np.ndarray:
    """One stationary-bootstrap resample of indices 0..n-1 (Politis-Romano).

    Block lengths are geometric with mean `avg_block`; wraps circularly.
    """
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


def stationary_bootstrap_ci(hourly_returns: np.ndarray, avg_block: float = 24,
                            B_iter: int = 5000, alpha: float = 0.05,
                            seed: int = C.SEED) -> dict:
    """CI for the total (summed) per-hour return via stationary bootstrap.

    Operates on the full hourly return series (0 on non-trade hours), preserving
    autocorrelation. Returns point estimate + percentile CI + p(total <= 0).
    """
    rng = np.random.default_rng(seed)
    n = len(hourly_returns)
    point = float(hourly_returns.sum())
    boot = np.empty(B_iter)
    for b in range(B_iter):
        idx = _stationary_bootstrap_index(n, avg_block, rng)
        boot[b] = hourly_returns[idx].sum()
    return {
        "point_total_return": point,
        "ci_lo": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "p_total_le_0": float((boot <= 0).mean()),
    }


# ----------------------------------------------------------------------------
# 2. Permutation test — circular block shift of the signal series
# ----------------------------------------------------------------------------
def circular_shift_permutation(direction: pd.Series, path_table: pd.DataFrame,
                               threshold: float, cost: float,
                               B_iter: int = 2000, seed: int = C.SEED) -> dict:
    """Empirical p-value that the observed return exceeds randomly-aligned signals.

    Keeps the signal series (its counts and clustering) intact and circularly
    shifts it against the fixed price paths. Reports p-values for net and gross
    total return. p = (1 + #{T_b >= T_obs}) / (B + 1).
    """
    rng = np.random.default_rng(seed)
    d = direction.to_numpy()
    n = len(d)

    obs_led = B.simulate(direction, path_table, threshold, cost)
    obs_net = _trade_stats(obs_led["net_ret"].to_numpy())["total_return"]
    obs_gross = _trade_stats(obs_led["gross_ret"].to_numpy())["total_return"]

    ge_net = ge_gross = 0
    for _ in range(B_iter):
        shift = int(rng.integers(1, n))
        d_shift = np.roll(d, shift)
        led = B.simulate(pd.Series(d_shift, index=direction.index),
                         path_table, threshold, cost)
        s_net = _trade_stats(led["net_ret"].to_numpy())["total_return"]
        s_gross = _trade_stats(led["gross_ret"].to_numpy())["total_return"]
        ge_net += (s_net >= obs_net)
        ge_gross += (s_gross >= obs_gross)
    return {
        "obs_net_total_return": obs_net,
        "obs_gross_total_return": obs_gross,
        "p_net": (1 + ge_net) / (B_iter + 1),
        "p_gross": (1 + ge_gross) / (B_iter + 1),
        "B": B_iter,
    }


# ----------------------------------------------------------------------------
# 3. White's Reality Check
# ----------------------------------------------------------------------------
def whites_reality_check(returns_matrix: np.ndarray, avg_block: float = 24,
                         B_iter: int = 2000, seed: int = C.SEED) -> dict:
    """White's Reality Check p-value over a candidate family vs a zero benchmark.

    returns_matrix: shape (T, K) of per-hour returns for K candidate strategies
    (benchmark = always-flat = 0, so f_k = strategy return). Test statistic
    V = max_k sqrt(T) * mean(f_k); the null distribution centers each strategy's
    bootstrap mean on its sample mean (Politis-Romano stationary bootstrap).
    """
    rng = np.random.default_rng(seed)
    T, K = returns_matrix.shape
    means = returns_matrix.mean(axis=0)
    V_obs = np.sqrt(T) * means.max()

    V_boot = np.empty(B_iter)
    for b in range(B_iter):
        idx = _stationary_bootstrap_index(T, avg_block, rng)
        boot_means = returns_matrix[idx].mean(axis=0)
        V_boot[b] = np.sqrt(T) * (boot_means - means).max()
    p = float((V_boot >= V_obs).mean())
    return {
        "V_obs": float(V_obs),
        "best_strategy_mean": float(means.max()),
        "reality_check_p": p,
        "n_candidates": K,
        "B": B_iter,
    }
