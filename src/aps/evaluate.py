"""Regression evaluation — the required metrics (Project-2).

Validation and test metrics:
  * MAE
  * Spearman rank correlation RHO (pred vs actual return)
  * Profit Factor (on a realized-return series)
  * Directional accuracy by quantile, compared to the baseline of always betting
    the unconditional majority direction.

The quantile view is the trading-relevant one: bucket days by predicted return,
and ask whether the model's directional accuracy in each bucket beats simply
always betting the sample's majority direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

from . import config as C


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def spearman_rho(y_true, y_pred) -> float:
    rho, _ = stats.spearmanr(np.asarray(y_true), np.asarray(y_pred))
    return float(rho)


def profit_factor(returns) -> float:
    """Gross profit / gross loss on a realized per-trade (or per-day) return series."""
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def directional_accuracy(y_true, y_pred) -> float:
    yt, yp = np.sign(np.asarray(y_true)), np.sign(np.asarray(y_pred))
    yt[yt == 0] = 1
    yp[yp == 0] = 1
    return float((yt == yp).mean())


def directional_accuracy_by_quantile(y_true, y_pred, n_q: int = C.N_QUANTILES
                                     ) -> pd.DataFrame:
    """Per predicted-return quantile: model directional accuracy vs a baseline
    that always bets the unconditional majority direction of the sample."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    # unconditional majority direction on this sample
    majority = 1.0 if (yt > 0).mean() >= 0.5 else -1.0

    ranks = pd.qcut(pd.Series(yp).rank(method="first"), n_q, labels=False) + 1
    sign_t = np.where(yt >= 0, 1.0, -1.0)
    sign_p = np.where(yp >= 0, 1.0, -1.0)

    rows = []
    for q in range(1, n_q + 1):
        m = ranks.to_numpy() == q
        n = int(m.sum())
        rows.append({
            "quantile": q, "n": n,
            "mean_pred": float(yp[m].mean()),
            "mean_actual": float(yt[m].mean()),
            "up_rate": float((yt[m] > 0).mean()),
            "model_dir_acc": float((sign_p[m] == sign_t[m]).mean()),
            "baseline_dir_acc": float((majority == sign_t[m]).mean()),
        })
    df = pd.DataFrame(rows)
    df["edge_vs_baseline"] = df["model_dir_acc"] - df["baseline_dir_acc"]
    return df


def regression_metrics(y_true, y_pred, *, model: str = "", split: str = "") -> dict:
    """Scalar summary: MAE, Spearman RHO, directional accuracy, and the
    directional edge in the extreme (traded) quantiles."""
    dq = directional_accuracy_by_quantile(y_true, y_pred)
    top = dq[dq["quantile"] == C.LONG_QUANTILE].iloc[0]
    bot = dq[dq["quantile"] == C.SHORT_QUANTILE].iloc[0]
    return {
        "model": model, "split": split, "n": len(y_true),
        "mae": mae(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
        "top_q_dir_acc": float(top["model_dir_acc"]),
        "top_q_edge": float(top["edge_vs_baseline"]),
        "bot_q_dir_acc": float(bot["model_dir_acc"]),
        "bot_q_edge": float(bot["edge_vs_baseline"]),
    }
