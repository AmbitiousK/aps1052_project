"""Trading-signal construction from model probabilities (plan Section 10).

Maps predicted class probabilities to a per-hour direction in {-1, 0, +1}. Two
mappings:
  * argmax  — trade the most probable class (no filtering);
  * confidence-filtered — only open a position when an extreme-class probability
    clears a threshold tau, otherwise stay flat.

Because class balancing makes the probabilities over-confident (Stage 4), tau is
an operating point calibrated on validation, not a literal probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

P_COLS = [f"p_{c}" for c in C.CLASSES]  # p_-1, p_0, p_1


def argmax_signals(proba_df: pd.DataFrame) -> pd.Series:
    """Direction = argmax over the three classes."""
    idx = proba_df[P_COLS].to_numpy().argmax(1)
    return pd.Series([C.CLASSES[i] for i in idx], index=proba_df.index,
                     name="direction")


def confidence_signals(proba_df: pd.DataFrame, tau: float) -> pd.Series:
    """Direction from an extreme-probability threshold.

    +1 if P(+1) >= tau, -1 if P(-1) >= tau, else 0. If both clear tau (rare),
    take the larger probability.
    """
    p_up = proba_df["p_1"].to_numpy()
    p_dn = proba_df["p_-1"].to_numpy()
    long = p_up >= tau
    short = p_dn >= tau
    d = np.where(long & short, np.where(p_up >= p_dn, 1, -1),
        np.where(long, 1, np.where(short, -1, 0)))
    return pd.Series(d, index=proba_df.index, name="direction")


def signal_coverage(direction: pd.Series) -> dict:
    """Basic signal-frequency stats for a direction series."""
    n = len(direction)
    nz = int((direction != 0).sum())
    return {
        "n": n,
        "n_signals": nz,
        "coverage": nz / n if n else 0.0,
        "n_long": int((direction == 1).sum()),
        "n_short": int((direction == -1).sum()),
    }


def model_proba(preds: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    """Pull one (model, split) probability frame from the Stage-3 predictions,
    indexed by timestamp with columns p_-1/p_0/p_1 and y."""
    sub = preds[(preds["model"] == model) & (preds["split"] == split)].copy()
    sub = sub.set_index("ts").sort_index()
    return sub[P_COLS + ["y"]]
