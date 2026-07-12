"""Classification evaluation — the single scoring standard (plan Section 9).

Accuracy alone is meaningless at 99:1 imbalance (predicting all-0 scores ~99%),
so we always report balanced accuracy, macro/weighted P/R/F1, MCC, per-class
metrics, the confusion matrix, the extreme-event (|label|=1) binary view, and —
when probabilities are available — log loss / Brier / PR-AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, confusion_matrix, log_loss,
    average_precision_score,
)

from . import config as C


def classification_metrics(
    y_true, y_pred, y_proba: np.ndarray | None = None, *,
    model: str = "", split: str = "",
) -> dict:
    """Scalar metrics for one (model, split). y_proba is (n, 3) over CLASSES order."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=C.CLASSES, average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=C.CLASSES, average="weighted", zero_division=0)
    p_cls, r_cls, f_cls, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=C.CLASSES, average=None, zero_division=0)

    out = {
        "model": model, "split": split, "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": p_macro, "macro_recall": r_macro, "macro_f1": f_macro,
        "weighted_f1": f_w,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for i, c in enumerate(C.CLASSES):
        out[f"precision_{c}"] = p_cls[i]
        out[f"recall_{c}"] = r_cls[i]
        out[f"f1_{c}"] = f_cls[i]
        out[f"support_{c}"] = int(sup[i])

    # -- extreme-event binary view (|label| == 1) ---------------------------
    z_true = (y_true != 0).astype(int)
    z_pred = (y_pred != 0).astype(int)
    zp, zr, zf, _ = precision_recall_fscore_support(
        z_true, z_pred, average="binary", zero_division=0)
    out["extreme_precision"] = zp
    out["extreme_recall"] = zr
    out["extreme_f1"] = zf

    # -- probability-based metrics ------------------------------------------
    if y_proba is not None:
        proba = np.asarray(y_proba)
        # guard against zeros for log loss
        eps = 1e-15
        proba_c = np.clip(proba, eps, 1 - eps)
        out["log_loss"] = log_loss(y_true, proba_c, labels=C.CLASSES)
        # extreme-event score = 1 - P(flat); PR-AUC on the binary extreme target
        flat_idx = C.CLASSES.index(0)
        extreme_score = 1.0 - proba[:, flat_idx]
        if z_true.sum() > 0:
            out["extreme_pr_auc"] = average_precision_score(z_true, extreme_score)
        else:
            out["extreme_pr_auc"] = np.nan
        # multiclass Brier = mean squared error vs one-hot
        onehot = np.zeros_like(proba)
        for i, c in enumerate(C.CLASSES):
            onehot[:, i] = (y_true == c).astype(float)
        out["brier"] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    else:
        out["log_loss"] = np.nan
        out["extreme_pr_auc"] = np.nan
        out["brier"] = np.nan

    return out


def confusion(y_true, y_pred) -> pd.DataFrame:
    """3x3 confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=C.CLASSES)
    names = [C.CLASS_NAMES[c] for c in C.CLASSES]
    return pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                        columns=[f"pred_{n}" for n in names])


def metrics_frame(results: list[dict]) -> pd.DataFrame:
    """Stack many metric dicts into one comparison table."""
    return pd.DataFrame(results)


def extreme_calibration(y_true, extreme_score, n_bins: int = 10):
    """Reliability curve for the extreme-event probability.

    Bins the predicted extreme probability (= 1 - P(flat)) and returns, per bin,
    the mean predicted probability vs the observed extreme frequency. A calibrated
    model tracks the diagonal.
    """
    from sklearn.calibration import calibration_curve
    z = (np.asarray(y_true) != 0).astype(int)
    score = np.asarray(extreme_score)
    # calibration_curve needs both classes present
    if z.sum() == 0 or z.sum() == len(z):
        return np.array([]), np.array([])
    frac_pos, mean_pred = calibration_curve(
        z, score, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos
