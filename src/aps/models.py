"""Model zoo — baselines and the model factory.

Every estimator exposes the sklearn-style interface (`fit`, `predict`, and where
meaningful `predict_proba` over CLASSES order) so the evaluation pipeline treats
baselines and real models identically. Preprocessing (scaling) is folded into a
Pipeline where the model needs it, and every scaler is fit on train only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from . import config as C


# ----------------------------------------------------------------------------
# Baseline 1 — majority class
# ----------------------------------------------------------------------------
class MajorityClassifier(BaseEstimator, ClassifierMixin):
    """Always predict the most frequent training class (usually flat = 0)."""

    def fit(self, X, y):
        y = np.asarray(y)
        vals, counts = np.unique(y, return_counts=True)
        self.majority_ = int(vals[np.argmax(counts)])
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=int)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(C.CLASSES)))
        proba[:, C.CLASSES.index(self.majority_)] = 1.0
        return proba


# ----------------------------------------------------------------------------
# Baseline 2 — random prediction from class priors
# ----------------------------------------------------------------------------
class PriorRandomClassifier(BaseEstimator, ClassifierMixin):
    """Predict by sampling from the training class distribution (seeded).

    predict_proba returns the constant class priors — a proper probabilistic
    baseline for log loss / PR-AUC.
    """

    def __init__(self, seed: int = C.SEED):
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        self.priors_ = np.array([(y == c).mean() for c in C.CLASSES])
        return self

    def predict(self, X):
        rng = np.random.default_rng(self.seed)
        return rng.choice(C.CLASSES, size=len(X), p=self.priors_)

    def predict_proba(self, X):
        return np.tile(self.priors_, (len(X), 1))


# ----------------------------------------------------------------------------
# Baseline 3 — simple momentum rule
# ----------------------------------------------------------------------------
class MomentumRule(BaseEstimator, ClassifierMixin):
    """Threshold the short-term momentum feature: +1 if ret_1h>θ, -1 if <-θ.

    θ is selected on TRAIN by maximizing macro-F1 over a grid of |ret_1h|
    quantiles — a genuine (train-only) rule, not a peek at validation/test.
    """

    def __init__(self, feature: str = "ret_1h",
                 quantile_grid: tuple[float, ...] = (0.80, 0.90, 0.95, 0.975, 0.99)):
        self.feature = feature
        self.quantile_grid = quantile_grid

    def fit(self, X, y):
        x = np.asarray(X[self.feature])
        y = np.asarray(y)
        best_theta, best_f1 = None, -1.0
        for q in self.quantile_grid:
            theta = np.quantile(np.abs(x), q)
            pred = np.where(x > theta, 1, np.where(x < -theta, -1, 0))
            f1 = f1_score(y, pred, labels=C.CLASSES, average="macro",
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_theta = f1, theta
        self.theta_ = float(best_theta)
        self.train_macro_f1_ = float(best_f1)
        return self

    def predict(self, X):
        x = np.asarray(X[self.feature])
        return np.where(x > self.theta_, 1, np.where(x < -self.theta_, -1, 0))


# ----------------------------------------------------------------------------
# Primary model factories
# ----------------------------------------------------------------------------
def make_logistic(seed: int = C.SEED) -> Pipeline:
    """Multinomial logistic regression with standardized inputs, balanced classes.

    Scaling matters for a linear model; class_weight='balanced' counters the
    99:1 imbalance. The scaler lives inside the Pipeline so it is fit only on the
    data passed to `.fit` (i.e. train).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            C=1.0,
            random_state=seed,
        )),
    ])


def make_random_forest(seed: int = C.SEED) -> "RandomForestClassifier":
    """Random forest — non-linear, scale-free, class-balanced."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_lightgbm(seed: int = C.SEED):
    """LightGBM gradient boosting with balanced class weights."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def class_sample_weights(y) -> np.ndarray:
    """Balanced per-sample weights w_c = N / (K * N_c) — for models without
    a native class_weight argument (e.g. XGBoost multiclass)."""
    y = np.asarray(y)
    n, k = len(y), len(C.CLASSES)
    counts = {c: max((y == c).sum(), 1) for c in C.CLASSES}
    return np.array([n / (k * counts[c]) for c in y], dtype=float)


class XGBMulticlass(BaseEstimator, ClassifierMixin):
    """XGBoost multiclass wrapper that keeps the native {-1,0,1} label space.

    XGBoost 2.x requires 0-based consecutive targets, so we map {-1,0,1}->{0,1,2}
    internally and expose `classes_ = [-1,0,1]` plus predict/predict_proba in the
    original label space. Class balance is applied via balanced sample weights at
    fit time (XGBoost has no multiclass class_weight).
    """

    def __init__(self, seed: int = C.SEED, balanced: bool = True):
        self.seed = seed
        self.balanced = balanced

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.classes_ = np.array(C.CLASSES)
        y_idx = remap_labels_to_index(y)
        sw = class_sample_weights(y) if self.balanced else None
        self._model = XGBClassifier(
            n_estimators=600, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", random_state=self.seed, n_jobs=-1,
            eval_metric="mlogloss",
        )
        self._model.fit(X, y_idx, sample_weight=sw)
        return self

    def predict(self, X):
        idx = self._model.predict(X)
        return np.array([C.CLASSES[i] for i in idx])

    def predict_proba(self, X):
        # xgb columns already in index order 0,1,2 == CLASSES order
        return self._model.predict_proba(X)


def make_xgboost(seed: int = C.SEED) -> "XGBMulticlass":
    """XGBoost multiclass with balanced sample weights, native label space."""
    return XGBMulticlass(seed=seed, balanced=True)


def remap_labels_to_index(y) -> np.ndarray:
    """Map labels {-1,0,1} -> {0,1,2} (XGBoost needs 0-based). Inverse: CLASSES[i]."""
    lut = {c: i for i, c in enumerate(C.CLASSES)}
    return np.array([lut[v] for v in np.asarray(y)], dtype=int)


def predict_proba_aligned(model, X) -> np.ndarray:
    """Return predict_proba re-ordered to CLASSES = [-1, 0, 1].

    sklearn orders columns by `model.classes_`; if a class is absent from the
    fitted model its column is filled with zeros.
    """
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", C.CLASSES))
    out = np.zeros((len(X), len(C.CLASSES)))
    for i, c in enumerate(C.CLASSES):
        if c in classes:
            out[:, i] = proba[:, classes.index(c)]
    return out
