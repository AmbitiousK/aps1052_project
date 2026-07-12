"""Pipeline 1 — tabular ML models with a manual grid-search CV (Project-2).

Each model is an sklearn Pipeline: StandardScaler -> SelectKBest -> estimator,
so scaling and feature selection happen INSIDE cross-validation (fit on train
folds only). Model and hyperparameter selection use a manual grid-search loop
over TimeSeriesSplit folds, scored by Spearman rank correlation (trading-relevant
and robust to the trivial near-zero predictor that MAE would reward on noisy
returns). MAE is still reported.

Feature selection uses SelectKBest with f_regression for the linear model and
mutual_info_regression for the non-linear models (per the assignment).
"""
from __future__ import annotations

from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import (SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from . import config as C

_mi = partial(mutual_info_regression, random_state=C.SEED)


def _score_func(name: str):
    return f_regression if name == "linear" else _mi


def _estimator(name: str, **p):
    if name == "linear":
        return LinearRegression()
    if name == "svr":
        return SVR(kernel="rbf", C=p.get("C", 1.0), gamma=p.get("gamma", "scale"),
                   epsilon=p.get("epsilon", 0.01))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 400),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            max_features=p.get("max_features", "sqrt"),
            n_jobs=-1, random_state=C.SEED)
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=p.get("n_estimators", 400),
            learning_rate=p.get("learning_rate", 0.03),
            num_leaves=p.get("num_leaves", 31),
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=C.SEED, verbose=-1)
    raise ValueError(name)


def build_pipeline(name: str, k: int, **params) -> Pipeline:
    """StandardScaler -> SelectKBest -> estimator."""
    k_eff = min(k, len(C.FEATURES))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=_score_func(name), k=k_eff)),
        ("model", _estimator(name, **params)),
    ])


# Hyperparameter grids (kept small for a clean manual CV). `k` = SelectKBest.
MODEL_GRIDS: dict[str, dict] = {
    "linear": {"k": [5, 10, len(C.FEATURES)]},
    "svr": {"k": [8, len(C.FEATURES)], "C": [1.0, 10.0], "gamma": ["scale", 0.1],
            "epsilon": [0.005, 0.01]},
    "random_forest": {"k": [10, len(C.FEATURES)], "n_estimators": [300, 600],
                      "max_depth": [4, 8, None], "min_samples_leaf": [5, 20]},
    "lightgbm": {"k": [10, len(C.FEATURES)], "n_estimators": [300, 600],
                 "learning_rate": [0.02, 0.05], "num_leaves": [15, 31]},
}

MODELS = list(MODEL_GRIDS)


def _param_combos(grid: dict):
    keys = list(grid)
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _cv_spearman(X, y, name: str, params: dict, n_splits: int = 5) -> float:
    """Mean out-of-fold Spearman RHO over TimeSeriesSplit folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, va in tscv.split(X):
        pipe = build_pipeline(name, params.get("k", len(C.FEATURES)),
                              **{k: v for k, v in params.items() if k != "k"})
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[va])
        rho, _ = stats.spearmanr(y.iloc[va], pred)
        scores.append(0.0 if np.isnan(rho) else rho)
    return float(np.mean(scores))


def manual_grid_search(X, y, name: str, n_splits: int = 5) -> dict:
    """Manual grid-search CV over a model's grid; returns best params + all rows."""
    rows = []
    best = {"cv_spearman": -np.inf, "params": None}
    for params in _param_combos(MODEL_GRIDS[name]):
        s = _cv_spearman(X, y, name, params, n_splits)
        rows.append({**params, "cv_spearman": s})
        if s > best["cv_spearman"]:
            best = {"cv_spearman": s, "params": params}
    return {"model": name, "best_params": best["params"],
            "best_cv_spearman": best["cv_spearman"],
            "results": pd.DataFrame(rows)}


def fit_best(X, y, name: str, best_params: dict) -> Pipeline:
    k = best_params.get("k", len(C.FEATURES))
    pipe = build_pipeline(name, k, **{k_: v for k_, v in best_params.items() if k_ != "k"})
    pipe.fit(X, y)
    return pipe
