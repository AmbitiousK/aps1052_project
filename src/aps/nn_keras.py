"""Pipeline 2 — Keras MLP regressor (Project-2; PyTorch is forbidden).

The neural pipeline uses rolling-scaled features (applied to the whole matrix
before the split, per the assignment), an MLP with L2 weight penalties and
dropout, MAE loss, Adam, and early stopping. Model/hyperparameter selection uses
a manual grid-search loop over TimeSeriesSplit folds scored by Spearman RHO.

Loss: mean absolute error. Penalties: L2 (kernel_regularizer) + dropout.
Sample weights: none (documented choice — the target is a homoscedastic daily
log return, so uniform weighting is appropriate).
"""
from __future__ import annotations

import os
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config as C


def _seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class KerasMLPRegressor(BaseEstimator, RegressorMixin):
    """Small MLP regressor with L2 + dropout, MAE loss, early stopping."""

    def __init__(self, hidden=(32, 16), l2=1e-4, dropout=0.2, lr=1e-3,
                 epochs=300, patience=25, batch_size=32, val_frac=0.15,
                 seed: int = C.SEED):
        self.hidden = hidden
        self.l2 = l2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed

    def _build(self, n_in: int) -> keras.Model:
        m = keras.Sequential([layers.Input((n_in,))])
        for h in self.hidden:
            m.add(layers.Dense(h, activation="relu",
                               kernel_regularizer=regularizers.l2(self.l2)))
            m.add(layers.Dropout(self.dropout))
        m.add(layers.Dense(1))
        m.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mae")
        return m

    def fit(self, X, y):
        _seed(self.seed)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        n = len(X)
        cut = int(n * (1 - self.val_frac))
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_ = self._build(X.shape[1])
        self.model_.fit(
            X[:cut], y[:cut], validation_data=(X[cut:], y[cut:]),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[es], verbose=0)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype="float32"),
                                   verbose=0).ravel()


NN_GRID = {
    "hidden": [(32, 16), (64, 32)],
    "l2": [1e-4, 1e-3],
    "dropout": [0.2],
    "lr": [1e-3],
}


def _combos(grid: dict):
    keys = list(grid)
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def manual_grid_search_nn(X, y, n_splits: int = 4) -> dict:
    """Manual grid-search CV for the MLP over TimeSeriesSplit, scored by Spearman."""
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows, best = [], {"cv_spearman": -np.inf, "params": None}
    for params in _combos(NN_GRID):
        scores = []
        for tr, va in tscv.split(X):
            mdl = KerasMLPRegressor(**params).fit(X[tr], y[tr])
            pred = mdl.predict(X[va])
            rho, _ = stats.spearmanr(y[va], pred)
            scores.append(0.0 if np.isnan(rho) else rho)
        s = float(np.mean(scores))
        rows.append({**params, "cv_spearman": s})
        if s > best["cv_spearman"]:
            best = {"cv_spearman": s, "params": params}
    return {"model": "keras_mlp", "best_params": best["params"],
            "best_cv_spearman": best["cv_spearman"],
            "results": pd.DataFrame(rows)}
