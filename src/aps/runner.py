"""Small orchestration helper: rebuild the selected model and predict any split.

Routes tabular models (X_ml, StandardScaler+SelectKBest pipeline) and the Keras
MLP (X_nn, rolling-scaled) through one interface so the test/trading pipelines do
not care which model was selected. Training is deterministic (seeded), so a fresh
fit reproduces the validation-stage model exactly.
"""
from __future__ import annotations

import ast

import pandas as pd

from . import config as C
from . import features as F
from . import models as M
from .data import chronological_split


def _parse_params(params) -> dict:
    if isinstance(params, dict):
        return params
    return ast.literal_eval(params) if params else {}


def prepare_splits(ds: pd.DataFrame):
    """Return (prep, split_ml, split_nn, y_by_part) on the common index."""
    prep = F.prepare(ds)
    split_ml = chronological_split(prep.X_ml.assign(**{C.TARGET: prep.y,
                                                       C.COL_CLOSE: prep.close}))
    split_nn = chronological_split(prep.X_nn.assign(**{C.TARGET: prep.y,
                                                       C.COL_CLOSE: prep.close}))
    y = {p: split_ml.parts[p][C.TARGET] for p in ("train", "val", "test")}
    return prep, split_ml, split_nn, y


def fit_predict(name: str, params, ds: pd.DataFrame,
                parts=("train", "val", "test")) -> dict:
    """Train `name` on train and predict the requested parts.

    Returns {'model': fitted, 'pred': {part: Series}, 'split_ml', 'split_nn'}.
    """
    params = _parse_params(params)
    prep, split_ml, split_nn, y = prepare_splits(ds)

    if name == "keras_mlp":
        from .nn_keras import KerasMLPRegressor
        Xtr = split_nn.train[C.FEATURES]
        mdl = KerasMLPRegressor(**params).fit(Xtr, y["train"])
        pred = {p: pd.Series(mdl.predict(split_nn.parts[p][C.FEATURES]),
                             index=split_nn.parts[p].index) for p in parts}
    else:
        Xtr = split_ml.train[C.FEATURES]
        mdl = M.fit_best(Xtr, y["train"], name, params)
        pred = {p: pd.Series(mdl.predict(split_ml.parts[p][C.FEATURES]),
                             index=split_ml.parts[p].index) for p in parts}
    return {"model": mdl, "pred": pred, "split_ml": split_ml,
            "split_nn": split_nn, "y": y, "prep": prep}
