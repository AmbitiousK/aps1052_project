"""R2/R3 — feature preparation (two representations on one common index) and
feature-selection ranking.

Both pipelines must be trained and tested on the SAME rows so their metrics are
comparable and the final test set is identical. Rolling scaling (for the neural
pipeline) drops the first `std_window` warm-up rows, so that trimmed index is the
common index used by BOTH pipelines:

  * `X_ml` : meaning-preserving features (StandardScaler applied later, inside the
             Scikit-Learn grid pipeline);
  * `X_nn` : the same features after rolling scaling (for the Keras MLP).

`feature_ranking` reports SelectKBest scores (f_regression + mutual information)
on the training rows so the weakest 1-2 features can be identified/discarded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression

from . import config as C
from . import scaling as SC


@dataclass(frozen=True)
class Prepared:
    X_ml: pd.DataFrame     # meaning-preserving (pre-StandardScaler)
    X_nn: pd.DataFrame     # rolling-scaled
    y: pd.Series           # next-day log return target
    close: pd.Series       # for backtesting
    index: pd.DatetimeIndex


def prepare(ds: pd.DataFrame) -> Prepared:
    """Build both feature representations aligned on the rolling-valid index."""
    base = SC.meaning_preserving_transform(ds)          # 1060 rows
    nn = SC.rolling_scale_matrix(base).dropna()         # ~971 rows after warm-up
    idx = nn.index
    return Prepared(
        X_ml=base.loc[idx],
        X_nn=nn,
        y=ds[C.TARGET].loc[idx],
        close=ds[C.COL_CLOSE].loc[idx],
        index=idx,
    )


def feature_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """f_regression F-score and mutual information per feature (on given rows)."""
    Xn = X.to_numpy()
    yn = np.asarray(y)
    f_sc, _ = f_regression(Xn, yn)
    mi = mutual_info_regression(Xn, yn, random_state=C.SEED)
    df = pd.DataFrame({"feature": X.columns, "f_score": f_sc, "mutual_info": mi})
    df["f_rank"] = df["f_score"].rank(ascending=False).astype(int)
    df["mi_rank"] = df["mutual_info"].rank(ascending=False).astype(int)
    df["avg_rank"] = (df["f_rank"] + df["mi_rank"]) / 2
    return df.sort_values("avg_rank").reset_index(drop=True)
