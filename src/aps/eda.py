"""Stage 1b — exploratory data analysis (analysis only, no data redesign).

Pure functions returning tables. Plots and file IO live in the pipeline. Covers
the plan's Section 6: descriptive stats, correlation, redundancy (VIF, mutual
information), feature-by-label separation, distribution shift across splits, and
extreme-event time clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from . import config as C
from .data import Split


# ----------------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------------
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extended per-feature descriptive statistics (train-agnostic, whole df)."""
    X = df[C.FEATURES]
    desc = X.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    desc["skew"] = X.skew()
    desc["kurtosis"] = X.kurtosis()
    desc.index.name = "feature"
    return desc.reset_index()


# ----------------------------------------------------------------------------
# Correlation & redundancy
# ----------------------------------------------------------------------------
def correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Feature-feature correlation matrix (pearson or spearman)."""
    return df[C.FEATURES].corr(method=method)


def top_correlated_pairs(corr: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k most strongly correlated feature pairs (by |corr|)."""
    m = corr.abs().copy()
    np.fill_diagonal(m.values, np.nan)
    pairs = (
        m.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "abs_corr"})
    )
    # each pair appears twice (a,b)/(b,a) -> keep one ordering
    pairs = pairs[pairs["feat_a"] < pairs["feat_b"]]
    pairs["corr"] = [corr.loc[a, b] for a, b in zip(pairs["feat_a"], pairs["feat_b"])]
    return pairs.sort_values("abs_corr", ascending=False).head(k).reset_index(drop=True)


def variance_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """VIF per feature via R^2 of regressing each feature on the others.

    VIF_j = 1 / (1 - R^2_j). >5 warns of moderate multicollinearity, >10 severe.
    Computed without statsmodels: R^2 from a least-squares fit on standardized X.
    """
    X = df[C.FEATURES].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    n_feat = X.shape[1]
    vifs = []
    for j in range(n_feat):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        # add intercept
        A = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
        vifs.append(vif)
    return pd.DataFrame({"feature": C.FEATURES, "VIF": vifs}).sort_values(
        "VIF", ascending=False).reset_index(drop=True)


def mutual_information(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Mutual information between each feature and the 3-class label.

    A non-linear complement to correlation: how much each feature reduces
    uncertainty about the label. Higher = more informative.
    """
    X = df[C.FEATURES]
    y = df[C.COL_LABEL]
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=seed)
    return pd.DataFrame({"feature": C.FEATURES, "mutual_info": mi}).sort_values(
        "mutual_info", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Feature separation by label
# ----------------------------------------------------------------------------
def feature_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each feature within each label class (long format)."""
    g = df.groupby(C.COL_LABEL)[C.FEATURES].mean().T
    g.columns = [f"mean_label_{c}" for c in g.columns]
    g.index.name = "feature"
    return g.reset_index()


# ----------------------------------------------------------------------------
# Distribution shift across splits
# ----------------------------------------------------------------------------
def distribution_shift(split: Split) -> pd.DataFrame:
    """Two-sample KS statistic (train vs val, train vs test) per feature.

    Large KS = the feature's marginal distribution moved between periods, i.e.
    a non-stationarity / regime-shift warning for that variable.
    """
    rows = []
    tr = split.train[C.FEATURES]
    for feat in C.FEATURES:
        ks_val = stats.ks_2samp(tr[feat], split.val[feat]).statistic
        ks_test = stats.ks_2samp(tr[feat], split.test[feat]).statistic
        rows.append({"feature": feat,
                     "ks_train_val": ks_val,
                     "ks_train_test": ks_test})
    return pd.DataFrame(rows).sort_values("ks_train_test", ascending=False
                                          ).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Extreme-event time clustering
# ----------------------------------------------------------------------------
def extreme_events_by_period(df: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Count extreme events (label != 0) per calendar period (default monthly).

    Reveals whether extreme hours cluster in specific regimes rather than spread
    uniformly — key context for the imbalance and for time-stability robustness.
    """
    ext = (df[C.COL_LABEL] != 0).astype(int)
    up = (df[C.COL_LABEL] == 1).astype(int)
    dn = (df[C.COL_LABEL] == -1).astype(int)
    grp = pd.DataFrame({"extreme": ext, "up": up, "down": dn}, index=df.index)
    out = grp.resample(freq).sum()
    out["total_hours"] = grp["extreme"].resample(freq).count()
    out["extreme_ratio"] = out["extreme"] / out["total_hours"].replace(0, np.nan)
    return out
