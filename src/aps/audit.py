"""Stage 1 — data integrity audit.

Pure functions that take DataFrames / Splits and return audit tables. No plotting,
no file IO here; the pipeline (`pipelines/s1_audit.py`) orchestrates and writes.

The audit answers the plan's Section 4 checks:
  * data dictionary (every column: role, dtype, range, missing)
  * timestamp audit (monotone, unique, hourly grid, gaps)
  * missing-value audit (per feature, per split)
  * class-distribution audit (per threshold, per split)
  * label-integrity audit (labels consistent with the raw forward stats)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import Split


# ----------------------------------------------------------------------------
# Data dictionary
# ----------------------------------------------------------------------------
def _role(col: str) -> str:
    if col in C.FEATURES:
        return "feature"
    if col == C.COL_LABEL:
        return "target"
    if col == C.COL_CLOSE:
        return "price (backtest only)"
    if col in C.FORWARD_COLS:
        return "forward stat (relabel only)"
    return "other"


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: role, dtype, summary stats, missing count."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "role": _role(col),
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "min": s.min() if np.issubdtype(s.dtype, np.number) else "",
            "max": s.max() if np.issubdtype(s.dtype, np.number) else "",
            "mean": s.mean() if np.issubdtype(s.dtype, np.number) else "",
            "std": s.std() if np.issubdtype(s.dtype, np.number) else "",
            "description": C.FEATURE_DESCRIPTIONS.get(col, ""),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Timestamp audit
# ----------------------------------------------------------------------------
def timestamp_audit(df: pd.DataFrame) -> dict:
    """Check the hourly index: monotone, unique, on the hour, and gap structure.

    The dataset legitimately has gaps (invalid exchange hours and incomplete
    forward windows were dropped by the pipeline), so gaps are *reported*, not
    treated as errors. What would be an error: non-monotone order, duplicate
    timestamps, or timestamps not aligned to the hour.
    """
    idx = df.index
    deltas = idx.to_series().diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    gaps = deltas[deltas > one_hour]

    expected_full_grid = int((idx[-1] - idx[0]) / one_hour) + 1
    top_gaps = (
        gaps.sort_values(ascending=False)
        .head(10)
        .apply(lambda d: d / one_hour)
        .rename("gap_hours")
    )

    return {
        "n_rows": len(df),
        "start": idx[0],
        "end": idx[-1],
        "is_monotonic_increasing": bool(idx.is_monotonic_increasing),
        "has_duplicates": bool(idx.has_duplicates),
        "n_duplicates": int(idx.duplicated().sum()),
        "all_on_the_hour": bool((idx.minute == 0).all() and (idx.second == 0).all()),
        "expected_rows_if_no_gaps": expected_full_grid,
        "n_missing_hours": expected_full_grid - len(df),
        "coverage_pct": round(100 * len(df) / expected_full_grid, 3),
        "n_gaps": int((deltas > one_hour).sum()),
        "max_gap_hours": float(gaps.max() / one_hour) if len(gaps) else 0.0,
        "top_gaps": top_gaps,
    }


# ----------------------------------------------------------------------------
# Missing-value audit
# ----------------------------------------------------------------------------
def missing_audit(split: Split) -> pd.DataFrame:
    """Missing count per feature in each split part (rows = features)."""
    out = {}
    for part, frame in split.parts.items():
        out[f"{part}_missing"] = frame[C.FEATURES].isna().sum()
    res = pd.DataFrame(out)
    res.index.name = "feature"
    res["total_missing"] = res.sum(axis=1)
    return res.reset_index()


# ----------------------------------------------------------------------------
# Class-distribution audit
# ----------------------------------------------------------------------------
def class_distribution(split: Split) -> pd.DataFrame:
    """Per-split class counts + shares + extreme ratio for one threshold."""
    rows = []
    for part, frame in split.parts.items():
        y = frame[C.COL_LABEL]
        n = len(y)
        counts = {c: int((y == c).sum()) for c in C.CLASSES}
        extreme = counts[-1] + counts[1]
        rows.append({
            "threshold": split.threshold,
            "split": part,
            "n": n,
            "class_-1": counts[-1],
            "class_0": counts[0],
            "class_+1": counts[1],
            "share_-1": counts[-1] / n,
            "share_0": counts[0] / n,
            "share_+1": counts[1] / n,
            "extreme_ratio": extreme / n,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Label-integrity audit
# ----------------------------------------------------------------------------
def label_integrity(df: pd.DataFrame, threshold: float) -> dict:
    """Check labels are consistent with the raw forward path statistics.

    First-touch semantics recoverable from magnitudes alone:
      * label == +1  requires the upper barrier was reachable: max_up_1h >= thr
      * label == -1  requires the lower barrier was reachable: max_dn_1h <= -thr
      * label ==  0  requires that NOT exactly one barrier was touched.

    Ordering (which side was hit *first*) is not recoverable from magnitudes, so
    a +1 with max_dn also beyond -thr is allowed (upper simply came first).

    Same-minute double touch is a documented, legitimate label-0 case: if both
    barriers are exceeded yet the label is 0, the two touches must have occurred
    in the *same* 1-minute bar (otherwise the earlier one would win a directional
    label). We count these separately as `same_minute_double_touch`, not as
    violations. A real flat violation is label 0 with *exactly one* barrier hit.
    """
    up, dn, y = df["max_up_1h"], df["max_dn_1h"], df[C.COL_LABEL]
    up_hit = up >= threshold
    dn_hit = dn <= -threshold

    viol_pos = int(((y == 1) & ~up_hit).sum())
    viol_neg = int(((y == -1) & ~dn_hit).sum())
    # exactly one side touched but labelled flat -> genuine inconsistency
    viol_flat = int(((y == 0) & (up_hit ^ dn_hit)).sum())
    # both sides touched but labelled flat -> legitimate same-minute double touch
    double_touch = int(((y == 0) & up_hit & dn_hit).sum())

    return {
        "threshold": threshold,
        "n": len(df),
        "violations_pos_label_no_up_touch": viol_pos,
        "violations_neg_label_no_dn_touch": viol_neg,
        "violations_flat_exactly_one_touch": viol_flat,
        "total_violations": viol_pos + viol_neg + viol_flat,
        "same_minute_double_touch": double_touch,
    }
