"""Dataset loading and chronological splitting.

The only place that reads the processed parquet files and that decides where the
train/validation/test boundaries fall. Everything downstream consumes `Split`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C


@dataclass(frozen=True)
class Split:
    """A chronological train/validation/test split of one dataset.

    Each attribute is a DataFrame carrying every column of the source dataset
    (features + close + forward stats + label). Use the helper properties to pull
    the model matrix `X` and target `y`.
    """

    threshold: float
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    # -- convenience accessors -------------------------------------------------
    def X(self, part: str) -> pd.DataFrame:
        return getattr(self, part)[C.FEATURES]

    def y(self, part: str) -> pd.Series:
        return getattr(self, part)[C.COL_LABEL]

    @property
    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def boundaries(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        """First/last timestamp of each part — for reporting the split dates."""
        return {k: (v.index[0], v.index[-1]) for k, v in self.parts.items()}


def load_dataset(threshold: float = C.MAIN_THRESHOLD) -> pd.DataFrame:
    """Load one modelling dataset, time-sorted, indexed by hourly timestamp `ts`."""
    df = pd.read_parquet(C.dataset_path(threshold))
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def chronological_split(
    df: pd.DataFrame,
    threshold: float = C.MAIN_THRESHOLD,
    train_frac: float = C.SPLIT_TRAIN,
    val_frac: float = C.SPLIT_VAL,
) -> Split:
    """Split a time-ordered dataset into train/val/test by position (no shuffle).

    Boundaries are index positions, so no row is ever shared across parts and the
    test period is strictly the most recent slice of real future data.
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return Split(
        threshold=threshold,
        train=df.iloc[:i_train],
        val=df.iloc[i_train:i_val],
        test=df.iloc[i_val:],
    )


def load_split(threshold: float = C.MAIN_THRESHOLD) -> Split:
    """Load and chronologically split in one call — the common entry point."""
    return chronological_split(load_dataset(threshold), threshold=threshold)
