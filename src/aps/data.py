"""Daily dataset loading and chronological splitting (regression).

The one place that reads the assembled dataset and fixes the train/val/test
boundaries. `X`/`y` pull the feature matrix and the next-day-log-return target.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C
from . import datasets


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def X(self, part: str) -> pd.DataFrame:
        return getattr(self, part)[C.FEATURES]

    def y(self, part: str) -> pd.Series:
        return getattr(self, part)[C.TARGET]

    def close(self, part: str) -> pd.Series:
        return getattr(self, part)[C.COL_CLOSE]

    @property
    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def boundaries(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {k: (v.index[0], v.index[-1]) for k, v in self.parts.items()}


def load_dataset(rebuild: bool = False) -> pd.DataFrame:
    """Load the assembled daily dataset, building it on first use."""
    path = C.DATA_PROCESSED / "btc_daily_dataset.parquet"
    if rebuild or not path.exists():
        return datasets.assemble(save=True)
    df = pd.read_parquet(path)
    return df.sort_index()


def chronological_split(df: pd.DataFrame,
                        train_frac: float = C.SPLIT_TRAIN,
                        val_frac: float = C.SPLIT_VAL) -> Split:
    """Position-based split (no shuffle); test is the most recent slice."""
    n = len(df)
    i_tr = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return Split(train=df.iloc[:i_tr], val=df.iloc[i_tr:i_val], test=df.iloc[i_val:])


def load_split(rebuild: bool = False) -> Split:
    return chronological_split(load_dataset(rebuild=rebuild))
