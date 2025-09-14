from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np


class PurgedTimeSeriesSplit:
    """Time-series split with purge and embargo to avoid leakage.

    Parameters
    -----------
    n_splits: int
        Number of splits.
    embargo: int
        Number of observations to exclude on both sides of the split boundary.
    min_train: int
        Minimum train size required to yield a split.
    """

    def __init__(self, n_splits: int = 3, embargo: int = 5, min_train: int = 50):
        self.n_splits = max(2, int(n_splits))
        self.embargo = max(0, int(embargo))
        self.min_train = max(1, int(min_train))

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        if fold_size <= 0:
            yield np.arange(0, max(self.min_train, n - 1)), np.arange(n - 1, n)
            return
        for i in range(1, self.n_splits + 1):
            end_train = i * fold_size
            start_valid = end_train + self.embargo
            end_valid = min(n, (i + 1) * fold_size)
            if end_train < self.min_train or start_valid >= end_valid:
                continue
            train_idx = np.arange(0, end_train)
            valid_idx = np.arange(start_valid, end_valid)
            yield train_idx, valid_idx

    def get_n_splits(self, *args, **kwargs) -> int:  # sklearn compatibility
        return self.n_splits

