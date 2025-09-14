from typing import Optional

import numpy as np
from sklearn.model_selection import KFold


class PurgedTimeSeriesSplit(KFold):
    """
    Extend KFold to work with labels that are not i.i.d.
    The train set is purged of observations overlapping in time with the test set.
    The test set is embargoed so that it does not influence the training set.
    """

    def __init__(
        self,
        n_splits: int = 10,
        t1: Optional[np.ndarray] = None,
        pct_embargo: float = 0.01,
    ):
        super(PurgedTimeSeriesSplit, self).__init__(n_splits, shuffle=False, random_state=None)

        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if self.t1 is None:
            self.t1 = np.arange(X.shape[0])

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for i, j in test_starts:
            t0 = self.t1[i]  # start of test set
            test_indices = indices[i:j]
            max_t1_idx = self.t1[test_indices].max()
            train_indices = np.where(self.t1 <= t0)[0]
            if train_indices.shape[0] > 0:
                yield train_indices, test_indices
