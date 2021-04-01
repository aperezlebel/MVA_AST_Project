"""Implement the Benchmark class."""
import sys
from abc import ABC
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from ..methods import BaseMethod

memory = Memory('joblib_cache/', verbose=0)


class BaseBenchmark(ABC):
    """Implement functions to benchmark methods."""

    def __init__(self, method, timeseries_list, n_splits=5, cv=None):
        assert isinstance(method, BaseMethod)

        self.method = method
        if not isinstance(timeseries_list, list):
            timeseries_list = [timeseries_list]
        self.timeseries_list = timeseries_list
        self.cv = cv
        self.n_splits = n_splits

    def data_split(self, timeseries):
        n_splits = self.n_splits
        cv = TimeSeriesSplit(n_splits=n_splits) if self.cv is None else self.cv
        cv_split = cv.split(timeseries)

        for train_idx, test_idx in cv_split:
            yield timeseries[train_idx], timeseries[test_idx]

    def cross_val_wrapper(self, f, *args, **kwargs):
        r1 = defaultdict(list)

        f = memory.cache(f)

        for timeseries in tqdm(self.timeseries_list):
            r2 = defaultdict(list)

            for X_train, X_test in tqdm(list(self.data_split(timeseries)), leave=False):
                res = f(X_train, X_test, *args, **kwargs)

                if not isinstance(res, tuple):
                    res = (res,)

                for i in range(len(res)):
                    r2[i].append(res[i])

            for k, v in r2.items():
                r1[k].append(v)

        for k, v in r1.items():
            r1[k] = np.array(v)

        return tuple(r1.values())

    @staticmethod
    def aggregator(res, avg_on_fold=True):

        def aggregate(r):
            if avg_on_fold:
                r = np.mean(r, axis=1)  # Average over folds

            avg = np.mean(r, axis=0)  # Average over time series
            std = np.std(r, axis=0)  # Std over time series

            return avg, std

        if not isinstance(res, tuple):
            return aggregate(res)

        aggregates = []
        for r in res:
            aggregates.append(aggregate(r))

        return tuple(aggregates)

    @staticmethod
    def size_of(x):
        return sys.getsizeof(x)

    @staticmethod
    def codes_to_compressed_data(X_codes):
        compressed_data = X_codes[~np.isclose(X_codes, 0)]
        return compressed_data

    @staticmethod
    def compression_rate(X_test, X_pred_codes):
        uncompressed_size = X_test.size*X_test.itemsize
        n_nonzero = np.sum(~np.isclose(X_pred_codes, 0))
        compressed_size = n_nonzero*X_pred_codes.itemsize
        ratio = uncompressed_size/compressed_size

        return ratio

    @staticmethod
    def get_or_create_ax(ax=None):
        if ax is None:
            _, ax = plt.subplots()

        return ax
