"""Implement the Benchmark class."""
import sys
import numpy as np
from sklearn.base import clone
import itertools
from sklearn.model_selection import TimeSeriesSplit
from joblib import Memory
import matplotlib.pyplot as plt
from dtaidistance import dtw
from collections import defaultdict
from tqdm import tqdm
from abc import ABC

from ..methods import BaseMethod


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    # 'font.size': 15,
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (8,4.8),
})
figsize = (10, 6)
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

    # @staticmethod
    # def compression_rate(uncompressed_objects, compressed_objects):
    #     uncompressed_size = np.sum([BaseBenchmark.size_of(x) for x in uncompressed_objects])
    #     compressed_size = np.sum([BaseBenchmark.size_of(x) for x in compressed_objects])

    #     ratio = uncompressed_size/compressed_size

    #     print(f'Raw: {uncompressed_size}')
    #     print(f'Compressed: {compressed_size}')
    #     print(f'Ratio: {ratio}')
    #     return ratio

    @staticmethod
    def compression_rate(X_test, X_pred_codes):
        uncompressed_size = X_test.size*X_test.itemsize
        n_nonzero = np.sum(~np.isclose(X_pred_codes, 0))
        compressed_size = n_nonzero*X_pred_codes.itemsize
        ratio = uncompressed_size/compressed_size

        # print(f'Raw: {uncompressed_size}')
        # print(f'Compressed: {compressed_size}')
        # print(f'Ratio: {ratio}')
        return ratio

    # def get_atoms(self, n_atoms, timeseries):
    #     X_train, _ = next(self.data_split(timeseries))
    #     method = clone(self.method)
    #     method.estimator.set_params(**{'n_components': n_atoms})
    #     method.fit(X_train)
    #     return method.get_atoms()

    ############ Plotting functions ############
    @staticmethod
    def get_or_create_ax(ax=None):
        if ax is None:
            _, ax = plt.subplots()

        return ax

    # def plot_atoms(self, n_atoms):
    #     fig, ax_arr = plt.subplots(
    #         nrows=int(np.ceil(n_atoms//3)),
    #         ncols=3,
    #         figsize=(20, 3 * (n_atoms // 3 + 1)),
    #     )

    #     atoms = self.get_atoms(n_atoms, self.timeseries_list[0])

    #     print(atoms.shape)

    #     for (ind, (component, ax)) in enumerate(
    #         zip(atoms.T, ax_arr.flatten())
    #     ):
    #         ax.plot(component)
    #         ax.set_xlim(0, component.size)
    #         ax.set_ylim(-2, 2)
    #         ax.set_title(f"Component n°{ind+1}")
