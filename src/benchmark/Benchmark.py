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


class Benchmark():
    """Implement functions to benchmark methods."""

    def __init__(self, method, timeseries_list, cv=None):
        assert isinstance(method, BaseMethod)

        self.method = method
        if not isinstance(timeseries_list, list):
            timeseries_list = [timeseries_list]
        self.timeseries_list = timeseries_list
        self.cv = cv

    def data_split(self, timeseries):
        n_splits = 5
        cv = TimeSeriesSplit(n_splits=n_splits) if self.cv is None else self.cv
        cv_split = cv.split(timeseries)
        # train_idx, test_idx = next(itertools.islice(cv_split, n_splits-1, None))
        for train_idx, test_idx in cv_split:
            yield timeseries[train_idx], timeseries[test_idx]

    def cross_val_wrapper(self, f, *args, **kwargs):
        r1 = defaultdict(list)
        for timeseries in self.timeseries_list:
            r2 = defaultdict(list)

            for X_train, X_test in self.data_split(timeseries):
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
    def aggregator(res):

        def aggregate(r):
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
    def compression_rate(uncompressed_objects, compressed_objects):
        uncompressed_size = np.sum([Benchmark.size_of(x) for x in uncompressed_objects])
        compressed_size = np.sum([Benchmark.size_of(x) for x in compressed_objects])

        return uncompressed_size/compressed_size

    def compression_rate_evolution(self, n_atoms):
        X_train, X_test = self.data_split()
        method = clone(self.method)

        rates = []

        @memory.cache
        def cached_fit(n):
            # method.estimator.set_params(**{'n_components': n})
            method.estimator.set_params(**{'transform_n_nonzero_coefs': n})
            method.fit(X_train)
            X_pred_codes = method.transform_codes(X_test)
            return X_pred_codes

        for n in n_atoms:
            X_pred_codes = cached_fit(n)
            print(X_pred_codes.shape)
            s = np.sum(np.isclose(X_pred_codes, 0))
            tot = np.sum(np.ones_like(X_pred_codes))
            print(f'{s}/{tot}')
            # compressed_data = X_pred_codes[~np.isclose(X_pred_codes, 0)]
            rate = self.compression_rate([X_test], [compressed_data])
            rates.append(rate)

        return rates

    def quality_vs_cr(self, sparse_levels, n_atoms, dist='dtw'):
        X_train, X_test = self.data_split()
        method = clone(self.method)

        @memory.cache
        def cached_fit(cr, n_atoms):
            method.estimator.set_params(**{'transform_n_nonzero_coefs': cr,
                                           'n_components': n_atoms})
            method.fit(X_train)
            X_pred_codes = method.transform_codes(X_test)
            X_pred = method.codes_to_signal(X_pred_codes)
            compressed_data = self.codes_to_compressed_data(X_pred_codes)
            rate = self.compression_rate([X_test], [compressed_data])

            return X_pred, rate

        dists = []
        rates = []
        for level in sparse_levels:
            X_pred, rate = cached_fit(level, n_atoms)
            a1 = np.array(X_test)
            a2 = np.array(X_pred)
            if dist == 'dtw':
                dist = dtw.distance_fast(a1, a2)
            elif dist == 'rmsre':
                dist = np.sqrt(np.mean(np.square(np.divide(a1 - a2, a1))))
            else:
                raise ValueError(f'Unknown distance {dist}')
            dists.append(dist)
            rates.append(rate)

        return dists, rates

    def get_atoms(self, n_atoms):
        X_train, _ = self.data_split()
        method = clone(self.method)
        method.estimator.set_params(**{'n_components': n_atoms})
        method.fit(X_train)
        return method.get_atoms()


    ############ Plotting functions ############
    @staticmethod
    def get_or_create_ax(ax=None):
        if ax is None:
            _, ax = plt.subplots()

        return ax

    def plot_compression_rate_evolution(self, n_atoms, ax=None):
        rates = self.compression_rate_evolution(n_atoms)
        ax = self.get_or_create_ax(ax)
        ax.plot(n_atoms, rates)
        ax.set_xlabel('Number of atoms')
        ax.set_ylabel('Compression rate')

    def plot_quality_vs_cr(self, sparse_levels, n_atoms, dist='dtw', ax=None):
        try:
            sparse_levels = np.linspace(1, n_atoms, sparse_levels).astype(int)
        except AttributeError:
            pass

        dists, rates = self.quality_vs_cr(sparse_levels, n_atoms, dist=dist)
        ax = self.get_or_create_ax(ax)
        twinx = ax.twinx()
        ax.plot(sparse_levels, dists, color='tab:blue')
        twinx.plot(sparse_levels, rates, color='tab:orange')
        ax.set_xlabel(r'Sparsity constraint $\tau$')
        ax.set_ylabel('DTW')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        twinx.set_ylabel('Compression rate')
        twinx.tick_params(axis='y', labelcolor='tab:orange')

    def plot_atoms(self, n_atoms):
        # ax = self.get_or_create_ax(ax)
        fig, ax_arr = plt.subplots(
            nrows=int(np.ceil(n_atoms//3)),
            ncols=3,
            figsize=(20, 3 * (n_atoms // 3 + 1)),
        )

        atoms = self.get_atoms(n_atoms)

        print(atoms.shape)

        for (ind, (component, ax)) in enumerate(
            zip(atoms.T, ax_arr.flatten())
        ):
            ax.plot(component)
            ax.set_xlim(0, component.size)
            ax.set_ylim(-2, 2)
            ax.set_title(f"Component n°{ind+1}")
