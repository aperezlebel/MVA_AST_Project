"""Implement the Benchmark class."""
import sys
import numpy as np
from sklearn.base import clone
import itertools
from sklearn.model_selection import TimeSeriesSplit
from joblib import Memory
import matplotlib.pyplot as plt
from dtaidistance import dtw

from ..methods import BaseMethod


memory = Memory('joblib_cache/', verbose=0)


class Benchmark():
    """Implement functions to benchmark methods."""

    def __init__(self, method, timeseries, cv=None):
        assert isinstance(method, BaseMethod)

        self.method = method
        self.timeseries = timeseries
        self.cv = cv

    def data_split(self):
        n_splits = 5
        cv = TimeSeriesSplit(n_splits=n_splits) if self.cv is None else self.cv
        cv_split = cv.split(self.timeseries)
        train_idx, test_idx = next(itertools.islice(cv_split, n_splits-1, None))
        return self.timeseries[train_idx], self.timeseries[test_idx]

    @staticmethod
    def size_of(x):
        return sys.getsizeof(x)

    @staticmethod
    def compression_rate(uncompressed_objects, compressed_objects):
        uncompressed_size = np.sum([Benchmark.size_of(x) for x in uncompressed_objects])
        compressed_size = np.sum([Benchmark.size_of(x) for x in compressed_objects])

        return compressed_size/uncompressed_size

    def compression_rate_evolution(self, n_atoms):
        X_train, X_test = self.data_split()
        method = clone(self.method)

        rates = []

        @memory.cache
        def cached_fit(n):
            # method.estimator.set_params(**{'n_components': n})
            method.estimator.set_params(**{'transform_n_nonzero_coefs': n})
            method.fit(X_train)
            X_pred_codes = method.transform_code(X_test)
            return X_pred_codes

        for n in n_atoms:
            X_pred_codes = cached_fit(n)
            print(X_pred_codes.shape)
            s = np.sum(np.isclose(X_pred_codes, 0))
            tot = np.sum(np.ones_like(X_pred_codes))
            print(f'{s}/{tot}')
            compressed_data = X_pred_codes[~np.isclose(X_pred_codes, 0)]
            rate = self.compression_rate([X_test], [compressed_data])
            rates.append(rate)

        return rates

    def quality_vs_cr(self, compression_rates, n_atoms):
        X_train, X_test = self.data_split()
        method = clone(self.method)

        @memory.cache
        def cached_fit(cr, n_atoms):
            method.estimator.set_params(**{'transform_n_nonzero_coefs': cr,
                                           'n_components': n_atoms})
            method.fit(X_train)
            X_pred = method.transform(X_test)
            return X_pred

        dists = []
        for cr in compression_rates:
            X_pred = cached_fit(cr, n_atoms)
            DTW = dtw.distance_fast(np.array(X_test), np.array(X_pred))
            dists.append(DTW)

        return dists

    ############ Plotting functions ############
    @staticmethod
    def get_or_create_ax(ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        return ax

    def plot_compression_rate_evolution(self, n_atoms, ax=None):
        rates = self.compression_rate_evolution(n_atoms)
        ax = self.get_or_create_ax(ax)
        ax.plot(n_atoms, rates)
        ax.set_xlabel('Number of atoms')
        ax.set_ylabel('Compression rate')

    def plot_quality_vs_cr(self, compression_rates, n_atoms, ax=None):
        try:
            compression_rates = np.linspace(1, n_atoms, compression_rates).astype(int)
        except AttributeError:
            pass

        dists = self.quality_vs_cr(compression_rates, n_atoms)
        ax = self.get_or_create_ax(ax)
        ax.plot(compression_rates, dists)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('DTW')
