"""Implement the Sparsity class."""
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

from ..methods import BaseMethod
from .BaseBenchmark import BaseBenchmark


memory = Memory('joblib_cache/', verbose=0)


class SparsityBenchmark(BaseBenchmark):
    """Implement functions to benchmark influence sparsity on the results."""

    # @staticmethod
    # def size_of(x):
    #     return sys.getsizeof(x)

    # @staticmethod
    # def codes_to_compressed_data(X_codes):
    #     compressed_data = X_codes[~np.isclose(X_codes, 0)]
    #     return compressed_data

    # @staticmethod
    # def compression_rate(uncompressed_objects, compressed_objects):
    #     uncompressed_size = np.sum([Benchmark.size_of(x) for x in uncompressed_objects])
    #     compressed_size = np.sum([Benchmark.size_of(x) for x in compressed_objects])

    #     return uncompressed_size/compressed_size

    # @staticmethod
    # def compression_rate_evolution(X_train, X_test, method, n_atoms):
    #     method = clone(method)

    #     rates = []

    #     # @memory.cache
    #     def cached_fit(n):
    #         method.estimator.set_params(**{'n_components': n})
    #         method.fit(X_train)
    #         X_pred_codes = method.transform_codes(X_test)
    #         return X_pred_codes

    #     for n in tqdm(n_atoms, leave=False):
    #         X_pred_codes = cached_fit(n)
    #         compressed_data = Benchmark.codes_to_compressed_data(X_pred_codes)
    #         rate = Benchmark.compression_rate([X_test], [compressed_data])
    #         rates.append(rate)

    #     return rates

    @staticmethod
    def quality_vs_cr(X_train, X_test, method, sparse_levels, n_atoms, dist='dtw'):
        method = clone(method)

        # @memory.cache
        def cached_fit(cr, n_atoms):
            method.estimator.set_params(**{
                'transform_n_nonzero_coefs': cr,
                'n_components': n_atoms,
                'verbose': 0,
            })
            method.fit(X_train)
            X_pred_codes = method.transform_codes(X_test)
            X_pred = method.codes_to_signal(X_pred_codes)
            compressed_data = SparsityBenchmark.codes_to_compressed_data(X_pred_codes)
            rate = SparsityBenchmark.compression_rate([X_test], [compressed_data])

            return X_pred, rate

        dists = []
        rates = []
        for level in tqdm(sparse_levels, leave=False):
            X_pred, rate = cached_fit(level, n_atoms)

            # Must truncate test timeseries to prediction timeseries
            a1 = np.array(X_test)
            a2 = np.array(X_pred)
            assert a1.shape[0] >= a2.shape[0]
            a1.resize(a2.shape)

            # Compute distance
            if dist == 'dtw':
                d = dtw.distance_fast(a1, a2)
            elif dist == 'rmsre':
                d = np.sqrt(np.mean(np.square(np.divide(a1 - a2, a1))))
            else:
                raise ValueError(f'Unknown distance {dist}')

            dists.append(d)
            rates.append(rate)

        return dists, rates


    ############ Plotting functions ############

    # def plot_compression_rate_evolution(self, n_atoms, ax=None):
    #     f = self.compression_rate_evolution
    #     res = self.cross_val_wrapper(f, self.method, n_atoms)
    #     agg = self.aggregator(res)

    #     rates_avg, rates_std = agg[0]

    #     ax = self.get_or_create_ax(ax)

    #     ax.plot(n_atoms, rates_avg, color='tab:blue')
    #     ax.fill_between(n_atoms, np.maximum(rates_avg-2*rates_std, 0),
    #                     rates_avg+2*rates_std, color='tab:blue', alpha=0.3)

    #     ax.set_xlabel('Number of atoms')
    #     ax.set_ylabel('Compression rate')

    def plot_quality_vs_cr(self, sparse_levels, n_atoms, dist='dtw', ax=None):
        try:
            sparse_levels = np.linspace(1, n_atoms, sparse_levels).astype(int)
        except AttributeError:
            pass

        f = self.quality_vs_cr
        res = self.cross_val_wrapper(f, self.method, sparse_levels, n_atoms, dist=dist)
        agg = self.aggregator(res)

        dists_avg, dists_std = agg[0]
        rates_avg, rates_std = agg[1]

        ax = self.get_or_create_ax(ax)
        twinx = ax.twinx()

        ax.plot(sparse_levels, dists_avg, color='tab:blue')
        ax.fill_between(sparse_levels, np.maximum(dists_avg-2*dists_std, 0), dists_avg+2*dists_std,
                        color='tab:blue', alpha=0.3)

        twinx.plot(sparse_levels, rates_avg, color='tab:orange')
        twinx.fill_between(sparse_levels, np.maximum(0, rates_avg-2*rates_std), rates_avg+2*rates_std,
                           color='tab:orange', alpha=0.3)

        ax.set_xlabel(r'Sparsity constraint $\tau$')
        ax.set_ylabel(f'{dist.upper()}')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        twinx.set_ylabel('Compression rate')
        twinx.tick_params(axis='y', labelcolor='tab:orange')
