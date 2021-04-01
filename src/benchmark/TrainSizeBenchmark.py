"""Implement the TrainSparsityBenchmark class."""
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


class TrainSizeBenchmark(BaseBenchmark):
    """Implement functions to benchmark influence of train size on quality."""

    @staticmethod
    def quality_vs_trainsize(X_train, X_test, method, dist='dtw'):
        method = clone(method)

        # @memory.cache
        def fit():
            method.estimator.set_params(**{
                'verbose': 0,
            })
            method.fit(X_train)
            X_pred_codes = method.transform_codes(X_test)
            X_pred = method.codes_to_signal(X_pred_codes)
            rate = BaseBenchmark.compression_rate(X_test, X_pred_codes)
            inv_rate = 1/rate

            return X_pred, rate, inv_rate

        X_pred, rate, inv_rate = fit()

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

        return X_train.shape[0], d, rate, inv_rate

    ############ Plotting functions ############
    def plot_quality_vs_trainsize(self, dist='dtw', ax=None):
        f = self.quality_vs_trainsize
        res = self.cross_val_wrapper(f, self.method, dist=dist)
        agg = self.aggregator(res, avg_on_fold=False)

        sizes, _ = agg[0]
        dists_avg, dists_std = agg[1]
        rates_avg, rates_std = agg[2]
        inv_rates_avg, inv_rates_std = agg[3]

        # res = self.cross_val_wrapper(f, self.method, sparse_levels, n_atoms, dist=dist)
        # agg = self.aggregator(res)

        # dists_avg, dists_std = agg[0]
        # rates_avg, rates_std = agg[1]
        # inv_rates_avg, inv_rates_std = agg[2]

        ax = self.get_or_create_ax(ax)
        # twinx = ax.twinx()

        ax.plot(sizes, dists_avg, color='tab:blue')
        ax.fill_between(sizes, np.maximum(dists_avg-2*dists_std, 0), dists_avg+2*dists_std,
                        color='tab:blue', alpha=0.3)

        # twinx.plot(sparse_levels, rates_avg, color='tab:orange')
        # twinx.fill_between(sparse_levels, np.maximum(0, rates_avg-2*rates_std), rates_avg+2*rates_std,
        #                    color='tab:orange', alpha=0.3)

        # twinx.plot(sizes, inv_rates_avg, color='tab:orange')
        # twinx.fill_between(sizes, np.maximum(0, inv_rates_avg-2*inv_rates_std), inv_rates_avg+2*inv_rates_std,
        #                    color='tab:orange', alpha=0.3)

        ax.set_xlabel(r'Size of the training set')
        ax.set_ylabel(f'{dist.upper()}')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        # twinx.set_ylabel('Compression rate')
        # twinx.tick_params(axis='y', labelcolor='tab:orange')
