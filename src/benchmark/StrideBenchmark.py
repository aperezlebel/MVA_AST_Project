"""Implement the StrideBenchmark class."""
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


class StrideBenchmark(BaseBenchmark):
    """Implement functions to benchmark influence of the stride."""

    @staticmethod
    def quality_vs_stride(X_train, X_test, method, strides, n_atoms, dist='dtw'):
        method = clone(method)

        # @memory.cache
        def fit(stride):
            method.set_params(**{
                'stride': stride,
            })
            method.estimator.set_params(**{
                'n_components': n_atoms,
                'verbose': 0,
            })
            method.fit(X_train)
            X_pred_codes = method.transform_codes(X_test)
            X_pred = method.codes_to_signal(X_pred_codes)
            rate = BaseBenchmark.compression_rate(X_test, X_pred_codes)
            inv_rate = 1/rate

            return X_pred, rate, inv_rate

        dists = []
        rates = []
        inv_rates = []
        for stride in strides:
            X_pred, rate, inv_rate = fit(stride)

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
            inv_rates.append(inv_rate)

        return dists, rates, inv_rates


    ############ Plotting functions ############

    def plot_quality_vs_stride(self, strides, n_atoms, dist='dtw', ax=None):
        f = self.quality_vs_stride
        res = self.cross_val_wrapper(f, self.method, strides, n_atoms, dist=dist)
        agg = self.aggregator(res)

        dists_avg, dists_std = agg[0]
        rates_avg, rates_std = agg[1]
        inv_rates_avg, inv_rates_std = agg[2]

        ax = self.get_or_create_ax(ax)
        twinx = ax.twinx()

        ax.plot(strides, dists_avg, color='tab:blue')
        ax.fill_between(strides, np.maximum(dists_avg-2*dists_std, 0), dists_avg+2*dists_std,
                        color='tab:blue', alpha=0.3)

        # twinx.plot(widths, rates_avg, color='tab:orange')
        # twinx.fill_between(widths, np.maximum(0, rates_avg-2*rates_std), rates_avg+2*rates_std,
        #                    color='tab:orange', alpha=0.3)

        twinx.plot(strides, inv_rates_avg, color='tab:orange')
        twinx.fill_between(strides, np.maximum(0, inv_rates_avg-2*inv_rates_std), inv_rates_avg+2*inv_rates_std,
                           color='tab:orange', alpha=0.3)

        ax.set_xlabel(r'Stride $s$')
        ax.set_ylabel(f'{dist.upper()}')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        twinx.set_ylabel('Compression rate')
        twinx.tick_params(axis='y', labelcolor='tab:orange')