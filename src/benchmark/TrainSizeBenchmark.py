"""Implement the TrainSizeBenchmark class."""
import numpy as np
from dtaidistance import dtw
from sklearn.base import clone
from tqdm import tqdm

from .BaseBenchmark import BaseBenchmark


class TrainSizeBenchmark(BaseBenchmark):
    """Implement functions to benchmark influence of train size on quality."""

    @staticmethod
    def quality_vs_trainsize(X_train, X_test, method, dist='dtw'):
        method = clone(method)

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

    def plot_quality_vs_trainsize(self, dist='dtw', ax=None):
        f = self.quality_vs_trainsize
        res = self.cross_val_wrapper(f, self.method, dist=dist)
        agg = self.aggregator(res, avg_on_fold=False)

        n = agg[0][0].shape[0]
        sizes = np.cumsum(np.ones(n)/n)
        dists_avg, dists_std = agg[1]
        rates_avg, rates_std = agg[2]
        inv_rates_avg, inv_rates_std = agg[3]

        ax = self.get_or_create_ax(ax)

        ax.plot(sizes, dists_avg, color='tab:blue', label='Mean')
        ax.fill_between(sizes, np.maximum(dists_avg-2*dists_std, 0), dists_avg+2*dists_std,
                        color='tab:blue', alpha=0.15, label='$[(\mu-2\sigma)^{+}, \mu+2\sigma]$')

        ax.set_xlabel(r'Relative size of the training set')
        ax.set_ylabel(f'{dist.upper()}')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        ax.legend(loc='upper left')
