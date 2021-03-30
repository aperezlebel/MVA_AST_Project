import os
import numpy as np
import matplotlib.pyplot as plt
import quandl

from .Benchmark import Benchmark
from ..methods import available_methods
from ..datasets import available_datasets


def run(args):
    method = available_methods[args.m]
    method = method(args.w, args.s)

    ds = available_datasets[args.ds]()
    bm = Benchmark(method, [ds.X], n_splits=2)#, ds.X])
    os.makedirs('figs', exist_ok=True)

    suffix = f'w_{args.w}-s_{args.s}'

    if args.action == 'plot-qcr':
        bm.plot_quality_vs_cr(3, n_atoms=10, dist='rmsre')  #  n_atoms=[3, 5, 10, 20, 100])
        plt.savefig(f'figs/quality_vs_cr-{suffix}.pdf', bbox_inches='tight')

    elif args.action == 'plot-atoms':
        n_atoms = 30
        bm.plot_atoms(n_atoms=n_atoms)
        plt.savefig(f'figs/atoms-{suffix}-n_{n_atoms}.pdf', bbox_inches='tight')

    elif args.action == 'test':
        def f(X_train, X_test):
            return np.ones(4), 2, 3*np.ones(4)
        res = bm.cross_val_wrapper(f)
        res_ag = bm.aggregator(res)
        print(res)
        print(res_ag)

    elif args.action == 'quandl':
        quandl.ApiConfig.api_key = 'your_api_key'
        # res = quandl.get('WIKI/PRICES')
        # res = quandl.get_table('ZACKS/FC', ticker=['AAPL', 'MSFT'], qopts={'columns':['ticker', 'date', 'close'] })
        res = quandl.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']}, ticker=['GOOGL'], date={'gte': '2000-01-01', 'lte': '2020-01-01'})
        # res = quandl.export_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close']}, ticker=['AAPL', 'MSFT', 'GOOGL'], date={'gte': '2000-01-01', 'lte': '2020-01-01'})
        print(res)
        print(res.shape)

        # print(type(res))
        res.plot('date', 'close')


    else:
        raise ValueError(f'Unkown action {args.action}')

    plt.savefig('figs/last_figure.pdf', bbox_inches='tight')
    # plt.tight_layout()
    plt.show()
