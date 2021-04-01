"""File that runs the experiments."""
import os

import matplotlib.pyplot as plt
import numpy as np

from ..datasets import available_datasets
from ..methods import available_methods
from .SizeBenchmark import SizeBenchmark
from .StrideBenchmark import StrideBenchmark
from .SparsityBenchmark import SparsityBenchmark
from .TrainSizeBenchmark import TrainSizeBenchmark
from .NumAtomsBenchmark import NumAtomsBenchmark


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'figure.figsize': (8, 4.8),
})


def run(args):
    """Run the desired experiment."""
    method = available_methods[args.m]
    method_params = {
        'n_components': args.atoms,
        'alpha': 1,
        'verbose': 1,
        'random_state': 0,
        'n_jobs': 4,
        'max_iter': args.iter,
    }
    method = method(args.w, args.s, **method_params)
    ds = available_datasets[args.ds]()

    os.makedirs('figs', exist_ok=True)
    suffix = f'w_{args.w}-s_{args.s}-splits_{args.splits}-ds_{ds.__class__.__name__}-dist_{args.dist}-iter_{args.iter}'

    # Figure 1
    if args.action == 'qcr':
        bm = SparsityBenchmark(method, ds.timeseries, n_splits=args.splits)
        bm.plot_quality_vs_cr(5, n_atoms=5, dist=args.dist)
        plt.savefig(f'figs/quality_vs_cr-{suffix}.pdf', bbox_inches='tight')

    # Figure 2
    elif args.action == 'widths':
        bm = SizeBenchmark(method, ds.timeseries)
        bm.plot_quality_vs_width(widths=[1, 5, 10, 15, 30, 50, 75, 100], stride=10, n_atoms=10, dist=args.dist)
        plt.savefig(f'figs/quality_vs_widths-{suffix}.pdf', bbox_inches='tight')

    # Figure 2
    elif args.action == 'strides':
        bm = StrideBenchmark(method, ds.timeseries)
        bm.plot_quality_vs_stride(strides=[1, 5, 10, 50], n_atoms=10, dist=args.dist)
        plt.savefig(f'figs/quality_vs_strides-{suffix}.pdf', bbox_inches='tight')

    # Figure 3
    elif args.action == 'atoms':
        bm = NumAtomsBenchmark(method, ds.timeseries)
        bm.plot_quality_vs_numatoms(n_atoms=[1, 5, 10, 50, 100], dist=args.dist)
        plt.savefig(f'figs/quality_vs_atoms-{suffix}.pdf', bbox_inches='tight')

    # Figure 4
    elif args.action == 'qts':
        bm = TrainSizeBenchmark(method, ds.timeseries, n_splits=args.splits)
        bm.plot_quality_vs_trainsize(dist=args.dist)
        plt.savefig(f'figs/quality_vs_trainsize-{suffix}.pdf', bbox_inches='tight')

    else:
        raise ValueError(f'Unkown action {args.action}')

    plt.savefig('figs/last_figure.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()
