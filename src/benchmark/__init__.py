import matplotlib.pyplot as plt

from .Benchmark import Benchmark
from ..methods import available_methods
from ..datasets import available_datasets


def run(args):
    method = available_methods[args.action]
    method = method(args.w, args.s)

    print(method)

    ds = available_datasets[args.ds]()
    bm = Benchmark(method, ds.X)
    bm.plot_compression_rate_evolution(n_atoms=[1, 2, 3, 4, 5]) #  n_atoms=[3, 5, 10, 20, 100])

    plt.show()
