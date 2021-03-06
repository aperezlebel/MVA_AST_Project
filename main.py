import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import DictionaryLearning
from joblib import Memory

from datasets import BTCDataset


memory = Memory('joblib_cache/', verbose=0)


def window_split(X, s, w):
    """From a signal, create an array of overlapping windows."""
    X = np.array(X).reshape(-1, 1)

    n_h = X.shape[0]
    c = int((n_h - w)/s + 1)

    Xs = []
    for k in range(c):
        i = w + k*s
        x = X[i-w:i]
        Xs.append(x)

    return np.concatenate(Xs, axis=1)


def window_merge(X_h, s):
    """From an array of overlapping windows, reconstruct the original signal.

    Parameters:
    -----------
        X_h : np.array of shape (w, c)
            Array of overlapping windows.
        s : int
            Stride

    Returns:
    --------
        X : np.array of shape

    """
    w, c = X_h.shape
    W = np.zeros((c, w+s*(c-1)))

    for i in range(c):
        W[i, i*s:i*s+w] = X_h[:, i]

    N = np.sum(W != 0, axis=0)
    x_hat = np.divide(np.sum(W, axis=0), N)
    return x_hat

# Dictionnary learning
@memory.cache
def dict_learning(X_h, **kwargs):
    dl = DictionaryLearning(**kwargs, verbose=2)
    dl.fit(X_h)
    return dl


if __name__ == '__main__':
    ds = BTCDataset()
    # print(ds.X)

    # ds.X.plot()
    # plt.yscale('log')
    # plt.show()

    cv = TimeSeriesSplit(n_splits=2, test_size=10000)
    cv_split = cv.split(ds.X)
    next(cv_split)
    train_idx, test_idx = next(cv_split)

    X_train = ds.X[train_idx]
    X_test = ds.X[test_idx]

    # X_train.plot()
    # X_test.plot()
    # plt.show()
    s = 12
    w = 24

    X_h = window_split(X_train, s, w)
    dl = dict_learning(X_h.T, n_components=10, alpha=1,
                       random_state=0, n_jobs=4, max_iter=10)

    # Retrieve dictionnary
    D = dl.components_.T

    # print(dl.components_)
    # print(dl.components_.shape)

    # D = dl.components_.T

    X_h_test = window_split(X_test, s, w)
    X_pred_codes = dl.transform(X_h_test.T).T

    X_h_pred = D@X_pred_codes

    print(X_h_pred)
    print(X_h_pred.shape, X_h_test.shape)

    X_pred_data = window_merge(X_h_pred, s=s)

    print(X_test.shape, X_pred_data.shape)

    # print(X_test)

    X_test = X_test[:X_pred_data.shape[0]]
    X_pred = pd.Series(X_pred_data, index=X_test.index)

    X_pred.plot()
    X_test.plot()
    # plt.yscale('log')
    plt.show()
