"""Implement the DictionaryLearningMethod."""
import numpy as np
from joblib import Memory
from sklearn.decomposition import DictionaryLearning

from .BaseMethod import BaseMethod


memory = Memory('joblib_cache/', verbose=0)


class DictionaryLearningMethod(BaseMethod):
    """Implement the dict learning method of the paper using sklearn."""

    def __init__(self, width=24, stride=12, n_components=10, alpha=1,
                 verbose=1, random_state=0, n_jobs=4, max_iter=1):
        self.width = width
        self.stride = stride

        self.n_components = n_components
        self.alpha = alpha
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_iter = max_iter

        self.estimator = DictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=max_iter,
        )

    @staticmethod
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

    @staticmethod
    def window_merge(X_h, s):
        """From array of overlapping windows, reconstruct the original signal.

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

    def fit(self, X, y=None):
        X_h = self.window_split(X, self.stride, self.width)
        self.estimator.fit(X_h.T)

    def transform_codes(self, X):
        X_h = self.window_split(X, self.stride, self.width)
        X_pred_codes = self.estimator.transform(X_h.T).T
        return X_pred_codes

    def codes_to_signal(self, X_codes):
        D = self.estimator.components_.T
        X_h = D@X_codes
        X = self.window_merge(X_h, self.stride)
        return X

    def transform(self, X):
        X_pred_codes = self.transform_codes(X)
        X_pred = self.codes_to_signal(X_pred_codes)
        return X_pred

    def get_atoms(self):
        return self.estimator.components_.T
