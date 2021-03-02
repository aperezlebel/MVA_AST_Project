import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import DictionaryLearning
from joblib import Memory

from datasets import BTCDataset


memory = Memory('joblib_cache/', verbose=0)

ds = BTCDataset()
print(ds.X)

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

# Dictionnary learning
@memory.cache
def dict_learning(X_train, s, w):
    n_h = X_train.shape[0]

    c = int((n_h - w)/s + 1)

    X_train = np.array(X_train).reshape(-1, 1)

    Xs = []
    for k in range(c):
        i = w + k*s
        x = X_train[i-w:i]
        Xs.append(x)

    X_h = np.concatenate(Xs, axis=1)

    print(X_h)
    print(X_h.shape)

    dl = DictionaryLearning(n_components=10, alpha=1, verbose=2, random_state=0, n_jobs=4, max_iter=10)
    dl.fit(X_h.T)

    return dl


dl = dict_learning(X_train, s, w)

print(dl.components_)
print(dl.components_.shape)

D = dl.components_.T
