import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import DictionaryLearning
from joblib import Memory

from datasets import BTCDataset


memory = Memory('joblib_cache/', verbose=0)

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

def window_split(X, s, w):
    X = np.array(X).reshape(-1, 1)

    n_h = X.shape[0]
    c = int((n_h - w)/s + 1)

    Xs = []
    for k in range(c):
        i = w + k*s
        x = X[i-w:i]
        Xs.append(x)

    return np.concatenate(Xs, axis=1)

# Dictionnary learning
@memory.cache
def dict_learning(X_h):
    dl = DictionaryLearning(n_components=10, alpha=1, verbose=2, random_state=0, n_jobs=4, max_iter=10)
    dl.fit(X_h)
    return dl


X_h = window_split(X_train, s, w)
dl = dict_learning(X_h.T)

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
