import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.decomposition import DictionaryLearning
# from joblib import Memory
from dtaidistance import dtw

from datasets import BTCDataset
from methods import DictionaryLearningMethod


if __name__ == '__main__':
    ds = BTCDataset()

    ds.X.plot()
    plt.yscale('log')
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
    s = 12*7
    w = 24*7

    estimator = DictionaryLearningMethod(width=w, stride=s)
    estimator.fit(X_train)
    X_pred_data = estimator.transform(X_test)

    X_test = X_test[:X_pred_data.shape[0]]
    X_pred = pd.Series(X_pred_data, index=X_test.index)

    plt.figure()
    X_pred.plot()
    X_test.plot()
    # plt.yscale('log')

    MSE = np.linalg.norm(X_test.values - X_pred.values)

    DTW = dtw.distance_fast(X_test.values, X_pred.values)
    print(f'MSE: {MSE:.0f}')
    print(f'DTW: {DTW:.0f}')

    plt.show()
