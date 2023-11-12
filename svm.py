import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from config import random_state


def basic_svm_fit(partition: dict | pd.DataFrame, c=None, sigma=None) -> tuple[float, float, float, float]:
    def _svm(c=None, sigma=None):
        if sigma is not None:
            gamma = 1 / (sigma ** 2)
            svm = SVC(C=c, kernel='rbf', random_state=random_state, gamma=gamma)
        else:
            svm = SVC(kernel='rbf', random_state=random_state)
        return svm

    def _fit(svm, train_x, train_y, test_x, test_y):
        svm.fit(train_x, train_y)
        pred_y = svm.predict(test_x)
        acc = accuracy_score(test_y, pred_y)
        tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(test_y, pred_y)
        return acc, sensitivity, specificity, auc

    if partition.get('train') is None or partition.get('test') is None:
        # 10-CV
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
        avg_ca, avg_sensitivity, avg_specificity, avg_auc = [], [], [], []
        for train_index, test_index in kf.split(partition):
            # Split the dataset into the current train and test parts for this fold
            X_train, X_test = partition.iloc[train_index, :-1], partition.iloc[test_index,
                                                                :-1]  # assuming last column is target
            y_train, y_test = partition.iloc[train_index, -1], partition.iloc[test_index, -1]
            svm = _svm(c, sigma)

            result = _fit(svm, X_train, y_train, X_test, y_test)
            avg_ca.append(result[0])
            avg_sensitivity.append(result[1])
            avg_specificity.append(result[2])
            avg_auc.append(result[3])
        return np.mean(avg_ca), np.mean(avg_sensitivity), np.mean(avg_specificity), np.mean(avg_auc)

    train = partition['train']
    test = partition['test']
    # drop the last column, -1
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]
    svm = _svm(c, sigma)
    return _fit(svm, train_x, train_y, test_x, test_y)
