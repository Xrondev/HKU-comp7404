from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC

from config import random_state


def basic_svm_fit(partition: dict, c=None, sigma=None) -> tuple[float, float, float, float]:
    train = partition['train']
    test = partition['test']
    # drop the last column, -1
    train_x = train.iloc[:, :-1]
    train_y = train.iloc[:, -1]
    test_x = test.iloc[:, :-1]
    test_y = test.iloc[:, -1]
    if sigma is not None:
        gamma = 1 / (sigma ** 2)
        svm = SVC(C=c, kernel='rbf', random_state=random_state, gamma=gamma)
    else:
        svm = SVC(kernel='rbf', random_state=random_state)
    svm.fit(train_x, train_y)
    pred_y = svm.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(test_y, pred_y)
    return acc, sensitivity, specificity, auc
