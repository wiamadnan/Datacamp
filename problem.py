import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import rampwf as rw

problem_title = "Purchasing Intention prediction for online shoppers"


_target_column_name = "Revenue"
_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3)
]


def get_train_data(path="."):
    data = pd.read_csv(os.path.join(path, "data","train.csv"))
    y_train = data[_target_column_name].values
    X_train = data.drop(_target_column_name, axis=1)
    return X_train, y_train

def get_test_data(path="."):
    data = pd.read_csv(os.path.join(path, "data", "test.csv"))
    y_test = data[_target_column_name].values
    X_test = data.drop(_target_column_name, axis=1)
    return X_test, y_test


def get_cv(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle = True, random_state=42)
    return cv.split(X, y)



