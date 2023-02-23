#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:27:23 2023

@author: nepal
"""

import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score


def record_experiment(preds: np.array, distance: np.array, X_test: pd.DataFrame, model: str):
    date = datetime.now().isoformat()
    file_path = "../output/" + date + "_" + model + ".csv"
    preds_series = pd.Series(preds, index=X_test.index, name="prediction")
    distance_series = pd.Series(distance, index=X_test.index, name="distance")
    preds_distance_df = pd.concat([preds_series, distance_series], axis=1)
    preds_distance_df = preds_distance_df.sort_values("distance")
    preds_distance_df.to_csv(file_path)


def normalize(X: pd.DataFrame) -> pd.array:
    return (X - X.min()) / (X.max() - X.min())


# IF NOT PCA
# laod training set and validation set
X_train = pd.read_csv("../data10/train_test_open/X_train.csv")
X_test = pd.read_csv("../data10/train_test_open/X_test.csv")
y_test = pd.read_csv("../data10/train_test_open/y_test.csv")
y_train = pd.read_csv("../data10/train_test_open/y_train.csv")

features = ['be_duration', 'pa_duration',
            "be_duration_mean", "pa_duration_mean",
            "be_duration_std", "pa_duration_std",
            "be_duration_skewness", "pa_duration_skewness",
            "be_duration_kurtosis", "pa_duration_kurtosis",
            'be_amount', 'pa_amount',
            'be_amount_mean', 'pa_amount_mean',
            'be_amount_std', 'pa_amount_std',
            'be_amount_skewness', 'pa_amount_skewness',
            'be_amount_kurtosis', 'pa_amount_kurtosis',
            "be_med_ann_revenue", "pa_med_ann_expenditure",
            'n_winners',
            'id_lsf', 'id_forma_giuridica', 'uber_forma_giuridica', 'cpv',
            ]
X_train = X_train[features]
X_test = X_test[features]


# If PCA
# X_train = pd.read_csv("../data10/train_test_open_PCA/X_train.csv")
# X_test = pd.read_csv("../data10/train_test_open_PCA/X_test.csv")
# y_test = pd.read_csv("../data10/train_test_open_PCA/y_test.csv")
# y_train = pd.read_csv("../data10/train_test_open_PCA/y_train.csv")
# X_train = X_train.iloc[:, [0, 1]]  # more 99% of explained variance
# X_test = X_test.iloc[:, [0, 1]]
# X_train = normalize(X_train)
# X_test = normalize(X_test)

outliers_fraction = (y_train == -1).sum().mean() / len(y_train)
model = OneClassSVM(nu=outliers_fraction)

print("starting optimization")
start = time.time()
model.fit(X_train)
print(f"optimization elapsed {time.time() - start}")

preds = model.predict(X_test)
distance = model.decision_function(X_test)

# record_experiment(preds, distance, X_test, model="oc-svm")


# def draw_contuours(X: pd.DataFrame, y: pd.DataFrame, model):

#     xx = np.linspace(-.1, 1)
#     yy = np.linspace(-.1, 1)

#     YY, XX = np.meshgrid(xx, yy)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T

#     Z = model.decision_function(xy).reshape(XX.shape)

#     CS = plt.contour(XX, YY, Z)
#     plt.clabel(CS)

#     for i in range(4):
#         X_outliers = X[y.iloc[:, i] == -1]
#         plt.scatter(X_outliers.iloc[:, 0],
#                     X_outliers.iloc[:, 1], label=y.columns[i])
#     plt.legend(title="Type of outliers")


# draw_contuours(X_test, y_test, model)
# plt.savefig("../images/ocsvm", dpi=200)


for outlier_label in ["extreme_amount", "extreme_duration", "rule_amount"]:
    fpr, tpr, thr = roc_curve(y_test[outlier_label], preds)
    auc = roc_auc_score(y_test[outlier_label], preds)
    # fpr, tpr, thr = roc_curve(y_train[outlier_label], preds)
    plt.plot(fpr, tpr, label=outlier_label)
    # print(features)
    print(f"{outlier_label} auc {round(auc*100, 2)}")
plt.plot([0, 1], [0, 1], color="grey", zorder=1)
plt.grid()
plt.legend()
plt.xlabel("False Alarm Rate")  # "False Positive Rate"
plt.ylabel("Hit Rate")  # True Positive Rate
plt.savefig("../images/roc/oc-svm/roc_oc-svm", dpi=200)
