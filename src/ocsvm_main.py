#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 18:27:23 2023

@author: nepal
"""

import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, roc_auc_score


# laod training set and validation set
X_train = pd.read_csv("../data10/train_test_open_full/X_train.csv")
X_test = pd.read_csv("../data10/train_test_open_full/X_test.csv")
y_test = pd.read_csv("../data10/train_test_open_full/y_test.csv")
y_train = pd.read_csv("../data10/train_test_open_full/y_train.csv")

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

outliers_fraction = (y_train == -1).sum().mean() / len(y_train)
model = OneClassSVM(nu=outliers_fraction)

print("starting optimization")
start = time.time()
model.fit(X_train)
print(f"optimization elapsed {time.time() - start}")

preds = model.predict(X_test)
# preds = model.test(X_train.values)

for outlier_label in ["extreme_amount", "extreme_duration", "rule_amount"]:
    fpr, tpr, thr = roc_curve(y_test[outlier_label], preds)
    auc = roc_auc_score(y_test[outlier_label], preds)
    # fpr, tpr, thr = roc_curve(y_train[outlier_label], preds)
    plt.plot(fpr, tpr, label=outlier_label)
    print(features)
    print(f"{outlier_label} auc {round(auc*100, 2)}")
plt.plot([0, 1], [0, 1], color="grey", zorder=1)
plt.grid()
plt.legend()
plt.xlabel("False Alarm Rate")  # "False Positive Rate"
plt.ylabel("Hit Rate")  # True Positive Rate
