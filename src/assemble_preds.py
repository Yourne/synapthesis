#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:41:27 2023

@author: nepal
"""
import pandas as pd

contracts = pd.read_csv("../data10/contracts.csv", index_col="index")
contracts = contracts.drop(
    columns=['be_med_ann_revenue', 'pa_med_ann_expenditure', 'n_winners',
             'id_lsf', 'id_forma_giuridica', 'uber_forma_giuridica', 'cpv',
             "id_award_procedure", 'be_std', 'amount',
             'be_skewness', 'be_kurtosis', 'pa_std', 'pa_skewness', 'pa_kurtosis'])
X_test = pd.read_csv("../data10/train_test_open_full/X_test.csv")
y_test = pd.read_csv("../data10/train_test_open_full/y_test.csv")

test_set = pd.concat([X_test, y_test], axis=1)
test_set = test_set.join(contracts)

gmm_preds = pd.read_csv(
    "../output/2023-01-31T15:04:19.454163_gmm_normed.csv", index_col=0)
Hpi_preds = pd.read_csv(
    "../output/2023-01-31 15:38:19_Hpi_aperta.csv", index_col=0)
Hpi_preds = Hpi_preds.rename(
    columns={"predict.fhat.Hpi..x...X_test.": "estimate"})
svm_preds = pd.read_csv(
    "../output/2023-01-30T17:45:32.382519_oc-svm.csv", index_col=0)

gmm_preds_extended = gmm_preds.join(test_set).sort_values(by="prediction")
Hpi_preds_extended = Hpi_preds.join(test_set).sort_values(
    by="estimate")
svm_preds_extended = svm_preds.join(test_set).sort_values("distance")

gmm_preds_extended.to_csv("../output/sheets/gmm_preds.csv", index_label=False)
Hpi_preds_extended.to_csv("../output/sheets/hpi_preds.csv", index_label=False)
svm_preds_extended.to_csv("../output/sheets/svm_preds.csv", index_label=False)
